#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H200 GPU Full Implementation for BMR Model Merge
Author: Ravikumar Shah
"""

import os
import sys
import subprocess
import argparse
import shutil
import datetime
import torch
import logging
import json
import numpy as np
# import cmd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("workflow.log")]
)
logger = logging.getLogger("h200-workflow")

def setup_gpu_env():
    """Configure environment variables for optimal H200 GPU usage."""
    logger.info("=== Setting up H200 GPU environment for full-scale model merging ===")
    
    # Set environment variables for optimal GPU performance
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Use all available GPUs
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    os.environ["NCCL_P2P_DISABLE"] = "1"  # Can help with multi-GPU communication
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"  # Optimize connection pooling
    
    # Set PyTorch to use TF32 on Ampere GPUs
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
    
    # Configure tensor parallelism for model loading
    os.environ["TP_DEGREE"] = "2"
    os.environ["DP_DEGREE"] = "2"
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        logger.info("Clearing CUDA cache")
        torch.cuda.empty_cache()
    else:
        logger.warning("CUDA not available, running on CPU only")

def setup_data_directories():
    """Create data directories for datasets if they don't exist."""
    logger.info("=== Creating data directories ===")
    os.makedirs("data/jp_invoices/validation", exist_ok=True)
    os.makedirs("data/jp_receipts/validation", exist_ok=True)
    logger.info("Data directories created")

def install_system_dependencies():
    """Install system dependencies required for OpenCV."""
    logger.info("=== Installing system dependencies for OpenCV ===")
    try:
        # Check if we're on a Debian/Ubuntu system
        if os.path.exists("/usr/bin/apt-get"):
            logger.info("Detected Debian/Ubuntu system, installing OpenGL libraries...")
            subprocess.run(["apt-get", "update", "-y"], check=True)
            subprocess.run(["apt-get", "install", "-y", "libgl1-mesa-glx", "libglib2.0-0"], check=True)
            logger.info("System dependencies installed successfully")
        # Check if we're on a RHEL/CentOS/Fedora system
        elif os.path.exists("/usr/bin/yum"):
            logger.info("Detected RHEL/CentOS/Fedora system, installing OpenGL libraries...")
            subprocess.run(["yum", "install", "-y", "mesa-libGL", "glib2"], check=True)
            logger.info("System dependencies installed successfully")
        else:
            logger.warning("Unknown Linux distribution, skipping system dependencies installation")
            logger.warning("If the workflow fails with OpenCV errors, please manually install OpenGL libraries")
    except subprocess.CalledProcessError as e:
        logger.error(f"System dependencies installation failed: {e}")
        logger.warning("If you're not running as root or don't have sudo privileges, you'll need to install these manually")
        logger.warning("Required packages: libgl1-mesa-glx (Debian/Ubuntu) or mesa-libGL (RHEL/CentOS/Fedora)")
    except Exception as e:
        logger.error(f"Unexpected error during system dependencies installation: {e}")

def install_dependencies():
    """Install required packages for full-scale implementation."""
    logger.info("=== Installing necessary packages ===")
    packages = [
        # "torch==2.1.0",
        # "accelerate==0.27.2",
        # "transformers==4.38.2",
        # "bitsandbytes==0.41.0",
        # "flash-attn==2.5.0",
        # "vllm==0.3.0",
        "huggingface_hub"
    ]
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade"] + packages, check=True)
        logger.info("Packages installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Package installation failed: {e}")
        logger.info("Continuing workflow despite package installation issues")

def run_complete_workflow(config_path, output_dir, algorithms=None, generations=None, models=None, minimal=False, force_download=False):
    """Run the complete workflow using the specified configuration."""
    logger.info(f"=== Starting full workflow with H200 optimized configuration ===")
    
    # Create output directory with timestamp if not specified
    if not output_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/h200_run_{timestamp}"
    
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {output_dir}")
    
    # Build command arguments
    # Create a Python script that will inject our mock cv2 module if needed
    script = f"""
import sys
import os

# If minimal mode, inject our mock cv2 module
if {str(minimal)}:
    # Add current directory to path
    sys.path.insert(0, os.path.abspath('.'))
    
    # Import our mock before anything else
    import cv2_mock
    sys.modules['cv2'] = cv2_mock
    print("Using cv2 mock implementation to avoid OpenGL dependency")

# Now execute the main script
with open('run_complete_workflow.py') as f:
    exec(f.read())
"""
    
    # Use the script with command line arguments
    command_list = [sys.executable, "-c", script]
    
    # Add all the original arguments
    for arg in ["--config", config_path, "--output-dir", output_dir]:
        if arg:
            command_list.append(arg)
    
    # Add optional arguments
    if algorithms:
        command_list.extend(["--algorithms"] + algorithms)
    
    if generations:
        command_list.extend(["--generations", str(generations)])
    
    if models:
        command_list.extend(["--models", str(models)])
    
    # Add force download flag if specified
    if force_download:
        command_list.append("--force-download")
    
    # Run the workflow
    try:
        logger.info(f"Running command: {' '.join(command_list)}")
        subprocess.run(command_list, check=True)
        logger.info("Workflow completed successfully")
        return output_dir
    except subprocess.CalledProcessError as e:
        logger.error(f"Workflow failed: {e}")
        sys.exit(1)

def save_to_pytorch_format(output_dir):
    """Convert the best model to PyTorch format."""
    logger.info("=== Converting model to PyTorch format for inference and Hugging Face ===")
    
    try:
        # Create output model directory
        optimized_dir = os.path.join(output_dir, "optimized_model")
        os.makedirs(optimized_dir, exist_ok=True)
        logger.info(f"Created output directory: {optimized_dir}")
        
        # Find the best model
        best_model_path = None
        algorithm = None
        
        # Try different algorithm outputs in order of preference
        logger.info("Searching for best model checkpoints...")
        for alg in ["bmr", "bwr", "genetic"]:
            path = os.path.join(output_dir, "checkpoints", f"{alg}_best.npy")
            logger.info(f"Checking for {alg} checkpoint at: {path}")
            if os.path.exists(path):
                best_model_path = path
                algorithm = alg
                logger.info(f"Found {alg} checkpoint")
                break
        
        if not best_model_path:
            logger.error("No best model found in checkpoints directory")
            logger.error(f"Contents of {os.path.join(output_dir, 'checkpoints')}: {os.listdir(os.path.join(output_dir, 'checkpoints'))}")
            return None
        
        logger.info(f"Using best model from {algorithm}: {best_model_path}")
        
        # Output path for PyTorch model
        pt_output_path = os.path.join(optimized_dir, "model_optimized.pt")
        logger.info(f"Will save PyTorch model to: {pt_output_path}")
        
        # Import required modules dynamically to avoid potential issues
        logger.info(f"Importing load_best_model from {algorithm} module")
        
        # For BMR algorithm
        if algorithm == "bmr":
            logger.info("Importing from evomerge.evolution.bmr")
            from evomerge.evolution.bmr import load_best_model
        # For BWR algorithm
        elif algorithm == "bwr":
            logger.info("Importing from evomerge.evolution.bwr")
            from evomerge.evolution.bwr import load_best_model
        # For genetic algorithm
        else:
            logger.info("Importing from evomerge.evolution.genetic")
            from evomerge.evolution.genetic import load_best_model
        
        # Load the model
        logger.info(f"Loading model from {best_model_path}")
        model = load_best_model(best_model_path)
        logger.info(f"Model loaded successfully: {type(model).__name__}")
        
        # Save in PyTorch format
        logger.info(f"Saving model to {pt_output_path}")
        torch.save(model, pt_output_path)
        logger.info(f"Model saved in PyTorch format to {pt_output_path}")
        
        # Save metadata
        metadata = {
            'algorithm': algorithm,
            'creation_date': datetime.datetime.now().strftime("%Y-%m-%d"),
            'author': 'Ravikumar Shah',
            'description': 'Japanese OCR model created using evolutionary model merging'
        }
        
        metadata_path = os.path.join(optimized_dir, "model_metadata.json")
        logger.info(f"Saving metadata to {metadata_path}")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info("Metadata saved successfully")
        
        return pt_output_path
    except Exception as e:
        logger.error(f"Error converting model to PyTorch format: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def main():
    """Main function to run the H200 full implementation workflow."""
    parser = argparse.ArgumentParser(description="H200 GPU Full Implementation for BMR Model Merge")
    parser.add_argument("--config", default="configs/evolution/h200_full.yaml",
                        help="Path to configuration file (default: configs/evolution/h200_full.yaml)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to store results (default: results/h200_run_TIMESTAMP)")
    parser.add_argument("--algorithms", nargs="+", choices=["bmr", "bwr", "genetic"], default=["bmr", "bwr", "genetic"],
                        help="Algorithms to run (default: all three)")
    parser.add_argument("--generations", type=int, default=100,
                        help="Number of generations to run (default: 100)")
    parser.add_argument("--models", type=int, default=10,
                        help="Number of models to use (default: 10)")
    parser.add_argument("--skip-env-setup", action="store_true",
                        help="Skip environment setup (default: False)")
    parser.add_argument("--skip-deps", action="store_true",
                        help="Skip dependency installation (default: False)")
    parser.add_argument("--minimal", action="store_true",
                        help="Run with minimal dependencies (tries to avoid OpenCV usage) (default: False)")
    parser.add_argument("--force-download", action="store_true",
                        help="Force download of models even if they exist locally (default: False)")
    parser.add_argument("--vista-data-path", default="/workspace/vista-data",
                        help="Path to vista data directory (default: /workspace/vista-data)")
    
    args = parser.parse_args()
    
    # Setup environment and directories
    if not args.skip_env_setup:
        setup_gpu_env()
        setup_data_directories()
    
    # Install dependencies
    if not args.skip_deps:
        # First install system dependencies for OpenCV
        install_system_dependencies()
        # Then install Python dependencies
        install_dependencies()
    
    # Update the config file to use the vista data path
    if args.vista_data_path:
        try:
            import yaml
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            
            # Update the datasets to use the vista data path
            if 'evaluation' in config and 'datasets' in config['evaluation']:
                for i, dataset in enumerate(config['evaluation']['datasets']):
                    if 'vista' in dataset.get('name', '').lower():
                        if dataset['name'] == 'vista-train':
                            config['evaluation']['datasets'][i]['path'] = f"{args.vista_data_path}/train"
                        elif dataset['name'] == 'vista-test':
                            config['evaluation']['datasets'][i]['path'] = f"{args.vista_data_path}/test"
            
            # Write the updated config back to the file
            with open(args.config, 'w') as f:
                yaml.dump(config, f)
            
            logger.info(f"Updated config file to use vista data path: {args.vista_data_path}")
        except Exception as e:
            logger.error(f"Error updating config file: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Run the complete workflow
    output_dir = run_complete_workflow(
        args.config,
        args.output_dir,
        args.algorithms,
        args.generations,
        args.models,
        True,  # Set minimal=True to use cv2_mock and avoid OpenGL dependencies
        args.force_download  # Pass force_download parameter
    )
    
    # Convert to PyTorch format
    pt_model_path = save_to_pytorch_format(output_dir)
    
    if pt_model_path:
        # Print information for publishing to HuggingFace
        logger.info("\n=== Model Conversion Successful ===")
        logger.info(f"PyTorch model saved to: {pt_model_path}")
        logger.info("\n=== Publishing to Hugging Face Hub (Optional) ===")
        logger.info("To publish this model to Hugging Face Hub, run:")
        logger.info(f"python publish_to_hf.py --model_path {pt_model_path} --base_model stockmark/stockmark-2-vl-100b-beta --repo_name your-username/japanese-ocr-merged --token YOUR_HF_TOKEN")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())