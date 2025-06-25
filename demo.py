#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interactive demo for the BMR-model-merge framework.
This script provides both a CLI and web UI for testing Japanese OCR capabilities.
"""

import argparse
import os
import sys
import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("bmr-demo")

# Import project modules
from evomerge.models import get_model
from evomerge.data.processor import JapaneseDocumentProcessor
from evomerge.models.field_extractor import InvoiceFieldExtractor
from evomerge.utils import load_config, setup_environment, ensure_font

# Optional imports for web UI
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    logger.warning("Gradio not installed. Web UI will not be available.")


class OCRDemo:
    """Interactive demo for the BMR-model-merge Japanese OCR system."""
    
    def __init__(self, config_path: str = None, model_path: str = None):
        """
        Initialize the OCR demo with configuration and models.
        
        Args:
            config_path: Path to configuration file
            model_path: Path to model checkpoint
        """
        # Load configuration
        self.config = load_config(config_path) if config_path else {}
        
        # Setup environment
        setup_environment()
        
        # Ensure Japanese fonts are available
        self.font_path = ensure_font()
        
        # Initialize processor
        self.processor = JapaneseDocumentProcessor(
            normalize_characters=self.config.get("normalize_characters", True),
            correct_orientation=self.config.get("correct_orientation", True)
        )
        
        # Load OCR model
        model_config = self.config.get("model", {})
        model_type = model_config.get("type", "japanese_ocr")
        self.model = None
        
        if model_path:
            logger.info(f"Loading model from {model_path}")
            self.model = get_model(model_type, model_config)
            self.model.load_model(model_path)
        else:
            logger.warning("No model path provided. Will attempt to use best available model.")
            self.model = self._find_best_model()
        
        # Initialize field extractor
        self.field_extractor = InvoiceFieldExtractor()
    
    def _find_best_model(self):
        """Find the best available model from checkpoints directory."""
        checkpoints_dir = Path("checkpoints")
        
        if not checkpoints_dir.exists():
            logger.warning("No checkpoints directory found")
            return None
        
        # Look for the best model
        model_files = list(checkpoints_dir.glob("*best*.pt"))
        if not model_files:
            model_files = list(checkpoints_dir.glob("*.pt"))
        
        if not model_files:
            logger.error("No model files found in checkpoints directory")
            return None
        
        # Sort by modification time (newest first)
        model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        best_model_path = str(model_files[0])
        
        logger.info(f"Using most recent model: {best_model_path}")
        model_config = self.config.get("model", {})
        model_type = model_config.get("type", "japanese_ocr")
        
        model = get_model(model_type, model_config)
        model.load_model(best_model_path)
        return model
    
    def process_image(self, image_path: Union[str, Path, np.ndarray], 
                     visualize: bool = False) -> Dict:
        """
        Process an image with OCR and field extraction.
        
        Args:
            image_path: Path to image or numpy array
            visualize: Whether to create visualization
            
        Returns:
            Dictionary with OCR results
        """
        start_time = time.time()
        
        # Load image if path provided
        if isinstance(image_path, (str, Path)):
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path
        
        # Preprocess image
        logger.info("Preprocessing image")
        processed_image = self.processor.process_image(image)
        
        # Run OCR if model is available
        if self.model is None:
            logger.error("No OCR model available. Please provide a model path.")
            return {"error": "No OCR model available"}
        
        logger.info("Running OCR")
        ocr_result = self.model(processed_image)
        
        # Extract fields
        logger.info("Extracting fields")
        fields = self.field_extractor.extract_fields(ocr_result)
        
        # Process elapsed time
        elapsed = time.time() - start_time
        
        # Prepare results
        result = {
            "ocr_text": ocr_result.get("text", ""),
            "fields": fields,
            "processing_time": elapsed
        }
        
        # Generate visualization if requested
        if visualize:
            result["visualization"] = self._create_visualization(
                image, processed_image, ocr_result, fields
            )
        
        return result
    
    def _create_visualization(self, original_image, processed_image, 
                             ocr_result, fields):
        """Create visualization of OCR results."""
        # Convert to PIL for easier text drawing
        h, w = original_image.shape[:2]
        
        # Create a figure with 3 subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")
        
        # Processed image
        axes[0, 1].imshow(processed_image, cmap='gray')
        axes[0, 1].set_title("Processed Image")
        axes[0, 1].axis("off")
        
        # OCR text result
        axes[1, 0].text(0.05, 0.95, ocr_result.get("text", "No text detected"),
                 horizontalalignment='left', verticalalignment='top',
                 wrap=True, fontname="MS Gothic" if os.name == 'nt' else "Noto Sans CJK JP",
                 fontsize=8)
        axes[1, 0].set_title("Extracted Text")
        axes[1, 0].axis("off")
        
        # Extracted fields
        field_text = "\n".join(f"{k}: {v}" for k, v in fields.items())
        axes[1, 1].text(0.05, 0.95, field_text,
                 horizontalalignment='left', verticalalignment='top',
                 wrap=True, fontsize=10)
        axes[1, 1].set_title("Extracted Fields")
        axes[1, 1].axis("off")
        
        plt.tight_layout()
        
        # Save to BytesIO
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        plt.close(fig)
        
        # Return PIL Image
        return Image.open(buf)
    
    def cli_demo(self, image_path: str, output_path: Optional[str] = None):
        """Run demo in CLI mode."""
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return
        
        logger.info(f"Processing image: {image_path}")
        result = self.process_image(image_path, visualize=True)
        
        # Print OCR results
        print("\n" + "="*80)
        print("OCR RESULTS")
        print("="*80)
        print(result["ocr_text"])
        print("\n" + "="*80)
        print("EXTRACTED FIELDS")
        print("="*80)
        for field, value in result["fields"].items():
            print(f"{field}: {value}")
        print("\n" + "="*80)
        print(f"Processing time: {result['processing_time']:.2f} seconds")
        
        # Save visualization if requested
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            result["visualization"].save(output_path)
            logger.info(f"Visualization saved to {output_path}")
            
            # Also save JSON results
            json_path = os.path.splitext(output_path)[0] + ".json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json_data = {
                    "ocr_text": result["ocr_text"],
                    "fields": result["fields"],
                    "processing_time": result["processing_time"]
                }
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            logger.info(f"JSON results saved to {json_path}")
    
    def web_ui(self):
        """Launch Gradio web UI for interactive demo."""
        if not GRADIO_AVAILABLE:
            logger.error("Gradio is required for web UI. Install with 'pip install gradio'")
            return
        
        def process_image_for_gradio(image, algorithm_choice, confidence):
            """Wrapper for Gradio interface"""
            if algorithm_choice != "Default":
                logger.info(f"Using {algorithm_choice} algorithm")
                # This would normally load a different model based on algorithm choice
                # For demo purposes, we'll just log it
            
            # Convert confidence to float
            conf = float(confidence) / 100.0
            self.field_extractor.confidence_threshold = conf
            
            # Process the image
            result = self.process_image(np.array(image), visualize=True)
            
            # Return results
            fields_text = "\n".join([f"**{k}**: {v}" for k, v in result["fields"].items()])
            time_text = f"Processing time: {result['processing_time']:.2f} seconds"
            
            return (
                result["visualization"], 
                result["ocr_text"], 
                fields_text,
                time_text
            )
        
        # Create Gradio interface
        with gr.Blocks(title="Japanese OCR Demo") as demo:
            gr.Markdown("# Japanese OCR with BMR Model Merging")
            gr.Markdown("Upload an image of a Japanese invoice or receipt to extract text and fields.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(type="pil", label="Input Image")
                    
                    with gr.Row():
                        algorithm = gr.Dropdown(
                            ["Default", "BMR Optimized", "BWR Optimized", "Genetic Algorithm"], 
                            label="Algorithm", 
                            value="Default"
                        )
                        confidence_slider = gr.Slider(
                            minimum=50, 
                            maximum=95, 
                            value=70, 
                            step=5, 
                            label="Confidence Threshold (%)"
                        )
                    
                    process_btn = gr.Button("Process Image", variant="primary")
                
                with gr.Column(scale=1):
                    output_viz = gr.Image(label="Visualization")
                    output_text = gr.Textbox(label="Extracted Text", lines=10)
                    output_fields = gr.Markdown(label="Extracted Fields")
                    output_time = gr.Textbox(label="Processing Time")
            
            # Sample images
            gr.Markdown("### Sample Images")
            with gr.Row():
                sample1 = gr.Image(value="data/samples/invoice_1.jpg", label="Sample Invoice 1")
                sample2 = gr.Image(value="data/samples/receipt_1.jpg", label="Sample Receipt 1")
                sample3 = gr.Image(value="data/samples/invoice_2.jpg", label="Sample Invoice 2")
            
            # Set up event handlers
            process_btn.click(
                process_image_for_gradio, 
                inputs=[input_image, algorithm, confidence_slider], 
                outputs=[output_viz, output_text, output_fields, output_time]
            )
            
            # Sample image click handlers
            sample1.click(
                lambda: "data/samples/invoice_1.jpg",
                outputs=input_image
            )
            sample2.click(
                lambda: "data/samples/receipt_1.jpg",
                outputs=input_image
            )
            sample3.click(
                lambda: "data/samples/invoice_2.jpg",
                outputs=input_image
            )
            
            # Instructions and information
            with gr.Accordion("Instructions", open=False):
                gr.Markdown("""
                ## How to use this demo
                
                1. Upload an image of a Japanese invoice or receipt using the upload button
                2. Or click one of the sample images below
                3. Select the optimization algorithm to use:
                   - Default: Standard OCR model
                   - BMR Optimized: Best-Mean-Random algorithm optimized model
                   - BWR Optimized: Best-Worst-Random algorithm optimized model
                   - Genetic Algorithm: Traditional evolutionary algorithm optimized model
                4. Adjust confidence threshold as needed (lower for more results, higher for better precision)
                5. Click "Process Image" and wait for results
                
                ## About the algorithms
                
                - **BMR (Best-Mean-Random)**: A parameter-free optimization algorithm that leverages the best solution, population mean, and a randomly selected solution
                - **BWR (Best-Worst-Random)**: A variant that uses the worst solution for directional guidance away from poor solutions
                - **Genetic Algorithm**: Traditional evolutionary optimization with crossover and mutation operations
                
                For detailed information, see our [documentation](https://github.com/yourusername/bmr-model-merge).
                """)
        
        # Launch the interface
        demo.launch(share=True)


def main():
    """Main entry point for the demo script."""
    parser = argparse.ArgumentParser(
        description="Japanese OCR Demo with BMR Model Merging"
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        default="cli", 
        choices=["cli", "web"],
        help="Demo mode: cli or web (default: cli)"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/demo_config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=None,
        help="Path to model checkpoint (will use best available if not provided)"
    )
    
    parser.add_argument(
        "--image_path", 
        type=str, 
        default=None,
        help="Path to image file (required for CLI mode)"
    )
    
    parser.add_argument(
        "--output_path", 
        type=str, 
        default=None,
        help="Path to save visualization and results (CLI mode only)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = args.config if os.path.exists(args.config) else None
    if config_path is None:
        logger.warning(f"Config file not found: {args.config}")
    
    # Initialize demo
    demo = OCRDemo(config_path=config_path, model_path=args.model_path)
    
    # Run in requested mode
    if args.mode == "web":
        if not GRADIO_AVAILABLE:
            logger.error("Gradio is required for web mode. Install with 'pip install gradio'")
            return
        logger.info("Starting web UI")
        demo.web_ui()
    else:  # CLI mode
        if not args.image_path:
            logger.error("Image path is required for CLI mode")
            parser.print_help()
            return
        
        logger.info("Running in CLI mode")
        demo.cli_demo(args.image_path, args.output_path)


if __name__ == "__main__":
    main()