#!/usr/bin/env python
"""
Entry point script for running evolutionary model merging optimization.

This script sets up and runs the genetic algorithm optimization process
for model merging to improve Japanese invoice/receipt OCR performance.
"""

import os
import sys
import argparse
import logging
import yaml
import glob
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

import torch

from evomerge.evolution.genetic import GeneticOptimizer
from evomerge.evolution.fitness import OCRFitnessEvaluator, create_evaluator_from_config


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('evolution_run.log')
    ]
)

logger = logging.getLogger("run_evolution")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run evolutionary optimization for model merging'
    )
    
    # Basic parameters
    parser.add_argument('--config', type=str, default='evolution_config.yaml',
                        help='Path to evolution configuration file')
    parser.add_argument('--models-dir', type=str, default='./configs',
                        help='Directory containing model configuration files')
    parser.add_argument('--vlm-models', type=str, nargs='*', 
                        help='Space-separated list of VLM model names to include')
    parser.add_argument('--llm-models', type=str, nargs='*',
                        help='Space-separated list of LLM model names to include')
    parser.add_argument('--test-data', type=str, 
                        help='Path to test data for evaluation')
    
    # Directories
    parser.add_argument('--results-dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory for checkpoints')
    
    # Evolutionary parameters
    parser.add_argument('--population-size', type=int, default=10,
                        help='Size of the population')
    parser.add_argument('--generations', type=int, default=20,
                        help='Number of generations to run')
    parser.add_argument('--crossover-rate', type=float, default=0.7,
                        help='Probability of crossover')
    parser.add_argument('--mutation-rate', type=float, default=0.2,
                        help='Probability of mutation')
    parser.add_argument('--crossover-method', type=str, default='uniform',
                        choices=['uniform', 'single_point', 'sliced', 'weighted'],
                        help='Method for crossover')
    parser.add_argument('--mutation-method', type=str, default='parameter_noise',
                        choices=['parameter_noise', 'layer_dropout', 'recipe_params'],
                        help='Method for mutation')
    parser.add_argument('--elitism-rate', type=float, default=0.2,
                        help='Percentage of elite individuals to preserve')
    
    # Resume from checkpoint
    parser.add_argument('--resume', type=str,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int,
                        help='Random seed for reproducibility')
    
    # Device settings
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run on (e.g., "cuda", "cpu")')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for evaluation')
    
    # Debug mode
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode (smaller models/dataset)')
    
    return parser.parse_args()


def load_config_file(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}")
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def find_model_configs(
    models_dir: str,
    vlm_models: Optional[List[str]] = None,
    llm_models: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Find model configuration files.
    
    Args:
        models_dir: Base directory for model configs
        vlm_models: List of VLM model names to include
        llm_models: List of LLM model names to include
        
    Returns:
        Dictionary mapping model names to config file paths
    """
    config_files = {}
    
    # Look for VLM models
    vlm_dir = os.path.join(models_dir, "vlm")
    if os.path.exists(vlm_dir):
        vlm_configs = glob.glob(os.path.join(vlm_dir, "*.yaml"))
        
        for config_path in vlm_configs:
            model_name = os.path.splitext(os.path.basename(config_path))[0]
            
            # Filter by requested models if specified
            if vlm_models is None or model_name in vlm_models:
                config_files[f"vlm_{model_name}"] = config_path
    
    # Look for LLM models
    llm_dir = os.path.join(models_dir, "llm")
    if os.path.exists(llm_dir):
        llm_configs = glob.glob(os.path.join(llm_dir, "*.yaml"))
        
        for config_path in llm_configs:
            model_name = os.path.splitext(os.path.basename(config_path))[0]
            
            # Filter by requested models if specified
            if llm_models is None or model_name in llm_models:
                config_files[f"llm_{model_name}"] = config_path
    
    return config_files


def setup_evaluator(args, evolution_config: Dict[str, Any]) -> OCRFitnessEvaluator:
    """
    Set up the fitness evaluator.
    
    Args:
        args: Command line arguments
        evolution_config: Evolution configuration
        
    Returns:
        Configured evaluator
    """
    # Determine test data path
    test_data_path = args.test_data
    if not test_data_path and "test_data" in evolution_config:
        test_data_path = evolution_config["test_data"]
    
    if not test_data_path:
        raise ValueError("Test data path must be provided")
    
    # Get metric weights from config if available
    metrics_weights = None
    if "metrics_weights" in evolution_config:
        metrics_weights = evolution_config["metrics_weights"]
    
    # Create evaluator
    evaluator = OCRFitnessEvaluator(
        test_data_path=test_data_path,
        metrics_weights=metrics_weights,
        batch_size=args.batch_size,
        device=args.device
    )
    
    return evaluator


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Load evolution configuration
    evolution_config = load_config_file(args.config)
    
    # Create results and checkpoint directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.results_dir, f"run_{timestamp}")
    checkpoint_dir = os.path.join(args.checkpoint_dir, f"run_{timestamp}")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Set up file logger
    file_handler = logging.FileHandler(os.path.join(results_dir, "evolution.log"))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Log basic info
    logger.info(f"Starting evolutionary optimization run at {timestamp}")
    logger.info(f"Results will be saved to: {results_dir}")
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")
    
    # Set device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Find model configurations
    model_configs = find_model_configs(
        args.models_dir,
        vlm_models=args.vlm_models,
        llm_models=args.llm_models
    )
    
    if not model_configs:
        logger.error("No model configurations found!")
        return 1
    
    logger.info(f"Found {len(model_configs)} model configurations")
    for model_name, config_path in model_configs.items():
        logger.info(f"  - {model_name}: {config_path}")
    
    # Set up evaluator
    try:
        evaluator = setup_evaluator(args, evolution_config)
        logger.info("Fitness evaluator initialized")
    except Exception as e:
        logger.error(f"Error setting up evaluator: {e}")
        return 1
    
    # Create optimizer
    try:
        optimizer = GeneticOptimizer(
            population_size=args.population_size,
            n_generations=args.generations,
            crossover_rate=args.crossover_rate,
            mutation_rate=args.mutation_rate,
            elitism_rate=args.elitism_rate,
            results_dir=results_dir,
            checkpoint_dir=checkpoint_dir,
            evaluator=evaluator,
            crossover_method=args.crossover_method,
            mutation_method=args.mutation_method,
            random_seed=args.seed,
            config_paths=list(model_configs.values()),
            resume_from=args.resume
        )
        logger.info("Genetic optimizer initialized")
    except Exception as e:
        logger.error(f"Error creating optimizer: {e}")
        return 1
    
    # Run optimization
    try:
        logger.info("Starting evolutionary optimization")
        population = optimizer.run()
        
        # Get best individual
        best_individual = population.get_best_individual()
        if best_individual:
            logger.info(f"Best individual: {best_individual.id}")
            logger.info(f"Best fitness: {best_individual.overall_fitness:.4f}")
            
            # Save best individual details
            best_path = os.path.join(results_dir, "best_individual.yaml")
            with open(best_path, 'w') as f:
                yaml.dump(best_individual.to_dict(), f)
                
            logger.info(f"Best individual details saved to: {best_path}")
        
        logger.info("Evolutionary optimization completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Error during optimization: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())