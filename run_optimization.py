#!/usr/bin/env python
"""
Run optimization for model merging with choice of algorithm.

This script allows choosing between Genetic, BMR (Best-Mean-Random), or 
BWR (Best-Worst-Random) algorithms for optimizing model merges.
"""

import os
import sys
import argparse
import yaml
import json
import logging
import random
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

import torch
import numpy as np

from evomerge.evolution import (
    GeneticOptimizer, 
    BMROptimizer, 
    BWROptimizer,
    ModelPopulation,
    OCRFitnessEvaluator
)
from evomerge import load_config, set_seed
from evomerge.models.japanese_ocr import JapaneseOCRModel


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("optimization.log")
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run model merging optimization with choice of algorithm",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # General parameters
    parser.add_argument(
        "--algorithm", type=str, default="genetic", choices=["genetic", "bmr", "bwr"],
        help="Optimization algorithm to use"
    )
    parser.add_argument(
        "--config_path", type=str, default="configs/evolution/default.yaml",
        help="Path to evolution configuration file"
    )
    parser.add_argument(
        "--models_dir", type=str, default="models",
        help="Directory containing models to merge"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Output directory for optimization results"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints",
        help="Directory for saving/loading checkpoints"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to run on (cuda, cpu). Defaults to CUDA if available."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--eval_dataset", type=str, default=None,
        help="Path to evaluation dataset"
    )
    
    # Algorithm-specific parameters
    parser.add_argument(
        "--population_size", type=int, default=None,
        help="Size of population (overrides config)"
    )
    parser.add_argument(
        "--generations", type=int, default=None,
        help="Number of generations to run (overrides config)"
    )
    parser.add_argument(
        "--mutation_rate", type=float, default=None,
        help="Mutation rate (overrides config)"
    )
    parser.add_argument(
        "--crossover_rate", type=float, default=None,
        help="Crossover rate for genetic algorithm (overrides config)"
    )
    parser.add_argument(
        "--mutation_probability", type=float, default=None,
        help="Probability of mutation for BMR/BWR (overrides config)"
    )
    parser.add_argument(
        "--resume_from", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    
    return parser.parse_args()


def setup_environment(args):
    """Setup environment for optimization."""
    # Set random seed
    if args.seed is not None:
        set_seed(args.seed)
    
    # Set default device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    logger.info(f"Using device: {args.device}")
    logger.info(f"Using random seed: {args.seed}")
    logger.info(f"Output directory: {args.output_dir}")


def load_evolution_config(config_path: str) -> Dict[str, Any]:
    """Load evolution configuration from file."""
    try:
        config = load_config(config_path)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}


def load_model_configs(models_dir: str) -> List[Dict[str, Any]]:
    """Load model configurations from models directory."""
    model_configs = []
    
    # Check all subdirectories in models_dir for config.yaml files
    for model_dir in Path(models_dir).iterdir():
        if model_dir.is_dir():
            config_path = model_dir / "config.yaml"
            if config_path.exists():
                try:
                    with open(config_path, "r") as f:
                        config = yaml.safe_load(f)
                    
                    model_configs.append({
                        "name": model_dir.name,
                        "path": str(model_dir),
                        "config": config
                    })
                    logger.info(f"Loaded model config: {model_dir.name}")
                except Exception as e:
                    logger.error(f"Error loading model config {config_path}: {e}")
    
    return model_configs


def setup_fitness_evaluator(args, config: Dict[str, Any]) -> OCRFitnessEvaluator:
    """Setup fitness evaluator for optimization."""
    # Extract evaluation parameters from config if available
    eval_config = config.get("evaluation", {})
    
    # Override with command line arguments if provided
    if args.eval_dataset:
        eval_config["dataset_path"] = args.eval_dataset
    
    # Create evaluator
    evaluator = OCRFitnessEvaluator(
        dataset_path=eval_config.get("dataset_path"),
        metric_weights=eval_config.get("metric_weights", {
            "character_accuracy": 0.6,
            "field_extraction_accuracy": 0.3,
            "processing_speed": 0.1
        }),
        device=args.device
    )
    
    return evaluator


def setup_population(args, config: Dict[str, Any], model_configs: List[Dict[str, Any]]) -> ModelPopulation:
    """Setup population for optimization."""
    # Extract population parameters from config
    population_size = args.population_size or config.get("evolution", {}).get("population_size", 10)
    
    # Create population
    population = ModelPopulation(
        size=population_size,
        model_class=JapaneseOCRModel,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    # Initialize population with base models
    if not args.resume_from:  # Only initialize if not resuming
        for model_config in model_configs:
            model_params = {
                "model_name": model_config["name"],
                "model_path": model_config["path"],
                "config": model_config["config"]
            }
            population.add_model(model_params)
        
        # If more individuals needed, create initial population
        if len(population.individuals) < population_size:
            population.initialize_random(population_size - len(population.individuals))
    
    return population


def setup_optimizer(args, config: Dict[str, Any], population: ModelPopulation, evaluator: OCRFitnessEvaluator):
    """Setup optimizer based on selected algorithm."""
    # Extract common parameters
    evolution_config = config.get("evolution", {})
    generations = args.generations or evolution_config.get("generations", 20)
    
    if args.algorithm == "genetic":
        # Extract genetic algorithm parameters
        crossover_rate = args.crossover_rate or evolution_config.get("crossover_rate", 0.7)
        mutation_rate = args.mutation_rate or evolution_config.get("mutation_rate", 0.2)
        elitism_rate = evolution_config.get("elitism_rate", 0.2)
        
        optimizer = GeneticOptimizer(
            population=population,
            n_generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elitism_rate=elitism_rate,
            results_dir=args.output_dir,
            checkpoint_dir=args.checkpoint_dir,
            evaluator=evaluator,
            resume_from=args.resume_from
        )
        logger.info(f"Using Genetic optimizer with {generations} generations, {crossover_rate} crossover rate, {mutation_rate} mutation rate")
        
    elif args.algorithm == "bmr":
        # Extract BMR parameters
        mutation_probability = args.mutation_probability or evolution_config.get("mutation_probability", 0.5)
        elitism_count = int(population.size * evolution_config.get("elitism_rate", 0.2))
        
        optimizer = BMROptimizer(
            population=population,
            fitness_evaluator=evaluator,
            mutation_probability=mutation_probability,
            elitism_count=elitism_count
        )
        logger.info(f"Using BMR optimizer with {mutation_probability} mutation probability")
        
    else:  # bwr
        # Extract BWR parameters
        mutation_probability = args.mutation_probability or evolution_config.get("mutation_probability", 0.5)
        elitism_count = int(population.size * evolution_config.get("elitism_rate", 0.2))
        
        optimizer = BWROptimizer(
            population=population,
            fitness_evaluator=evaluator,
            mutation_probability=mutation_probability,
            elitism_count=elitism_count
        )
        logger.info(f"Using BWR optimizer with {mutation_probability} mutation probability")
    
    return optimizer, generations


def run_optimization(optimizer, generations: int, algorithm: str):
    """Run optimization process."""
    logger.info(f"Starting optimization with {algorithm.upper()} algorithm for {generations} generations")
    
    # For BMR and BWR, we need to manually control the generation loop
    if algorithm in ["bmr", "bwr"]:
        # Evaluate initial population
        optimizer.evaluate_fitness()
        
        # Log initial stats
        best_individual = optimizer.get_best_individual()
        avg_fitness = optimizer.get_average_fitness()
        logger.info(f"Initial population - Best fitness: {best_individual.fitness:.4f}, Average fitness: {avg_fitness:.4f}")
        
        # Create results directory structure
        results_dir = os.path.join("results", algorithm)
        os.makedirs(results_dir, exist_ok=True)
        
        # Store fitness history
        history = {
            "best_fitness": [],
            "avg_fitness": [],
            "generation": []
        }
        
        # Run for specified number of generations
        for gen in range(generations):
            logger.info(f"Generation {gen+1}/{generations}")
            
            # Evolve population
            optimizer.evolve()
            
            # Evaluate new population
            optimizer.evaluate_fitness()
            
            # Log stats
            best_individual = optimizer.get_best_individual()
            avg_fitness = optimizer.get_average_fitness()
            logger.info(f"Generation {gen+1} - Best fitness: {best_individual.fitness:.4f}, Average fitness: {avg_fitness:.4f}")
            
            # Save best model of this generation
            best_dir = os.path.join(results_dir, f"gen_{gen+1}_best")
            os.makedirs(best_dir, exist_ok=True)
            best_individual.save(os.path.join(best_dir, "model.pt"))
            
            # Update history
            history["best_fitness"].append(best_individual.fitness)
            history["avg_fitness"].append(avg_fitness)
            history["generation"].append(gen+1)
            
            # Save history
            with open(os.path.join(results_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=2)
        
        logger.info(f"Optimization complete. Final best fitness: {best_individual.fitness:.4f}")
        return best_individual
    else:
        # For genetic algorithm, use its built-in run method
        optimizer.run(evaluate_initial=True)
        
        # Get best individual
        best_individual = optimizer.population.get_best_individual()
        logger.info(f"Optimization complete. Final best fitness: {best_individual.overall_fitness:.4f}")
        return best_individual


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup environment
    setup_environment(args)
    
    # Load configurations
    config = load_evolution_config(args.config_path)
    model_configs = load_model_configs(args.models_dir)
    
    if not model_configs:
        logger.error("No model configurations found. Please ensure models directory contains model configs.")
        return
    
    # Setup components
    evaluator = setup_fitness_evaluator(args, config)
    population = setup_population(args, config, model_configs)
    optimizer, generations = setup_optimizer(args, config, population, evaluator)
    
    # Run optimization
    best_model = run_optimization(optimizer, generations, args.algorithm)
    
    # Save final results
    final_dir = os.path.join(args.output_dir, f"{args.algorithm}_final")
    os.makedirs(final_dir, exist_ok=True)
    
    # Save best model configuration
    with open(os.path.join(final_dir, "best_model.json"), "w") as f:
        if hasattr(best_model, "overall_fitness"):
            # For genetic algorithm
            json.dump({
                "fitness": best_model.overall_fitness,
                "model_path": best_model.model_path,
                "model_config": best_model.model_config,
            }, f, indent=2)
        else:
            # For BMR/BWR
            json.dump({
                "fitness": best_model.fitness,
                "model_path": best_model.model_path if hasattr(best_model, "model_path") else None,
                "genotype": best_model.genotype,
            }, f, indent=2)
    
    logger.info(f"Optimization results saved to {final_dir}")


if __name__ == "__main__":
    main()