#!/usr/bin/env python
"""
Run evolutionary model merging with checkpoint saving.

This script runs the evolutionary optimization process to find optimal
model merging recipes for Japanese invoice/receipt OCR and saves
checkpoints of the best models during the process.
"""

import os
import sys
import argparse
import logging
import json
import yaml
from pathlib import Path
import time
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import project modules
from evomerge import load_config, set_seed
from evomerge.evolution.genetic import GeneticAlgorithm
from evomerge.evolution.population import ModelPopulation
from evomerge.evolution.fitness import FitnessEvaluator
from evomerge.models.japanese_ocr import JapaneseOCRModel
from evomerge.eval.metrics import calculate_character_accuracy


# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Set environment variable to avoid tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class CheckpointConfig:
    """Configuration for checkpointing evolutionary runs."""
    
    enabled: bool = True
    interval: int = 5  # Save checkpoint every N generations
    save_top_k: int = 3  # Save top K individuals in each checkpoint
    path: str = "checkpoints"
    save_population: bool = True


@dataclass
class EvolutionConfig:
    """Configuration for an evolutionary run."""
    
    population_size: int = 20
    generations: int = 50
    mutation_rate: float = 0.2
    crossover_rate: float = 0.7
    elitism_count: int = 2
    tournament_size: int = 3
    fitness_weights: Optional[Dict[str, float]] = None
    checkpoint: Optional[CheckpointConfig] = None
    

def parse_args():
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run evolutionary model merging with checkpointing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config_path", type=str, default="configs/evolution/default.yaml",
        help="Path to evolution configuration YAML file",
    )
    parser.add_argument(
        "--output_dir", type=str, default="evolution_results",
        help="Directory to save results and checkpoints",
    )
    parser.add_argument(
        "--population_size", type=int, default=None,
        help="Size of the population (overrides config file)",
    )
    parser.add_argument(
        "--generations", type=int, default=None,
        help="Number of generations to run (overrides config file)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to run on ('cuda', 'cpu', etc.)",
    )
    parser.add_argument(
        "--checkpoint_interval", type=int, default=None,
        help="Save checkpoint every N generations (overrides config file)",
    )
    parser.add_argument(
        "--resume_from", type=str, default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--eval_dataset", type=str, default=None,
        help="Path to evaluation dataset for fitness calculation",
    )
    parser.add_argument(
        "--no_visualize", action="store_true",
        help="Disable visualization of evolution progress",
    )
    
    args = parser.parse_args()
    
    # Set default device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    return args


def load_evolution_config(config_path: str) -> EvolutionConfig:
    """Load evolution configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Evolution configuration
    """
    # Check if config file exists
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}")
        logger.info("Using default evolution configuration")
        return EvolutionConfig(
            checkpoint=CheckpointConfig()
        )
    
    # Load config from file
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Parse evolution parameters
    evolution_params = config.get("evolution", {})
    checkpoint_params = evolution_params.get("checkpoint", {})
    
    # Create checkpoint config
    checkpoint_config = CheckpointConfig(
        enabled=checkpoint_params.get("enabled", True),
        interval=checkpoint_params.get("interval", 5),
        save_top_k=checkpoint_params.get("save_top_k", 3),
        path=checkpoint_params.get("path", "checkpoints"),
        save_population=checkpoint_params.get("save_population", True),
    )
    
    # Create evolution config
    evolution_config = EvolutionConfig(
        population_size=evolution_params.get("population_size", 20),
        generations=evolution_params.get("generations", 50),
        mutation_rate=evolution_params.get("mutation_rate", 0.2),
        crossover_rate=evolution_params.get("crossover_rate", 0.7),
        elitism_count=evolution_params.get("elitism_count", 2),
        tournament_size=evolution_params.get("tournament_size", 3),
        fitness_weights=evolution_params.get("fitness_weights"),
        checkpoint=checkpoint_config,
    )
    
    return evolution_config


def prepare_fitness_evaluator(eval_dataset: Optional[str], device: str) -> FitnessEvaluator:
    """Prepare fitness evaluator for evolutionary optimization.
    
    Args:
        eval_dataset: Path to evaluation dataset
        device: Device to run on
        
    Returns:
        Fitness evaluator
    """
    # Create fitness evaluator
    evaluator = FitnessEvaluator(
        metrics=[
            "character_accuracy",
            "field_extraction_accuracy",
            "processing_speed",
        ],
        device=device,
    )
    
    # Set evaluation dataset if provided
    if eval_dataset:
        evaluator.set_dataset(eval_dataset)
    
    return evaluator


def save_checkpoint(
    generation: int,
    population: ModelPopulation,
    best_individuals: List[Any],
    output_dir: str,
    config: EvolutionConfig,
):
    """Save checkpoint of the evolutionary run.
    
    Args:
        generation: Current generation number
        population: Current population
        best_individuals: List of best individuals
        output_dir: Directory to save checkpoint to
        config: Evolution configuration
    """
    if not config.checkpoint or not config.checkpoint.enabled:
        return
        
    # Create checkpoint directory
    checkpoint_dir = os.path.join(output_dir, config.checkpoint.path)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create generation-specific checkpoint directory
    gen_checkpoint_dir = os.path.join(checkpoint_dir, f"generation_{generation}")
    os.makedirs(gen_checkpoint_dir, exist_ok=True)
    
    # Save best individuals
    for i, individual in enumerate(best_individuals[:config.checkpoint.save_top_k]):
        # Save model
        model_path = os.path.join(gen_checkpoint_dir, f"best_{i+1}.pt")
        individual.save(model_path)
        
        # Save metadata
        metadata_path = os.path.join(gen_checkpoint_dir, f"best_{i+1}_metadata.json")
        metadata = {
            "fitness": individual.fitness,
            "genotype": individual.genotype,
            "generation": generation,
            "rank": i + 1,
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            
    # Save population state if enabled
    if config.checkpoint.save_population:
        population_path = os.path.join(gen_checkpoint_dir, "population.pkl")
        population.save(population_path)
    
    # Save configuration
    config_path = os.path.join(gen_checkpoint_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, ensure_ascii=False, indent=2)
        
    logger.info(f"Checkpoint saved at generation {generation} to {gen_checkpoint_dir}")


def resume_from_checkpoint(checkpoint_path: str, device: str) -> Tuple[ModelPopulation, int, List[Dict]]:
    """Resume evolution from a checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        device: Device to run on
        
    Returns:
        Tuple of (population, generation, history)
    """
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    # Load population
    population_path = os.path.join(checkpoint_path, "population.pkl")
    if not os.path.exists(population_path):
        raise FileNotFoundError(f"Population file not found: {population_path}")
        
    population = ModelPopulation.load(population_path)
    
    # Determine generation
    checkpoint_dir = os.path.dirname(checkpoint_path)
    generation_part = os.path.basename(checkpoint_path)
    if generation_part.startswith("generation_"):
        generation = int(generation_part.split("_")[1])
    else:
        # Try to determine generation from metadata files
        metadata_files = [f for f in os.listdir(checkpoint_path) if f.endswith("_metadata.json")]
        generation = 0
        
        if metadata_files:
            metadata_path = os.path.join(checkpoint_path, metadata_files[0])
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                generation = metadata.get("generation", 0)
    
    # Load history (if available)
    history_path = os.path.join(checkpoint_dir, "history.json")
    history = []
    
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
    
    logger.info(f"Resumed from checkpoint at generation {generation}")
    return population, generation, history


def visualize_evolution(history: List[Dict], output_dir: str):
    """Visualize evolution progress.
    
    Args:
        history: List of dictionaries with evolution history
        output_dir: Directory to save visualization to
    """
    generations = [entry["generation"] for entry in history]
    best_fitness = [entry["best_fitness"] for entry in history]
    avg_fitness = [entry["avg_fitness"] for entry in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_fitness, "b-", label="Best Fitness")
    plt.plot(generations, avg_fitness, "r--", label="Average Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Evolution Progress")
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plot_path = os.path.join(output_dir, "evolution_progress.png")
    plt.savefig(plot_path)
    logger.info(f"Evolution visualization saved to {plot_path}")


def run_evolution(args):
    """Run evolutionary model merging.
    
    Args:
        args: Command line arguments
    """
    # Load evolution config
    config = load_evolution_config(args.config_path)
    
    # Override config with command line arguments
    if args.population_size is not None:
        config.population_size = args.population_size
        
    if args.generations is not None:
        config.generations = args.generations
        
    if args.checkpoint_interval is not None:
        config.checkpoint.interval = args.checkpoint_interval
    
    # Create checkpoint directory in output_dir
    checkpoint_dir = os.path.join(args.output_dir, config.checkpoint.path)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Prepare fitness evaluator
    evaluator = prepare_fitness_evaluator(args.eval_dataset, args.device)
    
    # Initialize or resume population
    generation_start = 0
    history = []
    
    if args.resume_from:
        # Resume from checkpoint
        population, generation_start, history = resume_from_checkpoint(args.resume_from, args.device)
    else:
        # Initialize population
        population = ModelPopulation(
            size=config.population_size,
            model_class=JapaneseOCRModel,
            device=args.device,
        )
        population.initialize()
    
    # Create genetic algorithm
    ga = GeneticAlgorithm(
        population=population,
        fitness_evaluator=evaluator,
        mutation_rate=config.mutation_rate,
        crossover_rate=config.crossover_rate,
        elitism_count=config.elitism_count,
        tournament_size=config.tournament_size,
    )
    
    # Run evolution
    logger.info(f"Starting evolutionary optimization with {config.population_size} individuals for {config.generations} generations")
    start_time = time.time()
    
    for generation in range(generation_start, config.generations):
        logger.info(f"Generation {generation+1}/{config.generations}")
        
        # Evaluate fitness
        ga.evaluate_fitness()
        
        # Get current population statistics
        best_individual = ga.get_best_individual()
        best_fitness = best_individual.fitness
        avg_fitness = ga.get_average_fitness()
        
        # Log progress
        logger.info(f"  Best fitness: {best_fitness:.4f}")
        logger.info(f"  Average fitness: {avg_fitness:.4f}")
        
        # Save history
        history.append({
            "generation": generation,
            "best_fitness": best_fitness,
            "avg_fitness": avg_fitness,
            "elapsed_time": time.time() - start_time,
        })
        
        # Save checkpoint if interval is reached
        if config.checkpoint.enabled and (generation + 1) % config.checkpoint.interval == 0:
            save_checkpoint(
                generation=generation,
                population=population,
                best_individuals=ga.get_top_individuals(config.checkpoint.save_top_k),
                output_dir=args.output_dir,
                config=config,
            )
        
        # Save history
        with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
            
        # Early stopping if maximum fitness is reached
        if best_fitness >= 0.99:
            logger.info("Maximum fitness reached, stopping early")
            break
        
        # Evolve to next generation (skip for the last generation)
        if generation < config.generations - 1:
            ga.evolve()
    
    # Save final results
    final_checkpoint_dir = os.path.join(checkpoint_dir, "final")
    os.makedirs(final_checkpoint_dir, exist_ok=True)
    
    # Save best model
    best_model = ga.get_best_individual()
    best_model_path = os.path.join(final_checkpoint_dir, "best_model.pt")
    best_model.save(best_model_path)
    
    # Save best model metadata
    best_metadata_path = os.path.join(final_checkpoint_dir, "best_model_metadata.json")
    with open(best_metadata_path, "w", encoding="utf-8") as f:
        json.dump({
            "fitness": best_model.fitness,
            "genotype": best_model.genotype,
            "generation": config.generations - 1,
        }, f, ensure_ascii=False, indent=2)
    
    # Create evolution summary
    summary = {
        "best_fitness": best_model.fitness,
        "generations": config.generations,
        "population_size": config.population_size,
        "elapsed_time": time.time() - start_time,
        "parameters": asdict(config),
    }
    
    summary_path = os.path.join(args.output_dir, "evolution_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Visualize evolution progress
    if not args.no_visualize:
        visualize_evolution(history, args.output_dir)
    
    logger.info(f"Evolutionary optimization completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Best model saved to {best_model_path}")
    logger.info(f"Final fitness: {best_model.fitness:.4f}")


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Run evolution
    run_evolution(args)


if __name__ == "__main__":
    main()