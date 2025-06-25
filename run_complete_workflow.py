#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Complete Workflow Automation Script for BMR-model-merge

This script orchestrates the entire workflow for Japanese OCR optimization:
1. Downloads and prepares resources (models, datasets)
2. Runs optimization using multiple algorithms (BMR, BWR, Genetic)
3. Evaluates and compares the results
4. Selects the best model
5. Prepares the demo environment

Usage:
    python run_complete_workflow.py --config configs/evolution/default.yaml
    python run_complete_workflow.py --quick-run --models 3 --epochs 10
"""

import os
import sys
import time
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import concurrent.futures
import json
import shutil

import numpy as np
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("workflow.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("bmr-workflow")

# Import project modules
from evomerge.utils import load_config, setup_environment, ensure_font
from evomerge.data.datasets import JapaneseInvoiceDataset, JapaneseReceiptDataset
from evomerge.evolution.bmr import BMROptimizer
from evomerge.evolution.bwr import BWROptimizer
from evomerge.evolution.genetic import GeneticOptimizer
from evomerge.eval.metrics import calculate_metrics


class WorkflowManager:
    """Manager for the complete BMR-model-merge workflow."""
    
    def __init__(self, config_path: str, output_dir: str = "results"):
        """
        Initialize the workflow manager.
        
        Args:
            config_path: Path to configuration file
            output_dir: Directory to store results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        logger.info(f"Loading configuration from {config_path}")
        self.config = load_config(config_path)
        
        # Create results directories
        self.results_dir = self.output_dir / "results"
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.plots_dir = self.output_dir / "plots"
        self.logs_dir = self.output_dir / "logs"
        
        for d in [self.results_dir, self.checkpoints_dir, self.plots_dir, self.logs_dir]:
            d.mkdir(exist_ok=True)
        
        # Setup environment
        setup_environment()
        
        # Ensure Japanese fonts are available
        self.font_path = ensure_font()
        
        # Initialize state
        self.datasets = {}
        self.models = {}
        self.optimizers = {}
        self.results = {}
        
        # Configure
        self._configure()
    
    def _configure(self):
        """Configure the workflow based on settings."""
        # Set random seed for reproducibility
        if "seed" in self.config.get("optimization", {}):
            seed = self.config["optimization"]["seed"]
            np.random.seed(seed)
            logger.info(f"Set random seed to {seed}")
    
    def setup_resources(self, force_download: bool = False, 
                       models_limit: Optional[int] = None):
        """
        Download and prepare resources.
        
        Args:
            force_download: Force download even if files exist locally
            models_limit: Limit the number of models to download
        """
        logger.info("Setting up resources")
        
        # Download models
        self._download_models(force_download, models_limit)
        
        # Prepare datasets
        self._prepare_datasets()
        
        logger.info("Resource setup complete")
    
    def _download_models(self, force_download: bool = False, 
                        models_limit: Optional[int] = None):
        """Download required models."""
        logger.info("Downloading models")
        
        # Get list of models to download
        models = self.config.get("models", {}).get("base_models", [])
        
        if models_limit is not None and models_limit > 0:
            logger.info(f"Limiting to {models_limit} models")
            models = models[:models_limit]
        
        # Ensure models directory exists
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        for i, model_config in enumerate(models):
            model_name = model_config.get("name", f"model_{i}")
            model_path = model_config.get("path", f"models/{model_name}")
            
            logger.info(f"Processing model {model_name} ({i+1}/{len(models)})")
            
            # Skip if already exists and not force download
            if Path(model_path).exists() and not force_download:
                logger.info(f"Model {model_name} already exists, skipping download")
                continue
            
            logger.info(f"Downloading model {model_name} to {model_path}")
            
            # This would normally use a specific model download function
            # For this example, we'll just create a placeholder
            model_dir = Path(model_path)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a placeholder file with model info
            with open(model_dir / "info.json", "w") as f:
                json.dump({
                    "name": model_name,
                    "type": model_config.get("type", "vlm"),
                    "downloaded": time.strftime("%Y-%m-%d %H:%M:%S"),
                }, f, indent=2)
            
            logger.info(f"Model {model_name} processed")
    
    def _prepare_datasets(self):
        """Prepare evaluation datasets."""
        logger.info("Preparing datasets")
        
        # Get dataset configurations
        datasets_config = self.config.get("evaluation", {}).get("datasets", [])
        
        for dataset_config in datasets_config:
            name = dataset_config.get("name")
            path = dataset_config.get("path")
            split = dataset_config.get("split", "validation")
            
            # Use path as name if name not provided
            if name is None and path:
                # Extract last directory name from path
                import os
                name = os.path.basename(path)
            
            logger.info(f"Preparing dataset {name} (split: {split})")
            
            # Skip if no path provided
            if not path:
                logger.warning(f"No path provided for dataset {name}, skipping")
                continue
            
            # Initialize dataset based on type
            if name and "invoice" in name.lower():
                dataset = JapaneseInvoiceDataset(path, split=split)
            elif name and "receipt" in name.lower():
                dataset = JapaneseReceiptDataset(path, split=split)
            # Fallback to determining type from path if name doesn't help
            elif path and "invoice" in path.lower():
                dataset = JapaneseInvoiceDataset(path, split=split)
            elif path and "receipt" in path.lower():
                dataset = JapaneseReceiptDataset(path, split=split)
            else:
                logger.warning(f"Unknown dataset type {name}, skipping")
                continue
            
            # Store dataset
            self.datasets[name] = dataset
            logger.info(f"Dataset {name} prepared with {len(dataset)} samples")
    
    def run_optimizations(self, algorithms: List[str] = None,
                        generations: Optional[int] = None):
        """
        Run optimization algorithms.
        
        Args:
            algorithms: List of algorithms to run (default: all enabled in config)
            generations: Override number of generations
        """
        logger.info("Running optimization algorithms")
        
        # Determine which algorithms to run
        if algorithms is None:
            algorithms = []
            if self.config.get("bmr", {}).get("enabled", True):
                algorithms.append("bmr")
            if self.config.get("bwr", {}).get("enabled", True):
                algorithms.append("bwr")
            if self.config.get("genetic", {}).get("enabled", True):
                algorithms.append("genetic")
        
        # Get default number of generations
        if generations is None:
            generations = self.config.get("optimization", {}).get("max_generations", 30)
        
        # Run each algorithm
        for algorithm in algorithms:
            logger.info(f"Running {algorithm} optimization")
            
            # Set up optimizer
            optimizer = self._create_optimizer(algorithm)
            if optimizer is None:
                logger.warning(f"Failed to create optimizer for {algorithm}, skipping")
                continue
            
            # Store optimizer
            self.optimizers[algorithm] = optimizer
            
            # Run optimization
            start_time = time.time()
            results = self._run_optimization(optimizer, generations)
            elapsed = time.time() - start_time
            
            # Store results
            self.results[algorithm] = {
                "best_individual": results["best_individual"],
                "best_fitness": results["best_fitness"],
                "history": results["history"],
                "runtime": elapsed
            }
            
            # Save checkpoint
            self._save_checkpoint(algorithm, results)
            
            logger.info(f"{algorithm} optimization completed in {elapsed:.2f} seconds")
            logger.info(f"Best fitness: {results['best_fitness']:.4f}")
    
    def _create_optimizer(self, algorithm: str) -> Any:
        """Create an optimizer instance."""
        # Get optimization configuration
        opt_config = self.config.get("optimization", {})
        pop_size = opt_config.get("population_size", 40)
        
        # Get algorithm-specific configuration
        alg_config = self.config.get(algorithm, {})
        
        # Create optimizer
        if algorithm == "bmr":
            return BMROptimizer(
                population_size=pop_size,
                dimension=100,  # This would normally be determined from models
                lower_bound=0.0,
                upper_bound=1.0,
                fitness_function=self._fitness_function,
                T=alg_config.get("T_parameter", 1.0)
            )
        elif algorithm == "bwr":
            return BWROptimizer(
                population_size=pop_size,
                dimension=100,  # This would normally be determined from models
                lower_bound=0.0,
                upper_bound=1.0,
                fitness_function=self._fitness_function,
                T=alg_config.get("T_parameter", 1.0)
            )
        elif algorithm == "genetic":
            return GeneticOptimizer(
                population_size=pop_size,
                dimension=100,  # This would normally be determined from models
                lower_bound=0.0,
                upper_bound=1.0,
                fitness_function=self._fitness_function,
                mutation_rate=alg_config.get("mutation_rate", 0.1),
                crossover_rate=alg_config.get("crossover_rate", 0.7)
            )
        else:
            logger.error(f"Unknown algorithm: {algorithm}")
            return None
    
    def _fitness_function(self, individual: np.ndarray) -> float:
        """
        Fitness function for optimization.
        
        Args:
            individual: Individual to evaluate
            
        Returns:
            Fitness score
        """
        # This would normally create a model from weights and evaluate
        # For this example, we'll just use a dummy function
        
        # Add small random noise to make it interesting
        return (
            0.5 * np.mean(individual) + 
            0.3 * np.max(individual) + 
            0.2 * (1 - np.std(individual)) +
            0.05 * np.random.random()
        )
    
    def _run_optimization(self, optimizer, generations: int) -> Dict:
        """Run an optimization algorithm."""
        history = []
        
        # Run optimization for specified number of generations
        for generation in range(generations):
            # Evolve one generation
            optimizer.evolve()
            
            # Get current best
            best_individual = optimizer.get_best_individual()
            best_fitness = optimizer.get_best_fitness()
            
            # Store history
            history.append({
                "generation": generation,
                "best_fitness": best_fitness,
                "population_mean": np.mean([
                    optimizer.fitness_function(ind) 
                    for ind in optimizer.population
                ]),
                "population_diversity": np.std([
                    np.mean(ind) for ind in optimizer.population
                ])
            })
            
            # Log progress
            if (generation + 1) % 5 == 0 or generation == 0:
                logger.info(f"Generation {generation+1}/{generations}: "
                           f"Best fitness = {best_fitness:.4f}")
        
        return {
            "best_individual": optimizer.get_best_individual(),
            "best_fitness": optimizer.get_best_fitness(),
            "history": history
        }
    
    def _save_checkpoint(self, algorithm: str, results: Dict):
        """Save optimization checkpoint."""
        # Save best individual
        checkpoint_path = self.checkpoints_dir / f"{algorithm}_best.npy"
        np.save(checkpoint_path, results["best_individual"])
        
        # Save history
        history_path = self.logs_dir / f"{algorithm}_history.json"
        with open(history_path, "w") as f:
            json.dump(results["history"], f, indent=2)
        
        logger.info(f"Saved checkpoint for {algorithm} to {checkpoint_path}")
    
    def compare_results(self):
        """Compare results of different optimization algorithms."""
        logger.info("Comparing optimization results")
        
        if not self.results:
            logger.warning("No results to compare")
            return
        
        # Prepare comparison data
        comparison = {
            "algorithms": list(self.results.keys()),
            "best_fitness": {alg: res["best_fitness"] for alg, res in self.results.items()},
            "runtime": {alg: res["runtime"] for alg, res in self.results.items()},
            "convergence": {}
        }
        
        # Extract convergence data
        for alg, res in self.results.items():
            history = res["history"]
            comparison["convergence"][alg] = {
                "generations": [h["generation"] for h in history],
                "best_fitness": [h["best_fitness"] for h in history],
                "population_mean": [h["population_mean"] for h in history],
                "diversity": [h["population_diversity"] for h in history]
            }
        
        # Determine best algorithm
        best_algorithm = max(comparison["best_fitness"].items(), key=lambda x: x[1])[0]
        comparison["best_algorithm"] = best_algorithm
        
        # Save comparison results
        comparison_path = self.results_dir / "algorithm_comparison.json"
        with open(comparison_path, "w") as f:
            json.dump(comparison, f, indent=2)
        
        # Create visualizations
        self._create_comparison_plots(comparison)
        
        logger.info(f"Best algorithm: {best_algorithm} with "
                   f"fitness {comparison['best_fitness'][best_algorithm]:.4f}")
        logger.info(f"Comparison results saved to {comparison_path}")
        
        return best_algorithm
    
    def _create_comparison_plots(self, comparison: Dict):
        """Create comparison plots."""
        # Create convergence plot
        plt.figure(figsize=(10, 6))
        
        for alg, data in comparison["convergence"].items():
            plt.plot(data["generations"], data["best_fitness"], label=f"{alg}")
        
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.title("Convergence Comparison")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        
        # Save plot
        convergence_plot_path = self.plots_dir / "convergence_comparison.png"
        plt.savefig(convergence_plot_path, dpi=150)
        plt.close()
        
        # Create diversity plot
        plt.figure(figsize=(10, 6))
        
        for alg, data in comparison["convergence"].items():
            plt.plot(data["generations"], data["diversity"], label=f"{alg}")
        
        plt.xlabel("Generation")
        plt.ylabel("Population Diversity")
        plt.title("Diversity Comparison")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        
        # Save plot
        diversity_plot_path = self.plots_dir / "diversity_comparison.png"
        plt.savefig(diversity_plot_path, dpi=150)
        plt.close()
        
        # Create bar chart for final fitness
        plt.figure(figsize=(8, 5))
        
        algorithms = list(comparison["best_fitness"].keys())
        fitnesses = [comparison["best_fitness"][alg] for alg in algorithms]
        
        plt.bar(algorithms, fitnesses)
        plt.xlabel("Algorithm")
        plt.ylabel("Best Fitness")
        plt.title("Final Fitness Comparison")
        plt.grid(True, axis="y", linestyle="--", alpha=0.7)
        
        # Save plot
        fitness_plot_path = self.plots_dir / "fitness_comparison.png"
        plt.savefig(fitness_plot_path, dpi=150)
        plt.close()
        
        logger.info(f"Comparison plots saved to {self.plots_dir}")
    
    def prepare_demo(self, best_algorithm: str = None):
        """
        Prepare demo environment with the best model.
        
        Args:
            best_algorithm: Algorithm to use (if None, use the best from comparison)
        """
        logger.info("Preparing demo environment")
        
        # If no algorithm specified, use the best from comparison
        if best_algorithm is None and self.results:
            best_fitnesses = {alg: res["best_fitness"] 
                             for alg, res in self.results.items()}
            best_algorithm = max(best_fitnesses.items(), key=lambda x: x[1])[0]
        
        if best_algorithm and best_algorithm in self.results:
            # Copy best model to demo location
            src_path = self.checkpoints_dir / f"{best_algorithm}_best.npy"
            demo_dir = Path("demo_model")
            demo_dir.mkdir(exist_ok=True)
            
            dst_path = demo_dir / "best_model.npy"
            shutil.copy(src_path, dst_path)
            
            # Create demo config
            config = {
                "model": {
                    "type": "japanese_ocr",
                    "path": str(dst_path)
                },
                "normalize_characters": True,
                "correct_orientation": True
            }
            
            # Save demo config
            config_path = Path("configs/demo_config.yaml")
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, "w") as f:
                yaml.dump(config, f)
            
            logger.info(f"Demo prepared with {best_algorithm} model")
            logger.info(f"Demo model saved to {dst_path}")
            logger.info(f"Demo config saved to {config_path}")
        else:
            logger.warning("No best algorithm determined, demo not prepared")
    
    def generate_report(self):
        """Generate a final report of the workflow."""
        logger.info("Generating final report")
        
        # Create report
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": self.config,
            "datasets": {name: len(dataset) for name, dataset in self.datasets.items()},
            "results": {
                alg: {
                    "best_fitness": res["best_fitness"],
                    "runtime": res["runtime"]
                } for alg, res in self.results.items()
            },
            "plots": [str(p) for p in self.plots_dir.glob("*.png")]
        }
        
        # Determine best algorithm
        if self.results:
            best_fitnesses = {alg: res["best_fitness"] 
                             for alg, res in self.results.items()}
            report["best_algorithm"] = max(best_fitnesses.items(), key=lambda x: x[1])[0]
        
        # Save report
        report_path = self.results_dir / "workflow_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        # Create human-readable report
        md_report = [
            "# BMR-Model-Merge Workflow Report",
            f"Generated on: {report['timestamp']}",
            "",
            "## Overview",
            f"- Datasets processed: {len(report['datasets'])}",
            f"- Algorithms compared: {len(report['results'])}",
            f"- Best algorithm: {report.get('best_algorithm', 'Not determined')}",
            "",
            "## Dataset Summary",
        ]
        
        for name, count in report["datasets"].items():
            md_report.append(f"- {name}: {count} samples")
        
        md_report.extend([
            "",
            "## Algorithm Results",
            "",
            "| Algorithm | Best Fitness | Runtime (s) |",
            "|-----------|--------------|-------------|",
        ])
        
        for alg, res in report["results"].items():
            md_report.append(
                f"| {alg} | {res['best_fitness']:.4f} | {res['runtime']:.2f} |"
            )
        
        md_report.extend([
            "",
            "## Plots",
            "",
            "### Convergence Comparison",
            "![Convergence](../plots/convergence_comparison.png)",
            "",
            "### Diversity Comparison",
            "![Diversity](../plots/diversity_comparison.png)",
            "",
            "### Final Fitness Comparison",
            "![Fitness](../plots/fitness_comparison.png)",
        ])
        
        # Save markdown report
        md_report_path = self.results_dir / "workflow_report.md"
        with open(md_report_path, "w") as f:
            f.write("\n".join(md_report))
        
        logger.info(f"Final report saved to {report_path} and {md_report_path}")


def main():
    """Main entry point for the complete workflow script."""
    parser = argparse.ArgumentParser(
        description="Complete Workflow for Japanese OCR with BMR Model Merging"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/evolution/default.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="results",
        help="Directory to store results"
    )
    
    parser.add_argument(
        "--force-download", 
        action="store_true",
        help="Force download of models even if they exist locally"
    )
    
    parser.add_argument(
        "--algorithms", 
        type=str, 
        nargs="+",
        choices=["bmr", "bwr", "genetic"],
        help="Algorithms to run (default: all enabled in config)"
    )
    
    parser.add_argument(
        "--generations", 
        type=int, 
        default=None,
        help="Override number of generations to run"
    )
    
    parser.add_argument(
        "--models", 
        type=int, 
        default=None,
        help="Limit number of models to download"
    )
    
    parser.add_argument(
        "--quick-run", 
        action="store_true",
        help="Run a quick version with limited models and generations"
    )
    
    parser.add_argument(
        "--skip-setup", 
        action="store_true",
        help="Skip resource setup (assume already done)"
    )
    
    parser.add_argument(
        "--skip-demo", 
        action="store_true",
        help="Skip demo preparation"
    )
    
    args = parser.parse_args()
    
    # Handle quick run mode
    if args.quick_run:
        if args.models is None:
            args.models = 2
        if args.generations is None:
            args.generations = 10
        
        logger.info(f"Quick run mode: limiting to {args.models} models and {args.generations} generations")
    
    # Initialize workflow manager
    manager = WorkflowManager(args.config, args.output_dir)
    
    try:
        # Setup resources
        if not args.skip_setup:
            manager.setup_resources(args.force_download, args.models)
        
        # Run optimizations
        manager.run_optimizations(args.algorithms, args.generations)
        
        # Compare results
        best_algorithm = manager.compare_results()
        
        # Prepare demo
        if not args.skip_demo:
            manager.prepare_demo(best_algorithm)
        
        # Generate report
        manager.generate_report()
        
        logger.info("Workflow completed successfully")
    
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())