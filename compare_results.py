#!/usr/bin/env python
"""
Compare results from different optimization algorithms.

This script analyzes and visualizes the performance of different
optimization algorithms (genetic, BMR, BWR) for model merging.
"""

import os
import sys
import argparse
import json
import glob
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare results from different optimization algorithms",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--results_dirs", type=str, nargs="+", required=True,
        help="List of directories containing optimization results"
    )
    parser.add_argument(
        "--output_dir", type=str, default="comparison_results",
        help="Directory to save comparison results"
    )
    parser.add_argument(
        "--metrics", type=str, nargs="+",
        default=["best_fitness", "avg_fitness"],
        help="Metrics to compare"
    )
    
    return parser.parse_args()


def load_history_files(results_dirs: List[str]) -> Dict[str, Any]:
    """
    Load history files from results directories.
    
    Args:
        results_dirs: List of directories containing results
        
    Returns:
        Dictionary mapping algorithm names to history data
    """
    histories = {}
    
    for results_dir in results_dirs:
        # Determine algorithm name from directory path
        algorithm = os.path.basename(results_dir)
        
        # Look for history files
        history_file = None
        potential_paths = [
            os.path.join(results_dir, "history.json"),
            os.path.join(results_dir, "evolution_stats.json"),
            *glob.glob(os.path.join(results_dir, "**", "history.json"), recursive=True),
            *glob.glob(os.path.join(results_dir, "**", "evolution_stats.json"), recursive=True)
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                history_file = path
                break
                
        if history_file:
            try:
                with open(history_file, "r") as f:
                    history_data = json.load(f)
                    
                    # Handle different history file formats
                    if "best_fitness_per_gen" in history_data:
                        # Genetic algorithm format
                        histories[algorithm] = {
                            "best_fitness": history_data["best_fitness_per_gen"],
                            "avg_fitness": history_data["avg_fitness_per_gen"],
                            "generations": list(range(1, len(history_data["best_fitness_per_gen"]) + 1)),
                        }
                    elif "best_fitness" in history_data:
                        # BMR/BWR format
                        histories[algorithm] = history_data
                    else:
                        logger.warning(f"Unknown history file format in {history_file}")
                        
                logger.info(f"Loaded history data for {algorithm} from {history_file}")
            except Exception as e:
                logger.error(f"Error loading history file {history_file}: {e}")
        else:
            logger.warning(f"No history file found in {results_dir}")
    
    return histories


def find_best_models(results_dirs: List[str]) -> Dict[str, Any]:
    """
    Find best models from results directories.
    
    Args:
        results_dirs: List of directories containing results
        
    Returns:
        Dictionary mapping algorithm names to best model data
    """
    best_models = {}
    
    for results_dir in results_dirs:
        # Determine algorithm name from directory path
        algorithm = os.path.basename(results_dir)
        
        # Look for best model files
        best_model_file = None
        potential_paths = [
            os.path.join(results_dir, "best_model.json"),
            os.path.join(results_dir, f"{algorithm}_final", "best_model.json"),
            *glob.glob(os.path.join(results_dir, "**", "best_model.json"), recursive=True),
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                best_model_file = path
                break
                
        if best_model_file:
            try:
                with open(best_model_file, "r") as f:
                    model_data = json.load(f)
                    best_models[algorithm] = model_data
                    
                logger.info(f"Loaded best model data for {algorithm} from {best_model_file}")
            except Exception as e:
                logger.error(f"Error loading best model file {best_model_file}: {e}")
        else:
            logger.warning(f"No best model file found in {results_dir}")
    
    return best_models


def create_fitness_plots(histories: Dict[str, Any], output_dir: str, metrics: List[str]):
    """
    Create fitness comparison plots.
    
    Args:
        histories: Dictionary mapping algorithm names to history data
        output_dir: Directory to save plots
        metrics: Metrics to plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot settings
    colors = ["blue", "green", "red", "purple", "orange", "brown"]
    plt.figure(figsize=(12, 8))
    
    # Create plots for each metric
    for metric in metrics:
        if metric not in next(iter(histories.values())):
            logger.warning(f"Metric {metric} not found in history data, skipping")
            continue
            
        plt.clf()
        
        # Plot each algorithm
        for i, (algorithm, history) in enumerate(histories.items()):
            # Skip if metric not in this algorithm's history
            if metric not in history:
                continue
                
            generations = history.get("generations", list(range(1, len(history[metric]) + 1)))
            plt.plot(generations, history[metric], label=algorithm, color=colors[i % len(colors)], linewidth=2)
        
        plt.title(f"{metric.replace('_', ' ').title()} Comparison", fontsize=16)
        plt.xlabel("Generation", fontsize=14)
        plt.ylabel(metric.replace("_", " ").title(), fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(output_dir, f"{metric}_comparison.png")
        plt.savefig(output_path, dpi=300)
        logger.info(f"Saved plot to {output_path}")


def create_comparison_table(histories: Dict[str, Any], best_models: Dict[str, Any], output_dir: str):
    """
    Create comparison table of algorithms.
    
    Args:
        histories: Dictionary mapping algorithm names to history data
        best_models: Dictionary mapping algorithm names to best model data
        output_dir: Directory to save table
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for table
    data = []
    for algorithm in histories.keys():
        history = histories[algorithm]
        best_model = best_models.get(algorithm, {})
        
        # Get final fitness values
        best_fitness = max(history.get("best_fitness", [0]))
        avg_fitness = history.get("avg_fitness", [0])[-1] if history.get("avg_fitness") else 0
        
        # Get number of generations
        num_generations = len(history.get("generations", history.get("best_fitness", [])))
        
        # Get best model fitness
        model_fitness = best_model.get("fitness", best_model.get("overall_fitness", 0))
        
        data.append({
            "Algorithm": algorithm,
            "Best Fitness": best_fitness,
            "Average Fitness (Final)": avg_fitness,
            "Generations": num_generations,
            "Best Model Fitness": model_fitness,
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, "algorithm_comparison.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved comparison table to {csv_path}")
    
    # Also create a HTML table for easy viewing
    html_path = os.path.join(output_dir, "algorithm_comparison.html")
    with open(html_path, "w") as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Optimization Algorithm Comparison</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                h1 { color: #333; }
            </style>
        </head>
        <body>
            <h1>Optimization Algorithm Comparison</h1>
        """)
        
        f.write(df.to_html(index=False))
        
        f.write("""
        </body>
        </html>
        """)
    
    logger.info(f"Saved HTML comparison table to {html_path}")
    
    return df


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load history files from each results directory
    histories = load_history_files(args.results_dirs)
    
    if not histories:
        logger.error("No history files found. Cannot generate comparison.")
        return
    
    # Find best models
    best_models = find_best_models(args.results_dirs)
    
    # Create fitness plots
    create_fitness_plots(histories, args.output_dir, args.metrics)
    
    # Create comparison table
    df = create_comparison_table(histories, best_models, args.output_dir)
    
    # Print summary to console
    print("\n" + "=" * 50)
    print("OPTIMIZATION ALGORITHM COMPARISON")
    print("=" * 50)
    print(df.to_string(index=False))
    print("\n" + "=" * 50)
    print(f"Detailed results saved to: {args.output_dir}")
    
    # Determine best algorithm
    best_idx = df["Best Model Fitness"].idxmax()
    best_algorithm = df.iloc[best_idx]["Algorithm"]
    best_fitness = df.iloc[best_idx]["Best Model Fitness"]
    
    print(f"\nBest algorithm: {best_algorithm} (fitness: {best_fitness:.4f})")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()