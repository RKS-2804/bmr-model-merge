#!/usr/bin/env python
"""
Main evaluation script for Japanese invoice/receipt OCR models.

This script provides a unified interface for evaluating OCR model performance
on Japanese invoices and receipts using various evaluation metrics and tasks.
"""

import os
import argparse
import gc
import json
import logging
import yaml
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Any, Union

import torch

from evomerge import instantiate_from_config, load_config, set_seed
from evomerge.models.japanese_ocr import JapaneseOCRModel
from evomerge.models.field_extractor import InvoiceFieldExtractor
from evomerge.eval.ja_invoice_vqa import JapaneseInvoiceVQA
from evomerge.eval.ja_receipt_extraction import JapaneseReceiptExtraction
from evomerge.eval.ja_field_detection import JapaneseFieldDetection
from evomerge.eval.metrics import OCRLanguageDetector


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
class EvaluationConfig:
    """Configuration for an evaluation run."""
    task: str
    data_path: str
    batch_size: int = 4
    device: Optional[str] = None
    output_dir: Optional[str] = None
    additional_params: Optional[Dict[str, Any]] = None


def parse_args():
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Evaluate Japanese invoice/receipt OCR models")
    
    parser.add_argument(
        "--config_path", type=str, required=True,
        help="Path to model configuration YAML file",
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Path to save evaluation results (default: results/{config_name}.json)",
    )
    parser.add_argument(
        "--data_dir", type=str, default="data",
        help="Base directory for evaluation datasets",
    )
    parser.add_argument(
        "--tasks", type=str, nargs="+", 
        choices=["vqa", "extraction", "field_detection", "all"],
        default=["all"],
        help="Evaluation tasks to run (default: all)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for evaluation (default: 4)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device for evaluation ('cuda', 'cpu', etc.)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate visualizations of evaluation results",
    )
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output_path is None:
        config_name = os.path.splitext(os.path.basename(args.config_path))[0]
        args.output_path = f"results/{config_name}.json"
        
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        
    return args


def load_model(config: Dict[str, Any], device: Optional[str] = None) -> torch.nn.Module:
    """Load a model from configuration.
    
    Args:
        config: Model configuration dictionary
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    if "model" not in config:
        raise ValueError("Configuration must contain a 'model' section")
        
    # Set device if provided
    if device:
        config["model"]["params"]["device"] = device
        
    # Instantiate model from config
    model = instantiate_from_config(config["model"])
    logger.info(f"Model loaded: {model.__class__.__name__}")
    
    return model


def create_evaluator(
    task: str, 
    config: Dict[str, Any], 
    data_dir: str,
    batch_size: int,
    device: Optional[str] = None,
    visualize: bool = True,
) -> Any:
    """Create an evaluator for a specific task.
    
    Args:
        task: Evaluation task name
        config: Evaluation configuration
        data_dir: Base data directory
        batch_size: Batch size for evaluation
        device: Device for evaluation
        visualize: Whether to generate visualizations
        
    Returns:
        Evaluator object
    """
    eval_name = f"{task}_{config.get('name', '')}"
    
    # Default data paths based on task
    default_data_paths = {
        "vqa": os.path.join(data_dir, "ja_invoice_vqa"),
        "extraction": os.path.join(data_dir, "ja_receipts"),
        "field_detection": os.path.join(data_dir, "ja_invoices_fields"),
    }
    
    # Set default data path if not in config
    data_path = config.get("data_path", default_data_paths.get(task))
    if not data_path:
        raise ValueError(f"No data path provided for task {task}")
        
    # Set output directory
    output_dir = config.get("output_dir", f"results/{eval_name}")
    
    # Create evaluator based on task
    if task == "vqa":
        evaluator = JapaneseInvoiceVQA(
            name=eval_name,
            data_path=data_path,
            batch_size=batch_size,
            device=device,
            output_dir=output_dir,
            question_templates=config.get("question_templates"),
        )
    elif task == "extraction":
        evaluator = JapaneseReceiptExtraction(
            name=eval_name,
            data_path=data_path,
            batch_size=batch_size,
            device=device,
            output_dir=output_dir,
            normalize_text=config.get("normalize_text", True),
            visualization=visualize,
        )
    elif task == "field_detection":
        # Create field extractor if specified in config
        field_extractor = None
        if "field_extractor" in config:
            field_extractor_config = config["field_extractor"]
            field_extractor = InvoiceFieldExtractor(
                model_name=field_extractor_config.get("model_name", "cl-tohoku/bert-base-japanese-v2"),
                device=device,
                confidence_threshold=field_extractor_config.get("confidence_threshold", 0.7),
            )
            
        evaluator = JapaneseFieldDetection(
            name=eval_name,
            data_path=data_path,
            batch_size=batch_size,
            device=device,
            output_dir=output_dir,
            field_extractor=field_extractor,
            visualization=visualize,
        )
    else:
        raise ValueError(f"Unknown evaluation task: {task}")
        
    return evaluator


def run_evaluation(
    model: torch.nn.Module,
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Run evaluation tasks on a model.
    
    Args:
        model: Model to evaluate
        config: Evaluation configuration
        args: Command line arguments
        
    Returns:
        Dictionary of evaluation results
    """
    # Determine which tasks to run
    tasks_to_run = []
    if "all" in args.tasks:
        tasks_to_run = ["vqa", "extraction", "field_detection"]
    else:
        tasks_to_run = args.tasks
        
    # Initialize results dictionary
    results = {}
    
    # Get evaluation configs
    eval_configs = config.get("eval", {})
    if not eval_configs:
        logger.warning("No evaluation configurations found in config file")
        
        # Create default evaluation configurations
        eval_configs = {}
        for task in tasks_to_run:
            eval_configs[task] = {
                "name": f"{task}_default",
                "data_path": os.path.join(args.data_dir, f"ja_{task.replace('vqa', 'invoice_vqa')}"),
            }
    
    # Run evaluations
    for task in tasks_to_run:
        # Skip if task not in config and not using default config
        if task not in eval_configs and "all" not in args.tasks:
            logger.warning(f"Skipping task {task} as it is not in config file")
            continue
            
        # Get task config
        task_config = eval_configs.get(task, {})
        
        # Create evaluator
        logger.info(f"Creating evaluator for task: {task}")
        evaluator = create_evaluator(
            task=task,
            config=task_config,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            device=args.device,
            visualize=args.visualize,
        )
        
        # Run evaluation
        logger.info(f"Running evaluation for task: {task}")
        eval_output = evaluator(model)
        
        # Store results
        results[evaluator.name] = asdict(eval_output)
        
        # Clean up to free memory
        del evaluator
        torch.cuda.empty_cache()
        gc.collect()
    
    return results


def generate_summary_report(results: Dict[str, Any], output_path: str) -> str:
    """Generate a summary report of evaluation results.
    
    Args:
        results: Evaluation results
        output_path: Path to save summary report
        
    Returns:
        Summary report text
    """
    # Create report header
    summary = "# Japanese Invoice/Receipt OCR Evaluation Summary\n\n"
    
    # Add evaluation results for each task
    for eval_name, eval_data in results.items():
        summary += f"## {eval_name}\n\n"
        
        # Add metrics table
        summary += "### Metrics\n\n"
        summary += "| Metric | Value |\n"
        summary += "|--------|-------|\n"
        
        for metric, value in eval_data["metrics"].items():
            summary += f"| {metric} | {value:.2f} |\n"
            
        summary += "\n"
        
    # Save summary report
    report_path = output_path.replace(".json", "_summary.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(summary)
        
    logger.info(f"Summary report saved to {report_path}")
    return summary


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load config
    logger.info(f"Loading config from {args.config_path}")
    config = load_config(args.config_path)
    
    # Log config
    logger.info(f"Config:\n{json.dumps(config, indent=2, ensure_ascii=False)}")
    
    # Load model
    logger.info("Loading model")
    model = load_model(config, device=args.device)
    
    # Run evaluations
    logger.info("Running evaluations")
    results = run_evaluation(model, config, args)
    
    # Save results
    logger.info(f"Saving results to {args.output_path}")
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    # Generate summary report
    generate_summary_report(results, args.output_path)
    
    logger.info("Evaluation completed successfully")


if __name__ == "__main__":
    main()