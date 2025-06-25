import logging
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from datasets import Dataset

from .metrics import (
    rouge_ja, 
    OCRLanguageDetector, 
    calculate_character_accuracy, 
    calculate_field_extraction_accuracy, 
    normalize_japanese_text_for_ocr
)


logger = logging.getLogger(__name__)


@dataclass
class EvalOutput:
    """Container for evaluation outputs."""
    metrics: Dict[str, float]
    results: List[Dict[str, Any]]


def dict_collation_fn(samples: List) -> Dict:
    """Collation function for dictionary-type samples.
    
    Args:
        samples: List of sample dictionaries
        
    Returns:
        Batched dictionary
    """
    keys = set.intersection(*[set(sample.keys()) for sample in samples])
    batched = {key: [] for key in keys}

    for s in samples:
        [batched[key].append(s[key]) for key in batched]

    result = {}
    for key in batched:
        result[key] = list(batched[key])

    del samples
    del batched
    return result


def flatten_list(
    results: List[Dict[str, List[Union[str, bool]]]]
) -> Dict[str, List[Union[str, bool]]]:
    """Flatten a list of result dictionaries.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Flattened dictionary
    """
    flatten_results = {}
    for res in results:
        for k in res:
            if k not in flatten_results:
                flatten_results[k] = res[k]
            else:
                flatten_results[k].extend(res[k])
    return flatten_results


def _evaluate_ocr(
    model, example: Dict[str, List[Any]]
) -> Dict[str, List[Union[str, bool, Dict[str, str]]]]:
    """Evaluate OCR model on a batch of images.
    
    Args:
        model: OCR model
        example: Dictionary containing images and ground truth
        
    Returns:
        Dictionary with predictions and ground truth
    """
    images = example["image"]
    ground_truth = example.get("text", [""] * len(images))
    field_ground_truth = example.get("fields", [{}] * len(images))
    
    # Generate OCR predictions
    ocr_results = model(images=images)
    
    # Extract text and field predictions
    text_predictions = [res.get("text", "") for res in ocr_results]
    field_predictions = [res.get("fields", {}) for res in ocr_results]
    
    return {
        "image_paths": example.get("image_path", [""] * len(images)),
        "ground_truth": ground_truth,
        "prediction": text_predictions,
        "ground_truth_fields": field_ground_truth,
        "prediction_fields": field_predictions,
    }


def compute_ocr_scores(
    results: Dict[str, List[Any]], 
    lang_detect: Optional[OCRLanguageDetector] = None
) -> Dict[str, float]:
    """Compute metrics for OCR results.
    
    Args:
        results: Dictionary containing predictions and ground truth
        lang_detect: Language detector for filtering non-Japanese text
        
    Returns:
        Dictionary of metric scores
    """
    # Initialize metrics dictionary
    metrics = {}
    
    # Calculate ROUGE scores for full text
    metrics.update(rouge_ja(refs=results["ground_truth"], preds=results["prediction"]))
    
    # Calculate character accuracy
    char_accuracies = []
    for gt, pred in zip(results["ground_truth"], results["prediction"]):
        char_accuracies.append(calculate_character_accuracy(gt, pred))
    metrics["character_accuracy"] = sum(char_accuracies) / len(char_accuracies) if char_accuracies else 0.0
    
    # Calculate field extraction accuracy if available
    if "ground_truth_fields" in results and "prediction_fields" in results:
        field_accuracies = []
        field_specific_accuracies = {}
        
        for gt_fields, pred_fields in zip(results["ground_truth_fields"], results["prediction_fields"]):
            if gt_fields and pred_fields:
                overall, per_field = calculate_field_extraction_accuracy(gt_fields, pred_fields)
                field_accuracies.append(overall)
                
                # Aggregate per-field metrics
                for field_name, accuracy in per_field.items():
                    if field_name not in field_specific_accuracies:
                        field_specific_accuracies[field_name] = []
                    field_specific_accuracies[field_name].append(accuracy)
        
        # Calculate overall field extraction accuracy
        if field_accuracies:
            metrics["field_extraction_accuracy"] = sum(field_accuracies) / len(field_accuracies)
            
            # Calculate per-field average accuracies
            for field_name, accuracies in field_specific_accuracies.items():
                metrics[f"field_{field_name}_accuracy"] = sum(accuracies) / len(accuracies)
    
    # Apply language detection filtering if requested
    if lang_detect:
        # Filter for only Japanese text
        ja_preds = []
        ja_refs = []
        for ref, pred in zip(results["ground_truth"], results["prediction"]):
            if lang_detect.is_japanese(ref):
                ja_refs.append(ref)
                
                # If prediction is not Japanese, replace with empty string
                if lang_detect.is_japanese(pred):
                    ja_preds.append(pred)
                else:
                    ja_preds.append("")
                    
        # Calculate metrics for Japanese-only subset
        if ja_refs:
            ja_metrics = rouge_ja(refs=ja_refs, preds=ja_preds)
            ja_metrics = {f"{k}_ja_only": v for k, v in ja_metrics.items()}
            metrics.update(ja_metrics)
            
            # Character accuracy for Japanese-only
            ja_char_accuracies = []
            for ref, pred in zip(ja_refs, ja_preds):
                ja_char_accuracies.append(calculate_character_accuracy(ref, pred))
            
            metrics["character_accuracy_ja_only"] = sum(ja_char_accuracies) / len(ja_char_accuracies) if ja_char_accuracies else 0.0
    
    return metrics


def evaluate(
    name: str,
    model: torch.nn.Module,
    dataset: Dataset,
    loader_kwargs: dict,
    lang_detect: Optional[OCRLanguageDetector] = None,
):
    """Evaluate an OCR model on a dataset.
    
    Args:
        name: Name of the evaluation
        model: OCR model to evaluate
        dataset: Dataset containing images and ground truth
        loader_kwargs: Arguments for DataLoader
        lang_detect: Optional language detector for filtering
        
    Returns:
        EvalOutput containing metrics and results
    """
    results = []
    dataloader = DataLoader(dataset, collate_fn=dict_collation_fn, **loader_kwargs)

    for example in tqdm(dataloader, desc=f"Evaluating {name}"):
        res = _evaluate_ocr(model, example)
        results.append(res)
    
    results = flatten_list(results)
    metrics = compute_ocr_scores(results, lang_detect=lang_detect)
    
    return EvalOutput(metrics=metrics, results=results)


def process_ocr_result(
    image_path: str, 
    raw_text: str,
    predicted_fields: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Process raw OCR result into a standardized format.
    
    Args:
        image_path: Path to the image
        raw_text: Raw OCR text output
        predicted_fields: Dictionary of extracted fields
        
    Returns:
        Processed OCR result
    """
    result = {
        "image_path": image_path,
        "text": raw_text,
    }
    
    # Add predicted fields if available
    if predicted_fields:
        result["fields"] = predicted_fields
    
    return result


def generate_ocr_report(eval_output: EvalOutput, output_path: Optional[str] = None) -> str:
    """Generate a report from OCR evaluation results.
    
    Args:
        eval_output: Evaluation output
        output_path: Path to write report to
        
    Returns:
        Report text
    """
    # Format metrics section
    metrics_text = "## OCR Evaluation Metrics\n\n"
    metrics_text += "| Metric | Value |\n"
    metrics_text += "|--------|-------|\n"
    
    for metric, value in eval_output.metrics.items():
        metrics_text += f"| {metric} | {value:.2f} |\n"
    
    # Sample results section
    samples_text = "\n## Sample Results\n\n"
    
    # Show up to 5 samples
    sample_count = min(5, len(eval_output.results.get("image_paths", [])))
    
    for i in range(sample_count):
        image_path = eval_output.results["image_paths"][i]
        gt = eval_output.results["ground_truth"][i]
        pred = eval_output.results["prediction"][i]
        
        samples_text += f"### Sample {i+1}: {image_path}\n\n"
        samples_text += f"**Ground Truth:**\n```\n{gt}\n```\n\n"
        samples_text += f"**Prediction:**\n```\n{pred}\n```\n\n"
        
        # Add field extraction results if available
        if "ground_truth_fields" in eval_output.results and "prediction_fields" in eval_output.results:
            gt_fields = eval_output.results["ground_truth_fields"][i]
            pred_fields = eval_output.results["prediction_fields"][i]
            
            if gt_fields and pred_fields:
                samples_text += "**Fields:**\n\n"
                samples_text += "| Field | Ground Truth | Prediction |\n"
                samples_text += "|-------|-------------|------------|\n"
                
                for field in gt_fields:
                    gt_value = gt_fields.get(field, "")
                    pred_value = pred_fields.get(field, "")
                    samples_text += f"| {field} | {gt_value} | {pred_value} |\n"
                
                samples_text += "\n"
    
    # Combine sections
    report = metrics_text + samples_text
    
    # Write to file if requested
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
    
    return report