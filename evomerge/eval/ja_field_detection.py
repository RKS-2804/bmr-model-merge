"""Japanese invoice/receipt field detection evaluation module."""

import os
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from datasets import Dataset
from tqdm.auto import tqdm

from ..models.japanese_ocr import JapaneseOCRModel
from ..models.field_extractor import InvoiceFieldExtractor
from .metrics import (
    calculate_character_accuracy, 
    calculate_field_extraction_accuracy,
    normalize_japanese_text_for_ocr
)
from .utils import EvalOutput, dict_collation_fn, flatten_list


logger = logging.getLogger(__name__)


class JapaneseFieldDetection:
    """Evaluation class for Japanese invoice/receipt field detection.
    
    This evaluator tests a model's ability to detect and extract specific fields
    from Japanese invoices and receipts, such as company names, dates, prices, etc.
    """
    
    # Standard field definitions for Japanese invoices/receipts
    STANDARD_FIELDS = [
        "invoice_number",  # 請求書番号
        "issue_date",      # 発行日
        "due_date",        # 支払期限
        "total_amount",    # 合計金額
        "tax_amount",      # 消費税
        "subtotal",        # 小計
        "vendor_name",     # 販売者名
        "vendor_address",  # 販売者住所
        "vendor_phone",    # 販売者電話番号
        "customer_name",   # 顧客名
        "customer_address",# 顧客住所
    ]
    
    # Field display names (for reports)
    FIELD_DISPLAY_NAMES = {
        "invoice_number": "請求書番号 (Invoice Number)",
        "issue_date": "発行日 (Issue Date)",
        "due_date": "支払期限 (Due Date)",
        "total_amount": "合計金額 (Total Amount)",
        "tax_amount": "消費税 (Tax Amount)",
        "subtotal": "小計 (Subtotal)",
        "vendor_name": "販売者名 (Vendor Name)",
        "vendor_address": "販売者住所 (Vendor Address)",
        "vendor_phone": "販売者電話番号 (Vendor Phone)",
        "customer_name": "顧客名 (Customer Name)",
        "customer_address": "顧客住所 (Customer Address)",
    }
    
    def __init__(
        self,
        name: str = "ja_field_detection",
        data_path: str = "data/ja_invoices_fields",
        batch_size: int = 4,
        device: Optional[str] = None,
        output_dir: str = "results/ja_field_detection",
        field_extractor: Optional[InvoiceFieldExtractor] = None,
        visualization: bool = True,
    ):
        """Initialize Japanese field detection evaluator.
        
        Args:
            name: Name of this evaluation
            data_path: Path to evaluation data directory
            batch_size: Batch size for evaluation
            device: Device to run evaluation on
            output_dir: Directory to save results
            field_extractor: Optional field extractor to use
            visualization: Whether to generate visualizations
        """
        self.name = name
        self.data_path = data_path
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.visualization = visualization
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create visualization directory if needed
        if visualization:
            self.viz_dir = os.path.join(output_dir, "visualizations")
            os.makedirs(self.viz_dir, exist_ok=True)
        
        # Initialize field extractor if needed
        self.field_extractor = field_extractor or InvoiceFieldExtractor(device=self.device)
        
        # Try to load a Japanese font for visualizations
        self.font = None
        try:
            # Common Japanese font paths
            font_paths = [
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
                "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",  # macOS
                "C:/Windows/Fonts/msgothic.ttc",  # Windows
                "C:/Windows/Fonts/YuGothR.ttc",   # Windows
            ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    self.font = ImageFont.truetype(font_path, 16)
                    break
                    
            if self.font is None:
                # Fallback to default
                self.font = ImageFont.load_default()
                logger.warning("Japanese font not found, using default font for visualizations")
        except Exception as e:
            logger.warning(f"Error loading font: {e}")
            self.font = ImageFont.load_default()
    
    def _load_dataset(self) -> Dataset:
        """Load field detection dataset.
        
        Returns:
            Dataset for evaluation
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset path not found: {self.data_path}")
            
        # Check for metadata file
        metadata_path = os.path.join(self.data_path, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            
        # Prepare dataset items
        items = []
        for item in metadata:
            image_path = os.path.join(self.data_path, item["image_filename"])
            
            # Skip if image doesn't exist
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                continue
                
            # Add dataset item with fields
            data_item = {
                "image_path": image_path,
                "image": Image.open(image_path).convert("RGB"),
                "text": item.get("text", ""),
                "fields": item.get("fields", {}),
            }
            
            # Add field boxes if available
            if "field_boxes" in item:
                data_item["field_boxes"] = item["field_boxes"]
                
            items.append(data_item)
        
        return Dataset.from_list(items)
    
    def _evaluate_field_detection(self, model, example: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """Evaluate field detection on a batch of images.
        
        Args:
            model: OCR model
            example: Dictionary with images and ground truth
            
        Returns:
            Dictionary with predictions and ground truth
        """
        # Extract images
        images = example["image"]
        image_paths = example["image_path"]
        ground_truth_fields = example.get("fields", [{} for _ in range(len(images))])
        
        # Run OCR inference
        ocr_results = model(images=images)
        
        # Extract fields using field extractor if the model doesn't extract fields itself
        field_predictions = []
        text_predictions = []
        
        for result in ocr_results:
            # Extract text
            text = result.get("text", "")
            text_predictions.append(text)
            
            # Get fields if already extracted by the model
            if "fields" in result:
                field_predictions.append(result["fields"])
            else:
                # Extract fields using the field extractor
                fields = self.field_extractor.extract_fields(text)
                field_predictions.append(fields)
        
        # Prepare evaluation results
        result = {
            "image_path": image_paths,
            "prediction": text_predictions,
            "ground_truth_fields": ground_truth_fields,
            "prediction_fields": field_predictions,
        }
        
        # Add field boxes if available
        if "field_boxes" in example:
            result["field_boxes"] = example["field_boxes"]
            
        return result
    
    def compute_metrics(self, results: Dict[str, List[Any]]) -> Dict[str, float]:
        """Compute field detection metrics.
        
        Args:
            results: Dictionary with predictions and ground truth
            
        Returns:
            Dictionary of metrics
        """
        # Initialize metrics
        metrics = {}
        
        # Calculate overall field extraction accuracy
        field_accuracies = []
        field_specific_accuracies = {field: [] for field in self.STANDARD_FIELDS}
        field_presence_accuracy = {field: {"tp": 0, "fp": 0, "fn": 0} for field in self.STANDARD_FIELDS}
        
        for gt_fields, pred_fields in zip(results["ground_truth_fields"], results["prediction_fields"]):
            # Calculate overall field accuracy
            if gt_fields:
                overall, per_field = calculate_field_extraction_accuracy(gt_fields, pred_fields)
                field_accuracies.append(overall)
                
                # Track field-specific accuracies
                for field, accuracy in per_field.items():
                    if field in field_specific_accuracies:
                        field_specific_accuracies[field].append(accuracy)
                
                # Track field presence/absence (detection)
                for field in self.STANDARD_FIELDS:
                    if field in gt_fields and field in pred_fields:
                        # True positive
                        field_presence_accuracy[field]["tp"] += 1
                    elif field not in gt_fields and field in pred_fields:
                        # False positive
                        field_presence_accuracy[field]["fp"] += 1
                    elif field in gt_fields and field not in pred_fields:
                        # False negative
                        field_presence_accuracy[field]["fn"] += 1
        
        # Calculate overall field extraction accuracy
        if field_accuracies:
            metrics["overall_field_accuracy"] = sum(field_accuracies) / len(field_accuracies)
        
        # Calculate per-field accuracy
        for field in self.STANDARD_FIELDS:
            if field_specific_accuracies[field]:
                metrics[f"field_{field}_accuracy"] = (
                    sum(field_specific_accuracies[field]) / 
                    len(field_specific_accuracies[field])
                )
        
        # Calculate precision, recall, and F1 for field detection
        for field in self.STANDARD_FIELDS:
            tp = field_presence_accuracy[field]["tp"]
            fp = field_presence_accuracy[field]["fp"]
            fn = field_presence_accuracy[field]["fn"]
            
            # Precision
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            metrics[f"field_{field}_precision"] = precision * 100
            
            # Recall
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics[f"field_{field}_recall"] = recall * 100
            
            # F1 Score
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            metrics[f"field_{field}_f1"] = f1 * 100
        
        # Calculate average precision, recall, and F1 across all fields
        avg_precision = sum(metrics[f"field_{field}_precision"] for field in self.STANDARD_FIELDS) / len(self.STANDARD_FIELDS)
        avg_recall = sum(metrics[f"field_{field}_recall"] for field in self.STANDARD_FIELDS) / len(self.STANDARD_FIELDS)
        avg_f1 = sum(metrics[f"field_{field}_f1"] for field in self.STANDARD_FIELDS) / len(self.STANDARD_FIELDS)
        
        metrics["avg_field_precision"] = avg_precision
        metrics["avg_field_recall"] = avg_recall
        metrics["avg_field_f1"] = avg_f1
        
        # If field boxes are available, calculate spatial accuracy
        if "field_boxes" in results:
            spatial_accuracies = []
            for i, (gt_field_boxes, pred_fields) in enumerate(zip(results["field_boxes"], results["prediction_fields"])):
                # TODO: Implement spatial accuracy calculation if the model provides predicted field locations
                pass
                
            if spatial_accuracies:
                metrics["field_spatial_accuracy"] = sum(spatial_accuracies) / len(spatial_accuracies)
        
        return metrics
    
    def visualize_field_detection(self, results: Dict[str, List[Any]], indices: List[int]) -> None:
        """Generate visualization of field detection results.
        
        Args:
            results: Evaluation results
            indices: Indices of samples to visualize
        """
        if not self.visualization:
            return
            
        for idx in indices:
            # Get image path and load image
            image_path = results["image_path"][idx]
            img = Image.open(image_path).convert("RGB")
            img_draw = ImageDraw.Draw(img)
            
            # Get predictions
            pred_fields = results["prediction_fields"][idx]
            gt_fields = results["ground_truth_fields"][idx]
            
            # Create field comparison visualization
            plt.figure(figsize=(10, 8))
            plt.title(f"Field Detection Results: {Path(image_path).stem}")
            
            # Set up table data
            field_names = []
            gt_values = []
            pred_values = []
            match_status = []
            
            # Collect all fields present in either ground truth or predictions
            all_fields = set(gt_fields.keys()) | set(pred_fields.keys())
            
            for field in all_fields:
                display_name = self.FIELD_DISPLAY_NAMES.get(field, field)
                field_names.append(display_name)
                
                gt_value = gt_fields.get(field, "")
                pred_value = pred_fields.get(field, "")
                
                # Truncate very long values
                if len(gt_value) > 30:
                    gt_value = gt_value[:27] + "..."
                if len(pred_value) > 30:
                    pred_value = pred_value[:27] + "..."
                    
                gt_values.append(gt_value)
                pred_values.append(pred_value)
                
                # Determine match status
                if field not in gt_fields and field not in pred_fields:
                    match_status.append("N/A")
                elif field not in gt_fields:
                    match_status.append("False Positive")
                elif field not in pred_fields:
                    match_status.append("False Negative")
                elif gt_value.strip() == pred_value.strip():
                    match_status.append("Exact Match")
                else:
                    # Calculate character accuracy for partial match
                    char_acc = calculate_character_accuracy(gt_value, pred_value)
                    if char_acc > 70:
                        match_status.append(f"Partial ({char_acc:.1f}%)")
                    else:
                        match_status.append(f"Mismatch ({char_acc:.1f}%)")
            
            # Create table
            plt.axis('off')
            table_data = [field_names, gt_values, pred_values, match_status]
            table = plt.table(
                cellText=list(zip(*table_data)),
                rowLabels=None,
                colLabels=["Field", "Ground Truth", "Prediction", "Status"],
                loc='center',
                cellLoc='left',
                colWidths=[0.3, 0.3, 0.3, 0.2]
            )
            
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            
            # Color cells based on match status
            for row_idx, status in enumerate(match_status):
                cell = table[row_idx + 1, 3]  # +1 for header row
                if status == "Exact Match":
                    cell.set_facecolor((0.7, 1.0, 0.7))
                elif status.startswith("Partial"):
                    cell.set_facecolor((1.0, 1.0, 0.7))
                elif status == "False Positive" or status == "False Negative" or status.startswith("Mismatch"):
                    cell.set_facecolor((1.0, 0.7, 0.7))
            
            # Save visualization
            viz_filename = os.path.join(self.viz_dir, f"{Path(image_path).stem}_fields.png")
            plt.savefig(viz_filename, dpi=150, bbox_inches="tight")
            plt.close()
            
            # Create field annotation visualization if field boxes are available
            if "field_boxes" in results and idx < len(results.get("field_boxes", [])):
                field_boxes = results["field_boxes"][idx]
                if field_boxes:
                    field_viz = img.copy()
                    draw = ImageDraw.Draw(field_viz)
                    
                    # Draw field locations with different colors for each field type
                    colors = [
                        "red", "blue", "green", "orange", "purple",
                        "yellow", "cyan", "magenta", "brown", "pink"
                    ]
                    
                    for field, box in field_boxes.items():
                        color_idx = hash(field) % len(colors)
                        color = colors[color_idx]
                        
                        # Draw box
                        draw.rectangle(box, outline=color, width=3)
                        
                        # Draw field label
                        label_pos = (box[0], box[1] - 20) if box[1] > 20 else (box[0], box[3] + 5)
                        draw.text(label_pos, field, fill=color, font=self.font)
                    
                    # Save field visualization
                    field_viz_filename = os.path.join(self.viz_dir, f"{Path(image_path).stem}_field_boxes.png")
                    field_viz.save(field_viz_filename)
    
    def generate_report(self, eval_output: EvalOutput, output_path: str) -> str:
        """Generate evaluation report.
        
        Args:
            eval_output: Evaluation output
            output_path: Path to save report
            
        Returns:
            Report text
        """
        # Format metrics section
        report = "# Japanese Invoice/Receipt Field Detection Evaluation\n\n"
        report += "## Overall Metrics\n\n"
        report += "| Metric | Value |\n"
        report += "|--------|-------|\n"
        
        # Add main metrics first
        main_metrics = [
            "overall_field_accuracy", "avg_field_precision", 
            "avg_field_recall", "avg_field_f1"
        ]
        
        for metric in main_metrics:
            if metric in eval_output.metrics:
                report += f"| {metric} | {eval_output.metrics[metric]:.2f} |\n"
        
        # Per-field metrics
        report += "\n## Per-field Metrics\n\n"
        report += "| Field | Accuracy | Precision | Recall | F1 Score |\n"
        report += "|-------|----------|-----------|--------|----------|\n"
        
        for field in self.STANDARD_FIELDS:
            field_display = self.FIELD_DISPLAY_NAMES.get(field, field)
            
            # Gather metrics for this field
            accuracy = eval_output.metrics.get(f"field_{field}_accuracy", 0)
            precision = eval_output.metrics.get(f"field_{field}_precision", 0)
            recall = eval_output.metrics.get(f"field_{field}_recall", 0)
            f1 = eval_output.metrics.get(f"field_{field}_f1", 0)
            
            report += f"| {field_display} | {accuracy:.2f} | {precision:.2f} | {recall:.2f} | {f1:.2f} |\n"
        
        # Sample results
        report += "\n## Sample Results\n\n"
        
        # Show up to 5 samples
        sample_count = min(5, len(eval_output.results["image_path"]))
        
        for i in range(sample_count):
            report += f"### Sample {i+1}\n\n"
            report += f"**Image:** {eval_output.results['image_path'][i]}\n\n"
            
            # Field extraction results
            gt_fields = eval_output.results["ground_truth_fields"][i]
            pred_fields = eval_output.results["prediction_fields"][i]
            
            report += "**Field Extraction Results:**\n\n"
            report += "| Field | Ground Truth | Prediction | Status |\n"
            report += "|-------|-------------|------------|--------|\n"
            
            # Combine all fields from ground truth and prediction
            all_fields = set(gt_fields.keys()) | set(pred_fields.keys())
            
            for field in sorted(all_fields):
                field_display = self.FIELD_DISPLAY_NAMES.get(field, field)
                gt_value = gt_fields.get(field, "")
                pred_value = pred_fields.get(field, "")
                
                # Determine match status
                status = ""
                if field not in gt_fields and field not in pred_fields:
                    status = "N/A"
                elif field not in gt_fields:
                    status = "False Positive"
                elif field not in pred_fields:
                    status = "False Negative"
                elif gt_value.strip() == pred_value.strip():
                    status = "✓ Exact Match"
                else:
                    # Calculate character accuracy for partial match
                    char_acc = calculate_character_accuracy(gt_value, pred_value)
                    if char_acc > 70:
                        status = f"△ Partial ({char_acc:.1f}%)"
                    else:
                        status = f"✗ Mismatch ({char_acc:.1f}%)"
                
                report += f"| {field_display} | {gt_value} | {pred_value} | {status} |\n"
                
            # Add visualization links if available
            if self.visualization:
                image_name = Path(eval_output.results['image_path'][i]).stem
                report += f"\n**Visualizations:**\n\n"
                report += f"- [Field Results](visualizations/{image_name}_fields.png)\n"
                
                if os.path.exists(os.path.join(self.viz_dir, f"{image_name}_field_boxes.png")):
                    report += f"- [Field Locations](visualizations/{image_name}_field_boxes.png)\n"
                
            # Add separator
            report += "\n---\n\n"
        
        # Write report to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
            
        return report
    
    def __call__(self, model) -> EvalOutput:
        """Run field detection evaluation on model.
        
        Args:
            model: Model to evaluate
            
        Returns:
            Evaluation output
        """
        # Load dataset
        logger.info(f"Loading dataset from {self.data_path}")
        dataset = self._load_dataset()
        logger.info(f"Loaded {len(dataset)} examples")
        
        # Setup dataloader
        loader_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": 0
        }
        
        # Evaluate
        results = []
        dataloader = torch.utils.data.DataLoader(
            dataset, collate_fn=dict_collation_fn, **loader_kwargs
        )
        
        for example in tqdm(dataloader, desc=f"Evaluating {self.name}"):
            res = self._evaluate_field_detection(model, example)
            results.append(res)
        
        # Flatten and compute metrics
        results = flatten_list(results)
        metrics = self.compute_metrics(results)
        
        # Create output
        output = EvalOutput(metrics=metrics, results=results)
        
        # Generate visualizations for a few examples
        if self.visualization and len(results.get("image_path", [])) > 0:
            # Visualize first 10 examples or fewer if dataset is smaller
            viz_indices = list(range(min(10, len(results["image_path"]))))
            self.visualize_field_detection(results, viz_indices)
        
        # Generate report
        report_path = os.path.join(self.output_dir, f"{self.name}_report.md")
        self.generate_report(output, report_path)
        logger.info(f"Report saved to {report_path}")
        
        return output