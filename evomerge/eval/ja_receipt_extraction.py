"""Japanese receipt text extraction evaluation module."""

import os
import json
import logging
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from datasets import Dataset
from tqdm.auto import tqdm

from ..models.japanese_ocr import JapaneseOCRModel
from .metrics import (
    rouge_ja, 
    calculate_character_accuracy, 
    normalize_japanese_text_for_ocr,
    OCRLanguageDetector
)
from .utils import EvalOutput, dict_collation_fn, flatten_list


logger = logging.getLogger(__name__)


class JapaneseReceiptExtraction:
    """Evaluation class for Japanese receipt text extraction.
    
    This evaluator tests an OCR model's ability to extract text from Japanese
    receipts, with special attention to layout understanding and challenging
    Japanese characters.
    """
    
    def __init__(
        self,
        name: str = "ja_receipt_extraction",
        data_path: str = "data/ja_receipts",
        batch_size: int = 8,
        device: Optional[str] = None,
        output_dir: str = "results/ja_receipt_extraction",
        normalize_text: bool = True,
        visualization: bool = True,
    ):
        """Initialize Japanese receipt extraction evaluator.
        
        Args:
            name: Name of this evaluation
            data_path: Path to evaluation data directory
            batch_size: Batch size for evaluation
            device: Device to run evaluation on
            output_dir: Directory to save results
            normalize_text: Whether to normalize text before comparison
            visualization: Whether to generate visualizations
        """
        self.name = name
        self.data_path = data_path
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.normalize_text = normalize_text
        self.visualization = visualization
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create visualization directory if needed
        if visualization:
            self.viz_dir = os.path.join(output_dir, "visualizations")
            os.makedirs(self.viz_dir, exist_ok=True)
        
        # Language detector for Japanese-specific processing
        self.lang_detector = OCRLanguageDetector()
        
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
        """Load receipt dataset.
        
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
                
            # Add dataset item
            data_item = {
                "image_path": image_path,
                "image": Image.open(image_path).convert("RGB"),
                "text": item["text"],
            }
            
            # Add bounding box ground truth if available
            if "boxes" in item:
                data_item["boxes"] = item["boxes"]
                
            # Add line segments if available
            if "lines" in item:
                data_item["lines"] = item["lines"]
                
            items.append(data_item)
        
        return Dataset.from_list(items)
    
    def _evaluate_extraction(self, model, example: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """Evaluate text extraction on a batch of images.
        
        Args:
            model: OCR model
            example: Dictionary with images and ground truth
            
        Returns:
            Dictionary with predictions and ground truth
        """
        # Extract images
        images = example["image"]
        image_paths = example["image_path"]
        ground_truth = example.get("text", [""] * len(images))
        
        # Run OCR inference
        ocr_results = model(images=images)
        
        # Process results
        text_predictions = []
        boxes_predictions = []
        scores_predictions = []
        
        for result in ocr_results:
            # Extract text
            text = result.get("text", "")
            if self.normalize_text:
                text = normalize_japanese_text_for_ocr(text)
            text_predictions.append(text)
            
            # Extract boxes and scores if available
            boxes = result.get("boxes", [])
            boxes_predictions.append(boxes)
            
            scores = result.get("scores", [])
            scores_predictions.append(scores)
            
        # Prepare evaluation results
        result = {
            "image_path": image_paths,
            "ground_truth": ground_truth,
            "prediction": text_predictions,
            "boxes": boxes_predictions,
            "scores": scores_predictions,
        }
        
        # Add ground truth boxes if available
        if "boxes" in example:
            result["ground_truth_boxes"] = example["boxes"]
            
        # Add line segments if available
        if "lines" in example:
            result["ground_truth_lines"] = example["lines"]
            
        return result
    
    def compute_metrics(self, results: Dict[str, List[Any]]) -> Dict[str, float]:
        """Compute text extraction metrics.
        
        Args:
            results: Dictionary with predictions and ground truth
            
        Returns:
            Dictionary of metrics
        """
        # Initialize metrics
        metrics = {}
        
        # Calculate ROUGE scores
        metrics.update(rouge_ja(refs=results["ground_truth"], preds=results["prediction"]))
        
        # Calculate character error rate and accuracy
        char_accuracies = []
        for gt, pred in zip(results["ground_truth"], results["prediction"]):
            char_accuracies.append(calculate_character_accuracy(gt, pred))
        metrics["character_accuracy"] = sum(char_accuracies) / len(char_accuracies) if char_accuracies else 0.0
        metrics["character_error_rate"] = 100.0 - metrics["character_accuracy"]
        
        # Handle layout evaluation if ground truth boxes are available
        if "ground_truth_boxes" in results and "boxes" in results:
            layout_scores = []
            for gt_boxes, pred_boxes in zip(results["ground_truth_boxes"], results["boxes"]):
                if gt_boxes and pred_boxes:
                    # Calculate IoU-based layout score
                    layout_score = self._calculate_layout_score(gt_boxes, pred_boxes)
                    layout_scores.append(layout_score)
                    
            if layout_scores:
                metrics["layout_accuracy"] = sum(layout_scores) / len(layout_scores)
                
        # Calculate text direction detection accuracy (vertical vs. horizontal)
        if "ground_truth_lines" in results:
            orientation_correct = 0
            orientation_total = 0
            
            for gt_lines, pred_boxes in zip(results.get("ground_truth_lines", []), results["boxes"]):
                if gt_lines and pred_boxes:
                    # Detect if prediction correctly identified vertical/horizontal text segments
                    orientation_score = self._calculate_orientation_accuracy(gt_lines, pred_boxes)
                    orientation_correct += orientation_score[0]
                    orientation_total += orientation_score[1]
                    
            if orientation_total > 0:
                metrics["orientation_accuracy"] = (orientation_correct / orientation_total) * 100
        
        # Calculate Japanese-specific metrics
        metrics.update(self._calculate_japanese_specific_metrics(results))
        
        return metrics
    
    def _calculate_layout_score(self, gt_boxes: List[List[float]], pred_boxes: List[List[float]]) -> float:
        """Calculate layout accuracy based on bounding box IoU.
        
        Args:
            gt_boxes: Ground truth bounding boxes [[x1, y1, x2, y2], ...]
            pred_boxes: Predicted bounding boxes [[x1, y1, x2, y2], ...]
            
        Returns:
            Layout accuracy score (0-100)
        """
        if not gt_boxes or not pred_boxes:
            return 0.0
            
        # Calculate IoU for each GT box with best matching pred box
        ious = []
        
        for gt_box in gt_boxes:
            best_iou = 0.0
            
            for pred_box in pred_boxes:
                # Calculate intersection
                x1 = max(gt_box[0], pred_box[0])
                y1 = max(gt_box[1], pred_box[1])
                x2 = min(gt_box[2], pred_box[2])
                y2 = min(gt_box[3], pred_box[3])
                
                if x1 < x2 and y1 < y2:
                    intersection = (x2 - x1) * (y2 - y1)
                    
                    # Calculate areas
                    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                    
                    # Calculate IoU
                    iou = intersection / (gt_area + pred_area - intersection)
                    best_iou = max(best_iou, iou)
            
            ious.append(best_iou)
            
        # Average IoU
        return sum(ious) / len(ious) * 100 if ious else 0.0
    
    def _calculate_orientation_accuracy(self, gt_lines: List[Dict[str, Any]], pred_boxes: List[List[float]]) -> Tuple[int, int]:
        """Calculate text orientation detection accuracy.
        
        Args:
            gt_lines: Ground truth text lines with orientation information
            pred_boxes: Predicted bounding boxes [[x1, y1, x2, y2], ...]
            
        Returns:
            Tuple of (correct_orientations, total_orientations)
        """
        correct = 0
        total = 0
        
        for line in gt_lines:
            if "orientation" in line:
                total += 1
                
                # Find matching predicted box
                gt_box = line.get("box", [0, 0, 0, 0])
                best_match = None
                best_iou = 0.0
                
                for pred_box in pred_boxes:
                    # Calculate IoU
                    x1 = max(gt_box[0], pred_box[0])
                    y1 = max(gt_box[1], pred_box[1])
                    x2 = min(gt_box[2], pred_box[2])
                    y2 = min(gt_box[3], pred_box[3])
                    
                    if x1 < x2 and y1 < y2:
                        intersection = (x2 - x1) * (y2 - y1)
                        gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                        pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                        iou = intersection / (gt_area + pred_area - intersection)
                        
                        if iou > best_iou:
                            best_iou = iou
                            best_match = pred_box
                
                if best_match and best_iou > 0.5:
                    # Determine predicted orientation based on box dimensions
                    pred_width = best_match[2] - best_match[0]
                    pred_height = best_match[3] - best_match[1]
                    pred_orientation = "vertical" if pred_height > pred_width * 1.5 else "horizontal"
                    
                    # Check if orientation matches
                    if pred_orientation == line["orientation"]:
                        correct += 1
                        
        return correct, total
    
    def _calculate_japanese_specific_metrics(self, results: Dict[str, List[Any]]) -> Dict[str, float]:
        """Calculate Japanese-specific OCR metrics.
        
        Args:
            results: Dictionary with predictions and ground truth
            
        Returns:
            Dictionary of Japanese-specific metrics
        """
        metrics = {}
        
        # Check for complex Japanese characters
        complex_char_accuracies = []
        
        # Characters with high stroke count or complex shapes
        complex_chars = set("難漢鬱璽識臓鑑鬱贈藤麗曖")
        
        for gt, pred in zip(results["ground_truth"], results["prediction"]):
            # Find complex characters in ground truth
            complex_in_gt = [c for c in gt if c in complex_chars]
            if not complex_in_gt:
                continue
                
            # Check character accuracy for complex characters only
            correct = 0
            for c in complex_in_gt:
                if c in pred:
                    correct += 1
            
            accuracy = (correct / len(complex_in_gt)) * 100 if complex_in_gt else 0
            complex_char_accuracies.append(accuracy)
            
        if complex_char_accuracies:
            metrics["complex_character_accuracy"] = sum(complex_char_accuracies) / len(complex_char_accuracies)
        
        # Special symbol recognition (¥, 〒, 税, etc.)
        special_symbols = set("¥〒税％＊※◎○●△▲▽▼◆□■☆★♪♭♯")
        symbol_accuracies = []
        
        for gt, pred in zip(results["ground_truth"], results["prediction"]):
            # Find special symbols in ground truth
            symbols_in_gt = [c for c in gt if c in special_symbols]
            if not symbols_in_gt:
                continue
                
            # Check symbol accuracy
            correct = 0
            for s in symbols_in_gt:
                if s in pred:
                    correct += 1
            
            accuracy = (correct / len(symbols_in_gt)) * 100 if symbols_in_gt else 0
            symbol_accuracies.append(accuracy)
            
        if symbol_accuracies:
            metrics["symbol_recognition_accuracy"] = sum(symbol_accuracies) / len(symbol_accuracies)
        
        # Calculate kanji-to-kana ratio accuracy
        # This measures how well the model preserves the distribution of character types
        ratio_diffs = []
        
        for gt, pred in zip(results["ground_truth"], results["prediction"]):
            if not gt or not pred:
                continue
                
            # Calculate kanji ratio in ground truth
            import unicodedata
            
            def is_kanji(c):
                return "CJK UNIFIED IDEOGRAPH" in unicodedata.name(c, "")
                
            gt_kanji_count = sum(1 for c in gt if is_kanji(c))
            gt_kanji_ratio = gt_kanji_count / len(gt) if gt else 0
            
            pred_kanji_count = sum(1 for c in pred if is_kanji(c))
            pred_kanji_ratio = pred_kanji_count / len(pred) if pred else 0
            
            # Calculate difference in ratios
            ratio_diff = abs(gt_kanji_ratio - pred_kanji_ratio)
            ratio_diffs.append(ratio_diff)
            
        if ratio_diffs:
            # Convert to accuracy (lower difference = higher accuracy)
            avg_ratio_diff = sum(ratio_diffs) / len(ratio_diffs)
            metrics["kanji_ratio_accuracy"] = 100 * (1 - avg_ratio_diff)
            
        return metrics
    
    def visualize_results(self, results: Dict[str, List[Any]], indices: List[int]) -> None:
        """Generate visualization of OCR results.
        
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
            pred_text = results["prediction"][idx]
            boxes = results["boxes"][idx] if idx < len(results["boxes"]) else []
            
            # Create figure with subplots
            plt.figure(figsize=(12, 8))
            
            # Plot original image
            plt.subplot(1, 2, 1)
            plt.imshow(np.array(img))
            plt.title("Original Image")
            plt.axis("off")
            
            # Plot image with bounding boxes
            img_with_boxes = img.copy()
            draw = ImageDraw.Draw(img_with_boxes)
            
            # Draw boxes
            for i, box in enumerate(boxes):
                # Draw bounding box
                draw.rectangle(box, outline="red", width=2)
                
                # Add index number
                draw.text((box[0], box[1]), str(i), fill="red", font=self.font)
            
            plt.subplot(1, 2, 2)
            plt.imshow(np.array(img_with_boxes))
            plt.title("OCR Detection")
            plt.axis("off")
            
            # Add text at the bottom
            plt.figtext(0.5, 0.01, f"Prediction: {pred_text[:100]}{'...' if len(pred_text) > 100 else ''}", 
                      wrap=True, horizontalalignment='center', fontsize=12)
            
            # Save visualization
            viz_filename = os.path.join(self.viz_dir, f"{Path(image_path).stem}_viz.png")
            plt.savefig(viz_filename, dpi=150, bbox_inches="tight")
            plt.close()
            
            # Create text overlay image
            text_overlay = img.copy()
            draw = ImageDraw.Draw(text_overlay)
            
            # Draw extracted text at box locations
            if "raw_texts" in results and idx < len(results.get("raw_texts", [])):
                raw_texts = results["raw_texts"][idx]
                for i, (box, text) in enumerate(zip(boxes, raw_texts)):
                    # Draw semi-transparent background
                    overlay = Image.new("RGBA", text_overlay.size, (0, 0, 0, 0))
                    overlay_draw = ImageDraw.Draw(overlay)
                    overlay_draw.rectangle(box, fill=(255, 255, 255, 128))
                    text_overlay = Image.alpha_composite(text_overlay.convert("RGBA"), overlay).convert("RGB")
                    
                    draw = ImageDraw.Draw(text_overlay)
                    draw.text((box[0], box[1]), text, fill="blue", font=self.font)
            
            # Save text overlay visualization
            overlay_filename = os.path.join(self.viz_dir, f"{Path(image_path).stem}_overlay.png")
            text_overlay.save(overlay_filename)
            
            # Create error highlighting visualization if ground truth is available
            if "ground_truth" in results and results["ground_truth"][idx]:
                gt_text = results["ground_truth"][idx]
                error_viz = img.copy()
                error_draw = ImageDraw.Draw(error_viz)
                
                # Mark errors on text overlay image
                # This is a simplified approach - in practice, you would use more sophisticated
                # alignment algorithms to match ground truth with predictions
                for i, (box, text) in enumerate(zip(boxes, raw_texts if "raw_texts" in results else [""]*len(boxes))):
                    # Check if text appears in ground truth
                    if text and text in gt_text:
                        # Correct text - draw green box
                        error_draw.rectangle(box, outline="green", width=2)
                    else:
                        # Error - draw red box
                        error_draw.rectangle(box, outline="red", width=2)
                
                # Save error visualization
                error_filename = os.path.join(self.viz_dir, f"{Path(image_path).stem}_errors.png")
                error_viz.save(error_filename)
    
    def generate_report(self, eval_output: EvalOutput, output_path: str) -> str:
        """Generate evaluation report.
        
        Args:
            eval_output: Evaluation output
            output_path: Path to save report
            
        Returns:
            Report text
        """
        # Format metrics section
        report = "# Japanese Receipt Text Extraction Evaluation\n\n"
        report += "## Metrics\n\n"
        report += "| Metric | Value |\n"
        report += "|--------|-------|\n"
        
        for metric, value in eval_output.metrics.items():
            report += f"| {metric} | {value:.2f} |\n"
        
        # Sample results
        report += "\n## Sample Results\n\n"
        
        # Show up to 5 samples
        sample_count = min(5, len(eval_output.results["image_path"]))
        
        for i in range(sample_count):
            report += f"### Sample {i+1}\n\n"
            report += f"**Image:** {eval_output.results['image_path'][i]}\n\n"
            
            # Add ground truth if available
            if "ground_truth" in eval_output.results:
                report += f"**Ground Truth:**\n```\n{eval_output.results['ground_truth'][i]}\n```\n\n"
                
            # Add prediction
            report += f"**Prediction:**\n```\n{eval_output.results['prediction'][i]}\n```\n\n"
            
            # Add character accuracy
            gt = eval_output.results['ground_truth'][i]
            pred = eval_output.results['prediction'][i]
            char_acc = calculate_character_accuracy(gt, pred)
            report += f"**Character Accuracy:** {char_acc:.2f}%\n\n"
            
            # Add visualization links if available
            if self.visualization:
                image_name = Path(eval_output.results['image_path'][i]).stem
                report += f"**Visualizations:**\n\n"
                report += f"- [Detection](visualizations/{image_name}_viz.png)\n"
                report += f"- [Text Overlay](visualizations/{image_name}_overlay.png)\n"
                report += f"- [Error Analysis](visualizations/{image_name}_errors.png)\n\n"
                
            # Add separator
            report += "---\n\n"
        
        # Write report to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
            
        return report
    
    def __call__(self, model) -> EvalOutput:
        """Run receipt extraction evaluation on model.
        
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
            res = self._evaluate_extraction(model, example)
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
            self.visualize_results(results, viz_indices)
        
        # Generate report
        report_path = os.path.join(self.output_dir, f"{self.name}_report.md")
        self.generate_report(output, report_path)
        logger.info(f"Report saved to {report_path}")
        
        return output