"""Japanese invoice Visual QA evaluation module."""

import os
import logging
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import AutoProcessor

from ..models.japanese_ocr import JapaneseOCRModel
from .metrics import rouge_ja, OCRLanguageDetector, calculate_character_accuracy
from .utils import EvalOutput, process_ocr_result, evaluate, dict_collation_fn, generate_ocr_report


logger = logging.getLogger(__name__)


class JapaneseInvoiceVQA:
    """Evaluation class for Japanese invoice visual question answering tasks.
    
    This evaluator tests a model's ability to answer questions about Japanese invoices
    based on visual information in the document.
    """
    
    def __init__(
        self,
        name: str = "ja_invoice_vqa",
        data_path: str = "data/ja_invoice_vqa",
        batch_size: int = 4,
        device: Optional[str] = None,
        output_dir: str = "results/ja_invoice_vqa",
        question_templates: Optional[Dict[str, str]] = None,
    ):
        """Initialize Japanese invoice VQA evaluator.
        
        Args:
            name: Name of this evaluation
            data_path: Path to evaluation data directory
            batch_size: Batch size for evaluation
            device: Device to run evaluation on
            output_dir: Directory to save results
            question_templates: Templates for invoice questions
        """
        self.name = name
        self.data_path = data_path
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Default question templates for invoices
        self.question_templates = question_templates or {
            "invoice_number": "この請求書の番号はなんですか？",  # What is the invoice number?
            "issue_date": "この請求書はいつ発行されましたか？",  # When was this invoice issued?
            "total_amount": "この請求書の合計金額はいくらですか？",  # What is the total amount of this invoice?
            "vendor_name": "販売者の名前は何ですか？",  # What is the vendor's name?
            "customer_name": "顧客の名前は何ですか？",  # What is the customer's name?
        }
        
        # Language detector for filtering
        self.lang_detector = OCRLanguageDetector()
        
    def _load_dataset(self) -> Dataset:
        """Load invoice VQA dataset.
        
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
                
            # Add each question-answer pair as a separate item
            for q_type, question in self.question_templates.items():
                if q_type in item["answers"]:
                    items.append({
                        "image_path": image_path,
                        "image": Image.open(image_path).convert("RGB"),
                        "question": question,
                        "question_type": q_type,
                        "answer": item["answers"][q_type]
                    })
        
        return Dataset.from_list(items)
    
    def _evaluate_vqa(self, model, example: Dict[str, List[Any]]) -> Dict[str, List[Union[str, bool]]]:
        """Evaluate VQA model on a batch of examples.
        
        Args:
            model: VQA model
            example: Dictionary with images and questions
            
        Returns:
            Dictionary with predictions and ground truth
        """
        # Extract batch items
        images = example["image"]
        questions = example["question"]
        image_paths = example["image_path"]
        answers = example["answer"]
        question_types = example["question_type"]
        
        # Get predictions from model
        predictions = []
        for img, question in zip(images, questions):
            if hasattr(model, "process_vqa"):
                # If model has VQA capability
                pred = model.process_vqa(img, question)
                predictions.append(pred)
            else:
                # Fallback to OCR + text search
                ocr_result = model([img])[0]
                text = ocr_result.get("text", "")
                
                # Simple extraction based on question type
                q_type = question_types[predictions.index(None) if predictions else 0]
                pred = self._extract_answer_from_text(text, q_type)
                predictions.append(pred)
                
        return {
            "image_path": image_paths,
            "question": questions,
            "question_type": question_types,
            "ground_truth": answers,
            "prediction": predictions
        }
            
    def _extract_answer_from_text(self, text: str, question_type: str) -> str:
        """Extract answer from OCR text based on question type.
        
        Args:
            text: OCR text
            question_type: Type of question
            
        Returns:
            Extracted answer
        """
        # Use patterns similar to those in InvoiceFieldExtractor
        import re
        
        patterns = {
            "invoice_number": [
                re.compile(r"請求書番号[:\s]*([\w\d\-]{5,14})"),
                re.compile(r"インボイス番号[:\s]*([\w\d\-]{5,14})"),
            ],
            "issue_date": [
                re.compile(r"(?:発行日|日付)[:\s]*(?:令和|平成|大正|昭和)?(\d{1,2})年(\d{1,2})月(\d{1,2})日"),
                re.compile(r"(?:発行日|日付)[:\s]*(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})"),
            ],
            "total_amount": [
                re.compile(r"(?:合計|総額|総計)[:\s]*￥?([0-9,]+)(?:円)?"),
                re.compile(r"(?:合計|総額|総計)[:\s]*([0-9,]+)円"),
            ],
            "vendor_name": [
                re.compile(r"(?:販売者|発行者|請求元)[:\s]*(.+?)(?=[\n\r]|$)"),
            ],
            "customer_name": [
                re.compile(r"(?:宛先|請求先)[:\s]*(.+?)(?=[\n\r]|$)"),
            ],
        }
        
        # Try to match patterns
        if question_type in patterns:
            for pattern in patterns[question_type]:
                match = pattern.search(text)
                if match:
                    if question_type == "issue_date" and len(match.groups()) >= 3:
                        # Format date consistently
                        year, month, day = match.groups()[:3]
                        if len(year) == 1 or len(year) == 2:  # Japanese era year
                            # Convert era year to western calendar (simplified)
                            base_year = 2018  # Reiwa era starts from 2019
                            year = str(base_year + int(year))
                        return f"{year}-{month:0>2s}-{day:0>2s}"
                    else:
                        return match.group(1).strip()
                        
        # No match found
        return ""
    
    def compute_metrics(self, results: Dict[str, List[Any]]) -> Dict[str, float]:
        """Compute VQA metrics.
        
        Args:
            results: Dictionary with predictions and ground truth
            
        Returns:
            Dictionary of metrics
        """
        # Initialize metrics
        metrics = {}
        
        # Calculate overall accuracy
        correct = 0
        total = len(results["ground_truth"])
        
        # Per question type metrics
        question_types = set(results["question_type"])
        type_correct = {qt: 0 for qt in question_types}
        type_total = {qt: 0 for qt in question_types}
        
        for gt, pred, q_type in zip(results["ground_truth"], results["prediction"], results["question_type"]):
            # Normalize for comparison
            gt_norm = gt.strip().lower()
            pred_norm = pred.strip().lower()
            
            # Update type counters
            type_total[q_type] += 1
            
            # Exact match
            if gt_norm == pred_norm:
                correct += 1
                type_correct[q_type] += 1
                
        # Overall accuracy
        metrics["accuracy"] = (correct / total) * 100 if total > 0 else 0
        
        # Per-type accuracy
        for q_type in question_types:
            accuracy = (type_correct[q_type] / type_total[q_type]) * 100 if type_total[q_type] > 0 else 0
            metrics[f"accuracy_{q_type}"] = accuracy
        
        # Calculate ROUGE scores for text comparison
        metrics.update(rouge_ja(refs=results["ground_truth"], preds=results["prediction"]))
        
        # Character-level accuracy
        char_accuracies = []
        for gt, pred in zip(results["ground_truth"], results["prediction"]):
            char_accuracies.append(calculate_character_accuracy(gt, pred))
        metrics["character_accuracy"] = sum(char_accuracies) / len(char_accuracies) if char_accuracies else 0.0
        
        return metrics
    
    def generate_report(self, eval_output: EvalOutput, output_path: str) -> str:
        """Generate evaluation report.
        
        Args:
            eval_output: Evaluation output
            output_path: Path to save report
            
        Returns:
            Report text
        """
        # Format metrics section
        report = "# Japanese Invoice Visual QA Evaluation\n\n"
        report += "## Metrics\n\n"
        report += "| Metric | Value |\n"
        report += "|--------|-------|\n"
        
        for metric, value in eval_output.metrics.items():
            report += f"| {metric} | {value:.2f} |\n"
        
        # Sample results
        report += "\n## Sample Results\n\n"
        
        # Show up to 10 samples
        sample_count = min(10, len(eval_output.results["image_path"]))
        
        for i in range(sample_count):
            report += f"### Sample {i+1}\n\n"
            report += f"**Image:** {eval_output.results['image_path'][i]}\n\n"
            report += f"**Question:** {eval_output.results['question'][i]}\n\n"
            report += f"**Question Type:** {eval_output.results['question_type'][i]}\n\n"
            report += f"**Ground Truth:** {eval_output.results['ground_truth'][i]}\n\n"
            report += f"**Prediction:** {eval_output.results['prediction'][i]}\n\n"
            
            # Add visual separator
            report += "---\n\n"
        
        # Write report to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
            
        return report
    
    def __call__(self, model) -> EvalOutput:
        """Run VQA evaluation on model.
        
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
            res = self._evaluate_vqa(model, example)
            results.append(res)
        
        # Flatten and compute metrics
        from .utils import flatten_list
        results = flatten_list(results)
        metrics = self.compute_metrics(results)
        
        # Create output
        output = EvalOutput(metrics=metrics, results=results)
        
        # Generate report
        report_path = os.path.join(self.output_dir, f"{self.name}_report.md")
        self.generate_report(output, report_path)
        logger.info(f"Report saved to {report_path}")
        
        return output