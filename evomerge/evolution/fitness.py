"""Fitness functions for OCR model evaluation in evolutionary optimization."""

import os
import json
import logging
import tempfile
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import yaml
from dataclasses import dataclass

import torch
import numpy as np
from PIL import Image

from ..eval.metrics import (
    rouge_ja, 
    calculate_character_accuracy, 
    calculate_field_extraction_accuracy,
    JapaneseMecabTokenizer,
    OCRLanguageDetector,
    normalize_japanese_text_for_ocr
)
from ..models.japanese_ocr import JapaneseOCRModel
from ..models.field_extractor import FieldExtractor
from .population import Individual

logger = logging.getLogger(__name__)


@dataclass
class OCRMetricsResult:
    """Results from OCR evaluation."""
    character_accuracy: float = 0.0
    field_extraction_accuracy: float = 0.0
    rouge1: float = 0.0
    rouge2: float = 0.0
    rougeL: float = 0.0
    japanese_confidence: float = 0.0
    inference_time: float = 0.0
    sample_count: int = 0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "character_accuracy": self.character_accuracy,
            "field_extraction_accuracy": self.field_extraction_accuracy,
            "rouge1": self.rouge1,
            "rouge2": self.rouge2,
            "rougeL": self.rougeL,
            "japanese_confidence": self.japanese_confidence,
            "inference_time": self.inference_time,
            "sample_count": self.sample_count
        }


class OCRFitnessEvaluator:
    """Evaluator for OCR model fitness in evolutionary optimization.
    
    This class handles the evaluation of OCR models on Japanese invoice/receipt datasets,
    computing various metrics and combining them into an overall fitness score.
    """
    
    def __init__(
        self,
        test_data_path: str,
        metrics_weights: Optional[Dict[str, float]] = None,
        batch_size: int = 16,
        device: Optional[str] = None,
        language_detector: Optional[OCRLanguageDetector] = None
    ):
        """
        Initialize OCR fitness evaluator.
        
        Args:
            test_data_path: Path to test dataset
            metrics_weights: Weights for different metrics in overall fitness
            batch_size: Batch size for evaluation
            device: Device for model inference
            language_detector: OCR language detector
        """
        self.test_data_path = test_data_path
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Default metric weights if none provided
        if metrics_weights is None:
            self.metrics_weights = {
                "character_accuracy": 0.4,
                "field_extraction_accuracy": 0.3,
                "rouge1": 0.1,
                "rouge2": 0.1, 
                "rougeL": 0.1,
                "japanese_confidence": 0.0,  # Not used by default in overall score
                "inference_time": 0.0  # Not used by default in overall score
            }
        else:
            self.metrics_weights = metrics_weights
        
        # Initialize language detector if needed
        self.language_detector = language_detector or OCRLanguageDetector()
        
        # Japanese tokenizer for text processing
        self.tokenizer = JapaneseMecabTokenizer()
        
        # Load test data
        self.test_data = self._load_test_data()
        logger.info(f"Loaded {len(self.test_data)} test samples for OCR evaluation")
    
    def _load_test_data(self) -> List[Dict[str, Any]]:
        """
        Load test data for OCR evaluation.
        
        Returns:
            List of test samples
        """
        # Check if path exists
        if not os.path.exists(self.test_data_path):
            logger.warning(f"Test data path not found: {self.test_data_path}")
            # Return empty list as fallback
            return []
        
        # Load based on file type
        if os.path.isdir(self.test_data_path):
            return self._load_test_data_from_dir()
        elif self.test_data_path.endswith('.json'):
            return self._load_test_data_from_json()
        elif self.test_data_path.endswith('.jsonl'):
            return self._load_test_data_from_jsonl()
        else:
            logger.warning(f"Unsupported test data format: {self.test_data_path}")
            return []
    
    def _load_test_data_from_dir(self) -> List[Dict[str, Any]]:
        """
        Load test data from a directory structure.
        
        Expects:
            - images/ (containing image files)
            - annotations.json (containing text annotations)
        
        Returns:
            List of test samples
        """
        samples = []
        images_dir = os.path.join(self.test_data_path, "images")
        annotations_path = os.path.join(self.test_data_path, "annotations.json")
        
        if not os.path.exists(images_dir) or not os.path.exists(annotations_path):
            logger.error(f"Invalid test data directory structure: {self.test_data_path}")
            return []
        
        # Load annotations
        with open(annotations_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        # Match images with annotations
        for img_name in os.listdir(images_dir):
            img_path = os.path.join(images_dir, img_name)
            if img_name in annotations:
                samples.append({
                    "image_path": img_path,
                    "ground_truth": annotations[img_name]
                })
        
        return samples
    
    def _load_test_data_from_json(self) -> List[Dict[str, Any]]:
        """
        Load test data from a JSON file.
        
        Returns:
            List of test samples
        """
        with open(self.test_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert data to list if it's a dictionary
        if isinstance(data, dict):
            return list(data.values())
        return data
    
    def _load_test_data_from_jsonl(self) -> List[Dict[str, Any]]:
        """
        Load test data from a JSONL file.
        
        Returns:
            List of test samples
        """
        samples = []
        with open(self.test_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples
    
    def evaluate_model(self, model_or_path: Union[str, JapaneseOCRModel]) -> OCRMetricsResult:
        """
        Evaluate an OCR model on the test dataset.
        
        Args:
            model_or_path: OCR model or path to model
            
        Returns:
            Evaluation metrics
        """
        # Load model if path is provided
        if isinstance(model_or_path, str):
            model = JapaneseOCRModel(model_name=model_or_path, device=self.device)
            model.load_model()
        else:
            model = model_or_path
        
        # If no test data available, return empty metrics
        if not self.test_data:
            logger.warning("No test data available for evaluation")
            return OCRMetricsResult()
        
        # Process all test samples
        all_refs = []
        all_preds = []
        all_char_accuracies = []
        all_field_accuracies = []
        all_jp_confidences = []
        inference_time_total = 0.0
        
        # Optional field extractor for structured information
        field_extractor = None
        if any("fields" in sample.get("ground_truth", {}) for sample in self.test_data):
            try:
                field_extractor = FieldExtractor()
            except:
                logger.warning("Could not initialize field extractor")
        
        # Process in batches
        for i in range(0, len(self.test_data), self.batch_size):
            batch = self.test_data[i:i+self.batch_size]
            batch_images = [sample["image_path"] for sample in batch]
            
            # Measure inference time
            import time
            start_time = time.time()
            
            # Get model predictions
            batch_results = model(batch_images)
            
            # Record inference time
            inference_time = time.time() - start_time
            inference_time_total += inference_time
            
            # Process each sample
            for j, sample in enumerate(batch):
                if j < len(batch_results):
                    result = batch_results[j]
                    pred_text = result.get("text", "")
                    ref_text = sample.get("ground_truth", {}).get("text", "")
                    
                    # Add to list for ROUGE calculation
                    if ref_text:
                        all_refs.append(ref_text)
                        all_preds.append(pred_text)
                    
                    # Calculate character accuracy
                    char_acc = calculate_character_accuracy(ref_text, pred_text)
                    all_char_accuracies.append(char_acc)
                    
                    # Calculate Japanese confidence
                    jp_confidence = self.language_detector.is_japanese(pred_text, threshold=0)
                    all_jp_confidences.append(jp_confidence)
                    
                    # Calculate field extraction accuracy if available
                    if field_extractor and "fields" in sample.get("ground_truth", {}):
                        ref_fields = sample["ground_truth"]["fields"]
                        try:
                            pred_fields = field_extractor.extract_fields(pred_text)
                            field_acc, _ = calculate_field_extraction_accuracy(ref_fields, pred_fields)
                            all_field_accuracies.append(field_acc)
                        except Exception as e:
                            logger.warning(f"Field extraction failed: {e}")
        
        # Compile results
        metrics = OCRMetricsResult()
        metrics.sample_count = len(self.test_data)
        
        # Character accuracy
        if all_char_accuracies:
            metrics.character_accuracy = sum(all_char_accuracies) / len(all_char_accuracies)
        
        # Field extraction accuracy
        if all_field_accuracies:
            metrics.field_extraction_accuracy = sum(all_field_accuracies) / len(all_field_accuracies)
        
        # ROUGE scores
        if all_refs and all_preds:
            rouge_scores = rouge_ja(all_refs, all_preds)
            metrics.rouge1 = rouge_scores.get("rouge1", 0.0)
            metrics.rouge2 = rouge_scores.get("rouge2", 0.0)
            metrics.rougeL = rouge_scores.get("rougeL", 0.0)
        
        # Japanese confidence
        if all_jp_confidences:
            metrics.japanese_confidence = sum(all_jp_confidences) / len(all_jp_confidences)
        
        # Inference time (average per sample)
        if metrics.sample_count > 0:
            metrics.inference_time = inference_time_total / metrics.sample_count
        
        return metrics
    
    def evaluate_individual(self, individual: Individual) -> Individual:
        """
        Evaluate an individual's fitness.
        
        Args:
            individual: Individual to evaluate
            
        Returns:
            The same individual with updated fitness
        """
        logger.info(f"Evaluating individual {individual.id} from generation {individual.generation}")
        
        # Get model path
        model_path = individual.model_path or individual.model_config.get("base_model")
        if not model_path:
            logger.error("No model path available for evaluation")
            return individual
        
        # Evaluate model
        metrics = self.evaluate_model(model_path)
        
        # Store fitness metrics
        individual.fitness = metrics.to_dict()
        
        # Calculate overall fitness
        overall_fitness = 0.0
        for metric_name, weight in self.metrics_weights.items():
            if metric_name in individual.fitness:
                # Special handling for inference time (lower is better)
                if metric_name == "inference_time" and individual.fitness[metric_name] > 0:
                    # Normalize and invert (so lower time = higher fitness)
                    normalized_time = 1.0 / (1.0 + individual.fitness[metric_name])
                    overall_fitness += weight * normalized_time
                else:
                    overall_fitness += weight * individual.fitness[metric_name]
        
        individual.overall_fitness = overall_fitness
        individual.evaluated = True
        
        logger.info(f"Individual {individual.id} fitness: {overall_fitness:.4f}")
        return individual
    
    def calculate_fitness_improvement(
        self, 
        parent1: Individual, 
        parent2: Individual, 
        child: Individual
    ) -> Tuple[float, float]:
        """
        Calculate fitness improvement of child over parents.
        
        Args:
            parent1: First parent individual
            parent2: Second parent individual
            child: Child individual
            
        Returns:
            Tuple of (absolute improvement, relative improvement)
        """
        if not all(ind.evaluated for ind in [parent1, parent2, child]):
            logger.warning("Cannot calculate improvement: some individuals not evaluated")
            return 0.0, 0.0
        
        # Average parent fitness
        parent_avg_fitness = (parent1.overall_fitness + parent2.overall_fitness) / 2
        
        # Calculate improvements
        abs_improvement = child.overall_fitness - parent_avg_fitness
        rel_improvement = (abs_improvement / parent_avg_fitness) if parent_avg_fitness > 0 else 0.0
        
        return abs_improvement, rel_improvement


def create_evaluator_from_config(config_path: str) -> OCRFitnessEvaluator:
    """
    Create an OCR fitness evaluator from a configuration file.
    
    Args:
        config_path: Path to evaluator configuration file
        
    Returns:
        Initialized OCRFitnessEvaluator
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return OCRFitnessEvaluator(
        test_data_path=config.get('test_data_path', ''),
        metrics_weights=config.get('metrics_weights', None),
        batch_size=config.get('batch_size', 16),
        device=config.get('device', None)
    )