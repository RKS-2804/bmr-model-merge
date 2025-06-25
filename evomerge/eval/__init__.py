"""
Evaluation modules for Japanese OCR and field extraction.

This package contains metrics and utilities for evaluating OCR models
and field extraction on Japanese documents.
"""

from evomerge.eval.metrics import (
    character_accuracy_metric,
    field_extraction_f1_metric,
    calculate_metrics,
    invoice_accuracy_metric,
    receipt_accuracy_metric
)

# Registry of available metrics
METRICS_REGISTRY = {
    "character_accuracy": character_accuracy_metric,
    "field_extraction_f1": field_extraction_f1_metric,
    "invoice_accuracy": invoice_accuracy_metric,
    "receipt_accuracy": receipt_accuracy_metric,
}

__all__ = [
    "character_accuracy_metric",
    "field_extraction_f1_metric",
    "calculate_metrics",
    "invoice_accuracy_metric", 
    "receipt_accuracy_metric",
    "METRICS_REGISTRY",
]