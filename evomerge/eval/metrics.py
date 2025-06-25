"""
Evaluation metrics for Japanese OCR and field extraction.

This module provides metrics to evaluate the performance of OCR models
and field extraction on Japanese documents.
"""

import re
import string
import unicodedata
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np

def levenshtein_distance(s1, s2):
    """
    Calculate the Levenshtein distance between two strings.
    
    This is a simple implementation that doesn't require the Levenshtein package.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        The Levenshtein distance between s1 and s2
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one character longer
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def normalize_japanese_text(text: str) -> str:
    """
    Normalize Japanese text for fair comparison.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Convert to NFKC form (compatibility composition)
    text = unicodedata.normalize('NFKC', text)
    
    # Strip whitespace
    text = text.strip()
    
    # Remove duplicate whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Convert full-width to half-width for alphanumeric
    chars = []
    for c in text:
        if 0xFF10 <= ord(c) <= 0xFF19:  # Full-width digits to half-width
            chars.append(chr(ord(c) - 0xFF10 + 0x30))
        elif 0xFF21 <= ord(c) <= 0xFF3A:  # Full-width uppercase to half-width
            chars.append(chr(ord(c) - 0xFF21 + 0x41))
        elif 0xFF41 <= ord(c) <= 0xFF5A:  # Full-width lowercase to half-width
            chars.append(chr(ord(c) - 0xFF41 + 0x61))
        else:
            chars.append(c)
    
    return ''.join(chars)


def character_error_rate(pred_text: str, target_text: str) -> float:
    """
    Calculate Character Error Rate (CER).
    
    CER = (S + D + I) / N
    where:
    - S is the number of substitutions
    - D is the number of deletions
    - I is the number of insertions
    - N is the number of characters in the reference
    
    Args:
        pred_text: Predicted text
        target_text: Target (reference) text
        
    Returns:
        Character Error Rate
    """
    # Normalize texts
    pred = normalize_japanese_text(pred_text)
    target = normalize_japanese_text(target_text)
    
    if not target:
        return 1.0 if pred else 0.0
    
    # Calculate Levenshtein distance
    edit_distance = levenshtein_distance(pred, target)
    
    # Calculate CER
    cer = edit_distance / len(target)
    
    return cer


def character_accuracy(pred_text: str, target_text: str) -> float:
    """
    Calculate Character Accuracy (1 - CER).
    
    Args:
        pred_text: Predicted text
        target_text: Target (reference) text
        
    Returns:
        Character Accuracy
    """
    cer = character_error_rate(pred_text, target_text)
    return 1.0 - cer


def character_accuracy_metric(ocr_result: Dict[str, Any], reference: Dict[str, Any]) -> float:
    """
    Calculate character accuracy from OCR result and reference.
    
    Args:
        ocr_result: OCR result dictionary
        reference: Reference dictionary
        
    Returns:
        Character accuracy score
    """
    if "text" not in ocr_result or "text" not in reference:
        return 0.0
    
    return character_accuracy(ocr_result["text"], reference["text"])


def field_extraction_f1_metric(ocr_result: Dict[str, Any], reference: Dict[str, Any]) -> float:
    """
    Calculate F1 score for field extraction.
    
    Args:
        ocr_result: OCR result dictionary with extracted fields
        reference: Reference dictionary with ground truth fields
        
    Returns:
        F1 score for field extraction
    """
    if "fields" not in ocr_result or "fields" not in reference:
        return 0.0
    
    pred_fields = ocr_result["fields"]
    true_fields = reference["fields"]
    
    # Count true positives, false positives, and false negatives
    tp = 0
    fp = 0
    fn = 0
    
    # Check each field in predictions
    for field_name, pred_value in pred_fields.items():
        if field_name in true_fields:
            # Field exists in reference
            true_value = true_fields[field_name]
            
            # Extract values from dictionaries if needed
            if isinstance(pred_value, dict) and "value" in pred_value:
                pred_value = pred_value["value"]
            if isinstance(true_value, dict) and "value" in true_value:
                true_value = true_value["value"]
            
            # Convert to strings for comparison
            pred_str = str(pred_value)
            true_str = str(true_value)
            
            # Normalize for comparison
            pred_norm = normalize_japanese_text(pred_str)
            true_norm = normalize_japanese_text(true_str)
            
            # Check if values match
            if pred_norm == true_norm:
                tp += 1
            else:
                # Allow partial match (80% character overlap)
                char_accuracy = character_accuracy(pred_norm, true_norm)
                if char_accuracy >= 0.8:
                    tp += 0.5  # Partial credit
                    fn += 0.5  # Partial miss
                else:
                    fn += 1
        else:
            # Field doesn't exist in reference
            fp += 1
    
    # Check for fields in reference that are not in predictions
    for field_name in true_fields:
        if field_name not in pred_fields:
            fn += 1
    
    # Calculate precision, recall, and F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    return f1


def calculate_metrics(ocr_result: Dict[str, Any], reference: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate all metrics for an OCR result.
    
    Args:
        ocr_result: OCR result dictionary
        reference: Reference dictionary
        
    Returns:
        Dictionary with metric names and values
    """
    metrics = {
        "character_accuracy": character_accuracy_metric(ocr_result, reference),
        "field_extraction_f1": field_extraction_f1_metric(ocr_result, reference),
    }
    
    # Add document type specific metrics if available
    if "document_type" in ocr_result and ocr_result["document_type"] == reference.get("document_type"):
        if ocr_result["document_type"] == "invoice":
            # Add invoice-specific metrics
            metrics["invoice_accuracy"] = invoice_accuracy_metric(ocr_result, reference)
        elif ocr_result["document_type"] == "receipt":
            # Add receipt-specific metrics
            metrics["receipt_accuracy"] = receipt_accuracy_metric(ocr_result, reference)
    
    # Calculate overall score as weighted average
    metrics["overall_score"] = (
        0.4 * metrics["character_accuracy"] +
        0.6 * metrics["field_extraction_f1"]
    )
    
    return metrics


def invoice_accuracy_metric(ocr_result: Dict[str, Any], reference: Dict[str, Any]) -> float:
    """
    Calculate accuracy specifically for invoice fields.
    
    Args:
        ocr_result: OCR result dictionary
        reference: Reference dictionary
        
    Returns:
        Invoice accuracy score
    """
    if "fields" not in ocr_result or "fields" not in reference:
        return 0.0
    
    # Important fields for invoices with weights
    important_fields = {
        "invoice_number": 0.25,
        "date": 0.15,
        "due_date": 0.15,
        "total_amount": 0.3,
        "company_name": 0.15,
    }
    
    total_weight = sum(important_fields.values())
    score = 0.0
    
    for field, weight in important_fields.items():
        if field in ocr_result.get("fields", {}) and field in reference.get("fields", {}):
            pred_value = ocr_result["fields"][field]
            true_value = reference["fields"][field]
            
            # Extract values from dictionaries if needed
            if isinstance(pred_value, dict) and "value" in pred_value:
                pred_value = pred_value["value"]
            if isinstance(true_value, dict) and "value" in true_value:
                true_value = true_value["value"]
            
            # Calculate character accuracy for the field
            field_accuracy = character_accuracy(str(pred_value), str(true_value))
            score += weight * field_accuracy
    
    return score / total_weight if total_weight > 0 else 0.0


def receipt_accuracy_metric(ocr_result: Dict[str, Any], reference: Dict[str, Any]) -> float:
    """
    Calculate accuracy specifically for receipt fields.
    
    Args:
        ocr_result: OCR result dictionary
        reference: Reference dictionary
        
    Returns:
        Receipt accuracy score
    """
    if "fields" not in ocr_result or "fields" not in reference:
        return 0.0
    
    # Important fields for receipts with weights
    important_fields = {
        "receipt_number": 0.15,
        "date": 0.15,
        "total_amount": 0.4,
        "items": 0.3,  # Special handling for items
    }
    
    total_weight = sum(important_fields.values())
    score = 0.0
    
    for field, weight in important_fields.items():
        if field == "items":
            # Special handling for item lists
            if "items" in ocr_result.get("fields", {}) and "items" in reference.get("fields", {}):
                items_score = _calculate_items_accuracy(
                    ocr_result["fields"]["items"],
                    reference["fields"]["items"]
                )
                score += weight * items_score
        elif field in ocr_result.get("fields", {}) and field in reference.get("fields", {}):
            pred_value = ocr_result["fields"][field]
            true_value = reference["fields"][field]
            
            # Extract values from dictionaries if needed
            if isinstance(pred_value, dict) and "value" in pred_value:
                pred_value = pred_value["value"]
            if isinstance(true_value, dict) and "value" in true_value:
                true_value = true_value["value"]
            
            # Calculate character accuracy for the field
            field_accuracy = character_accuracy(str(pred_value), str(true_value))
            score += weight * field_accuracy
    
    return score / total_weight if total_weight > 0 else 0.0


def _calculate_items_accuracy(pred_items, true_items) -> float:
    """
    Calculate accuracy for item lists in receipts.
    
    Args:
        pred_items: Predicted items list
        true_items: True items list
        
    Returns:
        Items accuracy score
    """
    if not isinstance(pred_items, list) or not isinstance(true_items, list):
        return 0.0
    
    if not true_items:
        return 0.0 if pred_items else 1.0
    
    # Best matching for each true item
    item_scores = []
    
    for true_item in true_items:
        best_match_score = 0.0
        
        # Extract true item values
        true_name = true_item.get("name", "")
        true_price = true_item.get("price", "")
        
        for pred_item in pred_items:
            # Extract predicted item values
            pred_name = pred_item.get("name", "")
            pred_price = pred_item.get("price", "")
            
            # Calculate name and price accuracy
            name_accuracy = character_accuracy(pred_name, true_name)
            price_accuracy = character_accuracy(pred_price, true_price)
            
            # Combined score with higher weight on price
            item_score = 0.6 * name_accuracy + 0.4 * price_accuracy
            
            # Keep the best match
            best_match_score = max(best_match_score, item_score)
        
        item_scores.append(best_match_score)
    
    # Calculate overall items accuracy
    if not item_scores:
        return 0.0
    
    # Penalize for count mismatch
    count_penalty = min(1.0, len(true_items) / max(1, len(pred_items)))
    
    # Final score
    return sum(item_scores) / len(item_scores) * count_penalty