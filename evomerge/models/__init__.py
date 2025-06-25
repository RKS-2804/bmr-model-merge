"""
OCR model implementations and related utilities.
"""

from typing import Dict, Any
from evomerge.models.base_ocr import OCRModel
from evomerge.models.japanese_ocr import JapaneseOCRModel
from evomerge.models.field_extractor import InvoiceFieldExtractor

# Registry of available models
MODEL_REGISTRY = {
    "ocr": OCRModel,
    "japanese_ocr": JapaneseOCRModel,
}


def get_model(model_type: str, config: Dict[str, Any] = None) -> OCRModel:
    """
    Get a model instance by type.
    
    Args:
        model_type: Type of model to get
        config: Optional configuration dictionary
    
    Returns:
        Instantiated model
        
    Raises:
        ValueError: If model_type is not registered
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Model type '{model_type}' not found. Available types: {list(MODEL_REGISTRY.keys())}"
        )
    
    model_class = MODEL_REGISTRY[model_type]
    return model_class(config or {})


__all__ = [
    "OCRModel", 
    "JapaneseOCRModel", 
    "InvoiceFieldExtractor", 
    "get_model"
]