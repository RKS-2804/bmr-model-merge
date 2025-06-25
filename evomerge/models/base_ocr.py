"""
Base OCR model implementation.
"""

import os
import numpy as np
from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import torch


class OCRModel(ABC):
    """
    Base class for OCR models.
    
    This abstract class defines the interface for all OCR models in the system.
    Subclasses must implement the load_model, preprocess, postprocess, and __call__ methods.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the OCR model.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.model = None
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = self.config.get("confidence_threshold", 0.5)
        self.model_path = self.config.get("model_path", None)
        
        # Load model if path is provided
        if self.model_path and os.path.exists(self.model_path):
            self.load_model(self.model_path)
    
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """
        Load the model from a file.
        
        Args:
            model_path: Path to the model file
        """
        pass
    
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> Any:
        """
        Preprocess an image for OCR.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image or model input
        """
        pass
    
    @abstractmethod
    def postprocess(self, model_output: Any) -> Dict[str, Any]:
        """
        Postprocess model output to extract text and other information.
        
        Args:
            model_output: Raw output from the model
            
        Returns:
            Dictionary with extracted information
        """
        pass
    
    @abstractmethod
    def __call__(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run OCR on an image.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with OCR results
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "type": self.__class__.__name__,
            "config": self.config,
            "device": self.device,
            "model_path": self.model_path,
        }
    
    def _prepare_tensor(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert image to torch tensor and move to appropriate device.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tensor on the appropriate device
        """
        if not isinstance(image, torch.Tensor):
            # Convert to float and normalize if needed
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            
            # Convert to tensor
            tensor = torch.from_numpy(image).float()
            
            # Add batch dimension if needed
            if tensor.dim() == 3:  # HxWxC
                tensor = tensor.permute(2, 0, 1)  # CxHxW
            if tensor.dim() == 3:  # CxHxW
                tensor = tensor.unsqueeze(0)  # 1xCxHxW
        else:
            tensor = image
        
        # Move to device
        return tensor.to(self.device)
    
    def merge_with(self, other_model: 'OCRModel', weights: np.ndarray) -> 'OCRModel':
        """
        Merge this model with another model using weights.
        
        Args:
            other_model: Another OCR model to merge with
            weights: Weights for merging (typically between 0 and 1)
            
        Returns:
            A new merged model
            
        Raises:
            NotImplementedError: If the model does not support merging
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement model merging."
        )