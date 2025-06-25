"""
Dataset class for Vista OCR document datasets.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from evomerge.data.datasets import JapaneseDocumentDataset


class VistaDataset(JapaneseDocumentDataset):
    """Dataset for Vista document images."""
    
    def load_dataset(self):
        """Load Vista dataset samples."""
        # Define split directories
        split_dir = self.data_dir
        
        # Load samples
        self.samples = []
        
        # Check for image files with common extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(split_dir.glob(f"*{ext}")))
        
        # Create empty annotations if none exist
        self.annotations = {}
        
        # Log the number of images found
        import logging
        logger = logging.getLogger("bmr-workflow")
        logger.info(f"Found {len(image_files)} images in {split_dir}")
        
        # Process each image file
        for image_path in image_files:
            doc_id = image_path.stem
            
            self.samples.append({
                "image_path": image_path,
                "document_id": doc_id,
                "annotations": self.annotations.get(doc_id, {})
            })
        
        print(f"Loaded {len(self.samples)} vista samples from {split_dir}")
    
    def get_document_type(self):
        """Return the document type."""
        # Determine if this is likely a receipt or invoice based on directory name
        dir_name = self.data_dir.name.lower()
        if "receipt" in dir_name:
            return "receipt"
        elif "invoice" in dir_name:
            return "invoice"
        else:
            # Default to generic document type
            return "document"