"""
Dataset classes for Japanese OCR document datasets.
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from evomerge.data.processor import JapaneseDocumentProcessor


class JapaneseDocumentDataset(Dataset):
    """Base class for Japanese document datasets."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        transform=None,
        processor_config: Optional[Dict] = None,
    ):
        """
        Initialize a Japanese document dataset.
        
        Args:
            data_dir: Directory containing the dataset
            split: Data split to use ('train', 'val', 'test')
            transform: Optional transform to apply to images
            processor_config: Configuration for the document processor
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.processor = JapaneseDocumentProcessor(**(processor_config or {}))
        
        # Ensure the data directory exists
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            print(f"Created dataset directory: {self.data_dir}")
        
        # Initialize with empty list, subclasses will load data
        self.samples = []
        self.load_dataset()
    
    def load_dataset(self):
        """Load dataset samples. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement load_dataset method")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a dataset item by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing image and annotations
        """
        sample = self.samples[idx]
        
        # Load image
        image_path = sample["image_path"]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply processor
        processed_image = self.processor.process_image(image)
        
        # Apply transform if provided
        if self.transform:
            processed_image = self.transform(processed_image)
        
        # Convert to torch tensor if not already
        if not isinstance(processed_image, torch.Tensor):
            processed_image = torch.from_numpy(processed_image).float()
        
        # Return data
        return {
            "image": processed_image,
            "original_image": image,
            "annotations": sample["annotations"],
            "metadata": {
                "image_path": str(image_path),
                "document_id": sample.get("document_id", ""),
                "document_type": self.get_document_type(),
            }
        }
    
    def get_document_type(self):
        """Return the document type. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement get_document_type method")


class JapaneseInvoiceDataset(JapaneseDocumentDataset):
    """Dataset for Japanese invoice images."""
    
    def load_dataset(self):
        """Load invoice dataset samples."""
        # Define split directories
        split_dir = self.data_dir / self.split
        
        # Create placeholder data if directory doesn't exist or is empty
        if not split_dir.exists() or not any(split_dir.iterdir()):
            print(f"Warning: Creating placeholder data for {self.split} split in {split_dir}")
            split_dir.mkdir(exist_ok=True)
            self._create_placeholder_data(split_dir, num_samples=5)
        
        # Load annotations
        annotations_file = split_dir / "annotations.json"
        if annotations_file.exists():
            with open(annotations_file, "r", encoding="utf-8") as f:
                self.annotations = json.load(f)
        else:
            self.annotations = {}
        
        # Load samples
        self.samples = []
        for image_path in split_dir.glob("*.jpg"):
            doc_id = image_path.stem
            
            self.samples.append({
                "image_path": image_path,
                "document_id": doc_id,
                "annotations": self.annotations.get(doc_id, {})
            })
        
        print(f"Loaded {len(self.samples)} invoice samples from {split_dir}")
    
    def _create_placeholder_data(self, output_dir: Path, num_samples: int = 5):
        """Create placeholder data for testing."""
        annotations = {}
        
        for i in range(num_samples):
            # Create a blank image
            img = np.ones((800, 600, 3), dtype=np.uint8) * 255
            
            # Add some text to make it look like an invoice
            cv2.putText(img, f"Invoice #{i+1}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(img, "Date: 2025-06-25", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(img, "Amount: ¥10,000", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Save the image
            img_path = output_dir / f"invoice_{i+1}.jpg"
            cv2.imwrite(str(img_path), img)
            
            # Create annotations
            doc_id = f"invoice_{i+1}"
            annotations[doc_id] = {
                "invoice_number": f"INV-2025-{i+1:04d}",
                "date": "2025-06-25",
                "amount": "¥10,000",
                "fields": {
                    "invoice_number": {"value": f"INV-2025-{i+1:04d}", "confidence": 0.95},
                    "date": {"value": "2025-06-25", "confidence": 0.92},
                    "amount": {"value": "¥10,000", "confidence": 0.98},
                }
            }
        
        # Save annotations
        with open(output_dir / "annotations.json", "w", encoding="utf-8") as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)
    
    def get_document_type(self):
        return "invoice"


class JapaneseReceiptDataset(JapaneseDocumentDataset):
    """Dataset for Japanese receipt images."""
    
    def load_dataset(self):
        """Load receipt dataset samples."""
        # Define split directories
        split_dir = self.data_dir / self.split
        
        # Create placeholder data if directory doesn't exist or is empty
        if not split_dir.exists() or not any(split_dir.iterdir()):
            print(f"Warning: Creating placeholder data for {self.split} split in {split_dir}")
            split_dir.mkdir(exist_ok=True)
            self._create_placeholder_data(split_dir, num_samples=5)
        
        # Load annotations
        annotations_file = split_dir / "annotations.json"
        if annotations_file.exists():
            with open(annotations_file, "r", encoding="utf-8") as f:
                self.annotations = json.load(f)
        else:
            self.annotations = {}
        
        # Load samples
        self.samples = []
        for image_path in split_dir.glob("*.jpg"):
            doc_id = image_path.stem
            
            self.samples.append({
                "image_path": image_path,
                "document_id": doc_id,
                "annotations": self.annotations.get(doc_id, {})
            })
        
        print(f"Loaded {len(self.samples)} receipt samples from {split_dir}")
    
    def _create_placeholder_data(self, output_dir: Path, num_samples: int = 5):
        """Create placeholder data for testing."""
        annotations = {}
        
        for i in range(num_samples):
            # Create a blank image (narrower for receipts)
            img = np.ones((800, 400, 3), dtype=np.uint8) * 255
            
            # Add some text to make it look like a receipt
            cv2.putText(img, "コンビニ レシート", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(img, f"No. {i+1:04d}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(img, "Date: 2025-06-25", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            cv2.putText(img, "商品 A: ¥500", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            cv2.putText(img, "商品 B: ¥750", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            cv2.putText(img, "合計: ¥1,250", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Save the image
            img_path = output_dir / f"receipt_{i+1}.jpg"
            cv2.imwrite(str(img_path), img)
            
            # Create annotations
            doc_id = f"receipt_{i+1}"
            annotations[doc_id] = {
                "receipt_number": f"{i+1:04d}",
                "date": "2025-06-25",
                "amount": "¥1,250",
                "items": [
                    {"name": "商品 A", "price": "¥500"},
                    {"name": "商品 B", "price": "¥750"}
                ],
                "fields": {
                    "receipt_number": {"value": f"{i+1:04d}", "confidence": 0.93},
                    "date": {"value": "2025-06-25", "confidence": 0.91},
                    "total_amount": {"value": "¥1,250", "confidence": 0.97},
                }
            }
        
        # Save annotations
        with open(output_dir / "annotations.json", "w", encoding="utf-8") as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)
    
    def get_document_type(self):
        return "receipt"


class AugmentedOCRDataset(Dataset):
    """Dataset with augmentation for OCR training."""
    
    def __init__(
        self,
        base_dataset: JapaneseDocumentDataset,
        augmentations_config: Optional[Dict] = None
    ):
        """
        Initialize an augmented dataset for OCR training.
        
        Args:
            base_dataset: Base Japanese document dataset
            augmentations_config: Configuration for augmentations
        """
        self.base_dataset = base_dataset
        self.augmentations_config = augmentations_config or {}
        
        # Default augmentation settings
        self.augmentations = {
            "rotation": self.augmentations_config.get("rotation", True),
            "noise": self.augmentations_config.get("noise", True),
            "blur": self.augmentations_config.get("blur", False),
            "contrast": self.augmentations_config.get("contrast", True),
            "probability": self.augmentations_config.get("probability", 0.5),
        }
        
        print(f"Initialized augmented dataset with config: {self.augmentations}")
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        """Get a dataset item with augmentation applied."""
        # Get base item
        item = self.base_dataset[idx]
        
        # Apply augmentations with probability
        if random.random() < self.augmentations["probability"]:
            item["image"] = self._apply_augmentations(item["image"])
        
        return item
    
    def _apply_augmentations(self, image):
        """Apply augmentations to the image."""
        # Convert to numpy if tensor
        is_tensor = isinstance(image, torch.Tensor)
        if is_tensor:
            image = image.numpy()
        
        # Apply configured augmentations
        if self.augmentations["rotation"] and random.random() < 0.5:
            angle = random.uniform(-5, 5)  # Small rotation only
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        if self.augmentations["noise"] and random.random() < 0.5:
            noise = np.random.normal(0, 0.03, image.shape).astype(np.float32)
            image = np.clip(image + noise, 0, 1) if image.max() <= 1 else np.clip(image + noise * 255, 0, 255)
        
        if self.augmentations["blur"] and random.random() < 0.5:
            kernel_size = random.choice([3, 5])
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        if self.augmentations["contrast"] and random.random() < 0.5:
            alpha = random.uniform(0.8, 1.2)
            image = np.clip(image * alpha, 0, 1) if image.max() <= 1 else np.clip(image * alpha, 0, 255)
        
        # Convert back to tensor if it was a tensor
        if is_tensor:
            image = torch.from_numpy(image)
        
        return image