"""Image preprocessing for Japanese OCR tasks."""

import cv2
import numpy as np
from typing import Optional, Tuple, Union, List
from PIL import Image
import logging
import os


logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Image preprocessor specialized for Japanese invoices and receipts.
    
    This class provides preprocessing functions to improve OCR accuracy
    on Japanese documents by applying various image enhancement techniques.
    """
    
    def __init__(
        self,
        resize_width: Optional[int] = 1280,
        binarization: bool = True,
        denoise: bool = True,
        contrast_enhancement: bool = True,
        deskew: bool = True,
    ):
        """Initialize the image preprocessor.
        
        Args:
            resize_width: Width to resize images to. None for no resizing.
            binarization: Whether to apply binarization
            denoise: Whether to apply denoising
            contrast_enhancement: Whether to apply contrast enhancement
            deskew: Whether to apply deskewing
        """
        self.resize_width = resize_width
        self.binarization = binarization
        self.denoise = denoise
        self.contrast_enhancement = contrast_enhancement
        self.deskew = deskew
    
    def process(
        self, 
        image: Union[str, np.ndarray, Image.Image]
    ) -> Union[np.ndarray, Image.Image]:
        """Process an image for OCR.
        
        Args:
            image: Input image as file path, numpy array, or PIL Image
            
        Returns:
            Processed image
        """
        # Convert to numpy array if needed
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not read image: {image}")
        elif isinstance(image, Image.Image):
            img = np.array(image)
            # Convert RGB to BGR for OpenCV
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img = image.copy()
        
        # Process the image
        img = self._preprocess(img)
        
        # Convert back to PIL Image if input was PIL
        if isinstance(image, Image.Image):
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            
        return img
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing steps to image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Processed image
        """
        # Make a copy to avoid modifying the original
        img = image.copy()
        
        # Resize if requested
        if self.resize_width is not None:
            h, w = img.shape[:2]
            if w > self.resize_width:
                new_height = int(h * (self.resize_width / w))
                img = cv2.resize(img, (self.resize_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale if not already
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # Deskew (correct rotation)
        if self.deskew:
            gray = self._deskew(gray)
        
        # Denoise
        if self.denoise:
            gray = self._denoise(gray)
        
        # Enhance contrast
        if self.contrast_enhancement:
            gray = self._enhance_contrast(gray)
        
        # Binarize
        if self.binarization:
            gray = self._binarize(gray)
        
        return gray
    
    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """Correct skew in the image.
        
        Args:
            image: Grayscale image
            
        Returns:
            Deskewed image
        """
        try:
            # Find all non-zero points
            coords = np.column_stack(np.where(image > 0))
            
            # Get rotated rectangle
            angle = cv2.minAreaRect(coords)[-1]
            
            # Adjust angle reference
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
                
            # Skip if angle is small
            if abs(angle) < 0.5:
                return image
                
            # Rotate the image
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                image, M, (w, h), 
                flags=cv2.INTER_CUBIC, 
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255
            )
            
            return rotated
        except Exception as e:
            logger.warning(f"Deskew failed: {e}")
            return image
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Remove noise from image.
        
        Args:
            image: Grayscale image
            
        Returns:
