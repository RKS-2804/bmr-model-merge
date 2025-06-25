"""
Document processing utilities for Japanese OCR.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


class JapaneseDocumentProcessor:
    """Processor for Japanese document images and text."""
    
    def __init__(
        self,
        normalize_characters: bool = True,
        correct_orientation: bool = True,
        is_thermal_paper: bool = False,
        denoise_level: int = 1,
    ):
        """
        Initialize a Japanese document processor.
        
        Args:
            normalize_characters: Whether to normalize character width and variants
            correct_orientation: Whether to detect and correct orientation
            is_thermal_paper: Whether documents are thermal paper receipts
            denoise_level: Level of denoising to apply (0-3)
        """
        self.normalize_characters = normalize_characters
        self.correct_orientation = correct_orientation
        self.is_thermal_paper = is_thermal_paper
        self.denoise_level = denoise_level
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Process a document image for OCR.
        
        Args:
            image: Input image
            
        Returns:
            Processed image
        """
        # Copy the image to avoid modifying the original
        processed = image.copy()
        
        # Convert to grayscale if color
        if len(processed.shape) == 3:
            gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
        else:
            gray = processed.copy()
        
        # Check and correct orientation if enabled
        if self.correct_orientation:
            gray = self._correct_orientation(gray)
        
        # Apply preprocessing based on document type
        if self.is_thermal_paper:
            # Special processing for thermal paper receipts
            gray = self._denoise_thermal_paper(gray)
        else:
            # Standard document processing
            gray = self._enhance_document(gray)
        
        # Final touches
        enhanced = self._adaptive_threshold(gray)
        
        return enhanced
    
    def process_text(self, text: str) -> str:
        """
        Process OCR-extracted text.
        
        Args:
            text: Input text
            
        Returns:
            Processed text
        """
        if not text:
            return ""
        
        # Normalize characters if enabled
        if self.normalize_characters:
            processed_text = self._normalize_character_width(text)
            processed_text = self._normalize_character_variants(processed_text)
            processed_text = self._fix_common_ocr_errors(processed_text)
        else:
            processed_text = text
        
        return processed_text
    
    def _correct_orientation(self, image: np.ndarray) -> np.ndarray:
        """Detect and correct document orientation."""
        # In a real implementation, this would use a model to detect orientation
        # For this demo, we'll just assume the image is already correctly oriented
        
        # Placeholder for orientation detection
        # Would typically use a model or heuristic to detect orientation
        # and rotate if needed
        
        return image
    
    def _denoise_thermal_paper(self, image: np.ndarray) -> np.ndarray:
        """Apply specialized denoising for thermal paper receipts."""
        # Thermal receipts often have background noise and faded text
        denoised = cv2.fastNlMeansDenoising(
            image, None, h=10 * self.denoise_level, templateWindowSize=7, searchWindowSize=21
        )
        
        # Enhance contrast for faded text
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        return enhanced
    
    def _enhance_document(self, image: np.ndarray) -> np.ndarray:
        """Enhance general document image."""
        # Apply moderate denoising
        denoised = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        return enhanced
    
    def _adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding for better text extraction."""
        # Only apply thresholding for grayscale images
        if len(image.shape) == 2:
            block_size = 11
            constant = 2
            binary = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, block_size, constant
            )
            return binary
        else:
            return image
    
    def _normalize_character_width(self, text: str) -> str:
        """
        Convert between full-width and half-width forms consistently.
        
        This is particularly important for Japanese text where the same character
        may appear in different width forms.
        """
        # This would normally use unicodedata or other libraries to normalize
        # For this demo, we'll implement a simple version
        
        # Convert full-width digits to half-width
        for i in range(10):
            # Full-width digits are at Unicode code points U+FF10 through U+FF19
            text = text.replace(chr(0xFF10 + i), str(i))
        
        # Convert full-width alphabets to half-width
        for i in range(26):
            # Full-width uppercase A-Z: U+FF21 through U+FF3A
            text = text.replace(chr(0xFF21 + i), chr(ord('A') + i))
            # Full-width lowercase a-z: U+FF41 through U+FF5A
            text = text.replace(chr(0xFF41 + i), chr(ord('a') + i))
        
        return text
    
    def _normalize_character_variants(self, text: str) -> str:
        """Normalize different variants of the same character."""
        # In a real implementation, this would normalize things like:
        # - Traditional vs. simplified kanji
        # - Character variants (異体字/異字体)
        # - Old vs. new character forms
        
        # This is just a placeholder for demonstration
        replacements = {
            '−': '-',  # Replace minus sign with hyphen
            '~': '〜',  # Replace ASCII tilde with wave dash
            '･': '・',  # Normalize middle dots
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _fix_common_ocr_errors(self, text: str) -> str:
        """Fix common OCR errors in Japanese text."""
        # Common OCR confusion pairs
        replacements = {
            'て゛': 'で',
            'は゛': 'ば',
            'は゜': 'ぱ',
            'へ゛': 'べ',
            'へ゜': 'ぺ',
            # Add more common OCR errors here
        }
        
        for error, correction in replacements.items():
            text = text.replace(error, correction)
        
        return text