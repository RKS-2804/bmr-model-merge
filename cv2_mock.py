#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mock implementation of cv2 for environments without OpenGL libraries.
This provides basic functionality needed for the workflow to run.
Author: Ravikumar Shah
"""

import numpy as np
from PIL import Image

# Common constants
IMREAD_GRAYSCALE = 0
IMREAD_COLOR = 1
IMREAD_UNCHANGED = -1

# Basic image I/O functions
def imread(filename, flags=IMREAD_COLOR):
    """Read an image file using PIL instead of OpenCV."""
    try:
        img = Image.open(filename)
        if flags == IMREAD_GRAYSCALE:
            img = img.convert('L')
        elif flags == IMREAD_COLOR:
            img = img.convert('RGB')
        return np.array(img)
    except Exception as e:
        print(f"Error reading image {filename}: {e}")
        return None

def imwrite(filename, img):
    """Write an image file using PIL instead of OpenCV."""
    try:
        Image.fromarray(img).save(filename)
        return True
    except Exception as e:
        print(f"Error writing image {filename}: {e}")
        return False

# Minimal image processing functions
def resize(img, dsize, fx=None, fy=None, interpolation=None):
    """Resize an image using PIL."""
    try:
        pil_img = Image.fromarray(img)
        resized_img = pil_img.resize(dsize, Image.BILINEAR)
        return np.array(resized_img)
    except Exception as e:
        print(f"Error resizing image: {e}")
        return img

def cvtColor(img, code):
    """Convert image color space using PIL."""
    try:
        pil_img = Image.fromarray(img)
        if code == COLOR_BGR2GRAY:
            pil_img = pil_img.convert('L')
        elif code == COLOR_RGB2GRAY:
            pil_img = pil_img.convert('L')
        return np.array(pil_img)
    except Exception as e:
        print(f"Error converting color: {e}")
        return img

# Color conversion codes
COLOR_BGR2GRAY = 6
COLOR_RGB2GRAY = 7
COLOR_GRAY2BGR = 8
COLOR_GRAY2RGB = 8

# Other common constants and functions
FONT_HERSHEY_SIMPLEX = 0
LINE_AA = 16

def putText(*args, **kwargs):
    """Mock for putText - does nothing but avoids errors."""
    return

def rectangle(*args, **kwargs):
    """Mock for rectangle - does nothing but avoids errors."""
    return