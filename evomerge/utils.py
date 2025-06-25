import importlib
import json
import os
import yaml
import random
import numpy as np
import torch


def default(val, d):
    """Return val if it is not None, otherwise return d."""
    if val is not None:
        return val
    return d


def load_config(config_path: str) -> dict:
    """Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        The loaded configuration as a dictionary
    """
    ext = os.path.splitext(config_path)[-1]
    if ext in [".yaml", ".yml"]:
        with open(config_path, "r", encoding="utf-8") as fp:
            config = yaml.safe_load(fp)
    elif ext == ".json":
        with open(config_path, "r", encoding="utf-8") as fp:
            config = json.load(fp)
    else:
        raise RuntimeError(f"Unsupported configuration file extension: {ext}")
    return config


def get_obj_from_str(string, reload=False, invalidate_cache=True):
    """Get a Python object from its string representation.
    
    Args:
        string: String representation of the object
        reload: Whether to reload the module
        invalidate_cache: Whether to invalidate import caches
        
    Returns:
        The Python object
    """
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    """Instantiate an object from a configuration dictionary.
    
    Args:
        config: Configuration dictionary with 'target' and optionally 'params'
        
    Returns:
        The instantiated object
    """
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def set_seed(seed: int):
    """Set random seeds for reproducibility.
    
    Args:
        seed: The seed to set for random, numpy, and torch
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_environment():
    """Setup environment variables and configurations for the project.
    
    This function ensures that all necessary environment variables are set
    and resources are available for the project to run properly.
    """
    # Set up environment variables for better Japanese text handling
    os.environ["PYTHONIOENCODING"] = "utf-8"
    
    # Check CUDA availability and set environment variables appropriately
    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("CUDA not available, using CPU only")
    
    # Create necessary directories if they don't exist
    for dir_path in ["models", "data", "results", "checkpoints"]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Return environment configuration
    return {
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "project_root": os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    }


def ensure_font():
    """Ensure that a Japanese-compatible font is available for visualization.
    
    Returns:
        str: Path to the best available font
    """
    # List of potential font paths to check
    potential_fonts = []
    
    # Windows fonts
    if os.name == 'nt':
        potential_fonts.extend([
            "C:/Windows/Fonts/msgothic.ttc",
            "C:/Windows/Fonts/meiryo.ttc",
            "C:/Windows/Fonts/yugothm.ttc"
        ])
    
    # macOS fonts
    elif os.name == 'posix' and os.uname().sysname == 'Darwin':
        potential_fonts.extend([
            "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",
            "/Library/Fonts/Osaka.ttf",
            "/System/Library/Fonts/AppleGothic.ttf"
        ])
    
    # Linux fonts
    else:
        potential_fonts.extend([
            "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansJP-Regular.otf"
        ])
    
    # Check each potential font
    for font_path in potential_fonts:
        if os.path.exists(font_path):
            print(f"Found Japanese font: {font_path}")
            return font_path
    
    # If no font found, print warning and return None
    print("Warning: Could not find a Japanese font. Text rendering may be incorrect.")
    return None