#!/usr/bin/env python
"""
Setup and resource download script for BMR Model Merge project.

This script handles the downloading of necessary models, fonts, 
and datasets required to run the evolutionary model merging process.
"""

import os
import sys
import argparse
import logging
import json
import yaml
import shutil
from pathlib import Path
import zipfile
import tarfile
import gzip
from typing import Dict, List, Optional, Union, Any
import urllib.request
from tqdm import tqdm
import torch
from huggingface_hub import hf_hub_download, snapshot_download
import requests

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        """Update progress bar.
        
        Args:
            b: Number of blocks transferred
            bsize: Size of each block
            tsize: Total size
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: str):
    """Download a file from a URL with progress bar.
    
    Args:
        url: URL to download from
        output_path: Path to save the file to
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    logger.info(f"Downloading {url} to {output_path}")
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def extract_archive(archive_path: str, extract_path: str):
    """Extract an archive file.
    
    Args:
        archive_path: Path to the archive file
        extract_path: Path to extract to
    """
    os.makedirs(extract_path, exist_ok=True)
    
    logger.info(f"Extracting {archive_path} to {extract_path}")
    
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
    elif archive_path.endswith(".tar.gz") or archive_path.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tar_ref:
            tar_ref.extractall(extract_path)
    elif archive_path.endswith(".tar"):
        with tarfile.open(archive_path, "r") as tar_ref:
            tar_ref.extractall(extract_path)
    elif archive_path.endswith(".gz") and not archive_path.endswith(".tar.gz"):
        with gzip.open(archive_path, "rb") as f_in:
            with open(archive_path[:-3], "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        logger.warning(f"Unsupported archive format: {archive_path}")


def download_huggingface_model(repo_id: str, output_dir: str, model_name: str, use_auth_token: Optional[str] = None):
    """Download a model from Hugging Face Hub.
    
    Args:
        repo_id: Hugging Face model repository ID
        output_dir: Directory to save the model to
        model_name: Name of the model
        use_auth_token: Optional authentication token for private repositories
    """
    try:
        logger.info(f"Downloading {model_name} from Hugging Face Hub: {repo_id}")
        
        # Create model directory
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Download the model
        snapshot_download(
            repo_id=repo_id,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            token=use_auth_token
        )
        
        logger.info(f"Successfully downloaded {model_name} to {model_dir}")
        return True
    except Exception as e:
        logger.error(f"Error downloading {model_name} from {repo_id}: {e}")
        return False


def download_fasttext_model(output_dir: str):
    """Download FastText language identification model.
    
    Args:
        output_dir: Directory to save the model to
    """
    fasttext_url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
    output_path = os.path.join(output_dir, "lid.176.ftz")
    
    if os.path.exists(output_path):
        logger.info(f"FastText model already exists at {output_path}")
        return True
    
    try:
        download_url(fasttext_url, output_path)
        return True
    except Exception as e:
        logger.error(f"Error downloading FastText model: {e}")
        return False


def download_fonts(output_dir: str):
    """Download Japanese fonts required for visualization.
    
    Args:
        output_dir: Directory to save the fonts to
    """
    fonts_dir = os.path.join(output_dir, "fonts")
    os.makedirs(fonts_dir, exist_ok=True)
    
    # Noto Sans Japanese
    noto_sans_jp_url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/Japanese/NotoSansCJKjp-Regular.otf"
    noto_sans_jp_path = os.path.join(fonts_dir, "NotoSansCJKjp-Regular.otf")
    
    if not os.path.exists(noto_sans_jp_path):
        try:
            download_url(noto_sans_jp_url, noto_sans_jp_path)
        except Exception as e:
            logger.error(f"Error downloading Noto Sans Japanese font: {e}")
    else:
        logger.info(f"Noto Sans Japanese font already exists at {noto_sans_jp_path}")
    
    # Return the path to the font that can be used for visualization
    if os.path.exists(noto_sans_jp_path):
        return noto_sans_jp_path
    else:
        return None


def download_sample_datasets(output_dir: str):
    """Download sample Japanese invoice/receipt datasets.
    
    Args:
        output_dir: Directory to save the datasets to
    """
    datasets_dir = os.path.join(output_dir, "data")
    os.makedirs(datasets_dir, exist_ok=True)
    
    # Create sample directories
    samples_dir = os.path.join(datasets_dir, "samples")
    receipts_dir = os.path.join(samples_dir, "receipts")
    invoices_dir = os.path.join(samples_dir, "invoices")
    demo_dir = os.path.join(samples_dir, "demo")
    
    for d in [samples_dir, receipts_dir, invoices_dir, demo_dir]:
        os.makedirs(d, exist_ok=True)
        os.makedirs(os.path.join(d, "images"), exist_ok=True)
        os.makedirs(os.path.join(d, "annotations"), exist_ok=True)
    
    # URLs for sample images
    # Note: In a real implementation, you would point to actual datasets
    # For this example, we're using placeholder URLs
    sample_urls = [
        "https://github.com/NVIDIA/DIGITS/raw/master/examples/semantic-segmentation/example_image.png",
        "https://github.com/matterport/Mask_RCNN/raw/master/images/43900001_1.jpg",
    ]
    
    # Download sample images (in a real implementation, these would be actual invoices/receipts)
    for i, url in enumerate(sample_urls):
        try:
            receipt_path = os.path.join(receipts_dir, "images", f"sample_receipt_{i+1}.png")
            invoice_path = os.path.join(invoices_dir, "images", f"sample_invoice_{i+1}.png")
            demo_path = os.path.join(demo_dir, "images", f"sample_demo_{i+1}.png")
            
            if not os.path.exists(receipt_path):
                download_url(url, receipt_path)
            
            if not os.path.exists(invoice_path):
                download_url(url, invoice_path)
                
            if not os.path.exists(demo_path):
                download_url(url, demo_path)
                
            # Create sample annotation files
            for name, path in [
                ("receipt", receipt_path), 
                ("invoice", invoice_path), 
                ("demo", demo_path)
            ]:
                create_sample_annotation(path, name, i+1)
                
        except Exception as e:
            logger.error(f"Error downloading sample image {i+1}: {e}")
    
    logger.info(f"Sample datasets created at {samples_dir}")
    return samples_dir


def create_sample_annotation(image_path: str, doc_type: str, index: int):
    """Create a sample annotation file for an image.
    
    Args:
        image_path: Path to the image
        doc_type: Type of document ('receipt' or 'invoice')
        index: Index for naming
    """
    # Extract base image name
    image_name = os.path.basename(image_path)
    base_name = os.path.splitext(image_name)[0]
    
    # Determine annotation path
    annotation_dir = os.path.join(os.path.dirname(os.path.dirname(image_path)), "annotations")
    annotation_path = os.path.join(annotation_dir, f"{base_name}.json")
    
    # Skip if annotation already exists
    if os.path.exists(annotation_path):
        return
    
    # Create sample annotations based on document type
    if doc_type == "receipt":
        annotation = {
            "text": "サンプルレシート\n株式会社サンプル\n2023年5月1日\n商品A 1点 1,000円\n商品B 2点 2,400円\n小計: 3,400円\n消費税: 340円\n合計: 3,740円",
            "fields": {
                "vendor_name": "株式会社サンプル",
                "issue_date": "2023-05-01",
                "subtotal": "3400",
                "tax_amount": "340",
                "total_amount": "3740"
            },
            "metadata": {
                "source": "sample_dataset",
                "id": f"receipt_{index}"
            }
        }
    elif doc_type == "invoice":
        annotation = {
            "text": "請求書\n請求書番号: INV-2023-{:04d}\n発行日: 2023年5月1日\n支払期限: 2023年5月31日\n\n株式会社サンプル\n〒123-4567 東京都サンプル区サンプル町1-2-3\n\n商品の詳細:\n商品A 1点 10,000円\n商品B 2点 24,000円\n\n小計: 34,000円\n消費税(10%): 3,400円\n合計: 37,400円".format(index),
            "fields": {
                "invoice_number": f"INV-2023-{index:04d}",
                "issue_date": "2023-05-01",
                "due_date": "2023-05-31",
                "vendor_name": "株式会社サンプル",
                "vendor_address": "〒123-4567 東京都サンプル区サンプル町1-2-3",
                "subtotal": "34000",
                "tax_amount": "3400",
                "total_amount": "37400"
            },
            "line_items": [
                {
                    "item_name": "商品A",
                    "item_quantity": "1",
                    "item_price": "10000",
                    "item_amount": "10000"
                },
                {
                    "item_name": "商品B",
                    "item_quantity": "2",
                    "item_price": "12000",
                    "item_amount": "24000"
                }
            ],
            "metadata": {
                "source": "sample_dataset",
                "id": f"invoice_{index}"
            }
        }
    else:  # demo
        annotation = {
            "text": "サンプルドキュメント\n株式会社サンプル\n2023年5月1日\n\nこれはOCRテスト用のサンプルドキュメントです。\n日本語テキストの認識精度を確認します。",
            "fields": {
                "vendor_name": "株式会社サンプル",
                "issue_date": "2023-05-01"
            },
            "metadata": {
                "source": "sample_dataset",
                "id": f"demo_{index}"
            }
        }
    
    # Save annotation
    with open(annotation_path, "w", encoding="utf-8") as f:
        json.dump(annotation, f, ensure_ascii=False, indent=2)


def download_models_from_config(config_dir: str, models_dir: str, use_auth_token: Optional[str] = None):
    """Download models specified in configuration files.
    
    Args:
        config_dir: Directory containing model configuration files
        models_dir: Directory to save models to
        use_auth_token: Optional authentication token for private repositories
    """
    os.makedirs(models_dir, exist_ok=True)
    
    # Track downloaded models
    downloaded_models = []
    
    # Process VLM configs
    vlm_config_dir = os.path.join(config_dir, "vlm")
    if os.path.exists(vlm_config_dir):
        for config_file in os.listdir(vlm_config_dir):
            if config_file.endswith(".yaml"):
                try:
                    config_path = os.path.join(vlm_config_dir, config_file)
                    with open(config_path, "r", encoding="utf-8") as f:
                        config = yaml.safe_load(f)
                    
                    if "model" in config and "params" in config["model"]:
                        model_params = config["model"]["params"]
                        if "model_name" in model_params:
                            model_name = model_params["model_name"]
                            # Extract repo_id if present, otherwise use model_name as repo_id
                            repo_id = model_params.get("repo_id", model_name)
                            
                            model_dir = os.path.join(models_dir, "vlm", model_name)
                            
                            if not os.path.exists(model_dir):
                                logger.info(f"Downloading VLM: {model_name}")
                                success = download_huggingface_model(
                                    repo_id=repo_id,
                                    output_dir=os.path.join(models_dir, "vlm"),
                                    model_name=model_name,
                                    use_auth_token=use_auth_token
                                )
                                if success:
                                    downloaded_models.append({"type": "vlm", "name": model_name, "path": model_dir})
                            else:
                                logger.info(f"VLM {model_name} already exists at {model_dir}")
                                downloaded_models.append({"type": "vlm", "name": model_name, "path": model_dir})
                except Exception as e:
                    logger.error(f"Error processing config file {config_file}: {e}")
    
    # Process LLM configs
    llm_config_dir = os.path.join(config_dir, "llm")
    if os.path.exists(llm_config_dir):
        for config_file in os.listdir(llm_config_dir):
            if config_file.endswith(".yaml"):
                try:
                    config_path = os.path.join(llm_config_dir, config_file)
                    with open(config_path, "r", encoding="utf-8") as f:
                        config = yaml.safe_load(f)
                    
                    if "model" in config and "params" in config["model"]:
                        model_params = config["model"]["params"]
                        if "model_name" in model_params:
                            model_name = model_params["model_name"]
                            # Extract repo_id if present, otherwise use model_name as repo_id
                            repo_id = model_params.get("repo_id", model_name)
                            
                            model_dir = os.path.join(models_dir, "llm", model_name)
                            
                            if not os.path.exists(model_dir):
                                logger.info(f"Downloading LLM: {model_name}")
                                success = download_huggingface_model(
                                    repo_id=repo_id,
                                    output_dir=os.path.join(models_dir, "llm"),
                                    model_name=model_name,
                                    use_auth_token=use_auth_token
                                )
                                if success:
                                    downloaded_models.append({"type": "llm", "name": model_name, "path": model_dir})
                            else:
                                logger.info(f"LLM {model_name} already exists at {model_dir}")
                                downloaded_models.append({"type": "llm", "name": model_name, "path": model_dir})
                except Exception as e:
                    logger.error(f"Error processing config file {config_file}: {e}")
    
    return downloaded_models


def create_model_registry(models_info: List[Dict[str, Any]], registry_path: str):
    """Create a registry file with information about downloaded models.
    
    Args:
        models_info: List of dictionaries with model information
        registry_path: Path to save the registry file to
    """
    registry = {
        "models": models_info,
        "timestamp": import_time_module().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Model registry created at {registry_path}")


def import_time_module():
    """Import time module to avoid circular imports"""
    import datetime
    return datetime.datetime.now()


def setup_environment(base_dir: str):
    """Setup environment variables for the project.
    
    Args:
        base_dir: Base directory of the project
    """
    # Set environment variables for models and fonts
    os.environ["BMR_MODELS_DIR"] = os.path.join(base_dir, "models")
    os.environ["BMR_DATA_DIR"] = os.path.join(base_dir, "data")
    os.environ["BMR_FONTS_DIR"] = os.path.join(base_dir, "resources", "fonts")
    
    # Set FastText model path
    fasttext_path = os.path.join(base_dir, "resources", "lid.176.ftz")
    if os.path.exists(fasttext_path):
        os.environ["LID176FTZ_PATH"] = fasttext_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Setup and download resources for BMR Model Merge",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--output_dir", type=str, default="./downloaded_resources",
        help="Directory to save downloaded resources to"
    )
    parser.add_argument(
        "--download_models", action="store_true",
        help="Download models specified in configuration files"
    )
    parser.add_argument(
        "--download_samples", action="store_true",
        help="Download sample datasets"
    )
    parser.add_argument(
        "--download_fonts", action="store_true",
        help="Download fonts for visualization"
    )
    parser.add_argument(
        "--download_fasttext", action="store_true",
        help="Download FastText language identification model"
    )
    parser.add_argument(
        "--download_all", action="store_true",
        help="Download all resources"
    )
    parser.add_argument(
        "--use_auth_token", type=str, default=None,
        help="Hugging Face authentication token for private repositories"
    )
    parser.add_argument(
        "--config_dir", type=str, default="./configs",
        help="Directory containing model configuration files"
    )
    
    args = parser.parse_args()
    
    # If no specific download option is selected, default to downloading all
    if not any([
        args.download_models,
        args.download_samples,
        args.download_fonts,
        args.download_fasttext,
        args.download_all
    ]):
        logger.info("No specific download option selected, defaulting to --download_all")
        args.download_all = True
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    return args


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Apply download_all flag
    if args.download_all:
        args.download_models = True
        args.download_samples = True
        args.download_fonts = True
        args.download_fasttext = True
    
    # Create resource directories
    models_dir = os.path.join(args.output_dir, "models")
    resources_dir = os.path.join(args.output_dir, "resources")
    
    for directory in [models_dir, resources_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Download resources
    downloaded_models = []
    
    # Download FastText model
    if args.download_fasttext:
        download_fasttext_model(resources_dir)
    
    # Download fonts
    if args.download_fonts:
        download_fonts(resources_dir)
    
    # Download sample datasets
    if args.download_samples:
        download_sample_datasets(args.output_dir)
    
    # Download models
    if args.download_models:
        downloaded_models = download_models_from_config(
            config_dir=args.config_dir,
            models_dir=models_dir,
            use_auth_token=args.use_auth_token
        )
    
    # Create model registry
    if downloaded_models:
        create_model_registry(
            models_info=downloaded_models,
            registry_path=os.path.join(args.output_dir, "model_registry.json")
        )
    
    # Setup environment
    setup_environment(args.output_dir)
    
    logger.info(f"Resource setup complete. Resources saved to {args.output_dir}")
    
    # Output summary
    print("\n" + "=" * 50)
    print("Setup Summary")
    print("=" * 50)
    print(f"Resources directory: {args.output_dir}")
    
    if args.download_models:
        print(f"\nDownloaded Models: {len(downloaded_models)}")
        for model in downloaded_models:
            print(f"  - {model['type'].upper()}: {model['name']}")
    
    if args.download_samples:
        print("\nSample datasets created in:")
        print(f"  - {os.path.join(args.output_dir, 'data', 'samples')}")
    
    if args.download_fonts:
        print("\nDownloaded fonts for visualization")
    
    if args.download_fasttext:
        print("\nDownloaded FastText language identification model")
    
    print("\nTo use these resources with the project, set the following environment variables:")
    print(f"  BMR_MODELS_DIR={os.path.join(args.output_dir, 'models')}")
    print(f"  BMR_DATA_DIR={os.path.join(args.output_dir, 'data')}")
    print(f"  BMR_FONTS_DIR={os.path.join(args.output_dir, 'resources', 'fonts')}")
    print(f"  LID176FTZ_PATH={os.path.join(args.output_dir, 'resources', 'lid.176.ftz')}")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()