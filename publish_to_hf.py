#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publish model to Hugging Face Hub
Author: Ravikumar Shah
"""

import os
import sys
import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, login
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("hf-publisher")

def setup_args():
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(description="Publish model to Hugging Face Hub")
    
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the .pt model file")
    parser.add_argument("--base_model", type=str, required=True,
                        help="Base model architecture to use (e.g., stockmark/stockmark-2-vl-100b-beta)")
    parser.add_argument("--repo_name", type=str, required=True,
                        help="Name of the repository on Hugging Face (e.g., yourusername/model-name)")
    parser.add_argument("--token", type=str, required=False,
                        help="Hugging Face token. If not provided, will look for HF_TOKEN env variable")
    parser.add_argument("--commit_message", type=str, default="Upload BMR/BWR merged model",
                        help="Commit message for the model upload")
    
    return parser.parse_args()

def convert_and_publish(model_path, base_model, repo_name, token, commit_message):
    """Convert the model to HF format and publish it."""
    
    logger.info(f"Loading model from {model_path}...")
    
    try:
        # Load the .pt model
        merged_model_state = torch.load(model_path, map_location="cpu")
        
        # Load base model and tokenizer from HF
        logger.info(f"Loading base model architecture from {base_model}...")
        base_model_obj = AutoModelForCausalLM.from_pretrained(base_model)
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        # Apply merged weights to base model
        logger.info("Applying merged weights to base model...")
        # This is a simplified approach - actual implementation depends on how weights are merged
        if isinstance(merged_model_state, dict) and 'state_dict' in merged_model_state:
            base_model_obj.load_state_dict(merged_model_state['state_dict'], strict=False)
        elif isinstance(merged_model_state, dict):
            base_model_obj.load_state_dict(merged_model_state, strict=False)
        else:
            # Handle other formats as needed
            logger.warning("Model format not recognized, attempting direct loading...")
            base_model_obj = merged_model_state
        
        # Create model card
        logger.info("Creating model card...")
        model_card = f"""---
language: ja
license: apache-2.0
tags:
- japanese
- ocr
- document-understanding
- evolutionary-merging
- bmr
- bwr
datasets:
- japanese-invoices
- japanese-receipts
---

# BMR/BWR Evolved Model for Japanese OCR

This model was created using Evolutionary Model Merging techniques (BMR/BWR algorithms) to optimize performance on Japanese invoice and receipt OCR tasks.

## Model Details

- **Architecture:** Based on {base_model}
- **Training:** Evolved using BMR/BWR algorithms with optimized merging weights
- **Developer:** Ravikumar Shah
- **Language:** Japanese

## Use Cases

- Japanese invoice processing
- Receipt text extraction
- Document field detection
- Japanese OCR tasks

## Performance

This model was optimized for:
- Character recognition accuracy 
- Field extraction accuracy
- Processing speed

## Limitations

- Optimized specifically for Japanese business documents
- May not perform as well on handwritten text
- Requires clear document images for best results
"""

        # Login to Hugging Face
        if token:
            login(token)
        else:
            # Try to get token from environment variable
            env_token = os.environ.get("HF_TOKEN")
            if not env_token:
                raise ValueError("No Hugging Face token provided. Please provide --token or set HF_TOKEN environment variable")
            login(env_token)
        
        # Create repo and push model
        logger.info(f"Publishing model to {repo_name}...")
        
        # Save model and tokenizer to temporary directory
        temp_dir = "temp_model_dir"
        os.makedirs(temp_dir, exist_ok=True)
        
        base_model_obj.save_pretrained(temp_dir)
        tokenizer.save_pretrained(temp_dir)
        
        # Write model card
        with open(f"{temp_dir}/README.md", "w") as f:
            f.write(model_card)
            
        # Create metadata file with additional information
        metadata = {
            "base_model": base_model,
            "merging_algorithm": "bmr_bwr_evolutionary",
            "author": "Ravikumar Shah",
            "version": "1.0.0"
        }
        
        with open(f"{temp_dir}/merging_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        # Push to hub
        api = HfApi()
        api.create_repo(repo_id=repo_name, exist_ok=True)
        api.upload_folder(
            folder_path=temp_dir,
            repo_id=repo_name,
            commit_message=commit_message
        )
        
        logger.info(f"Successfully published model to {repo_name}")
        logger.info(f"Model URL: https://huggingface.co/{repo_name}")
        
    except Exception as e:
        logger.error(f"Error publishing model: {str(e)}")
        raise

if __name__ == "__main__":
    args = setup_args()
    convert_and_publish(
        args.model_path, 
        args.base_model, 
        args.repo_name,
        args.token,
        args.commit_message
    )