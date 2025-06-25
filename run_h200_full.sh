#!/bin/bash
# Full implementation script for running BMR model merge on H200 GPU
# Created by Ravikumar Shah

# Stop on first error
set -e

echo "=== Setting up H200 GPU environment for full-scale model merging ==="

# Create data directories if they don't exist
mkdir -p data/jp_invoices/validation
mkdir -p data/jp_receipts/validation

# Set optimal environment variables for H200 GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use all available GPUs
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export NCCL_P2P_DISABLE=1  # Can help with multi-GPU communication
export CUDA_DEVICE_MAX_CONNECTIONS=1  # Optimize connection pooling

# GPU memory optimization for 100B+ models
echo "=== Configuring system for large models ==="
# Increase shared memory limit if needed
if [ $(df -kh /dev/shm | awk 'NR==2 {print $2}' | tr -d 'G') -lt 64 ]; then
    echo "Increasing shared memory size..."
    sudo mount -o remount,size=64G /dev/shm
fi

# Set PyTorch to use TF32 on Ampere GPUs (better performance with acceptable precision)
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export TORCH_CUDNN_V8_API_ENABLED=1

# Configure tensor parallelism for model loading
export TP_DEGREE=2
export DP_DEGREE=2

echo "=== Clearing CUDA cache and optimizing GPU memory ==="
python -c "import torch; torch.cuda.empty_cache(); print('CUDA cache cleared')"

echo "=== Installing necessary packages if missing ==="
pip install --upgrade torch==2.1.0 accelerate==0.27.2 transformers==4.38.2 bitsandbytes==0.41.0 flash-attn==2.5.0 vllm==0.3.0

echo "=== Starting full workflow with H200 optimized configuration ==="
# Create output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="results/h200_run_${TIMESTAMP}"

# Run the workflow with our optimized H200 configuration
python run_complete_workflow.py \
    --config configs/evolution/h200_full.yaml \
    --output-dir ${OUTPUT_DIR} \
    --algorithms bmr bwr genetic \
    --generations 100 \
    --models 10

echo "=== Workflow complete ==="
echo "Results saved to: ${OUTPUT_DIR}"
echo "Best model saved to: ${OUTPUT_DIR}/checkpoints/bmr_best.npy (or similar)"

# Print GPU stats at the end
echo "=== Final GPU stats ==="
nvidia-smi

echo "=== Converting model to PyTorch format for inference and Hugging Face ==="
# Convert the best model to PyTorch format
OUTPUT_MODEL_DIR="${OUTPUT_DIR}/optimized_model"
mkdir -p ${OUTPUT_MODEL_DIR}

# First try BMR (best algorithm), fall back to other algorithms if needed
if [ -f "${OUTPUT_DIR}/checkpoints/bmr_best.npy" ]; then
    BEST_MODEL="${OUTPUT_DIR}/checkpoints/bmr_best.npy"
    ALGORITHM="bmr"
elif [ -f "${OUTPUT_DIR}/checkpoints/bwr_best.npy" ]; then
    BEST_MODEL="${OUTPUT_DIR}/checkpoints/bwr_best.npy"
    ALGORITHM="bwr"
else
    BEST_MODEL="${OUTPUT_DIR}/checkpoints/genetic_best.npy"
    ALGORITHM="genetic"
fi

echo "Using best model from ${ALGORITHM} algorithm: ${BEST_MODEL}"

# Save as PyTorch format (.pt)
python -c "
import torch
import numpy as np
from evomerge.evolution.bmr import load_best_model

# Load the best model
model_path = '${BEST_MODEL}'
print(f'Loading model from {model_path}')
model = load_best_model(model_path)

# Save in PyTorch format
torch_path = '${OUTPUT_MODEL_DIR}/model_optimized.pt'
torch.save(model, torch_path, _use_new_zipfile_serialization=True)
print(f'Model saved in PyTorch format to {torch_path}')

# Also save metadata for Hugging Face
import json
metadata = {
    'algorithm': '${ALGORITHM}',
    'creation_date': '$(date +"%Y-%m-%d")',
    'author': 'Ravikumar Shah',
    'description': 'Japanese OCR model created using evolutionary model merging'
}
with open('${OUTPUT_MODEL_DIR}/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
"

echo "=== Run complete ==="
echo "You can now use the optimized model for inference with:"
echo "python demo.py --model ${OUTPUT_MODEL_DIR}/model_optimized.pt"

echo ""
echo "=== Publishing to Hugging Face Hub (Optional) ==="
echo "To publish this model to Hugging Face Hub, run:"
echo "python publish_to_hf.py --model_path ${OUTPUT_MODEL_DIR}/model_optimized.pt --base_model stockmark/stockmark-2-vl-100b-beta --repo_name your-username/japanese-ocr-merged --token YOUR_HF_TOKEN"