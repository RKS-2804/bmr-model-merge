# BMR-Model-Merge Workflow Guide

This guide provides detailed step-by-step instructions for using the BMR-model-merge framework for Japanese OCR optimization. It covers various workflows from basic usage to advanced optimization scenarios.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Basic Workflows](#basic-workflows)
  - [Interactive Demo](#interactive-demo)
  - [Single Document OCR](#single-document-ocr)
- [Optimization Workflows](#optimization-workflows)
  - [Complete Automated Workflow](#complete-automated-workflow)
  - [Custom Optimization Run](#custom-optimization-run)
  - [Algorithm Comparison](#algorithm-comparison)
- [Configuration Guide](#configuration-guide)
  - [Dataset Configuration](#dataset-configuration)
  - [Model Configuration](#model-configuration)
  - [Optimization Parameters](#optimization-parameters)
- [Checkpoint Management](#checkpoint-management)
- [Advanced Usage](#advanced-usage)
  - [Custom Model Integration](#custom-model-integration)
  - [Custom Metrics](#custom-metrics)
  - [Hyperparameter Optimization](#hyperparameter-optimization)
- [Troubleshooting](#troubleshooting)

## Prerequisites

Before starting, ensure you have:

1. Python 3.9 or higher installed
2. CUDA-compatible GPU for optimal performance (optional but recommended)
3. Approximately 20GB of free disk space for model downloads
4. Japanese language support installed on your system

## Basic Workflows

### Interactive Demo

The interactive demo provides a user-friendly interface to test OCR capabilities:

1. **Launch the Web UI**:
   ```bash
   python demo.py --mode web
   ```

2. **Use the Interface**:
   - Upload an image or use sample images
   - Adjust processing parameters if needed
   - View OCR results and field extraction
   - Try different model combinations

3. **CLI Alternative**:
   ```bash
   python demo.py --mode cli --image_path data/samples/invoice_1.jpg --output_path results/ocr_output.json
   ```

### Single Document OCR

For processing individual documents:

1. **Run the OCR script**:
   ```bash
   python run_ocr.py --image_path path/to/document.jpg --model_checkpoint checkpoints/best_model.pt --output_format json
   ```

2. **Process multiple documents**:
   ```bash
   python run_ocr.py --image_dir path/to/documents/ --model_checkpoint checkpoints/best_model.pt --output_dir path/to/results/
   ```

## Optimization Workflows

### Complete Automated Workflow

For an end-to-end optimization experience:

1. **Run the complete workflow**:
   ```bash
   python run_complete_workflow.py --config configs/evolution/default.yaml
   ```

2. **This will automatically**:
   - Download required models
   - Prepare evaluation datasets
   - Run optimization using all algorithms
   - Compare results
   - Save the best model
   - Generate evaluation reports

### Custom Optimization Run

For more control over the optimization process:

1. **Choose your algorithm**:
   ```bash
   # For BMR optimization
   python run_optimization.py --algorithm bmr --config configs/evolution/bmr_config.yaml
   
   # For BWR optimization
   python run_optimization.py --algorithm bwr --config configs/evolution/bwr_config.yaml
   
   # For genetic algorithm optimization
   python run_optimization.py --algorithm genetic --config configs/evolution/genetic_config.yaml
   ```

2. **Customize population size and generations**:
   ```bash
   python run_optimization.py --algorithm bmr --population_size 50 --generations 30 --save_checkpoints
   ```

3. **Fine-tune for specific dataset**:
   ```bash
   python run_optimization.py --algorithm bmr --dataset ja_invoices --target_metric f1_score --eval_steps 5
   ```

### Algorithm Comparison

To compare the performance of different optimization algorithms:

1. **Run the comparison script**:
   ```bash
   python compare_results.py --checkpoints checkpoints/bmr_best.pt checkpoints/bwr_best.pt checkpoints/genetic_best.pt --output_path results/comparison.json
   ```

2. **Visualize the comparison**:
   ```bash
   python visualize_results.py --comparison_file results/comparison.json --plot_type convergence
   ```

## Configuration Guide

### Dataset Configuration

Customize the dataset settings in `data/sample_config.yaml`:

```yaml
dataset:
  name: "japanese_invoices"
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  augmentations:
    rotation: true
    noise: true
    blur: false
    contrast: true
  preprocessing:
    resize_width: 800
    resize_height: 1200
    normalize: true
    grayscale: false
```

### Model Configuration

Configure VLM models in `configs/vlm/{model_name}.yaml`:

```yaml
model:
  name: "vila-jp"
  checkpoint: "path/to/vila-jp-checkpoint"
  weights_path: "path/to/weights"
  backbone: "transformer"
  vision_encoder:
    type: "vit"
    patch_size: 16
    hidden_size: 1024
  language_decoder:
    type: "transformer"
    layers: 24
    hidden_size: 1024
  params:
    temperature: 0.7
    max_length: 512
    top_p: 0.9
```

### Optimization Parameters

Configure optimization parameters in `configs/evolution/{algorithm}_config.yaml`:

```yaml
optimization:
  algorithm: "bmr"  # or "bwr" or "genetic"
  population_size: 30
  generations: 20
  mutation_rate: 0.1  # only for genetic
  crossover_rate: 0.7  # only for genetic
  elitism_rate: 0.1
  fitness_metrics:
    - name: "character_accuracy"
      weight: 0.4
    - name: "field_extraction_f1"
      weight: 0.6
  early_stopping:
    patience: 5
    min_delta: 0.001
```

## Checkpoint Management

To manage optimization checkpoints:

1. **List available checkpoints**:
   ```bash
   python utils/checkpoint_manager.py --list
   ```

2. **Resume from checkpoint**:
   ```bash
   python run_optimization.py --algorithm bmr --resume_from checkpoints/bmr_gen_10.pt
   ```

3. **Export checkpoint to ONNX**:
   ```bash
   python utils/convert_to_onnx.py --checkpoint checkpoints/best_model.pt --output models/best_model.onnx
   ```

## Advanced Usage

### Custom Model Integration

To integrate a custom model:

1. Create a new model class in `evomerge/models/`:
   ```python
   from evomerge.models.base_ocr import OCRModel
   
   class CustomOCRModel(OCRModel):
       def __init__(self, config):
           super().__init__(config)
           # Custom initialization
           
       def load_model(self):
           # Custom loading logic
           
       def preprocess(self, image):
           # Custom preprocessing
           
       def postprocess(self, output):
           # Custom postprocessing
           
       def __call__(self, image):
           # Custom inference
   ```

2. Register your model in `evomerge/models/__init__.py`:
   ```python
   from evomerge.models.custom_ocr import CustomOCRModel
   
   MODEL_REGISTRY = {
       # Existing models
       "japanese_ocr": JapaneseOCRModel,
       "custom_ocr": CustomOCRModel,
   }
   ```

3. Create a configuration file in `configs/vlm/custom_model.yaml`

### Custom Metrics

To implement a custom evaluation metric:

1. Create a new metric in `evomerge/eval/metrics.py`:
   ```python
   def custom_japanese_metric(pred_text, target_text):
       # Implement custom metric
       return score
   ```

2. Register it in `evomerge/eval/__init__.py`:
   ```python
   METRICS_REGISTRY = {
       # Existing metrics
       "character_accuracy": character_accuracy_metric,
       "field_extraction_f1": field_extraction_f1_metric,
       "custom_metric": custom_japanese_metric,
   }
   ```

### Hyperparameter Optimization

To run hyperparameter optimization:

```bash
python utils/hyperparameter_search.py --algorithm bmr --search_space configs/hparam_search.yaml --trials 20
```

## Troubleshooting

### Common Issues and Solutions

1. **CUDA Out of Memory**:
   - Reduce batch size in configuration
   - Use a smaller model variant
   - Try CPU-only mode with `--no_cuda` flag

2. **Japanese Character Rendering Issues**:
   - Ensure proper CJK font installation
   - Set environment variable: `export PYTHONIOENCODING=utf-8`
   - Check terminal/console encoding supports UTF-8

3. **Slow Optimization Performance**:
   - Reduce population size or generations
   - Disable unnecessary evaluation metrics
   - Use checkpointing to save progress

4. **Field Extraction Errors**:
   - Check document orientation
   - Adjust confidence threshold: `--field_confidence 0.7`
   - Try template matching: `--use_templates true`

5. **Model Loading Errors**:
   - Verify correct model paths in configuration
   - Run `python utils/verify_downloads.py` to check model integrity
   - Try re-downloading models: `python setup_resources.py --force_download`

For further assistance, submit an issue on GitHub with:
- Error message and stack trace
- Configuration used
- System information