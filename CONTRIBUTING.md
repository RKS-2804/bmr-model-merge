# Contributing to BMR Model Merge

Thank you for your interest in contributing to the BMR Model Merge project! This document provides guidelines and instructions for contributing to the project.

## Development Setup

### Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/) for dependency management
- CUDA-compatible GPU (recommended for training and inference)

### Installation for Development

1. Clone the repository:
```bash
git clone https://github.com/your-org/bmr-model-merge.git
cd bmr-model-merge
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Activate the virtual environment:
```bash
poetry shell
```

## Project Structure

- `configs/` - Model configuration files
  - `llm/` - Language model configs
  - `vlm/` - Vision-language model configs
- `evomerge/` - Core implementation
  - `data/` - Dataset handling and preprocessing
  - `eval/` - Evaluation metrics and benchmarks
  - `evolution/` - Evolutionary optimization algorithms
  - `models/` - Model implementations
  - `processors/` - Text and image processing utilities
- `data/` - Data configurations and sample datasets
- `demo.py` - Interactive demo script
- `evaluate.py` - Evaluation script
- `run_evolution.py` - Script to run evolutionary model merging

## Development Workflow

### Adding New Features

1. Create a new branch for your feature:
```bash
git checkout -b feature/your-feature-name
```

2. Implement your feature, following the project's coding style and guidelines
3. Write tests for your feature
4. Update documentation as necessary
5. Submit a pull request

### Code Style

- Follow PEP 8 guidelines for Python code
- Use type hints wherever possible
- Document your code using docstrings in Google style
- Keep line length to 88 characters (Black formatter compatible)

### Testing

Before submitting your changes, run the tests to ensure everything works as expected:

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_your_feature.py
```

## Working with Japanese OCR

### Japanese Language Considerations

- Always handle text using UTF-8 encoding
- Use appropriate Japanese text normalization tools (included in `evomerge.data.processor`)
- Consider both vertical and horizontal text orientations
- Handle mixed script (Kanji, Hiragana, Katakana, Latin) appropriately

### Data Handling

When working with the dataset classes:

1. Ensure proper character encoding in file paths and text
2. Use `JapaneseDocumentProcessor` for preprocessing
3. Apply appropriate augmentations depending on the document type

Example:
```python
from evomerge.data import JapaneseReceiptDataset, JapaneseDocumentProcessor

# Create a processor with Japanese-specific settings
processor = JapaneseDocumentProcessor(
    apply_deskew=True,
    apply_contrast=True,
    apply_noise_reduction=True,
    detect_orientation=True
)

# Create dataset with the processor
dataset = JapaneseReceiptDataset(
    data_dir="path/to/data",
    processor=processor,
    split="train"
)
```

### Model Development

When developing or modifying OCR models:

1. Always test with Japanese text samples
2. Validate that the model handles various fonts and styles
3. Check performance on both printed and handwritten text
4. Ensure proper integration with the field extraction pipeline

## Evolutionary Merging

### Adding New Evolution Strategies

1. Implement your strategy in the `evomerge.evolution` package
2. Follow the existing APIs for compatibility
3. Document parameters and expected behavior
4. Add appropriate metrics for fitness evaluation

### Configuration Files

When creating configuration files for models or evolutionary runs:

1. Place in the appropriate directory (`configs/llm/` or `configs/vlm/`)
2. Follow the YAML structure of existing configs
3. Document all parameters
4. Include example usage in comments

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create a new release on GitHub with release notes
4. Ensure documentation is updated for the new version

## Getting Help

If you have questions or need assistance:

- Open an issue on GitHub
- Check existing documentation
- Contact the project maintainers

Thank you for contributing to BMR Model Merge!