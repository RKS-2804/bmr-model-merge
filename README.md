# BMR-Model-Merge

An advanced framework for Japanese invoice/receipt OCR optimization using evolutionary model merging techniques, featuring BMR (Best-Mean-Random) and BWR (Best-Worst-Random) parameter-free optimization algorithms.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<div align="center">
  <img src="assets/japanese_invoice_example.png" alt="Japanese Invoice Example" width="400"/>
</div>

## Features

- **Japanese OCR Specialization**: Optimized for Japanese invoices and receipts with specialized text normalization and field extraction
- **Multiple Optimization Algorithms**:
  - **Genetic Algorithm**: Traditional evolutionary optimization with crossover and mutation
  - **BMR Algorithm**: Parameter-free Best-Mean-Random optimization 
  - **BWR Algorithm**: Parameter-free Best-Worst-Random optimization
- **Interactive Demo**: Web UI for testing OCR capabilities with visual feedback
- **CLI Workflow**: Complete automation pipeline for training and evaluation
- **Checkpoint Management**: Track and compare optimization runs
- **Detailed Documentation**: Architecture diagrams and workflow guides

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/bmr-model-merge.git
cd bmr-model-merge

# Install dependencies
pip install -e .
```

## Quick Start

### Interactive Demo

```bash
# Launch the Gradio web UI
python demo.py --mode web

# Or use CLI mode
python demo.py --mode cli --image_path path/to/invoice.jpg
```

### Running Optimization

```bash
# Run complete workflow (download models, prepare data, optimize, evaluate)
python run_complete_workflow.py --config configs/evolution/default.yaml

# Run specific optimization algorithm
python run_optimization.py --algorithm bmr --config configs/evolution/bmr_config.yaml
```

## Project Structure

```
bmr-model-merge/
├── assets/                   # Architecture diagrams and images
├── configs/                  # Configuration files
│   ├── evolution/           # Optimization algorithm configs
│   ├── llm/                 # LLM model configs
│   └── vlm/                 # Vision-language model configs
├── data/                    # Sample data and datasets
├── evomerge/                # Core library modules
│   ├── data/                # Dataset and processing modules
│   ├── eval/                # Evaluation metrics and utilities
│   ├── evolution/           # Optimization algorithms
│   ├── models/              # Model definitions
│   └── processors/          # Text and image processors
├── scripts/                 # Utility scripts
├── tests/                   # Unit and integration tests
├── demo.py                  # Interactive demo application
├── evaluate.py              # Model evaluation script
├── run_optimization.py      # Optimization runner
└── run_complete_workflow.py # End-to-end workflow
```

## Optimization Algorithms

### BMR (Best-Mean-Random)

BMR is a parameter-free optimization algorithm that leverages the best solution, mean of the population, and a randomly selected solution to guide the search process. It balances exploitation and exploration with a simple adaptive mechanism.

Key features:
- No hyperparameters to tune
- Balances exploration and exploitation dynamically
- Resilient to local optima

### BWR (Best-Worst-Random)

BWR is a variation of BMR that uses the worst solution instead of the mean to provide directional guidance away from poor solutions. It can be more effective in problems with deceptive local optima.

Key features:
- Learns from both good and bad solutions
- More aggressive exploration than BMR
- Effective for complex, multi-modal landscapes

## Japanese OCR Workflow

The framework specifically addresses challenges in Japanese document OCR:

1. **Preprocessing**: Orientation correction, contrast enhancement
2. **Text Extraction**: Specialized OCR for vertical and horizontal Japanese text
3. **Field Detection**: Layout analysis for invoice/receipt structure
4. **Text Normalization**: Converting between character forms (half-width/full-width)
5. **Field Extraction**: Intelligent mapping of text to structured data

## Architecture

For detailed architecture information, see the [Architecture Document](assets/architecture.md).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project implements algorithms based on research in parameter-free optimization
- Special thanks to contributors and the open-source community