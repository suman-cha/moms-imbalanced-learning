# MOMS: Minority Oversampling with Majority Selection

A deep learning-based approach for handling imbalanced classification problems using Maximum Mean Discrepancy (MMD) and triplet loss.

## Overview

MOMS (Minority Oversampling with Majority Selection) is a novel method for generating synthetic minority class samples in imbalanced learning scenarios. The model uses a neural network-based transformation map to learn a mapping from majority to minority class distributions, leveraging MMD loss for distribution matching and triplet loss for local structure preservation.

## Features

- **MMD-based Distribution Matching**: Uses Maximum Mean Discrepancy to align majority and minority distributions
- **Triplet Loss for Local Structure**: Preserves local neighborhood structure through triplet constraints
- **Adaptive Kernel Bandwidth**: Automatically adjusts kernel bandwidth based on data characteristics
- **Comprehensive Ablation Studies**: Framework for hyperparameter sensitivity analysis
- **Scalability Testing**: Tools for testing high-dimensional and sparse data scenarios

## Project Structure

```
learnig_majority_to_minority/
├── src/                          # Source code
│   ├── models/                   # Model definitions
│   │   ├── moms_losses.py       # MMD and triplet loss functions
│   │   ├── moms_generate.py     # Data generation functions
│   │   ├── moms_metrics.py      # Evaluation metrics
│   │   └── kernels.py           # Kernel functions
│   ├── training/                 # Training scripts
│   │   └── moms_train.py        # Main training function
│   ├── utils/                    # Utility functions
│   │   ├── moms_utils.py        # General utilities
│   │   └── moms_visualize.py    # Visualization tools
│   └── experiments/              # Experiment scripts
│       ├── run_ablation_study.py
│       ├── run_scalability_test.py
│       ├── run_sparse_test.py
│       └── visualize_*.py
├── experiments/                  # Experiment configurations
│   └── configs/
│       ├── ablation_study/       # Ablation study configs
│       └── scalability_test/     # Scalability test configs
├── data/                         # Data directory
│   ├── raw/                      # Raw datasets
│   └── processed/                # Processed datasets
├── results/                      # Experiment results
│   ├── ablation_study/           # Ablation study results
│   └── scalability_analysis/     # Scalability test results
├── figures/                      # Generated figures
│   ├── ablation_study/           # Ablation study plots
│   └── scalability/              # Scalability plots
├── custom_packages/              # Custom packages
│   └── boost/                    # Boosting algorithms
├── pydpc/                        # Local pydpc package
└── requirements.txt              # Python dependencies
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/REPOSITORY_NAME.git
cd REPOSITORY_NAME
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Install PyTorch (if needed)

For CPU-only:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For GPU support (CUDA):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Install Local pydpc Package (Optional)

If you need the `pydpc` package for clustering-based selection:

```bash
cd pydpc
pip install -e .
cd ..
```

**Note:** On Windows, `pydpc` may require C++ build tools. If installation fails, the code includes a fallback mechanism.

### Step 5: Set Up Environment Variables (Windows)

For Windows, you may need to set `PYTHONPATH`:

```powershell
$env:PYTHONPATH = "$PWD;$PWD\src;$PWD\src\models;$PWD\src\training;$PWD\src\utils;$PWD\pydpc"
```

Or create a batch file:

```batch
set PYTHONPATH=%CD%;%CD%\src;%CD%\src\models;%CD%\src\training;%CD%\src\utils;%CD%\pydpc
```

## Usage

### Quick Start

1. **Prepare your data**: Place your dataset in `data/raw/` directory. Supported formats: CSV, DAT (LIBSVM format).

2. **Run a quick test**:
```bash
python src/experiments/run_ablation_study.py --config experiments/configs/ablation_study/quick_test.yaml
```

### Running Ablation Studies

The framework supports ablation studies for four key hyperparameters:

1. **Triplet Margin (α)**
```bash
python src/experiments/run_ablation_study.py --config experiments/configs/ablation_study/triplet_margin_ablation.yaml
```

2. **MMD Kernel Bandwidth (σ)**
```bash
python src/experiments/run_ablation_study.py --config experiments/configs/ablation_study/mmd_bandwidth_ablation.yaml
```

3. **Danger Set k Value**
```bash
python src/experiments/run_ablation_study.py --config experiments/configs/ablation_study/danger_k_ablation.yaml
```

4. **Lambda (β) - Triplet Loss Weight**
```bash
python src/experiments/run_ablation_study.py --config experiments/configs/ablation_study/lambda_ablation_extended.yaml
```

### Running Scalability Tests

1. **High-Dimensional Test**:
```bash
python src/experiments/run_scalability_test.py --config experiments/configs/scalability_test/high_dimensional_test.yaml
```

2. **Sparse Setting Test**:
```bash
python src/experiments/run_sparse_test.py --config experiments/configs/scalability_test/sparse_minority_test.yaml
```

### Visualizing Results

After running experiments, generate visualizations:

```bash
# Ablation study visualizations
python src/experiments/visualize_ablation_results.py --results_dir results/ablation_study/triplet_margin

# Scalability test visualizations
python src/experiments/visualize_scalability.py --results_dir results/scalability_analysis
```

### Generating Reports

Generate markdown reports with analysis:

```bash
# Ablation study report
python src/experiments/generate_ablation_report.py --results_dir results/ablation_study

# Scalability test report
python src/experiments/generate_scalability_report.py --results_dir results/scalability_analysis
```

## Configuration Files

Experiments are configured using YAML files in `experiments/configs/`. Each configuration file specifies:

- Model hyperparameters (network architecture, learning rate, epochs)
- Fixed hyperparameters (not being tested)
- Evaluation settings (number of runs, cross-validation folds, metrics)
- Dataset list
- Output directories

Example configuration structure:

```yaml
experiment_name: "triplet_margin_ablation"
parameter_name: "triplet_margin"
parameter_values: [0.1, 0.5, 1.0, 2.0, 5.0]

model:
  n_epochs: 2000
  lr: 0.001
  hidden_dims: [64, 32]
  latent_dim: 16

fixed_params:
  mmd_sigma: 1.0
  danger_k: 5
  lambda_beta: 0.01

evaluation:
  n_runs: 10
  n_splits: 10
  metrics: ["f1_score", "roc_auc", "g_mean"]
```

## Key Hyperparameters

- **`triplet_margin` (α)**: Margin for triplet loss, controls separation between positive and negative samples
- **`mmd_sigma` (σ)**: Bandwidth parameter for MMD Gaussian kernel
- **`danger_k`**: Number of nearest neighbors for identifying "danger" minority samples
- **`lambda_beta` (β)**: Weight for triplet loss in the combined objective

## Evaluation Metrics

The framework evaluates performance using multiple metrics:

- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **G-Mean**: Geometric mean of sensitivity and specificity
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **Balanced Accuracy**: Average of sensitivity and specificity

## Data Format

### Supported Formats

1. **CSV**: Comma-separated values, last column is the target label
2. **DAT (LIBSVM)**: Space-separated, format: `label feature1:value1 feature2:value2 ...`

### Dataset Requirements

- Binary classification (will be converted automatically)
- Last column should be the target label
- Features should be numeric (categorical features will be encoded)

## Troubleshooting

### Import Errors

If you encounter `ModuleNotFoundError`:

1. Check that all dependencies are installed: `pip install -r requirements.txt`
2. Verify `PYTHONPATH` is set correctly (especially on Windows)
3. Ensure you're running from the project root directory

### pydpc Installation Issues

If `pydpc` fails to install (common on Windows):

- The code includes a fallback mechanism that uses alternative methods
- You can skip `pydpc` installation if you're not using `OUBoost`
- For Linux/Mac, try: `pip install pydpc` or build from source in `pydpc/` directory

### CUDA/GPU Issues

If CUDA is not available:

- The code will automatically fall back to CPU
- Set `device: "cpu"` in your configuration file
- Install CPU-only PyTorch if you don't have a GPU

### Memory Issues

For large datasets or high-dimensional data:

- Reduce `n_epochs` in configuration
- Use smaller network architectures (`hidden_dims`, `latent_dim`)
- Process datasets in batches
- Consider using dimensionality reduction (PCA/UMAP) for very high-dimensional data

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

[Specify your license here]

## Citation

If you use this code in your research, please cite:

```bibtex
@article{moms2024,
  title={MOMS: Minority Oversampling with Majority Selection},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].

## Acknowledgments

- This project uses the SMOTE variants library
- Thanks to the imbalanced-learn community
- Built with PyTorch and scikit-learn

