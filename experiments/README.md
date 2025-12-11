# Experiments

This directory contains scripts and configurations for running experiments from the paper:

**"Learning Majority-to-Minority Transformations with MMD and Triplet Loss for Imbalanced Classification"** (arXiv:2509.11511)

## Directory Structure

```
experiments/
├── run_main_experiments.py      # Main experiment script
├── run_all_experiments.sh       # Shell script to run all experiments
├── configs/
│   ├── main_experiment.yaml     # Configuration for main experiments
│   └── default_config.yaml      # Default hyperparameter settings
└── notebooks/                   # Jupyter notebooks for analysis
```

## Quick Start

### Run Experiments

```bash
# Run with default settings (uses GPU)
python experiments/run_main_experiments.py --datasets us_crime oil car_eval_34

# Specify methods and classifiers
python experiments/run_main_experiments.py \
    --datasets us_crime oil \
    --methods ROS SMOTE bSMOTE ADASYN MWMOTE CTGAN GAMO MGVAE MMD MMD+T \
    --classifiers SVM RandomForest \
    --device cuda:0

# Quick test with reduced epochs
python experiments/run_main_experiments.py \
    --datasets us_crime \
    --n_epochs 100 \
    --n_runs 2

# Run all experiments using shell script
bash experiments/run_all_experiments.sh
```

## Paper Hyperparameters (Section 4.2)

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Regularization | λ | 0.01 | Balances MMD and triplet loss |
| Triplet neighbors | k | 5 | Neighbors for triplet mining |
| Triplet margin | α | 1.0 | Triplet loss margin |
| Optimizer | - | Adam | Default PyTorch settings |
| Architecture | - | Auto-scaled | Based on input dimension d |

The network architecture is automatically scaled based on input dimension:
- Hidden dims: `[d×2, d×4, d×8, d×16]`
- Latent dim: `d×32`

## Available Methods

| Method | Description | Package Required |
|--------|-------------|------------------|
| `MMD+T` | Proposed method (MMD + Triplet Loss) | Built-in |
| `MMD` | Proposed method (MMD only, λ=0) | Built-in |
| `ROS` | Random Oversampling | imblearn |
| `SMOTE` | Synthetic Minority Oversampling | imblearn |
| `bSMOTE` | Borderline-SMOTE | imblearn |
| `ADASYN` | Adaptive Synthetic Sampling | imblearn |
| `MWMOTE` | Majority Weighted Minority Oversampling | SMOTE_variants |
| `CTGAN` | Conditional Tabular GAN | ctgan |
| `GAMO` | GAN-based Minority Oversampling | gamosampler |
| `MGVAE` | Majority-Guided VAE | Built-in |

## Datasets

The experiments use 29 real-world imbalanced datasets (Table 1 in paper):

### From imblearn.datasets.fetch_datasets():
- us_crime, oil, car_eval_34, arrhythmia, coil_2000
- letter_img, mammography, optical_digits, ozone_level
- pen_digits, satimage, sick_euthyroid, spectrometer
- thyroid_sick, wine_quality, yeast_me2

### From .dat files (KEEL format):
- abalone9-18, abalone19, cleveland-0_vs_4, ecoli4
- glass5, led7digit-0-2-4-5-6-7-8-9_vs_1
- page-blocks-1-3_vs_4, shuttle-c0-vs-c4
- vowel0, yeast4, yeast5, yeast6

## Evaluation Protocol

Following the paper (Section 4.1):
- 10-fold cross-validation
- 10 independent trials
- Metrics: AUROC, G-mean, F1-score, MCC

## Configuration Files

### main_experiment.yaml

```yaml
# Training hyperparameters (Paper Section 4.2)
n_epochs: 1000
lr: 0.01
beta: 0.01  # λ in paper

# Triplet loss settings
k_neighbors: 5
triplet_margin: 1.0

# Evaluation settings
n_runs: 10
n_splits: 10
```

## Command Line Arguments

### run_main_experiments.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | None | Path to YAML config file |
| `--datasets` | [us_crime, oil, car_eval_34] | Datasets to evaluate |
| `--methods` | [Ours, SMOTE, ...] | Oversampling methods |
| `--classifiers` | None (all) | Classifier selection |
| `--n_epochs` | 1000 | Training epochs |
| `--lr` | 0.01 | Learning rate |
| `--beta` | 0.01 | Regularization (λ) |
| `--k_neighbors` | 5 | Triplet neighbors (k) |
| `--triplet_margin` | 1.0 | Triplet margin (α) |
| `--n_runs` | 10 | Number of trials |
| `--n_splits` | 10 | CV folds |
| `--device` | cuda | Device (cuda/cpu) |
| `--save_path` | ./results | Output directory |
| `--visualize` | False | Generate t-SNE plots |

## Results

Results are saved in CSV format with the following columns:
- Dataset name
- Method name
- Classifier name
- Metric values (mean ± std across folds)

Example output: `results/us_crime_results_gaussian_beta_0.01.csv`

