# Hyperparameter Ablation Study Configuration

This directory contains configuration files for comprehensive hyperparameter sensitivity analysis of the MOMS model.

## Configuration Files

1. **triplet_margin_ablation.yaml** - Tests triplet margin (α) values: [0.1, 0.5, 1.0, 2.0, 5.0]
2. **mmd_bandwidth_ablation.yaml** - Tests MMD kernel bandwidth (σ) values: [0.1, 0.5, 1.0, 2.0, 5.0]
3. **danger_k_ablation.yaml** - Tests danger set k values: [3, 5, 7, 10, 15]
4. **lambda_ablation_extended.yaml** - Tests lambda (β) values: [0.0, 0.1, 0.5, 1.0, 2.0]

## Usage

### Step 1: Run Ablation Studies

Run each ablation study using the configuration files:

```bash
# Triplet margin ablation
python src/experiments/run_ablation_study.py \
    --config experiments/configs/ablation_study/triplet_margin_ablation.yaml

# MMD bandwidth ablation
python src/experiments/run_ablation_study.py \
    --config experiments/configs/ablation_study/mmd_bandwidth_ablation.yaml

# Danger k ablation
python src/experiments/run_ablation_study.py \
    --config experiments/configs/ablation_study/danger_k_ablation.yaml

# Lambda ablation
python src/experiments/run_ablation_study.py \
    --config experiments/configs/ablation_study/lambda_ablation_extended.yaml
```

### Step 2: Generate Visualizations

After running the experiments, generate visualizations:

```bash
# Triplet margin
python src/experiments/visualize_ablation_results.py \
    --results-dir results/ablation_study/triplet_margin \
    --experiment-name triplet_margin_ablation

# MMD bandwidth
python src/experiments/visualize_ablation_results.py \
    --results-dir results/ablation_study/mmd_bandwidth \
    --experiment-name mmd_bandwidth_ablation

# Danger k
python src/experiments/visualize_ablation_results.py \
    --results-dir results/ablation_study/danger_k \
    --experiment-name danger_k_ablation

# Lambda
python src/experiments/visualize_ablation_results.py \
    --results-dir results/ablation_study/lambda_beta \
    --experiment-name lambda_ablation_extended
```

### Step 3: Generate Report

Generate comprehensive analysis report:

```bash
python src/experiments/generate_ablation_report.py \
    --results-dir results/ablation_study \
    --output results/ablation_study_report.md
```

## Configuration Structure

Each configuration file contains:

- **experiment_name**: Unique identifier for the experiment
- **parameter_name**: Name of the hyperparameter being tested
- **parameter_values**: List of values to test
- **model**: Model architecture configuration
- **fixed_params**: Other hyperparameters held constant
- **training**: Training configuration
- **evaluation**: Cross-validation settings
- **datasets**: List of datasets to test on
- **output**: Output directory paths

## Output Structure

Results are saved in the following structure:

```
results/ablation_study/
├── triplet_margin/
│   └── triplet_margin_ablation_results.csv
├── mmd_bandwidth/
│   └── mmd_bandwidth_ablation_results.csv
├── danger_k/
│   └── danger_k_ablation_results.csv
└── lambda_beta/
    └── lambda_ablation_extended_results.csv

figures/ablation_study/
├── triplet_margin/
│   ├── triplet_margin_ablation_detailed.png
│   ├── triplet_margin_ablation_aggregated.png
│   └── triplet_margin_ablation_summary.csv
└── ...

results/
└── ablation_study_report.md
```

## Notes

- Each experiment uses 10-fold cross-validation with 10 runs (100 total evaluations per parameter value)
- Results include mean and standard deviation for each metric
- Primary metrics: F1-score, ROC-AUC, G-mean
- Additional metrics: Precision, Recall, Balanced Accuracy

