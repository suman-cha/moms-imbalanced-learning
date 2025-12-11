<p align="center">
  <h1 align="center">MOMS: Majority-to-Minority Transformation<br/>with MMD and Triplet Loss</h1>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2509.11511"><img src="https://img.shields.io/badge/arXiv-2509.11511-b31b1b.svg" alt="arXiv"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch"></a>
  <a href="mailto:oldrain123@yonsei.ac.kr"><img src="https://img.shields.io/badge/contact-oldrain123%40yonsei.ac.kr-green.svg" alt="Contact"></a>
</p>

<p align="center">
  <b>Official implementation of <a href="https://arxiv.org/abs/2509.11511">"Learning Majority-to-Minority Transformations with MMD and Triplet Loss for Imbalanced Classification"</a></b>
</p>

---

## Abstract

Class imbalance in supervised classification often degrades model performance by biasing predictions toward the majority class. Traditional oversampling techniques generate synthetic minority samples via local interpolation but fail to capture global data distributions in high-dimensional spaces. Deep generative models offer richer distribution modeling yet suffer from training instability under severe imbalance.

We introduce **MOMS**, an oversampling framework that learns a parametric transformation to map majority samples into the minority distribution. Our approach minimizes the **Maximum Mean Discrepancy (MMD)** between transformed and true minority samples for global alignment, and incorporates a **triplet loss regularizer** to enforce boundary awareness by guiding synthesized samples toward challenging borderline regions.

<p align="center">
  <img src="https://img.shields.io/badge/Evaluated_on-29_datasets-orange" alt="Datasets">
  <img src="https://img.shields.io/badge/Metrics-AUROC_|_G--mean_|_F1_|_MCC-blue" alt="Metrics">
</p>

---

## Key Features

- **TransMap Network**: Encoder-decoder architecture with skip connections for majority → minority transformation
- **MMD Loss**: Global distribution alignment with adaptive bandwidth (median heuristic)
- **Local Triplet Loss**: Boundary-aware regularization using danger/safe set decomposition
- **Comprehensive Baselines**: ROS, SMOTE, bSMOTE, ADASYN, MWMOTE, CTGAN, GAMO, MGVAE
- **Config-Driven Experiments**: YAML-based hyperparameter ablations with checkpointing and multi-run aggregation

---

## Installation

### Requirements
- Python ≥ 3.8
- PyTorch ≥ 2.0
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone repository
git clone https://github.com/oldrain123/moms-imbalanced-learning.git
cd moms-imbalanced-learning

# Option 1: Conda (recommended for GPU)
conda env create -f experiments/configs/environment.yml
conda activate imb_clf

# Option 2: pip
pip install -r requirements.txt
```

### Environment Variables

```bash
# Linux/Mac
export PYTHONPATH=$PWD:$PWD/src

# Windows (PowerShell)
$env:PYTHONPATH = "$PWD;$PWD\src"
```

---

## Repository Structure

```
moms-imbalanced-learning/
├── src/
│   ├── models/
│   │   ├── moms_losses.py      # MMD, triplet loss, danger set computation
│   │   ├── moms_generate.py    # Sample generation utilities
│   │   ├── moms_metrics.py     # AUROC, G-mean, MCC, F1, mAP
│   │   └── kernels.py          # Kernel functions (Gaussian, Laplacian, IMQ, RQ)
│   ├── training/
│   │   └── moms_train.py       # TransMap training loop
│   └── utils/
│       ├── moms_utils.py       # Seed management, I/O
│       └── moms_visualize.py   # t-SNE, plotting utilities
├── experiments/
│   ├── run_main_experiments.py # Main experiment script
│   ├── run_all_experiments.sh  # Shell script to run all experiments
│   └── configs/
│       ├── ablation_study/     # Hyperparameter ablation configs
│       ├── default_config.yaml
│       └── environment.yml     # Conda environment specification
└── requirements.txt
```

---

## Quick Start

### 1. Prepare Data

Place datasets in `data/raw/` (KEEL `.dat` format or `.csv`) or use datasets from `imbalanced-learn`:

```python
from imblearn.datasets import fetch_datasets
datasets = fetch_datasets()  # 27 curated imbalanced datasets
```

### 2. Run Experiments

Run the full experiment pipeline with standard baselines:

```bash
# Run on specific datasets
python experiments/run_main_experiments.py \
    --datasets us_crime oil \
    --methods MMD+T MMD SMOTE ADASYN \
    --device cuda

# Run all experiments using shell script
bash experiments/run_all_experiments.sh
```

## Experiments

### Available Methods

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

### Hyperparameter Ablations

| Parameter | Config File | Description |
|-----------|------------|-------------|
| Triplet margin | `triplet_margin_ablation.yaml` | Controls positive/negative separation |
| MMD bandwidth | `mmd_bandwidth_ablation.yaml` | Kernel bandwidth for distribution matching |
| Danger-k | `danger_k_ablation.yaml` | k-NN threshold for borderline detection |
| Lambda | `lambda_ablation.yaml` | Triplet loss weight |

### Evaluation Protocol

- **Cross-validation**: 10-fold stratified CV × 10 runs
- **Metrics**: AUROC, G-mean, F1-score, MCC
- **Classifiers**: SVM, Decision Tree, Random Forest, k-NN, MLP

---

## Configuration

Example configuration (`triplet_margin_ablation.yaml`):

```yaml
experiment_name: "triplet_margin_ablation"
parameter_name: "triplet_margin"
parameter_values: [0.1, 0.5, 1.0, 2.0, 5.0]

model:
  n_epochs: 2000
  lr: 0.001
  hidden_dims: [16, 32, 64, 128]
  latent_dim: 256

fixed_params:
  mmd_sigma: 1.0
  danger_k: 5
  lambda_beta: 0.01

evaluation:
  n_runs: 10
  n_splits: 10
  random_state: 1203
  metrics: ["roc_auc", "g_mean", "mcc", "f1_score"]

datasets:
  - name: "ecoli3"
    path: "data/raw/ecoli3.dat"
  - name: "yeast6"
    path: "imblearn"  # Load from imbalanced-learn
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{cha2025moms,
  title     = {Learning Majority-to-Minority Transformations with MMD and 
               Triplet Loss for Imbalanced Classification},
  author    = {Cha, Suman and Kim, Hyunjoong},
  journal   = {arXiv preprint arXiv:2509.11511},
  year      = {2025},
  url       = {https://arxiv.org/abs/2509.11511}
}
```

---

## Contact

- **Suman Cha** — [oldrain123@yonsei.ac.kr](mailto:oldrain123@yonsei.ac.kr)
- Issues and PRs are welcome!

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [imbalanced-learn](https://imbalanced-learn.org/) for benchmark datasets
- [SMOTE-variants](https://github.com/analyticalmindsltd/smote_variants) library
- PyTorch team for the deep learning framework
