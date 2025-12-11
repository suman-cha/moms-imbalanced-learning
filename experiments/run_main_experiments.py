#!/usr/bin/env python3
"""
Main Experiment Script for Imbalanced Classification

This script runs the main experiments for the paper:
"Learning Majority-to-Minority Transformations with MMD and Triplet Loss 
for Imbalanced Classification" (arXiv:2509.11511)

Usage:
    python run_main_experiments.py --device cuda:0 --methods Ours SMOTE --save_path ./results
    python run_main_experiments.py --config experiments/configs/default_experiment.yaml

Reference Paper Hyperparameters:
    - lambda (beta): 0.01
    - k (triplet neighbors): 5
    - margin (alpha): 1.0
    - Optimizer: Adam with default settings
"""

import os
import sys
import argparse
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


@dataclass
class ExperimentConfig:
    """Configuration for experiments following paper settings.
    
    Paper Hyperparameters (Section 4.2):
        - lambda (beta): 0.01 - Regularization parameter balancing MMD and triplet loss
        - k: 5 - Number of neighbors for triplet mining
        - margin (alpha): 1.0 - Triplet loss margin
        - Architecture: Automatically scaled based on input dimension d
    """
    # Dataset settings
    data_source: str = "imblearn"  # "imblearn" or "custom"
    data_path: Optional[str] = None  # Path for custom datasets
    
    # Training hyperparameters (Paper Section 4.2)
    n_epochs: int = 1000
    lr: float = 0.01
    beta: float = 0.01  # lambda in paper - regularization coefficient
    
    # Architecture (auto-scaled based on input dim if None)
    hidden_dims: Optional[List[int]] = None
    latent_dim: Optional[int] = None
    
    # Triplet loss settings (Paper Section 3.2)
    k_neighbors: int = 5  # k in paper
    triplet_margin: float = 1.0  # alpha in paper
    
    # MMD kernel settings
    kernel_type: str = "gaussian"
    
    # Evaluation settings (Paper Section 4.1)
    n_runs: int = 10
    n_splits: int = 10
    seed: int = 1203
    
    # Output settings
    save_path: str = "./results"
    visualize: bool = False
    
    # Device settings
    device: str = "cuda"
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def get_architecture(self, input_dim: int) -> tuple:
        """Get network architecture scaled by input dimension.
        
        Following paper: network capacity is automatically scaled based on input dimension d.
        """
        if self.hidden_dims is None:
            # Auto-scale architecture based on input dimension
            hidden_dims = [
                input_dim * 2,
                input_dim * 4,
                input_dim * 8,
                input_dim * 16
            ]
        else:
            hidden_dims = self.hidden_dims
            
        if self.latent_dim is None:
            latent_dim = input_dim * 32
        else:
            latent_dim = self.latent_dim
            
        return hidden_dims, latent_dim


# Dataset configurations for experiments
# Following Table 1 in the paper (29 real-world imbalanced datasets)
DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    # Datasets from imblearn.datasets.fetch_datasets()
    "us_crime": {"source": "imblearn", "maj_target": -1, "cat_idx": []},
    "oil": {"source": "imblearn", "maj_target": -1, "cat_idx": []},
    "car_eval_34": {"source": "imblearn", "maj_target": -1, "cat_idx": []},
    "arrhythmia": {"source": "imblearn", "maj_target": -1, "cat_idx": []},
    "coil_2000": {"source": "imblearn", "maj_target": -1, "cat_idx": []},
    "letter_img": {"source": "imblearn", "maj_target": -1, "cat_idx": []},
    "mammography": {"source": "imblearn", "maj_target": -1, "cat_idx": []},
    "optical_digits": {"source": "imblearn", "maj_target": -1, "cat_idx": []},
    "ozone_level": {"source": "imblearn", "maj_target": -1, "cat_idx": []},
    "pen_digits": {"source": "imblearn", "maj_target": -1, "cat_idx": []},
    "satimage": {"source": "imblearn", "maj_target": -1, "cat_idx": []},
    "sick_euthyroid": {"source": "imblearn", "maj_target": -1, "cat_idx": []},
    "spectrometer": {"source": "imblearn", "maj_target": -1, "cat_idx": []},
    "thyroid_sick": {"source": "imblearn", "maj_target": -1, "cat_idx": []},
    "wine_quality": {"source": "imblearn", "maj_target": -1, "cat_idx": []},
    "yeast_me2": {"source": "imblearn", "maj_target": -1, "cat_idx": []},
    
    # Datasets from .dat files (KEEL format)
    "abalone9-18": {"source": "file", "cat_idx": [0]},
    "abalone19": {"source": "file", "cat_idx": [0]},
    "cleveland-0_vs_4": {"source": "file", "cat_idx": []},
    "ecoli4": {"source": "file", "cat_idx": []},
    "glass5": {"source": "file", "cat_idx": []},
    "led7digit-0-2-4-5-6-7-8-9_vs_1": {"source": "file", "cat_idx": []},
    "page-blocks-1-3_vs_4": {"source": "file", "cat_idx": []},
    "shuttle-c0-vs-c4": {"source": "file", "cat_idx": []},
    "vowel0": {"source": "file", "cat_idx": []},
    "yeast4": {"source": "file", "cat_idx": []},
    "yeast5": {"source": "file", "cat_idx": []},
    "yeast6": {"source": "file", "cat_idx": []},
}

# Network architecture is automatically scaled based on input dimension d (Paper Section 4.2)
# - Hidden dims: [d×2, d×4, d×8, d×16]
# - Latent dim: d×32
# Per-dataset overrides are optional and generally not needed
DATASET_HYPERPARAMS: Dict[str, Dict[str, Any]] = {}


def get_base_classifiers(seed: int = 1203) -> Dict[str, Any]:
    """
    Get dictionary of base classifiers for evaluation.
    
    Following standard protocols in imbalanced learning (Krawczyk 2016),
    SVM with RBF kernel is the primary classifier.
    """
    return {
        "SVM": SVC(kernel='rbf', probability=True, random_state=seed),
        "DecisionTree": DecisionTreeClassifier(max_depth=6, random_state=seed),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(100,), 
            max_iter=1000, 
            early_stopping=True, 
            random_state=seed
        ),
        "RandomForest": RandomForestClassifier(n_estimators=500, random_state=seed),
    }


def run_experiment(
    config: ExperimentConfig,
    datasets: List[str],
    methods: List[str],
    classifiers: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run experiments on specified datasets with given methods.
    
    Args:
        config: Experiment configuration
        datasets: List of dataset names to run
        methods: List of oversampling methods to evaluate
        classifiers: List of classifier names (None = all)
        
    Returns:
        Dictionary mapping dataset names to result DataFrames
    """
    from imblearn.datasets import fetch_datasets
    from src.models.experiment import run_exp, load_dataset
    
    # Setup
    os.makedirs(config.save_path, exist_ok=True)
    all_results = {}
    
    # Load imblearn datasets once
    imblearn_data = None
    if any(DATASET_CONFIGS.get(d, {}).get("source") == "imblearn" for d in datasets):
        print("Loading imblearn datasets...")
        imblearn_data = fetch_datasets()
    
    # Get classifiers
    base_models = get_base_classifiers(config.seed)
    if classifiers is not None:
        base_models = {k: v for k, v in base_models.items() if k in classifiers}
    
    for data_name in datasets:
        print(f"\n{'='*60}")
        print(f"Running experiment on: {data_name}")
        print(f"{'='*60}")
        
        # Get dataset config
        ds_config = DATASET_CONFIGS.get(data_name, {})
        source = ds_config.get("source", "file")
        cat_idx = ds_config.get("cat_idx", [])
        maj_target = ds_config.get("maj_target", None)
        
        # Get dataset-specific hyperparameters
        ds_params = DATASET_HYPERPARAMS.get(data_name, {})
        hidden_dims = ds_params.get("hidden_dims", config.hidden_dims)
        latent_dim = ds_params.get("latent_dim", config.latent_dim)
        lr = ds_params.get("lr", config.lr)
        beta = ds_params.get("beta", config.beta)
        
        # Load data
        try:
            if source == "imblearn":
                data = imblearn_data
            else:
                if config.data_path is None:
                    raise ValueError(f"data_path required for file-based dataset: {data_name}")
                data = pd.read_csv(f"{config.data_path}/{data_name}.dat", header=None)
                maj_target = None  # File datasets use different label format
        except Exception as e:
            print(f"Error loading dataset {data_name}: {e}")
            continue
        
        # Build loss parameters
        loss_params = {
            "k": config.k_neighbors,
            "margin": config.triplet_margin,
        }
        
        # Run experiment
        try:
            results = run_exp(
                data=data,
                cat_idx=cat_idx,
                methods=methods,
                base_model=base_models,
                device=config.device,
                n_epochs=config.n_epochs,
                hidden_dims=hidden_dims,
                latent_dim=latent_dim,
                lr=lr,
                beta=beta,
                data_name=data_name,
                maj_target_name=maj_target,
                n_runs=config.n_runs,
                n_splits=config.n_splits,
                seed=config.seed,
                visualize=config.visualize,
                save_path=config.save_path,
                kernel_type=config.kernel_type,
                loss_params=loss_params,
            )
            all_results[data_name] = results
            
        except Exception as e:
            print(f"Error running experiment on {data_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return all_results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run imbalanced classification experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default settings on specific datasets
    python run_main_experiments.py --datasets us_crime oil --device cuda:0
    
    # Run all methods with custom save path
    python run_main_experiments.py --methods Ours SMOTE ADASYN --save_path ./my_results
    
    # Use configuration file
    python run_main_experiments.py --config experiments/configs/main_exp.yaml
    
    # Run quick test (reduced epochs and runs)
    python run_main_experiments.py --datasets us_crime --n_epochs 100 --n_runs 2
        """
    )
    
    # Configuration file
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML configuration file"
    )
    
    # Dataset selection
    parser.add_argument(
        "--datasets", nargs="+", type=str,
        default=["us_crime", "oil", "car_eval_34"],
        help="List of dataset names to run experiments on"
    )
    
    # Method selection
    parser.add_argument(
        "--methods", nargs="+", type=str,
        default=["ROS", "SMOTE", "bSMOTE", "ADASYN", "MWMOTE", "CTGAN", "GAMO", "MGVAE", "MMD", "MMD+T"],
        help="List of oversampling methods to evaluate"
    )
    
    # Classifier selection
    parser.add_argument(
        "--classifiers", nargs="+", type=str, default=None,
        help="List of classifiers (default: all)"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--n_epochs", type=int, default=1000,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01,
        help="Learning rate"
    )
    parser.add_argument(
        "--beta", type=float, default=0.01,
        help="Regularization coefficient (lambda in paper)"
    )
    
    # Triplet loss settings
    parser.add_argument(
        "--k_neighbors", type=int, default=5,
        help="Number of neighbors for triplet mining (k in paper)"
    )
    parser.add_argument(
        "--triplet_margin", type=float, default=1.0,
        help="Triplet loss margin (alpha in paper)"
    )
    
    # Evaluation settings
    parser.add_argument(
        "--n_runs", type=int, default=10,
        help="Number of experimental runs"
    )
    parser.add_argument(
        "--n_splits", type=int, default=10,
        help="Number of CV folds"
    )
    parser.add_argument(
        "--seed", type=int, default=1203,
        help="Random seed"
    )
    
    # I/O settings
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="Path to dataset directory (for .dat files)"
    )
    parser.add_argument(
        "--save_path", type=str, default="./results",
        help="Path to save results"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run on (cuda, cuda:0, cpu)"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate t-SNE visualizations"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create configuration
    if args.config is not None:
        config = ExperimentConfig.from_yaml(args.config)
    else:
        config = ExperimentConfig(
            data_path=args.data_path,
            n_epochs=args.n_epochs,
            lr=args.lr,
            beta=args.beta,
            k_neighbors=args.k_neighbors,
            triplet_margin=args.triplet_margin,
            n_runs=args.n_runs,
            n_splits=args.n_splits,
            seed=args.seed,
            save_path=args.save_path,
            device=args.device,
            visualize=args.visualize,
        )
    
    # Set CUDA device if specified
    if config.device.startswith("cuda"):
        device_id = config.device.split(":")[-1] if ":" in config.device else "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    
    print("=" * 60)
    print("Imbalanced Classification Experiment")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Datasets: {args.datasets}")
    print(f"Methods: {args.methods}")
    print(f"Hyperparameters:")
    print(f"  - n_epochs: {config.n_epochs}")
    print(f"  - lr: {config.lr}")
    print(f"  - beta (lambda): {config.beta}")
    print(f"  - k_neighbors: {config.k_neighbors}")
    print(f"  - triplet_margin (alpha): {config.triplet_margin}")
    print(f"  - n_runs: {config.n_runs}, n_splits: {config.n_splits}")
    print(f"Save path: {config.save_path}")
    print("=" * 60)
    
    # Run experiments
    results = run_experiment(
        config=config,
        datasets=args.datasets,
        methods=args.methods,
        classifiers=args.classifiers,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("Experiment Complete!")
    print("=" * 60)
    print(f"Results saved to: {config.save_path}")
    print(f"Datasets processed: {len(results)}")
    
    return results


if __name__ == "__main__":
    main()

