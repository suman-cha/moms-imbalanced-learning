"""
Hyperparameter Ablation Study Runner for MOMS Model.

This script performs comprehensive hyperparameter sensitivity analysis
for the MOMS (Minority Oversampling with Majority Selection) model.

Hyperparameters tested:
1. Triplet margin (α)
2. MMD kernel bandwidth (σ)
3. Danger set k value
4. Lambda (β) - triplet loss weight
"""

import os
import sys
import yaml
import json
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    balanced_accuracy_score
)

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from src.training.moms_train import train_map
from src.models.moms_losses import (
    MMD_est_torch, compute_danger_set, local_triplet_loss,
    adaptive_kernel_width
)
from src.utils.moms_utils import set_seed
from src.models.moms_generate import transform
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


def load_dataset_from_file(file_path: str):
    """
    Load dataset from file (supports .dat and .csv formats).
    
    Parameters
    ----------
    file_path : str
        Path to the dataset file.
        
    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : np.ndarray
        Binary labels (0: majority, 1: minority).
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    if file_path.suffix == '.dat':
        # Load KEEL format dataset
        data = pd.read_csv(file_path, header=None, sep=',')
        # Last column is typically the target
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1].values
        
        # Convert labels to binary (assume minority is the less frequent class)
        unique_labels, counts = np.unique(y, return_counts=True)
        minority_label = unique_labels[np.argmin(counts)]
        y_binary = np.where(y == minority_label, 1, 0)
        
        return X, y_binary
    elif file_path.suffix == '.csv':
        data = pd.read_csv(file_path)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1].values
        
        # Convert labels to binary
        unique_labels, counts = np.unique(y, return_counts=True)
        minority_label = unique_labels[np.argmin(counts)]
        y_binary = np.where(y == minority_label, 1, 0)
        
        return X, y_binary
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def get_base_classifiers(seed):
    """
    Get base classifiers for evaluation.
    
    Returns
    -------
    classifiers : dict
        Dictionary of classifier name to classifier instance.
    """
    return {
        'SVM': SVC(kernel='rbf', probability=True, random_state=seed),
        'DT': DecisionTreeClassifier(random_state=seed),
        'RF': RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
        'kNN': KNeighborsClassifier(n_neighbors=5),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(100,), 
            max_iter=1000,  # Increased from 500
            early_stopping=True,  # Enable early stopping
            n_iter_no_change=10,  # Stop if no improvement for 10 iterations
            random_state=seed
        )
    }


def compute_metrics(y_true, y_pred, y_proba=None):
    """
    Compute evaluation metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    y_proba : np.ndarray, optional
        Predicted probabilities.
        
    Returns
    -------
    metrics : dict
        Dictionary of computed metrics.
    """
    metrics = {
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
    }
    
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['roc_auc'] = 0.0
    
    # G-mean = sqrt(recall_majority * recall_minority)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        recall_majority = tn / (tn + fp) if (tn + fp) > 0 else 0
        recall_minority = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['g_mean'] = np.sqrt(recall_majority * recall_minority)
    else:
        metrics['g_mean'] = 0.0
    
    return metrics


def run_single_experiment(
    X_train, y_train, X_test, y_test,
    config, param_value, param_name,
    device, seed
):
    """
    Run a single experiment with given hyperparameter value.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Test labels.
    config : dict
        Configuration dictionary.
    param_value : float or int
        Value of the hyperparameter being tested.
    param_name : str
        Name of the hyperparameter.
    device : str
        Device to use ('cuda' or 'cpu').
    seed : int
        Random seed.
        
    Returns
    -------
    metrics : dict
        Evaluation metrics.
    """
    set_seed(seed)
    
    # Separate majority and minority classes
    X_maj = X_train[y_train == 0].astype(np.float32)
    X_min = X_train[y_train == 1].astype(np.float32)
    
    if len(X_min) == 0 or len(X_maj) == 0:
        return {metric: 0.0 for metric in config['evaluation']['metrics']}
    
    # Prepare hyperparameters
    model_config = config['model']
    fixed_params = config['fixed_params']
    
    # Set the hyperparameter being tested
    if param_name == 'triplet_margin':
        margin = param_value
        triplet_alpha = fixed_params.get('triplet_alpha', 0.3)
        beta = fixed_params['lambda_beta']
        k = fixed_params['danger_k']
        sigma = fixed_params['mmd_sigma']
    elif param_name == 'mmd_sigma':
        sigma = param_value
        margin = fixed_params['triplet_margin']
        triplet_alpha = fixed_params.get('triplet_alpha', 0.3)
        beta = fixed_params['lambda_beta']
        k = fixed_params['danger_k']
    elif param_name == 'danger_k':
        k = int(param_value)
        margin = fixed_params['triplet_margin']
        triplet_alpha = fixed_params.get('triplet_alpha', 0.3)
        beta = fixed_params['lambda_beta']
        sigma = fixed_params['mmd_sigma']
    elif param_name == 'lambda_beta':
        beta = param_value
        margin = fixed_params['triplet_margin']
        triplet_alpha = fixed_params.get('triplet_alpha', 0.3)
        k = fixed_params['danger_k']
        sigma = fixed_params['mmd_sigma']
    else:
        raise ValueError(f"Unknown parameter name: {param_name}")
    
    # Train TransMap model
    try:
        # Prepare loss function parameters
        loss_params = config['training'].get('loss_params', {}).copy()
        if not config['model'].get('median_bw', True):
            loss_params['h'] = sigma
        
        # Train model
        model = train_map(
            X_maj=X_maj,
            X_min=X_min,
            in_dim=X_train.shape[1],
            latent_dim=model_config.get('latent_dim'),
            hidden_dims=model_config.get('hidden_dims'),
            loss_fn=MMD_est_torch,
            kernel_type=config['training']['kernel_type'],
            device=device,
            n_epochs=model_config['n_epochs'],
            lr=model_config['lr'],
            beta=beta,
            k=k,
            seed=seed,
            residual=model_config.get('residual', True),
            median_bw=config['model'].get('median_bw', True),
            loss_params=loss_params,
            triplet_margin=margin,
            triplet_alpha=triplet_alpha
        )
        
        # Generate synthetic samples
        n_trans = len(X_maj) - len(X_min)
        if n_trans > 0:
            X_trans_init = torch.tensor(X_maj[:n_trans], dtype=torch.float32).to(device)
            with torch.no_grad():
                _, X_trans = model(X_trans_init)
            X_synthetic = X_trans.cpu().numpy()
            
            # Combine with original data
            X_train_aug = np.vstack([X_train, X_synthetic])
            y_train_aug = np.hstack([y_train, np.ones(len(X_synthetic))])
        else:
            X_train_aug = X_train
            y_train_aug = y_train
        
        # Evaluate with multiple base classifiers
        classifiers = get_base_classifiers(seed)
        all_metrics = {}
        
        for clf_name, classifier in classifiers.items():
            try:
                classifier.fit(X_train_aug, y_train_aug)
                y_pred = classifier.predict(X_test)
                y_proba = classifier.predict_proba(X_test)[:, 1] if hasattr(classifier, 'predict_proba') else None
                
                clf_metrics = compute_metrics(y_test, y_pred, y_proba)
                
                # Store metrics with classifier name prefix
                for metric_name, metric_value in clf_metrics.items():
                    all_metrics[f"{clf_name}_{metric_name}"] = metric_value
                    
            except Exception as e:
                print(f"  Warning: {clf_name} failed: {e}")
                # Store zeros for failed classifier
                for metric_name in config['evaluation']['metrics']:
                    all_metrics[f"{clf_name}_{metric_name}"] = 0.0
        
        # Average metrics across all classifiers
        metrics = {}
        for metric_name in config['evaluation']['metrics']:
            metric_values = [all_metrics.get(f"{clf_name}_{metric_name}", 0.0) 
                           for clf_name in classifiers.keys()]
            metrics[metric_name] = np.mean(metric_values) if metric_values else 0.0
        
    except Exception as e:
        print(f"Error in experiment: {e}")
        import traceback
        traceback.print_exc()
        metrics = {metric: 0.0 for metric in config['evaluation']['metrics']}
    
    return metrics


def run_ablation_study(config_path: str, output_dir: str = None):
    """
    Run complete ablation study for a hyperparameter.
    
    Parameters
    ----------
    config_path : str
        Path to YAML configuration file.
    output_dir : str, optional
        Output directory for results. If None, uses config output settings.
    """
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Set output directory
    if output_dir is None:
        output_dir = Path(config['output']['results_dir'])
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = config['training']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Get parameter values
    param_name = config['parameter_name']
    param_values = config['parameter_values']
    datasets = config['datasets']
    
    # Evaluation settings
    n_runs = config['evaluation']['n_runs']
    n_splits = config['evaluation']['n_splits']
    random_state = config['evaluation']['random_state']
    metrics_list = config['evaluation']['metrics']
    
    # Store results
    all_results = []
    
    print(f"\n{'='*60}")
    print(f"Starting Ablation Study: {config['experiment_name']}")
    print(f"Parameter: {param_name}")
    print(f"Values: {param_values}")
    print(f"Datasets: {len(datasets)}")
    print(f"{'='*60}\n")
    
    # Iterate over datasets
    for dataset_info in datasets:
        dataset_name = dataset_info['name']
        dataset_path = dataset_info['path']
        
        print(f"\nProcessing dataset: {dataset_name}")
        print(f"  Path: {dataset_path}")
        
        try:
            # Load dataset
            X, y = load_dataset_from_file(dataset_path)
            print(f"  Shape: {X.shape}, Imbalance ratio: {np.sum(y==0)/np.sum(y==1):.2f}")
            
            # Iterate over parameter values
            for param_value in param_values:
                print(f"  Testing {param_name}={param_value}...")
                
                # Store results for this parameter value
                param_results = defaultdict(list)
                
                # Run cross-validation
                for run in range(n_runs):
                    skf = StratifiedKFold(
                        n_splits=n_splits,
                        random_state=random_state + run,
                        shuffle=True
                    )
                    
                    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                        X_train, X_test = X.iloc[train_idx].values, X.iloc[test_idx].values
                        y_train, y_test = y[train_idx], y[test_idx]
                        
                        # Standardize features
                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)
                        
                        # Run experiment
                        fold_seed = random_state + run * 1000 + fold * 100
                        metrics = run_single_experiment(
                            X_train, y_train, X_test, y_test,
                            config, param_value, param_name,
                            device, fold_seed
                        )
                        
                        # Store results
                        for metric_name, metric_value in metrics.items():
                            param_results[metric_name].append(metric_value)
                
                # Compute statistics
                for metric_name in metrics_list:
                    values = param_results[metric_name]
                    if len(values) > 0:
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        
                        all_results.append({
                            'dataset': dataset_name,
                            'parameter': param_name,
                            'parameter_value': param_value,
                            'metric': metric_name,
                            'mean': mean_val,
                            'std': std_val,
                            'n_runs': n_runs,
                            'n_splits': n_splits
                        })
                
                # Print summary
                print(f"    Results: F1={np.mean(param_results['f1_score']):.4f}±{np.std(param_results['f1_score']):.4f}, "
                      f"AUC={np.mean(param_results.get('roc_auc', [0])):.4f}±{np.std(param_results.get('roc_auc', [0])):.4f}")
        
        except Exception as e:
            print(f"  Error processing dataset {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_file = output_dir / f"{config['experiment_name']}_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description='Run hyperparameter ablation study')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Run ablation study
    results = run_ablation_study(args.config, args.output_dir)
    
    print(f"\n{'='*60}")
    print("Ablation Study Complete!")
    print(f"Total experiments: {len(results)}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

