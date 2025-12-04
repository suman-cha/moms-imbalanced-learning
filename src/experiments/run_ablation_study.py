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
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    f1_score, roc_auc_score, matthews_corrcoef
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

# Import imblearn for dataset loading
try:
    from imblearn.datasets import fetch_datasets
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False


def encode_categorical_features(X):
    """
    Encode categorical features to numerical values.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix that may contain categorical columns.
        
    Returns
    -------
    X_encoded : pd.DataFrame
        Feature matrix with all categorical columns encoded as numerical.
    """
    X_encoded = X.copy()
    
    # Identify categorical columns (object or category dtype)
    categorical_columns = X_encoded.select_dtypes(include=['object', 'category']).columns
    
    # Encode each categorical column
    for col in categorical_columns:
        le = LabelEncoder()
        # Handle missing values by converting to string first
        X_encoded[col] = X_encoded[col].astype(str)
        X_encoded[col] = le.fit_transform(X_encoded[col])
    
    return X_encoded


def load_dataset_from_file(file_path: str, dataset_name: str = None):
    """
    Load dataset from file or imblearn.datasets.
    
    Supports:
    - File paths (.dat and .csv formats)
    - imblearn.datasets.fetch_datasets() when file_path == "imblearn"
    - Automatic encoding of categorical features
    
    Parameters
    ----------
    file_path : str
        Path to the dataset file, or "imblearn" to use fetch_datasets().
    dataset_name : str, optional
        Dataset name when using imblearn (required if file_path == "imblearn").
        
    Returns
    -------
    X : pd.DataFrame
        Feature matrix with all numerical values.
    y : np.ndarray
        Binary labels (0: majority, 1: minority).
    """
    # Handle imblearn.datasets
    if file_path == "imblearn" or str(file_path) == "imblearn":
        if not IMBLEARN_AVAILABLE:
            raise ImportError("imblearn package is not installed. Install it with: pip install imbalanced-learn")
        
        if dataset_name is None:
            raise ValueError("dataset_name is required when using imblearn")
        
        # Fetch all datasets
        datasets_dict = fetch_datasets()
        
        if dataset_name not in datasets_dict:
            available = list(datasets_dict.keys())[:10]  # Show first 10
            raise ValueError(
                f"Dataset '{dataset_name}' not found in imblearn.datasets. "
                f"Available datasets (showing first 10): {available}"
            )
        
        # Get the specific dataset
        dataset = datasets_dict[dataset_name]
        X = pd.DataFrame(dataset.data)
        y = dataset.target
        
        # Encode categorical features
        X = encode_categorical_features(X)
        
        # Convert labels to binary (assume minority is the less frequent class)
        unique_labels, counts = np.unique(y, return_counts=True)
        minority_label = unique_labels[np.argmin(counts)]
        y_binary = np.where(y == minority_label, 1, 0)
        
        return X, y_binary
    
    # Handle file paths
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    if file_path.suffix == '.dat':
        # Load KEEL format dataset
        data = pd.read_csv(file_path, header=None, sep=',')
        # Last column is typically the target
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1].values
        
        # Encode categorical features
        X = encode_categorical_features(X)
        
        # Convert labels to binary (assume minority is the less frequent class)
        unique_labels, counts = np.unique(y, return_counts=True)
        minority_label = unique_labels[np.argmin(counts)]
        y_binary = np.where(y == minority_label, 1, 0)
        
        return X, y_binary
    elif file_path.suffix == '.csv':
        data = pd.read_csv(file_path)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1].values
        
        # Encode categorical features
        X = encode_categorical_features(X)
        
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
    Compute evaluation metrics for imbalanced classification.
    
    Core metrics: AUROC, G-mean, F1-score, MCC
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    y_proba : np.ndarray, optional
        Predicted probabilities (required for AUROC).
        
    Returns
    -------
    metrics : dict
        Dictionary containing only the 4 core metrics.
    """
    from sklearn.metrics import confusion_matrix
    
    # Initialize metrics dictionary
    metrics = {}
    
    # 1. F1-score: Harmonic mean of precision and recall
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    
    # 2. MCC: Matthews Correlation Coefficient
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    # 3. AUROC: Area Under the ROC Curve (requires probability estimates)
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['roc_auc'] = 0.0
    else:
        metrics['roc_auc'] = 0.0
    
    # 4. G-mean: Geometric mean of sensitivity and specificity
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
        # triplet_alpha should equal triplet_margin
        triplet_alpha = margin
        beta = fixed_params['lambda_beta']
        k = fixed_params['danger_k']
        sigma = fixed_params['mmd_sigma']
    elif param_name == 'mmd_sigma':
        sigma = param_value
        margin = fixed_params['triplet_margin']
        # triplet_alpha should equal triplet_margin
        triplet_alpha = margin
        beta = fixed_params['lambda_beta']
        k = fixed_params['danger_k']
    elif param_name == 'danger_k':
        k = int(param_value)
        margin = fixed_params['triplet_margin']
        # triplet_alpha should equal triplet_margin
        triplet_alpha = margin
        beta = fixed_params['lambda_beta']
        sigma = fixed_params['mmd_sigma']
    elif param_name == 'lambda_beta':
        beta = param_value
        margin = fixed_params['triplet_margin']
        # triplet_alpha should equal triplet_margin
        triplet_alpha = margin
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


def run_ablation_study(config_path: str, output_dir: str = None, resume: bool = True):
    """
    Run complete ablation study for a hyperparameter.
    
    Parameters
    ----------
    config_path : str
        Path to YAML configuration file.
    output_dir : str, optional
        Output directory for results. If None, uses config output settings.
    resume : bool, optional
        If True, resume from checkpoint if available. Default is True.
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
    
    # Define checkpoint file
    checkpoint_file = output_dir / f"{config['experiment_name']}_checkpoint.csv"
    
    # Load existing results if resuming
    completed_experiments = set()
    all_results = []
    
    if resume and checkpoint_file.exists():
        print(f"\n{'='*60}")
        print(f"Resuming from checkpoint: {checkpoint_file}")
        print(f"{'='*60}\n")
        
        checkpoint_df = pd.read_csv(checkpoint_file)
        all_results = checkpoint_df.to_dict('records')
        
        # Track completed (dataset, parameter_value) combinations
        for row in all_results:
            key = (row['dataset'], row['parameter_value'])
            completed_experiments.add(key)
        
        print(f"Loaded {len(completed_experiments)} completed experiments")
        print(f"Total rows in checkpoint: {len(all_results)}\n")
    
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
    
    print(f"\n{'='*60}")
    print(f"Starting Ablation Study: {config['experiment_name']}")
    print(f"Parameter: {param_name}")
    print(f"Values: {param_values}")
    print(f"Datasets: {len(datasets)}")
    print(f"Completed experiments: {len(completed_experiments)}")
    print(f"{'='*60}\n")
    
    # Iterate over datasets with progress bar
    dataset_pbar = tqdm(datasets, desc="Datasets", position=0, leave=True)
    for dataset_info in dataset_pbar:
        dataset_name = dataset_info['name']
        dataset_path = dataset_info['path']
        
        dataset_pbar.set_description(f"Dataset: {dataset_name}")
        
        try:
            # Load dataset
            # Pass dataset_name for imblearn datasets
            X, y = load_dataset_from_file(dataset_path, dataset_name=dataset_name)
            dataset_pbar.set_postfix({
                'shape': X.shape,
                'imbalance': f"{np.sum(y==0)/np.sum(y==1):.2f}"
            })
            
            # Iterate over parameter values with progress bar
            param_pbar = tqdm(param_values, desc=f"  {param_name} values", position=1, leave=False)
            for param_value in param_pbar:
                param_pbar.set_description(f"  {param_name}={param_value}")
                
                # Check if this experiment is already completed
                experiment_key = (dataset_name, param_value)
                if experiment_key in completed_experiments:
                    dataset_pbar.write(f"  Skipping {dataset_name} with {param_name}={param_value} (already completed)")
                    continue
                
                # Store results for this parameter value
                param_results = defaultdict(list)
                
                # Calculate total folds for progress tracking
                total_folds = n_runs * n_splits
                
                # Run cross-validation with progress bar
                fold_pbar = tqdm(total=total_folds, desc="    Folds", position=2, leave=False)
                fold_idx = 0
                
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
                        
                        fold_idx += 1
                        fold_pbar.update(1)
                        fold_pbar.set_postfix({
                            'F1': f"{np.mean(param_results.get('f1_score', [0])):.3f}",
                            'AUC': f"{np.mean(param_results.get('roc_auc', [0])):.3f}"
                        })
                
                fold_pbar.close()
                
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
                
                # Save checkpoint after each parameter value
                try:
                    results_df = pd.DataFrame(all_results)
                    results_df.to_csv(checkpoint_file, index=False)
                    dataset_pbar.write(f"  Checkpoint saved: {len(all_results)} total rows")
                except Exception as e:
                    dataset_pbar.write(f"  Warning: Failed to save checkpoint: {e}")
                
                # Mark this experiment as completed
                completed_experiments.add(experiment_key)
                
                # Update parameter progress bar with summary
                param_pbar.set_postfix({
                    'F1': f"{np.mean(param_results['f1_score']):.4f}±{np.std(param_results['f1_score']):.4f}",
                    'AUC': f"{np.mean(param_results.get('roc_auc', [0])):.4f}±{np.std(param_results.get('roc_auc', [0])):.4f}"
                })
            
            param_pbar.close()
        
        except Exception as e:
            dataset_pbar.write(f"  Error processing dataset {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    dataset_pbar.close()
    
    # Save final results
    results_df = pd.DataFrame(all_results)
    results_file = output_dir / f"{config['experiment_name']}_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\n{'='*60}")
    print(f"Final results saved to: {results_file}")
    print(f"Total experiments completed: {len(completed_experiments)}")
    print(f"Total result rows: {len(all_results)}")
    print(f"{'='*60}")
    
    # Create a backup of the checkpoint file
    if checkpoint_file.exists():
        backup_file = output_dir / f"{config['experiment_name']}_checkpoint_backup.csv"
        results_df.to_csv(backup_file, index=False)
        print(f"Checkpoint backup saved to: {backup_file}")
    
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
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Start from scratch instead of resuming from checkpoint'
    )
    
    args = parser.parse_args()
    
    # Run ablation study
    results = run_ablation_study(
        args.config, 
        args.output_dir,
        resume=not args.no_resume
    )
    
    print(f"\n{'='*60}")
    print("Ablation Study Complete!")
    print(f"Total result rows: {len(results)}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

