"""
Extreme Sparse Setting Test - MOMS vs SMOTE with Few Minority Samples.

This script tests MOMS vs traditional methods (SMOTE, ADASYN, Borderline-SMOTE)
when there are very few minority samples (10, 20, 50).
"""

import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE

# Add paths
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.training.moms_train import train_map
from src.models.moms_losses import MMD_est_torch
from src.utils.moms_utils import set_seed
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def load_dataset_from_file(file_path: str):
    """Load dataset from file."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    data = pd.read_csv(file_path, header=None, sep=',')
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    # Convert to binary labels
    unique_labels, counts = np.unique(y, return_counts=True)
    minority_label = unique_labels[np.argmin(counts)]
    y_binary = np.where(y == minority_label, 1, 0)
    
    return X, y_binary


def create_sparse_setting(X, y, n_minority_samples, seed=42):
    """
    Create sparse setting by limiting minority samples.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Labels.
    n_minority_samples : int
        Number of minority samples to keep.
    seed : int
        Random seed.
        
    Returns
    -------
    X_sparse, y_sparse : Sparse dataset
    """
    np.random.seed(seed)
    
    # Get indices
    maj_idx = np.where(y == 0)[0]
    min_idx = np.where(y == 1)[0]
    
    # Limit minority samples
    if len(min_idx) > n_minority_samples:
        min_idx_selected = np.random.choice(min_idx, size=n_minority_samples, replace=False)
    else:
        min_idx_selected = min_idx
    
    # Combine (keep all majority samples)
    selected_idx = np.concatenate([maj_idx, min_idx_selected])
    np.random.shuffle(selected_idx)
    
    return X[selected_idx], y[selected_idx]


def apply_oversampling(X_train, y_train, method, k_neighbors, seed):
    """
    Apply oversampling method.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    method : str
        Method name ('SMOTE', 'ADASYN', 'BorderlineSMOTE', 'MOMS', 'Original').
    k_neighbors : int
        Number of neighbors for SMOTE-based methods.
    seed : int
        Random seed.
        
    Returns
    -------
    X_resampled, y_resampled : Resampled data
    """
    if method == 'Original':
        return X_train, y_train
    
    n_minority = np.sum(y_train == 1)
    
    if method == 'SMOTE':
        try:
            if n_minority < k_neighbors + 1:
                k_neighbors = max(1, n_minority - 1)
            smote = SMOTE(k_neighbors=k_neighbors, random_state=seed)
            return smote.fit_resample(X_train, y_train)
        except Exception as e:
            print(f"    SMOTE failed: {e}, using original data")
            return X_train, y_train
    
    elif method == 'ADASYN':
        try:
            if n_minority < k_neighbors + 1:
                k_neighbors = max(1, n_minority - 1)
            adasyn = ADASYN(n_neighbors=k_neighbors, random_state=seed)
            return adasyn.fit_resample(X_train, y_train)
        except Exception as e:
            print(f"    ADASYN failed: {e}, using original data")
            return X_train, y_train
    
    elif method == 'BorderlineSMOTE':
        try:
            if n_minority < k_neighbors + 1:
                k_neighbors = max(1, n_minority - 1)
            bsmote = BorderlineSMOTE(k_neighbors=k_neighbors, random_state=seed)
            return bsmote.fit_resample(X_train, y_train)
        except Exception as e:
            print(f"    BorderlineSMOTE failed: {e}, using original data")
            return X_train, y_train
    
    else:
        raise ValueError(f"Unknown method: {method}")


def run_sparse_experiment(
    X_train, y_train, X_test, y_test,
    config, method, device, seed
):
    """
    Run single sparse setting experiment.
    
    Returns
    -------
    metrics : dict
        Performance metrics.
    """
    set_seed(seed)
    
    try:
        n_minority = np.sum(y_train == 1)
        
        # Get k_neighbors from config
        k_neighbors_map = config['smote_config']['k_neighbors_map']
        k_neighbors = k_neighbors_map.get(n_minority, 5)
        
        if method == 'MOMS':
            # Use MOMS
            X_maj = X_train[y_train == 0].astype(np.float32)
            X_min = X_train[y_train == 1].astype(np.float32)
            
            if len(X_min) < 5:
                print(f"    Too few minority samples ({len(X_min)}), skipping MOMS")
                return None
            
            model_config = config['model']
            fixed_params = config['fixed_params']
            
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
                beta=fixed_params['lambda_beta'],
                k=min(fixed_params['danger_k'], len(X_min) - 1),  # Adjust k
                seed=seed,
                residual=model_config.get('residual', True),
                median_bw=config['model'].get('median_bw', True),
                triplet_margin=fixed_params['triplet_margin'],
                triplet_alpha=fixed_params['triplet_alpha']
            )
            
            # Generate synthetic samples
            n_trans = len(X_maj) - len(X_min)
            if n_trans > 0:
                X_trans_init = torch.tensor(X_maj[:n_trans], dtype=torch.float32).to(device)
                with torch.no_grad():
                    _, X_trans = model(X_trans_init)
                X_synthetic = X_trans.cpu().numpy()
                
                X_train_aug = np.vstack([X_train, X_synthetic])
                y_train_aug = np.hstack([y_train, np.ones(len(X_synthetic))])
            else:
                X_train_aug = X_train
                y_train_aug = y_train
        else:
            # Use traditional oversampling
            X_train_aug, y_train_aug = apply_oversampling(
                X_train, y_train, method, k_neighbors, seed
            )
        
        # Train classifiers
        classifiers = {
            'RF': RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=-1),
            'SVM': SVC(kernel='rbf', probability=True, random_state=seed),
            'DT': DecisionTreeClassifier(random_state=seed),
            'MLP': MLPClassifier(
                hidden_layer_sizes=(100,),
                max_iter=1000,
                early_stopping=True,
                n_iter_no_change=10,
                random_state=seed
            )
        }
        
        all_metrics = {}
        for clf_name, classifier in classifiers.items():
            try:
                classifier.fit(X_train_aug, y_train_aug)
                y_pred = classifier.predict(X_test)
                y_proba = classifier.predict_proba(X_test)[:, 1] if hasattr(classifier, 'predict_proba') else None
                
                all_metrics[f"{clf_name}_f1"] = f1_score(y_test, y_pred, zero_division=0)
                all_metrics[f"{clf_name}_precision"] = precision_score(y_test, y_pred, zero_division=0)
                all_metrics[f"{clf_name}_recall"] = recall_score(y_test, y_pred, zero_division=0)
                
                if y_proba is not None:
                    try:
                        all_metrics[f"{clf_name}_auc"] = roc_auc_score(y_test, y_proba)
                    except ValueError:
                        all_metrics[f"{clf_name}_auc"] = 0.0
            except Exception as e:
                print(f"    {clf_name} failed: {e}")
                all_metrics[f"{clf_name}_f1"] = 0.0
                all_metrics[f"{clf_name}_auc"] = 0.0
        
        # Average metrics
        metrics = {
            'f1_score': np.mean([v for k, v in all_metrics.items() if 'f1' in k]),
            'roc_auc': np.mean([v for k, v in all_metrics.items() if 'auc' in k]),
            'precision': np.mean([v for k, v in all_metrics.items() if 'precision' in k]),
            'recall': np.mean([v for k, v in all_metrics.items() if 'recall' in k]),
        }
        
        return metrics
        
    except Exception as e:
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_sparse_minority_test(config_path: str):
    """
    Run extreme sparse setting test.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"\n{'='*60}")
    print(f"Extreme Sparse Setting Test")
    print(f"{'='*60}\n")
    
    device = config['training']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    output_dir = Path(config['output']['results_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # For each minority sample count
    for n_minority in config['minority_sample_counts']:
        print(f"\n{'='*60}")
        print(f"Minority Samples: {n_minority}")
        print(f"{'='*60}\n")
        
        # For each dataset
        for dataset_info in config['datasets']:
            dataset_name = dataset_info['name']
            print(f"\nDataset: {dataset_name}")
            
            try:
                # Load dataset
                X, y = load_dataset_from_file(dataset_info['path'])
                
                # Create sparse setting
                X_sparse, y_sparse = create_sparse_setting(X, y, n_minority)
                print(f"  Shape: {X_sparse.shape}, Minority: {np.sum(y_sparse == 1)}")
                
                # Run experiments with each method
                for method in config['comparison_methods']:
                    print(f"\n  Method: {method}")
                    
                    method_results = []
                    n_runs = config['evaluation']['n_runs']
                    n_splits = config['evaluation']['n_splits']
                    random_state = config['evaluation']['random_state']
                    
                    for run in range(n_runs):
                        skf = StratifiedKFold(
                            n_splits=n_splits,
                            random_state=random_state + run,
                            shuffle=True
                        )
                        
                        for fold, (train_idx, test_idx) in enumerate(skf.split(X_sparse, y_sparse)):
                            X_train, X_test = X_sparse[train_idx], X_sparse[test_idx]
                            y_train, y_test = y_sparse[train_idx], y_sparse[test_idx]
                            
                            # Skip if too few minority samples in train
                            if np.sum(y_train == 1) < 5:
                                continue
                            
                            # Standardize
                            scaler = StandardScaler()
                            X_train = scaler.fit_transform(X_train)
                            X_test = scaler.transform(X_test)
                            
                            # Run experiment
                            fold_seed = random_state + run * 1000 + fold * 100
                            metrics = run_sparse_experiment(
                                X_train, y_train, X_test, y_test,
                                config, method, device, fold_seed
                            )
                            
                            if metrics is not None:
                                metrics.update({
                                    'dataset': dataset_name,
                                    'n_minority': n_minority,
                                    'method': method,
                                    'run': run,
                                    'fold': fold
                                })
                                all_results.append(metrics)
                                method_results.append(metrics['f1_score'])
                    
                    if method_results:
                        print(f"    F1: {np.mean(method_results):.4f} ± {np.std(method_results):.4f}")
            
            except Exception as e:
                print(f"  Error processing {dataset_name}: {e}")
                continue
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_file = output_dir / f"{config['experiment_name']}_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description='Run sparse minority setting test')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    results = run_sparse_minority_test(args.config)
    
    print(f"\n{'='*60}")
    print("Sparse Setting Test Complete!")
    print(f"Total experiments: {len(results)}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

