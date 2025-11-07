"""
High-Dimensional Scalability Test for MOMS Model.

This script tests MOMS performance on high-dimensional datasets (500+, 1000+ dimensions)
to address reviewer concerns about scalability.
"""

import os
import sys
import yaml
import time
import tracemalloc
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score

# Add paths
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.training.moms_train import train_map
from src.models.moms_losses import MMD_est_torch
from src.utils.moms_utils import set_seed
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


def load_dataset_from_file(file_path: str):
    """Load dataset from file."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    # Load data
    data = pd.read_csv(file_path, header=None, sep=',')
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].values
    
    # Convert to binary labels
    unique_labels, counts = np.unique(y, return_counts=True)
    minority_label = unique_labels[np.argmin(counts)]
    y_binary = np.where(y == minority_label, 1, 0)
    
    return X.values, y_binary


def augment_dimensions(X, target_dim, method='random'):
    """
    Augment dataset to target dimensionality.
    
    Parameters
    ----------
    X : np.ndarray
        Original features.
    target_dim : int
        Target number of dimensions.
    method : str
        Method for augmentation ('random', 'polynomial').
        
    Returns
    -------
    X_augmented : np.ndarray
        Augmented feature matrix.
    """
    n_samples, n_features = X.shape
    
    if target_dim <= n_features:
        return X
    
    additional_dims = target_dim - n_features
    
    if method == 'random':
        # Add random noise features (scaled to match original variance)
        noise = np.random.randn(n_samples, additional_dims)
        noise = noise * np.std(X, axis=0).mean()
        X_augmented = np.hstack([X, noise])
        
    elif method == 'polynomial':
        # Add polynomial features and random if needed
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        if X_poly.shape[1] >= target_dim:
            X_augmented = X_poly[:, :target_dim]
        else:
            # Add random features to reach target_dim
            remaining = target_dim - X_poly.shape[1]
            noise = np.random.randn(n_samples, remaining)
            noise = noise * np.std(X_poly, axis=0).mean()
            X_augmented = np.hstack([X_poly, noise])
    else:
        raise ValueError(f"Unknown augmentation method: {method}")
    
    return X_augmented


def measure_performance(
    X_train, y_train, X_test, y_test,
    config, device, seed
):
    """
    Measure MOMS performance including time and memory.
    
    Returns
    -------
    results : dict
        Dictionary with performance metrics and measurements.
    """
    set_seed(seed)
    
    # Start memory tracking
    tracemalloc.start()
    start_memory = tracemalloc.get_traced_memory()[0]
    
    results = {
        'training_time': 0,
        'mmd_time': 0,
        'knn_time': 0,
        'prediction_time': 0,
        'memory_peak': 0,
    }
    
    try:
        # Separate classes
        X_maj = X_train[y_train == 0].astype(np.float32)
        X_min = X_train[y_train == 1].astype(np.float32)
        
        if len(X_min) < 5 or len(X_maj) < 5:
            tracemalloc.stop()
            return None
        
        # Train MOMS model
        model_config = config['model']
        fixed_params = config['fixed_params']
        
        start_time = time.time()
        
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
            k=fixed_params['danger_k'],
            seed=seed,
            residual=model_config.get('residual', True),
            median_bw=config['model'].get('median_bw', True),
            triplet_margin=fixed_params['triplet_margin'],
            triplet_alpha=fixed_params['triplet_alpha']
        )
        
        results['training_time'] = time.time() - start_time
        
        # Generate synthetic samples
        n_trans = len(X_maj) - len(X_min)
        if n_trans > 0:
            X_trans_init = torch.tensor(X_maj[:n_trans], dtype=torch.float32).to(device)
            
            gen_start = time.time()
            with torch.no_grad():
                _, X_trans = model(X_trans_init)
            X_synthetic = X_trans.cpu().numpy()
            results['prediction_time'] = time.time() - gen_start
            
            # Augment training data
            X_train_aug = np.vstack([X_train, X_synthetic])
            y_train_aug = np.hstack([y_train, np.ones(len(X_synthetic))])
        else:
            X_train_aug = X_train
            y_train_aug = y_train
        
        # Measure peak memory
        current, peak = tracemalloc.get_traced_memory()
        results['memory_peak'] = (peak - start_memory) / (1024 ** 2)  # MB
        
        # Train classifiers and evaluate
        classifier_metrics = {}
        classifiers = {
            'RF': RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=-1),
            'SVM': SVC(kernel='rbf', probability=True, random_state=seed),
            'MLP': MLPClassifier(
                hidden_layer_sizes=(100,), 
                max_iter=1000,
                early_stopping=True,
                n_iter_no_change=10,
                random_state=seed
            )
        }
        
        for clf_name, classifier in classifiers.items():
            try:
                classifier.fit(X_train_aug, y_train_aug)
                y_pred = classifier.predict(X_test)
                y_proba = classifier.predict_proba(X_test)[:, 1] if hasattr(classifier, 'predict_proba') else None
                
                classifier_metrics[f"{clf_name}_f1"] = f1_score(y_test, y_pred, zero_division=0)
                classifier_metrics[f"{clf_name}_bacc"] = balanced_accuracy_score(y_test, y_pred)
                
                if y_proba is not None:
                    try:
                        classifier_metrics[f"{clf_name}_auc"] = roc_auc_score(y_test, y_proba)
                    except ValueError:
                        classifier_metrics[f"{clf_name}_auc"] = 0.0
            except Exception as e:
                print(f"  Warning: {clf_name} failed: {e}")
                classifier_metrics[f"{clf_name}_f1"] = 0.0
                classifier_metrics[f"{clf_name}_bacc"] = 0.0
                classifier_metrics[f"{clf_name}_auc"] = 0.0
        
        # Average across classifiers
        results['f1_score'] = np.mean([v for k, v in classifier_metrics.items() if 'f1' in k])
        results['roc_auc'] = np.mean([v for k, v in classifier_metrics.items() if 'auc' in k])
        results['balanced_accuracy'] = np.mean([v for k, v in classifier_metrics.items() if 'bacc' in k])
        
    except Exception as e:
        print(f"Error in measurement: {e}")
        traceback.print_exc()
        tracemalloc.stop()
        return None
    
    tracemalloc.stop()
    return results


def run_high_dim_scalability_test(config_path: str):
    """
    Run high-dimensional scalability test.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file.
    """
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"\n{'='*60}")
    print(f"High-Dimensional Scalability Test")
    print(f"{'='*60}\n")
    
    device = config['training']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU")
    
    output_dir = Path(config['output']['results_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # Test each dimension
    for target_dim in config['dimensions']['target_dims']:
        print(f"\n{'='*60}")
        print(f"Testing Dimension: {target_dim}")
        print(f"{'='*60}\n")
        
        # Test each base dataset
        for dataset_info in config['base_datasets']:
            dataset_name = dataset_info['name']
            print(f"\nDataset: {dataset_name}")
            
            try:
                # Load dataset
                X, y = load_dataset_from_file(dataset_info['path'])
                print(f"  Original shape: {X.shape}")
                
                # Augment to target dimension
                X_aug = augment_dimensions(X, target_dim, method='random')
                print(f"  Augmented shape: {X_aug.shape}")
                
                # Run experiments
                n_runs = config['evaluation']['n_runs']
                n_splits = config['evaluation']['n_splits']
                random_state = config['evaluation']['random_state']
                
                for run in range(n_runs):
                    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state + run, shuffle=True)
                    
                    for fold, (train_idx, test_idx) in enumerate(skf.split(X_aug, y)):
                        X_train, X_test = X_aug[train_idx], X_aug[test_idx]
                        y_train, y_test = y[train_idx], y[test_idx]
                        
                        # Standardize
                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)
                        
                        # Run measurement
                        fold_seed = random_state + run * 1000 + fold * 100
                        results = measure_performance(
                            X_train, y_train, X_test, y_test,
                            config, device, fold_seed
                        )
                        
                        if results is not None:
                            results.update({
                                'dataset': dataset_name,
                                'dimension': target_dim,
                                'run': run,
                                'fold': fold
                            })
                            all_results.append(results)
                            
                            print(f"  Run {run+1}/{n_runs}, Fold {fold+1}/{n_splits}: "
                                  f"F1={results['f1_score']:.4f}, "
                                  f"Time={results['training_time']:.2f}s, "
                                  f"Memory={results['memory_peak']:.1f}MB")
            
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
    parser = argparse.ArgumentParser(description='Run high-dimensional scalability test')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    results = run_high_dim_scalability_test(args.config)
    
    print(f"\n{'='*60}")
    print("Scalability Test Complete!")
    print(f"Total experiments: {len(results)}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

