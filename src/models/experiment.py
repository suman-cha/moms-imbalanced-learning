import os
import sys
sys.path.append("/home/oldrain123/IMBALANCED_CLASSIFICATION/MOMs")
sys.path.append("/home/oldrain123/IMBALANCED_CLASSIFICATION/boost")
sys.path.append('/home/oldrain123/IMBALANCED_CLASSIFICATION/')
sys.path.append('/home/oldrain123/IMBALANCED_CLASSIFICATION/SMOTE_variants/')

import time
from typing import Any
import torch
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from ctgan import CTGAN
from mgvae import MGVAE
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler
from SMOTE_variants.sm_variants.oversampling.mwmote import MWMOTE
from gamosampler import GAMOtabularSampler
from osman.oversampler import VAEify, WGANify
from custom_packages.boost import AdaBoostClassifier, SMOTEBoost, RUSBoost, OUBoost
from sklearn.tree import DecisionTreeClassifier

from src.utils.moms_utils import set_seed, save_results
from src.models.moms_losses import MMD_est_torch
from src.models.moms_metrics import Metrics
from src.models.moms_generate import transform
from src.utils.moms_visualize import plot_tsne


np.bool = np.bool_

def load_dataset(data, data_name=None, maj_target_name=None):
    """
    Handles dataset loading and label transformation based on input format.

    Parameters:
    - data: Can be either:
        - A dictionary containing datasets (`fetch_datasets()[data_name]` format).
        - A Pandas DataFrame containing features and the target variable.
    - data_name (str, optional): The dataset key when using `fetch_datasets()`.
    - maj_target_name (str, optional): The majority class name (only required for format [A]).

    Returns:
    - X (numpy.ndarray): Feature matrix.
    - y (numpy.ndarray): Target vector (binary labels: 0 for majority, 1 for minority).
    """
    if isinstance(data, dict) and data_name is not None:
        # Format [A]: Dictionary-based dataset (e.g., fetch_datasets()[data_name])
        dataset = data[data_name]
        X, y = pd.DataFrame(dataset.data), dataset.target
        print(f"Dataset loaded: {data_name}")
    elif isinstance(data, pd.DataFrame):
        # Format [B]: DataFrame format (features + last column as target)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1].values
        print("Dataset loaded from DataFrame")
    else:
        raise ValueError("Invalid dataset format. Expected a dictionary with data_name or a Pandas DataFrame.")

    # Convert labels to binary format
    if maj_target_name is not None:
        y = np.where(y == maj_target_name, 0, 1)  # Format [A]
    else:
        y = np.where(y == 'negative', 0, 1)  # Format [B]
    return X, y

def run_exp(
    data, cat_idx, methods, base_model, device, 
    n_epochs, hidden_dims, latent_dim, lr, beta, 
    seed, data_name, maj_target_name=None, 
    n_runs=10, n_splits=10, visualize=False, save_path=None,
    kernel_type='gaussian', loss_params=None
):
    """
    Runs the experiment pipeline, including data preprocessing, model training, and evaluation.
    
    Args:
        data: Raw dataset.
        cat_idx: List of categorical feature indices.
        methods: List of methods to evaluate.
        base_model: The classifier model(s) to evaluate. 
                    Can be a single model or a dict of models (e.g., {"DecisionTree": ..., "kNN": ...}).
        device: 'cpu' or 'cuda' for model training.
        n_epochs: Number of training epochs for TransMap.
        hidden_dims: List of hidden layer sizes for TransMap.
        latent_dim: Latent dimension for TransMap.
        lr: Learning rate.
        beta: Regularization coefficient.
        seed: Random seed for reproducibility.
        data_name: Name of the dataset.
        maj_target_name: Majority class target name (if applicable).
        n_runs: Number of experimental runs.
        n_splits: Number of folds for cross-validation.
        visualize: Whether to generate t-SNE visualization.
        save_path: Path to save experiment results.
        kernel_type: kernel type to calculate MMD (default: gaussian)
        loss_params: additional parameters for MMD loss (e.g. {'h': 1.0, 'alpha':1.0})

    Returns:
        final_res: Dictionary containing performance metrics for each method (and classifier).
    """
    X, y = load_dataset(data, data_name, maj_target_name)
    n_maj = np.sum(y == 0)
    n_min = np.sum(y == 1)
    ir = np.round(n_maj / n_min, 2) if n_min > 0 else np.inf

    print(f"X shape : {X.shape}")
    print(f"target : {Counter(y)}")
    print(f"IR: {ir}")
    
    final_res = {}

    for run in range(n_runs):
        skf = StratifiedKFold(n_splits=n_splits, random_state=seed+run, shuffle=True)
        print(f"\nStarting experiment {run+1}/{n_runs}...")
        res = {}
        
        for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
            # print(f"    [EXP {run+1}/{n_runs}] Fold {fold+1}/{n_splits}")
            f_seed = seed + fold + 10 * run
            set_seed(f_seed)

            X_tr, X_te = X.iloc[tr_idx, :], X.iloc[te_idx, :]
            y_tr, y_te = y[tr_idx], y[te_idx]

            supported_methods = {"Original", "Boost", "SMOTE", "ADASYN", "bSMOTE", 
                        "ROS", "MWMOTE", "CTGAN", "VAE", "WGAN", "GAMO", "MGVAE", "Ours"}
            selected_methods = supported_methods if methods is None else set(methods).intersection(supported_methods)

            # -----------------------------
            # Use raw data for CTGAN
            # -----------------------------
            X_tr_raw = X_tr.copy()  
            X_maj_raw = X_tr_raw[y_tr == 0]
            X_min_raw = X_tr_raw[y_tr == 1]

            column_names = [f"col_{i}" for i in range(X_min_raw.shape[1])]
            X_min_raw.columns = column_names
            ctgan = CTGAN(epochs=100)
            cat_features_ctgan = [column_names[i] for i in cat_idx] if len(cat_idx) > 0 else []
            ctgan.fit(X_min_raw, discrete_columns=cat_features_ctgan)
            
            # Generate synthetic minority samples so that the augmented set has balanced majority/minority
            n_trans = len(X_maj_raw) - len(X_min_raw)
            X_ctgan = ctgan.sample(n=n_trans)
            
            # -----------------------------
            # Create one-hot encoded data for transformation
            # -----------------------------
            X_tr_arr = X_tr.values  
            X_te_arr = X_te.values
            if len(cat_idx) > 0:
                encoder = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
                cat_data = encoder.fit_transform(X_tr_arr[:, cat_idx])
                cat_data_te = encoder.transform(X_te_arr[:, cat_idx])
                num_idx = [i for i in range(X_tr_arr.shape[1]) if i not in cat_idx]
                num_data = X_tr_arr[:, num_idx]
                num_data_te = X_te_arr[:, num_idx]
                num_scaler = StandardScaler()
                num_data = num_scaler.fit_transform(num_data)
                num_data_te = num_scaler.transform(num_data_te)
                X_tr_enc = np.hstack((num_data, cat_data))
                X_te_enc = np.hstack((num_data_te, cat_data_te))
            else:
                X_tr_enc = X_tr_arr.copy()
                scaler = StandardScaler()
                X_tr_enc = scaler.fit_transform(X_tr_enc)
                X_te_enc = scaler.transform(X_te_arr)

            ctgan_arr = X_ctgan.values  # CTGAN : DataFrame -> Numpy array

            if len(cat_idx) > 0:
                cat_data_ctgan = encoder.transform(ctgan_arr[:, cat_idx])
                num_data_ctgan = num_scaler.transform(ctgan_arr[:, [i for i in range(ctgan_arr.shape[1]) if i not in cat_idx]])
                X_ctgan_enc = np.hstack((num_data_ctgan, cat_data_ctgan))
            else:
                X_ctgan_enc = scaler.transform(ctgan_arr)
            
            X_maj_enc = X_tr_enc[y_tr == 0]
            X_min_enc = X_tr_enc[y_tr == 1]
            in_dim = X_tr_enc.shape[1]
   
            datasets = {}
            if "Original" in selected_methods:
                datasets["Original"] = (X_tr_enc, y_tr)
            if "Boost" in selected_methods:
                datasets["Boost"] = (X_tr_enc, y_tr)
            if "SMOTE" in selected_methods:
                datasets["SMOTE"] = SMOTE(random_state=seed).fit_resample(X_tr_enc, y_tr)
            if "ADASYN" in selected_methods:
                datasets["ADASYN"] = ADASYN(random_state=seed).fit_resample(X_tr_enc, y_tr)
            if "bSMOTE" in selected_methods:
                datasets["bSMOTE"] = BorderlineSMOTE(random_state=seed).fit_resample(X_tr_enc, y_tr)
            if "ROS" in selected_methods:
                datasets["ROS"] = RandomOverSampler(random_state=seed).fit_resample(X_tr_enc, y_tr)
            if "MWMOTE" in selected_methods:
                datasets["MWMOTE"] = MWMOTE(random_state=seed).sample(X_tr_enc, y_tr)
            if "CTGAN" in selected_methods:
                datasets["CTGAN"] = (np.vstack((X_maj_enc, X_min_enc, X_ctgan_enc)), 
                                     np.hstack((np.zeros(len(X_maj_enc)), np.ones(len(X_min_enc) + len(X_ctgan_enc)))))
            if "VAE" in selected_methods:
                datasets['VAE'] = VAEify(pd.DataFrame(X_tr_enc), pd.Series(y_tr.flatten()), n_epochs, seed, device)
            if "WGAN" in selected_methods:
                datasets['WGAN'] = WGANify(pd.DataFrame(X_tr_enc), pd.Series(y_tr.flatten()), n_epochs=n_epochs)
            if "GAMO" in selected_methods:
                all_minority_X = {0: X_maj_enc, 1: X_min_enc}
                gamo = GAMOtabularSampler(input_dim = X_min_enc.shape[1], all_minority_X = all_minority_X, class_counts=[len(X_maj_enc), len(X_min_enc)], latent_dim=X_maj_enc.shape[1], hidden_dim=X_maj_enc.shape[1]*2, device=device)
                class_X_dict = {0: X_maj_enc, 1: X_min_enc}
                gamo.fit(class_X_dict=class_X_dict, n_epochs=n_epochs, batch_size=len(X_min_enc), lr=lr)
                X_gamo = gamo.sample(n_samples=n_trans, class_id=1)
                datasets["GAMO"] = (np.vstack((X_maj_enc, X_min_enc, X_gamo)),
                    np.hstack((np.zeros(len(X_maj_enc)), np.ones(len(X_min_enc) + len(X_gamo)))))
            if "MGVAE" in selected_methods:
                X_maj_tensor = torch.tensor(X_maj_enc, dtype=torch.float32).to(device)
                X_min_tensor = torch.tensor(X_min_enc, dtype=torch.float32).to(device)
                
                latent_dim = X_maj_enc.shape[1]  # Adapt as needed
                hidden_dims = [X_maj_enc.shape[1] * 2, X_maj_enc.shape[1] * 4, X_maj_enc.shape[1] * 8]
                
                mgvae = MGVAE(input_dim=X_maj_enc.shape[1],
                            latent_dim=latent_dim,
                            hidden_dims=hidden_dims,
                            device=device,
                            majority_subsample=64)  # Choose subsample size per your memory capacity
                
                # Pretraining on majority samples
                mgvae.pretrain(X_maj_tensor, epochs=n_epochs)
                
                # Store pre-trained parameters for EWC
                pretrain_params = {n: p.clone().detach() for n, p in mgvae.named_parameters()}
                
                # Compute Fisher information on majority data (used for EWC)
                fisher_info = mgvae.compute_fisher(X_maj_tensor)
                
                # Fine-tune on minority data with EWC regularization enabled (set lambda>0 to activate)
                mgvae.finetune(X_min_tensor, X_maj_tensor, fisher_info, pretrain_params,
                            epochs=n_epochs, ewc_lambda=500)  # Adjust ewc_lambda based on experiments
                
                # Generate synthetic minority samples using majority-based prior mixture
                X_mgvae = mgvae.sample(X_maj_tensor, n_samples=n_trans)
                
                # Compose augmented dataset: majority + original minority + generated minority
                datasets["MGVAE"] = (np.vstack((X_maj_enc, X_min_enc, X_mgvae)),
                                    np.hstack((np.zeros(len(X_maj_enc)), 
                                                np.ones(len(X_min_enc) + len(X_mgvae)))))

            if "Ours" in selected_methods:
                # -----------------------------
                # Apply transformation using the one-hot encoded data
                # -----------------------------
                X_maj_T, X_min_T, X_trans = transform(
                    X_maj=X_maj_enc,
                    X_min=X_min_enc,
                    in_dim=in_dim,
                    hidden_dims=hidden_dims,
                    latent_dim=latent_dim,
                    loss_fn=MMD_est_torch,
                    kernel_type=kernel_type,
                    loss_params=loss_params,
                    device=device,
                    method="direct",
                    n_epochs=n_epochs,
                    lr=lr,
                    beta=beta,
                    seed=f_seed
                )
                datasets["Ours"] = (np.vstack((X_maj_enc, X_min_enc, X_trans)),
                                    np.hstack((np.zeros(len(X_maj_enc)), np.ones(len(X_min_enc) + len(X_trans)))))

            for method, (X_bal, y_bal) in datasets.items():
                if method == "Boost":
                    # Boosting methods
                    # AdaBoost
                    model_ada = AdaBoostClassifier(
                        DecisionTreeClassifier(max_depth=5),
                        n_estimators=100,
                        algorithm="SAMME",
                        learning_rate=0.1,
                        random_state=f_seed,
                    )
                    model_ada.fit(X_bal, y_bal)
                    pred_ada = model_ada.predict(X_te_enc)
                    proba_ada = model_ada.predict_proba(X_te_enc)[:, 1]
                    metrics_ada = Metrics(y_te, pred_ada, proba_ada)
                    if "AdaBoost" not in res:
                        res["AdaBoost"] = {"AUROC": [], "G-mean": [], "MCC": [], "F1-score": [], "mAP": []}
                    for metric_name, metric_value in metrics_ada.all_metrics().items():
                        res["AdaBoost"][metric_name].append(np.round(metric_value, 4))
                    
                    # SMOTEBoost
                    smote_boost = SMOTEBoost(
                        learning_rate=0.1, n_samples=5, n_estimators=100, random_state=f_seed
                    )
                    smote_boost.fit(X_bal, y_bal)
                    pred_smb = smote_boost.predict(X_te_enc)
                    proba_smb = smote_boost.predict_proba(X_te_enc)[:, 1]
                    metrics_smb = Metrics(y_te, pred_smb, proba_smb)
                    if "SMOTEBoost" not in res:
                        res["SMOTEBoost"] = {"AUROC": [], "G-mean": [], "MCC": [], "F1-score": [], "mAP": []}

                    for metric_name, metric_value in metrics_smb.all_metrics().items():
                        res["SMOTEBoost"][metric_name].append(np.round(metric_value, 4))
                    
                    # RUSBoost
                    rus_boost = RUSBoost(
                        learning_rate=0.1, n_samples=5, n_estimators=100, random_state=f_seed
                    )
                    rus_boost.fit(X_bal, y_bal)
                    pred_rus = rus_boost.predict(X_te_enc)
                    proba_rus = rus_boost.predict_proba(X_te_enc)[:, 1]
                    metrics_rus = Metrics(y_te, pred_rus, proba_rus)
                    if "RUSBoost" not in res:
                        res["RUSBoost"] = {"AUROC": [], "G-mean": [], "MCC": [], "F1-score": [], "mAP": []}
                    for metric_name, metric_value in metrics_rus.all_metrics().items():
                        res["RUSBoost"][metric_name].append(np.round(metric_value, 4))
                    
                    # OUBoost
                    ou_boost = OUBoost(
                        learning_rate=0.1, n_samples=5, n_estimators=100, random_state=f_seed
                    )
                    ou_boost.fit(X_bal, y_bal)
                    pred_oub = ou_boost.predict(X_te_enc)
                    proba_oub = ou_boost.predict_proba(X_te_enc)[:, 1]
                    metrics_oub = Metrics(y_te, pred_oub, proba_oub)
                    if "OUBoost" not in res:
                        res["OUBoost"] = {"AUROC": [], "G-mean": [], "MCC": [], "F1-score": [], "mAP": []}
                    for metric_name, metric_value in metrics_oub.all_metrics().items():
                        res["OUBoost"][metric_name].append(np.round(metric_value, 4))
                else:
                    # For oversampling methods other than Boost:
                    # Check if base_model is a dict (i.e., multiple classifiers) or a single model.
                    if isinstance(base_model, dict):
                        for clf_name, clf in base_model.items():
                            new_key = f"{method} - {clf_name}"
                            clf.fit(X_bal, y_bal)
                            y_pred = clf.predict(X_te_enc)
                            y_proba = clf.predict_proba(X_te_enc)[:, 1]
                            metrics_os = Metrics(y_te, y_pred, y_proba)
                            if new_key not in res:
                                res[new_key] = {"AUROC": [], "G-mean": [], "MCC": [], "F1-score": [], "mAP": []}
                            for metric_name, metric_value in metrics_os.all_metrics().items():
                                res[new_key][metric_name].append(np.round(metric_value, 4))
                    else:
                        # Single classifier case
                        base_model.fit(X_bal, y_bal)
                        y_pred = base_model.predict(X_te_enc)
                        y_proba = base_model.predict_proba(X_te_enc)[:, 1]
                        metrics_os = Metrics(y_te, y_pred, y_proba)
                        if method not in res:
                            res[method] = {"AUROC": [], "G-mean": [], "MCC": [], "F1-score": [], "mAP": []}
                        for metric_name, metric_value in metrics_os.all_metrics().items():
                            res[method][metric_name].append(np.round(metric_value, 4))
            
            # print(f"    [Fold {fold+1}] Results")
            # for mth, metrics in res.items():
            #     print(f"      [{mth}] AUROC = {metrics['AUROC'][-1]:.4f}  |  "
            #           f"G-mean = {metrics['G-mean'][-1]:.4f}  |  "
            #           f"MCC = {metrics['MCC'][-1]:.4f}  |  "
            #           f"F1-score = {metrics['F1-score'][-1]:.4f}  |  "
            #           f"mAP = {metrics['mAP'][-1]:.4f}")
            
            # Visualization (t-SNE)
            if visualize:
                from sklearn.manifold import TSNE
                import matplotlib.pyplot as plt
                methods_plot = ['SMOTE', 'CTGAN', 'GAMO', 'MGVAE', 'Ours']
                synth = {}
                if 'SMOTE' in datasets:
                    X_res, y_res = datasets['SMOTE']
                    start = len(X_tr_enc)
                    X_new = X_res[start:][y_res[start]==1]
                    synth['SMOTE'] = X_new 
                if 'CTGAN' in datasets:
                    synth['CTGAN'] = X_ctgan_enc
                if 'GAMO' in datasets:
                    synth['GAMO'] = X_gamo
                if 'MGVAE' in datasets:
                    synth['MGVAE'] = X_mgvae
                if 'Ours' in datasets:
                    synth['Ours'] = X_trans
                
                # 3) TSNE subplot 생성
                n = len(methods_plot)
                fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
                for i, m in enumerate(methods_plot):
                    X_T = synth[m]
                    X_T = np.squeeze(X_T)
                    # TSNE embedding
                    tsne = TSNE(
                        n_components=2, perplexity=30,
                        early_exaggeration=4, random_state=seed, init="pca"
                    )
                    X_comb = np.vstack((X_tr_enc, X_T))
                    y_comb = np.hstack((y_tr, np.full(len(X_T), 2)))
                    X_emb = tsne.fit_transform(X_comb)

                    ax = axes[i]
                    p0 = ax.scatter(
                        X_emb[y_comb==0, 0], X_emb[y_comb==0, 1],
                        label='Majority', alpha=0.5, s=30, edgecolor='k', linewidth=0.3
                    )
                    p1 = ax.scatter(
                        X_emb[y_comb==1, 0], X_emb[y_comb==1, 1],
                        label='Minority', alpha=0.5, s=30, edgecolor='k', linewidth=0.3
                    )
                    p2 = ax.scatter(
                        X_emb[y_comb==2, 0], X_emb[y_comb==2, 1],
                        label='Synthetic', marker='x', s=50, c='red', linewidth=1.0
                    )
                    ax.set_title(m, fontsize=12)
                    ax.set_xticks([]); ax.set_yticks([])
                    # ax.legend(loc='lower center', fontsize=8, frameon=True, ncol=3)
                    ax.grid(False)

                plt.tight_layout()

                fig.legend(
                    handles=[p0, p1, p2],
                    labels=['Majority', 'Minority', 'Synthetic'],
                    loc='lower center',
                    ncol=3,
                    frameon=True,
                    fontsize=10,
                    bbox_to_anchor=(0.5, -0.02)
                )

                # 4) PDF로 저장
                if save_path:
                    os.makedirs(save_path, exist_ok=True)
                    pdf_path = os.path.join(save_path, f"{data_name}_tsne_comparison.pdf")
                    fig.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
                    print(f"[Saved] t-SNE comparison figure saved to: {pdf_path}")

                plt.show()
                
        for mth, metrics in res.items():
            if mth not in final_res:
                final_res[mth] = {"AUROC": [], "G-mean": [], "MCC": [], "F1-score": [], "mAP": []}
            for metric, values in metrics.items():
                final_res[mth][metric].append(np.round(np.mean(values), 4))
        
        print(f"\n[Aggregated Results after {run+1} runs]")
        for mth, metrics in final_res.items():
            print(f"  [{mth}]")
            for metric, values in metrics.items():
                avg_value = np.mean(values)
                print(f"    {metric}: {avg_value:.4f}")
            
    # Print final averaged results
    # print("\n[Final Results]")
    # for mth, metrics in final_res.items():
    #     print(f"    [{mth}]")
    #     for metric, values in metrics.items():
    #         print(f"    [{metric}]: {np.mean(values):.4f}")
    
    if save_path is not None:
        save_results(final_res, data_name, kernel_type, beta, save_path)
    
    return final_res

def run_ablation(
    data_name,
    beta,
    methods,
    base_models,
    device,
    save_path,
    data_path=None,     
    fetch_data_func=None,  
    maj_target_name=None,
    visualize=False,
    cat_idx = [],
    n_epochs=2000,
    hidden_dims=[16, 32, 64, 128],
    latent_dim=256,
    lr=0.001,
    seed=1203,
    kernel_type='gaussian',
    loss_params=None
):
    if fetch_data_func is not None:
        data = fetch_data_func()
    elif data_path is not None:
        data = pd.read_csv(f"{data_path}/{data_name}.dat", header=None)
    else:
        raise ValueError("Either `data_path` or `fetch_data_func` must be provided.")

    exp_kwargs = {
        "data": data,
        "cat_idx": cat_idx,
        "methods": methods,
        "base_model": base_models,
        "device": device,
        "n_epochs": n_epochs,
        "hidden_dims": hidden_dims,
        "latent_dim": latent_dim,
        "lr": lr,
        "beta": beta,
        "data_name": data_name,
        "seed": seed,
        "visualize": visualize,
        "save_path": save_path,
        "kernel_type": kernel_type,
        "loss_params": loss_params
    }

    if maj_target_name is not None:
        exp_kwargs["maj_target_name"] = maj_target_name

    # Run experiment
    res = run_exp(**exp_kwargs)
    return res

def run_time(
        method_name: str,
        X: np.ndarray, 
        y: np.ndarray,
        sampler: Any = None,
        device: str = 'cpu',
        seed: int = 1203,
        ctgan_epochs: int = 100, 
        transform_epochs: int = 100, 
        n_runs: int = 10,
) -> float:
    """
    Measure the average runtime for a given oversampling method.
    """
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        if method_name == 'CTGAN':
            X_min = pd.DataFrame(X[y == 1])
            ctgan = CTGAN(epochs=ctgan_epochs)
            ctgan.fit(X_min)
            n_samples = (y == 0).sum() - (y == 1).sum()
            ctgan.sample(n=n_samples)
        elif method_name == 'Ours':
            X_maj = X[y == 0]
            X_min = X[y == 1]
            transform(
                X_maj=X_maj,
                X_min=X_min,
                in_dim=X.shape[1],
                hidden_dims = [X.shape[1] * 2, X.shape[1] * 4],
                latent_dim = X.shape[1] * 8,
                loss_fn = MMD_est_torch,
                kernel_type = 'gaussian',
                loss_params={},
                device=device,
                method='direct',
                n_epochs=transform_epochs,
                lr=0.001,
                beta=0.01,
                seed=seed,
                residual=True,
            )
        else:
            _ = sampler.fit_resample(X, y) 
        end = time.perf_counter()
        times.append(end - start) 
    return float(np.mean(times))