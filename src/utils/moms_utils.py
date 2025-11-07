import pandas as pd
import numpy as np
import torch 
import os

def set_seed(seed=1203):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_results(final_results, data_name, kernel_type, beta, save_path="./results"):
    """
    Save final evaluation results into a CSV file.

    Parameters:
    - final_results (dict): Dictionary containing metric results for each method-classifier pair.
    - data_name (str): Name of the dataset.
    - beta (float): Regularization coefficient (used in filename).
    - save_path (str): Directory where the CSV file will be saved.
    """
    res_data = {
        "Classifier": [],
        "Method": [],
        "Metric": [],
        "Value": [],
    }

    for key, metrics in final_results.items():
        # Split key into oversampling method and classifier if available
        if " - " in key:
            method_name, clf_name = key.split(" - ", 1)
        else:
            method_name, clf_name = key, "Default"
        
        for metric, values in metrics.items():
            avg_val = np.mean(values) if values else "N/A"
            res_data["Classifier"].append(clf_name)
            res_data["Method"].append(method_name)
            res_data["Metric"].append(metric)
            res_data["Value"].append(avg_val)

    res_df = pd.DataFrame(res_data)

    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f"{data_name}_results_{kernel_type}_beta_{beta}.csv")
    res_df.to_csv(save_file, index=False)
    print(f"\n[Saved] Final results are saved to {save_file}")