"""
Visualization Script for Ablation Study Results.

Generates publication-quality plots for hyperparameter sensitivity analysis.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


def load_results(results_dir: str, experiment_name: str) -> pd.DataFrame:
    """
    Load ablation study results from CSV files.
    
    Parameters
    ----------
    results_dir : str
        Directory containing result CSV files.
    experiment_name : str
        Name of the experiment (e.g., 'triplet_margin_ablation').
        
    Returns
    -------
    results_df : pd.DataFrame
        Combined results dataframe.
    """
    results_path = Path(results_dir) / f"{experiment_name}_results.csv"
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    results_df = pd.read_csv(results_path)
    return results_df


def plot_parameter_sensitivity(
    results_df: pd.DataFrame,
    parameter_name: str,
    metrics: List[str] = None,
    output_path: str = None,
    figsize: tuple = (12, 8)
):
    """
    Create line plots showing parameter sensitivity for each metric.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe with columns: dataset, parameter_value, metric, mean, std
    parameter_name : str
        Name of the parameter being tested.
    metrics : List[str], optional
        List of metrics to plot. If None, plots all available metrics.
    output_path : str, optional
        Path to save the figure.
    figsize : tuple
        Figure size (width, height).
    """
    if metrics is None:
        metrics = results_df['metric'].unique().tolist()
    
    # Filter to primary metrics if specified
    primary_metrics = ['f1_score', 'roc_auc', 'g_mean']
    metrics_to_plot = [m for m in primary_metrics if m in metrics]
    
    n_metrics = len(metrics_to_plot)
    n_datasets = results_df['dataset'].nunique()
    
    # Create subplots: one row per metric, one column per dataset
    fig, axes = plt.subplots(
        n_metrics, n_datasets,
        figsize=(figsize[0] * n_datasets, figsize[1] * n_metrics),
        sharex=True, sharey='row'
    )
    
    if n_metrics == 1:
        axes = axes.reshape(1, -1)
    if n_datasets == 1:
        axes = axes.reshape(-1, 1)
    
    datasets = sorted(results_df['dataset'].unique())
    
    for metric_idx, metric in enumerate(metrics_to_plot):
        for dataset_idx, dataset in enumerate(datasets):
            ax = axes[metric_idx, dataset_idx]
            
            # Filter data
            metric_data = results_df[
                (results_df['dataset'] == dataset) &
                (results_df['metric'] == metric)
            ].sort_values('parameter_value')
            
            if len(metric_data) > 0:
                x = metric_data['parameter_value'].values
                y = metric_data['mean'].values
                yerr = metric_data['std'].values
                
                # Plot with error bars
                ax.errorbar(x, y, yerr=yerr, marker='o', capsize=5, capthick=2,
                           linewidth=2, markersize=8, label=metric)
                
                # Fill area for standard deviation
                ax.fill_between(x, y - yerr, y + yerr, alpha=0.2)
            
            # Labels and formatting
            if metric_idx == 0:
                ax.set_title(dataset, fontsize=10, fontweight='bold')
            if dataset_idx == 0:
                metric_label = metric.replace('_', ' ').title()
                ax.set_ylabel(f'{metric_label}', fontsize=10, fontweight='bold')
            
            if metric_idx == n_metrics - 1:
                param_label = parameter_name.replace('_', ' ').title()
                ax.set_xlabel(f'{param_label}', fontsize=10)
            
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)
    
    plt.suptitle(
        f'Hyperparameter Sensitivity: {parameter_name.replace("_", " ").title()}',
        fontsize=14, fontweight='bold', y=0.995
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_aggregated_sensitivity(
    results_df: pd.DataFrame,
    parameter_name: str,
    output_path: str = None,
    figsize: tuple = (10, 6)
):
    """
    Create aggregated plot showing average performance across all datasets.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe.
    parameter_name : str
        Name of the parameter.
    output_path : str, optional
        Path to save the figure.
    figsize : tuple
        Figure size.
    """
    # Aggregate across datasets
    aggregated = results_df.groupby(['parameter_value', 'metric']).agg({
        'mean': 'mean',
        'std': lambda x: np.sqrt(np.mean(x**2))  # RMS of std
    }).reset_index()
    
    # Focus on primary metrics
    primary_metrics = ['f1_score', 'roc_auc', 'g_mean']
    aggregated = aggregated[aggregated['metric'].isin(primary_metrics)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for metric in primary_metrics:
        metric_data = aggregated[
            aggregated['metric'] == metric
        ].sort_values('parameter_value')
        
        if len(metric_data) > 0:
            x = metric_data['parameter_value'].values
            y = metric_data['mean'].values
            yerr = metric_data['std'].values
            
            label = metric.replace('_', ' ').title()
            ax.errorbar(x, y, yerr=yerr, marker='o', capsize=5, capthick=2,
                       linewidth=2, markersize=8, label=label)
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.2)
    
    param_label = parameter_name.replace('_', ' ').title()
    ax.set_xlabel(f'{param_label}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance Score', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Aggregated Sensitivity: {param_label}',
        fontsize=14, fontweight='bold'
    )
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def create_summary_table(
    results_df: pd.DataFrame,
    parameter_name: str,
    output_path: str = None
) -> pd.DataFrame:
    """
    Create summary table with optimal values and robust ranges.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe.
    parameter_name : str
        Name of the parameter.
    output_path : str, optional
        Path to save LaTeX table.
        
    Returns
    -------
    summary_df : pd.DataFrame
        Summary dataframe.
    """
    # Focus on F1-score as primary metric
    f1_data = results_df[results_df['metric'] == 'f1_score'].copy()
    
    summary_rows = []
    
    for dataset in f1_data['dataset'].unique():
        dataset_data = f1_data[f1_data['dataset'] == dataset].sort_values('parameter_value')
        
        if len(dataset_data) == 0:
            continue
        
        # Find optimal value
        optimal_idx = dataset_data['mean'].idxmax()
        optimal_value = dataset_data.loc[optimal_idx, 'parameter_value']
        optimal_score = dataset_data.loc[optimal_idx, 'mean']
        
        # Find robust range (within 5% of optimal)
        threshold = optimal_score * 0.95
        robust_mask = dataset_data['mean'] >= threshold
        robust_values = dataset_data[robust_mask]['parameter_value'].values
        
        if len(robust_values) > 0:
            robust_min = robust_values.min()
            robust_max = robust_values.max()
        else:
            robust_min = robust_max = optimal_value
        
        summary_rows.append({
            'dataset': dataset,
            'optimal_value': optimal_value,
            'optimal_f1': optimal_score,
            'robust_min': robust_min,
            'robust_max': robust_max,
            'robust_range': f"[{robust_min:.2f}, {robust_max:.2f}]"
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    if output_path:
        # Save as CSV
        csv_path = Path(output_path).with_suffix('.csv')
        summary_df.to_csv(csv_path, index=False)
        
        # Save as LaTeX
        latex_path = Path(output_path).with_suffix('.tex')
        with open(latex_path, 'w') as f:
            f.write(summary_df.to_latex(index=False, float_format="%.4f"))
        
        print(f"Summary table saved to: {csv_path}")
        print(f"LaTeX table saved to: {latex_path}")
    
    return summary_df


def main():
    parser = argparse.ArgumentParser(
        description='Visualize ablation study results'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        required=True,
        help='Directory containing result CSV files'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        required=True,
        help='Name of the experiment (e.g., triplet_margin_ablation)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for figures'
    )
    parser.add_argument(
        '--parameter-name',
        type=str,
        default=None,
        help='Parameter name (auto-detected if not provided)'
    )
    
    args = parser.parse_args()
    
    # Load results
    results_df = load_results(args.results_dir, args.experiment_name)
    
    # Detect parameter name if not provided
    if args.parameter_name is None:
        args.parameter_name = results_df['parameter'].iloc[0]
    
    # Set output directory
    if args.output_dir is None:
        output_dir = Path(args.results_dir).parent.parent / 'figures' / 'ablation_study'
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    print(f"\nGenerating visualizations for {args.experiment_name}...")
    
    # 1. Detailed sensitivity plots
    detailed_path = output_dir / f"{args.experiment_name}_detailed.png"
    plot_parameter_sensitivity(
        results_df, args.parameter_name,
        output_path=str(detailed_path)
    )
    
    # 2. Aggregated sensitivity plot
    aggregated_path = output_dir / f"{args.experiment_name}_aggregated.png"
    plot_aggregated_sensitivity(
        results_df, args.parameter_name,
        output_path=str(aggregated_path)
    )
    
    # 3. Summary table
    summary_path = output_dir / f"{args.experiment_name}_summary"
    summary_df = create_summary_table(
        results_df, args.parameter_name,
        output_path=str(summary_path)
    )
    
    print(f"\nVisualization complete!")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()

