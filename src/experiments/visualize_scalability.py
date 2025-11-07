"""
Visualization Scripts for Scalability Tests.

Generates publication-quality figures for:
1. High-dimensional scalability analysis
2. Extreme sparse setting comparison
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


def plot_dimension_vs_performance(results_df, output_path=None):
    """
    Plot dimension vs performance (F1, AUC).
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results with columns: dimension, f1_score, roc_auc, etc.
    output_path : str, optional
        Path to save figure.
    """
    # Aggregate by dimension
    agg_results = results_df.groupby('dimension').agg({
        'f1_score': ['mean', 'std'],
        'roc_auc': ['mean', 'std'],
        'balanced_accuracy': ['mean', 'std']
    }).reset_index()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    metrics = [
        ('f1_score', 'F1-Score', axes[0]),
        ('roc_auc', 'ROC-AUC', axes[1]),
        ('balanced_accuracy', 'Balanced Accuracy', axes[2])
    ]
    
    for metric_name, metric_label, ax in metrics:
        x = agg_results['dimension'].values
        y = agg_results[(metric_name, 'mean')].values
        yerr = agg_results[(metric_name, 'std')].values
        
        ax.errorbar(x, y, yerr=yerr, marker='o', capsize=5, capthick=2,
                   linewidth=2, markersize=8)
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.2)
        
        ax.set_xlabel('Dimension', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_label, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_label} vs Dimension', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
    
    plt.suptitle('MOMS Performance on High-Dimensional Data', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_dimension_vs_runtime(results_df, output_path=None):
    """
    Plot dimension vs runtime (log scale).
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results with columns: dimension, training_time, etc.
    output_path : str, optional
        Path to save figure.
    """
    agg_results = results_df.groupby('dimension').agg({
        'training_time': ['mean', 'std'],
        'prediction_time': ['mean', 'std']
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = agg_results['dimension'].values
    y_train = agg_results[('training_time', 'mean')].values
    yerr_train = agg_results[('training_time', 'std')].values
    
    y_pred = agg_results[('prediction_time', 'mean')].values
    yerr_pred = agg_results[('prediction_time', 'std')].values
    
    ax.errorbar(x, y_train, yerr=yerr_train, marker='o', capsize=5,
               label='Training Time', linewidth=2, markersize=8)
    ax.errorbar(x, y_pred, yerr=yerr_pred, marker='s', capsize=5,
               label='Prediction Time', linewidth=2, markersize=8)
    
    ax.set_xlabel('Dimension', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Runtime vs Dimension', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_dimension_vs_memory(results_df, output_path=None):
    """
    Plot dimension vs memory usage.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results with column: memory_peak
    output_path : str, optional
        Path to save figure.
    """
    agg_results = results_df.groupby('dimension').agg({
        'memory_peak': ['mean', 'std']
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = agg_results['dimension'].values
    y = agg_results[('memory_peak', 'mean')].values
    yerr = agg_results[('memory_peak', 'std')].values
    
    ax.errorbar(x, y, yerr=yerr, marker='o', capsize=5, capthick=2,
               linewidth=2, markersize=8, color='#e74c3c')
    ax.fill_between(x, y - yerr, y + yerr, alpha=0.2, color='#e74c3c')
    
    ax.set_xlabel('Dimension', fontsize=12, fontweight='bold')
    ax.set_ylabel('Peak Memory (MB)', fontsize=12, fontweight='bold')
    ax.set_title('Memory Usage vs Dimension', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_sparse_comparison(results_df, output_path=None):
    """
    Plot minority samples vs performance for all methods.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results with columns: n_minority, method, f1_score, etc.
    output_path : str, optional
        Path to save figure.
    """
    # Aggregate by method and n_minority
    agg_results = results_df.groupby(['n_minority', 'method']).agg({
        'f1_score': ['mean', 'std'],
        'roc_auc': ['mean', 'std']
    }).reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    metrics = [
        ('f1_score', 'F1-Score', axes[0]),
        ('roc_auc', 'ROC-AUC', axes[1])
    ]
    
    methods = results_df['method'].unique()
    colors = sns.color_palette("husl", len(methods))
    
    for metric_name, metric_label, ax in metrics:
        for method, color in zip(methods, colors):
            method_data = agg_results[agg_results['method'] == method]
            
            if len(method_data) > 0:
                x = method_data['n_minority'].values
                y = method_data[(metric_name, 'mean')].values
                yerr = method_data[(metric_name, 'std')].values
                
                ax.errorbar(x, y, yerr=yerr, marker='o', capsize=5,
                           label=method, linewidth=2, markersize=8, color=color)
                ax.fill_between(x, y - yerr, y + yerr, alpha=0.15, color=color)
        
        ax.set_xlabel('Number of Minority Samples', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_label, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_label} vs Minority Sample Size', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        ax.set_xscale('log')
        ax.set_xticks([10, 20, 50, 100, 200])
        ax.set_xticklabels(['10', '20', '50', '100', '200'])
    
    plt.suptitle('Performance in Extreme Sparse Settings',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_sparse_comparison_by_dataset(results_df, output_path=None):
    """
    Plot sparse comparison with separate subplot for each dataset.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe.
    output_path : str, optional
        Path to save figure.
    """
    datasets = sorted(results_df['dataset'].unique())
    n_datasets = len(datasets)
    
    fig, axes = plt.subplots(2, (n_datasets + 1) // 2, figsize=(15, 8))
    axes = axes.flatten()
    
    methods = sorted(results_df['method'].unique())
    colors = sns.color_palette("husl", len(methods))
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        dataset_data = results_df[results_df['dataset'] == dataset]
        
        agg = dataset_data.groupby(['n_minority', 'method']).agg({
            'f1_score': ['mean', 'std']
        }).reset_index()
        
        for method, color in zip(methods, colors):
            method_data = agg[agg['method'] == method]
            
            if len(method_data) > 0:
                x = method_data['n_minority'].values
                y = method_data[('f1_score', 'mean')].values
                yerr = method_data[('f1_score', 'std')].values
                
                ax.errorbar(x, y, yerr=yerr, marker='o', capsize=3,
                           label=method, linewidth=2, markersize=6, color=color)
        
        ax.set_title(dataset, fontsize=10, fontweight='bold')
        ax.set_xlabel('N Minority', fontsize=9)
        ax.set_ylabel('F1-Score', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        if idx == 0:
            ax.legend(fontsize=8, loc='best')
    
    # Hide unused subplots
    for idx in range(n_datasets, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Sparse Setting Performance by Dataset',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def create_scalability_table(results_df, output_path=None):
    """
    Create LaTeX table for scalability results.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe.
    output_path : str, optional
        Path to save table.
        
    Returns
    -------
    table_df : pd.DataFrame
        Summary table.
    """
    summary = results_df.groupby('dimension').agg({
        'f1_score': ['mean', 'std'],
        'roc_auc': ['mean', 'std'],
        'training_time': ['mean', 'std'],
        'memory_peak': ['mean', 'std']
    }).reset_index()
    
    # Format table
    table_data = []
    for _, row in summary.iterrows():
        table_data.append({
            'Dimension': int(row['dimension']),
            'F1-Score': f"{row[('f1_score', 'mean')]:.3f} ± {row[('f1_score', 'std')]:.3f}",
            'ROC-AUC': f"{row[('roc_auc', 'mean')]:.3f} ± {row[('roc_auc', 'std')]:.3f}",
            'Training Time (s)': f"{row[('training_time', 'mean')]:.1f} ± {row[('training_time', 'std')]:.1f}",
            'Memory (MB)': f"{row[('memory_peak', 'mean')]:.1f} ± {row[('memory_peak', 'std')]:.1f}"
        })
    
    table_df = pd.DataFrame(table_data)
    
    if output_path:
        # Save CSV
        csv_path = Path(output_path).with_suffix('.csv')
        table_df.to_csv(csv_path, index=False)
        
        # Save LaTeX
        latex_path = Path(output_path).with_suffix('.tex')
        with open(latex_path, 'w') as f:
            f.write(table_df.to_latex(index=False, escape=False))
        
        print(f"Table saved to: {csv_path}")
        print(f"LaTeX saved to: {latex_path}")
    
    return table_df


def create_sparse_comparison_table(results_df, output_path=None):
    """
    Create comparison table for sparse setting.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe.
    output_path : str, optional
        Path to save table.
        
    Returns
    -------
    table_df : pd.DataFrame
        Comparison table.
    """
    # Aggregate by n_minority and method
    summary = results_df.groupby(['n_minority', 'method']).agg({
        'f1_score': ['mean', 'std'],
        'roc_auc': ['mean', 'std']
    }).reset_index()
    
    # Pivot to create comparison table
    f1_table = summary.pivot(index='n_minority', columns='method', values=('f1_score', 'mean'))
    f1_std_table = summary.pivot(index='n_minority', columns='method', values=('f1_score', 'std'))
    
    # Format table
    table_data = []
    for n_min in f1_table.index:
        row = {'N_Minority': int(n_min)}
        for method in f1_table.columns:
            mean_val = f1_table.loc[n_min, method]
            std_val = f1_std_table.loc[n_min, method]
            row[method] = f"{mean_val:.3f}±{std_val:.3f}"
        table_data.append(row)
    
    table_df = pd.DataFrame(table_data)
    
    if output_path:
        csv_path = Path(output_path).with_suffix('.csv')
        table_df.to_csv(csv_path, index=False)
        
        latex_path = Path(output_path).with_suffix('.tex')
        with open(latex_path, 'w') as f:
            f.write(table_df.to_latex(index=False, escape=False))
        
        print(f"Table saved to: {csv_path}")
        print(f"LaTeX saved to: {latex_path}")
    
    return table_df


def main():
    parser = argparse.ArgumentParser(description='Visualize scalability test results')
    parser.add_argument(
        '--results-file',
        type=str,
        required=True,
        help='Path to results CSV file'
    )
    parser.add_argument(
        '--test-type',
        type=str,
        choices=['high_dim', 'sparse'],
        required=True,
        help='Type of test (high_dim or sparse)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for figures'
    )
    
    args = parser.parse_args()
    
    # Load results
    results_df = pd.read_csv(args.results_file)
    print(f"Loaded {len(results_df)} results from {args.results_file}")
    
    # Set output directory
    if args.output_dir is None:
        output_dir = Path(args.results_file).parent.parent.parent / 'figures' / 'scalability'
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    if args.test_type == 'high_dim':
        print("\nGenerating high-dimensional scalability plots...")
        
        # Performance vs dimension
        plot_dimension_vs_performance(
            results_df,
            output_path=str(output_dir / 'dimension_vs_performance.png')
        )
        
        # Runtime vs dimension
        if 'training_time' in results_df.columns:
            plot_dimension_vs_runtime(
                results_df,
                output_path=str(output_dir / 'dimension_vs_runtime.png')
            )
        
        # Memory vs dimension
        if 'memory_peak' in results_df.columns:
            plot_dimension_vs_memory(
                results_df,
                output_path=str(output_dir / 'dimension_vs_memory.png')
            )
        
        # Summary table
        create_scalability_table(
            results_df,
            output_path=str(output_dir / 'scalability_summary')
        )
    
    elif args.test_type == 'sparse':
        print("\nGenerating sparse setting comparison plots...")
        
        # Comparison plot
        plot_sparse_comparison(
            results_df,
            output_path=str(output_dir / 'sparse_comparison.png')
        )
        
        # By-dataset plot
        plot_sparse_comparison_by_dataset(
            results_df,
            output_path=str(output_dir / 'sparse_comparison_by_dataset.png')
        )
        
        # Comparison table
        create_sparse_comparison_table(
            results_df,
            output_path=str(output_dir / 'sparse_comparison_table')
        )
    
    print(f"\nVisualization complete! Output directory: {output_dir}")


if __name__ == '__main__':
    main()

