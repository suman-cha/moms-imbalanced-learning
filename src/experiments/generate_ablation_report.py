"""
Generate Comprehensive Ablation Study Report.

Creates a detailed markdown report analyzing hyperparameter sensitivity
and providing recommendations for optimal values.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import yaml


def load_all_results(results_base_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load all ablation study results.
    
    Parameters
    ----------
    results_base_dir : str
        Base directory containing ablation study results.
        
    Returns
    -------
    results_dict : Dict[str, pd.DataFrame]
        Dictionary mapping experiment names to result dataframes.
    """
    results_base = Path(results_base_dir)
    results_dict = {}
    
    experiment_names = [
        'triplet_margin_ablation',
        'mmd_bandwidth_ablation',
        'danger_k_ablation',
        'lambda_ablation_extended'
    ]
    
    for exp_name in experiment_names:
        results_file = results_base / exp_name / f"{exp_name}_results.csv"
        if results_file.exists():
            results_dict[exp_name] = pd.read_csv(results_file)
        else:
            print(f"Warning: Results file not found: {results_file}")
    
    return results_dict


def analyze_parameter_sensitivity(
    results_df: pd.DataFrame,
    parameter_name: str
) -> Dict:
    """
    Analyze sensitivity of a parameter.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe.
    parameter_name : str
        Name of the parameter.
        
    Returns
    -------
    analysis : Dict
        Analysis results including optimal values, robust ranges, etc.
    """
    # Focus on F1-score as primary metric
    f1_data = results_df[results_df['metric'] == 'f1_score'].copy()
    
    analysis = {
        'parameter_name': parameter_name,
        'datasets': {},
        'overall_optimal': None,
        'overall_robust_range': None,
        'sensitivity_score': None
    }
    
    # Analyze per dataset
    for dataset in f1_data['dataset'].unique():
        dataset_data = f1_data[f1_data['dataset'] == dataset].sort_values('parameter_value')
        
        if len(dataset_data) == 0:
            continue
        
        # Optimal value
        optimal_idx = dataset_data['mean'].idxmax()
        optimal_value = dataset_data.loc[optimal_idx, 'parameter_value']
        optimal_score = dataset_data.loc[optimal_idx, 'mean']
        
        # Robust range (within 5% of optimal)
        threshold = optimal_score * 0.95
        robust_mask = dataset_data['mean'] >= threshold
        robust_values = dataset_data[robust_mask]['parameter_value'].values
        
        if len(robust_values) > 0:
            robust_min = robust_values.min()
            robust_max = robust_values.max()
        else:
            robust_min = robust_max = optimal_value
        
        # Sensitivity: coefficient of variation
        cv = dataset_data['mean'].std() / dataset_data['mean'].mean() if dataset_data['mean'].mean() > 0 else 0
        
        analysis['datasets'][dataset] = {
            'optimal_value': optimal_value,
            'optimal_f1': optimal_score,
            'robust_min': robust_min,
            'robust_max': robust_max,
            'sensitivity': cv
        }
    
    # Overall analysis
    all_optimal_values = [v['optimal_value'] for v in analysis['datasets'].values()]
    all_robust_mins = [v['robust_min'] for v in analysis['datasets'].values()]
    all_robust_maxs = [v['robust_max'] for v in analysis['datasets'].values()]
    
    if all_optimal_values:
        analysis['overall_optimal'] = np.median(all_optimal_values)
        analysis['overall_robust_range'] = (
            np.median(all_robust_mins),
            np.median(all_robust_maxs)
        )
        analysis['sensitivity_score'] = np.mean([v['sensitivity'] for v in analysis['datasets'].values()])
    
    return analysis


def generate_report(
    results_dict: Dict[str, pd.DataFrame],
    output_path: str
):
    """
    Generate comprehensive ablation study report.
    
    Parameters
    ----------
    results_dict : Dict[str, pd.DataFrame]
        Dictionary of results dataframes.
    output_path : str
        Path to save the report.
    """
    report_lines = []
    
    # Header
    report_lines.append("# Hyperparameter Sensitivity Analysis Report")
    report_lines.append("")
    report_lines.append("## Executive Summary")
    report_lines.append("")
    report_lines.append("This report presents a comprehensive analysis of hyperparameter sensitivity ")
    report_lines.append("for the MOMS (Minority Oversampling with Majority Selection) model. ")
    report_lines.append("Four key hyperparameters were systematically evaluated across 10 diverse datasets.")
    report_lines.append("")
    
    # Parameter mapping
    param_mapping = {
        'triplet_margin_ablation': {
            'name': 'Triplet Margin (α)',
            'description': 'Controls the margin in triplet loss, determining the separation between positive and negative pairs.',
            'default': 1.0,
            'range': '[0.1, 5.0]'
        },
        'mmd_bandwidth_ablation': {
            'name': 'MMD Kernel Bandwidth (σ)',
            'description': 'Bandwidth parameter for the Gaussian kernel in Maximum Mean Discrepancy (MMD) loss.',
            'default': 1.0,
            'range': '[0.1, 5.0]'
        },
        'danger_k_ablation': {
            'name': 'Danger Set k Value',
            'description': 'Number of nearest neighbors used to identify danger set samples (minority samples near majority class boundary).',
            'default': 5,
            'range': '[3, 15]'
        },
        'lambda_ablation_extended': {
            'name': 'Lambda (β) - Triplet Loss Weight',
            'description': 'Weight coefficient for triplet loss regularization term in the overall loss function.',
            'default': 0.01,
            'range': '[0.0, 2.0]'
        }
    }
    
    # Analyze each parameter
    analyses = {}
    for exp_name, results_df in results_dict.items():
        param_name = results_df['parameter'].iloc[0]
        analyses[exp_name] = analyze_parameter_sensitivity(results_df, param_name)
    
    # Detailed analysis for each parameter
    report_lines.append("## Detailed Analysis")
    report_lines.append("")
    
    for exp_name, analysis in analyses.items():
        param_info = param_mapping.get(exp_name, {})
        param_display_name = param_info.get('name', exp_name)
        
        report_lines.append(f"### {param_display_name}")
        report_lines.append("")
        report_lines.append(f"**Description:** {param_info.get('description', 'N/A')}")
        report_lines.append("")
        report_lines.append(f"**Tested Range:** {param_info.get('range', 'N/A')}")
        report_lines.append("")
        report_lines.append(f"**Default Value:** {param_info.get('default', 'N/A')}")
        report_lines.append("")
        
        # Overall recommendations
        if analysis['overall_optimal'] is not None:
            report_lines.append("#### Overall Recommendations")
            report_lines.append("")
            report_lines.append(f"- **Optimal Value:** {analysis['overall_optimal']:.4f}")
            report_lines.append(f"- **Robust Range:** [{analysis['overall_robust_range'][0]:.4f}, {analysis['overall_robust_range'][1]:.4f}]")
            report_lines.append(f"- **Sensitivity Score:** {analysis['sensitivity_score']:.4f} (lower is more robust)")
            report_lines.append("")
        
        # Per-dataset results
        report_lines.append("#### Dataset-Specific Results")
        report_lines.append("")
        report_lines.append("| Dataset | Optimal Value | Optimal F1 | Robust Range | Sensitivity |")
        report_lines.append("|---------|---------------|------------|--------------|------------|")
        
        for dataset, dataset_analysis in analysis['datasets'].items():
            robust_range_str = f"[{dataset_analysis['robust_min']:.2f}, {dataset_analysis['robust_max']:.2f}]"
            report_lines.append(
                f"| {dataset} | {dataset_analysis['optimal_value']:.4f} | "
                f"{dataset_analysis['optimal_f1']:.4f} | {robust_range_str} | "
                f"{dataset_analysis['sensitivity']:.4f} |"
            )
        
        report_lines.append("")
    
    # Summary and recommendations
    report_lines.append("## Summary and Recommendations")
    report_lines.append("")
    
    report_lines.append("### Key Findings")
    report_lines.append("")
    
    for exp_name, analysis in analyses.items():
        param_info = param_mapping.get(exp_name, {})
        param_display_name = param_info.get('name', exp_name)
        
        if analysis['overall_optimal'] is not None:
            sensitivity_level = "Low" if analysis['sensitivity_score'] < 0.1 else \
                               "Medium" if analysis['sensitivity_score'] < 0.2 else "High"
            
            report_lines.append(f"1. **{param_display_name}**")
            report_lines.append(f"   - Optimal value: {analysis['overall_optimal']:.4f}")
            report_lines.append(f"   - Sensitivity: {sensitivity_level}")
            report_lines.append(f"   - Recommendation: {'Robust across datasets' if sensitivity_level == 'Low' else 'Dataset-specific tuning recommended'}")
            report_lines.append("")
    
    report_lines.append("### Default Value Recommendations")
    report_lines.append("")
    report_lines.append("Based on the analysis across all datasets, the following default values are recommended:")
    report_lines.append("")
    
    for exp_name, analysis in analyses.items():
        param_info = param_mapping.get(exp_name, {})
        param_display_name = param_info.get('name', exp_name)
        
        if analysis['overall_optimal'] is not None:
            report_lines.append(f"- **{param_display_name}:** {analysis['overall_optimal']:.4f}")
    
    report_lines.append("")
    report_lines.append("### Robust Range Guidelines")
    report_lines.append("")
    report_lines.append("For datasets with different characteristics, the following robust ranges can be used:")
    report_lines.append("")
    
    for exp_name, analysis in analyses.items():
        param_info = param_mapping.get(exp_name, {})
        param_display_name = param_info.get('name', exp_name)
        
        if analysis['overall_robust_range'] is not None:
            robust_min, robust_max = analysis['overall_robust_range']
            report_lines.append(f"- **{param_display_name}:** [{robust_min:.4f}, {robust_max:.4f}]")
    
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("*Report generated automatically from ablation study results.*")
    
    # Write report
    report_content = "\n".join(report_lines)
    with open(output_path, 'w') as f:
        f.write(report_content)
    
    print(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate ablation study report'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        required=True,
        help='Base directory containing ablation study results'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/ablation_study_report.md',
        help='Output path for the report'
    )
    
    args = parser.parse_args()
    
    # Load all results
    print("Loading ablation study results...")
    results_dict = load_all_results(args.results_dir)
    
    if len(results_dict) == 0:
        print("No results found. Please run ablation studies first.")
        return
    
    print(f"Loaded {len(results_dict)} experiment results")
    
    # Generate report
    print("Generating report...")
    generate_report(results_dict, args.output)
    
    print("Report generation complete!")


if __name__ == '__main__':
    main()

