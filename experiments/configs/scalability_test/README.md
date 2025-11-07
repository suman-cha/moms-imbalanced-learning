# Scalability Test Configurations

## Purpose

These configurations address reviewer concerns about:
1. **High-dimensional scalability** (500+, 1000+ dimensions)
2. **Extreme sparse settings** (10, 20, 50 minority samples)

## Configuration Files

### 1. high_dimensional_test.yaml

Tests MOMS performance on high-dimensional data.

**Key Settings:**
- Target dimensions: [294, 500, 768, 1000, 1500, 2000]
- Base datasets: ionosphere, penbased, wdbc
- Dimension augmentation: Random noise features
- Metrics: Performance + Time + Memory

**Why These Dimensions:**
- 294: Typical intermediate dimension
- 500-768: Common embedding dimensions (BERT-like)
- 1000-2000: High-dimensional scenarios (image features)

**Measurements:**
- F1-score, ROC-AUC, Balanced Accuracy
- Training time
- Prediction time
- Peak memory usage
- MMD computation time
- k-NN graph construction time

### 2. sparse_minority_test.yaml

Compares MOMS vs traditional methods with few minority samples.

**Key Settings:**
- Minority sample counts: [10, 20, 50, 100, 200]
- Methods: Original, SMOTE, ADASYN, BorderlineSMOTE, MOMS
- Datasets: ecoli3, glass4, yeast4, ionosphere, pima
- Classifiers: RF, SVM, DT

**Adaptive k-neighbors:**
```yaml
k_neighbors_map:
  10: 3   # If 10 samples, use k=3
  20: 5   # If 20 samples, use k=5
  50: 5
  100: 5
  200: 5
```

**Statistical Tests:**
- Wilcoxon signed-rank test
- Significance level: α = 0.05
- Critical threshold detection

### 3. quick_test.yaml (Optional)

Faster version for testing setup:
- Fewer runs (2 instead of 5/10)
- Fewer splits (3 instead of 5)
- Fewer dimensions/counts
- **Use this first to verify everything works!**

## Experiment Design

### High-Dimensional Test

```
For each target dimension (6 dims):
  For each base dataset (3 datasets):
    Augment dataset to target_dim
    For each run (5 runs):
      For each fold (5 folds):
        Train MOMS
        Generate synthetics
        Evaluate with 3 classifiers
        Measure time & memory
        
Total: 6 × 3 × 5 × 5 × 3 = 2,700 classifier trainings
Time: ~4-8 hours (CPU)
```

### Sparse Setting Test

```
For each minority count (5 counts):
  For each dataset (5 datasets):
    Create sparse version (limit minority samples)
    For each method (5 methods):
      For each run (10 runs):
        For each fold (5 folds):
          Apply method
          Evaluate with 3 classifiers
          
Total: 5 × 5 × 5 × 10 × 5 × 3 = 18,750 classifier trainings
Time: ~3-6 hours (CPU)
```

## Key Parameters

### High-Dimensional Test

| Parameter | Value | Reason |
|-----------|-------|--------|
| n_epochs | 1000 | Reduced for efficiency |
| hidden_dims | [128, 64] | Larger for high-dim |
| latent_dim | 32 | Larger latent space |
| n_runs | 5 | Balance speed/reliability |
| n_splits | 5 | Standard CV |

### Sparse Setting Test

| Parameter | Value | Reason |
|-----------|-------|--------|
| n_epochs | 1000 | Standard |
| hidden_dims | [64, 32] | Standard |
| danger_k | 3 | Reduced for sparse |
| n_runs | 10 | More runs for significance |
| n_splits | 5 | Standard CV |

## Expected Outputs

### High-Dimensional Test

**CSV Results:**
```
dimension, dataset, run, fold, f1_score, roc_auc, balanced_accuracy,
training_time, prediction_time, memory_peak, mmd_time, knn_time
```

**Figures:**
- `dimension_vs_performance.png`: 3-panel plot (F1, AUC, BAcc)
- `dimension_vs_runtime.png`: Log-scale time plot
- `dimension_vs_memory.png`: Memory usage plot

**Tables:**
- `scalability_summary.csv`: Aggregated statistics
- `scalability_summary.tex`: LaTeX format

### Sparse Setting Test

**CSV Results:**
```
n_minority, dataset, method, run, fold, f1_score, roc_auc,
precision, recall
```

**Figures:**
- `sparse_comparison.png`: Methods comparison (F1 + AUC)
- `sparse_comparison_by_dataset.png`: 5-panel subplot

**Tables:**
- `sparse_comparison_table.csv`: N_minority × Methods
- `sparse_comparison_table.tex`: LaTeX format

## Quick Test First!

Before running full experiments, test with reduced settings:

1. Edit config files:
```yaml
evaluation:
  n_runs: 2
  n_splits: 3

# For high_dim
dimensions:
  target_dims: [294, 500, 1000]

# For sparse
minority_sample_counts: [10, 50, 100]
```

2. Run:
```cmd
run_quick_test.bat
```

3. Should complete in **~1-2 hours**

4. Verify results look reasonable

5. Then run full experiments

## 논문에 추가할 내용

리포트 생성 후 자동으로 포함됨:

1. **Scalability Section:**
   - High-dimensional performance
   - Computational complexity analysis
   - Memory efficiency

2. **Limitations Section:**
   - Sparse setting recommendations
   - When to use SMOTE vs MOMS
   - Critical sample size threshold

3. **Supplementary Materials:**
   - All figures and tables
   - Detailed experimental setup
   - Statistical test results

## Troubleshooting

### "Dimension too high, out of memory"
- Reduce `hidden_dims` and `latent_dim`
- Reduce `n_epochs`
- Test on fewer dimensions

### "Too few minority samples for SMOTE"
- This is expected and handled automatically
- k_neighbors is adjusted based on sample count
- Methods that fail return baseline performance

### "Taking too long"
- Run overnight
- Reduce `n_runs` and `n_splits`
- Test fewer dimensions/counts first

## Summary

**Total Experiments:**
- High-dim: ~450 evaluations → 4-8 hours
- Sparse: ~6,250 evaluations → 3-6 hours
- **Combined: ~7-14 hours (CPU)**

**Outputs:**
- 2 CSV results files
- 8 visualization files (PNG)
- 4 table files (CSV + LaTeX)
- 1 comprehensive report (Markdown)
- LaTeX paragraph for paper

**Ready to Run:**
```cmd
run_scalability_tests.bat
```

Or start with quick test to verify setup works!

