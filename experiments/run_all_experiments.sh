#!/bin/bash
# =============================================================================
# Run All Experiments Script
# Paper: "Learning Majority-to-Minority Transformations with MMD and Triplet Loss 
#         for Imbalanced Classification" (arXiv:2509.11511)
# 
# Paper Hyperparameters (Section 4.2):
#   - lambda (beta): 0.01
#   - k (triplet neighbors): 5  
#   - margin (alpha): 1.0
#   - Architecture: Auto-scaled based on input dimension d
#     - Hidden dims: [d×2, d×4, d×8, d×16]
#     - Latent dim: d×32
# =============================================================================

set -e  # Exit on error

# Configuration
DEVICE="${CUDA_DEVICE:-cuda:0}"
SAVE_PATH="${SAVE_PATH:-./results}"
SEED="${SEED:-1203}"

echo "=============================================="
echo "Imbalanced Classification Experiments"
echo "=============================================="
echo "Device: $DEVICE"
echo "Save path: $SAVE_PATH"
echo "Seed: $SEED"
echo ""

# Create results directory
mkdir -p "$SAVE_PATH"

# =============================================================================
# Main Experiments (Paper Section 4.3)
# =============================================================================
echo "----------------------------------------------"
echo "Running Main Experiments on imblearn datasets..."
echo "----------------------------------------------"

python experiments/run_main_experiments.py \
    --datasets \
        us_crime oil car_eval_34 arrhythmia coil_2000 \
        letter_img mammography optical_digits ozone_level \
        pen_digits satimage sick_euthyroid spectrometer \
        thyroid_sick wine_quality yeast_me2 \
    --methods ROS SMOTE bSMOTE ADASYN MWMOTE CTGAN GAMO MGVAE MMD MMD+T \
    --classifiers SVM \
    --device "$DEVICE" \
    --save_path "$SAVE_PATH" \
    --seed "$SEED"

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "Results saved to: $SAVE_PATH"
echo "=============================================="

