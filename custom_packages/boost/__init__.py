"""
Boosting algorithms for imbalanced classification.

This module implements various boosting algorithms designed specifically
for imbalanced learning scenarios, including:
- OUBoost: Over-Under Boosting
- SMOTEBoost: SMOTE-based Boosting
- RUSBoost: Random Under-Sampling Boosting
"""

from .boost import OUBoost, SMOTEBoost, RUSBoost, SMOTE, Sampler
from ._weight_boosting import (
    AdaBoostClassifier,
    AdaBoostClassifierOUBoost,
    AdaBoostRegressor,
)

__all__ = [
    "OUBoost",
    "SMOTEBoost",
    "RUSBoost",
    "SMOTE",
    "Sampler",
    "AdaBoostClassifier",
    "AdaBoostClassifierOUBoost",
    "AdaBoostRegressor",
]

