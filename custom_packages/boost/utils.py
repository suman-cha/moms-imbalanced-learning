"""
Utility Functions for Imbalanced Learning.

This module provides helper functions for configuration management,
logging, and common operations in imbalanced learning experiments.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import yaml
import logging
import numpy as np


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
        
    Returns
    -------
    config : Dict[str, Any]
        Loaded configuration dictionary.
        
    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    yaml.YAMLError
        If the file contains invalid YAML.
        
    Examples
    --------
    >>> config = load_config("config/default_config.yaml")
    >>> print(config['smote']['k_neighbors'])
    5
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(
            f"Error parsing configuration file {config_path}: {e}"
        )


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Parameters
    ----------
    level : str, default="INFO"
        Logging level: DEBUG, INFO, WARNING, ERROR, or CRITICAL.
    log_file : Optional[str], default=None
        Path to log file. If None, logs only to console.
    format_string : Optional[str], default=None
        Custom log format string. If None, uses default format.
        
    Returns
    -------
    logger : logging.Logger
        Configured logger instance.
        
    Examples
    --------
    >>> logger = setup_logging(level="DEBUG", log_file="experiment.log")
    >>> logger.info("Experiment started")
    """
    if format_string is None:
        format_string = (
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(),
        ]
    )
    
    # Add file handler if log_file is specified
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_string))
        logging.getLogger().addHandler(file_handler)
    
    return logging.getLogger(__name__)


def set_random_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Parameters
    ----------
    seed : int
        Random seed value.
        
    Notes
    -----
    This function sets seeds for:
    - NumPy random number generator
    - Python's built-in random module (if imported)
    
    For full reproducibility, also set seeds in:
    - scikit-learn estimators (via random_state parameter)
    - Other libraries used in the experiment
    
    Examples
    --------
    >>> set_random_seed(42)
    >>> # All subsequent random operations will be reproducible
    """
    np.random.seed(seed)
    
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass


def compute_class_weights(
    y: np.ndarray,
    strategy: str = "balanced"
) -> Dict[int, float]:
    """
    Compute class weights for imbalanced datasets.
    
    Parameters
    ----------
    y : np.ndarray of shape (n_samples,)
        Class labels.
    strategy : str, default="balanced"
        Weighting strategy:
        - "balanced": weights inversely proportional to class frequencies
        - "equal": all classes weighted equally
        
    Returns
    -------
    class_weights : Dict[int, float]
        Dictionary mapping class labels to weights.
        
    Examples
    --------
    >>> y = np.array([0, 0, 0, 0, 1])
    >>> weights = compute_class_weights(y)
    >>> print(weights)
    {0: 0.625, 1: 2.5}
    """
    unique_classes, class_counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(unique_classes)
    
    if strategy == "balanced":
        # Compute balanced class weights
        weights = n_samples / (n_classes * class_counts)
    elif strategy == "equal":
        # Equal weights for all classes
        weights = np.ones(n_classes)
    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            "Valid options: 'balanced', 'equal'"
        )
    
    return dict(zip(unique_classes, weights))


def validate_input_data(
    X: np.ndarray,
    y: np.ndarray,
    min_samples_per_class: int = 2
) -> None:
    """
    Validate input data for imbalanced learning algorithms.
    
    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray of shape (n_samples,)
        Target labels.
    min_samples_per_class : int, default=2
        Minimum required samples per class.
        
    Raises
    ------
    ValueError
        If validation fails.
        
    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y = np.array([0, 0, 1])
    >>> validate_input_data(X, y)  # Passes validation
    """
    # Check dimensions
    if X.ndim != 2:
        raise ValueError(f"X must be 2D array, got shape {X.shape}")
    
    if y.ndim != 1:
        raise ValueError(f"y must be 1D array, got shape {y.shape}")
    
    # Check sample counts match
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y have different sample counts: "
            f"{X.shape[0]} vs {y.shape[0]}"
        )
    
    # Check for NaN values
    if np.isnan(X).any():
        raise ValueError("X contains NaN values")
    
    if np.isnan(y).any():
        raise ValueError("y contains NaN values")
    
    # Check minimum samples per class
    unique_classes, class_counts = np.unique(y, return_counts=True)
    
    for class_label, count in zip(unique_classes, class_counts):
        if count < min_samples_per_class:
            raise ValueError(
                f"Class {class_label} has only {count} samples, "
                f"need at least {min_samples_per_class}"
            )

