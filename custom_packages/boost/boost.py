"""
Boosting Algorithms for Imbalanced Classification.

This module implements advanced boosting algorithms specifically designed
for handling imbalanced datasets in classification tasks.

References
----------
.. [1] Chawla, N. V., Lazarevic, A., Hall, L. O., & Bowyer, K. W. (2003).
       "SMOTEBoost: Improving Prediction of the Minority Class in Boosting."
       In European Conference on Principles of Data Mining and Knowledge Discovery (PKDD).

.. [2] Seiffert, C., Khoshgoftaar, T. M., Hulse, J. V., & Napolitano, A. (2008).
       "RUSBoost: Improving Classification Performance when Training Data is Skewed."
       In International Conference on Pattern Recognition (ICPR).

.. [3] Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002).
       "SMOTE: Synthetic Minority Over-Sampling Technique."
       Journal of Artificial Intelligence Research (JAIR), 16, 321-357.
"""

from typing import Optional, Tuple, Union
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import is_regressor
from sklearn.ensemble._forest import BaseForest
from sklearn.preprocessing import normalize
from sklearn.tree import BaseDecisionTree
from sklearn.utils import check_random_state, check_X_y, check_array
from sklearn.neighbors import NearestNeighbors

from ._weight_boosting import AdaBoostClassifierOUBoost, AdaBoostClassifier
from . import clustering_selection

class SMOTE:
    """
    Synthetic Minority Over-Sampling Technique (SMOTE).

    SMOTE performs oversampling of the minority class by generating synthetic
    samples along the line segments joining k nearest minority class neighbors.
    
    Given a minority sample :math:`x_i`, SMOTE generates a synthetic sample as:
    
    .. math::
        x_{syn} = x_i + \\lambda \\cdot (x_{nn} - x_i)
    
    where :math:`x_{nn}` is a randomly chosen k-nearest neighbor and
    :math:`\\lambda \\in [0, 1]` is a random interpolation factor.

    Parameters
    ----------
    k_neighbors : int, default=5
        Number of nearest neighbors to use for generating synthetic samples.
        Valid range: k_neighbors >= 1.
        
    random_state : Optional[int], default=None
        Controls the randomness of sample generation.
        - If int, random_state is the seed used by the random number generator.
        - If None, uses the current numpy random state.
        
    Attributes
    ----------
    X_ : np.ndarray of shape (n_minority_samples, n_features)
        The minority class training samples.
        
    n_minority_samples_ : int
        Number of minority class samples used for fitting.
        
    n_features_ : int
        Number of features in the training data.
        
    neigh_ : NearestNeighbors
        Fitted nearest neighbors estimator.

    References
    ----------
    .. [1] Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002).
           "SMOTE: Synthetic Minority Over-Sampling Technique."
           Journal of Artificial Intelligence Research (JAIR), 16, 321-357.

    Examples
    --------
    >>> import numpy as np
    >>> from custom_packages.boost import SMOTE
    >>> X_minority = np.array([[1, 2], [2, 3], [3, 4]])
    >>> smote = SMOTE(k_neighbors=2, random_state=42)
    >>> smote.fit(X_minority)
    >>> X_synthetic = smote.sample(n_samples=10)
    >>> X_synthetic.shape
    (10, 2)
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        random_state: Optional[int] = None
    ) -> None:
        if k_neighbors < 1:
            raise ValueError(
                f"k_neighbors must be >= 1, got {k_neighbors}"
            )
        
        self.k = k_neighbors
        self.random_state = random_state

    def sample(self, n_samples: int) -> np.ndarray:
        """
        Generate synthetic minority class samples.

        Parameters
        ----------
        n_samples : int
            Number of synthetic samples to generate.
            Valid range: n_samples >= 1.

        Returns
        -------
        X_synthetic : np.ndarray of shape (n_samples, n_features)
            Generated synthetic samples.
            
        Raises
        ------
        ValueError
            If n_samples < 1 or if the model has not been fitted.
            
        Notes
        -----
        Each synthetic sample is created by:
        1. Randomly selecting a minority sample x_i
        2. Finding its k nearest neighbors
        3. Randomly selecting one neighbor x_nn
        4. Interpolating: x_syn = x_i + λ·(x_nn - x_i), λ ~ U[0,1]
        """
        if not hasattr(self, 'X_'):
            raise ValueError(
                "SMOTE instance is not fitted. Call 'fit' before 'sample'."
            )
        
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        
        # Set random seed for reproducibility
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Pre-allocate array for synthetic samples
        X_synthetic = np.zeros((n_samples, self.n_features_), dtype=np.float64)
        
        # Generate synthetic samples
        for i in range(n_samples):
            # Randomly select a minority sample
            sample_idx = np.random.randint(0, self.X_.shape[0])
            
            # Find k-nearest neighbors (excluding the sample itself)
            neighbors_idx = self.neigh_.kneighbors(
                self.X_[sample_idx].reshape(1, -1),
                return_distance=False
            )[0, 1:]  # Exclude first neighbor (the sample itself)
            
            # Randomly select one neighbor
            neighbor_idx = np.random.choice(neighbors_idx)
            
            # Compute difference vector
            diff_vector = self.X_[neighbor_idx] - self.X_[sample_idx]
            
            # Generate random interpolation factor λ ∈ [0, 1]
            lambda_interp = np.random.random()
            
            # Generate synthetic sample: x_syn = x_i + λ·(x_nn - x_i)
            X_synthetic[i, :] = self.X_[sample_idx, :] + lambda_interp * diff_vector

        return X_synthetic

    def fit(self, X: np.ndarray) -> 'SMOTE':
        """
        Fit the SMOTE model on minority class samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_minority_samples, n_features)
            Minority class training samples.

        Returns
        -------
        self : SMOTE
            Fitted SMOTE instance.
            
        Raises
        ------
        ValueError
            If X has fewer samples than k_neighbors + 1.
        """
        # Validate input
        if not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=np.float64)
        
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2D array, got shape {X.shape}"
            )
        
        self.X_ = X
        self.n_minority_samples_, self.n_features_ = self.X_.shape
        
        # Check if we have enough samples
        if self.n_minority_samples_ <= self.k:
            raise ValueError(
                f"Cannot use SMOTE with k_neighbors={self.k} when only "
                f"{self.n_minority_samples_} minority samples are available. "
                f"Need at least {self.k + 1} samples."
            )

        # Fit nearest neighbors model (k+1 to include the sample itself)
        self.neigh_ = NearestNeighbors(n_neighbors=self.k + 1)
        self.neigh_.fit(self.X_)

        return self

class Sampler:
    """
    Combined Sampler for Imbalanced Learning with Clustering-based Selection.
    
    This sampler implements both SMOTE-like oversampling for the minority class
    and clustering-based intelligent undersampling for the majority class.
    
    The sampling strategy uses Density Peak Clustering (DPC) to identify
    representative majority class samples, combined with SMOTE to generate
    synthetic minority samples.
    
    Parameters
    ----------
    k_neighbors : int, default=5
        Number of nearest neighbors for SMOTE-based minority oversampling.
    with_replacement : bool, default=False
        Whether to sample with replacement (currently not used in clustering).
    return_indices : bool, default=False
        Whether to return indices of selected samples.
    random_state : Optional[int], default=None
        Random seed for reproducibility.
    alpha : float, default=0.5
        Weight for density component in clustering selection.
    beta : float, default=0.5
        Weight for distance component in clustering selection.
        
    Attributes
    ----------
    X_ : np.ndarray
        Stored training data.
    y_ : np.ndarray
        Stored training labels.
    n_features_ : int
        Number of features in the data.
    neigh_ : NearestNeighbors
        Fitted nearest neighbors model for minority sampling.
        
    References
    ----------
    Uses clustering_selection module for DPC-based majority undersampling.
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        with_replacement: bool = False,
        return_indices: bool = False,
        random_state: Optional[int] = None,
        alpha: float = 0.5,
        beta: float = 0.5
    ) -> None:
        if k_neighbors < 1:
            raise ValueError(f"k_neighbors must be >= 1, got {k_neighbors}")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if not (0.0 <= beta <= 1.0):
            raise ValueError(f"beta must be in [0, 1], got {beta}")
            
        self.k = k_neighbors
        self.return_indices = return_indices
        self.with_replacement = with_replacement
        self.random_state = random_state
        self.alpha = alpha
        self.beta = beta

    def sample_majority(self, n_samples: int) -> np.ndarray:
        """
        Sample majority class using clustering-based selection.
        
        Uses Density Peak Clustering to identify representative majority samples
        that are both dense and distant from minority class.
        
        Parameters
        ----------
        n_samples : int
            Target number of samples (not directly used; determined by clustering).
            
        Returns
        -------
        selected_indices : np.ndarray
            Indices of selected majority class samples.
            
        Notes
        -----
        This method assumes binary classification with labels 0 (majority)
        and 1 (minority).
        """
        if not hasattr(self, 'X_') or not hasattr(self, 'y_'):
            raise ValueError(
                "Sampler not fitted. Call 'fit_majority' first."
            )
        
        X_train = self.X_
        y_train = self.y_
        
        # Convert labels to pandas Series for easier indexing
        y_train_series = pd.Series(y_train)
        X_train = np.ascontiguousarray(X_train, dtype=np.float64)
        
        # Separate majority and minority classes
        majority_mask = (y_train_series == 0)
        minority_mask = (y_train_series == 1)
        
        X_train_majority = X_train[majority_mask]
        X_train_minority = X_train[minority_mask]
        y_train_majority = y_train_series[majority_mask].reset_index(drop=True)
        y_train_minority = y_train_series[minority_mask].reset_index(drop=True)
        
        # Apply clustering-based selection
        (cluster_indices, clusters_density, cluster_distances,
         cluster_instance_densities) = clustering_selection.clustering_dpc(
            X_train_majority, X_train_minority,
            y_train_majority, y_train_minority
        )
        
        # Select samples using density and distance criteria
        _, _, selected_indices = clustering_selection.selection(
            X_train_majority, X_train_minority,
            y_train_majority, y_train_minority,
            cluster_indices, clusters_density, cluster_distances,
            self.alpha, self.beta, cluster_instance_densities
        )
        
        return selected_indices

    def fit_majority(self, X: np.ndarray, y: np.ndarray) -> 'Sampler':
        """
        Fit sampler on the full dataset for majority class sampling.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples,)
            Training labels.
            
        Returns
        -------
        self : Sampler
            Fitted sampler instance.
        """
        self.X_ = X
        self.y_ = y
        self.n_samples_, self.n_features_ = X.shape
        
        return self

    def sample_minority(self, n_samples: int) -> np.ndarray:
        """
        Generate synthetic minority samples using SMOTE.
        
        Parameters
        ----------
        n_samples : int
            Number of synthetic samples to generate.
            
        Returns
        -------
        X_synthetic : np.ndarray of shape (n_samples, n_features)
            Generated synthetic minority samples.
            
        Raises
        ------
        ValueError
            If sampler has not been fitted with fit_minority.
        """
        if not hasattr(self, 'X_'):
            raise ValueError(
                "Sampler not fitted. Call 'fit_minority' first."
            )
        
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        
        # Set random seed for reproducibility
        if self.random_state is not None:
            np.random.seed(self.random_state)

        X_synthetic = np.zeros((n_samples, self.n_features_), dtype=np.float64)
        
        # Generate synthetic samples using SMOTE methodology
        for i in range(n_samples):
            # Randomly select a minority sample
            sample_idx = np.random.randint(0, self.X_.shape[0])
            
            # Find k-nearest neighbors (excluding the sample itself)
            neighbors_idx = self.neigh_.kneighbors(
                self.X_[sample_idx].reshape(1, -1),
                return_distance=False
            )[0, 1:]
            
            # Randomly select one neighbor
            neighbor_idx = np.random.choice(neighbors_idx)
            
            # Compute interpolation
            diff_vector = self.X_[neighbor_idx] - self.X_[sample_idx]
            lambda_interp = np.random.random()
            
            X_synthetic[i, :] = (
                self.X_[sample_idx, :] + lambda_interp * diff_vector
            )
        
        return X_synthetic

    def fit_minority(self, X_minority: np.ndarray) -> 'Sampler':
        """
        Fit sampler on minority class data for SMOTE-based oversampling.
        
        Parameters
        ----------
        X_minority : np.ndarray of shape (n_minority_samples, n_features)
            Minority class training samples.
            
        Returns
        -------
        self : Sampler
            Fitted sampler instance.
            
        Raises
        ------
        ValueError
            If too few minority samples for the specified k_neighbors.
        """
        self.X_ = X_minority
        self.n_minority_samples_, self.n_features_ = X_minority.shape
        
        # Validate sample count
        if self.n_minority_samples_ <= self.k:
            raise ValueError(
                f"Cannot use k_neighbors={self.k} with only "
                f"{self.n_minority_samples_} minority samples. "
                f"Need at least {self.k + 1} samples."
            )

        # Fit nearest neighbors model
        self.neigh_ = NearestNeighbors(n_neighbors=self.k + 1)
        self.neigh_.fit(self.X_)

        return self


class OUBoost(AdaBoostClassifierOUBoost):
    """Implementation of RUSBoost.
    RUSBoost introduces data sampling into the AdaBoost algorithm by
    undersampling the majority class using random undersampling (with or
    without replacement) on each boosting iteration [1].
    This implementation inherits methods from the scikit-learn 
    AdaBoostClassifier class, only modifying the `fit` method.
    Parameters
    ----------
    n_samples : int, optional (default=100)
        Number of new synthetic samples per boosting step.
    min_ratio : float (default=1.0)
        Minimum ratio of majority to minority class samples to generate.
    with_replacement : bool, optional (default=True)
        Undersample with replacement.
    estimator : object, optional (default=DecisionTreeClassifier)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper `classes_`
        and `n_classes_` attributes.
    n_estimators : int, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
    learning_rate : float, optional (default=1.)
        Learning rate shrinks the contribution of each classifier by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.
    algorithm : {'SAMME', 'SAMME.R'}, optional (default='SAMME.R')
        If 'SAMME.R' then use the SAMME.R real boosting algorithm.
        ``estimator`` must support calculation of class probabilities.
        If 'SAMME' then use the SAMME discrete boosting algorithm.
        The SAMME.R algorithm typically converges faster than SAMME,
        achieving a lower test error with fewer boosting iterations.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.
    References
    ----------
    .. [1] C. Seiffert, T. M. Khoshgoftaar, J. V. Hulse, and A. Napolitano.
           "RUSBoost: Improving Classification Performance when Training Data
           is Skewed". International Conference on Pattern Recognition
           (ICPR), 2008.
    """

    def __init__(self,
                 n_samples=100,
                 min_ratio=1.0,
                 with_replacement=False,
                 estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None):

        self.n_samples = n_samples
        self.min_ratio = min_ratio
        self.algorithm = algorithm
        self.ou = Sampler(with_replacement=with_replacement,
                                      return_indices=True,
                                      random_state=random_state)

        super(OUBoost, self).__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

    def fit(self, X, y, sample_weight=None, minority_target=None):
        """Build a boosted classifier/regressor from the training set (X, y),
        performing random undersampling during each boosting step.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR. The dtype is
            forced to DTYPE from tree._tree if the base classifier of this
            ensemble weighted boosting classifier is a tree or forest.
        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.
        minority_target : int
            Minority class label.
        Returns
        -------
        self : object
            Returns self.
        Notes
        -----
        Based on the scikit-learn v0.18 AdaBoostClassifier and
        BaseWeightBoosting `fit` methods.
        """
        # Check that algorithm is supported.
        if self.algorithm not in ('SAMME', 'SAMME.R'):
            raise ValueError("algorithm %s is not supported" % self.algorithm)

        # Check parameters.
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if (self.estimator is None or
                isinstance(self.estimator, (BaseDecisionTree,
                                                 BaseForest))):
            DTYPE = np.float64  # from fast_dict.pxd
            dtype = DTYPE
            accept_sparse = 'csc'
        else:
            dtype = None
            accept_sparse = ['csr', 'csc']

        X, y = check_X_y(X, y, accept_sparse=accept_sparse, dtype=dtype,
                         y_numeric=is_regressor(self))

        if sample_weight is None:
            # Initialize weights to 1 / n_samples.
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
        else:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            # Normalize existing weights.
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive.
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples.")

        if minority_target is None:
            # Determine the minority class label.
            X_org = X
            y_org = y
            sample_weight_org = sample_weight
            
            stats_c_ = Counter(y)
            maj_c_ = max(stats_c_, key=stats_c_.get)
            min_c_ = min(stats_c_, key=stats_c_.get)
            self.minority_target = min_c_
        else:
            self.minority_target = minority_target

        # Check parameters.
        self._validate_estimator()

        # Clear any previous fit results.
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        random_state = check_random_state(self.random_state)
        OX_min = X_org[np.where(y_org == self.minority_target)]

        for iboost in range(self.n_estimators):

            # Random undersampling step.
            X_maj = X_org[np.where(y_org != self.minority_target)]
            X_min = X_org[np.where(y_org == self.minority_target)]
            
            stats_ = Counter(y_org == 1)
           # print(stats_)

            # Compute minority class ratio
            ratio = np.sum(y_org) / y_org.shape[0]
            
            # Apply oversampling and undersampling if minority ratio < 0.5
            if ratio < 0.50:
                # Oversample minority class using SMOTE
                self.ou.fit_minority(X_min)
                X_syn = self.ou.sample_minority(self.n_samples)
                y_syn = np.full(
                    X_syn.shape[0],
                    fill_value=self.minority_target,
                    dtype=np.int64
                )
                
                # Normalize synthetic sample weights
                sample_weight_syn = np.empty(X_syn.shape[0], dtype=np.float64)
                sample_weight_syn[:] = 1. / X_org.shape[0]
                
                # Combine original and synthetic samples
                X_org = np.vstack((X_org, X_syn))
                y_org = np.append(y_org, y_syn)
                
                # Combine weights and normalize
                sample_weight_org = np.append(
                    sample_weight_org, sample_weight_syn
                ).reshape(-1, 1)
                sample_weight_org = np.squeeze(
                    normalize(sample_weight_org, axis=0, norm='l1')
                )

                # Undersample majority class using clustering-based selection
                self.ou.fit_majority(X_org, y_org)
                selected_indices = self.ou.sample_majority(self.n_samples)
                
                # Extract majority and minority samples
                majority_mask = (y_org != self.minority_target)
                minority_mask = (y_org == self.minority_target)
                
                X_maj = X_org[majority_mask]
                y_maj = y_org[majority_mask]
                w_maj = sample_weight_org[majority_mask]
                
                # Select undersampled majority samples
                X_selected_majority = X_maj[selected_indices]
                y_selected_majority = y_maj[selected_indices]
                w_selected_majority = w_maj[selected_indices]
                
                # Get all minority samples
                X_min = X_org[minority_mask]
                y_min = y_org[minority_mask]
                sample_weight_min = sample_weight_org[minority_mask]
                
                # Combine selected majority and minority samples
                X = np.vstack((X_selected_majority, X_min))
                y = np.append(y_selected_majority, y_min)
                
                # Combine and normalize weights
                sample_weight = np.append(
                    w_selected_majority, sample_weight_min
                ).reshape(-1, 1)
                sample_weight = np.squeeze(
                    normalize(sample_weight, axis=0, norm='l1')
                )
             
 
            # Boosting step.
            sample_weight, estimator_weight, estimator_error,sample_weight_org = self._boost(
                iboost,
                X, y,
                sample_weight,
                random_state,X_org,y_org,sample_weight_org)
             
            X = X_org
            y = y_org
            sample_weight = sample_weight_org
            
            # Early termination.
            if sample_weight_org is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero.
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight_org)

            # Stop if the sum of sample weights has become non-positive.
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize.
                sample_weight_org /= sample_weight_sum
               
        return self



class SMOTEBoost(AdaBoostClassifier):
    """Implementation of SMOTEBoost.

    SMOTEBoost introduces data sampling into the AdaBoost algorithm by
    oversampling the minority class using SMOTE on each boosting iteration [1].

    This implementation inherits methods from the scikit-learn 
    AdaBoostClassifier class, only modifying the `fit` method.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        Number of new synthetic samples per boosting step.

    k_neighbors : int, optional (default=5)
        Number of nearest neighbors.

    base_estimator : object, optional (default=DecisionTreeClassifier)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper `classes_`
        and `n_classes_` attributes.

    n_estimators : int, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.

    learning_rate : float, optional (default=1.)
        Learning rate shrinks the contribution of each classifier by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.

    algorithm : {'SAMME', 'SAMME.R'}, optional (default='SAMME.R')
        If 'SAMME.R' then use the SAMME.R real boosting algorithm.
        ``base_estimator`` must support calculation of class probabilities.
        If 'SAMME' then use the SAMME discrete boosting algorithm.
        The SAMME.R algorithm typically converges faster than SAMME,
        achieving a lower test error with fewer boosting iterations.

    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.

    References
    ----------
    .. [1] N. V. Chawla, A. Lazarevic, L. O. Hall, and K. W. Bowyer.
           "SMOTEBoost: Improving Prediction of the Minority Class in
           Boosting." European Conference on Principles of Data Mining and
           Knowledge Discovery (PKDD), 2003.
    """

    def __init__(
        self,
        n_samples=100,
        k_neighbors=5,
        base_estimator=None,
        n_estimators=50,
        learning_rate=1.,
        algorithm="SAMME.R",
        random_state=None,
    ):

        self.n_samples = n_samples
        self.algorithm = algorithm
        self.smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)

        super(SMOTEBoost, self).__init__(
            estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
        )

    def fit(self, X, y, sample_weight=None, minority_target=None):
        """Build a boosted classifier/regressor from the training set (X, y),
        performing SMOTE during each boosting step.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR. The dtype is
            forced to DTYPE from tree._tree if the base classifier of this
            ensemble weighted boosting classifier is a tree or forest.

        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.

        minority_target : int
            Minority class label.

        Returns
        -------
        self : object
            Returns self.

        Notes
        -----
        Based on the scikit-learn v0.18 AdaBoostClassifier and
        BaseWeightBoosting `fit` methods.
        """
        # Check that algorithm is supported.
        if self.algorithm not in ("SAMME", "SAMME.R"):
            raise ValueError("algorithm %s is not supported" % self.algorithm)

        # Check parameters.
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if (self.estimator is None or isinstance(
            self.estimator, (BaseDecisionTree, BaseForest)
        )):
            DTYPE = np.float64  # from fast_dict.pxd
            dtype = DTYPE
            accept_sparse = "csc"
        else:
            dtype = None
            accept_sparse = ["csr", "csc"]

        X, y = check_X_y(
            X,
            y,
            accept_sparse=accept_sparse,
            dtype=dtype,
            y_numeric=is_regressor(self),
        )

        if sample_weight is None:
            # Initialize weights to 1 / n_samples.
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
        else:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            # Normalize existing weights.
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive.
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples."
                )

        if minority_target is None:
            # Determine the minority class label.
            stats_c_ = Counter(y)
            min_c_ = min(stats_c_, key=stats_c_.get)
            self.minority_target = min_c_
        else:
            self.minority_target = minority_target

        # Check parameters.
        self._validate_estimator()

        # Clear any previous fit results.
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        random_state = check_random_state(self.random_state)

        for iboost in range(self.n_estimators):
            X_min = X[np.where(y == self.minority_target)]

            # SMOTE step.
            if len(X_min) >= self.smote.k:
                self.smote.fit(X_min)
                X_syn = self.smote.sample(self.n_samples)
                y_syn = np.full(
                    X_syn.shape[0],
                    fill_value=self.minority_target,
                    dtype=np.int64,
                )

                # Normalize synthetic sample weights based on current training set.
                sample_weight_syn = np.empty(X_syn.shape[0], dtype=np.float64)
                sample_weight_syn[:] = 1. / X.shape[0]

                # Combine the original and synthetic samples.
                X = np.vstack((X, X_syn))
                y = np.append(y, y_syn)

                # Combine the weights.
                sample_weight = np.append(
                    sample_weight, sample_weight_syn
                ).reshape(-1, 1)
                sample_weight = np.squeeze(
                    normalize(sample_weight, axis=0, norm="l1")
                )

                #X, y, sample_weight = shuffle(
                #    X, y, sample_weight, random_state=random_state
                #)

            # Boosting step.
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost,
                X,
                y,
                sample_weight,
                random_state,
            )

            # Early termination.
            if sample_weight is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero.
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive.
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize.
                sample_weight /= sample_weight_sum

        return self
    
class RandomUnderSampler(object):
    """Implementation of random undersampling (RUS).
    Undersample the majority class(es) by randomly picking samples with or
    without replacement.
    Parameters
    ----------
    with_replacement : bool, optional (default=True)
        Undersample with replacement.
    return_indices : bool, optional (default=False)
        Whether or not to return the indices of the samples randomly selected
        from the majority class.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.
    """

    def __init__(self, with_replacement=True, return_indices=False,
                 random_state=None):
        self.return_indices = return_indices
        self.with_replacement = with_replacement
        self.random_state = random_state

    def sample(self, n_samples):
        """Perform undersampling.
        Parameters
        ----------
        n_samples : int
            Number of samples to remove.
        Returns
        -------
        S : array, shape = [n_majority_samples - n_samples, n_features]
            Returns synthetic samples.
        """
        np.random.seed(seed=self.random_state)

        if self.n_majority_samples <= n_samples:
            n_samples = self.n_majority_samples

        idx = np.random.choice(self.n_majority_samples,
                               size=self.n_majority_samples - n_samples,
                               replace=self.with_replacement)

        if self.return_indices:
            return (self.X[idx], idx)
        else:
            return self.X[idx]

    def fit(self, X):
        """Train model based on input data.
        Parameters
        ----------
        X : array-like, shape = [n_majority_samples, n_features]
            Holds the majority samples.
        """
        self.X = X
        self.n_majority_samples, self.n_features = self.X.shape

        return self


class RUSBoost(AdaBoostClassifier):
    """Implementation of RUSBoost.
    RUSBoost introduces data sampling into the AdaBoost algorithm by
    undersampling the majority class using random undersampling (with or
    without replacement) on each boosting iteration [1].
    This implementation inherits methods from the scikit-learn 
    AdaBoostClassifier class, only modifying the `fit` method.
    Parameters
    ----------
    n_samples : int, optional (default=100)
        Number of new synthetic samples per boosting step.
    min_ratio : float (default=1.0)
        Minimum ratio of majority to minority class samples to generate.
    with_replacement : bool, optional (default=True)
        Undersample with replacement.
    estimator : object, optional (default=DecisionTreeClassifier)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper `classes_`
        and `n_classes_` attributes.
    n_estimators : int, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
    learning_rate : float, optional (default=1.)
        Learning rate shrinks the contribution of each classifier by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.
    algorithm : {'SAMME', 'SAMME.R'}, optional (default='SAMME.R')
        If 'SAMME.R' then use the SAMME.R real boosting algorithm.
        ``estimator`` must support calculation of class probabilities.
        If 'SAMME' then use the SAMME discrete boosting algorithm.
        The SAMME.R algorithm typically converges faster than SAMME,
        achieving a lower test error with fewer boosting iterations.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.
    References
    ----------
    .. [1] C. Seiffert, T. M. Khoshgoftaar, J. V. Hulse, and A. Napolitano.
           "RUSBoost: Improving Classification Performance when Training Data
           is Skewed". International Conference on Pattern Recognition
           (ICPR), 2008.
    """

    def __init__(self,
                 n_samples=100,
                 min_ratio=1.0,
                 with_replacement=True,
                 estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None):

        self.n_samples = n_samples
        self.min_ratio = min_ratio
        self.algorithm = algorithm
        self.rus = RandomUnderSampler(with_replacement=with_replacement,
                                      return_indices=True,
                                      random_state=random_state)

        super(RUSBoost, self).__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

    def fit(self, X, y, sample_weight=None, minority_target=None):
        """Build a boosted classifier/regressor from the training set (X, y),
        performing random undersampling during each boosting step.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR. The dtype is
            forced to DTYPE from tree._tree if the base classifier of this
            ensemble weighted boosting classifier is a tree or forest.
        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.
        minority_target : int
            Minority class label.
        Returns
        -------
        self : object
            Returns self.
        Notes
        -----
        Based on the scikit-learn v0.18 AdaBoostClassifier and
        BaseWeightBoosting `fit` methods.
        """
        # Check that algorithm is supported.
        if self.algorithm not in ('SAMME', 'SAMME.R'):
            raise ValueError("algorithm %s is not supported" % self.algorithm)

        # Check parameters.
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if (self.estimator is None or
                isinstance(self.estimator, (BaseDecisionTree,
                                                 BaseForest))):
            DTYPE = np.float64  # from fast_dict.pxd
            dtype = DTYPE
            accept_sparse = 'csc'
        else:
            dtype = None
            accept_sparse = ['csr', 'csc']

        X, y = check_X_y(X, y, accept_sparse=accept_sparse, dtype=dtype,
                         y_numeric=is_regressor(self))

        if sample_weight is None:
            # Initialize weights to 1 / n_samples.
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
        else:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            # Normalize existing weights.
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive.
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples.")

        if minority_target is None:
            # Determine the minority class label.
            stats_c_ = Counter(y)
            maj_c_ = max(stats_c_, key=stats_c_.get)
            min_c_ = min(stats_c_, key=stats_c_.get)
            self.minority_target = min_c_
        else:
            self.minority_target = minority_target

        # Check parameters.
        self._validate_estimator()

        # Clear any previous fit results.
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        random_state = check_random_state(self.random_state)

        for iboost in range(self.n_estimators):
            # Random undersampling step.
            X_maj = X[np.where(y != self.minority_target)]
            X_min = X[np.where(y == self.minority_target)]
            
            self.rus.fit(X_maj)

            n_maj = X_maj.shape[0]
            n_min = X_min.shape[0]
            if n_maj - self.n_samples < int(n_min * self.min_ratio):
                self.n_samples = n_maj - int(n_min * self.min_ratio)
            X_rus, X_idx = self.rus.sample(self.n_samples)

            y_rus = y[np.where(y != self.minority_target)][X_idx]
            y_min = y[np.where(y == self.minority_target)]

            sample_weight_rus = \
                sample_weight[np.where(y != self.minority_target)][X_idx]
          
            sample_weight_min = \
                sample_weight[np.where(y == self.minority_target)]
           

            # Combine the minority and majority class samples.
          
            X = np.vstack((X_rus, X_min))
            y = np.append(y_rus, y_min)
       
            # Combine the weights.
            sample_weight = \
                np.append(sample_weight_rus, sample_weight_min).reshape(-1, 1)
            sample_weight = \
                np.squeeze(normalize(sample_weight, axis=0, norm='l1'))

            # Boosting step.
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost,
                X, y,
                sample_weight,
                random_state)

            # Early termination.
            if sample_weight is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero.
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive.
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize.
                sample_weight /= sample_weight_sum
               
        return self