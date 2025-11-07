import numpy as np
import torch
from src.models.kernels import *
from src.utils.moms_utils import set_seed

# Maximum Mean Discrepancy (MMD)
def MMD_est_torch(x, y, kernel_type='gaussian', **kwargs):
    """
    Computes the Maximum Mean Discrepancy (MMD) between samples x and y
    using the specified kernel via a vectorized kernel trick.
    
    Parameters:
        x: Tensor, shape (n1, d)
            Samples from the first distribution.
        y: Tensor, shape (n2, d)
            Samples from the second distribution.
        kernel_type: str, default='gaussian'
            Type of kernel to use. Options: 'gaussian', 'laplacian', 'imq', 'rq'.
        **kwargs:
            Additional parameters for the chosen kernel:
              - For 'gaussian' and 'rq': h (bandwidth), default=1.0.
              - For 'laplacian': h (bandwidth), default=1.0.
              - For 'imq': c (constant, default=1.0), alpha (exponent, default=0.5).
              - For 'rq': alpha (default=1.0).
              
    Returns:
        mmd: Tensor
            The computed MMD value as a torch.Tensor.
    """
    n1 = x.shape[0]
    n2 = y.shape[0]
    
    # Compute kernel matrix
    Kxx = compute_kernel_matrix(x, kernel_type=kernel_type, **kwargs)
    Kyy = compute_kernel_matrix(y, kernel_type=kernel_type, **kwargs)
    Kxy = compute_kernel_matrix(x, y, kernel_type=kernel_type, **kwargs)

    # Compute the biased MMD estimator
    mmd_value = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
    return mmd_value

def adaptive_kernel_width(X, Y=None):
    if Y is None:
        Y = X
    if isinstance(X, torch.Tensor) or isinstance(Y, torch.Tensor):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
        if not isinstance(Y, torch.Tensor):
            Y = torch.tensor(Y)
        
        dists = torch.cdist(X, Y, p=2)
        bandwidth = torch.median(dists)
        return bandwidth
    else:
        dists = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2)
        bandwidth = np.median(dists)
        return bandwidth

def coverage_regularization(X_trans, X_minority, beta=0.001, gamma=0.001):
    """
    Compute the coverage regularization loss with an additional penalty for variance alignment.

    Parameters:
        X_trans: torch.Tensor
            Transformed Majority samples (after transformation).
        X_minority: torch.Tensor
            Minority samples.
        beta: float
            Weight for the coverage regularization term.
        gamma: float
            Weight for the variance alignment term.

    Returns:
        torch.Tensor
            The combined coverage and variance regularization loss.
    """
    dist_matrix = torch.cdist(X_trans, X_minority)  # Shape: (num_majority, num_minority)
    min_distances = torch.min(dist_matrix, dim=0).values  # Shape: (num_minority,)
    coverage_loss = torch.sum(min_distances ** 2)

    variance_trans = torch.var(X_trans, dim=0)  # Variance of transformed samples
    variance_minority = torch.var(X_minority, dim=0)  # Variance of minority samples
    variance_loss = torch.sqrt(torch.sum((variance_trans - variance_minority) ** 2))

    total_loss = beta * coverage_loss + gamma * variance_loss

    return total_loss

def boundary_loss(X_min, X_trans, X_maj, k=5, batch_size=500):
    """
    Efficiently compute a regularization term inspired by Borderline SMOTE to minimize distance to the DANGER set.

    Parameters:
        X_trans, X_min, X_maj: torch.Tensor
        beta, delta: float
        k: int
        batch_size: int

    Returns:
        torch.Tensor
    """
    def batched_cdist(X1, X2, batch_size):
        """
        Compute pairwise distances between two sets of vectors in a batched manner without using torch.cdist.

        Parameters:
            X1: torch.Tensor
                Tensor of shape (n, d), where n is the number of vectors and d is the dimension.
            X2: torch.Tensor
                Tensor of shape (m, d), where m is the number of vectors and d is the dimension.
            batch_size: int
                The size of each batch for computing distances.

        Returns:
            distances: torch.Tensor
                Pairwise distance matrix of shape (n, m).
        """
        n, m = X1.shape[0], X2.shape[0]
        distances = torch.zeros(n, m, device=X1.device)

        # Compute distances in batches
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            X1_batch = X1[i:end]  # Batch from X1
            # Pairwise distance: sqrt((x1 - x2)^2) for all pairs
            distances[i:end] = torch.sqrt(
                torch.sum((X1_batch[:, None, :] - X2[None, :, :]) ** 2, dim=-1)
            )

        return distances

    # Combine X_min and X_maj
    X_all = torch.cat([X_min, X_maj], dim=0)
    from sklearn.neighbors import NearestNeighbors

    # Convert tensors to NumPy arrays for NearestNeighbors
    X_min_np = X_min.cpu().numpy()
    X_all_np = X_all.cpu().numpy()

    # Use NearestNeighbors to find the k nearest neighbors
    nn = NearestNeighbors(n_neighbors=k+1, algorithm='auto', n_jobs=-1)
    nn.fit(X_all_np)
    _, nn_idx = nn.kneighbors(X_min_np)
    nn_idx = torch.tensor(nn_idx, device=X_min.device)[:, 1:]

    # Identify DANGER and refined DANGER sets
    majority_mask = nn_idx >= X_min.shape[0]
    majority_counts = majority_mask.sum(dim=1)

    danger_mask = majority_counts > (k // 2)  # DANGER set: > k/2 majority neighbors
    all_majority_mask = majority_mask.all(dim=1)  # NOISE set

    borderline_danger_set = X_min[danger_mask & ~all_majority_mask]

    # Proximity to Borderline DANGER Set
    if borderline_danger_set.shape[0] > 0:
        dist_trans_danger = batched_cdist(X_trans, borderline_danger_set, batch_size)
        prox_loss = torch.mean(dist_trans_danger.min(dim=1).values)
    else: 
        dist_trans_min = batched_cdist(X_trans, X_min, batch_size)
        prox_loss = torch.mean(dist_trans_min.min(dim=1).values)
    
    return prox_loss

def compute_danger_set(X_min, X_maj, seed, k=5):
    set_seed(seed)
    from sklearn.neighbors import NearestNeighbors
    # X_all = torch.cat([X_min, X_maj], dim=0)
    # X_min_np = X_min.cpu().numpy()
    # X_all_np = X_all.cpu().numpy()
    X_min_np = X_min.detach().cpu().numpy()
    X_all = torch.cat([X_min, X_maj], dim=0)
    X_all_np = X_all.detach().cpu().numpy()
    
    nn = NearestNeighbors(n_neighbors=k+1, algorithm='auto', n_jobs=-1)
    nn.fit(X_all_np)
    _, nn_idx = nn.kneighbors(X_min_np)
    nn_idx = torch.tensor(nn_idx, device=X_min.device)[:, 1:]

    maj_mask = nn_idx >= X_min.shape[0]
    maj_cnt = maj_mask.sum(dim=1)

    danger_mask = maj_cnt > (k//2)
    all_maj_mask = maj_mask.all(dim=1)

    danger_set = X_min[danger_mask & ~all_maj_mask]
    return danger_set

def compute_border_maj(X_min, X_maj):
    """Find majority samples closest to minority class"""
    n_neg = len(X_min)
    with torch.no_grad():
        dists = torch.cdist(X_maj, X_min)
        min_dists = dists.min(dim=1).values
        _, idx = torch.topk(min_dists, k=min(n_neg, len(X_maj)), largest=False)
    return X_maj[idx]

def compute_safe_maj(X_maj, X_min, q=0.75):
    min_dists = torch.cdist(X_maj, X_min).min(dim=1).values
    safe_thresh = torch.quantile(min_dists, q=q)
    safe_mask = min_dists > safe_thresh
    return X_maj[safe_mask]

def local_triplet_loss(danger_min, X_trans, safe_maj, margin=1.0, alpha=0.3):
    """triplet loss focusing on danger set"""
    pos_dist = torch.cdist(X_trans, danger_min).min(dim=1).values
    neg_dist = torch.cdist(X_trans, safe_maj).min(dim=1).values

    if danger_min.size(0) > 1:
        intra_dist = torch.pdist(danger_min).mean()
        margin_adj = margin + alpha * intra_dist
    else:
        margin_adj = margin
    
    return torch.relu(pos_dist - neg_dist + margin_adj).mean() 

# def local_triplet_loss(latent_min, latent_trans, latent_maj, margin=1.0, k=5):
#     """Density-aware triplet loss between transformed samples and minority/majority samples"""
#     eps = 1e-8  # small constant to avoid division by zero
#     B_min = latent_min.size(0)

#     # Step 1: Compute local density (average distance to k neighbors in latent_min)
#     with torch.no_grad():
#         if B_min <= 1:
#             # Edge case: only one point in danger set
#             weights = torch.ones(1, device=latent_min.device)
#         else:
#             k_eff = min(k, B_min - 1)
#             min_to_min_dist = torch.cdist(latent_min, latent_min, p=2)
#             # mask self-distance
#             min_to_min_dist += torch.eye(B_min, device=latent_min.device) * 1e6
#             density = min_to_min_dist.topk(k_eff, largest=False, dim=1).values.mean(dim=1)  # [B_min]
#             weights = 1.0 / (density + eps)  # inverse density => sparse = high weight
#             weights = weights / (weights.sum() + eps)  # normalize to sum to 1

#     # Step 2: Compute positive distances (transformed -> danger minority set)
#     pos_dist = torch.cdist(latent_trans, latent_min, p=2)  # [B_trans, B_min]
#     weighted_pos = (pos_dist * weights.unsqueeze(0)).sum(dim=1)  # weighted sum per transformed point
#     pull_loss = weighted_pos.mean()

#     # Step 3: Compute negative distances (transformed -> majority)
#     neg_dist = torch.cdist(latent_trans, latent_maj, p=2).min(dim=1).values
#     push_loss = torch.relu(margin - neg_dist).mean()

#     return pull_loss + push_loss


# 단일 커널 MMD 함수 (옵션: 기본 sigma=0.01)
def MMD_deep_kernel(x, y, sigma=0.01):
    def pairwise_distances(A, B):
        A_sq = (A ** 2).sum(1).view(-1, 1)
        B_sq = (B ** 2).sum(1).view(1, -1)
        dist = A_sq + B_sq - 2.0 * torch.mm(A, B.t())
        return torch.clamp(dist, min=0.0)
    
    dist_xx = pairwise_distances(x, x)
    dist_yy = pairwise_distances(y, y)
    dist_xy = pairwise_distances(x, y)
    
    Kxx = torch.exp(-dist_xx / (sigma ** 2))
    Kyy = torch.exp(-dist_yy / (sigma ** 2))
    Kxy = torch.exp(-dist_xy / (sigma ** 2))
    
    mmd = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
    return mmd
