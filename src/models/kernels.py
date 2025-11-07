import numpy as np
import torch

# Gaussian kernel
def gaussian_kernel(x1, x2, h=1.0):
    if isinstance(x1, torch.Tensor) or isinstance(x2, torch.Tensor):
        diff = x1 - x2
        dist_sq = torch.sum(diff * diff)
        return torch.exp(-dist_sq / (2 * h ** 2))
    else:
        dist_sq = np.linalg.norm(x1 - x2)**2
        return np.exp(-dist_sq / (2 * h **2))

# Laplacian kernel 
def laplacian_kernel(x1, x2, h=1.0):
    if isinstance(x1, torch.Tensor) or isinstance(x2, torch.Tensor):
        diff = x1 - x2
        # L1 norm along all dimensions
        dist = torch.sum(torch.abs(diff))
        return torch.exp(-dist/h)
    else:
        dist = np.linalg.norm(x1 - x2, ord=1)
        return np.exp(-dist/h)

# IMQ kernel
def imq_kernel(x1, x2, c=1.0, alpha=0.5):
    if isinstance(x1, torch.Tensor) or isinstance(x2, torch.Tensor):
        diff = x1 - x2
        dist_sq = torch.sum(diff * diff)
        return 1.0 / ((dist_sq + c**2)**alpha)
    else:
        dist_sq = np.linalg.norm(x1-x2)**2
        return 1.0 / ((dist_sq + c**2)**alpha)

# RQ kernel
def rq_kernel(x1, x2, h=1.0, alpha=1.0):
    if isinstance(x1, torch.Tensor) or isinstance(x2, torch.Tensor):
        diff = x1 - x2
        dist_sq = torch.sum(diff * diff)
        return (1 + dist_sq / (2 * alpha * h**2)) ** (-alpha)
    else:
        dist_sq = np.linalg.norm(x1-x2)**2
        return (1 + dist_sq / (2 * alpha * h**2)) ** (-alpha)

def compute_kernel_matrix(x, y=None, kernel_type='gaussian', **kwargs):
    """
    Computes the kernel matrix for samples in x and optionally between x and y using the specified kernel.
    
    Parameters:
        x: ndarray, shape (n_samples_X, d)
            Samples from the first distribution.
        y: ndarray, shape (n_samples_Y, d), optional
            Samples from the second distribution. If None, compute the kernel matrix for x.
        kernel_type: str, default='gaussian'
            Type of kernel to compute. Options include:
            'gaussian', 'laplacian', 'inverse_multiquadratic', 'rational_quadratic'.
        **kwargs: dict
            Additional keyword arguments for the chosen kernel:
                - For 'gaussian' and 'rational_quadratic': h (bandwidth), default=1.0.
                - For 'laplacian': h (bandwidth), default=1.0.
                - For 'inverse_multiquadratic': c (constant, default=1.0), alpha (exponent, default=0.5).
                - For 'rational_quadratic': alpha (default=1.0).
    
    Returns:
        kernel_matrix: ndarray, shape (n_samples_X, n_samples_Y)
            Computed kernel matrix.
    """

    if y is None:
        y = x

    use_torch = isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor)
    if use_torch:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)

        if kernel_type in ['gaussian', 'imq', 'rq']:
            x_norm = torch.sum(x**2, dim=1, keepdim=True)
            y_norm = torch.sum(y**2, dim=1, keepdim=True).t()
            dists_sq = x_norm + y_norm - 2 * torch.mm(x, y.t())
        
        if kernel_type == 'gaussian':
            h = kwargs.get('h', 1.0)
            if isinstance(h, torch.Tensor):
                h = h.item()
            return torch.exp(-dists_sq/(2*h**2))
        elif kernel_type == 'laplacian':
            h = kwargs.get('h', 1.0)
            if isinstance(h, torch.Tensor):
                h = h.item()
            dists = torch.sum(torch.abs(x[:, None, :] - y[None, :, :]), dim=2)
            return torch.exp(-dists/h)
        elif kernel_type == 'imq':
            c = kwargs.get('c', 1.0)
            alpha = kwargs.get('alpha', 0.5)
            if isinstance(c, torch.Tensor):
                c = c.item()
            if isinstance(alpha, torch.Tensor):
                alpha = alpha.item()
            return 1.0/((dists_sq + c**2)**alpha)
        elif kernel_type == 'rq':
            h = kwargs.get('h', 1.0)
            alpha = kwargs.get('alpha', 1.0)
            if isinstance(h, torch.Tensor):
                h = h.item()
            if isinstance(alpha, torch.Tensor):
                alpha = alpha.item()
            return (1+dists_sq / (2*alpha*h**2))**(-alpha)
        else:
            raise ValueError("Unknown kernel_type: {}".format(kernel_type))
    else:
        if kernel_type in ['gaussian', 'imq', 'rq']:
            x_norm = np.sum(x**2, axis=1).reshape(-1, 1)
            y_norm = np.sum(y**2, axis=1).reshape(1, -1)
            dists_sq = x_norm + y_norm - 2 * np.dot(x, y.T)
        
        if kernel_type == 'gaussian':
            h = kwargs.get('h', 1.0)
            if isinstance(h, torch.Tensor):
                h = float(h.item())
            return np.exp(-dists_sq/(2*h**2))
        elif kernel_type == 'laplacian':
            h = kwargs.get('h', 1.0)
            if isinstance(h, torch.Tensor):
                h = float(h.item())
            dists = np.sum(np.abs(x[:, None, :] - y[None, :, :]), axis=2)
            return np.exp(-dists/h)
        elif kernel_type == 'imq':
            c = kwargs.get('c', 1.0)
            alpha = kwargs.get('alpha', 0.5)
            if isinstance(c, torch.Tensor):
                c = float(c.item())
            if isinstance(alpha, torch.Tensor):
                alpha = float(alpha.item())
            return 1.0/((dists_sq + c**2)**alpha)
        elif kernel_type == 'rq':
            h = kwargs.get('h', 1.0)
            alpha = kwargs.get('alpha', 1.0)
            if isinstance(h, torch.Tensor):
                h = float(h.item())
            if isinstance(alpha, torch.Tensor):
                alpha = float(alpha.item())
            return (1+dists_sq / (2*alpha*h**2))**(-alpha)
        else:
            raise ValueError("Unknown kernel_type: {}".format(kernel_type))
