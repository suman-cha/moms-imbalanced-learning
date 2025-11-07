import torch
import numpy as np
from src.training.moms_train import train_map
from src.utils.moms_utils import set_seed

def transform(
    X_maj,
    X_min,
    in_dim,
    latent_dim=None,
    hidden_dims=None,
    loss_fn=None,
    kernel_type='gaussian',
    median_bw: bool = True,
    loss_params:dict = None,
    device="cpu",
    method="direct",
    n_epochs=100,
    lr=0.001,
    beta=0.01,
    seed=1203,
    residual=True
):
    """
    Apply transformation to selected majority samples.
    """
    # Set random seed for reproducibility
    set_seed(seed)

    n_maj = len(X_maj)
    n_min = len(X_min)
    n_trans = n_maj - n_min
    if n_trans <= 0:
        raise ValueError("The number of majority samples must exceed the number of minority samples for transformation.")

    # Use noise-based or direct transformation method
    if method == "noise":
        lower = np.min(X_maj, axis=0)
        upper = np.max(X_maj, axis=0)
        X_maj = np.random.uniform(low=lower, high=upper, size=(n_trans, in_dim))
    else:
        # direct 방식
        X_maj = X_maj.copy()
    # Train the transformation function using train_map
    f = train_map(
        X_maj=X_maj,
        X_min=X_min,
        in_dim=in_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        loss_fn=loss_fn,
        kernel_type=kernel_type,
        median_bw=median_bw,
        loss_params=loss_params,
        device=device,
        n_epochs=n_epochs,
        lr=lr,
        beta=beta,
        seed=seed,
        residual=residual
    )

    def generate_samples(f, X_sel, n_trans, seed):
        """
        Generate transformed samples using the trained transformation model.
        """
        set_seed(seed)
        f.eval()
        X_trans_list = []
        batch_size = max(2, min(n_min, n_trans))
        for i in range(0, n_trans, batch_size):
            X_b = X_sel[i : i + batch_size]
            X_b = torch.tensor(X_b, dtype=torch.float32).to(device)

            with torch.no_grad():
                _, X_trans_b = f(X_b)
            
            X_trans_list.append(X_trans_b.detach().cpu().numpy())

        return np.vstack(X_trans_list)

    # Generate transformed samples
    X_trans = generate_samples(f, X_maj, n_trans, seed=seed)
    return X_maj, X_min, X_trans
