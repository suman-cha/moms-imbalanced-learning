import torch
from torch import nn
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from src.models.moms_losses import *
from src.utils.moms_utils import set_seed


class TransMap(nn.Module):
    def __init__(self, input_dim, latent_dim=None, hidden_dims=None, 
                 use_batchnorm=True, use_layernorm=False, use_residual=True):
        super(TransMap, self).__init__()

        if latent_dim is None:
            latent_dim = max(1, input_dim // 2)
        
        if hidden_dims is None:
            hidden_dims = [input_dim * 2 if input_dim > 1 else input_dim]
        
        self.act = nn.GELU()
        self.use_ln = use_layernorm
        self.use_bn = use_batchnorm and not use_layernorm

        self.encoder_layers = nn.ModuleList()
        prev_dim = input_dim 
        for hid_dim in hidden_dims:
            self.encoder_layers.append(nn.Linear(prev_dim, hid_dim))
            prev_dim = hid_dim 
        
        self.fc_latent = nn.Linear(prev_dim, latent_dim)
        self.enc_norms = nn.ModuleList()
        for hid_dim in hidden_dims:
            if self.use_bn:
                self.enc_norms.append(nn.BatchNorm1d(hid_dim))
            elif self.use_ln:
                self.enc_norms.append(nn.LayerNorm(hid_dim))
            else:
                self.enc_norms.append(nn.Identity())

        if self.use_bn:
            self.latent_norm = nn.BatchNorm1d(latent_dim)
        elif self.use_ln:
            self.latent_norm = nn.LayerNorm(latent_dim)
        else:
            self.latent_norm = nn.Identity()
        
        self.decoder_layers = nn.ModuleList()
        self.dec_norms = nn.ModuleList() 
        prev_dim = latent_dim
        for hid_dim in reversed(hidden_dims):
            self.decoder_layers.append(nn.Linear(prev_dim, hid_dim))
            if self.use_bn:
                self.dec_norms.append(nn.BatchNorm1d(hid_dim))
            elif self.use_ln:
                self.dec_norms.append(nn.LayerNorm(hid_dim))
            else:
                self.dec_norms.append(nn.Identity())
            prev_dim = hid_dim            
        
        self.output_layer = nn.Linear(prev_dim, input_dim)
        self.use_residual = use_residual

    def forward(self, x):
        enc_outs = []
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            x = self.enc_norms[i](x)
            x = self.act(x) 
            enc_outs.append(x)
        x = self.fc_latent(x)
        x = self.latent_norm(x) 
        latent = x

        dec_x = latent
        for i, layer in enumerate(self.decoder_layers):
            dec_x = layer(dec_x)
            dec_x = self.dec_norms[i](dec_x)
            if self.use_residual and i < len(enc_outs):
                dec_x = dec_x + enc_outs[-1-i]
            dec_x = self.act(dec_x)
        out = self.output_layer(dec_x)
        return latent, out

def train_map(X_maj: torch.Tensor, 
              X_min: torch.Tensor, 
              in_dim: int, 
              latent_dim: int = None, 
              hidden_dims: list = None, 
              loss_fn=None,
              kernel_type: str = 'gaussian', 
              device: str = 'cuda', 
              n_epochs: int = 1000, 
              lr: float = 1e-3, 
              beta: float = 0.01, 
              k: int = 5,
              seed: int = 1203, 
              residual=True,
              median_bw: bool = True,
              loss_params: dict = None,
              triplet_margin: float = 1.0,
              triplet_alpha: float = 0.3) -> nn.Module:
    """
    Train the TransMap model to transform majority samples into minority-like samples by minimizing MMD,
    and introduce reg_loss (local_triplet_loss) only after a warm-up period to preserve global structure.
    
    Returns:
        TransMap: Trained TransMap model.
    """
    set_seed(seed)
    X_maj = torch.tensor(X_maj, dtype=torch.float32).to(device)
    X_min = torch.tensor(X_min, dtype=torch.float32).to(device)

    n_maj = X_maj.size(0)
    n_min = X_min.size(0)
    n_trans = n_maj - n_min

    perm = torch.randperm(n_maj)
    X_trans_init = X_maj[perm[:n_trans]].clone()

    model = TransMap(input_dim=in_dim, latent_dim=latent_dim, hidden_dims=hidden_dims, 
                     use_layernorm=False, use_residual=residual).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    
    if loss_params is None:
        loss_params = {}
    if loss_fn is None:
        loss_fn = MMD_est_torch
    
    trans_dataset = TensorDataset(X_trans_init)
    min_dataset = TensorDataset(X_min)
    trans_loader = DataLoader(trans_dataset, batch_size=len(trans_dataset), shuffle=True, drop_last=True)
    min_loader = DataLoader(min_dataset, batch_size=len(min_dataset), shuffle=True, drop_last=True) 

    # Training loop
    for epoch in range(n_epochs):
        model.train()  
        epoch_loss = 0.0
        epoch_reg_loss = 0.0

        for trans_batch, min_batch in zip(trans_loader, min_loader):
            b_trans = trans_batch[0].to(device)
            b_min = min_batch[0].to(device)
            
            _, X_trans = model(b_trans)

            # Adaptive kernel width (global structure)
            sigma = adaptive_kernel_width(X_trans, b_min) if median_bw else 1.0
            
            # Prepare loss_params with kernel bandwidth
            loss_params_with_h = (loss_params or {}).copy()
            if 'h' not in loss_params_with_h:
                loss_params_with_h['h'] = sigma
            
            mmd_loss = loss_fn(X_trans, b_min, kernel_type=kernel_type, **loss_params_with_h)
            
            if beta == 0.0:
                loss = mmd_loss
                reg_loss = torch.tensor(0.0, device=device)
            else:
                danger_set = compute_danger_set(X_min, X_maj, seed, k)
                if danger_set.size(0) == 0:
                   loss = mmd_loss
                   reg_loss = torch.tensor(0.0, device=device)
                else:
                    safe_maj = compute_safe_maj(X_maj, X_min)
                    # Use margin and alpha from parameters (with fallback to loss_params)
                    margin_val = loss_params.get('triplet_margin', triplet_margin) if loss_params else triplet_margin
                    alpha_val = loss_params.get('triplet_alpha', triplet_alpha) if loss_params else triplet_alpha
                    triplet_loss = local_triplet_loss(danger_set, X_trans, safe_maj, margin=margin_val, alpha=alpha_val)
                    reg_loss = beta * triplet_loss
            
            loss = mmd_loss + beta * reg_loss
            epoch_reg_loss += (beta * reg_loss).item()


            # reg_loss = boundary_loss(X_min=b_min, X_trans=X_trans,
            #                                 X_maj=X_maj)
            # loss = mmd_loss + beta * reg_loss
            # epoch_reg_loss += beta * reg_loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            epoch_loss += mmd_loss.item()
            
        scheduler.step()
        # 필요시 중간 결과 출력
        # if (epoch+1) % 100 == 0:
        #     print(f"Epoch [{epoch+1}/{n_epochs}]: MMD Loss = {epoch_loss:.4f}, Reg Loss = {epoch_reg_loss:.4f}")

    return model
