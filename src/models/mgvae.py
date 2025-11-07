import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims=[128, 128]):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.LeakyReLU(0.2))
            # layers.append(nn.LayerNorm(h))
            last_dim = h
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(last_dim, latent_dim)
        self.fc_logvar = nn.Linear(last_dim, latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

class PriorNet(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims=[128, 128], init_logvar=0.0):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.LeakyReLU(0.2))
            # layers.append(nn.LayerNorm(h))
            last_dim = h
        self.net = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(last_dim, latent_dim)
        self.logvar = nn.Parameter(torch.full((1,), init_logvar))

    def forward(self, x):
        h = self.net(x)
        mu = self.fc_mu(h)
        # # clamp to avoid pathological variance
        logvar = self.logvar.expand_as(mu) 
        # logvar = torch.clamp(self.logvar, min=-1.0, max=2.0).expand_as(mu)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dims=[128, 128]):
        super().__init__()
        layers = []
        last_dim = latent_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.LeakyReLU(0.2))
            # layers.append(nn.LayerNorm(h))
            last_dim = h
        layers.append(nn.Linear(last_dim, output_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        return self.decoder(z)

class MGVAE(nn.Module):
    """
    Majority-Guided VAE: two-stage pretrain (majority) and finetune (minority) with mixture prior and EWC.
    """
    def __init__(self, input_dim, latent_dim=32, hidden_dims=[128,128], device='cpu', majority_subsample=64):
        super().__init__()
        self.device = device
        self.encoder = Encoder(input_dim, latent_dim, hidden_dims).to(device)
        self.prior_net = PriorNet(input_dim, latent_dim, hidden_dims).to(device)
        self.decoder = Decoder(latent_dim, input_dim, hidden_dims).to(device)
        self.latent_dim = latent_dim
        self.majority_subsample = majority_subsample

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_min, x_maj):
        # Encode minority
        mu_q, logvar_q = self.encoder(x_min)
        z = self.reparameterize(mu_q, logvar_q)
        x_recon = self.decoder(z)
        # Subsample majority for mixture prior
        idx = torch.randint(0, x_maj.size(0), (self.majority_subsample,), device=self.device)
        x_sub = x_maj[idx]
        mu_p, logvar_p = self.prior_net(x_sub)
        return x_recon, mu_q, logvar_q, mu_p, logvar_p, z, x_sub

    @staticmethod
    def log_normal(x, mu, logvar):
        var = torch.exp(logvar)
        return -0.5 * ((x - mu)**2 / var + logvar + math.log(2*math.pi))

    def kl_divergence_mixture_prior(self, mu_q, logvar_q, z, mu_p, logvar_p):
        # q(z|x) log density
        log_qz = self.log_normal(z, mu_q, logvar_q).sum(dim=1)
        # mixture prior log p(z)
        batch, dim = z.size()
        S = mu_p.size(0)
        # expand for vectorized mixture
        z_ex = z.unsqueeze(1).expand(batch, S, dim)
        mu_ex = mu_p.unsqueeze(0).expand(batch, S, dim)
        lv_ex = logvar_p.unsqueeze(0).expand(batch, S, dim)
        log_r = self.log_normal(z_ex, mu_ex, lv_ex).sum(dim=2)
        m = torch.logsumexp(log_r, dim=1) - math.log(S)
        return (log_qz - m).mean()

    def loss_function(self, x_min, x_recon, mu_q, logvar_q, mu_p, logvar_p, z, ewc_lambda=0.0, ewc_term=0.0):
        recon = F.mse_loss(x_recon, x_min, reduction='mean')
        kl = self.kl_divergence_mixture_prior(mu_q, logvar_q, z, mu_p, logvar_p)
        loss = recon + kl
        if ewc_lambda > 0:
            loss += ewc_lambda * ewc_term
        return loss, recon, kl

    def pretrain(self, X_maj, epochs=100, batch_size=64, lr=1e-3):
        self.train()
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        ds = torch.utils.data.TensorDataset(X_maj)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
        for _ in range(epochs):
            for (xb,) in loader:
                xb = xb.to(self.device)
                xr, mu_q, lv_q, mu_p, lv_p, z, _ = self.forward(xb, xb)
                loss, _, _ = self.loss_function(xb, xr, mu_q, lv_q, mu_p, lv_p, z)
                opt.zero_grad()
                loss.backward()
                opt.step()

    def compute_fisher(self, X_maj, batch_size=256):
        self.eval()
        fisher = {n: torch.zeros_like(p) for n,p in self.named_parameters()}
        ds = torch.utils.data.TensorDataset(X_maj)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
        for (xb,) in loader:
            xb = xb.to(self.device)
            self.zero_grad()
            xr, mu_q, lv_q, mu_p, lv_p, z, _ = self.forward(xb, xb)
            loss, _, _ = self.loss_function(xb, xr, mu_q, lv_q, mu_p, lv_p, z)
            loss.backward()
            for n, p in self.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data.pow(2) * xb.size(0)
        N = len(ds)
        for n in fisher:
            fisher[n] /= N
        return fisher

    def finetune(self, X_min, X_maj, fisher, pre_params, epochs=100, batch_size=64, lr=1e-3, ewc_lambda=500):
        self.train()
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        ds = torch.utils.data.TensorDataset(X_min)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
        for _ in range(epochs):
            for (xm,) in loader:
                xm = xm.to(self.device)
                xr, mu_q, lv_q, mu_p, lv_p, z, _ = self.forward(xm, X_maj)
                # EWC penalty
                ewc_term = 0.0
                for n,p in self.named_parameters():
                    ewc_term += (fisher[n] * (p - pre_params[n])**2).sum()
                loss, _, _ = self.loss_function(xm, xr, mu_q, lv_q, mu_p, lv_p, z, ewc_lambda, ewc_term)
                opt.zero_grad()
                loss.backward()
                opt.step()

    def fit(self, X_maj, X_min, pretrain_epochs=100, finetune_epochs=100,
            batch_size=64, lr=1e-3, ewc_lambda=500):
        # Stage 1: pretrain on majority
        self.pretrain(X_maj, epochs=pretrain_epochs, batch_size=batch_size, lr=lr)
        # snapshot parameters and fisher
        pre_params = {n: p.detach().clone() for n,p in self.named_parameters()}
        fisher = self.compute_fisher(X_maj, batch_size=batch_size)
        # Stage 2: fine-tune on minority
        self.finetune(X_min, X_maj, fisher, pre_params,
                      epochs=finetune_epochs, batch_size=batch_size, lr=lr,
                      ewc_lambda=ewc_lambda)

    def sample(self, X_maj, n_samples):
        self.eval()
        with torch.no_grad():
            idx = torch.randint(0, X_maj.size(0), (n_samples,), device=self.device)
            x_sub = X_maj[idx]
            mu_p, lv_p = self.prior_net(x_sub)
            z = self.reparameterize(mu_p, lv_p)
            return self.decoder(z).cpu().numpy()
