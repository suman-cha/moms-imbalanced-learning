import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            norm_layer(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            norm_layer(dim)
        )
    def forward(self, x):
        return x + self.block(x)

class cTMU(nn.Module):
    def __init__(self, latent_dim, class_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + class_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
    def forward(self, z, class_onehot):
        x = torch.cat([z, class_onehot], dim=1)
        return self.net(x)

class IGU(nn.Module):
    def __init__(self, intermediate_dim, n_samples_class):
        super().__init__()
        self.fc_alpha = nn.Linear(intermediate_dim, n_samples_class)
    def forward(self, tvec):
        alpha = torch.softmax(self.fc_alpha(tvec), dim=1)
        return alpha

class ConvexGenerator(nn.Module):
    def __init__(self, latent_dim, n_classes, class_counts, class_dim, hidden_dim, all_minority_X):
        super().__init__()
        self.n_classes = n_classes
        self.class_dim = class_dim
        self.ctmu = cTMU(latent_dim, class_dim, hidden_dim)
        self.igus = nn.ModuleList([
            IGU(hidden_dim, class_counts[c]) for c in range(n_classes)
        ])
        self.class_counts = class_counts
        # Register all class samples (dictionary: class_id -> tensor of samples)
        for c in range(n_classes):
            self.register_buffer(f'X_class_{c}', torch.tensor(all_minority_X[c], dtype=torch.float32))
    def forward(self, z, class_ids):
        class_onehot = F.one_hot(class_ids, num_classes=self.n_classes).float()
        tvec = self.ctmu(z, class_onehot)
        synth_samples = []
        for i, cid in enumerate(class_ids):
            alpha = self.igus[cid](tvec[i:i+1]) # (1, n_samples_class)
            X_min = getattr(self, f'X_class_{cid.item()}') #(n_samples_class, dim)
            synth = torch.sum(alpha.unsqueeze(2) * X_min.unsqueeze(0), dim=1)
            synth_samples.append(synth)
        return torch.cat(synth_samples, dim=0)

class ConditionalDiscriminator(nn.Module):
    def __init__(self, input_dim, n_classes, class_emb_dim=16, hidden_dims=(128,128), num_res_blocks=2):
        super().__init__()
        self.class_emb = nn.Embedding(n_classes, class_emb_dim)
        layers = [nn.Linear(input_dim + class_emb_dim, hidden_dims[0]), nn.LeakyReLU(0.2)]
        current_dim = hidden_dims[0]
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(current_dim, norm_layer=nn.BatchNorm1d))
        for hid_dim in hidden_dims[1:]:
            layers.extend([nn.Linear(current_dim, hid_dim), nn.LeakyReLU(0.2), ResidualBlock(hid_dim, norm_layer=nn.BatchNorm1d)])
            current_dim = hid_dim
        self.net = nn.Sequential(*layers)
        self.output_layer = nn.Linear(current_dim, 1)
    def forward(self, x, class_ids):
        emb = self.class_emb(class_ids)
        xx = torch.cat([x, emb], dim=1)
        return self.output_layer(self.net(xx))

class MultiClassClassifier(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_dims=(128,128), num_res_blocks=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]
        current_dim = hidden_dims[0]
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(current_dim))
        for hid_dim in hidden_dims[1:]:
            layers.extend([nn.Linear(current_dim, hid_dim), nn.ReLU(), ResidualBlock(hid_dim)])
            current_dim = hid_dim
        self.net = nn.Sequential(*layers)
        self.output_layer = nn.Linear(current_dim, n_classes)
    def forward(self, x):
        return self.output_layer(self.net(x))

class GAMOtabularSampler:
    def __init__(self, input_dim, latent_dim, all_minority_X, class_counts, n_classes=2,  
                 class_emb_dim=16, hidden_dim=128, device='cpu'):
        self.n_classes = n_classes
        self.class_counts = class_counts
        self.device = device
        self.G = ConvexGenerator(latent_dim, n_classes, class_counts, n_classes, hidden_dim, all_minority_X).to(device)
        self.D = ConditionalDiscriminator(input_dim, n_classes, class_emb_dim, (hidden_dim,hidden_dim)).to(device)
        self.C = MultiClassClassifier(input_dim, n_classes, (hidden_dim,hidden_dim)).to(device)
        self.trained = False

    def fit(self, class_X_dict, n_epochs=2000, batch_size=128, lr=1e-3, lambda_cls=1.0, lambda_gan=1.0, seed=1203):
        torch.manual_seed(seed)
        np.random.seed(seed)
        class_tensors = {c: torch.tensor(data, dtype=torch.float32).to(self.device) for c, data in class_X_dict.items()}
        optim_G = torch.optim.Adam(self.G.parameters(), lr=lr)
        optim_D = torch.optim.Adam(self.D.parameters(), lr=lr)
        optim_C = torch.optim.Adam(self.C.parameters(), lr=lr)
        mse = nn.MSELoss()  # Least squares loss for D and G
        ls = nn.MSELoss()   # Least squares loss for C

        n_classes = self.n_classes
        for epoch in range(n_epochs):
            # Sample a batch: For each class, draw real and generate fake with class labels for conditionality
            class_sample_counts = {c: class_tensors[c].shape[0] for c in range(n_classes)}
            class_ids = torch.randint(0, n_classes, (batch_size,), device=self.device)
            x_real = torch.stack([
                class_tensors[c.item()][torch.randint(0, class_sample_counts[c.item()], (1,))]
                for c in class_ids
            ], dim=0).squeeze(1)
            # Generate fake
            z = torch.randn(batch_size, self.G.ctmu.net[0].in_features - n_classes, device=self.device)
            x_fake = self.G(z, class_ids)
            # Step D: real=1, fake=0 (LSGAN loss)
            d_real = self.D(x_real, class_ids)
            d_fake = self.D(x_fake.detach(), class_ids)
            d_loss = 0.5 * (mse(d_real, torch.ones_like(d_real)) + mse(d_fake, torch.zeros_like(d_fake)))
            optim_D.zero_grad()
            d_loss.backward()
            optim_D.step()
            # Step C: classifier, all classes, real and generated
            logits_real = self.C(x_real)
            logits_fake = self.C(x_fake.detach())
            y_true = class_ids
            # Real loss
            c_loss_real = ls(logits_real, F.one_hot(y_true, n_classes).float())
            # Fake loss: ignore if desired; include for strong adversarial scenario
            c_loss_fake = ls(logits_fake, F.one_hot(y_true, n_classes).float())
            c_loss = c_loss_real + c_loss_fake
            optim_C.zero_grad()
            c_loss.backward()
            optim_C.step()
            # Step G: adversarial - fool D into assigning 1, fool C into misclassifying toward boundaries
            g_adv_loss = mse(self.D(x_fake, class_ids), torch.ones_like(d_real))
            # Focus generator's samples on classifier boundaries: maximize classifier uncertainty
            logits_g = self.C(x_fake)
            g_cls_loss = -torch.mean(torch.std(F.softmax(logits_g, dim=1), dim=1))  # maximize classifier confusion
            g_loss = lambda_gan * g_adv_loss + lambda_cls * g_cls_loss
            optim_G.zero_grad()
            g_loss.backward()
            optim_G.step()
        self.trained = True

    def sample(self, n_samples, class_id):
        self.G.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.G.ctmu.net[0].in_features - self.n_classes, device=self.device)
            class_ids = torch.full((n_samples,), class_id, dtype=torch.long, device=self.device)
            synth = self.G(z, class_ids)
        return synth.cpu().numpy()
