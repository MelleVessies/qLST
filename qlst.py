from argparse import ArgumentParser

import torch
import torch.nn as nn

import pytorch_lightning as pl

class Attention1D(nn.Module):
    """Attention mechanism.
    Parameters
    ----------
    dim : int
        The input and out dimension of per token features.
    n_heads : int
        Number of attention heads.
    qkv_bias : bool
        If True then we include bias to the query, key and value projections.
    attn_p : float
        Dropout probability applied to the query, key and value tensors.
    proj_p : float
        Dropout probability applied to the output tensor.
    Attributes
    ----------
    scale : float
        Normalizing consant for the dot product.
    qkv : nn.Linear
        Linear projection for the query, key and value.
    proj : nn.Linear
        Linear mapping that takes in the concatenated output of all attention
        heads and maps it into a new space.
    attn_drop, proj_drop : nn.Dropout
        Dropout layers.
    """
    def __init__(self, dim, n_heads=16, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * n_heads * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim * n_heads, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        """Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        """
        n_samples, dim = x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)  # (n_samples, 3 * dim)

        qkv = qkv.reshape(n_samples, 3, self.n_heads, self.head_dim)  # (n_smaples, 3, n_heads, head_dim)

        qkv = qkv.permute(1, 0, 2, 3)  # (3, n_samples, n_heads, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)  # (n_samples, head_dim, n_heads)
        dp = (q @ k_t) * self.scale  # (n_samples, n_heads, n_heads)
        attn = dp.softmax(dim=-1)  # (n_samples, n_heads, n_heads)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v  # (n_samples, n_heads, head_dim)
        weighted_avg = weighted_avg.transpose(1, 2)  # (n_samples, head_dim, n_heads)
        weighted_avg = weighted_avg.flatten(1)  # (n_samples, dim)

        x = self.proj(weighted_avg)  # (n_samples, dim)
        x = self.proj_drop(x)  # (n_samples, dim)

        return x



class qLST(pl.LightningModule):
    def __init__(
        self,
        classification_model: pl.LightningModule,
        vae: pl.LightningModule,
        query_idx : int,
        lr : float = 1e-4,
        **kwargs
    ):
        super(qLST, self).__init__()

        self.query_idx = query_idx

        self.lr = lr

        self.latent_dim = vae.model.latent_dim
        self.delta_weight = 0.25

        self.classification_model = classification_model
        self.vae = vae

        self.classification_model.requires_grad_(False)
        self.vae.requires_grad_(False)

        self.encoder = self.vae.model.encoder
        self.encoder.requires_grad_(False)
        self.decoder = self.vae.model.decoder
        self.decoder.requires_grad_(True)

        self.num_classes = classification_model.num_classes

        self.exerator = nn.Sequential(*[
            Attention1D(self.latent_dim + self.num_classes + 1, 5, attn_p=0.1),
            nn.Linear(self.latent_dim + self.num_classes + 1, self.latent_dim)
        ])

    def forward(self, x, q):
        mu, log_var = self.encoder(x)
        z = mu

        z_query = torch.cat((z, q), dim=1)

        z_delta = self.exerator(z_query)
        z_e_recon = self.decoder(z + z_delta)
        z_e_class = self.classification_model(z_e_recon)

        return z, z_delta, z_e_recon, z_e_class

    def _run_step(self, x, q):
        mu, log_var = self.encoder(x)
        z = mu

        z_query = torch.cat((z, q), dim=1)

        z_delta = self.exerator(z_query)
        z_e_recon = self.decoder(z + z_delta)
        z_e_class = self.classification_model(z_e_recon)

        return z, z_delta, z_e_recon, z_e_class

    def step(self, batch, batch_idx):
        x = batch['waveform']

        self.classification_model.eval()
        self.vae.eval()
        self.encoder.eval()

        # Run classification
        q_orig = self.classification_model(x).sigmoid()
        
        # Create random queries
        q = torch.rand(q_orig[:, self.query_idx].shape).to(x.device)

        # Calculate query diff for loss and concatenate query and classifier output
        q_diff = (q_orig[:, self.query_idx] - q).abs()
        q_orig = torch.cat((q_orig, q.unsqueeze(-1)), dim=1) 

        z, z_delta, z_e_recon, z_e_class = self._run_step(x, q_orig)

        classification_loss = torch.functional.F.binary_cross_entropy_with_logits(z_e_class[:, self.query_idx], q, reduction='none')
        delta_loss = torch.functional.F.mse_loss(x, z_e_recon, reduction='none').flatten(start_dim=1).sum(dim=1)

        weighted_delta_loss = delta_loss * (1 - q_diff + 0.01) * self.delta_weight
        loss = (classification_loss + weighted_delta_loss).mean()

        logs = {
            "classification_loss": classification_loss.mean(),
            "delta_loss": delta_loss.mean(),
            "weighted_delta_loss": weighted_delta_loss.mean(),
            "delta_size (mean)": abs(z_delta).sum(dim=-1).mean(),
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict(
            {f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        logs = {f"val_{k}": v for k, v in logs.items()}

        self.log_dict(logs)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.exerator.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--lr", type=float, default=1e-6)

        parser.add_argument("--batch_size", type=int, default=512)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--data_dir", type=str, default=".")

        return parser