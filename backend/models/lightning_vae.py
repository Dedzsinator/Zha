import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import List, Optional, Tuple
from .golc_vae import GOLC_VAE

class LightningGOLCVAE(pl.LightningModule):
    """
    PyTorch Lightning module for the Group-Orbital Latent Consistency VAE (GOLC-VAE).
    
    This module handles:
    1. Training loop with orbital consistency loss
    2. Validation and monitoring
    3. Optimization configuration
    4. Automatic logging
    """
    
    def __init__(self, 
                 input_dim: int = 128, 
                 latent_dim: int = 128, 
                 beta: float = 1.0,
                 beta_orbit: float = 0.5,
                 transposition_range: int = 6,
                 learning_rate: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = GOLC_VAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            beta=beta,
            beta_orbit=beta_orbit,
            transposition_range=transposition_range
        )
        
        self.learning_rate = learning_rate
        self.transposition_range = transposition_range

    def forward(self, x):
        return self.model.encode(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

    def _compute_loss(self, batch):
        """
        Compute GOLC-VAE loss terms:
        1. Reconstruction loss
        2. KL Divergence (beta-VAE)
        3. Orbital Consistency Loss (GOLC)
        """
        x = batch
        
        # 1. Standard VAE pass
        # Encode
        mu_logvar = self.model.encode(x)
        mu, logvar = torch.chunk(mu_logvar, 2, dim=-1)
        
        # Reparameterize
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # Decode
        recon_x = self.model.decode(z)
        
        # Reconstruction Loss (BCE or MSE)
        # Assuming input is normalized [0, 1] or binary-like for MIDI piano roll
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.size(0)
        
        # KL Divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        
        # 2. Orbital Consistency Loss
        # Generate orbit: apply transpositions to input x
        # We simulate transpositions by rolling the input tensor along the pitch dimension
        # Assuming x shape is (Batch, Time, Pitch) or (Batch, Pitch)
        # If (Batch, Pitch), we roll along dim 1.
        
        orbit_loss = 0.0
        if self.hparams.beta_orbit > 0:
            orbit_encodings = []
            
            # Calculate encodings for the orbit (including original)
            # We sample a subset of the group to save memory if range is large
            shifts = list(range(-self.transposition_range, self.transposition_range + 1))
            
            # For efficiency, we can process the orbit in a batch if memory allows, 
            # or loop. Looping is safer for memory.
            
            # Original encoding (g=0)
            orbit_encodings.append(mu) # Use mean as the representation
            
            for shift in shifts:
                if shift == 0: continue
                
                # Apply transposition (roll)
                # Handle edge cases where roll wraps around: mask out wrapped notes?
                # For simplicity, we just roll. In MIDI, wrapping low to high is bad, 
                # but for small shifts it's a reasonable approximation or we should zero-pad.
                # Better: shift and zero-fill.
                
                x_shifted = self._shift_input(x, shift)
                
                # Encode shifted input
                mu_shifted_logvar = self.model.encode(x_shifted)
                mu_shifted, _ = torch.chunk(mu_shifted_logvar, 2, dim=-1)
                
                orbit_encodings.append(mu_shifted)
            
            # Stack encodings: (GroupSize, Batch, LatentDim)
            orbit_stack = torch.stack(orbit_encodings)
            
            # Calculate Canonical Representation (Mean of orbit)
            z_canonical = torch.mean(orbit_stack, dim=0)
            
            # Calculate Orbit Loss: MSE between each orbit element and canonical
            # Sum over group, mean over batch
            orbit_loss = torch.mean(torch.sum((orbit_stack - z_canonical.unsqueeze(0)) ** 2, dim=2))
            
        
        # Total Loss
        total_loss = recon_loss + (self.hparams.beta * kl_loss) + (self.hparams.beta_orbit * orbit_loss)
        
        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "orbit_loss": orbit_loss
        }

    def _shift_input(self, x, shift):
        """
        Shift input x by `shift` positions along the feature dimension.
        Zero-fill the empty space (do not wrap).
        """
        # Assuming x is (Batch, Features)
        result = torch.zeros_like(x)
        if shift > 0:
            result[:, shift:] = x[:, :-shift]
        elif shift < 0:
            result[:, :shift] = x[:, -shift:]
        else:
            result = x
        return result

    def training_step(self, batch, batch_idx):
        losses = self._compute_loss(batch)
        self.log("train_loss", losses["loss"], prog_bar=True)
        self.log("train_recon_loss", losses["recon_loss"])
        self.log("train_kl_loss", losses["kl_loss"])
        self.log("train_orbit_loss", losses["orbit_loss"])
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        losses = self._compute_loss(batch)
        self.log("val_loss", losses["loss"], prog_bar=True)
        self.log("val_recon_loss", losses["recon_loss"])
        self.log("val_kl_loss", losses["kl_loss"])
        self.log("val_orbit_loss", losses["orbit_loss"])
        return losses["loss"]
