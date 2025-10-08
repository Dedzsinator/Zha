import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

class ResidualBlock(nn.Module):
    """Residual block with normalization for better gradient flow"""
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.activation = nn.SiLU(inplace=True)
    
    def forward(self, x):
        return x + self.activation(self.norm(self.linear(x)))


class GOLC_VAE(nn.Module):
    """
    Group-Orbital Latent Consistency Variational Autoencoder (GOLC-VAE)
    
    This enhanced VAE implements group-orbital consistency to ensure that
    musical transformations (e.g., transpositions) map to the same latent
    representation, promoting invariance to musical symmetries.
    
    Key innovations:
    1. Orbit consistency loss - ensures Enc(g·x) ≈ Enc(x) for all g ∈ G
    2. Canonical representation averaging across group orbits
    3. Enhanced posterior stability across transformations
    """
    
    def __init__(self, 
                 input_dim=128, 
                 latent_dim=128, 
                 beta=1.0,
                 beta_orbit=0.5,
                 transposition_range=6):
        """
        Args:
            input_dim: Input feature dimension (MIDI note range)
            latent_dim: Latent space dimension
            beta: Weight for KL divergence term (β-VAE)
            beta_orbit: Weight for orbital consistency loss
            transposition_range: Range of transpositions to consider (±semitones)
        """
        super().__init__()
        
        # Encoder architecture (same as baseline for fair comparison)
        self.encoder_input = nn.Linear(input_dim, 512)
        self.encoder_norm1 = nn.LayerNorm(512)
        self.encoder_res1 = ResidualBlock(512)
        
        self.encoder_hidden = nn.Linear(512, 256)
        self.encoder_norm2 = nn.LayerNorm(256) 
        self.encoder_res2 = ResidualBlock(256)
        
        self.encoder_output = nn.Linear(256, latent_dim * 2)
        
        # Decoder architecture (same as baseline for fair comparison)
        self.decoder_input = nn.Linear(latent_dim, 256)
        self.decoder_norm1 = nn.LayerNorm(256)
        self.decoder_res1 = ResidualBlock(256)
        
        self.decoder_hidden = nn.Linear(256, 512)
        self.decoder_norm2 = nn.LayerNorm(512)
        self.decoder_res2 = ResidualBlock(512)
        
        self.decoder_output = nn.Linear(512, input_dim)
        
        # Initialize weights
        self._init_weights()
        
        # GOLC-specific parameters
        self.beta = beta  # β-VAE weight
        self.beta_orbit = beta_orbit  # Orbital consistency weight
        self.transposition_group = list(range(-transposition_range, transposition_range + 1))
        
        # Track orbital consistency during training
        self.orbit_distances = []

    def _init_weights(self):
        """Apply better weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode(self, x):
        """Encoder forward pass - maps input to latent parameters"""
        x = self.encoder_res1(F.silu(self.encoder_norm1(self.encoder_input(x)), inplace=True))
        x = self.encoder_res2(F.silu(self.encoder_norm2(self.encoder_hidden(x)), inplace=True))
        return self.encoder_output(x)
    
    def decode(self, z):
        """Decoder forward pass - reconstructs input from latent"""
        z = self.decoder_res1(F.silu(self.decoder_norm1(self.decoder_input(z)), inplace=True))
        z = self.decoder_res2(F.silu(self.decoder_norm2(self.decoder_hidden(z)), inplace=True))
        return torch.sigmoid(self.decoder_output(z))
    
    def reparameterize(self, mu, logvar, temperature=1.0):
        """Reparameterization trick with temperature control"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std * temperature
        else:
            return mu
    
    def transpose_sample(self, x: torch.Tensor, semitones: int) -> torch.Tensor:
        """
        Transpose a pitch distribution by a given number of semitones
        
        Mathematical operation: T_k(x)_i = x_{i-k} for valid indices
        
        Args:
            x: Input pitch distribution [batch_size, 128]
            semitones: Number of semitones to transpose
            
        Returns:
            Transposed distribution [batch_size, 128]
        """
        batch_size, input_dim = x.shape
        transposed = torch.zeros_like(x)
        
        for i in range(input_dim):
            new_idx = i + semitones
            if 0 <= new_idx < input_dim:
                transposed[:, new_idx] = x[:, i]
        
        return transposed
    
    def compute_canonical_representation(self, x: torch.Tensor, 
                                        sample_transformations: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute canonical (orbit-averaged) latent representation
        
        Mathematical formulation:
        z_c(x) = (1/|G|) * Σ_{g∈G} Enc_φ(g·x)
        
        Args:
            x: Input samples [batch_size, input_dim]
            sample_transformations: If True, sample subset of group; if False, use all
            
        Returns:
            Tuple of (canonical_mu, canonical_logvar)
        """
        # Encode original sample
        mu_logvar_original = self.encode(x)
        mu_original, logvar_original = torch.chunk(mu_logvar_original, 2, dim=-1)
        
        # Initialize accumulators
        mu_sum = mu_original.clone()
        logvar_sum = logvar_original.clone()
        count = 1
        
        # Sample transformations if requested (for efficiency during training)
        if sample_transformations and len(self.transposition_group) > 5:
            # Sample 3-5 random transformations
            import random
            sampled_group = random.sample([g for g in self.transposition_group if g != 0], 
                                         min(4, len(self.transposition_group) - 1))
        else:
            sampled_group = [g for g in self.transposition_group if g != 0]
        
        # Accumulate representations across orbit
        for transpose_amt in sampled_group:
            # Apply transformation
            x_transformed = self.transpose_sample(x, transpose_amt)
            
            # Encode transformed sample
            mu_logvar_trans = self.encode(x_transformed)
            mu_trans, logvar_trans = torch.chunk(mu_logvar_trans, 2, dim=-1)
            
            # Accumulate
            mu_sum = mu_sum + mu_trans
            logvar_sum = logvar_sum + logvar_trans
            count += 1
        
        # Average to get canonical representation
        canonical_mu = mu_sum / count
        canonical_logvar = logvar_sum / count
        
        return canonical_mu, canonical_logvar
    
    def orbital_consistency_loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute orbital consistency loss
        
        Mathematical formulation:
        L_orbit = (1/|G|) * Σ_{g∈G} ||Enc_φ(g·x) - z_c(x)||²
        
        Args:
            x: Input samples [batch_size, input_dim]
            
        Returns:
            Tuple of (loss_value, metrics_dict)
        """
        # Compute canonical representation
        canonical_mu, canonical_logvar = self.compute_canonical_representation(x, sample_transformations=True)
        
        # Encode original
        mu_logvar_original = self.encode(x)
        mu_original, logvar_original = torch.chunk(mu_logvar_original, 2, dim=-1)
        
        # Compute distance from original to canonical
        orbit_loss = F.mse_loss(mu_original, canonical_mu.detach(), reduction='mean')
        orbit_loss += F.mse_loss(logvar_original, canonical_logvar.detach(), reduction='mean')
        
        # Sample a few transformations and measure consistency
        import random
        sampled_transforms = random.sample([g for g in self.transposition_group if g != 0], 
                                          min(2, len(self.transposition_group) - 1))
        
        for transpose_amt in sampled_transforms:
            x_trans = self.transpose_sample(x, transpose_amt)
            mu_logvar_trans = self.encode(x_trans)
            mu_trans, _ = torch.chunk(mu_logvar_trans, 2, dim=-1)
            
            # Add to orbit loss
            orbit_loss += F.mse_loss(mu_trans, canonical_mu.detach(), reduction='mean')
        
        # Track orbital distance for monitoring
        with torch.no_grad():
            orbit_distance = torch.norm(mu_original - canonical_mu, p=2, dim=1).mean().item()
            self.orbit_distances.append(orbit_distance)
        
        metrics = {
            'orbit_distance': orbit_distance,
            'orbit_loss': orbit_loss.item()
        }
        
        return orbit_loss, metrics
    
    def forward(self, x, temperature=1.0, compute_orbit_loss=True):
        """
        Combined forward pass with optional orbit consistency
        
        Args:
            x: Input samples [batch_size, input_dim]
            temperature: Sampling temperature
            compute_orbit_loss: Whether to compute orbital consistency loss
            
        Returns:
            Tuple of (reconstruction, mu, logvar, orbit_loss, orbit_metrics)
        """
        # Standard VAE forward pass
        mu_logvar = self.encode(x)
        mu, logvar = torch.chunk(mu_logvar, 2, dim=-1)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar, temperature)
        
        # Decode
        recon_x = self.decode(z)
        
        # Compute orbital consistency loss if requested
        if compute_orbit_loss and self.training:
            orbit_loss, orbit_metrics = self.orbital_consistency_loss(x)
        else:
            orbit_loss = torch.tensor(0.0, device=x.device)
            orbit_metrics = {}
        
        return recon_x, mu, logvar, orbit_loss, orbit_metrics
    
    def loss_function(self, recon_x, x, mu, logvar, orbit_loss):
        """
        Complete GOLC-VAE loss function
        
        Mathematical formulation:
        L_total = L_recon + β_KL * L_KL + β_orbit * L_orbit
        
        Where:
        - L_recon = BCE(x, recon_x) - reconstruction loss
        - L_KL = KL(q(z|x) || p(z)) - KL divergence
        - L_orbit = orbital consistency loss
        
        Args:
            recon_x: Reconstructed samples
            x: Original samples
            mu: Latent mean
            logvar: Latent log variance
            orbit_loss: Orbital consistency loss
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Reconstruction loss (BCE for normalized inputs)
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.size(0)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        
        # Total loss with GOLC regularization
        total_loss = recon_loss + self.beta * kl_loss + self.beta_orbit * orbit_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'orbit_loss': orbit_loss.item() if isinstance(orbit_loss, torch.Tensor) else orbit_loss,
            'beta_kl': (self.beta * kl_loss).item(),
            'beta_orbit': (self.beta_orbit * orbit_loss).item() if isinstance(orbit_loss, torch.Tensor) else 0.0
        }
        
        return total_loss, loss_dict
    
    def sample(self, num_samples=1, temperature=0.8, device='cuda'):
        """Generate samples from the latent space with temperature control"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.decoder_input.in_features, device=device) * temperature
            return self.decode(z)
    
    def interpolate(self, x1, x2, steps=10):
        """Generate a smooth transition between two inputs"""
        with torch.no_grad():
            # Encode both inputs
            mu1, _ = torch.chunk(self.encode(x1), 2, dim=-1)
            mu2, _ = torch.chunk(self.encode(x2), 2, dim=-1)
            
            # Create interpolation steps
            alphas = torch.linspace(0, 1, steps, device=mu1.device)
            z_interp = torch.stack([(1-a)*mu1 + a*mu2 for a in alphas])
            
            # Decode interpolated latent vectors
            return self.decode(z_interp)
    
    def get_orbit_statistics(self) -> dict:
        """Get statistics about orbital consistency during training"""
        if not self.orbit_distances:
            return {}
        
        import numpy as np
        distances = np.array(self.orbit_distances)
        
        return {
            'mean_orbit_distance': float(np.mean(distances)),
            'std_orbit_distance': float(np.std(distances)),
            'min_orbit_distance': float(np.min(distances)),
            'max_orbit_distance': float(np.max(distances)),
            'recent_orbit_distance': float(distances[-1]) if len(distances) > 0 else 0.0
        }
