import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import script

class ResidualBlock(nn.Module):
    """Residual block with normalization for better gradient flow"""
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.activation = nn.SiLU(inplace=True)  # SiLU/Swish is more efficient than ReLU in practice
    
    def forward(self, x):
        return x + self.activation(self.norm(self.linear(x)))

class VAEModel(nn.Module):
    def __init__(self, input_dim=128, latent_dim=128):
        super().__init__()
        # Encoder with normalization and residual connections
        self.encoder_input = nn.Linear(input_dim, 512)
        self.encoder_norm1 = nn.LayerNorm(512)
        self.encoder_res1 = ResidualBlock(512)
        
        self.encoder_hidden = nn.Linear(512, 256)
        self.encoder_norm2 = nn.LayerNorm(256) 
        self.encoder_res2 = ResidualBlock(256)
        
        self.encoder_output = nn.Linear(256, latent_dim * 2)
        
        # Decoder with similar optimizations
        self.decoder_input = nn.Linear(latent_dim, 256)
        self.decoder_norm1 = nn.LayerNorm(256)
        self.decoder_res1 = ResidualBlock(256)
        
        self.decoder_hidden = nn.Linear(256, 512)
        self.decoder_norm2 = nn.LayerNorm(512)
        self.decoder_res2 = ResidualBlock(512)
        
        self.decoder_output = nn.Linear(512, input_dim)
        
        # Initialize weights for faster convergence 
        self._init_weights()

    def _init_weights(self):
        """Apply better weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode(self, x):
        """Optimized encoder forward pass"""
        x = F.silu(self.encoder_norm1(self.encoder_input(x)))
        x = self.encoder_res1(x)
        x = F.silu(self.encoder_norm2(self.encoder_hidden(x)))
        x = self.encoder_res2(x)
        return self.encoder_output(x)
    
    def decode(self, z):
        """Optimized decoder forward pass"""
        z = F.silu(self.decoder_norm1(self.decoder_input(z)))
        z = self.decoder_res1(z)
        z = F.silu(self.decoder_norm2(self.decoder_hidden(z)))
        z = self.decoder_res2(z)
        return torch.sigmoid(self.decoder_output(z))
    
    def reparameterize(self, mu, logvar):
        """Optimized reparameterization trick"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # During inference just use the mean for deterministic output
            return mu
    
    def forward(self, x):
        """Optimized combined forward pass"""
        # Encode and get latent parameters
        mu_logvar = self.encode(x)
        mu, logvar = torch.chunk(mu_logvar, 2, dim=-1)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        recon_x = self.decode(z)
        
        return recon_x, mu, logvar