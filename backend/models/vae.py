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
    def __init__(self, input_dim=128, latent_dim=128, beta=1.0):
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
        
        # Beta parameter for KL divergence weighting
        self.beta = beta

    def _init_weights(self):
        """Apply better weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode(self, x):
        """Optimized encoder forward pass"""
        # Combined operations for more efficiency
        x = self.encoder_res1(F.silu(self.encoder_norm1(self.encoder_input(x)), inplace=True))
        x = self.encoder_res2(F.silu(self.encoder_norm2(self.encoder_hidden(x)), inplace=True))
        return self.encoder_output(x)
    
    def decode(self, z):
        """Optimized decoder forward pass"""
        # Combined operations for more efficiency
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
            # During inference just use the mean for deterministic output
            return mu
    
    def forward(self, x, temperature=1.0):
        """Optimized combined forward pass with temperature control"""
        # Encode and get latent parameters
        mu_logvar = self.encode(x)
        mu, logvar = torch.chunk(mu_logvar, 2, dim=-1)
        
        # Reparameterize with temperature
        z = self.reparameterize(mu, logvar, temperature)
        
        # Decode
        recon_x = self.decode(z)
        
        return recon_x, mu, logvar
    
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

    def export_to_onnx(self, filepath, input_shape=(1, 128)):
        """Export model to ONNX format for efficient inference"""
        dummy_input = torch.randn(*input_shape)
        torch.onnx.export(
            self, 
            dummy_input, 
            filepath,
            input_names=['input'],
            output_names=['reconstruction', 'mu', 'logvar'],
            dynamic_axes={'input': {0: 'batch_size'}}
        )
        return filepath