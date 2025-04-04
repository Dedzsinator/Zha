import torch
import torch.nn as nn
import numpy as np

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=32):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x, t):
        # Process time embedding
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb.unsqueeze(-1).repeat(1, 1, x.shape[-1])

        # First conv block with time embedding
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h + time_emb)

        # Second conv block
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu(h)

        # Residual connection
        return h + self.residual(x)

class DiffusionModel(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, time_emb_dim=32):
        super().__init__()
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Input projection
        self.input_proj = nn.Conv1d(1, hidden_dim, kernel_size=1)

        # Down blocks
        self.down1 = UNetBlock(hidden_dim, hidden_dim, time_emb_dim)
        self.down2 = UNetBlock(hidden_dim, hidden_dim*2, time_emb_dim)

        # Middle blocks
        self.mid1 = UNetBlock(hidden_dim*2, hidden_dim*2, time_emb_dim)
        self.mid2 = UNetBlock(hidden_dim*2, hidden_dim*2, time_emb_dim)

        # Up blocks
        self.up1 = UNetBlock(hidden_dim*4, hidden_dim, time_emb_dim)
        self.up2 = UNetBlock(hidden_dim*2, hidden_dim, time_emb_dim)

        # Output projection
        self.output = nn.Conv1d(hidden_dim, 1, kernel_size=1)

        # Downsampling and upsampling
        self.downsample = nn.AvgPool1d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)

    def forward(self, x, t):
        # x shape: [batch_size, input_dim]
        # Reshape to [batch_size, 1, input_dim] for 1D convolutions
        x = x.unsqueeze(1)

        # Time embedding
        t_emb = self.time_mlp(t.unsqueeze(-1))

        # Initial projection
        h = self.input_proj(x)

        # Downsample blocks with skip connections
        h1 = self.down1(h, t_emb)
        h2 = self.down2(self.downsample(h1), t_emb)

        # Middle blocks
        h = self.mid1(h2, t_emb)
        h = self.mid2(h, t_emb)

        # Upsample blocks with skip connections
        h = self.upsample(h)
        h = torch.cat([h, h2], dim=1)
        h = self.up1(h, t_emb)

        h = self.upsample(h)
        h = torch.cat([h, h1], dim=1)
        h = self.up2(h, t_emb)

        # Output
        h = self.output(h)

        # Return to [batch_size, input_dim] shape
        return h.squeeze(1)

    def sample(self, x_T, steps=100, eta=0.0):
        """
        Sample from the diffusion model

        Args:
            x_T: Starting noise [batch_size, input_dim]
            steps: Number of diffusion steps
            eta: Controls stochasticity (0 is deterministic, 1 is stochastic)
        """
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            x_t = x_T.to(device)

            for t in torch.linspace(1.0, 0.0, steps+1)[:-1]:
                t_batch = torch.ones(x_t.shape[0], device=device) * t

                # Predict clean data
                pred = self.forward(x_t, t_batch)

                # Add noise according to eta
                if t > 0 and eta > 0:
                    sigma = eta * torch.sqrt(t)
                    noise = torch.randn_like(x_t) * sigma
                    x_t = pred + noise
                else:
                    x_t = pred

            return x_t