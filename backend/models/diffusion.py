import torch
import torch.nn as nn

class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.denoiser = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv1d(128, 1, 3, padding=1))

    def forward(self, x, t):
        noise = torch.randn_like(x)
        noisy_x = x + torch.sqrt(t) * noise
        return self.denoiser(noisy_x)