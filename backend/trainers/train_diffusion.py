import torch
from torch.utils.data import DataLoader
from backend.models.diffusion import DiffusionModel
from backend.util.audio_utils import load_audio_sequences

# Load dataset
sequences = load_audio_sequences("dataset/processed/audio")
dataloader = DataLoader(sequences, batch_size=32, shuffle=True)

# Initialize model
model = DiffusionModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(10):  # Number of epochs
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        t = torch.rand(1)
        denoised = model(batch, t)
        loss = torch.nn.functional.mse_loss(denoised, batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")