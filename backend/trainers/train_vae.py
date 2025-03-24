import torch
from torch.utils.data import DataLoader
from backend.models.vae import VAEModel
from backend.util.midi_utils import load_midi_sequences

# Load dataset
sequences = load_midi_sequences("dataset/processed/sequences")
dataloader = DataLoader(sequences, batch_size=32, shuffle=True)

# Initialize model
model = VAEModel(input_dim=128, latent_dim=64)
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
        recon, mu, logvar = model(batch)
        recon_loss = torch.nn.functional.mse_loss(recon, batch)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kld_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")