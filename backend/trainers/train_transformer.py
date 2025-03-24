import torch
from torch.utils.data import DataLoader
from backend.models.transformer import TransformerModel
from backend.util.midi_utils import load_midi_sequences

# Load dataset
sequences = load_midi_sequences("dataset/processed/sequences")
dataloader = DataLoader(sequences, batch_size=32, shuffle=True)

# Initialize model
model = TransformerModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to_device(device)

# Define optimizer
optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-3)

# Training loop
for epoch in range(10):  # Number of epochs
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        output = model.model(batch)
        loss = torch.nn.functional.cross_entropy(output.logits, batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")