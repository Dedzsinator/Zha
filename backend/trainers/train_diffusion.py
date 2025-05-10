import torch, os, numpy as np, warnings
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True

# Reuse the MIDIDataset from VAE trainer
from backend.trainers.train_vae import MIDIDataset
from backend.models.diffusion import DiffusionModel

def train_diffusion_model(epochs=50, batch_size=128, learning_rate=2e-4):
    """Diffusion model training with simplified output"""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Training on: {device}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
    
    # Initialize model
    model = DiffusionModel(input_dim=128, hidden_dim=256).to(device)
    print(f"🔄 Model initialized: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler()
    
    # Data loading
    print("📂 Setting up data loader...")
    dataset = MIDIDataset(midi_dir="dataset/midi/")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                          num_workers=4, pin_memory=True)
    
    # Noise schedule
    beta_start = 1e-4
    beta_end = 0.02
    timesteps = 1000
    betas = torch.linspace(beta_start, beta_end, timesteps)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # Training loop
    os.makedirs("output/trained_models", exist_ok=True)
    use_amp = torch.cuda.is_available()
    
    print(f"🚀 Starting diffusion training: {epochs} epochs")
    
    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        model.train()
        
        for batch in progress_bar:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            # Sample random timestep for each item
            t = torch.randint(0, timesteps, (batch.shape[0],), device=device)
            noise = torch.randn_like(batch)
            
            # Create noisy samples based on timestep
            alphas_t = alphas_cumprod[t].view(-1, 1)
            noisy_batch = torch.sqrt(alphas_t) * batch + torch.sqrt(1 - alphas_t) * noise
            
            # Training step with mixed precision if available
            if use_amp:
                with autocast():
                    # Predict noise
                    noise_pred = model(noisy_batch, t/timesteps)
                    loss = torch.nn.functional.mse_loss(noise_pred, noise)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Predict noise
                noise_pred = model(noisy_batch, t/timesteps)
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Update statistics
            epoch_loss += loss.item()
            
            # Update progress
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}"
            })
        
        # End of epoch reporting
        avg_loss = epoch_loss / len(dataloader)
        print(f"📈 Epoch {epoch+1}: Loss={avg_loss:.4f}")
        
        # Save checkpoints
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            torch.save(model.state_dict(), f"output/trained_models/diffusion_ep{epoch+1}.pt")
    
    # Save final model
    torch.save(model.state_dict(), "output/trained_models/trained_diffusion.pt")
    print("✅ Training complete! Model saved to output/trained_models/trained_diffusion.pt")

if __name__ == "__main__":
    train_diffusion_model(epochs=50, batch_size=128)