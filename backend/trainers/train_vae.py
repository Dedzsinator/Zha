import torch, os, numpy as np, warnings, matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from backend.models.vae import VAEModel

warnings.filterwarnings("ignore", category=UserWarning)
torch.backends.cudnn.benchmark = True

class MIDIDataset(Dataset):
    def __init__(self, midi_dir, max_files=None):
        self.midi_dir = midi_dir
        self.file_list = []
        self.data_cache = {}
        
        # Find MIDI files in directory structure
        print("🔍 Scanning for MIDI files...")
        for root, _, files in os.walk(midi_dir):
            for file in files:
                if file.endswith('.mid'):
                    self.file_list.append(os.path.join(os.path.relpath(root, midi_dir), file))
                    if max_files and len(self.file_list) >= max_files:
                        break
        print(f"✓ Found {len(self.file_list)} MIDI files")

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # Return from cache if available
        if idx in self.data_cache:
            return self.data_cache[idx]

        # Process MIDI file to feature vector
        midi_path = os.path.join(self.midi_dir, self.file_list[idx])
        try:
            from music21 import converter
            score = converter.parse(midi_path)
            
            # Extract notes and create pitch histogram
            notes = [element.pitch.midi for element in score.flatten() 
                     if hasattr(element, 'pitch')]
            
            # Create feature vector
            feature = np.zeros(128, dtype=np.float32)
            for note in notes:
                if 0 <= note < 128:
                    feature[note] += 1
            
            # Normalize
            if np.sum(feature) > 0:
                feature = feature / np.sum(feature)
            
            tensor = torch.from_numpy(feature).float()
            self.data_cache[idx] = tensor
            return tensor
            
        except Exception as e:
            print(f"⚠️ Error processing {midi_path}: {e}")
            return torch.zeros(128, dtype=torch.float32)

def train_vae_model(epochs=100, batch_size=128, learning_rate=2e-4, latent_dim=128):
    """VAE model training with simplified output reporting"""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Training on: {device}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
    
    # Initialize model
    model = VAEModel(input_dim=128, latent_dim=latent_dim).to(device)
    print(f"🔄 Model initialized: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate * 0.01)
    scaler = GradScaler()
    
    # Data loading
    print("📂 Setting up data loader...")
    dataset = MIDIDataset(midi_dir="dataset/midi/")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                          num_workers=4, pin_memory=True)
    
    # Training loop
    os.makedirs("output/trained_models", exist_ok=True)
    losses = []
    
    print(f"🚀 Starting training: {epochs} epochs")
    use_amp = torch.cuda.is_available()
    
    for epoch in range(epochs):
        epoch_loss = recon_loss_sum = kl_loss_sum = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        model.train()
        
        for batch in progress_bar:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            # Training step with mixed precision if available
            if use_amp:
                with autocast():
                    recon, mu, logvar = model(batch)
                    recon_loss = torch.nn.functional.mse_loss(recon, batch)
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch.size(0)
                    beta = min(1.0, 0.2 + epoch/50)
                    loss = recon_loss + beta * kl_loss
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                recon, mu, logvar = model(batch)
                recon_loss = torch.nn.functional.mse_loss(recon, batch)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch.size(0)
                beta = min(1.0, 0.2 + epoch/50)
                loss = recon_loss + beta * kl_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Update statistics
            epoch_loss += loss.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
            
            # Update progress
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "recon": f"{recon_loss.item():.4f}",
                "kl": f"{kl_loss.item():.4f}"
            })
        
        # End of epoch reporting
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        avg_recon = recon_loss_sum / len(dataloader)
        avg_kl = kl_loss_sum / len(dataloader)
        losses.append(avg_loss)
        
        print(f"📈 Epoch {epoch+1}: Loss={avg_loss:.4f} (Recon={avg_recon:.4f}, KL={avg_kl:.4f}), LR={scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoints
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            torch.save(model.state_dict(), f"output/trained_models/vae_ep{epoch+1}.pt")
    
    # Save final model
    torch.save(model.state_dict(), "output/trained_models/trained_vae.pt")
    print("✅ Training complete! Model saved to output/trained_models/trained_vae.pt")

if __name__ == "__main__":
    train_vae_model(epochs=100, batch_size=128)