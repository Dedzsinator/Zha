import torch, os, numpy as np, warnings
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from backend.models.transformer import TransformerModel

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True

# Reuse the MIDIDataset from VAE trainer
from backend.trainers.train_vae import MIDIDataset

def train_transformer_model(epochs=100, batch_size=64, learning_rate=1e-4):
    """Transformer model training with clear output"""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Training on: {device}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
    
    # Initialize model
    model = TransformerModel(
        input_dim=128, 
        embed_dim=512, 
        num_heads=8,
        num_layers=8,
        dim_feedforward=2048
    ).to(device)
    print(f"🔄 Model initialized: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, total_steps=epochs * 1000, 
        pct_start=0.1, anneal_strategy='cos', div_factor=25, final_div_factor=1e4)
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
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        model.train()
        
        for batch in progress_bar:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            # Training step with mixed precision if available
            if use_amp:
                with autocast():
                    output = model(batch)
                    loss = torch.nn.functional.mse_loss(output, batch)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(batch)
                loss = torch.nn.functional.mse_loss(output, batch)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
            
            # Update statistics
            epoch_loss += loss.item()
            scheduler.step()
            
            # Update progress
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.6f}"
            })
        
        # End of epoch reporting
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        print(f"📈 Epoch {epoch+1}: Loss={avg_loss:.4f}, LR={scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoints
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            torch.save(model.state_dict(), f"output/trained_models/transformer_ep{epoch+1}.pt")
    
    # Save final model
    torch.save(model.state_dict(), "output/trained_models/trained_transformer.pt")
    print("✅ Training complete! Model saved to output/trained_models/trained_transformer.pt")

if __name__ == "__main__":
    train_transformer_model(epochs=100, batch_size=64)