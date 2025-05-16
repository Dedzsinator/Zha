import torch, os, numpy as np, warnings
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from backend.models.transformer import TransformerModel
from backend.trainers.utils import train_epoch, LRSchedulerWithBatchOption, EarlyStopping
from backend.trainers.train_vae import MIDIDataset  # Reuse the dataset class

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True

def train_transformer_model(epochs=100, batch_size=64, learning_rate=1e-4, 
                           grad_accum_steps=1, patience=10):
    """Transformer model training with improved efficiency"""
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
        optimizer, max_lr=learning_rate, total_steps=epochs * 1000 // grad_accum_steps, 
        pct_start=0.1, anneal_strategy='cos', div_factor=25, final_div_factor=1e4)
    scheduler_wrapper = LRSchedulerWithBatchOption(scheduler, step_every_batch=True)
    scaler = GradScaler() if torch.cuda.is_available() else None
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience)
    
    # Data loading with memory-efficient caching
    print("📂 Setting up data loader...")
    dataset = MIDIDataset(midi_dir="dataset/midi/", cache_size=500)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                          num_workers=4, pin_memory=True)
    
    # Training loop
    os.makedirs("output/trained_models", exist_ok=True)
    all_metrics = []
    
    print(f"🚀 Starting training: {epochs} epochs (gradient accumulation: {grad_accum_steps} steps)")
    
    for epoch in range(epochs):
        # Train one epoch using the common training utility
        metrics = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler_wrapper,
            device=device,
            scaler=scaler,
            accumulation_steps=grad_accum_steps,
            epoch_idx=epoch
        )
        
        # Store metrics
        all_metrics.append(metrics)
        
        # Report metrics
        print(f"📈 Epoch {epoch+1}: Loss={metrics['loss']:.4f}, LR={scheduler.get_last_lr()[0]:.6f}")
        
        # Early stopping check
        if early_stopping(metrics['loss']):
            print(f"⚠️ Early stopping triggered after {epoch+1} epochs")
            break
        
        # Save checkpoints
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            torch.save(model.state_dict(), f"output/trained_models/transformer_ep{epoch+1}.pt")
    
    # Save final model and JIT scripted version for faster inference
    torch.save(model.state_dict(), "output/trained_models/trained_transformer.pt")
    
    # Create a JIT compiled version for faster inference
    try:
        scripted_model = torch.jit.script(model.cpu())
        scripted_model.save("output/trained_models/trained_transformer_jit.pt")
        print("✅ JIT compiled model saved for faster inference")
    except Exception as e:
        print(f"⚠️ JIT compilation failed: {e}")
    
    print("✅ Training complete! Model saved to output/trained_models/trained_transformer.pt")
    return model, all_metrics

if __name__ == "__main__":
    train_transformer_model(
        epochs=100, 
        batch_size=64,
        grad_accum_steps=4,  # Accumulate gradients for larger effective batch size
        patience=15          # Early stopping patience
    )