import torch
import os
import numpy as np
import warnings
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from backend.models.vae import VAEModel

# Enable faster training when input sizes don't change
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Filter warnings
warnings.filterwarnings("ignore", category=UserWarning)

class MIDIDataset(Dataset):
    def __init__(self, midi_dir, transform=None, max_files=None):
        self.midi_dir = midi_dir
        self.file_list = []
        self.data_cache = {}  # Cache for preprocessed data
        self.use_cache = True  # Enable caching for speed

        # Walk through all subdirectories to find MIDI files
        for root, _, files in os.walk(midi_dir):
            for file in files:
                if file.endswith('.mid'):
                    # Store relative path to the file
                    rel_path = os.path.join(os.path.relpath(root, midi_dir), file)
                    self.file_list.append(rel_path)
                    # Limit files for faster testing if needed
                    if max_files and len(self.file_list) >= max_files:
                        break

        self.transform = transform
        print(f"Found {len(self.file_list)} MIDI files")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if idx in self.data_cache and self.use_cache:
            return self.data_cache[idx]

        midi_path = os.path.join(self.midi_dir, self.file_list[idx])

        # Load MIDI file and convert to feature vectors
        from music21 import converter

        try:
            # Parse the MIDI file
            score = converter.parse(midi_path)

            # Extract notes and create a one-hot encoding
            notes = []
            for element in score.flatten():  # Use flatten() instead of flat
                if hasattr(element, 'pitch'):
                    notes.append(element.pitch.midi)

            # Create a simple feature vector (pitch histogram)
            feature = np.zeros(128, dtype=np.float32)
            for note in notes:
                if 0 <= note < 128:
                    feature[note] += 1

            # Normalize
            if np.sum(feature) > 0:
                feature = feature / np.sum(feature)

            tensor = torch.from_numpy(feature).float()

        except Exception as e:
            print(f"Error processing {midi_path}: {e}")
            tensor = torch.zeros(128, dtype=torch.float32)

        if self.transform:
            tensor = self.transform(tensor)

        # Cache the processed tensor
        if self.use_cache:
            self.data_cache[idx] = tensor

        return tensor

def setup_data_loading(batch_size=128, num_workers=4):
    """Configure optimized data loading with larger batch sizes"""

    # Load dataset with potential file limit for testing
    dataset = MIDIDataset(midi_dir="dataset/midi/", max_files=None)

    # GPU-optimized DataLoader with much larger batch size
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Faster transfer to GPU
        prefetch_factor=2,
        persistent_workers=True  # Keep workers alive between epochs
    )

    return dataloader

def train_vae_model(epochs=100, batch_size=128, learning_rate=2e-4, latent_dim=128):
    """Main training function with GPU optimizations"""

    # Set up device with CUDA memory optimizations
    if torch.cuda.is_available():
        # Set GPU memory strategy for better utilization
        torch.cuda.set_per_process_memory_fraction(0.9)  # Use up to 90% of GPU memory
        torch.cuda.empty_cache()
        device = torch.device("cuda")
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("No GPU available, training on CPU")

    # Initialize model with larger dimensions
    input_dim = 128
    model = VAEModel(input_dim=input_dim, latent_dim=latent_dim)
    model.to(device)

    # Use parallel processing if multiple GPUs available
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        print(f"Using {gpu_count} GPUs for training")
        model = torch.nn.DataParallel(model)

    # Try to compile model with torch.compile if available
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            print("Using torch.compile() for model acceleration")
        except Exception as e:
            print(f"torch.compile() not supported: {e}")

    # Optimizer with higher learning rate and weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=learning_rate * 0.01
    )

    # Initialize gradient scaler for mixed precision
    scaler = GradScaler()

    # Set up data loading with optimized parameters
    dataloader = setup_data_loading(batch_size=batch_size, num_workers=4)

    # Create directory for saving models
    os.makedirs("trained_models", exist_ok=True)

    # Training loop with optimized GPU memory usage
    losses = []

    print(f"Training VAE model on {device}")
    print(f"Batch size: {batch_size}, Epochs: {epochs}, Learning rate: {learning_rate}")

    # Check PyTorch version for autocast compatibility
    use_amp = torch.cuda.is_available()  # Only use AMP with CUDA
    print(f"Using Automatic Mixed Precision (AMP): {use_amp}")

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        model.train()

        for batch in progress_bar:
            batch = batch.to(device, non_blocking=True)  # non_blocking for async transfer
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

            # Mixed precision forward pass - compatibility fix for older PyTorch versions
            if use_amp:
                with autocast():  # No device_type parameter for compatibility
                    # Forward pass
                    recon, mu, logvar = model(batch)

                    # Calculate loss components
                    recon_loss = torch.nn.functional.mse_loss(recon, batch)
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch.size(0)

                    # Combined loss with beta weighting for KL divergence
                    beta = min(1.0, 0.2 + epoch/50)  # Gradually increase KL weight for better training
                    loss = recon_loss + beta * kl_loss

                # Use scaler for mixed precision training
                scaler.scale(loss).backward()

                # Gradient clipping to prevent explosion
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision training if AMP not available
                # Forward pass
                recon, mu, logvar = model(batch)

                # Calculate loss components
                recon_loss = torch.nn.functional.mse_loss(recon, batch)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch.size(0)

                # Combined loss with beta weighting for KL divergence
                beta = min(1.0, 0.2 + epoch/50)  # Gradually increase KL weight for better training
                loss = recon_loss + beta * kl_loss

                # Standard backprop
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Track losses
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()

            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.6f}",
                "recon": f"{recon_loss.item():.6f}",
                "kl": f"{kl_loss.item():.6f}"
            })

        # Update learning rate
        scheduler.step()

        # Calculate average losses
        avg_loss = epoch_loss / len(dataloader)
        avg_recon_loss = epoch_recon_loss / len(dataloader)
        avg_kl_loss = epoch_kl_loss / len(dataloader)
        losses.append(avg_loss)

        # Print current GPU memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f} (Recon: {avg_recon_loss:.6f}, KL: {avg_kl_loss:.6f}), LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint periodically
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            # Save model state without DataParallel wrapper if used
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), f"trained_models/vae_ep{epoch+1}.pt")
            else:
                torch.save(model.state_dict(), f"trained_models/vae_ep{epoch+1}.pt")

    # Save the final model
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), "trained_models/trained_vae.pt")
    else:
        torch.save(model.state_dict(), "trained_models/trained_vae.pt")

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("VAE Model Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("vae_training_loss.png")
    print("Training complete!")

if __name__ == "__main__":
    # You can adjust these parameters based on your GPU capabilities
    train_vae_model(epochs=100, batch_size=128, learning_rate=2e-4, latent_dim=128)