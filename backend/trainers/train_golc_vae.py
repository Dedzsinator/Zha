import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

from backend.models.golc_vae import GOLC_VAE


class GOLC_VAE_Trainer:
    """Trainer for Group-Orbital Latent Consistency VAE"""
    
    def __init__(self, 
                 model: GOLC_VAE,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 learning_rate: float = 1e-3,
                 device: str = 'cuda',
                 save_dir: str = 'output/trained_models'):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer with weight decay for better generalization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-5
        )
        
        # Learning rate scheduler - reduce on plateau
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_recon': [],
            'val_recon': [],
            'train_kl': [],
            'val_kl': [],
            'train_orbit': [],
            'val_orbit': [],
            'learning_rate': [],
            'orbit_distance': []
        }
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stop_patience = 30
        
    def train_epoch(self, temperature=1.0):
        """Train for one epoch with optional temperature annealing"""
        self.model.train()
        
        epoch_losses = {
            'total': 0.0,
            'recon': 0.0,
            'kl': 0.0,
            'orbit': 0.0
        }
        
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        for batch_idx, (data,) in enumerate(pbar):
            data = data.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with orbital consistency
            recon_x, mu, logvar, orbit_loss, orbit_metrics = self.model(
                data, 
                temperature=temperature,
                compute_orbit_loss=True
            )
            
            # Compute loss
            total_loss, loss_dict = self.model.loss_function(
                recon_x, data, mu, logvar, orbit_loss
            )
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Accumulate losses
            epoch_losses['total'] += loss_dict['total_loss']
            epoch_losses['recon'] += loss_dict['recon_loss']
            epoch_losses['kl'] += loss_dict['kl_loss']
            epoch_losses['orbit'] += loss_dict['orbit_loss']
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'recon': f"{loss_dict['recon_loss']:.4f}",
                'kl': f"{loss_dict['kl_loss']:.4f}",
                'orbit': f"{loss_dict['orbit_loss']:.4f}"
            })
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate(self, temperature=0.8):
        """Validate the model"""
        self.model.eval()
        
        epoch_losses = {
            'total': 0.0,
            'recon': 0.0,
            'kl': 0.0,
            'orbit': 0.0
        }
        
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation', leave=False)
            for data, in pbar:
                data = data.to(self.device)
                
                # Forward pass
                recon_x, mu, logvar, orbit_loss, orbit_metrics = self.model(
                    data,
                    temperature=temperature,
                    compute_orbit_loss=True
                )
                
                # Compute loss
                total_loss, loss_dict = self.model.loss_function(
                    recon_x, data, mu, logvar, orbit_loss
                )
                
                # Accumulate losses
                epoch_losses['total'] += loss_dict['total_loss']
                epoch_losses['recon'] += loss_dict['recon_loss']
                epoch_losses['kl'] += loss_dict['kl_loss']
                epoch_losses['orbit'] += loss_dict['orbit_loss']
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss_dict['total_loss']:.4f}",
                    'orbit': f"{loss_dict['orbit_loss']:.4f}"
                })
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def train(self, num_epochs=100, temperature_schedule=None):
        """
        Complete training loop with optional temperature annealing
        
        Args:
            num_epochs: Number of epochs to train
            temperature_schedule: Optional function epoch -> temperature
        """
        print(f"\n{'='*60}")
        print(f"Training GOLC-VAE for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            # Determine temperature for this epoch
            if temperature_schedule is not None:
                temperature = temperature_schedule(epoch)
            else:
                # Default: anneal from 1.0 to 0.8
                temperature = 1.0 - 0.2 * min(epoch / max(num_epochs // 4, 1), 1.0)
            
            # Train and validate
            train_losses = self.train_epoch(temperature=temperature)
            val_losses = self.validate(temperature=temperature)
            
            # Get orbital statistics
            orbit_stats = self.model.get_orbit_statistics()
            
            # Update learning rate scheduler
            self.scheduler.step(val_losses['total'])
            
            # Record history
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_losses['total'])
            self.history['val_loss'].append(val_losses['total'])
            self.history['train_recon'].append(train_losses['recon'])
            self.history['val_recon'].append(val_losses['recon'])
            self.history['train_kl'].append(train_losses['kl'])
            self.history['val_kl'].append(val_losses['kl'])
            self.history['train_orbit'].append(train_losses['orbit'])
            self.history['val_orbit'].append(val_losses['orbit'])
            self.history['learning_rate'].append(current_lr)
            self.history['orbit_distance'].append(orbit_stats.get('mean_orbit_distance', 0.0))
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_losses['total']:.4f} "
                  f"(Recon: {train_losses['recon']:.4f}, "
                  f"KL: {train_losses['kl']:.4f}, "
                  f"Orbit: {train_losses['orbit']:.4f})")
            print(f"  Val Loss:   {val_losses['total']:.4f} "
                  f"(Recon: {val_losses['recon']:.4f}, "
                  f"KL: {val_losses['kl']:.4f}, "
                  f"Orbit: {val_losses['orbit']:.4f})")
            print(f"  LR: {current_lr:.6f}, Temp: {temperature:.3f}")
            
            if orbit_stats:
                print(f"  Orbit Stats: Mean={orbit_stats['mean_orbit_distance']:.4f}, "
                      f"Std={orbit_stats['std_orbit_distance']:.4f}")
            
            # Save best model
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
                print(f"  \u2713 New best model saved! (Val Loss: {self.best_val_loss:.4f}, LR: {current_lr:.6f})")
            else:
                self.patience_counter += 1
                
            # Regular checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            # Early stopping
            if self.patience_counter >= self.early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        # Save final model and history
        self.save_checkpoint(num_epochs - 1, is_best=False, final=True)
        self.save_history()
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")
    
    def save_checkpoint(self, epoch, is_best=False, final=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'model_config': {
                'input_dim': self.model.encoder_input.in_features,
                'latent_dim': self.model.decoder_input.in_features,
                'beta': self.model.beta,
                'beta_orbit': self.model.beta_orbit,
                'transposition_range': max(self.model.transposition_group)
            }
        }
        
        if is_best:
            path = self.save_dir / 'golc_vae_best.pt'
            torch.save(checkpoint, path)
        
        if final:
            path = self.save_dir / 'golc_vae_final.pt'
            torch.save(checkpoint, path)
        
        # Always save latest
        path = self.save_dir / 'golc_vae_latest.pt'
        torch.save(checkpoint, path)
    
    def save_history(self):
        """Save training history as JSON"""
        history_path = self.save_dir / 'golc_vae_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {history_path}")


def load_data(data_dir='dataset/processed', batch_size=128, val_split=0.1):
    """Load preprocessed MIDI data from .pt file (markov_sequences format)"""
    print(f"Loading data from {data_dir}...")

    # Try full_dataset.pt first, fall back to markov_sequences.pt
    for fname in ('full_dataset.pt', 'markov_sequences.pt'):
        sequences_path = Path(data_dir) / fname
        if sequences_path.exists():
            break
    else:
        raise FileNotFoundError(
            f"No preprocessed data found in {data_dir}. "
            "Run scripts/preprocess_dataset.py first."
        )

    raw = torch.load(sequences_path)
    items = raw.get('sequences', raw) if isinstance(raw, dict) else raw

    # Build pitch-histogram tensors (128-dim, normalised) from note sequences
    tensors = []
    for item in items:
        seq_data = item.get('sequences', {}) if isinstance(item, dict) else {}
        notes = seq_data.get('full', seq_data.get('melody', []))
        if not notes and isinstance(item, dict):
            notes = item.get('sequence', [])
        feature = np.zeros(128, dtype=np.float32)
        for n in notes:
            if isinstance(n, (int, float)) and 0 <= int(n) < 128:
                feature[int(n)] += 1
        if feature.sum() > 0:
            feature /= feature.sum()
        tensors.append(torch.from_numpy(feature))

    if not tensors:
        raise ValueError("No valid sequences found in the preprocessed data.")

    data_tensor = torch.stack(tensors)          # [N, 128]
    data_tensor = torch.clamp(data_tensor, 0, 1)
    print(f"Loaded {len(data_tensor)} sequences, shape: {data_tensor.shape}")

    # Train / val split
    num_val   = max(1, int(len(data_tensor) * val_split))
    num_train = len(data_tensor) - num_val
    indices   = torch.randperm(len(data_tensor))
    train_data = data_tensor[indices[:num_train]]
    val_data   = data_tensor[indices[num_train:]]

    train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(TensorDataset(val_data),   batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    metadata = {'num_train': num_train, 'num_val': num_val,
                'source_file': str(sequences_path)}
    print(f"Train samples: {num_train}, Val samples: {num_val}")
    return train_loader, val_loader, metadata


def main():
    parser = argparse.ArgumentParser(description='Train GOLC-VAE')
    parser.add_argument('--data-dir', type=str, default='dataset/processed',
                       help='Path to preprocessed data')
    parser.add_argument('--output-dir', type=str, default='output/trained_models',
                       help='Output directory for models')
    parser.add_argument('--input-dim', type=int, default=128,
                       help='Input dimension (MIDI note range)')
    parser.add_argument('--latent-dim', type=int, default=128,
                       help='Latent space dimension')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Beta-VAE weight for KL divergence')
    parser.add_argument('--beta-orbit', type=float, default=0.5,
                       help='Weight for orbital consistency loss')
    parser.add_argument('--transposition-range', type=int, default=6,
                       help='Range of transpositions (±semitones)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation split ratio')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader, metadata = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split
    )
    
    # Create model
    model = GOLC_VAE(
        input_dim=args.input_dim,
        latent_dim=args.latent_dim,
        beta=args.beta,
        beta_orbit=args.beta_orbit,
        transposition_range=args.transposition_range
    )
    
    print(f"\nModel Configuration:")
    print(f"  Input dim: {args.input_dim}")
    print(f"  Latent dim: {args.latent_dim}")
    print(f"  Beta (KL): {args.beta}")
    print(f"  Beta (Orbit): {args.beta_orbit}")
    print(f"  Transposition range: ±{args.transposition_range}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = GOLC_VAE_Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        device=device,
        save_dir=args.output_dir
    )
    
    # Train model
    trainer.train(num_epochs=args.epochs)


if __name__ == '__main__':
    main()
