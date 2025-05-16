import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

def train_epoch(model, dataloader, optimizer, scheduler, device, 
               scaler=None, accumulation_steps=1, epoch_idx=0):
    """
    Efficient training loop for one epoch with support for:
    - Mixed precision training
    - Gradient accumulation
    - Progress reporting
    
    Args:
        model: PyTorch model
        dataloader: DataLoader instance
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        device: Device to use for training
        scaler: GradScaler for mixed precision (None to disable)
        accumulation_steps: Number of steps for gradient accumulation
        epoch_idx: Current epoch index for display
        
    Returns:
        Dictionary with metrics (loss, etc.)
    """
    model.train()
    use_amp = scaler is not None
    
    # Track metrics
    metrics = {'loss': 0.0, 'recon_loss': 0.0, 'kl_loss': 0.0}
    samples_processed = 0
    
    # Create progress bar
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_idx+1}")
    step = 0
    
    # Determine if this is a VAE model (has reparameterize method)
    is_vae = hasattr(model, 'reparameterize')
    
    for batch in progress_bar:
        batch = batch.to(device, non_blocking=True)
        step += 1
        samples_processed += batch.size(0)
        
        # Compute loss based on model type
        if use_amp:
            with autocast():
                if is_vae:  # VAE model
                    recon, mu, logvar = model(batch)
                    recon_loss = nn.functional.mse_loss(recon, batch)
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch.size(0)
                    beta = min(1.0, 0.2 + epoch_idx/50)
                    loss = recon_loss + beta * kl_loss
                    metrics['recon_loss'] += recon_loss.item() * batch.size(0)
                    metrics['kl_loss'] += kl_loss.item() * batch.size(0)
                else:  # Other model types (e.g., Transformer)
                    output = model(batch)
                    loss = nn.functional.mse_loss(output, batch)
                
                # Scale loss by accumulation steps
                loss = loss / accumulation_steps
                
            # Update with scaled gradients
            scaler.scale(loss).backward()
            
            # Accumulate gradients
            if step % accumulation_steps == 0 or step == len(dataloader):
                scaler.unscale_(optimizer)
                max_norm = 1.0 if is_vae else 0.5
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                # Update scheduler if it's batch-based
                if scheduler is not None and hasattr(scheduler, 'step_every_batch') and scheduler.step_every_batch:
                    scheduler.step()
        else:
            # Similar non-AMP implementation
            if is_vae:  # VAE model
                recon, mu, logvar = model(batch)
                recon_loss = nn.functional.mse_loss(recon, batch)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch.size(0)
                beta = min(1.0, 0.2 + epoch_idx/50)
                loss = recon_loss + beta * kl_loss
                metrics['recon_loss'] += recon_loss.item() * batch.size(0)
                metrics['kl_loss'] += kl_loss.item() * batch.size(0)
            else:  # Other model types
                output = model(batch)
                loss = nn.functional.mse_loss(output, batch)
            
            # Scale loss by accumulation steps
            loss = loss / accumulation_steps
            loss.backward()
            
            # Accumulate gradients
            if step % accumulation_steps == 0 or step == len(dataloader):
                max_norm = 1.0 if is_vae else 0.5
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
                # Update scheduler if it's batch-based
                if scheduler is not None and hasattr(scheduler, 'step_every_batch') and scheduler.step_every_batch:
                    scheduler.step()
        
        # Update total loss
        metrics['loss'] += loss.item() * accumulation_steps * batch.size(0)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item() * accumulation_steps:.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
        })
    
    # Normalize metrics by samples processed
    for key in metrics:
        metrics[key] /= samples_processed
        
    return metrics

class LRSchedulerWithBatchOption:
    """Wrapper for LR schedulers to track whether they update per batch"""
    def __init__(self, scheduler, step_every_batch=False):
        self.scheduler = scheduler
        self.step_every_batch = step_every_batch
        
    def step(self):
        self.scheduler.step()
        
    def get_last_lr(self):
        return self.scheduler.get_last_lr()

class EarlyStopping:
    """Early stopping implementation to stop training when not improving"""
    def __init__(self, patience=10, min_delta=0, mode='min', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode
        self.verbose = verbose
        
    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
            return False
            
        if self.mode == 'min':
            improvement = self.best_score - val_score > self.min_delta
        else:  # mode == 'max'
            improvement = val_score - self.best_score > self.min_delta
            
        if improvement:
            self.best_score = val_score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False

class MemoryEfficientDataset(torch.utils.data.Dataset):
    """Memory-efficient dataset with LRU cache"""
    def __init__(self, dataset, cache_size=100):
        self.dataset = dataset
        self.cache_size = cache_size
        self._cache = {}
        self._access_order = []
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if idx in self._cache:
            # Move to end of access order (most recently used)
            self._access_order.remove(idx)
            self._access_order.append(idx)
            return self._cache[idx]
            
        # Get item from dataset
        item = self.dataset[idx]
        
        # Add to cache
        self._cache[idx] = item
        self._access_order.append(idx)
        
        # If cache too large, remove least recently used
        if len(self._cache) > self.cache_size:
            old_idx = self._access_order.pop(0)
            del self._cache[old_idx]
            
        return item
