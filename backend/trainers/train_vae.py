import torch, os, json, numpy as np, warnings
import sys
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from backend.models.vae import VAEModel
from backend.trainers.utils import LRSchedulerWithBatchOption, EarlyStopping
from backend.datamodules.hf_midi_dataset import build_hf_dataloader

warnings.filterwarnings("ignore", category=UserWarning)
torch.backends.cudnn.benchmark = True

class MIDIDataset(Dataset):
    def __init__(self, midi_dir=None, max_files=None, cache_size=500, use_preprocessed=True, track_type='full'):
        self.use_preprocessed = use_preprocessed
        self.track_type = track_type
        self.data_cache = {}
        self.cache_size = cache_size
        self._access_order = []
        
        if use_preprocessed:
            # Load from preprocessed data
            processed_data_path = "dataset/processed/markov_sequences.pt"
            print(f"Loading preprocessed data from {processed_data_path} (track: {track_type})...")
            
            if not os.path.exists(processed_data_path):
                print(f"Preprocessed data not found, falling back to MIDI processing")
                self.use_preprocessed = False
                midi_dir = midi_dir or "dataset/midi/"
            else:
                try:
                    import torch
                    data = torch.load(processed_data_path)
                    sequences = data['sequences']
                    
                    # Extract sequences for the specified track type
                    self.processed_sequences = []
                    valid_files = 0
                    
                    for item in sequences:
                        seq_data = item['sequences']
                        if track_type in seq_data and len(seq_data[track_type]) > 0:
                            # Convert sequence to pitch histogram for VAE
                            notes = seq_data[track_type]
                            feature = np.zeros(128, dtype=np.float32)
                            for note in notes:
                                if isinstance(note, (int, float)) and 0 <= int(note) < 128:
                                    feature[int(note)] += 1
                            
                            # Normalize
                            if np.sum(feature) > 0:
                                feature = feature / np.sum(feature)
                            
                            self.processed_sequences.append(torch.from_numpy(feature).float())
                            valid_files += 1
                        elif track_type == 'full' and 'sequence' in item:
                            # Fallback for backward compatibility
                            notes = item['sequence']
                            feature = np.zeros(128, dtype=np.float32)
                            for note in notes:
                                if isinstance(note, (int, float)) and 0 <= int(note) < 128:
                                    feature[int(note)] += 1
                            
                            if np.sum(feature) > 0:
                                feature = feature / np.sum(feature)
                            
                            self.processed_sequences.append(torch.from_numpy(feature).float())
                            valid_files += 1
                    
                    print(f"Loaded {len(self.processed_sequences)} {track_type} sequences from {valid_files} files")
                    
                    if len(self.processed_sequences) == 0:
                        print(f"No {track_type} sequences found, falling back to MIDI processing")
                        self.use_preprocessed = False
                        midi_dir = midi_dir or "dataset/midi/"
                        
                except Exception as e:
                    print(f"Error loading preprocessed data: {e}, falling back to MIDI processing")
                    self.use_preprocessed = False
                    midi_dir = midi_dir or "dataset/midi/"
        
        if not self.use_preprocessed:
            # Original MIDI file processing
            self.midi_dir = midi_dir
            self.file_list = []
            
            # Find MIDI files in directory structure
            print(" Scanning for MIDI files...")
            for root, _, files in os.walk(midi_dir):
                for file in files:
                    if file.endswith(('.mid', '.midi')):
                        self.file_list.append(os.path.join(os.path.relpath(root, midi_dir), file))
                        if max_files and len(self.file_list) >= max_files:
                            break
            print(f" Found {len(self.file_list)} MIDI files")

    def __len__(self):
        if self.use_preprocessed:
            return len(self.processed_sequences)
        else:
            return len(self.file_list)

    def __getitem__(self, idx):
        if self.use_preprocessed:
            return self.processed_sequences[idx]
        
        # Memory-efficient LRU cache for MIDI processing
        if idx in self.data_cache:
            # Move to most recently used
            self._access_order.remove(idx)
            self._access_order.append(idx)
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

            # Add to cache with LRU policy
            self.data_cache[idx] = tensor
            self._access_order.append(idx)

            # Enforce cache size limit
            if len(self.data_cache) > self.cache_size:
                old_idx = self._access_order.pop(0)
                del self.data_cache[old_idx]

            return tensor

        except Exception as e:
            print(f"Error processing {midi_path}: {e}")
            return torch.zeros(128, dtype=torch.float32)

def train_epoch(model, dataloader, optimizer, scheduler, device, scaler=None,
                accumulation_steps=1, epoch_idx=0, consistency_weight=0.1,
                show_progress=True):
    """Training epoch with consistency loss to promote smoother transitions"""
    model.train()
    epoch_loss = 0
    epoch_recon_loss = 0
    epoch_kl_loss = 0
    epoch_consistency_loss = 0
    # IterableDataset (HuggingFace streaming) has no __len__; fall back to None
    try:
        n_batches_total = len(dataloader)
    except TypeError:
        n_batches_total = None
    _device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    i = -1  # track actual batch count in case of streaming dataset

    optimizer.zero_grad()

    with tqdm(
        total=n_batches_total,
        desc=f"Epoch {epoch_idx+1}",
        ncols=100,
        disable=not show_progress,
        dynamic_ncols=True,
        mininterval=1.0,
    ) as pbar:
        for i, batch in enumerate(dataloader):
            batch = batch.to(device, non_blocking=True)

            # Forward pass with mixed precision training
            with autocast(device_type=_device_type, enabled=scaler is not None):
                recon_batch, mu, logvar = model(batch)

            # Calculate losses in full precision
            recon_batch_float = recon_batch.float()
            recon_loss = F.binary_cross_entropy(recon_batch_float, batch, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar.float() - mu.float().pow(2) - logvar.float().exp())
            beta_kl_loss = model.beta * kl_loss

            # Consistency loss
            note_diffs = torch.abs(recon_batch_float[:, 1:] - recon_batch_float[:, :-1])
            consistency_loss = torch.mean(note_diffs) * batch.size(0) * consistency_weight

            # Total loss
            loss = (recon_loss + beta_kl_loss + consistency_loss) / batch.size(0)

            # Backward pass with gradient accumulation (flush every N steps)
            if scaler is not None:
                scaler.scale(loss / accumulation_steps).backward()
                if (i + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if scheduler.step_every_batch:
                        scheduler.step()
            else:
                (loss / accumulation_steps).backward()
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    if scheduler.step_every_batch:
                        scheduler.step()

            # Update metrics
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item() / batch.size(0)
            epoch_kl_loss += kl_loss.item() / batch.size(0)
            epoch_consistency_loss += consistency_loss.item() / batch.size(0)

            # Update progress bar
            pbar.update(1)
            if show_progress:
                pbar.set_postfix({
                    'loss': f"{epoch_loss/(i+1):.4f}",
                    'recon': f"{epoch_recon_loss/(i+1):.4f}",
                    'kl': f"{epoch_kl_loss/(i+1):.4f}"
                })

    # Flush any remaining accumulated gradients after the last batch
    n_batches = i + 1
    if n_batches % accumulation_steps != 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            optimizer.step()
            optimizer.zero_grad()

    # Return average metrics
    return {
        'loss': epoch_loss / n_batches,
        'recon_loss': epoch_recon_loss / n_batches,
        'kl_loss': epoch_kl_loss / n_batches,
        'consistency_loss': epoch_consistency_loss / n_batches
    }

def train_vae_model(epochs=100, batch_size=128, learning_rate=2e-4, latent_dim=128,
                   grad_accum_steps=1, patience=10, cache_size=500, beta=0.5,
                   consistency_weight=0.2, use_preprocessed=True, track_type='full',
                   use_huggingface=False, hf_genre_filter=None, resume_checkpoint=None):
    """VAE model training with beta-VAE and consistency loss for better music generation
    
    Args:
        use_preprocessed: Whether to use preprocessed data instead of processing MIDI files
        track_type: Type of track to train on ('full', 'melody', 'bass', 'drums')
        use_huggingface: Stream data from amaai-lab/MidiCaps on HuggingFace (no download)
        hf_genre_filter: List of genre strings to filter, e.g. ['pop', 'rock'].
                         None = use all 168k tracks.
    """
    # Setup device
    use_tqdm = sys.stdout.isatty()

    def _progress_log(message):
        if use_tqdm:
            tqdm.write(message)
        else:
            print(message)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Initialize model
    model = VAEModel(input_dim=128, latent_dim=latent_dim, beta=beta).to(device)
    print(f"Model initialized: {sum(p.numel() for p in model.parameters())} parameters")
    print(f" Using beta={beta}, consistency_weight={consistency_weight}")
    print(f"Using preprocessed data: {use_preprocessed}, track: {track_type}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate * 0.01)
    scheduler_wrapper = LRSchedulerWithBatchOption(scheduler, step_every_batch=False)

    # Create gradient scaler for mixed precision training
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    # Early stopping
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # Data loading
    print("Setting up data loader...")
    if use_huggingface:
        print(f"Streaming from HuggingFace (amaai-lab/MidiCaps), genre_filter={hf_genre_filter}")
        dataloader = build_hf_dataloader(
            batch_size=batch_size,
            genre_filter=hf_genre_filter,
            shuffle_buffer=2000,
            num_workers=0,
            pin_memory=True,
        )
    else:
        dataset = MIDIDataset(midi_dir="dataset/midi/", cache_size=cache_size,
                             use_preprocessed=use_preprocessed, track_type=track_type)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4 if not use_preprocessed else 0, pin_memory=True)

    # Training loop setup
    os.makedirs("output/trained_models", exist_ok=True)
    all_metrics = []
    start_epoch = 0
    best_loss_so_far = float('inf')

    # Optional resume from checkpoint
    if resume_checkpoint:
        resume_path = resume_checkpoint
        if resume_checkpoint == "latest":
            resume_path = "output/trained_models/vae_latest.pt"
        if os.path.exists(resume_path):
            ckpt = torch.load(resume_path, map_location=device)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
                if 'optimizer_state_dict' in ckpt:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                if 'scheduler_state_dict' in ckpt:
                    try:
                        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                    except Exception as e:
                        _progress_log(f"Could not restore scheduler state: {e}")
                if scaler is not None and 'scaler_state_dict' in ckpt and ckpt['scaler_state_dict'] is not None:
                    try:
                        scaler.load_state_dict(ckpt['scaler_state_dict'])
                    except Exception as e:
                        _progress_log(f"Could not restore scaler state: {e}")
                start_epoch = int(ckpt.get('epoch', -1)) + 1
                all_metrics = ckpt.get('history', []) or []
                best_loss_so_far = float(ckpt.get('best_loss', best_loss_so_far))
                _progress_log(f" Resumed VAE from {resume_path} at epoch {start_epoch}")
            else:
                model.load_state_dict(ckpt)
                _progress_log(f" Loaded model weights from {resume_path} (optimizer state not found)")
        else:
            _progress_log(f"Resume checkpoint not found: {resume_path}. Starting fresh.")

    print(f" Starting training: {epochs} epochs (gradient accumulation: {grad_accum_steps} steps)")

    epoch_pbar = tqdm(
        range(start_epoch, epochs),
        desc=" Epochs",
        unit="epoch",
        position=0,
        disable=not use_tqdm,
        dynamic_ncols=True,
        mininterval=1.0,
    )
    for epoch in epoch_pbar:
        # Train one epoch
        metrics = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler_wrapper,
            device=device,
            scaler=scaler,
            accumulation_steps=grad_accum_steps,
            epoch_idx=epoch,
            consistency_weight=consistency_weight,
            show_progress=use_tqdm,
        )

        # Step scheduler if epoch-based
        if not scheduler_wrapper.step_every_batch:
            scheduler.step()

        # Store metrics
        all_metrics.append(metrics)

        # Update outer progress bar
        if use_tqdm:
            epoch_pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'recon': f"{metrics['recon_loss']:.4f}",
                'kl': f"{metrics['kl_loss']:.4f}",
            })

        if metrics['loss'] < best_loss_so_far:
            best_loss_so_far = metrics['loss']

        # Report metrics
        _progress_log(
            f"Epoch {epoch+1}: Loss={metrics['loss']:.4f} "
            f"(Recon={metrics['recon_loss']:.4f}, KL={metrics['kl_loss']:.4f}, "
            f"Consistency={metrics['consistency_loss']:.4f}), "
            f"LR={scheduler.get_last_lr()[0]:.6f}"
        )

        # Early stopping check
        if early_stopping(metrics['loss']):
            _progress_log(f"Early stopping triggered after {epoch+1} epochs")
            break

        # Generate samples periodically
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                samples = model.sample(num_samples=3, temperature=0.8, device=device)
                _progress_log(
                    f"Sample note distribution entropy: "
                    f"{-torch.sum(samples * torch.log(samples + 1e-8), dim=1).mean().item():.4f}"
                )

        # Save checkpoints
        checkpoint_payload = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
            'best_loss': best_loss_so_far,
            'history': all_metrics,
            'hyperparams': {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'latent_dim': latent_dim,
                'beta': beta,
                'consistency_weight': consistency_weight,
                'grad_accum_steps': grad_accum_steps,
                'track_type': track_type,
            },
        }
        torch.save(checkpoint_payload, "output/trained_models/vae_latest.pt")
        if metrics['loss'] <= best_loss_so_far:
            torch.save(checkpoint_payload, "output/trained_models/vae_best.pt")
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            torch.save(checkpoint_payload, f"output/trained_models/vae_ep{epoch+1}.pt")

    # Save final model
    torch.save(model.state_dict(), "output/trained_models/trained_vae.pt")
    print("Training complete! Model saved to output/trained_models/trained_vae.pt")

    # Save full metrics history to JSON
    os.makedirs("output/metrics", exist_ok=True)
    metrics_path = "output/metrics/vae_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            'epochs_run': len(all_metrics),
            'final_loss': all_metrics[-1]['loss'] if all_metrics else None,
            'final_recon_loss': all_metrics[-1]['recon_loss'] if all_metrics else None,
            'final_kl_loss': all_metrics[-1]['kl_loss'] if all_metrics else None,
            'final_consistency_loss': all_metrics[-1]['consistency_loss'] if all_metrics else None,
            'best_loss': min(m['loss'] for m in all_metrics) if all_metrics else None,
            'history': all_metrics,
            'hyperparams': {
                'epochs': epochs, 'batch_size': batch_size, 'learning_rate': learning_rate,
                'latent_dim': latent_dim, 'beta': beta, 'consistency_weight': consistency_weight,
                'grad_accum_steps': grad_accum_steps, 'track_type': track_type
            }
        }, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Export to ONNX format
    try:
        onnx_path = "output/trained_models/trained_vae.onnx"
        model.export_to_onnx(onnx_path)
        print(f"Model exported to ONNX format: {onnx_path}")
    except Exception as e:
        print(f"ONNX export failed: {e}")

    return model, all_metrics

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--hf", action="store_true", help="Stream from HuggingFace instead of local data")
    p.add_argument("--genre", nargs="*", default=None, help="Genre filter for HF dataset, e.g. --genre pop rock")
    p.add_argument("--resume", nargs="?", const="latest", default=None,
                   help="Resume from checkpoint path, or use --resume for latest")
    args = p.parse_args()
    train_vae_model(
        epochs=100,
        batch_size=128,
        grad_accum_steps=2,
        patience=15,
        cache_size=500,
        beta=0.5,
        consistency_weight=0.2,
        use_preprocessed=not args.hf,
        track_type='full',
        use_huggingface=args.hf,
        hf_genre_filter=args.genre,
        resume_checkpoint=args.resume,
    )