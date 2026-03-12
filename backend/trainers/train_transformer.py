import torch, os, json, warnings
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.amp import GradScaler
import torch.nn.functional as F
from backend.models.transformer import TransformerModel
from backend.trainers.utils import train_epoch, LRSchedulerWithBatchOption, EarlyStopping
from backend.trainers.train_vae import MIDIDataset  # Reuse the dataset class
from backend.datamodules.hf_midi_dataset import build_hf_dataloader

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True

class TransformerMIDIDataset(MIDIDataset):
    """Extended dataset for Transformer training with chord/tempo conditioning"""
    
    def __init__(self, midi_dir=None, max_files=None, cache_size=500, use_preprocessed=True, track_type='full'):
        super().__init__(midi_dir, max_files, cache_size, use_preprocessed, track_type)
        
        if use_preprocessed:
            # Load additional conditioning data
            processed_data_path = "dataset/processed/markov_sequences.pt"
            if os.path.exists(processed_data_path):
                try:
                    data = torch.load(processed_data_path)
                    sequences = data['sequences']
                    
                    self.chord_sequences = []
                    self.tempo_sequences = []
                    
                    for item in sequences:
                        seq_data = item['sequences']
                        if track_type in seq_data and len(seq_data[track_type]) > 0:
                            # Load chord and tempo data if available
                            chords = seq_data.get('chords', [])
                            tempo_changes = seq_data.get('tempo_changes', [])
                            
                            self.chord_sequences.append(chords)
                            self.tempo_sequences.append(tempo_changes)
                        else:
                            # Default empty conditioning
                            self.chord_sequences.append([])
                            self.tempo_sequences.append([])
                    
                    print(f"✅ Loaded conditioning data for {len(self.chord_sequences)} sequences")
                    
                except Exception as e:
                    print(f"⚠️ Error loading conditioning data: {e}")
                    self.chord_sequences = [[]] * len(self.processed_sequences)
                    self.tempo_sequences = [[]] * len(self.processed_sequences)
    
    def __getitem__(self, idx):
        if self.use_preprocessed:
            # Get the base sequence
            sequence = self.processed_sequences[idx]
            
            # Get conditioning data
            chords = self.chord_sequences[idx] if hasattr(self, 'chord_sequences') else []
            tempo_changes = self.tempo_sequences[idx] if hasattr(self, 'tempo_sequences') else []
            
            return {
                'sequence': sequence,
                'chords': chords,
                'tempo_changes': tempo_changes
            }
        else:
            # Fallback to MIDI processing (simplified)
            return {
                'sequence': torch.zeros(128),  # Placeholder
                'chords': [],
                'tempo_changes': []
            }

def _transformer_collate_fn(batch):
    """Custom collate that handles the variable-length chord/tempo lists."""
    sequences = torch.stack([item['sequence'] for item in batch])
    chords = [item['chords'] for item in batch]
    tempo_changes = [item['tempo_changes'] for item in batch]
    return {'sequence': sequences, 'chords': chords, 'tempo_changes': tempo_changes}


def train_transformer_model(
    epochs=100,
    batch_size=64,
    learning_rate=1e-4,
    grad_accum_steps=1,
    patience=10,
    embed_dim=512,
    num_heads=8,
    num_layers=8,
    dim_feedforward=2048,
    dropout=0.1,
    use_preprocessed=True,
    track_type='full',
    use_huggingface=False,
    hf_genre_filter=None,
):
    """
    Train a transformer model for music generation with enhanced performance

    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Peak learning rate for the scheduler
        grad_accum_steps: Gradient accumulation steps for larger effective batch size
        patience: Early stopping patience
        embed_dim: Embedding dimension for the transformer
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout rate for regularization
        use_preprocessed: Whether to use preprocessed data instead of processing MIDI files
        track_type: Type of track to train on ('full', 'melody', 'bass', 'drums')

    Returns:
        Trained model and training metrics
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Training on: {device}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"📊 GPU: {torch.cuda.get_device_name(0)}")

    # Initialize model
    model = TransformerModel(
        input_dim=128,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        enable_conditioning=True  # Enable chord/tempo conditioning
    ).to(device)
    print(f"🔄 Model initialized: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"💾 Using preprocessed data: {use_preprocessed}, track: {track_type}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=epochs * 1000 // grad_accum_steps,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=1e4
    )
    scheduler_wrapper = LRSchedulerWithBatchOption(scheduler, step_every_batch=True)
    scaler = GradScaler('cuda') if torch.cuda.is_available() else None

    # Early stopping
    early_stopping = EarlyStopping(patience=patience)

    # Data loading with memory-efficient caching
    print("📂 Setting up data loader...")
    if use_huggingface:
        print(f"🌐 Streaming from HuggingFace (amaai-lab/MidiCaps), genre_filter={hf_genre_filter}")
        # HF dataset returns plain tensors; wrap them to match the dict format the loop expects
        _hf_base = build_hf_dataloader(
            batch_size=batch_size,
            genre_filter=hf_genre_filter,
            shuffle_buffer=2000,
            num_workers=0,
            pin_memory=True,
        )
        class _HFDictWrapper:
            """Wraps plain-tensor batches into the dict format TransformerMIDIDataset uses."""
            def __init__(self, loader): self._loader = loader
            def __iter__(self):
                for seq in self._loader:
                    yield {'sequence': seq, 'chords': [[] for _ in range(seq.size(0))],
                           'tempo_changes': [[] for _ in range(seq.size(0))]}
            def __len__(self): return len(self._loader) if hasattr(self._loader, '__len__') else 0
        dataloader = _HFDictWrapper(_hf_base)
    else:
        dataset = TransformerMIDIDataset(midi_dir="dataset/midi/", cache_size=500,
                                       use_preprocessed=use_preprocessed, track_type=track_type)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4 if not use_preprocessed else 0,
            pin_memory=True,
            collate_fn=_transformer_collate_fn
        )

    # Training loop
    os.makedirs("output/trained_models", exist_ok=True)
    all_metrics = []

    print(f"🚀 Starting training: {epochs} epochs (gradient accumulation: {grad_accum_steps} steps)")

    epoch_pbar = tqdm(range(epochs), desc="🎵 Epochs", unit="epoch", position=0)
    for epoch in epoch_pbar:
        # Custom training loop for Transformer with conditioning
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", unit="batch", leave=False, position=1)
        for batch_idx, batch in enumerate(batch_pbar):
            sequences = batch['sequence'].to(device)
            chords = batch['chords']  # List of chord lists
            tempo_changes = batch['tempo_changes']  # List of tempo lists
            
            # Prepare conditioning tensors for this batch
            batch_size = sequences.size(0)
            seq_len = sequences.size(1) if len(sequences.shape) > 1 else 1
            
            chord_conditioning = None
            tempo_conditioning = None
            
            if model.enable_conditioning:
                # Convert batch conditioning data to tensors
                max_chords = max(len(c) for c in chords) if chords else 0
                max_tempos = max(len(t) for t in tempo_changes) if tempo_changes else 0
                
                if max_chords > 0:
                    chord_conditioning = torch.zeros(batch_size, seq_len, 31, device=device)
                    for b, chord_list in enumerate(chords):
                        for i, chord in enumerate(chord_list[:seq_len]):
                            # Fill chord conditioning (same logic as in model)
                            if 'root' in chord and chord['root'] is not None:
                                root_idx = chord['root'] % 12
                                chord_conditioning[b, i, root_idx] = 1.0
                            # Add other chord features...
                
                if max_tempos > 0:
                    tempo_conditioning = torch.zeros(batch_size, seq_len, 1, device=device)
                    for b, tempo_list in enumerate(tempo_changes):
                        for i, tempo_change in enumerate(tempo_list[:seq_len]):
                            if 'tempo' in tempo_change:
                                tempo_val = tempo_change['tempo']
                                normalized_tempo = (tempo_val - 60) / (200 - 60)
                                normalized_tempo = max(0.0, min(1.0, normalized_tempo))
                                tempo_conditioning[b, i, 0] = normalized_tempo
            
            # Forward pass
            optimizer.zero_grad()
            _amp_enabled = scaler is not None
            with torch.amp.autocast('cuda', enabled=_amp_enabled):
                output = model(sequences, chord_conditioning=chord_conditioning, 
                             tempo_conditioning=tempo_conditioning)
                
                # model always returns [batch, seq_len, input_dim]; squeeze seq dim when 1
                if output.dim() == 3 and output.size(1) == 1:
                    output = output.squeeze(1)   # [batch, input_dim]

                # Compute loss (cross-entropy for next token prediction)
                if output.dim() == 2:
                    # Single step prediction: [batch, 128] vs [batch] target
                    loss = F.cross_entropy(output, sequences.argmax(dim=-1))
                else:
                    # Sequence prediction - predict next tokens
                    target = sequences[:, 1:].contiguous()
                    output = output[:, :-1].contiguous()
                    loss = F.cross_entropy(output.view(-1, output.size(-1)), 
                                         target.view(-1))
            
            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
                if (batch_idx + 1) % grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler_wrapper.step()
            else:
                loss.backward()
                if (batch_idx + 1) % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler_wrapper.step()
            
            epoch_loss += loss.item()
            num_batches += 1

            batch_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = epoch_loss / num_batches
        metrics = {'loss': avg_loss}
        all_metrics.append(metrics)

        # Update outer progress bar
        epoch_pbar.set_postfix({'loss': f"{avg_loss:.4f}", 'lr': f"{scheduler.get_last_lr()[0]:.2e}"})

        # Report metrics
        tqdm.write(f"📈 Epoch {epoch+1}: Loss={avg_loss:.4f}, LR={scheduler.get_last_lr()[0]:.6f}")

        # Early stopping check
        if early_stopping(avg_loss):
            tqdm.write(f"⚠️ Early stopping triggered after {epoch+1} epochs")
            break

        # Save checkpoints
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            torch.save(model.state_dict(), f"output/trained_models/transformer_ep{epoch+1}.pt")

    # Save final model and JIT scripted version for faster inference
    torch.save(model.state_dict(), "output/trained_models/trained_transformer.pt")

    # Save full metrics history to JSON
    os.makedirs("output/metrics", exist_ok=True)
    metrics_path = "output/metrics/transformer_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            'epochs_run': len(all_metrics),
            'final_loss': all_metrics[-1]['loss'] if all_metrics else None,
            'best_loss': min(m['loss'] for m in all_metrics) if all_metrics else None,
            'history': all_metrics,
            'hyperparams': {
                'epochs': epochs, 'batch_size': batch_size, 'learning_rate': learning_rate,
                'embed_dim': embed_dim, 'num_heads': num_heads, 'num_layers': num_layers,
                'dim_feedforward': dim_feedforward, 'dropout': dropout,
                'grad_accum_steps': grad_accum_steps, 'track_type': track_type
            }
        }, f, indent=2)
    print(f"📊 Metrics saved to {metrics_path}")

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
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--hf", action="store_true", help="Stream from HuggingFace instead of local data")
    p.add_argument("--genre", nargs="*", default=None, help="Genre filter, e.g. --genre pop rock")
    args = p.parse_args()
    train_transformer_model(
        epochs=100,
        batch_size=64,
        grad_accum_steps=4,
        patience=15,
        embed_dim=512,
        num_heads=8,
        num_layers=8,
        dim_feedforward=2048,
        dropout=0.1,
        use_preprocessed=not args.hf,
        track_type='full',
        use_huggingface=args.hf,
        hf_genre_filter=args.genre,
    )