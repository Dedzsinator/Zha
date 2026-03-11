import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import os
import numpy as np

class MidiDataset(Dataset):
    def __init__(self, data_path, seq_len=64, resolution=4):
        """
        Args:
            data_path: Path to the processed .pt file
            seq_len: Length of sequences to generate (in time steps)
            resolution: Time steps per quarter note (e.g., 4 = 16th notes)
        """
        self.seq_len = seq_len
        self.resolution = resolution
        
        # Load data
        # We assume the data is a list of dicts, each dict has 'full', 'melody', etc.
        # Each item in 'full' is (pitch, duration, velocity, offset)
        try:
            data_dict = torch.load(data_path, weights_only=False)
            if isinstance(data_dict, dict) and 'sequences' in data_dict:
                raw_data = data_dict['sequences']
            else:
                raw_data = data_dict
            
            self.sequences = []
            for item in raw_data:
                if isinstance(item, dict):
                    if 'sequences' in item and isinstance(item['sequences'], dict):
                        self.sequences.append(item['sequences'])
                    elif 'full' in item:
                        self.sequences.append(item)
                        
        except Exception as e:
            print(f"Error loading data: {e}")
            self.sequences = []

        # Filter out empty sequences
        self.sequences = [s for s in self.sequences if s and len(s.get('full', [])) > 0]
        
        # Pre-calculate valid indices or just sample randomly?
        # For a large dataset, we might want to iterate over all segments.
        # But sequences have different lengths.
        # Strategy: We will index by (sequence_idx, start_time_idx)
        # But that's too complex to pre-calculate for 17k files.
        # Simple strategy: Each item in the dataset is a random crop from a random sequence?
        # No, __getitem__ needs a deterministic index.
        
        # We'll treat each file as one item, and __getitem__ will return a random crop 
        # or a specific crop. For training, random crop is good.
        # But for validation, we want deterministic.
        
        # Better: Flatten the dataset into segments?
        # If dataset is huge, flattening might take too much memory.
        # Let's stick to: __getitem__(idx) returns a processed tensor from sequence[idx].
        # If sequence is longer than seq_len, we take a random crop (training) or center crop (eval).
        # If shorter, we pad.
        
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_data = self.sequences[idx]
        notes = seq_data['full'] # List of (pitch, dur, vel, offset)
        
        # Convert to Piano Roll
        # 1. Determine duration of the sequence
        if not notes:
            return torch.zeros(self.seq_len, 128)
            
        last_note = max(notes, key=lambda x: x[3] + x[1])
        total_duration = last_note[3] + last_note[1]
        
        # Convert to grid
        # Grid size: ceil(total_duration * resolution)
        grid_len = int(np.ceil(total_duration * self.resolution))
        
        # If grid is smaller than seq_len, pad
        if grid_len < self.seq_len:
            piano_roll = torch.zeros(self.seq_len, 128)
            start_step = 0
        else:
            # Random crop
            max_start = grid_len - self.seq_len
            start_step = np.random.randint(0, max_start + 1)
            piano_roll = torch.zeros(self.seq_len, 128)
            
        # Fill piano roll
        # We only fill the window [start_step, start_step + seq_len]
        # Time t in grid corresponds to t / resolution beats
        
        window_start_beat = start_step / self.resolution
        window_end_beat = (start_step + self.seq_len) / self.resolution
        
        for pitch, dur, vel, offset in notes:
            # Check if note overlaps with window
            note_end = offset + dur
            
            if note_end <= window_start_beat or offset >= window_end_beat:
                continue
                
            # Calculate start and end indices in the window
            # relative_offset = offset - window_start_beat
            # relative_end = note_end - window_start_beat
            
            # Convert to steps
            step_start = int((offset - window_start_beat) * self.resolution)
            step_end = int((note_end - window_start_beat) * self.resolution)
            
            # Clip to window
            step_start = max(0, step_start)
            step_end = min(self.seq_len, step_end)
            
            if step_start < step_end:
                # Set pitch
                # Normalize velocity to [0, 1]
                piano_roll[step_start:step_end, int(pitch)] = vel / 127.0
                
        return piano_roll

class MidiDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size=32, seq_len=64, num_workers=4):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_workers = num_workers
        
    def setup(self, stage=None):
        full_dataset = MidiDataset(self.data_path, seq_len=self.seq_len)
        
        # Split
        train_size = int(0.8 * len(full_dataset))
        val_size = int(0.1 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
