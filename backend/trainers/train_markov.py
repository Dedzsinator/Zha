import os
import numpy as np
from music21 import converter
import warnings
import multiprocessing as mp
from tqdm import tqdm
import inspect
from backend.models.markov_chain import MarkovChain
import matplotlib.pyplot as plt
import torch  # For GPU memory monitoring even though Markov doesn't use PyTorch

# Filter warnings
warnings.filterwarnings("ignore", category=UserWarning)

class MIDIProcessor:
    def __init__(self, midi_dir):
        self.midi_dir = midi_dir
        self.file_list = []

        # Walk through all subdirectories to find MIDI files
        for root, _, files in os.walk(midi_dir):
            for file in files:
                if file.endswith('.mid'):
                    full_path = os.path.join(root, file)
                    self.file_list.append(full_path)

        print(f"Found {len(self.file_list)} MIDI files")

        # Cache to avoid processing the same file multiple times
        self.cache = {}

    def extract_notes_from_midi(self, midi_path):
        """Extract note sequences from a MIDI file with caching"""
        if midi_path in self.cache:
            return self.cache[midi_path]

        try:
            # Parse the MIDI file
            score = converter.parse(midi_path)

            # Extract notes
            notes = []
            for element in score.flatten():  # Use flatten() instead of flat
                if hasattr(element, 'pitch'):
                    notes.append(element.pitch.midi)

            # Cache the result
            self.cache[midi_path] = notes
            return notes

        except Exception as e:
            print(f"Error processing {midi_path}: {e}")
            return []

    def process_file_batch(self, file_batch):
        """Process a batch of files and return their note sequences"""
        results = []
        for midi_file in file_batch:
            notes = self.extract_notes_from_midi(midi_file)
            if notes:  # Only include non-empty sequences
                results.append(notes)
        return results

    def process_in_parallel(self, batch_size=10, num_workers=None):
        """Process all files in parallel with efficient batching"""
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 1)

        # Group files into batches for better efficiency
        batches = [self.file_list[i:i+batch_size] for i in range(0, len(self.file_list), batch_size)]

        print(f"Processing {len(self.file_list)} files in {len(batches)} batches using {num_workers} workers")

        all_sequences = []

        # Using process pool for parallel processing
        with mp.Pool(processes=num_workers) as pool:
            # Process batches in parallel
            batch_results = list(tqdm(
                pool.imap(self.process_file_batch, batches),
                total=len(batches),
                desc="Processing MIDI batches"
            ))

            # Flatten results
            for batch_result in batch_results:
                all_sequences.extend(batch_result)

        print(f"Successfully processed {len(all_sequences)} sequences")
        return all_sequences

def train_markov_model(midi_dir="dataset/midi/",
                       order=2,
                       batch_size=20,
                       num_workers=None):
    """Train an optimized Markov Chain model with parallel processing"""
    print("Starting optimized Markov model training...")

    if torch.cuda.is_available():
        # Monitor GPU usage even though Markov doesn't use GPU directly
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"Will monitor GPU memory usage during processing")

        # Clear GPU memory from any previous tasks
        torch.cuda.empty_cache()
        start_mem = torch.cuda.memory_allocated() / 1e9
        print(f"Initial GPU memory allocated: {start_mem:.2f} GB")

    # Initialize the processor
    processor = MIDIProcessor(midi_dir)

    # Process files in parallel
    all_sequences = processor.process_in_parallel(batch_size=batch_size, num_workers=num_workers)

    if not all_sequences:
        print("No valid sequences found. Check your MIDI files.")
        return None

    # Display some statistics about the dataset
    seq_lengths = [len(seq) for seq in all_sequences]
    avg_length = sum(seq_lengths) / len(seq_lengths)
    print(f"Dataset statistics:")
    print(f"  - Total sequences: {len(all_sequences)}")
    print(f"  - Average sequence length: {avg_length:.2f} notes")
    print(f"  - Min sequence length: {min(seq_lengths)}")
    print(f"  - Max sequence length: {max(seq_lengths)}")

    # Initialize Markov model - don't pass order parameter as it's not supported
    print(f"\nTraining Markov model (desired order: {order})...")
    model = MarkovChain()

    # Check if the model has a method to set the order
    if hasattr(model, 'set_order'):
        model.set_order(order)
        print(f"Set Markov chain order to {order}")
    elif hasattr(model, 'order'):
        # Try setting the attribute directly if it exists
        try:
            model.order = order
            print(f"Set Markov chain order to {order}")
        except:
            print(f"Warning: Could not set Markov chain order to {order}. Using default.")
    else:
        print(f"Warning: MarkovChain doesn't support setting order parameter. Using default order.")

    # Check if the train method accepts a progress_callback parameter
    train_params = inspect.signature(model.train).parameters
    supports_progress = 'progress_callback' in train_params

    if supports_progress:
        # If the model supports progress callbacks, use them
        with tqdm(total=len(all_sequences), desc="Training Markov model") as pbar:
            model.train(all_sequences, progress_callback=lambda x: pbar.update(1))
    else:
        # Otherwise train without progress updates
        print("Training Markov model (no progress updates available)...")
        # Simply train on all sequences at once
        model.train(all_sequences)
        print("Training complete.")

    # Create directory for trained models if it doesn't exist
    os.makedirs("trained_models", exist_ok=True)

    # Save the trained model
    model.save("trained_models/trained_markov.npy")
    print("Markov model trained and saved to trained_models/trained_markov.npy")

    # Optional: visualize transition probabilities for common notes
    try:
        # Generate a visualization of transition probabilities
        plt.figure(figsize=(10, 8))

        # Get some of the most common notes from middle C region
        common_notes = list(range(60, 72))  # Middle C to B

        # Get transition probabilities for these common notes if available
        if hasattr(model, 'transitions') and len(common_notes) > 0:
            # Create a heatmap of transition probabilities
            probabilities = np.zeros((len(common_notes), len(common_notes)))

            for i, from_note in enumerate(common_notes):
                for j, to_note in enumerate(common_notes):
                    if from_note < model.transitions.shape[0] and to_note < model.transitions.shape[1]:
                        probabilities[i, j] = model.transitions[from_note, to_note]

            plt.imshow(probabilities, cmap='viridis')
            plt.colorbar(label="Transition Probability")
            plt.title("Markov Chain Transition Probabilities")
            plt.xlabel("To Note")
            plt.ylabel("From Note")

            # Label axes with note names
            note_names = ['C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4']
            plt.xticks(range(len(common_notes)), note_names, rotation=45)
            plt.yticks(range(len(common_notes)), note_names)

            plt.savefig("markov_transitions.png")
            print("Created visualization of transition probabilities at markov_transitions.png")
    except Exception as e:
        print(f"Could not create transition visualization: {e}")

    # Show final memory usage
    if torch.cuda.is_available():
        end_mem = torch.cuda.memory_allocated() / 1e9
        print(f"Final GPU memory allocated: {end_mem:.2f} GB")

    return model

if __name__ == "__main__":
    # You can adjust these parameters based on your system
    train_markov_model(
        midi_dir="dataset/midi/",
        order=2,
        batch_size=20,
        num_workers=None  # Auto-detect number of workers
    )