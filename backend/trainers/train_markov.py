import os
import glob
import multiprocessing
import warnings
import sys
import numpy as np
from music21 import converter, environment
from tqdm import tqdm
import logging
from backend.models.markov_chain import MarkovChain

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress ALL Music21 warnings and errors
warnings.filterwarnings("ignore")
environment.Environment().warn = False

def process_midi_file(file_path):
    """Process a single MIDI file for Markov chain training with robust error handling"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Using no21 environment to suppress errors
            us = environment.UserSettings()
            us['warnings'] = 0
            score = converter.parse(file_path)
        return score
    except Exception as e:
        return None

def train_markov_model(midi_dir="dataset/midi", order=2, max_interval=12, output_dir="output/trained_models"):
    """Train Markov chain model with optimized processing and clear feedback"""
    logger.info(f"Initializing Markov model (order={order}, max_interval={max_interval})")
    model = MarkovChain(order=order, max_interval=max_interval)
    
    # Scan for MIDI files efficiently
    logger.info("Scanning for MIDI files...")
    midi_files = []
    
    # Use glob for faster file finding
    for pattern in ['**/*.mid', '**/*.midi']:
        midi_files.extend(glob.glob(os.path.join(midi_dir, pattern), recursive=True))
    
    # Remove duplicates
    midi_files = list(set(midi_files))
    
    if not midi_files:
        logger.error(f"No MIDI files found in {midi_dir}")
        return None
    
    logger.info(f"Found {len(midi_files)} MIDI files")
    
    # Process files in parallel with optimized chunking
    logger.info("Processing MIDI files in parallel...")
    cpu_count = max(1, multiprocessing.cpu_count() - 1)
    
    # Use larger chunks for better performance
    chunk_size = max(1, min(100, len(midi_files) // (cpu_count * 2)))
    
    with multiprocessing.Pool(processes=cpu_count) as pool:
        scores = list(tqdm(
            pool.imap(process_midi_file, midi_files, chunksize=chunk_size),
            total=len(midi_files),
            desc="Processing MIDI files"
        ))
    
    # Filter out None results
    scores = [s for s in scores if s is not None]
    
    if not scores:
        logger.error("No valid scores were processed")
        return None
    
    logger.info(f"Successfully processed {len(scores)}/{len(midi_files)} files")
    
    # Train model with progress tracking
    logger.info("Training Markov model...")
    progress_bar = tqdm(total=100, desc="Training")
    
    def update_progress(percent):
        progress_bar.update(int(percent * 100) - progress_bar.n)
    
    model.train(scores, progress_callback=update_progress)
    progress_bar.close()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "trained_markov.npy")
    
    # Save model
    try:
        model.save(output_path)
        logger.info(f"Model saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        # Try an alternate save location
        try:
            alt_path = os.path.join(os.path.dirname(output_dir), "trained_markov_fallback.npy")
            model.save(alt_path)
            logger.info(f"Model saved to alternate location: {alt_path}")
        except Exception as e2:
            logger.error(f"All save attempts failed: {e2}")
    
    # Output model statistics
    logger.info("\nModel Statistics:")
    
    # Get note transition matrix dimensions
    if hasattr(model.transitions, 'shape'):
        logger.info(f"Note transitions: {model.transitions.shape}")
    
    # Count interval transitions
    interval_transitions = len(model.interval_transitions)
    logger.info(f"Interval transitions: {interval_transitions}")
    
    # Log musical features
    musical_features = [
        ("Common keys", model.musical_features['common_keys']),
        ("Chord progressions", model.musical_features['common_chord_progressions']),
        ("Rhythm patterns", model.musical_features['rhythm_patterns']),
        ("Time signatures", model.musical_features['time_signatures']),
        ("Roman numeral transitions", model.musical_features['roman_numeral_transitions'])
    ]
    
    for name, data in musical_features:
        if data:
            count = len(data)
            examples = str(list(data.keys())[:3]) if hasattr(data, 'keys') else "..."
            logger.info(f"{name}: {count} entries (e.g. {examples})")
    
    return model

if __name__ == "__main__":
    # Get dataset directory with validation
    default_dir = "dataset/midi"
    
    if len(sys.argv) > 1:
        custom_dir = sys.argv[1]
        if os.path.exists(custom_dir):
            midi_dir = custom_dir
        else:
            logger.error(f"Directory not found: {custom_dir}")
            sys.exit(1)
    elif not os.path.exists(default_dir):
        logger.warning(f"Default directory '{default_dir}' not found.")
        print("Enter the path to your MIDI directory:")
        custom_dir = input("> ").strip()
        if custom_dir and os.path.exists(custom_dir):
            midi_dir = custom_dir
        else:
            logger.error("Invalid directory")
            sys.exit(1)
    else:
        midi_dir = default_dir
    
    # Allow custom parameters from command line
    order = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    max_interval = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    
    train_markov_model(midi_dir=midi_dir, order=order, max_interval=max_interval)