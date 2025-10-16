import os
import glob
import time
import json
import hashlib
import warnings
import traceback
import concurrent.futures
from tqdm import tqdm
import torch
from music21 import converter, environment, note, chord
import logging
import multiprocessing as mp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress Music21 warnings and environment setup
warnings.filterwarnings("ignore")
try:
    env = environment.Environment()
    env.warn = False
except Exception as e:
    logger.error(f"Failed to initialize music21 environment: {e}")


def get_file_hash(filepath):
    """Generate a hash for a file to track if it's been processed."""
    try:
        stat = os.stat(filepath)
        hash_str = f"{filepath}_{stat.st_mtime}_{stat.st_size}"
        return hashlib.md5(hash_str.encode()).hexdigest()
    except FileNotFoundError:
        logger.warning(f"File not found for hashing: {filepath}")
        return None

def load_processed_cache(cache_file):
    """Load the cache of already processed files."""
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load cache file {cache_file}: {e}. Starting fresh.")
    return {}

def save_processed_cache(cache_file, cache):
    """Save the cache of processed files."""
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(cache, f, indent=2)
    except IOError as e:
        logger.warning(f"Could not save cache file {cache_file}: {e}")

def extract_enhanced_note_sequence(score):
    """Extracts an enhanced sequence of notes and chords from a music21 score."""
    note_sequence = []
    try:
        # Flatten the score to handle parts and nested structures
        for element in score.flatten().notes:
            if isinstance(element, note.Note):
                note_sequence.append((
                    element.pitch.midi,
                    float(element.duration.quarterLength),
                    element.volume.velocity,
                    float(element.offset)
                ))
            elif isinstance(element, chord.Chord):
                # Represent chord as a tuple of MIDI notes, sorted for consistency
                chord_notes = tuple(sorted([p.midi for p in element.pitches]))
                note_sequence.append((
                    chord_notes,
                    float(element.duration.quarterLength),
                    element.volume.velocity,
                    float(element.offset)
                ))
        # Sort by offset to ensure chronological order
        note_sequence.sort(key=lambda x: x[3])
    except Exception:
        logger.debug(f"Failed to extract enhanced note sequence:\n{traceback.format_exc()}")
        return None
    return note_sequence

def process_file(file_path):
    """
    Processes a single MIDI file. This function is designed to be run in a separate process.
    """
    try:
        # Each process should have its own environment settings if needed,
        # but top-level settings are often sufficient.
        score = converter.parse(file_path, forceSource=True)
        sequence = extract_enhanced_note_sequence(score)
        if sequence:
            return file_path, sequence
        else:
            # This path is logged from the main process based on the None return
            return file_path, None
    except Exception:
        # Log detailed error in the main process to avoid tqdm corruption
        error_info = traceback.format_exc()
        return file_path, (None, error_info)

def save_and_merge_data(output_file, temp_files):
    """
    Merges temporary batch files with the main output file to save progress.
    Uses a memory-efficient approach by only loading paths from the existing file,
    then appending new sequences.
    """
    logger.info(f"💾 Merging {len(temp_files)} temporary files into '{output_file}'...")
    
    # 1. Load only the PATHS from the existing file (not the full sequences)
    existing_paths = set()
    if os.path.exists(output_file):
        try:
            existing_data = torch.load(output_file)
            existing_paths = {item['path'] for item in existing_data.get('sequences', [])}
            logger.info(f"Found {len(existing_paths)} existing sequences in main file.")
        except Exception as e:
            logger.warning(f"⚠️ Could not load existing output file '{output_file}': {e}. Starting fresh.")

    # 2. Collect NEW sequences from temp files (skip duplicates)
    new_sequences = []
    duplicate_count = 0
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                batch_data = torch.load(temp_file)
                for item in batch_data.get('sequences', []):
                    if item['path'] not in existing_paths:
                        new_sequences.append(item)
                        existing_paths.add(item['path'])  # Track to avoid duplicates within temp files
                    else:
                        duplicate_count += 1
        except Exception as e:
            logger.warning(f"⚠️ Could not load temp file '{temp_file}': {e}")

    if duplicate_count > 0:
        logger.info(f"Skipped {duplicate_count} duplicate sequences.")

    # 3. Append new sequences to the existing file (memory-efficient)
    if new_sequences:
        if os.path.exists(output_file):
            # Load existing file, append, and save
            try:
                existing_data = torch.load(output_file)
                existing_data['sequences'].extend(new_sequences)
                torch.save(existing_data, output_file)
                logger.info(f"✅ Appended {len(new_sequences)} new sequences to existing file.")
            except Exception as e:
                logger.error(f"❌ Failed to append to existing file: {e}")
                # Fallback: save new sequences to a new file
                try:
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    torch.save({'sequences': new_sequences, 'created_at': time.time()}, output_file)
                    logger.info(f"✅ Saved {len(new_sequences)} sequences to '{output_file}'")
                except Exception as e2:
                    logger.error(f"❌ Fallback save also failed: {e2}")
        else:
            # No existing file, create new
            try:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                torch.save({'sequences': new_sequences, 'created_at': time.time()}, output_file)
                logger.info(f"✅ Saved {len(new_sequences)} sequences to '{output_file}'")
            except Exception as e:
                logger.error(f"❌ Failed to save: {e}")
    else:
        logger.info("No new sequences to save.")

    # 4. Clean up temp files
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except OSError as e:
                logger.warning(f"⚠️ Could not delete temp file '{temp_file}': {e}")
    logger.info("🧹 Cleaned up temporary files.")
    
    # Return total count
    total_count = len(existing_paths)
    return total_count

def main(midi_dir="dataset/midi", output_file="dataset/processed/markov_sequences.pt", cache_dir="output/trained_models"):
    """
    Scans a directory for MIDI files, processes them in parallel using a robust
    multiprocessing pool, and saves the extracted note sequences to a PyTorch file
    using a memory-efficient streaming approach.
    """
    logger.info("🚀 Starting ROBUST parallel MIDI preprocessing script with STREAMING.")
    
    # --- 1. Setup Stop Flag ---
    stop_file = ".stop_preprocessing"
    if os.path.exists(stop_file):
        os.remove(stop_file)
    logger.info(f"✅ To gracefully stop, create a file named '{stop_file}' in the root directory.")
    logger.info(f"(You can run this command in a separate terminal: touch {stop_file})")

    # --- 2. Scan for MIDI files ---
    logger.info(f"🔍 Scanning for MIDI files in '{midi_dir}'...")
    patterns = ['**/*.mid', '**/*.midi', '**/*.MID', '**/*.MIDI']
    midi_files = sorted(list(set(
        f for p in patterns for f in glob.glob(os.path.join(midi_dir, p), recursive=True)
    )))
    
    if not midi_files:
        logger.error(f"❌ No MIDI files found in '{midi_dir}'. Exiting.")
        return

    logger.info(f"✅ Found {len(midi_files)} total MIDI files.")

    # --- 3. Handle Caching ---
    cache_file = os.path.join(cache_dir, ".preprocess_cache.json")
    processed_cache = load_processed_cache(cache_file)
    files_to_process_map = {fp: get_file_hash(fp) for fp in midi_files}
    
    files_needing_processing = [
        fp for fp, h in files_to_process_map.items() if processed_cache.get(fp) != h
    ]

    if not files_needing_processing:
        logger.info("✅ All MIDI files are up-to-date. No new processing needed.")
        if not os.path.exists(output_file):
            logger.warning(f"⚠️ Output file '{output_file}' not found. Reprocessing all files.")
            files_needing_processing = midi_files
        else:
            return

    logger.info(f"🎵 Found {len(files_needing_processing)} new or modified files to process.")

    # --- 4. Parallel Processing with Streaming Saves ---
    total_failed = 0
    start_time = time.time()
    temp_files = []
    processed_paths_in_session = set()
    
    batch_size = 500  # Save to disk every 500 files
    num_workers = max(1, os.cpu_count())
    logger.info(f"⚡ Starting robust parallel processing with {num_workers} workers.")
    logger.info(f"💾 Streaming results to disk in batches of {batch_size}.")

    try:
        with mp.Pool(processes=num_workers, maxtasksperchild=1) as pool:
            pbar = tqdm(total=len(files_needing_processing), desc="🎵 Processing MIDI", unit="files")
            results_iterator = pool.imap_unordered(process_file, files_needing_processing)
            
            batch_results = []
            for i, result in enumerate(results_iterator):
                # Check for the stop file periodically
                if i % 10 == 0 and os.path.exists(stop_file):
                    logger.warning(f"\n🛑 '{stop_file}' detected. Stopping and saving progress...")
                    break

                path, sequence = result
                if sequence:
                    # Ensure we handle both normal and error returns from process_file
                    if isinstance(sequence, tuple) and len(sequence) == 2 and sequence[1] is not None:
                         # This is an error tuple from process_file
                        total_failed += 1
                    else:
                        batch_results.append({'path': path, 'sequence': sequence})
                        processed_paths_in_session.add(path)
                else:
                    total_failed += 1
                pbar.update(1)

                # --- Streaming Save Logic ---
                if len(batch_results) >= batch_size:
                    temp_file = f"{output_file}.part_{int(time.time())}_{i}"
                    torch.save({'sequences': batch_results}, temp_file)
                    temp_files.append(temp_file)
                    logger.info(f"\n📦 Saved batch of {len(batch_results)} to {temp_file}")
                    batch_results = []

            # Save any remaining results in the last batch
            if batch_results:
                temp_file = f"{output_file}.part_{int(time.time())}_final"
                torch.save({'sequences': batch_results}, temp_file)
                temp_files.append(temp_file)
                logger.info(f"\n📦 Saved final batch of {len(batch_results)} to {temp_file}")

            pbar.close()

    except KeyboardInterrupt:
        logger.warning("\n🛑 Keyboard interrupt detected. Stopping and saving progress...")
    finally:
        # --- 5. Final Merge and Cleanup ---
        duration = time.time() - start_time
        logger.info(f"⏱️ Processing finished in {duration:.2f} seconds.")
        
        total_saved = save_and_merge_data(output_file, temp_files)

        # --- 6. Update Cache ---
        for path in processed_paths_in_session:
            if path in files_to_process_map:
                processed_cache[path] = files_to_process_map[path]
        save_processed_cache(cache_file, processed_cache)
        logger.info(f"💾 Processed file cache updated at '{cache_file}'")

        # --- 7. Final Summary ---
        total_processed = len(processed_paths_in_session)
        total_attempted = total_processed + total_failed
        success_rate = total_processed / total_attempted * 100 if total_attempted > 0 else 0
        logger.info("\n" + "="*50)
        logger.info("📊 PREPROCESSING SUMMARY 📊")
        logger.info(f"  - Successfully processed: {total_processed} files")
        logger.info(f"  - Failed to process:     {total_failed} files")
        logger.info(f"  - Success rate:          {success_rate:.1f}%")
        logger.info(f"  - Total sequences saved: {total_saved}")
        logger.info("="*50)
        logger.info("✅ Preprocessing complete!")
        
        # Clean up the stop file
        if os.path.exists(stop_file):
            os.remove(stop_file)

if __name__ == "__main__":
    # This ensures the main function is called only when the script is executed directly
    main()
