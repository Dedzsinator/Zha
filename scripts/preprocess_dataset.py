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
                cache = json.load(f)
            logger.info(f"✅ Loaded cache with {len(cache)} entries")
            return cache
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load cache file {cache_file}: {e}")
    return {}

def save_processed_cache(cache_file, cache):
    """Save the cache of processed files."""
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(cache, f, indent=2)
        logger.info(f"💾 Saved cache with {len(cache)} entries")
    except IOError as e:
        logger.warning(f"Could not save cache file {cache_file}: {e}")

# Global in-memory cache for processed sequences
_memory_cache = {}
_cache_size_limit = 1000  # Keep up to 1000 processed files in memory

def get_cached_sequence(file_path, cache_file):
    """Get processed sequence from memory cache or disk cache."""
    file_hash = get_file_hash(file_path)
    if file_hash is None:
        return None
    
    # Check memory cache first
    if file_path in _memory_cache:
        cached_data = _memory_cache[file_path]
        if cached_data['hash'] == file_hash:
            logger.debug(f"💾 Memory cache hit for {file_path}")
            return cached_data['sequences']
        else:
            # File changed, remove from cache
            del _memory_cache[file_path]
    
    # Check disk cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                disk_cache = json.load(f)
            if file_path in disk_cache and disk_cache[file_path]['hash'] == file_hash:
                # Load from disk and add to memory cache
                sequences = disk_cache[file_path]['sequences']
                _memory_cache[file_path] = {
                    'hash': file_hash,
                    'sequences': sequences,
                    'accessed': time.time()
                }
                # Maintain cache size limit (LRU eviction)
                if len(_memory_cache) > _cache_size_limit:
                    oldest_key = min(_memory_cache.keys(), 
                                    key=lambda k: _memory_cache[k]['accessed'])
                    del _memory_cache[oldest_key]
                
                logger.debug(f"💾 Disk cache hit for {file_path}")
                return sequences
        except Exception as e:
            logger.debug(f"Could not load from disk cache: {e}")
    
    return None

def save_to_cache(file_path, sequences, cache_file):
    """Save processed sequences to both memory and disk cache."""
    file_hash = get_file_hash(file_path)
    if file_hash is None or not sequences:
        return
    
    cache_entry = {
        'hash': file_hash,
        'sequences': sequences,
        'processed_at': time.time()
    }
    
    # Save to memory cache
    _memory_cache[file_path] = {
        'hash': file_hash,
        'sequences': sequences,
        'accessed': time.time()
    }
    
    # Save to disk cache
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                disk_cache = json.load(f)
        else:
            disk_cache = {}
        
        disk_cache[file_path] = cache_entry
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(disk_cache, f, indent=2)
        
        logger.debug(f"💾 Cached processed data for {file_path}")
    except Exception as e:
        logger.debug(f"Could not save to disk cache: {e}")

def extract_enhanced_note_sequence(score):
    """
    Extracts an enhanced sequence of notes and chords from a music21 score.
    Separates tracks into melody, bass, and drums based on MIDI characteristics.
    
    Returns a dict with:
        - 'melody': Main melodic content (notes above bass range, non-percussion)
        - 'bass': Bass notes (MIDI 21-55, approximately E0-G3)
        - 'drums': Percussion (MIDI channel 10 or instruments with channel 9)
        - 'full': Complete sequence without separation
    """
    melody_sequence = []
    bass_sequence = []
    drums_sequence = []
    full_sequence = []
    
    try:
        # Check if score has parts (multi-track)
        parts = score.parts if hasattr(score, 'parts') and score.parts else [score]
        
        for part in parts:
            # Check if this is a drum part (channel 10 in MIDI, which is index 9)
            is_drum_part = False
            try:
                if hasattr(part, 'midiChannel') and part.midiChannel == 9:
                    is_drum_part = True
                # Also check instrument name
                if hasattr(part, 'partName') and part.partName:
                    drum_keywords = ['drum', 'percussion', 'perc', 'kit', 'rhythm']
                    if any(kw in part.partName.lower() for kw in drum_keywords):
                        is_drum_part = True
            except:
                pass
            
            # Process notes in this part
            for element in part.flatten().notes:
                offset = float(element.offset)
                duration = float(element.duration.quarterLength)
                velocity = element.volume.velocity if hasattr(element.volume, 'velocity') else 80
                
                if isinstance(element, note.Note):
                    midi_note = element.pitch.midi
                    note_data = (midi_note, duration, velocity, offset)
                    
                    # Add to full sequence
                    full_sequence.append(note_data)
                    
                    # Categorize by type
                    if is_drum_part or midi_note >= 27 and hasattr(element, 'pitch') and element.pitch.name in ['C', 'D', 'E', 'F', 'G', 'A', 'B']:
                        # Check if it's likely a drum sound (high velocity, short duration)
                        if velocity > 90 or duration < 0.5:
                            drums_sequence.append(note_data)
                        elif midi_note <= 55:  # Bass range: E0 (21) to G3 (55)
                            bass_sequence.append(note_data)
                        else:
                            melody_sequence.append(note_data)
                    elif midi_note <= 55:  # Bass range
                        bass_sequence.append(note_data)
                    else:  # Melody range
                        melody_sequence.append(note_data)
                        
                elif isinstance(element, chord.Chord):
                    # For chords, categorize by lowest note
                    chord_notes = tuple(sorted([p.midi for p in element.pitches]))
                    chord_data = (chord_notes, duration, velocity, offset)
                    
                    full_sequence.append(chord_data)
                    
                    lowest_note = min(chord_notes)
                    if lowest_note <= 55:  # Bass chord
                        bass_sequence.append(chord_data)
                    else:  # Melody chord
                        melody_sequence.append(chord_data)
        
        # Sort all sequences by offset
        melody_sequence.sort(key=lambda x: x[3])
        bass_sequence.sort(key=lambda x: x[3])
        drums_sequence.sort(key=lambda x: x[3])
        full_sequence.sort(key=lambda x: x[3])
        
        return {
            'melody': melody_sequence,
            'bass': bass_sequence,
            'drums': drums_sequence,
            'full': full_sequence,
            'track_count': len(parts)
        }
        
    except Exception:
        logger.debug(f"Failed to extract enhanced note sequence:\n{traceback.format_exc()}")
        return None

def process_file(file_path):
    """
    Processes a single MIDI file with track separation.
    Returns sequences separated into melody, bass, and drums.
    """
    try:
        # Each process should have its own environment settings if needed,
        # but top-level settings are often sufficient.
        score = converter.parse(file_path, forceSource=True)
        sequences = extract_enhanced_note_sequence(score)
        
        if sequences and sequences.get('full'):
            # Include metadata about track separation
            result = {
                'melody': sequences['melody'],
                'bass': sequences['bass'],
                'drums': sequences['drums'],
                'full': sequences['full'],
                'track_count': sequences.get('track_count', 1),
                'has_bass': len(sequences['bass']) > 0,
                'has_drums': len(sequences['drums']) > 0,
                'has_melody': len(sequences['melody']) > 0
            }
            return file_path, result
        else:
            # This path is logged from the main process based on the None return
            return file_path, None
    except Exception:
        # Log detailed error in the main process to avoid tqdm corruption
        error_info = traceback.format_exc()
        return file_path, (None, error_info)

def save_and_merge_data(output_file, temp_files, cached_sequences=None):
    """
    Merges temporary batch files with the main output file to save progress.
    Uses a memory-efficient approach by only loading paths from the existing file,
    then appending new sequences and cached sequences.
    """
    if cached_sequences is None:
        cached_sequences = []
        
    logger.info(f"💾 Merging {len(temp_files)} temporary files and {len(cached_sequences)} cached sequences into '{output_file}'...")
    
    # 1. Load only the PATHS from the existing file (not the full sequences)
    existing_paths = set()
    if os.path.exists(output_file):
        try:
            existing_data = torch.load(output_file)
            existing_paths = {item['path'] for item in existing_data.get('sequences', [])}
            logger.info(f"Found {len(existing_paths)} existing sequences in main file.")
        except Exception as e:
            logger.warning(f"⚠️ Could not load existing output file '{output_file}': {e}. Starting fresh.")

    # 2. Collect NEW sequences from temp files and cached sequences (skip duplicates)
    new_sequences = []
    duplicate_count = 0
    
    # Add cached sequences first
    for item in cached_sequences:
        if item['path'] not in existing_paths:
            new_sequences.append(item)
            existing_paths.add(item['path'])
        else:
            duplicate_count += 1
    
    # Add sequences from temp files
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
    
    # Check cache for already processed files
    files_needing_processing = []
    cached_sequences = []
    
    logger.info("🧠 Checking cache for processed files...")
    for fp, current_hash in files_to_process_map.items():
        cached_seq = get_cached_sequence(fp, cache_file)
        if cached_seq is not None:
            # File is cached and up-to-date
            cached_sequences.append({
                'path': fp,
                'sequences': cached_seq,
                'metadata': {
                    'track_count': cached_seq.get('track_count', 1),
                    'has_bass': cached_seq.get('has_bass', False),
                    'has_drums': cached_seq.get('has_drums', False),
                    'has_melody': cached_seq.get('has_melody', False)
                }
            })
        else:
            # File needs processing (not cached or cache is outdated)
            files_needing_processing.append(fp)
    
    logger.info(f"🧠 Loaded {len(cached_sequences)} sequences from cache")
    logger.info(f"🎵 Found {len(files_needing_processing)} files needing processing")

    # If we have cached sequences and no new files to process, just save and exit
    if not files_needing_processing and cached_sequences:
        logger.info("✅ All sequences available from cache. Updating output file...")
        torch.save({'sequences': cached_sequences}, output_file)
        logger.info(f"💾 Saved {len(cached_sequences)} cached sequences to {output_file}")
        return

    if not files_needing_processing and not cached_sequences:
        logger.info("✅ All MIDI files are up-to-date. No new processing needed.")
        if not os.path.exists(output_file):
            logger.warning(f"⚠️ Output file '{output_file}' not found. Reprocessing all files.")
            files_needing_processing = midi_files
        else:
            return

    logger.info(f"🧠 Loaded {len(cached_sequences)} sequences from cache")
    logger.info(f"🎵 Found {len(files_needing_processing)} new or modified files to process.")

    # If we have cached sequences and no new files to process, just save and exit
    if not files_needing_processing and cached_sequences:
        logger.info("✅ All sequences available from cache. Saving to output file...")
        torch.save({'sequences': cached_sequences}, output_file)
        logger.info(f"💾 Saved {len(cached_sequences)} cached sequences to {output_file}")
        return

    if not files_needing_processing and not cached_sequences:
        logger.info("✅ All MIDI files are up-to-date. No new processing needed.")
        if not os.path.exists(output_file):
            logger.warning(f"⚠️ Output file '{output_file}' not found. Reprocessing all files.")
            files_needing_processing = midi_files
        else:
            return

    # --- 4. Parallel Processing with Streaming Saves ---
    total_failed = 0
    start_time = time.time()
    temp_files = []
    processed_paths_in_session = set()
    
    batch_size = 500  # Save to disk every 500 files
    num_workers = max(1, os.cpu_count())
    logger.info(f"⚡ Starting robust parallel processing with {num_workers} workers.")
    logger.info(f"💾 Streaming results to disk in batches of {batch_size}.")
    logger.info(f"🧠 Using memory cache (limit: {_cache_size_limit} files) with disk fallback.")

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

                path, sequences = result
                if sequences:
                    # Ensure we handle both normal and error returns from process_file
                    if isinstance(sequences, tuple) and len(sequences) == 2 and sequences[1] is not None:
                         # This is an error tuple from process_file
                        total_failed += 1
                    else:
                        # Store the separated tracks
                        batch_results.append({
                            'path': path,
                            'sequences': sequences,  # Contains melody, bass, drums, full
                            'metadata': {
                                'track_count': sequences.get('track_count', 1),
                                'has_bass': sequences.get('has_bass', False),
                                'has_drums': sequences.get('has_drums', False),
                                'has_melody': sequences.get('has_melody', False)
                            }
                        })
                        processed_paths_in_session.add(path)
                        
                        # Save to cache for future use
                        save_to_cache(path, sequences, cache_file)
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
        
        # Merge temp files and cached sequences
        total_saved = save_and_merge_data(output_file, temp_files, cached_sequences)

        # --- 6. Update Cache ---
        for path in processed_paths_in_session:
            if path in files_to_process_map:
                processed_cache[path] = files_to_process_map[path]
        save_processed_cache(cache_file, processed_cache)
        logger.info(f"💾 Processed file cache updated at '{cache_file}'")

        # --- 7. Final Summary with Track Statistics ---
        total_processed = len(processed_paths_in_session)
        total_attempted = total_processed + total_failed
        success_rate = total_processed / total_attempted * 100 if total_attempted > 0 else 0
        
        # Calculate track separation statistics
        track_stats = {'with_bass': 0, 'with_drums': 0, 'with_melody': 0, 'multi_track': 0}
        if os.path.exists(output_file):
            try:
                data = torch.load(output_file)
                for item in data.get('sequences', []):
                    metadata = item.get('metadata', {})
                    if metadata.get('has_bass'):
                        track_stats['with_bass'] += 1
                    if metadata.get('has_drums'):
                        track_stats['with_drums'] += 1
                    if metadata.get('has_melody'):
                        track_stats['with_melody'] += 1
                    if metadata.get('track_count', 1) > 1:
                        track_stats['multi_track'] += 1
            except:
                pass
        
        logger.info("\n" + "="*60)
        logger.info("📊 PREPROCESSING SUMMARY 📊")
        logger.info(f"  - Successfully processed: {total_processed} files")
        logger.info(f"  - Failed to process:     {total_failed} files")
        logger.info(f"  - Success rate:          {success_rate:.1f}%")
        logger.info(f"  - Total sequences saved: {total_saved}")
        logger.info("\n🎵 TRACK SEPARATION STATISTICS:")
        logger.info(f"  - Files with bass:       {track_stats['with_bass']}")
        logger.info(f"  - Files with drums:      {track_stats['with_drums']}")
        logger.info(f"  - Files with melody:     {track_stats['with_melody']}")
        logger.info(f"  - Multi-track files:     {track_stats['multi_track']}")
        logger.info("="*60)
        logger.info("✅ Preprocessing complete with track separation!")
        
        # Clean up the stop file
        if os.path.exists(stop_file):
            os.remove(stop_file)

if __name__ == "__main__":
    # This ensures the main function is called only when the script is executed directly
    main()
