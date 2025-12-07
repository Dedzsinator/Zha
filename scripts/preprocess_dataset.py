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
import psutil
import gc
from collections import deque
import threading
from functools import partial
import gzip
import pickle
import mmap
import numpy as np

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

# ULTRA-FAST OPTIMIZATION CONSTANTS - MEMORY SAFE
MAX_MEMORY_PERCENT = 60  # Even lower memory limit for safety
BATCH_SIZE = 100  # Much smaller batches for memory control
MAX_WORKERS = min(mp.cpu_count(), 8)  # Conservative worker count
MEMORY_CHECK_INTERVAL = 25  # Check memory more frequently
FAST_MODE = True  # Skip heavy analysis for speed
USE_COMPRESSION = True  # Compress cached data
STREAMING_BATCH_SIZE = 200  # Smaller chunks to prevent memory spikes
MEMORY_CLEANUP_INTERVAL = 10  # Force cleanup every N files
WORKER_RESTART_INTERVAL = 50  # Restart workers more frequently
MAX_FILES_PER_WORKER = 5  # Max files per worker before restart

# Memory safety limits
MAX_MEMORY_CACHE_SIZE = 50  # Keep fewer files in memory cache
FORCE_GC_EVERY_N_FILES = 5  # Force garbage collection more often

# Thread-local storage for music21 objects to avoid recreation
thread_local = threading.local()

def get_music21_stream():
    """Get thread-local music21 stream to avoid recreation overhead."""
    if not hasattr(thread_local, 'stream'):
        thread_local.stream = converter.parse
    return thread_local.stream

def get_memory_usage():
    """Get current memory usage percentage."""
    return psutil.virtual_memory().percent

def force_memory_cleanup():
    """Force aggressive garbage collection and memory cleanup."""
    # Force multiple GC cycles
    for _ in range(3):
        gc.collect()

    # Clear any cached music21 objects
    try:
        from music21 import common
        common.cleanup()
        # Clear environment cache
        if hasattr(environment, 'Environment'):
            env = environment.Environment()
            if hasattr(env, 'refStreamOrPath'):
                env.refStreamOrPath = None
    except:
        pass

    # Clear thread local storage
    try:
        thread_local.__dict__.clear()
    except:
        pass

    # Clear memory cache if it's too large
    global _memory_cache
    if len(_memory_cache) > MAX_MEMORY_CACHE_SIZE // 2:
        # Keep only most recently accessed items
        sorted_items = sorted(_memory_cache.items(),
                            key=lambda x: x[1]['accessed'], reverse=True)
        _memory_cache = dict(sorted_items[:MAX_MEMORY_CACHE_SIZE // 4])

    # Force Python to release memory to OS
    import ctypes
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except:
        pass

def preload_file_batch(file_paths, batch_size=50):
    """Preload a batch of files into memory for faster processing."""
    preloaded = {}
    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i:i + batch_size]
        for file_path in batch:
            try:
                # Use memory mapping for faster file access
                with open(file_path, 'rb') as f:
                    preloaded[file_path] = f.read()
            except Exception as e:
                logger.debug(f"Failed to preload {file_path}: {e}")
                continue

        # Force cleanup between batches
        if i % (batch_size * 5) == 0:
            force_memory_cleanup()

    return preloaded

def process_file_batch(file_batch, preloaded_data=None):
    """Process a batch of files with optional preloaded data."""
    results = []
    for file_path in file_batch:
        try:
            # For now, skip preloaded data to avoid multiprocessing issues
            score = converter.parse(file_path, forceSource=True)

            sequences = extract_enhanced_note_sequence(score)

            if sequences and sequences.get('full'):
                result = {
                    'path': file_path,
                    'sequences': sequences,
                    'metadata': {
                        'track_count': sequences.get('track_count', 1),
                        'has_bass': len(sequences.get('bass', [])) > 0,
                        'has_drums': len(sequences.get('drums', [])) > 0,
                        'has_melody': len(sequences.get('melody', [])) > 0,
                        'processed_at': time.time(),
                        'fast_mode': FAST_MODE
                    }
                }
                results.append((file_path, result))
            else:
                results.append((file_path, None))

        except Exception as e:
            logger.debug(f"Failed to process {file_path}: {e}")
            results.append((file_path, None))

    return results

def check_memory_and_cleanup(force=False):
    """Check memory usage and cleanup if needed."""
    mem_percent = get_memory_usage()
    if force or mem_percent > MAX_MEMORY_PERCENT:
        logger.info(f"🧹 Memory usage at {mem_percent:.1f}%, forcing cleanup...")
        force_memory_cleanup()
        new_mem = get_memory_usage()
        logger.info(f"✅ Memory after cleanup: {new_mem:.1f}%")
        return True
    return False

def check_memory_before_chunk():
    """Check if memory is safe to start a new chunk."""
    mem_percent = get_memory_usage()
    if mem_percent > MAX_MEMORY_PERCENT:
        logger.warning(f"⚠️ Memory usage too high ({mem_percent:.1f}%) before starting chunk. Forcing cleanup...")
        force_memory_cleanup()
        new_mem = get_memory_usage()
        if new_mem > MAX_MEMORY_PERCENT:
            logger.error(f"❌ Memory still too high ({new_mem:.1f}%) after cleanup. Skipping chunk for safety.")
            return False
        logger.info(f"✅ Memory reduced to {new_mem:.1f}% after cleanup.")
    return True

def get_optimal_workers():
    """Get optimal number of workers based on memory."""
    mem_percent = get_memory_usage()
    if mem_percent > 85:
        return 1
    elif mem_percent > 70:
        return min(2, MAX_WORKERS)
    elif mem_percent > 50:
        return min(4, MAX_WORKERS)
    else:
        return MAX_WORKERS

def create_memory_efficient_pool(workers=None):
    """Create a memory-efficient process pool that restarts periodically."""
    if workers is None:
        workers = get_optimal_workers()
    return mp.Pool(processes=workers, maxtasksperchild=MAX_FILES_PER_WORKER)


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
_cache_size_limit = MAX_MEMORY_CACHE_SIZE  # Keep fewer files in memory cache

def get_cached_sequence(file_path, cache_file):
    """Get processed sequence from sequence cache file."""
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

    # Load from sequence cache file
    sequence_cache_file = cache_file.replace('.preprocess_cache.json', '.sequence_cache.json')
    if os.path.exists(sequence_cache_file):
        try:
            with open(sequence_cache_file, 'r') as f:
                sequence_cache = json.load(f)
            if file_path in sequence_cache:
                entry = sequence_cache[file_path]
                if isinstance(entry, dict) and entry.get('hash') == file_hash:
                    # Check if we have compressed sequences
                    if 'sequences_compressed' in entry and entry.get('compressed'):
                        try:
                            compressed_data = bytes.fromhex(entry['sequences_compressed'])
                            sequences = pickle.loads(gzip.decompress(compressed_data))
                            # Add to memory cache
                            _memory_cache[file_path] = {
                                'hash': file_hash,
                                'sequences': sequences,
                                'accessed': time.time()
                            }
                            logger.debug(f"💾 Sequence cache hit for {file_path}")
                            return sequences
                        except Exception as e:
                            logger.debug(f"Failed to decompress cached sequences: {e}")
                    elif 'sequences' in entry:
                        sequences = entry['sequences']
                        # Add to memory cache
                        _memory_cache[file_path] = {
                            'hash': file_hash,
                            'sequences': sequences,
                            'accessed': time.time()
                        }
                        logger.debug(f"💾 Sequence cache hit for {file_path}")
                        return sequences
        except Exception as e:
            logger.debug(f"Could not load from sequence cache: {e}")

    return None

def save_to_cache(file_path, sequences, cache_file):
    """Save processed sequences to sequence cache file."""
    file_hash = get_file_hash(file_path)
    if file_hash is None or not sequences:
        return

    # Use a separate sequence cache file
    sequence_cache_file = cache_file.replace('.preprocess_cache.json', '.sequence_cache.json')

    # Load existing sequence cache
    sequence_cache = {}
    if os.path.exists(sequence_cache_file):
        try:
            with open(sequence_cache_file, 'r') as f:
                sequence_cache = json.load(f)
        except Exception as e:
            logger.debug(f"Could not load sequence cache: {e}")

    # Compress sequences for storage efficiency
    if USE_COMPRESSION:
        try:
            compressed_sequences = gzip.compress(pickle.dumps(sequences))
            cache_entry = {
                'hash': file_hash,
                'sequences_compressed': compressed_sequences.hex(),
                'processed_at': time.time(),
                'compressed': True
            }
        except Exception as e:
            # Fallback to uncompressed
            cache_entry = {
                'hash': file_hash,
                'sequences': sequences,
                'processed_at': time.time(),
                'compressed': False
            }
    else:
        cache_entry = {
            'hash': file_hash,
            'sequences': sequences,
            'processed_at': time.time(),
            'compressed': False
        }

    # Save to sequence cache
    sequence_cache[file_path] = cache_entry

    # Limit cache size to prevent huge files (keep most recent 2000 entries)
    if len(sequence_cache) > 2000:
        # Sort by processed_at and keep most recent
        sorted_entries = sorted(sequence_cache.items(),
                              key=lambda x: x[1].get('processed_at', 0),
                              reverse=True)
        sequence_cache = dict(sorted_entries[:2000])

    # Save sequence cache
    try:
        with open(sequence_cache_file, 'w') as f:
            json.dump(sequence_cache, f, separators=(',', ':'))
    except Exception as e:
        logger.warning(f"Failed to save sequence cache: {e}")

    # Also save to memory cache (limit size)
    if len(_memory_cache) < _cache_size_limit:
        _memory_cache[file_path] = {
            'hash': file_hash,
            'sequences': sequences,  # Keep uncompressed in memory
            'accessed': time.time()
        }

def extract_enhanced_note_sequence(score):
    """
    ULTRA-FAST: Extracts only essential note sequences, skips all heavy analysis.
    Returns minimal dict for maximum speed with memory-efficient processing.
    """
    melody_sequence = []
    bass_sequence = []
    drums_sequence = []
    full_sequence = []

    try:
        # ULTRA-FAST: Skip all heavy analysis
        parts = score.parts if hasattr(score, 'parts') and score.parts else [score]

        # Pre-allocate lists for better memory efficiency
        melody_sequence = []
        bass_sequence = []
        drums_sequence = []
        full_sequence = []

        # ULTRA-FAST: Single pass through all notes with optimized processing
        for part in parts:
            # FAST: Quick drum detection - cache result per part
            is_drum_part = False
            try:
                if hasattr(part, 'midiChannel') and part.midiChannel == 9:
                    is_drum_part = True
                elif hasattr(part, 'partName') and part.partName:
                    part_name_lower = part.partName.lower()
                    if any(kw in part_name_lower for kw in ['drum', 'percussion', 'perc']):
                        is_drum_part = True
            except:
                pass

            # ULTRA-FAST: Process notes without music21 overhead
            # Use flatten() once and iterate efficiently
            try:
                flattened_notes = part.flatten().notes
                for element in flattened_notes:
                    try:
                        # Extract properties once to avoid repeated attribute access
                        offset = float(element.offset)
                        duration = float(element.duration.quarterLength)
                        velocity = getattr(element.volume, 'velocity', 80) if hasattr(element, 'volume') else 80

                        if isinstance(element, note.Note):
                            midi_note = element.pitch.midi
                            note_data = (midi_note, duration, velocity, offset)

                            # FAST: Categorize by pitch range with early exit
                            if is_drum_part:
                                drums_sequence.append(note_data)
                            elif midi_note <= 55:  # Bass range
                                bass_sequence.append(note_data)
                            else:  # Melody range
                                melody_sequence.append(note_data)

                            full_sequence.append(note_data)

                    except Exception:
                        # Skip problematic notes silently for speed
                        continue
            except Exception:
                # If flatten fails, try direct iteration
                try:
                    for element in part.notes:
                        try:
                            offset = float(element.offset)
                            duration = float(element.duration.quarterLength)
                            velocity = getattr(element.volume, 'velocity', 80) if hasattr(element, 'volume') else 80

                            if isinstance(element, note.Note):
                                midi_note = element.pitch.midi
                                note_data = (midi_note, duration, velocity, offset)

                                if is_drum_part:
                                    drums_sequence.append(note_data)
                                elif midi_note <= 55:
                                    bass_sequence.append(note_data)
                                else:
                                    melody_sequence.append(note_data)

                                full_sequence.append(note_data)

                        except Exception:
                            continue
                except Exception:
                    continue

        # ULTRA-FAST: Return minimal data structure with computed stats
        track_count = len(parts)
        has_bass = len(bass_sequence) > 0
        has_drums = len(drums_sequence) > 0
        has_melody = len(melody_sequence) > 0

        return {
            'melody': melody_sequence,
            'bass': bass_sequence,
            'drums': drums_sequence,
            'full': full_sequence,
            'chords': [],  # Empty in fast mode
            'tempo_changes': [],  # Empty in fast mode
            'polyphony_events': [],  # Empty in fast mode
            'track_count': track_count,
            'has_bass': has_bass,
            'has_drums': has_drums,
            'has_melody': has_melody
        }

    except Exception as e:
        logger.debug(f"Ultra-fast extraction failed: {e}")
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
                'chords': sequences.get('chords', []),
                'tempo_changes': sequences.get('tempo_changes', []),
                'polyphony_events': sequences.get('polyphony_events', []),
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
            existing_data = torch.load(output_file, weights_only=False)
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
                batch_data = torch.load(temp_file, weights_only=False)
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
                existing_data = torch.load(output_file, weights_only=False)
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

def create_final_merged_file(output_file, output_dir, cached_sequences=None):
    """
    Creates the final merged file from individual sequence files and cached sequences.
    """
    logger.info(f"🔄 Creating final merged file from individual sequences...")

    all_sequences = []
    if cached_sequences:
        all_sequences.extend(cached_sequences)

    # Load all individual files
    individual_files = glob.glob(os.path.join(output_dir, "*.pt"))
    logger.info(f"📂 Found {len(individual_files)} individual sequence files")

    for individual_file in tqdm(individual_files, desc="📖 Loading sequences", unit="files"):
        try:
            data = torch.load(individual_file, weights_only=False)
            all_sequences.append(data)
        except Exception as e:
            logger.warning(f"⚠️ Failed to load {individual_file}: {e}")

    # Save merged file
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        torch.save({'sequences': all_sequences, 'created_at': time.time()}, output_file)
        logger.info(f"✅ Saved {len(all_sequences)} sequences to '{output_file}'")

        # Clean up individual files
        logger.info("🧹 Cleaning up individual files...")
        for individual_file in individual_files:
            try:
                os.remove(individual_file)
            except Exception as e:
                logger.warning(f"⚠️ Failed to remove {individual_file}: {e}")

        return len(all_sequences)

    except Exception as e:
        logger.error(f"❌ Failed to create merged file: {e}")
        return 0


def main(midi_dir="dataset/midi", output_file="dataset/processed/markov_sequences.pt", cache_dir="output/trained_models", max_files=None):
    """
    Scans a directory for MIDI files, processes them in parallel using a robust
    multiprocessing pool, and saves the extracted note sequences to a PyTorch file
    using a memory-efficient streaming approach.

    Args:
        midi_dir: Directory containing MIDI files
        output_file: Output file for processed sequences
        cache_dir: Directory for cache files
        max_files: Maximum number of files to process (optional)
    """
    import sys
    # Parse command line arguments
    if len(sys.argv) > 1:
        import argparse
        parser = argparse.ArgumentParser(description='MIDI preprocessing script')
        parser.add_argument('--midi-dir', default=midi_dir, help='Directory containing MIDI files')
        parser.add_argument('--output-file', default=output_file, help='Output file for processed sequences')
        parser.add_argument('--cache-dir', default=cache_dir, help='Directory for cache files')
        parser.add_argument('--max-files', type=int, help='Maximum number of files to process')

        args = parser.parse_args()
        midi_dir = args.midi_dir
        output_file = args.output_file
        cache_dir = args.cache_dir
        max_files = args.max_files

    logger.info("🚀 Starting ROBUST parallel MIDI preprocessing script with STREAMING.")
    
    # --- 1. Setup Stop Flag ---
    stop_file = ".stop_preprocessing"
    if os.path.exists(stop_file):
        os.remove(stop_file)
    logger.info(f"✅ To gracefully stop, create a file named '{stop_file}' in the root directory.")
    logger.info(f"(You can run this command in a separate terminal: touch {stop_file})")
    
    # Setup output directory for individual files
    output_dir = os.path.join(os.path.dirname(output_file), "individual_sequences")
    os.makedirs(output_dir, exist_ok=True)

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

    # Apply max_files limit if specified
    if max_files and max_files > 0:
        midi_files = midi_files[:max_files]
        logger.info(f"📏 Limited to processing {len(midi_files)} files (--max-files={max_files})")

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

    # --- 4. Memory-Efficient Streaming Processing with Worker Recycling ---
    # SAFETY CHECK: Don't start if memory is already too high
    initial_mem = get_memory_usage()
    if initial_mem > MAX_MEMORY_PERCENT:
        logger.error(f"❌ Initial memory usage too high ({initial_mem:.1f}%). Please free up memory before running preprocessing.")
        return

    logger.info(f"🧠 Initial memory usage: {initial_mem:.1f}%")

    total_failed = 0
    start_time = time.time()
    processed_paths_in_session = set()
    files_processed = 0

    logger.info(f"🚀 Starting ultra-fast streaming processing with worker recycling...")
    logger.info(f"⚡ Max workers: {MAX_WORKERS} (conservative for memory safety)")
    logger.info(f"🧠 Memory limit: {MAX_MEMORY_PERCENT}% (strict limit to prevent crashes)")
    logger.info(f"🔄 Worker restart every {WORKER_RESTART_INTERVAL} files")
    logger.info(f"🧹 Memory cleanup every {MEMORY_CLEANUP_INTERVAL} files")
    logger.info(f"🗑️ Force GC every {FORCE_GC_EVERY_N_FILES} files")

    # Warn if settings might be too aggressive
    if MAX_WORKERS > 16:
        logger.warning(f"⚠️ MAX_WORKERS ({MAX_WORKERS}) is very high! This may cause memory issues.")
    if STREAMING_BATCH_SIZE > 500:
        logger.warning(f"⚠️ STREAMING_BATCH_SIZE ({STREAMING_BATCH_SIZE}) is large! This may cause memory spikes.")

    pbar = tqdm(total=len(files_needing_processing), desc="🎵 Processing MIDI", unit="files")

    try:
        # Process files in chunks with worker recycling (simplified)
        chunk_size = STREAMING_BATCH_SIZE
        for i in range(0, len(files_needing_processing), chunk_size):
            # Check memory before starting chunk
            if not check_memory_before_chunk():
                logger.warning(f"⏭️ Skipping chunk {i//chunk_size + 1} due to high memory usage")
                continue

            chunk = files_needing_processing[i:i + chunk_size]
            logger.info(f"📦 Processing chunk {i//chunk_size + 1}/{(len(files_needing_processing) + chunk_size - 1)//chunk_size} ({len(chunk)} files)")

            # Create fresh pool for each chunk to prevent memory leaks
            current_workers = get_optimal_workers()
            logger.debug(f"Using {current_workers} workers for chunk processing")

            with create_memory_efficient_pool(current_workers) as pool:
                results_iterator = pool.imap_unordered(process_file, chunk, chunksize=1)

                for result in results_iterator:
                    # Check for stop file
                    if os.path.exists(stop_file):
                        logger.warning(f"\n🛑 Stop file detected. Saving progress...")
                        break

                    path, sequences = result
                    files_processed += 1

                    if sequences and isinstance(sequences, dict) and 'full' in sequences:
                        # Success case - SAVE IMMEDIATELY
                        try:
                            # Create individual file for this sequence
                            file_hash = hashlib.md5(path.encode()).hexdigest()[:8]
                            individual_file = os.path.join(output_dir, f"{file_hash}.pt")

                            # Wrap in expected format for saving
                            data_to_save = {
                                'path': path,
                                'sequences': sequences,
                                'metadata': {
                                    'track_count': sequences.get('track_count', 1),
                                    'has_bass': len(sequences.get('bass', [])) > 0,
                                    'has_drums': len(sequences.get('drums', [])) > 0,
                                    'has_melody': len(sequences.get('melody', [])) > 0,
                                    'processed_at': time.time(),
                                    'fast_mode': FAST_MODE
                                }
                            }

                            # Use memory-efficient saving
                            torch.save(data_to_save, individual_file, pickle_protocol=4)
                            processed_paths_in_session.add(path)
                            save_to_cache(path, sequences, cache_file)

                            logger.debug(f"💾 Saved {os.path.basename(path)} -> {os.path.basename(individual_file)}")

                        except Exception as e:
                            logger.error(f"❌ Failed to save {path}: {e}")
                            total_failed += 1
                    else:
                        # Failed to process
                        total_failed += 1
                        logger.debug(f"❌ Failed to process {os.path.basename(path)}")

                    pbar.update(1)

                    # Ultra-aggressive memory management
                    if files_processed % FORCE_GC_EVERY_N_FILES == 0:
                        gc.collect()  # Force GC every few files

                    if files_processed % MEMORY_CLEANUP_INTERVAL == 0:
                        check_memory_and_cleanup(force=True)

            # Force aggressive cleanup between chunks
            force_memory_cleanup()

            # Check if we should stop
            if os.path.exists(stop_file):
                break

    except KeyboardInterrupt:
        logger.warning("\n🛑 Keyboard interrupt detected. Stopping and saving progress...")
    finally:
        pbar.close()
        # --- 5. Final Merge and Cleanup ---
        duration = time.time() - start_time
        logger.info(f"⏱️ Processing finished in {duration:.2f} seconds.")

        # Create final merged file from individual files
        total_saved = create_final_merged_file(output_file, output_dir, cached_sequences)

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
                data = torch.load(output_file, weights_only=False)
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
