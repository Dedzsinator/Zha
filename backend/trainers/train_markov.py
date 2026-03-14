import os
import sys
import gc
import json
import logging
import warnings
import traceback
import time
import psutil
import numpy as np
import torch
from tqdm import tqdm

from backend.models.markov_chain import MarkovChain

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Reduce noisy third-party logs that are usually non-fatal for long HMM runs
logging.getLogger("hmmlearn").setLevel(logging.ERROR)
logging.getLogger("hmmlearn.base").setLevel(logging.ERROR)

# Suppress warnings
warnings.filterwarnings("ignore")

def log_memory_usage(stage=""):
    """Log current memory usage."""
    mem = psutil.virtual_memory()
    logger.info(f"🧠 Memory {stage}: {mem.percent:.1f}% used ({mem.used/1e9:.2f}GB / {mem.total/1e9:.2f}GB available)")
    if mem.percent > 85:
        logger.warning(f"⚠️ Memory usage is very high ({mem.percent:.1f}%)!")

def log_process_memory(stage=""):
    """Log current process RSS/VMS to make htop RES/VIRT interpretation explicit."""
    try:
        proc = psutil.Process(os.getpid())
        mem = proc.memory_info()
        logger.info(
            f"🧾 Process memory {stage}: RSS={mem.rss/1e9:.2f}GB, VMS={mem.vms/1e9:.2f}GB"
        )
    except Exception as e:
        logger.debug(f"Could not read process memory: {e}")

def detect_gpu_capabilities():
    """Detect and configure GPU capabilities."""
    gpu_info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': 0,
        'memory_gb': 0
    }
    
    if gpu_info['cuda_available']:
        gpu_info['device_count'] = torch.cuda.device_count()
        try:
            gpu_info['memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"🚀 CUDA detected: {gpu_info['device_count']} device(s), {gpu_info['memory_gb']:.1f}GB memory")
        except Exception as e:
            logger.error(f"Could not get GPU properties: {e}")
            gpu_info['cuda_available'] = False
    else:
        logger.warning("❌ CUDA not available - Training will use CPU")
    
    return gpu_info

def demonstrate_hmm_algorithms(model, sequences):
    """Demonstrate the HMM algorithms with sample sequences."""
    logger.info("🔬 Running HMM Algorithm Demonstrations...")
    
    if not sequences or not model.hmm_model:
        logger.warning("⚠️ No sequences or HMM model available for demonstration.")
        return
    
    try:
        demo_sequence = sequences[0]
        if len(demo_sequence) < 10:
            logger.warning("⚠️ Demo sequence too short, skipping HMM demonstration.")
            return
            
        features = model._sequence_to_features(demo_sequence)
        if not features:
            logger.warning("⚠️ Could not extract features for demonstration.")
            return
            
        logger.info(f"📊 Demo sequence length: {len(demo_sequence)} notes")
        
        # Viterbi Algorithm Demonstration
        logger.info("🔄 Running Viterbi Algorithm...")
        state_sequence, log_prob = model.viterbi_algorithm(features)
        if state_sequence is not None:
            logger.info(f"✅ Viterbi Algorithm - Log Probability: {log_prob:.4f}")
            logger.info(f"🎯 Optimal state sequence (first 10): {state_sequence[:10]}...")
        else:
            logger.warning("⚠️ Viterbi Algorithm failed.")
        
        logger.info("✅ HMM Algorithm demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ HMM demonstration failed: {e}")
        logger.debug(traceback.format_exc())

# ---------------------------------------------------------------------------
# Chord quality → semitone intervals (root-relative)
# ---------------------------------------------------------------------------
_QUALITY_INTERVALS = {
    'major':          [0, 4, 7],
    'minor':          [0, 3, 7],
    'dominant':       [0, 4, 7, 10],
    'major seventh':  [0, 4, 7, 11],
    'minor seventh':  [0, 3, 7, 10],
    'diminished':     [0, 3, 6],
    'augmented':      [0, 4, 8],
    'suspended':      [0, 5, 7],
    'half-diminished':[0, 3, 6, 10],
}
_NAME_TO_PC = {'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}

def _chord_to_notes(chord: dict, base_octaves=(4, 5)) -> list:
    """Convert a MidiCaps chord dict to a richer list of MIDI note numbers."""
    root_str = chord.get('root') or chord.get('Root') or 'C'
    bass_str = chord.get('bass') or chord.get('Bass') or root_str
    quality  = (chord.get('quality') or chord.get('Quality') or 'major').lower()

    # Parse root note → pitch class
    root_pc = _NAME_TO_PC.get(root_str[0].upper(), 0)
    if len(root_str) > 1:
        root_pc += root_str[1:].count('#') - root_str[1:].count('b')
    root_pc %= 12

    bass_pc = _NAME_TO_PC.get(bass_str[0].upper(), 0)
    if len(bass_str) > 1:
        bass_pc += bass_str[1:].count('#') - bass_str[1:].count('b')
    bass_pc %= 12

    intervals = _QUALITY_INTERVALS.get('major')  # fallback
    for qname, ivs in _QUALITY_INTERVALS.items():
        if qname in quality:
            intervals = ivs
            break

    notes = []
    for octv in base_octaves:
        root_midi = root_pc + octv * 12
        notes.extend(min(127, max(0, root_midi + i)) for i in intervals)

    # Add bass reinforcement in lower octave for better movement diversity
    bass_midi = bass_pc + 3 * 12
    notes.append(min(127, max(0, bass_midi)))

    return sorted(set(notes))


def _load_sequences_from_hf():
    """Stream amaai-lab/MidiCaps and build note+chord sequences for Markov training."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "The 'datasets' library is required.  pip install datasets"
        ) from e

    logger.info("🌐 Streaming from HuggingFace (amaai-lab/MidiCaps) — ALL genres...")
    ds = load_dataset("amaai-lab/MidiCaps", split="train", streaming=True)

    note_sequences = []
    chord_sequences_out = []
    track_count = 0
    rows_seen = 0
    log_every = 5000

    use_tqdm = sys.stdout.isatty()
    for row in tqdm(
        ds,
        desc="🌐 Loading from HF",
        disable=not use_tqdm,
        dynamic_ncols=True,
        mininterval=1.0,
    ):
        rows_seen += 1
        # Try multiple field names — MidiCaps uses "chords" in some versions
        all_chords = row.get("all_chords") or row.get("chords") or []
        if isinstance(all_chords, str):
            try:
                import json as _json
                all_chords = _json.loads(all_chords)
            except Exception:
                all_chords = []

        note_seq = []
        chord_seq = []

        # Build a note sequence by expanding each chord into its pitches
        for chord in (all_chords if isinstance(all_chords, list) else []):
            if not isinstance(chord, dict):
                continue
            notes = _chord_to_notes(chord)
            note_seq.extend(notes)
            chord_seq.append(tuple(sorted(notes)))

        # Fallback: synthesise a scale from key metadata when no chord data
        if len(note_seq) < 4:
            key_str = str(row.get("key") or "")
            tokens = key_str.split()
            if tokens:
                rname = tokens[0]
                rpc = _NAME_TO_PC.get(rname[0].upper(), 0)
                if len(rname) > 1:
                    rpc += rname[1:].count('#') - rname[1:].count('b')
                rpc %= 12
                is_minor = len(tokens) > 1 and 'minor' in tokens[1].lower()
                # Natural minor / major diatonic scale (2 octaves)
                ivs = [0, 2, 3, 5, 7, 8, 10] if is_minor else [0, 2, 4, 5, 7, 9, 11]
                base = 48 + rpc  # root at C4
                note_seq = [min(127, base + i) for i in ivs + [i + 12 for i in ivs]]
                chord_seq = []

        if len(note_seq) >= 4:   # minimum viable sequence for Markov
            note_sequences.append(note_seq)
            chord_sequences_out.append(chord_seq)
            track_count += 1

        if (not use_tqdm) and (rows_seen % log_every == 0):
            logger.info(
                f"🌐 HF streaming progress: rows_seen={rows_seen:,}, usable_sequences={track_count:,}"
            )

    logger.info(f"✅ Built {track_count:,} note sequences from HF dataset (rows_seen={rows_seen:,})")
    return note_sequences, chord_sequences_out, track_count


def _load_sequences_from_local(track_type: str):
    """Load note sequences from the local preprocessed .pt file."""
    processed_data_path = "dataset/processed/full_dataset.pt"
    logger.info(f"💾 Loading pre-processed data from '{processed_data_path}' (track: {track_type})...")

    if not os.path.exists(processed_data_path):
        old_path = "dataset/processed/markov_sequences.pt"
        if os.path.exists(old_path):
            logger.warning(f"⚠️ '{processed_data_path}' not found, falling back to '{old_path}'")
            processed_data_path = old_path
        else:
            logger.error(f"❌ Processed data file not found at '{processed_data_path}'.")
            logger.error("Run: python scripts/preprocess_dataset.py")
            return None

    try:
        data = torch.load(processed_data_path)
        sequences_container = data.get('sequences') if isinstance(data, dict) else data
        if sequences_container is None:
            sequences_container = data
        sequences_iter = sequences_container if isinstance(sequences_container, (list, tuple)) else []

        note_sequences = []
        chord_sequences = []
        track_count = 0

        for item in sequences_iter:
            if isinstance(item, dict) and 'sequences' in item and isinstance(item['sequences'], dict):
                seq_data = item['sequences']
                if track_type in seq_data and seq_data[track_type]:
                    note_sequences.append(seq_data[track_type])
                    track_count += 1
                elif track_type == 'full' and 'full' in seq_data and seq_data['full']:
                    note_sequences.append(seq_data['full'])
                    track_count += 1
                if 'chords' in seq_data and seq_data['chords']:
                    chord_seq = [tuple(sorted(c['pitches'])) for c in seq_data['chords'] if 'pitches' in c]
                    if chord_seq:
                        chord_sequences.append(chord_seq)

        logger.info(f"✅ Loaded {len(note_sequences)} {track_type} sequences from {track_count} files.")

        if len(note_sequences) == 0:
            logger.error(f"❌ No {track_type} sequences found in '{processed_data_path}'.")
            return None

        return note_sequences, chord_sequences, track_count

    except Exception as e:
        logger.error(f"❌ Failed to load pre-processed data: {e}\n{traceback.format_exc()}")
        return None


def train_markov_model(order=3, max_interval=12, output_dir="output/trained_models",
                       n_hidden_states=16, use_gpu=True, track_type='full',
                       use_huggingface=False):
    """
    Trains the Markov chain model by loading pre-processed note sequences.

    Args:
        track_type: Type of track to train on ('full', 'melody', 'bass', 'drums')
        use_huggingface: If True, stream note sequences from amaai-lab/MidiCaps
                         on HuggingFace instead of the local preprocessed file.
    """
    gpu_info = detect_gpu_capabilities()
    use_gpu = use_gpu and gpu_info['cuda_available']
    
    logger.info(f"🚀 Initializing Markov model (order={order}, hidden_states={n_hidden_states}, track={track_type})")
    model = MarkovChain(
        order=order, 
        max_interval=max_interval, 
        n_hidden_states=n_hidden_states,
        use_gpu=use_gpu
    )
    logger.info(
        f"⚙️ Markov backend: use_gpu={model.use_gpu}, device={model.device}, "
        f"note_transition_backend={'torch-cuda' if model.use_gpu else 'numpy-cpu'}"
    )
    logger.info(
        "ℹ️ Training mixes CPU-heavy counting loops with some GPU tensor ops; "
        "seeing one CPU core at high usage is expected."
    )

    # -----------------------------------------------------------------------
    # Data loading: HuggingFace streaming OR local preprocessed file
    # -----------------------------------------------------------------------
    if use_huggingface:
        note_sequences, chord_sequences, track_count = _load_sequences_from_hf()
    else:
        result = _load_sequences_from_local(track_type)
        if result is None:
            return None
        note_sequences, chord_sequences, track_count = result

    log_process_memory("after data load")
    
    # --- 2. Train the Model ---
    logger.info("🧠 Training Markov model with HMM...")
    use_tqdm = sys.stdout.isatty()
    progress_bar = tqdm(
        total=100,
        desc="🚀 Training",
        unit="%",
        disable=not use_tqdm,
        dynamic_ncols=True,
        mininterval=1.0,
    )
    start_time = time.time()
    last_logged_percent = -1
    logger.info("⏳ Markov training started (progress logs every 5%)")
    
    def update_progress(percent):
        nonlocal last_logged_percent
        current = int(percent * 100)
        progress_bar.update(current - progress_bar.n)
        if (not use_tqdm) and current >= 0:
            if current >= min(100, last_logged_percent + 5):
                elapsed = time.time() - start_time
                eta = (elapsed / current) * (100 - current) if current > 0 else 0.0
                logger.info(
                    f"🚀 Training progress: {current}% | elapsed={elapsed/60:.1f}m | eta={eta/60:.1f}m"
                )
                last_logged_percent = current
    
    markov_model_logger = logging.getLogger("backend.models.markov_chain")
    previous_markov_level = markov_model_logger.level
    markov_model_logger.setLevel(logging.WARNING)
    try:
        training_success = model.train(
            note_sequences,
            chord_sequences=chord_sequences,
            progress_callback=update_progress,
        )
    finally:
        markov_model_logger.setLevel(previous_markov_level)
    progress_bar.close()
    elapsed_total = time.time() - start_time
    logger.info(f"✅ Markov training phase finished in {elapsed_total/60:.1f} minutes")
    log_process_memory("after model.train")
    
    # --- 3. Clean Up Memory ---
    logger.info("🧹 Cleaning up memory...")
    demo_sequences = model.musical_features.get('demo_sequences', [])
    if not demo_sequences and note_sequences:
        # Keep a tiny sample for post-train HMM demonstration.
        demo_sequences = note_sequences[:2]
    del note_sequences
    gc.collect()
    log_memory_usage("after training cleanup")
    log_process_memory("after cleanup")
    
    if not training_success:
        logger.error("❌ Training failed.")
        return None

    # --- 4. Demonstrate HMM Algorithms ---
    demonstrate_hmm_algorithms(model, demo_sequences)
    
    # --- 5. Save the Model ---
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "markov.npy")
    backup_path = os.path.join(output_dir, f"markov_backup_{order}_{n_hidden_states}.npy")
    
    try:
        model.save(output_path)
        logger.info(f"💾 Model saved to {output_path}")
        model.save(backup_path)
        logger.info(f"🔒 Backup saved to {backup_path}")
    except Exception as e:
        logger.error(f"❌ Failed to save model: {e}")

    # --- 6. Display Model Statistics ---
    logger.info("\n" + "="*60 + "\n🎼 MODEL STATISTICS\n" + "="*60)
    if hasattr(model.transitions, 'shape'):
        non_zero = np.count_nonzero(model.transitions) if isinstance(model.transitions, np.ndarray) else model.transitions.to_sparse()._nnz()
        total_possible = model.transitions.shape[0] * model.transitions.shape[1]
        sparsity = (1 - non_zero / total_possible) * 100 if total_possible > 0 else 0
        logger.info(f"🎯 Note transitions: {model.transitions.shape} ({non_zero:,} non-zero, {sparsity:.1f}% sparse)")
    
    total_higher_order = sum(len(transitions) for transitions in model.higher_order_transitions.values())
    logger.info(f"🧠 Higher-order transitions: {total_higher_order:,} contexts")
    
    if model.hmm_model:
        logger.info(f"🔮 HMM: {model.n_hidden_states} hidden states (ENABLED)")
    else:
        logger.info("🔮 HMM: DISABLED")
        
    logger.info("="*60 + "\n✅ TRAINING COMPLETE!\n" + "="*60)
    
    # Export model statistics to JSON
    os.makedirs(output_dir, exist_ok=True)
    stats = {
        'order': order,
        'max_interval': max_interval,
        'n_hidden_states': n_hidden_states,
        'track_type': track_type,
        'source': 'huggingface' if use_huggingface else 'local',
        'sequences_loaded': track_count,
        'hmm_enabled': bool(model.hmm_model),
    }
    if hasattr(model, 'transitions') and hasattr(model.transitions, 'shape'):
        transitions_np = model.transitions if isinstance(model.transitions, np.ndarray) else model.transitions.cpu().numpy()

        non_zero = int(np.count_nonzero(transitions_np))
        total_possible = int(transitions_np.shape[0] * transitions_np.shape[1])
        stats['transition_matrix_size'] = list(transitions_np.shape)
        stats['transition_nonzero'] = non_zero
        stats['transition_sparsity_pct'] = round((1 - non_zero / total_possible) * 100, 2) if total_possible > 0 else 0

        # More informative sparsity over active notes only (rows/cols with non-zero probability mass).
        row_activity = transitions_np.sum(axis=1)
        col_activity = transitions_np.sum(axis=0)
        active_notes = np.union1d(np.where(row_activity > 0)[0], np.where(col_activity > 0)[0])
        active_note_count = int(len(active_notes))
        stats['active_note_count'] = active_note_count

        if active_note_count > 0:
            active_submatrix = transitions_np[np.ix_(active_notes, active_notes)]
            active_nonzero = int(np.count_nonzero(active_submatrix))
            active_total = int(active_note_count * active_note_count)
            stats['active_transition_nonzero'] = active_nonzero
            stats['active_transition_sparsity_pct'] = round((1 - active_nonzero / active_total) * 100, 2) if active_total > 0 else 0
    stats['higher_order_contexts'] = sum(len(v) for v in model.higher_order_transitions.values())
    metrics_path = os.path.join("output/metrics", "markov_metrics.json")
    os.makedirs("output/metrics", exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"📊 Metrics saved to {metrics_path}")
    
    return model

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--order",          type=int,   default=4)
    p.add_argument("--max-interval",   type=int,   default=12)
    p.add_argument("--hidden-states",  type=int,   default=16)
    p.add_argument("--no-gpu",         action="store_true")
    p.add_argument("--track",          type=str,   default="full")
    p.add_argument("--hf",             action="store_true",
                   help="Stream data from HuggingFace (amaai-lab/MidiCaps) instead of local file")
    args = p.parse_args()

    logger.info("🚀 MARKOV TRAINING INITIATED 🚀")
    logger.info(f"🧠 Order: {args.order}")
    logger.info(f"🎵 Max interval: {args.max_interval}")
    logger.info(f"🔮 Hidden states: {args.hidden_states}")
    logger.info(f"⚡ GPU acceleration: {'DISABLED' if args.no_gpu else 'ENABLED'}")
    logger.info(f"🎼 Track type: {args.track}")
    logger.info(f"🌐 Data source: {'HuggingFace' if args.hf else 'local'}")

    train_markov_model(
        order=args.order,
        max_interval=args.max_interval,
        n_hidden_states=args.hidden_states,
        use_gpu=not args.no_gpu,
        track_type=args.track,
        use_huggingface=args.hf,
    )