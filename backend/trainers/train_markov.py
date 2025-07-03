import os
import glob
import multiprocessing
import warnings
import sys
import numpy as np
import torch
from music21 import converter, environment, note, chord
from tqdm import tqdm
import logging
from backend.models.markov_chain import MarkovChain

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress Music21 warnings
warnings.filterwarnings("ignore")
environment.Environment().warn = False

# CUDA GPU detection and optimization
def detect_gpu_capabilities():
    """Detect and configure GPU capabilities"""
    gpu_info = {
        'cuda_available': torch.cuda.is_available(),
        'cupy_available': False,
        'device_count': 0,
        'memory_gb': 0
    }
    
    if gpu_info['cuda_available']:
        gpu_info['device_count'] = torch.cuda.device_count()
        gpu_info['memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"🚀 CUDA detected: {gpu_info['device_count']} device(s), {gpu_info['memory_gb']:.1f}GB memory")
        
        # Try to import CuPy for advanced GPU operations
        try:
            import cupy as cp
            gpu_info['cupy_available'] = True
            logger.info("✅ CuPy detected - Advanced GPU acceleration enabled")
        except ImportError:
            logger.warning("⚠️ CuPy not found - Install with: pip install cupy-cuda11x")
    else:
        logger.warning("❌ CUDA not available - Training will use CPU")
    
    return gpu_info

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
    except Exception:
        return None

def extract_note_sequence_from_score(score):
    """Convert a music21 score to a sequence of (note, duration) pairs"""
    note_sequence = []
    try:
        # Flatten the score and process each note
        for element in score.flatten().notes:
            if isinstance(element, note.Note):
                # Add the note's MIDI pitch and duration
                note_sequence.append((element.pitch.midi, float(element.duration.quarterLength)))
            elif isinstance(element, chord.Chord):
                # For chords, add each pitch separately with the same duration
                for pitch in element.pitches:
                    note_sequence.append((pitch.midi, float(element.duration.quarterLength)))
    except Exception as e:
        logger.debug(f"Error extracting note sequence: {e}")
    
    return note_sequence

def process_midi_file_enhanced(file_path):
    """Enhanced MIDI file processing with better error handling and feature extraction"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Using no21 environment to suppress errors
            us = environment.UserSettings()
            us['warnings'] = 0
            score = converter.parse(file_path)
            
            # Basic validation
            if score and len(score.flatten().notes) > 0:
                return score
            else:
                return None
                
    except Exception:
        return None

def extract_enhanced_note_sequence_gpu_batch(scores_batch):
    """GPU-accelerated batch processing of note sequences"""
    sequences = []
    
    for score in scores_batch:
        if score is None:
            continue
            
        try:
            sequence = extract_enhanced_note_sequence(score)
            if sequence and len(sequence) >= 4:
                sequences.append(sequence)
        except Exception as e:
            logger.debug(f"Error in batch processing: {e}")
            continue
    
    return sequences

def worker_batch_process(batch_data):
    """Worker function for parallel batch processing - must be at module level for pickle"""
    batch_paths, enhanced_features = batch_data
    import warnings
    from music21 import converter
    
    sequences = []
    for path in batch_paths:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score = converter.parse(path)
                if enhanced_features:
                    seq = extract_enhanced_note_sequence(score)
                else:
                    seq = extract_note_sequence_from_score(score)
                if seq and len(seq) >= 4:
                    sequences.append(seq)
        except Exception:
            # Silently skip failed files to avoid log spam
            continue
    return sequences

def parallel_sequence_extraction_gpu(file_paths, enhanced_features=True, use_gpu=True):
    """Safe parallel sequence extraction: pass file paths, not music21 objects. Limit workers. Log memory usage."""
    import psutil
    logger.info("🚀 Starting safe parallel sequence extraction (file-path based)...")

    # Reduce workers even more to prevent OOM
    max_workers = min(2, max(1, multiprocessing.cpu_count() // 2))
    batch_size = min(25, max(5, len(file_paths) // (max_workers * 4)))
    logger.info(f"⚡ Using {max_workers} workers with batch size {batch_size}")

    # Split file paths into batches with enhanced_features flag
    batches = []
    for i in range(0, len(file_paths), batch_size):
        batch_paths = file_paths[i:i + batch_size]
        batches.append((batch_paths, enhanced_features))
    
    all_sequences = []

    with multiprocessing.Pool(processes=max_workers) as pool:
        batch_results = list(tqdm(
            pool.imap(worker_batch_process, batches),
            total=len(batches),
            desc="🎼 Parallel extraction",
            unit="batches"
        ))
    
    for batch_sequences in batch_results:
        all_sequences.extend(batch_sequences)

    # Log memory usage
    mem = psutil.virtual_memory()
    logger.info(f"🧠 Memory used: {mem.percent}% ({mem.used/1e9:.2f}GB/{mem.total/1e9:.2f}GB)")
    return all_sequences

def extract_enhanced_note_sequence(score):
    """Enhanced note sequence extraction with velocity, timing, and harmonic information"""
    note_sequence = []
    try:
        # Flatten the score and process each element
        flattened = score.flatten()
        
        for element in flattened.notes:
            if isinstance(element, note.Note):
                # Extract enhanced features
                pitch_midi = element.pitch.midi
                duration = float(element.duration.quarterLength)
                
                # Extract velocity if available
                velocity = getattr(element, 'velocity', 64)  # Default MIDI velocity
                
                # Extract timing information
                offset = float(element.offset) if hasattr(element, 'offset') else 0.0
                
                # Store as tuple with enhanced information
                note_data = (pitch_midi, duration, velocity, offset)
                note_sequence.append(note_data)
                
            elif isinstance(element, chord.Chord):
                # For chords, add each pitch with shared timing
                duration = float(element.duration.quarterLength)
                velocity = getattr(element, 'velocity', 64)
                offset = float(element.offset) if hasattr(element, 'offset') else 0.0
                
                for pitch in element.pitches:
                    note_data = (pitch.midi, duration, velocity, offset)
                    note_sequence.append(note_data)
                    
        # Sort by offset to ensure chronological order
        note_sequence.sort(key=lambda x: x[3])  # Sort by offset
        
    except Exception as e:
        logger.debug(f"Error extracting enhanced sequence: {e}")
    
    return note_sequence

def demonstrate_hmm_algorithms(model, sequences):
    """Demonstrate the HMM algorithms with sample sequences"""
    logger.info("🔬 Running HMM Algorithm Demonstrations...")
    
    if not sequences or not model.hmm_model:
        logger.warning("⚠️ No sequences or HMM model available for demonstration")
        return
    
    try:
        # Select a representative sequence
        demo_sequence = sequences[0] if sequences else None
        if not demo_sequence or len(demo_sequence) < 10:
            logger.warning("⚠️ Demo sequence too short, skipping HMM demonstration")
            return
            
        # Extract features for demonstration
        features = model._sequence_to_features(demo_sequence)
        if not features:
            logger.warning("⚠️ Could not extract features for demonstration")
            return
            
        logger.info(f"📊 Demo sequence length: {len(demo_sequence)} notes")
        logger.info(f"📊 Feature vectors: {len(features)}")
        
        # 1. Forward Algorithm Demonstration
        logger.info("🔄 Running Forward Algorithm...")
        alpha, log_likelihood = model.forward_algorithm(features)
        if alpha is not None:
            logger.info(f"✅ Forward Algorithm - Log Likelihood: {log_likelihood:.4f}")
            logger.info(f"📈 Forward probabilities shape: {alpha.shape}")
        else:
            logger.warning("⚠️ Forward Algorithm failed")
        
        # 2. Backward Algorithm Demonstration
        logger.info("🔄 Running Backward Algorithm...")
        beta = model.backward_algorithm(features)
        if beta is not None:
            logger.info(f"✅ Backward Algorithm completed")
            logger.info(f"📈 Backward probabilities shape: {beta.shape}")
        else:
            logger.warning("⚠️ Backward Algorithm failed")
        
        # 3. Forward-Backward Algorithm Demonstration
        logger.info("🔄 Running Forward-Backward Algorithm...")
        gamma, xi = model.forward_backward_algorithm(features)
        if gamma is not None and xi is not None:
            logger.info(f"✅ Forward-Backward Algorithm completed")
            logger.info(f"📈 State posteriors (gamma) shape: {gamma.shape}")
            logger.info(f"📈 Transition posteriors (xi) shape: {xi.shape}")
            
            # Show most likely states
            most_likely_states = np.argmax(gamma, axis=1)
            logger.info(f"🎯 Most likely state sequence: {most_likely_states[:10]}...")
        else:
            logger.warning("⚠️ Forward-Backward Algorithm failed")
        
        # 4. Viterbi Algorithm Demonstration
        logger.info("🔄 Running Viterbi Algorithm...")
        state_sequence, log_prob = model.viterbi_algorithm(features)
        if state_sequence is not None:
            logger.info(f"✅ Viterbi Algorithm - Log Probability: {log_prob:.4f}")
            logger.info(f"🎯 Optimal state sequence: {state_sequence[:10]}...")
            logger.info(f"📊 State sequence length: {len(state_sequence)}")
            
            # Show state distribution
            unique_states, counts = np.unique(state_sequence, return_counts=True)
            logger.info(f"📊 State distribution: {dict(zip(unique_states, counts))}")
        else:
            logger.warning("⚠️ Viterbi Algorithm failed")
        
        # 5. Baum-Welch Algorithm Demonstration
        logger.info("🔄 Running Baum-Welch Algorithm (limited iterations)...")
        trained_model, convergence = model.baum_welch_algorithm(features, max_iterations=5)
        if trained_model is not None and convergence:
            logger.info(f"✅ Baum-Welch Algorithm completed")
            logger.info(f"📈 Convergence history: {[f'{ll:.4f}' for ll in convergence[-3:]]}")
            
            # Compare before/after likelihood
            initial_ll = convergence[0] if convergence else 0
            final_ll = convergence[-1] if convergence else 0
            improvement = final_ll - initial_ll
            logger.info(f"📊 Likelihood improvement: {improvement:.4f}")
        else:
            logger.warning("⚠️ Baum-Welch Algorithm failed")
        
        # 6. Prediction Demonstration
        logger.info("🔄 Running HMM-based Prediction...")
        test_sequence = demo_sequence[:len(demo_sequence)//2]  # Use first half
        predicted_note, confidence = model.predict_next_note_hmm(test_sequence)
        
        logger.info(f"🎵 Input sequence (last 5 notes): {[n[0] if isinstance(n, (list, tuple)) else n for n in test_sequence[-5:]]}")
        logger.info(f"🎯 Predicted next note: {predicted_note}")
        logger.info(f"📊 Prediction confidence: {confidence:.3f}")
        
        # Compare with actual next note if available
        if len(demo_sequence) > len(test_sequence):
            actual_next = demo_sequence[len(test_sequence)]
            actual_note = actual_next[0] if isinstance(actual_next, (list, tuple)) else actual_next
            logger.info(f"🎵 Actual next note: {actual_note}")
            error = abs(predicted_note - actual_note)
            logger.info(f"📊 Prediction error: {error} semitones")
        
        logger.info("✅ HMM Algorithm demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ HMM demonstration failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())

def train_markov_model(midi_dir="dataset/midi", order=3, max_interval=12, output_dir="output/trained_models", 
                       n_hidden_states=16, use_gpu=True, enhanced_features=True):
    """Enhanced Markov chain model training with HMM, GPU acceleration, and hyperoptimization"""
    
    # Detect GPU capabilities
    gpu_info = detect_gpu_capabilities()
    
    # Force GPU usage if available with safer checks
    if gpu_info['cuda_available'] and use_gpu:
        try:
            torch.cuda.empty_cache()  # Clear GPU memory
            # Test a simple GPU operation
            test_tensor = torch.tensor([1.0]).cuda()
            _ = test_tensor + 1  # Simple operation
            logger.info(f"🔥 GPU acceleration ENABLED - {gpu_info['memory_gb']:.1f}GB VRAM available")
        except Exception as e:
            logger.warning(f"⚠️ GPU test failed ({e}) - falling back to CPU")
            use_gpu = False
    else:
        use_gpu = False
        logger.warning("⚠️ GPU acceleration DISABLED - training will be slower")
    
    logger.info(f"🚀 Initializing ENHANCED Markov model (order={order}, max_interval={max_interval}, hidden_states={n_hidden_states})")
    model = MarkovChain(
        order=order, 
        max_interval=max_interval, 
        n_hidden_states=n_hidden_states,
        use_gpu=use_gpu
    )
    
    # Scan for MIDI files efficiently with progress
    logger.info("🔍 Scanning for MIDI files...")
    midi_files = []
    
    # Use glob for faster file finding
    patterns = ['**/*.mid', '**/*.midi', '**/*.MID', '**/*.MIDI']
    for pattern in patterns:
        found_files = glob.glob(os.path.join(midi_dir, pattern), recursive=True)
        midi_files.extend(found_files)
    
    # Remove duplicates and sort
    midi_files = sorted(list(set(midi_files)))
    
    if not midi_files:
        logger.error(f"❌ No MIDI files found in {midi_dir}")
        return None
    
    logger.info(f"✅ Found {len(midi_files)} MIDI files")
    
    # Optimize multiprocessing for GPU systems
    if gpu_info['cuda_available']:
        # Use fewer CPU processes to leave resources for GPU
        cpu_count = max(2, multiprocessing.cpu_count() // 2)
        chunk_size = max(10, min(50, len(midi_files) // (cpu_count * 4)))
    else:
        # Use more CPU processes when no GPU
        cpu_count = max(1, multiprocessing.cpu_count() - 1)
        chunk_size = max(1, min(100, len(midi_files) // (cpu_count * 2)))
    
    logger.info(f"🔧 Processing with {cpu_count} CPU cores, chunk size: {chunk_size}")
    
    # Process files sequentially to avoid memory issues
    logger.info("⚡ Processing MIDI files sequentially (memory-safe)...")
    scores = []
    failed_count = 0
    
    for file_path in tqdm(midi_files, desc="🎵 Processing MIDI", unit="files"):
        try:
            score = process_midi_file_enhanced(file_path)
            if score is not None:
                scores.append(score)
            else:
                failed_count += 1
        except Exception as e:
            failed_count += 1
            logger.debug(f"Failed to process {file_path}: {e}")
        
        # Clean up memory every 50 files
        if len(scores) % 50 == 0:
            import gc
            gc.collect()
    
    if not scores:
        logger.error("❌ No valid scores were processed")
        return None
    
    success_rate = len(scores) / len(midi_files) * 100
    logger.info(f"✅ Successfully processed {len(scores)}/{len(midi_files)} files ({success_rate:.1f}% success rate)")
    
    # Convert scores to enhanced note sequences (memory-safe sequential processing)
    logger.info("🎼 Converting scores to enhanced note sequences (sequential processing)...")
    note_sequences = []
    
    for i, score in enumerate(tqdm(scores, desc="🎵 Extracting sequences", unit="scores")):
        try:
            if enhanced_features:
                sequence = extract_enhanced_note_sequence(score)
            else:
                sequence = extract_note_sequence_from_score(score)
            
            if sequence and len(sequence) >= 4:
                note_sequences.append(sequence)
                
        except Exception as e:
            logger.debug(f"Error extracting sequence from score {i}: {e}")
        
        # Clean up memory every 25 sequences
        if i % 25 == 0:
            import gc
            gc.collect()
    
    if not note_sequences:
        logger.error("❌ No valid note sequences could be extracted")
        return None
    
    avg_length = sum(len(seq) for seq in note_sequences) / len(note_sequences)
    logger.info(f"🎯 Extracted {len(note_sequences)} sequences (avg length: {avg_length:.1f} notes)")
    
    # Enhanced training with progress tracking
    logger.info("🧠 Training ENHANCED Markov model with HMM...")
    progress_bar = tqdm(total=100, desc="🚀 Training", unit="%")
    
    def update_progress(percent):
        current = int(percent * 100)
        progress_bar.update(current - progress_bar.n)
    
    # Train the enhanced model
    training_success = model.train(note_sequences, progress_callback=update_progress)
    progress_bar.close()
    
    if not training_success:
        logger.error("❌ Training failed")
        return None

    # Demonstrate HMM algorithms after training
    logger.info("🔬 Demonstrating HMM algorithms...")
    demonstrate_hmm_algorithms(model, note_sequences[:5])  # Use first 5 sequences for demo
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "enhanced_markov.npy")
    
    # Save enhanced model
    try:
        model.save(output_path)
        logger.info(f"💾 Enhanced model saved to {output_path}")
        
        # Also save a backup
        backup_path = os.path.join(output_dir, f"enhanced_markov_backup_{order}_{n_hidden_states}.npy")
        model.save(backup_path)
        logger.info(f"🔒 Backup saved to {backup_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save model: {e}")
        # Try an alternate save location
        try:
            alt_path = os.path.join(os.path.dirname(output_dir), "enhanced_markov_fallback.npy")
            model.save(alt_path)
            logger.info(f"💾 Model saved to alternate location: {alt_path}")
        except Exception as e2:
            logger.error(f"❌ All save attempts failed: {e2}")
    
    # Enhanced model statistics
    logger.info("\n" + "="*60)
    logger.info("🎼 ENHANCED MODEL STATISTICS")
    logger.info("="*60)
    
    # Core transition statistics
    if hasattr(model.transitions, 'shape'):
        non_zero = np.count_nonzero(model.transitions)
        total_possible = model.transitions.shape[0] * model.transitions.shape[1]
        sparsity = (1 - non_zero / total_possible) * 100
        logger.info(f"🎯 Note transitions: {model.transitions.shape} ({non_zero:,} non-zero, {sparsity:.1f}% sparse)")
    
    # Higher-order transitions
    total_higher_order = sum(len(transitions) for transitions in model.higher_order_transitions.values())
    logger.info(f"🧠 Higher-order transitions: {total_higher_order:,} contexts")
    
    for order, transitions in model.higher_order_transitions.items():
        logger.info(f"   📊 Order-{order}: {len(transitions):,} contexts")
    
    # Interval transitions
    interval_transitions = len(model.interval_transitions)
    logger.info(f"🎵 Interval transitions: {interval_transitions:,}")
    
    # HMM statistics
    if model.hmm_model:
        logger.info(f"🔮 HMM: {model.n_hidden_states} hidden states (ENABLED)")
    else:
        logger.info("🔮 HMM: DISABLED (insufficient data)")
    
    # Musical features with enhanced reporting
    enhanced_features_list = [
        ("🎼 Common keys", model.musical_features['common_keys']),
        ("🎵 Chord progressions", model.musical_features['common_chord_progressions']),
        ("🥁 Rhythm patterns", model.musical_features['rhythm_patterns']),
        ("⏱️ Time signatures", model.musical_features['time_signatures']),
        ("🎭 Roman numeral transitions", model.musical_features['roman_numeral_transitions']),
        ("🎨 Melodic contours", model.musical_features.get('melodic_contours', {})),
        ("💪 Dynamic patterns", model.musical_features.get('dynamic_patterns', {})),
        ("🎯 Phrase boundaries", model.musical_features.get('phrase_boundaries', {}))
    ]
    
    for name, data in enhanced_features_list:
        if data:
            count = len(data)
            if hasattr(data, 'keys') and count > 0:
                examples = list(data.keys())[:3]
                examples_str = ", ".join(str(e) for e in examples)
                logger.info(f"{name}: {count:,} entries (e.g., {examples_str})")
            else:
                logger.info(f"{name}: {count:,} entries")
        else:
            logger.info(f"{name}: 0 entries")
    
    # Performance statistics
    if gpu_info['cuda_available'] and use_gpu:
        try:
            gpu_memory_used = torch.cuda.max_memory_allocated() / 1e9
            logger.info(f"🚀 GPU memory used: {gpu_memory_used:.2f}GB")
            torch.cuda.reset_peak_memory_stats()
        except:
            pass
    
    logger.info("="*60)
    logger.info("✅ ENHANCED MARKOV TRAINING COMPLETE!")
    logger.info("="*60)
    
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
    
    # Enhanced parameters with hyperoptimization
    order = int(sys.argv[2]) if len(sys.argv) > 2 else 4  # Increased default order
    max_interval = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    n_hidden_states = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    use_gpu = sys.argv[5].lower() != 'false' if len(sys.argv) > 5 else True
    
    logger.info("🚀 HYPEROPTIMIZED MARKOV TRAINING INITIATED 🚀")
    logger.info(f"📂 Dataset: {midi_dir}")
    logger.info(f"🧠 Order: {order} (Higher-order transitions)")
    logger.info(f"🎵 Max interval: {max_interval}")
    logger.info(f"🔮 Hidden states: {n_hidden_states}")
    logger.info(f"⚡ GPU acceleration: {'ENABLED' if use_gpu else 'DISABLED'}")
    
    train_markov_model(
        midi_dir=midi_dir, 
        order=order, 
        max_interval=max_interval,
        n_hidden_states=n_hidden_states,
        use_gpu=use_gpu,
        enhanced_features=True
    )