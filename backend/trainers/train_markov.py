import os
import sys
import gc
import logging
import warnings
import traceback
import psutil
import numpy as np
import torch
from tqdm import tqdm

from backend.models.markov_chain import MarkovChain

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

def log_memory_usage(stage=""):
    """Log current memory usage."""
    mem = psutil.virtual_memory()
    logger.info(f"🧠 Memory {stage}: {mem.percent:.1f}% used ({mem.used/1e9:.2f}GB / {mem.total/1e9:.2f}GB available)")
    if mem.percent > 85:
        logger.warning(f"⚠️ Memory usage is very high ({mem.percent:.1f}%)!")

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

def train_markov_model(order=3, max_interval=12, output_dir="output/trained_models", 
                       n_hidden_states=16, use_gpu=True):
    """
    Trains the Markov chain model by loading pre-processed note sequences.
    """
    gpu_info = detect_gpu_capabilities()
    use_gpu = use_gpu and gpu_info['cuda_available']
    
    logger.info(f"🚀 Initializing Markov model (order={order}, hidden_states={n_hidden_states})")
    model = MarkovChain(
        order=order, 
        max_interval=max_interval, 
        n_hidden_states=n_hidden_states,
        use_gpu=use_gpu
    )
    
    # --- 1. Load Pre-processed Data ---
    processed_data_path = "dataset/processed/markov_sequences.pt"
    logger.info(f"💾 Loading pre-processed data from '{processed_data_path}'...")

    if not os.path.exists(processed_data_path):
        logger.error(f"❌ Processed data file not found at '{processed_data_path}'.")
        logger.error("Please run the preprocessing script first: python scripts/preprocess_dataset.py")
        return None

    try:
        data = torch.load(processed_data_path)
        note_sequences = [item['sequence'] for item in data['sequences']]
        logger.info(f"✅ Successfully loaded {len(note_sequences)} sequences.")
    except Exception as e:
        logger.error(f"❌ Failed to load or parse pre-processed data: {e}\n{traceback.format_exc()}")
        return None

    if not note_sequences:
        logger.error("❌ No valid note sequences found in the pre-processed file.")
        return None

    log_memory_usage("after loading sequences")
    
    # --- 2. Train the Model ---
    logger.info("🧠 Training Markov model with HMM...")
    progress_bar = tqdm(total=100, desc="🚀 Training", unit="%")
    
    def update_progress(percent):
        current = int(percent * 100)
        progress_bar.update(current - progress_bar.n)
    
    training_success = model.train(note_sequences, progress_callback=update_progress)
    progress_bar.close()
    
    # --- 3. Clean Up Memory ---
    logger.info("🧹 Cleaning up memory...")
    demo_sequences = model.musical_features.get('demo_sequences', [])
    del note_sequences
    gc.collect()
    log_memory_usage("after training cleanup")
    
    if not training_success:
        logger.error("❌ Training failed.")
        return None

    # --- 4. Demonstrate HMM Algorithms ---
    demonstrate_hmm_algorithms(model, demo_sequences)
    
    # --- 5. Save the Model ---
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "enhanced_markov.npy")
    backup_path = os.path.join(output_dir, f"enhanced_markov_backup_{order}_{n_hidden_states}.npy")
    
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
    
    return model

if __name__ == "__main__":
    # Simplified parameter parsing
    order = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    max_interval = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    n_hidden_states = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    use_gpu = sys.argv[4].lower() != 'false' if len(sys.argv) > 4 else True
    
    logger.info("🚀 MARKOV TRAINING INITIATED 🚀")
    logger.info(f"🧠 Order: {order}")
    logger.info(f"🎵 Max interval: {max_interval}")
    logger.info(f"🔮 Hidden states: {n_hidden_states}")
    logger.info(f"⚡ GPU acceleration: {'ENABLED' if use_gpu else 'DISABLED'}")
    
    train_markov_model(
        order=order, 
        max_interval=max_interval,
        n_hidden_states=n_hidden_states,
        use_gpu=use_gpu
    )