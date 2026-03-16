#!/bin/bash
#SBATCH --job-name=zha_full_training
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%j.out

# Full Production Training Script for Zha (SLURM)
# Trains all models on amaai-lab/MidiCaps streamed from HuggingFace by default.
# Use --local to fall back to local preprocessed data.

set -e

echo "Hostname: $(hostname)"
echo "Date: $(date)"

OUTPUT_DIR="output/trained_models"
LOG_DIR="output/logs"
DATASET_PATH="dataset/processed/full_dataset.pt"
MIDI_DIR="dataset/midi"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "output/metrics"
export PYTHONPATH="${PYTHONPATH:-.}:."

# Auto-resume flags (only if checkpoints exist)
GOLC_RESUME_ARG=""
TRANSFORMER_RESUME_ARG=""

if [ -f "$OUTPUT_DIR/golc_vae_latest.pt" ]; then
    GOLC_RESUME_ARG="--resume"
    echo "  GOLC-VAE checkpoint found: will resume from latest"
fi

if [ -f "$OUTPUT_DIR/transformer_latest.pt" ]; then
    TRANSFORMER_RESUME_ARG="--resume"
    echo "  Transformer checkpoint found: will resume from latest"
fi

# Optional conda activation (set CONDA_ENV before sbatch if needed)
if [[ -n "${CONDA_ENV:-}" ]]; then
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV"
    elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV"
    else
        echo "WARNING: CONDA_ENV is set, but conda.sh was not found; continuing without activation."
    fi
fi

# GPU / CUDA check
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    echo " GPU detected: $GPU_NAME"
else
    echo "WARNING: No CUDA GPU detected — training will run on CPU (very slow)."
fi

# Optional --local flag skips HF and uses local preprocessed data
USE_HF="--hf"
if [[ "$1" == "--local" ]]; then
    USE_HF=""
    echo "ℹ  Using LOCAL preprocessed data"
    if [ ! -f "$DATASET_PATH" ]; then
        echo "Preprocessing MIDI files from $MIDI_DIR..."
        python3 scripts/preprocess_dataset.py --midi-dir "$MIDI_DIR" --output-file "$DATASET_PATH"
    fi
else
    echo "Using HuggingFace dataset (amaai-lab/MidiCaps, ALL genres)"
fi

echo ""
echo "========================================================"
echo " STARTING FULL PRODUCTION TRAINING PIPELINE"
echo "========================================================"

# 1. Markov
echo ""
echo "--------------------------------------------------------"
echo "1⃣  TRAINING MARKOV CHAIN MODEL"
echo "--------------------------------------------------------"
python3 -m backend.trainers.train_markov \
    --order 4 \
    --max-interval 12 \
    --hidden-states 16 \
    --track full \
    $USE_HF \
    2>&1 | tee "$LOG_DIR/markov.log"

# 2. GOLC-VAE
echo ""
echo "--------------------------------------------------------"
echo "3⃣  TRAINING GOLC-VAE MODEL"
echo "--------------------------------------------------------"
python3 -m backend.trainers.train_golc_vae \
    --epochs 100 \
    --batch-size 128 \
    --lr 1e-3 \
    $GOLC_RESUME_ARG \
    $USE_HF \
    2>&1 | tee "$LOG_DIR/golc_vae.log"

# 3. Transformer
echo ""
echo "--------------------------------------------------------"
echo "4⃣  TRAINING TRANSFORMER MODEL"
echo "--------------------------------------------------------"
python3 -m backend.trainers.train_transformer \
    $TRANSFORMER_RESUME_ARG \
    $USE_HF \
    2>&1 | tee "$LOG_DIR/transformer.log"

echo ""
echo "========================================================"
echo "ALL MODELS TRAINED SUCCESSFULLY!"
echo "   Metrics: output/metrics/"
echo "   Weights: output/trained_models/"
echo "   Logs:    output/logs/"
echo "========================================================"
