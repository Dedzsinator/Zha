#!/bin/bash

# Full Production Training Script for Zha
# Trains ALL models on the full amaai-lab/MidiCaps dataset (168k MIDI files,
# all genres) streamed directly from HuggingFace — no local download needed.
# Falls back to local preprocessed data if --local flag is passed.

set -e

OUTPUT_DIR="output/trained_models"
LOG_DIR="output/logs"
DATASET_PATH="dataset/processed/full_dataset.pt"
MIDI_DIR="dataset/midi"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "output/metrics"
export PYTHONPATH="${PYTHONPATH:-.}:."

# ── auto-resume flags (only if checkpoints exist) ───────────────────────────
VAE_RESUME_ARG=""
GOLC_RESUME_ARG=""
TRANSFORMER_RESUME_ARG=""

if [ -f "$OUTPUT_DIR/vae_latest.pt" ]; then
    VAE_RESUME_ARG="--resume"
    echo "♻️  VAE checkpoint found: will resume from latest"
fi

if [ -f "$OUTPUT_DIR/golc_vae_latest.pt" ]; then
    GOLC_RESUME_ARG="--resume"
    echo "♻️  GOLC-VAE checkpoint found: will resume from latest"
fi

if [ -f "$OUTPUT_DIR/transformer_latest.pt" ]; then
    TRANSFORMER_RESUME_ARG="--resume"
    echo "♻️  Transformer checkpoint found: will resume from latest"
fi

# ── GPU / CUDA check ──────────────────────────────────────────────────────────
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    echo "🟢 GPU detected: $GPU_NAME"
else
    echo "⚠️  WARNING: No CUDA GPU detected — training will run on CPU (very slow)."
    echo "   To train on GPU, install the CUDA-enabled torch wheel:"
    echo "   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126"
    echo "   (replace cu126 with your CUDA version — check with: nvidia-smi)"
    echo ""
    read -r -p "Continue on CPU anyway? [y/N] " _cpu_confirm
    [[ "$_cpu_confirm" =~ ^[Yy]$ ]] || exit 1
fi

# ── optional --local flag skips HF and uses local preprocessed data ──────────
USE_HF="--hf"
if [[ "$1" == "--local" ]]; then
    USE_HF=""
    echo "ℹ️  Using LOCAL preprocessed data"
    if [ ! -f "$DATASET_PATH" ]; then
        echo "🔄 Preprocessing MIDI files from $MIDI_DIR..."
        python3 scripts/preprocess_dataset.py --midi-dir "$MIDI_DIR" --output-file "$DATASET_PATH"
    fi
else
    echo "🌐 Using HuggingFace dataset (amaai-lab/MidiCaps, ALL genres)"
fi

echo ""
echo "========================================================"
echo "🚀 STARTING FULL PRODUCTION TRAINING PIPELINE"
echo "========================================================"

# ── 1. Markov ─────────────────────────────────────────────────────────────────
echo ""
echo "--------------------------------------------------------"
echo "1️⃣  TRAINING MARKOV CHAIN MODEL"
echo "--------------------------------------------------------"
python3 -m backend.trainers.train_markov \
    --order 4 \
    --max-interval 12 \
    --hidden-states 16 \
    --track full \
    $USE_HF \
    2>&1 | tee "$LOG_DIR/markov.log"

# ── 2. VAE ────────────────────────────────────────────────────────────────────
echo ""
echo "--------------------------------------------------------"
echo "2️⃣  TRAINING VAE MODEL"
echo "--------------------------------------------------------"
python3 -m backend.trainers.train_vae \
    $VAE_RESUME_ARG \
    $USE_HF \
    2>&1 | tee "$LOG_DIR/vae.log"

# ── 3. GOLC-VAE ───────────────────────────────────────────────────────────────
echo ""
echo "--------------------------------------------------------"
echo "3️⃣  TRAINING GOLC-VAE MODEL"
echo "--------------------------------------------------------"
python3 -m backend.trainers.train_golc_vae \
    --epochs 100 \
    --batch-size 128 \
    --lr 1e-3 \
    $GOLC_RESUME_ARG \
    $USE_HF \
    2>&1 | tee "$LOG_DIR/golc_vae.log"

# ── 4. Transformer ────────────────────────────────────────────────────────────
echo ""
echo "--------------------------------------------------------"
echo "4️⃣  TRAINING TRANSFORMER MODEL"
echo "--------------------------------------------------------"
python3 -m backend.trainers.train_transformer \
    $TRANSFORMER_RESUME_ARG \
    $USE_HF \
    2>&1 | tee "$LOG_DIR/transformer.log"

echo ""
echo "========================================================"
echo "✅ ALL MODELS TRAINED SUCCESSFULLY!"
echo "   Metrics: output/metrics/"
echo "   Weights: output/trained_models/"
echo "   Logs:    output/logs/"
echo "========================================================"