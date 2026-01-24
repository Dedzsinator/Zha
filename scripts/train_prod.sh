#!/bin/bash

# Full Production Training Script for Zha
# Trains all 3 models: Markov Chain, GOLC-VAE, and Transformer

# Exit on error
set -e

# Configuration
DATASET_PATH="dataset/processed/full_dataset.pt"
MIDI_DIR="dataset/midi"
OUTPUT_DIR="output/trained_models"
LOG_DIR="output/logs"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "========================================================"
echo "🚀 STARTING FULL PRODUCTION TRAINING PIPELINE"
echo "========================================================"

# 1. Preprocessing
echo ""
echo "--------------------------------------------------------"
echo "1️⃣  PREPROCESSING DATASET"
echo "--------------------------------------------------------"
if [ -f "$DATASET_PATH" ]; then
    echo "✅ Dataset already exists at $DATASET_PATH"
    echo "   (Delete it if you want to re-process from scratch)"
else
    echo "🔄 Processing MIDI files from $MIDI_DIR..."
    python3 scripts/preprocess_dataset.py --midi-dir "$MIDI_DIR" --output-file "$DATASET_PATH"
fi

# 2. Train Markov Model
echo ""
echo "--------------------------------------------------------"
echo "2️⃣  TRAINING MARKOV CHAIN MODEL"
echo "--------------------------------------------------------"
# Order=4, MaxInterval=12, HiddenStates=16, GPU=True, Track=full
export PYTHONPATH=$PYTHONPATH:.
python3 backend/trainers/train_markov.py 4 12 16 True full

# 3. Train GOLC-VAE
echo ""
echo "--------------------------------------------------------"
echo "3️⃣  TRAINING GOLC-VAE MODEL"
echo "--------------------------------------------------------"
python3 scripts/train_lightning.py \
    --model vae \
    --data_path "$DATASET_PATH" \
    --epochs 100 \
    --batch_size 64 \
    --latent_dim 128 \
    --lr 1e-4

# 4. Train Transformer
echo ""
echo "--------------------------------------------------------"
echo "4️⃣  TRAINING TRANSFORMER MODEL"
echo "--------------------------------------------------------"
python3 scripts/train_lightning.py \
    --model transformer \
    --data_path "$DATASET_PATH" \
    --epochs 100 \
    --batch_size 32 \
    --embed_dim 256 \
    --num_heads 4 \
    --num_layers 4 \
    --lr 1e-4

echo ""
echo "========================================================"
echo "✅ ALL MODELS TRAINED SUCCESSFULLY!"
echo "========================================================"
