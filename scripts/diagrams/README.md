# Thesis Diagram Generation Guide

This directory contains scripts to generate all figures and diagrams for the thesis.

## 📊 Overview

**Total Placeholders:** 29 diagrams/figures across 5 chapters
- **Architecture diagrams:** 6 (no models required) ✅ READY
- **Markov diagrams:** 5 (requires Markov model)
- **VAE diagrams:** 6 (requires VAE model) ✅ READY
- **Transformer diagrams:** 9 (requires Transformer model)
- **System/Multi-track diagrams:** 3 (requires all models)

## 🚀 Quick Start

### Generate ALL diagrams (one command):
```bash
python scripts/generate_all_diagrams.py
```

### Generate specific chapter diagrams:
```bash
# Architecture diagrams (no training needed!)
python scripts/diagrams/generate_architecture_diagrams.py

# Markov diagrams
python scripts/diagrams/generate_markov_diagrams.py

# VAE diagrams
python scripts/diagrams/generate_vae_diagrams.py

# Transformer diagrams
python scripts/diagrams/generate_transformer_diagrams.py

# Multi-track diagrams
python scripts/diagrams/generate_multitrack_diagrams.py

# System integration diagrams
python scripts/diagrams/generate_system_diagrams.py
```

## 📋 Prerequisites

### Install Python packages:
```bash
pip install matplotlib seaborn numpy torch graphviz pillow music21 pandas
```

### Train models (if not already trained):
```bash
# Train all models
python backend/trainers/train_markov.py
python backend/trainers/train_vae.py
python backend/trainers/train_transformer.py

# Or use existing checkpoints in output/trained_models/
```

## 📁 Output Structure

```
output/figures/thesis/
├── architecture/          # 6 diagrams - READY TO GENERATE
│   ├── system_architecture.pdf
│   ├── transformer_architecture.pdf
│   ├── vae_architecture.pdf
│   ├── residual_block.pdf
│   ├── memory_mechanism.pdf
│   └── structured_generation_flowchart.pdf
├── markov/               # 5 diagrams
│   ├── stationary_distribution.pdf
│   ├── gpu_memory_usage.pdf
│   └── ... (+ 3 metric tables from generate_markov_metrics.py)
├── vae/                  # 6 diagrams - READY TO GENERATE
│   ├── reparameterization_trick.pdf
│   ├── training_curves.pdf
│   ├── latent_kl_distribution.pdf
│   └── temperature_ablation.pdf
├── transformer/          # 9 diagrams
│   └── ... (requires trained model)
├── multitrack/           # 3 diagrams
│   └── ... (requires trained multi-track model)
└── system/               # 3 diagrams
    └── ... (requires all models)
```

## 📝 Detailed Script Descriptions

### 1. generate_architecture_diagrams.py ✅ COMPLETE
**Status:** Ready to run (no training required)

**Generates:**
1. **System Architecture** (zha.tex line 306)
   - Full pipeline: MIDI → Markov/VAE/Transformer → Output
   - Weighted combination (0.5/0.3/0.2)
   - Scale filtering & register limiting
   - Multi-track branch

2. **Transformer Architecture** (transformer.tex line 124)
   - Input embedding → 8 Transformer layers → Output
   - Multi-head attention (8 heads)
   - Positional encoding (Rotary)
   - Residual connections

3. **VAE Encoder-Decoder** (vae.tex line 366)
   - Encoder: 128→512→256→latent
   - ResidualBlocks highlighted
   - Reparameterization in latent space
   - Decoder: latent→256→512→128

4. **ResidualBlock Detail** (vae.tex line 319)
   - LayerNorm → Linear → SiLU → Linear
   - Skip connection visualization
   - Formula: out = x + MLP(LN(x))

5. **Memory Mechanism** (transformer.tex line 178)
   - Section-based memory storage (Intro/Verse/Chorus/Bridge/Outro)
   - Memory recall for structural coherence
   - Timeline visualization

6. **Structured Generation Flowchart** (transformer.tex line 232)
   - Decision tree: Section in memory? → Recall vs Generate new
   - Apply variation → Store → Append
   - Loop for multiple sections

**Run:**
```bash
python scripts/diagrams/generate_architecture_diagrams.py
# Output: 6 PDF + 6 PNG files in output/figures/thesis/architecture/
```

### 2. generate_markov_diagrams.py
**Status:** Partially complete (uses simulated data if models not trained)

**Generates:**
1. **Stationary Distribution Histogram** (markov.tex line 50)
   - MIDI note equilibrium distribution
   - Color-coded by octave
   - Statistics: mean, std, entropy

2. **GPU Memory Usage Plot** (markov.tex line 188)
   - Memory timeline during training
   - Component breakdown (transition matrix, emission, buffers)
   - Spike detection during batch processing

**Also needs (from generate_markov_metrics.py):**
3. Entropy Rate Table (line 65) - LaTeX table
4. HMM Hidden State Interpretation (line 152) - LaTeX table
5. GPU Benchmark Table (line 180) - LaTeX table

**Run:**
```bash
# Generate diagrams
python scripts/diagrams/generate_markov_diagrams.py

# Generate metric tables
python scripts/generate_markov_metrics.py
```

### 3. generate_vae_diagrams.py ✅ COMPLETE
**Status:** Ready to run (uses simulated training data)

**Generates:**
1. **Reparameterization Trick Diagram** (vae.tex line 73)
   - Side-by-side comparison: without vs with reparameterization
   - Gradient flow visualization
   - Formula: z = μ + σ·ε, ε~N(0,1)

2. **Training Curves** (vae.tex line 109)
   - Total ELBO loss over epochs
   - Reconstruction loss vs KL divergence
   - Beta annealing schedule
   - Loss ratio (KL/Recon balance)

3. **Latent KL Distribution** (vae.tex line 116)
   - KL divergence per latent dimension (bar chart)
   - Grouped boxplot: Collapsed / Moderate / Active dims
   - Free bits threshold visualization
   - Statistics: active dims, collapsed dims

4. **Temperature Ablation** (vae.tex line 396)
   - 2D latent space projections at T=[0.5, 0.8, 1.0, 1.2, 1.5]
   - Contour plots showing sample distribution
   - Descriptions: Conservative vs Exploratory
   - Effect on creativity/coherence

**Run:**
```bash
python scripts/diagrams/generate_vae_diagrams.py
# Output: 4 PDF + 4 PNG files in output/figures/thesis/vae/
```

### 4. generate_transformer_diagrams.py
**Status:** Template created (needs implementation)

**Should generate:**
1. Sampling Strategies Comparison (line 293, 300, 310)
2. Attention Weight Heatmaps (line 334)
3. Attention Head Ablation Study (line 342)
4. Training Loss Curves (line 364)
5. Perplexity over Sequence Length (line 371)
6. Quality Metrics Table (line 391)

**TODO:** Implement visualization functions (see vae_diagrams.py for patterns)

### 5. generate_multitrack_diagrams.py
**Status:** Template created (needs implementation)

**Should generate:**
1. Multi-track Harmonic Coherence (transformer.tex line 910)
2. Drum Pattern Consistency (line 918)
3. Multi-track Generation Examples (line 926)

**TODO:** Implement multi-track visualizations

### 6. generate_system_diagrams.py
**Status:** Template created (needs implementation)

**Should generate:**
1. Generation Examples in Musical Notation (zha.tex line 315)
2. A/B Preference Testing Visualization (zha.tex line 419)

**TODO:** Implement system-level visualizations

## 🎯 Execution Order

### Phase 1: No Training Required ✅
```bash
# These work immediately with no trained models
python scripts/diagrams/generate_architecture_diagrams.py
python scripts/diagrams/generate_vae_diagrams.py
```
**Result:** 10 diagrams ready for thesis!

### Phase 2: After Training Markov Model
```bash
python backend/trainers/train_markov.py  # Train first
python scripts/diagrams/generate_markov_diagrams.py
python scripts/generate_markov_metrics.py
```

### Phase 3: After Training Transformer Model
```bash
python backend/trainers/train_transformer.py  # Train first
python scripts/diagrams/generate_transformer_diagrams.py
```

### Phase 4: After Training Multi-track Model
```bash
python backend/trainers/train_transformer.py --enable_multitrack  # Train
python scripts/diagrams/generate_multitrack_diagrams.py
```

### Phase 5: System Integration
```bash
python scripts/diagrams/generate_system_diagrams.py  # Needs all models
```

## 📤 Copying to Thesis

After generation, copy figures to thesis directory:

```bash
# Copy all PDFs to thesis figures directory
cp output/figures/thesis/*/*.pdf docs/thesis/figures/

# Or selectively by chapter
cp output/figures/thesis/architecture/*.pdf docs/thesis/figures/
cp output/figures/thesis/vae/*.pdf docs/thesis/figures/
# etc.
```

Then update LaTeX files to include actual figures:

```latex
% Before (placeholder):
% TODO: \includegraphics[width=0.8\textwidth]{images/vae_architecture.png}

% After:
\includegraphics[width=0.8\textwidth]{figures/vae_architecture.pdf}
```

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'matplotlib'"
```bash
pip install matplotlib seaborn numpy
```

### "FileNotFoundError: trained model not found"
- Check that model training completed successfully
- Look in `output/trained_models/` for model checkpoints
- Some diagrams can run with simulated data (check script for `simulate=True` option)

### Output directory doesn't exist
- Scripts automatically create output directories
- Ensure you have write permissions in project directory

### Figures look wrong in LaTeX
- Use PDF format for vector graphics (better quality)
- Adjust `\includegraphics[width=...]` to fit page
- Use `\centering` inside figure environment

## 📊 Metrics vs Diagrams

**Diagrams (visual):**
- Generated by `scripts/diagrams/*.py`
- Output: PDF/PNG files
- Include in thesis with `\includegraphics`

**Metrics (tables/numbers):**
- Generated by `scripts/generate_*_metrics.py`
- Output: LaTeX table code or CSV files
- Include in thesis with `\input{table.tex}` or copy-paste

## ⏱️ Estimated Generation Times

| Script | Time | Models Required |
|--------|------|-----------------|
| architecture_diagrams | ~10s | None ✅ |
| vae_diagrams | ~15s | None (uses simulated data) ✅ |
| markov_diagrams | ~20s | Markov model (optional) |
| transformer_diagrams | ~30s | Transformer model |
| multitrack_diagrams | ~25s | Multi-track Transformer |
| system_diagrams | ~40s | All models |

**Total time (all diagrams):** ~2-3 minutes

## 📚 Additional Resources

- **LaTeX Figure Tutorial:** docs/thesis/FIGURES_GUIDE.md (if exists)
- **Backend Training Guide:** backend/trainers/README.md
- **Model Architecture Docs:** docs/thesis/chapters/*.tex
- **Consistency Check Report:** docs/thesis/CONSISTENCY_CHECK_REPORT.md

## 🎨 Customization

All scripts use consistent styling:

```python
# Color scheme (in each script)
COLORS = {
    'markov': '#FF6B6B',      # Red
    'vae': '#4ECDC4',         # Teal
    'transformer': '#45B7D1', # Blue
    'combined': '#FFA07A',    # Salmon
    'multitrack': '#98D8C8',  # Mint
}

# DPI settings
PDF: 300 dpi (publication quality)
PNG: 150 dpi (screen/web quality)
```

To customize:
1. Edit color scheme in script
2. Adjust figure sizes: `figsize=(width, height)`
3. Change fonts: `plt.rcParams['font.size'] = 12`

## ✅ Checklist

- [ ] Install all Python dependencies
- [ ] Run architecture diagrams (no training needed)
- [ ] Run VAE diagrams (no training needed)
- [ ] Train Markov model
- [ ] Run Markov diagrams & metrics
- [ ] Train Transformer model
- [ ] Run Transformer diagrams
- [ ] Train multi-track model
- [ ] Run multi-track diagrams
- [ ] Run system integration diagrams
- [ ] Copy all PDFs to `docs/thesis/figures/`
- [ ] Update LaTeX `\includegraphics` commands
- [ ] Compile thesis to verify all figures appear correctly

---

**Created:** 2025-10-22  
**Last Updated:** 2025-10-22  
**Maintainer:** Zha Project Team
