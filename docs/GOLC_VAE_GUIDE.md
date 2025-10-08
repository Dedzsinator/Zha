# GOLC-VAE Implementation Guide

## Overview

This document describes the implementation and usage of the **Group-Orbital Latent Consistency Variational Autoencoder (GOLC-VAE)**, an enhanced VAE that ensures musical transformations (like transpositions) map to consistent latent representations.

## Theoretical Foundation

The GOLC-VAE is based on the mathematical framework described in the thesis (see `docs/thesis/chapters/vae.tex`). Key concepts:

### 1. Orbital Consistency
For a group of musical transformations G (e.g., transpositions), we ensure:
```
Enc_φ(g·x) ≈ Enc_φ(x)  for all g ∈ G
```

### 2. Canonical Representation
The canonical latent representation is computed by averaging across the orbit:
```
z_c(x) = (1/|G|) * Σ_{g∈G} Enc_φ(g·x)
```

### 3. Total Loss Function
```
L_total = L_recon + β_KL * L_KL + β_orbit * L_orbit

where:
- L_recon = BCE(x, recon_x)  # Reconstruction loss
- L_KL = KL(q(z|x) || p(z))  # KL divergence
- L_orbit = ||Enc_φ(x) - z_c(x)||²  # Orbital consistency loss
```

## Architecture Comparison

### Baseline VAE
- Standard encoder-decoder architecture
- Reparameterization trick
- β-VAE regularization
- No invariance to musical transformations

### GOLC-VAE (Enhanced)
- ✓ Same encoder-decoder architecture (fair comparison)
- ✓ Group-orbital consistency loss
- ✓ Canonical representation averaging
- ✓ Transposition-invariant latent space
- ✓ Enhanced posterior stability

## File Structure

```
backend/
├── models/
│   ├── vae.py              # Baseline VAE implementation
│   └── golc_vae.py         # GOLC-enhanced VAE implementation
├── trainers/
│   ├── train_vae.py        # Baseline VAE training script
│   └── train_golc_vae.py   # GOLC-VAE training script
└── util/
    └── vae_metrics.py      # Comprehensive metrics evaluation

scripts/
├── evaluate_baseline_vae.py  # Evaluate baseline VAE
└── compare_vae_models.py     # Compare baseline vs GOLC
```

## Usage

### 1. Training Baseline VAE (on feature branch)

Switch to the feature branch:
```bash
git checkout feature/vae-baseline-metrics
```

Train the baseline model:
```bash
python backend/trainers/train_vae.py \
  --data-dir dataset/processed \
  --output-dir output/trained_models \
  --latent-dim 128 \
  --batch-size 128 \
  --epochs 100 \
  --lr 1e-3
```

### 2. Training GOLC-VAE (on main branch)

Switch to main branch:
```bash
git checkout main
```

Train the GOLC-enhanced model:
```bash
python backend/trainers/train_golc_vae.py \
  --data-dir dataset/processed \
  --output-dir output/trained_models \
  --latent-dim 128 \
  --beta 1.0 \
  --beta-orbit 0.5 \
  --transposition-range 6 \
  --batch-size 128 \
  --epochs 100 \
  --lr 1e-3
```

### 3. Evaluating Models

#### Evaluate Baseline VAE:
```bash
python scripts/evaluate_baseline_vae.py \
  --checkpoint output/trained_models/vae_best.pt \
  --data-path dataset/processed/sequences.pkl \
  --output-dir output/baseline_evaluation
```

#### Compare Both Models:
```bash
python scripts/compare_vae_models.py \
  --baseline-checkpoint output/trained_models/vae_best.pt \
  --golc-checkpoint output/trained_models/golc_vae_best.pt \
  --test-data dataset/processed/sequences.pkl \
  --output-dir output/vae_comparison
```

## Hyperparameters

### Key GOLC-VAE Parameters

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|-------------------|
| `beta` | KL divergence weight (β-VAE) | 1.0 | 0.5 - 2.0 |
| `beta_orbit` | Orbital consistency weight | 0.5 | 0.1 - 1.0 |
| `transposition_range` | Max transposition (±semitones) | 6 | 3 - 12 |
| `latent_dim` | Latent space dimension | 128 | 64 - 256 |
| `batch_size` | Training batch size | 128 | 64 - 256 |

### Tuning Recommendations

**For stronger invariance to transpositions:**
- Increase `beta_orbit` (e.g., 0.8 - 1.0)
- Increase `transposition_range` (e.g., 9 - 12)

**For better reconstruction:**
- Decrease `beta` (e.g., 0.5 - 0.8)
- Decrease `beta_orbit` (e.g., 0.2 - 0.4)

**For more diverse generation:**
- Increase `beta` (e.g., 1.5 - 2.0)
- Moderate `beta_orbit` (e.g., 0.3 - 0.5)

## Evaluation Metrics

The implementation includes five comprehensive metrics:

### 1. Orbit Latent Distance (OLD)
**Purpose:** Measures consistency of latent representations under musical transformations

**Formula:**
```
OLD = (1/N) * Σ ||Enc(x) - Enc(T_k(x))||²
```

**Interpretation:** Lower is better. Indicates invariance to transpositions.

### 2. KL Divergence Variance
**Purpose:** Detects posterior collapse

**Formula:**
```
KL_var = Var(KL(q(z|x_i) || p(z)))
```

**Interpretation:** Lower variance indicates less posterior collapse.

### 3. Pitch-Class Entropy
**Purpose:** Quantifies tonal diversity in generated samples

**Formula:**
```
H = -Σ p(c) * log₂(p(c))
```

**Interpretation:** Higher entropy = more diverse pitch usage.

### 4. Tonality Preservation Score
**Purpose:** Evaluates key consistency between originals and reconstructions

**Method:** Uses music21 key detection

**Interpretation:** Higher score = better tonal preservation.

### 5. Reconstruction MSE
**Purpose:** Standard reconstruction quality metric

**Formula:**
```
MSE = (1/N) * Σ ||x - recon(x)||²
```

**Interpretation:** Lower is better.

## Expected Results

Based on theoretical predictions, GOLC-VAE should demonstrate:

1. **✓ Improved OLD** (~20-40% reduction)
   - Better invariance to transpositions
   - More consistent latent representations

2. **✓ Reduced KL Variance** (~10-30% reduction)
   - Less posterior collapse
   - More stable posterior distributions

3. **✓ Maintained Pitch Entropy** (±5%)
   - Preserves generative diversity
   - No loss of tonal variety

4. **✓ Improved Tonality Preservation** (~5-15% increase)
   - Better preservation of musical keys
   - More musically coherent reconstructions

5. **~ Similar Reconstruction MSE** (±10%)
   - Comparable reconstruction quality
   - Trade-off between reconstruction and invariance

## Output Files

After running the comparison, you'll get:

```
output/vae_comparison/
├── comparison_results.json     # Numerical results
├── vae_comparison.png          # Side-by-side visualization
└── comparison_report.md        # Comprehensive analysis
```

### Sample Output Structure

**comparison_results.json:**
```json
{
  "baseline": {
    "old": 0.0234,
    "kl_variance": 0.0156,
    "pitch_entropy": 3.45,
    "tonality": 0.67,
    "recon_mse": 0.0023
  },
  "golc": {
    "old": 0.0145,
    "kl_variance": 0.0112,
    "pitch_entropy": 3.48,
    "tonality": 0.74,
    "recon_mse": 0.0025
  },
  "comparison": {
    "improvements": {
      "old": 38.03,
      "kl_variance": 28.21,
      "pitch_entropy": 0.87,
      "tonality": 10.45,
      "recon_mse": -8.70
    }
  }
}
```

## Integration with Thesis

Results from these experiments should be added to:

1. **Experimental Results Section** (`docs/thesis/chapters/vae.tex`)
   - Add subsection: "5.5 Empirikai Eredmények" (Empirical Results)
   - Include comparison table with all metrics
   - Add bar chart visualization

2. **Discussion** 
   - Interpret improvements in OLD and KL Variance
   - Discuss trade-offs (if any)
   - Relate back to theoretical predictions

3. **Figures**
   - Add `vae_comparison.png` to thesis figures
   - Reference as "X. ábra: GOLC-VAE vs. Baseline VAE összehasonlítása"

## Troubleshooting

### Model not improving orbital consistency:
- **Solution:** Increase `beta_orbit` to 0.8-1.0
- Check that transpositions are being applied correctly
- Verify gradient flow through orbital loss

### Poor reconstruction quality:
- **Solution:** Decrease `beta_orbit` to 0.2-0.3
- Increase model capacity (more layers/wider)
- Reduce KL weight (`beta`)

### Training instability:
- **Solution:** Use gradient clipping (already implemented)
- Reduce learning rate to 5e-4
- Use learning rate warmup

### Posterior collapse:
- **Solution:** Use β-annealing (increase `beta` gradually)
- Monitor KL divergence per dimension
- Increase latent dimension

## Git Workflow

The implementation follows a two-branch strategy:

**Main Branch (main):**
- GOLC-VAE implementation
- Enhanced features
- Orbital consistency

**Feature Branch (feature/vae-baseline-metrics):**
- Baseline VAE evaluation
- Metrics implementation
- Standard architecture

To switch branches:
```bash
# Work on GOLC enhancement
git checkout main

# Work on baseline evaluation
git checkout feature/vae-baseline-metrics
```

## Next Steps

1. **Train Both Models:**
   - Baseline on feature branch
   - GOLC on main branch
   - Use same hyperparameters (except GOLC-specific)

2. **Run Comprehensive Comparison:**
   - Execute comparison script
   - Analyze results
   - Generate visualizations

3. **Update Thesis:**
   - Add experimental results section
   - Include comparison figures
   - Discuss findings in Hungarian

4. **Hyperparameter Search (Optional):**
   - Grid search over `beta_orbit` and `transposition_range`
   - Find optimal balance
   - Report best configuration

## References

Mathematical foundations and implementation details are based on:

1. **Kingma & Welling (2014)** - "Auto-Encoding Variational Bayes" - VAE fundamentals
2. **Temperley (2007)** - Music theory and tonality analysis
3. **Armstrong et al. (2020)** - "Group-Equivariant Neural Networks" - Group theory in ML
4. **Kondor & Trivedi (2018)** - "On the Generalization of Equivariance" - Invariance principles

See `docs/thesis/dolgozat.bib` for complete bibliography.

## Questions?

For implementation questions or issues:
1. Check `backend/models/golc_vae.py` for detailed code comments
2. Review `docs/thesis/chapters/vae.tex` for mathematical foundations
3. Examine training logs in `output/trained_models/golc_vae_history.json`
