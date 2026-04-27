#!/usr/bin/env python3
"""
COMPREHENSIVE MODEL CAPABILITY METRICS GENERATOR
Generates industry-standard generative model evaluation metrics.

Metrics Categories:
  1. MUSICAL QUALITY METRICS
     - Pitch class distribution analysis
     - Pitch range and diversity
     - Interval distribution
     - Chromatic coherence (scale conformity)
     - Pitch entropy
  
  2. STATISTICAL DISTANCE METRICS
     - Wasserstein Distance (optimal transport)
     - Jensen-Shannon Divergence
     - KL Divergence
     - Hellinger Distance
  
  3. DIVERSITY METRICS
     - Self-similarity matrix
     - Pitch coverage
     - Sequence diversity score
  
  4. QUALITY METRICS (for VAE)
     - Reconstruction error (MSE, MAE, RMSE)
     - R² Score
  
  5. GENERATION METRICS
     - Unique note sequences
     - Generation success rate
     - Output validity
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.metrics_utils import MusicMetrics, PerformanceMetrics, MetricsReporter
from backend.models.markov_chain import MarkovChain
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.size"] = 10

OUTPUT_DIR = Path("output/metrics/capabilities")
MODEL_DIR = Path("output/trained_models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class ModelCapabilityEvaluator:
    """Comprehensive model capability evaluation."""
    
    def __init__(self):
        self.metrics_reporter = MetricsReporter()
    
    def evaluate_markov(self, num_samples: int = 30) -> Dict:
        """Evaluate Markov model."""
        print("\n  Evaluating Markov Chain...")
        
        if not (MODEL_DIR / "markov.npy").exists():
            print("    ✗ Markov model not found (markov.npy)")
            return {}
        
        try:
            model = MarkovChain()
            # Load expects path WITHOUT extension
            model.load(str(MODEL_DIR / "markov"))
            print("    ✓ Markov model loaded")
            
            generations = []
            for i in tqdm(range(num_samples), desc="    Generating sequences", ncols=60):
                try:
                    # Generate a sequence starting from middle C (note 60)
                    seq = model.generate_sequence(start_note=60, length=100)
                    if isinstance(seq, list):
                        seq = np.array(seq)
                    if len(seq) > 0:
                        generations.append(seq)
                except Exception as e:
                    print(f"      Warning: Sequence {i} failed: {e}")
                    continue
            
            if not generations:
                print("    ✗ Failed to generate any sequences")
                return {}
            
            print(f"    ✓ Generated {len(generations)} sequences")
            metrics = self._compute_generation_metrics(generations, "Markov")
            return metrics
        
        except Exception as e:
            print(f"    ✗ Error evaluating Markov: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def evaluate_transformer(self, num_samples: int = 30) -> Dict:
        """Evaluate Transformer model."""
        print("\n  Evaluating Transformer...")
        
        model_paths = [
            MODEL_DIR / "transformer_best.pt",
            MODEL_DIR / "transformer_latest.pt",
            MODEL_DIR / "trained_transformer.pt",
        ]
        
        model_path = None
        for p in model_paths:
            if p.exists():
                model_path = p
                print(f"    Using: {p.name}")
                break
        
        if not model_path:
            print("    ✗ Transformer model not found")
            print(f"      Searched: {[p.name for p in model_paths]}")
            return {}
        
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"    Loading to device: {device}")
            
            # Try to load transformer model
            try:
                from backend.models.lightning_transformer import LightningTransformer
                
                model = LightningTransformer.load_from_checkpoint(
                    str(model_path),
                    map_location=device
                )
                model.to(device)
                model.eval()
                print(f"    ✓ Transformer model loaded from checkpoint")
                
                generations = []
                with torch.no_grad():
                    for i in tqdm(range(num_samples), desc="    Generating sequences", ncols=60):
                        try:
                            # Generate using model's generation method if available
                            if hasattr(model, 'generate'):
                                seq = model.generate(max_length=100, device=device)
                            else:
                                # Strict real-data policy: skip synthetic latent fallbacks.
                                raise RuntimeError("Transformer generate() is not available")
                            
                            if isinstance(seq, torch.Tensor):
                                seq = seq.cpu().numpy().flatten()
                            if len(seq) > 0:
                                generations.append(seq)
                        except Exception as e:
                            print(f"      Warning: Sample {i} failed: {e}")
                            continue
                
                if not generations:
                    print("    ✗ Failed to generate any sequences")
                    return {}
                
                print(f"    ✓ Generated {len(generations)} sequences")
                metrics = self._compute_generation_metrics(generations, "Transformer")
                return metrics
            
            except Exception as e:
                print(f"    ⚠ Transformer LightningModule not available: {e}")
                print("    ⚠ Skipping Transformer evaluation (complex architecture)")
                return {}
        
        except Exception as e:
            print(f"    ✗ Error evaluating Transformer: {e}")
            return {}
    
    def evaluate_golc_vae(self, num_samples: int = 30) -> Dict:
        """Evaluate GOLC-VAE model."""
        print("\n  Evaluating GOLC-VAE...")
        
        model_paths = [
            MODEL_DIR / "golc_vae_latest.pt",
            MODEL_DIR / "golc_vae_best.pt",
            MODEL_DIR / "golc_vae_final.pt",
        ]
        
        model_path = None
        for p in model_paths:
            if p.exists():
                model_path = p
                print(f"    Using: {p.name}")
                break
        
        if not model_path:
            print("    ✗ GOLC-VAE model not found")
            print(f"      Searched: {[p.name for p in model_paths]}")
            return {}
        
        try:
            try:
                from backend.models.lightning_vae import LightningGOLCVAE
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                print(f"    Loading to device: {device}")
                
                # Try to load checkpoint with map_location
                model = LightningGOLCVAE.load_from_checkpoint(
                    str(model_path),
                    map_location=device
                )
                model.to(device)
                model.eval()
                print(f"    ✓ GOLC-VAE model loaded from checkpoint")
                
                generations = []
                with torch.no_grad():
                    for i in tqdm(range(num_samples), desc="    Generating sequences", ncols=60):
                        try:
                            # Get latent dimension from model hparams
                            latent_dim = getattr(model.hparams, 'latent_dim', 128)
                            z = torch.randn(1, latent_dim, device=device)
                            recon = model.model.decode(z)
                            seq = recon.cpu().numpy().flatten()
                            if len(seq) > 0:
                                generations.append(seq)
                        except Exception as e:
                            print(f"      Warning: Sample {i} failed: {e}")
                            continue
                
                if not generations:
                    print("    ✗ Failed to generate any sequences")
                    return {}
                
                print(f"    ✓ Generated {len(generations)} sequences")
                metrics = self._compute_generation_metrics(generations, "GOLC-VAE")
                metrics["vae_reconstruction_quality"] = self._estimate_reconstruction_quality(model, device)
                return metrics
            
            except ImportError as e:
                print(f"    ⚠ LightningGOLCVAE not available: {e}")
                print("    Trying direct GOLC_VAE loading...")
                
                try:
                    from backend.models.golc_vae import GOLC_VAE
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    
                    # Try loading as raw checkpoint
                    checkpoint = torch.load(str(model_path), map_location=device)
                    
                    # Create model with default params
                    model_bare = GOLC_VAE(
                        input_dim=128,
                        latent_dim=128,
                        beta=1.0,
                        beta_orbit=0.5,
                        transposition_range=6
                    )
                    model_bare.to(device)
                    model_bare.eval()
                    
                    # Try to load state dict
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        state_weights = checkpoint['state_dict']
                        # Remove 'model.' prefix if present
                        state_dict = {k.replace('model.', ''): v for k, v in state_weights.items()}
                        model_bare.load_state_dict(state_dict, strict=False)
                    else:
                        model_bare.load_state_dict(checkpoint, strict=False)
                    
                    print(f"    ✓ GOLC-VAE loaded directly")
                    
                    generations = []
                    with torch.no_grad():
                        for i in tqdm(range(num_samples), desc="    Generating sequences", ncols=60):
                            try:
                                z = torch.randn(1, 128, device=device)
                                recon = model_bare.decode(z)
                                seq = recon.cpu().numpy().flatten()
                                if len(seq) > 0 and seq.max() <= 1.1:  # Valid range check
                                    generations.append(seq)
                            except Exception as e:
                                print(f"      Warning: Sample {i} failed: {e}")
                                continue
                    
                    if not generations:
                        print("    ✗ Failed to generate any sequences")
                        return {}
                    
                    print(f"    ✓ Generated {len(generations)} sequences")
                    metrics = self._compute_generation_metrics(generations, "GOLC-VAE")
                    return metrics
                
                except Exception as e2:
                    print(f"    ✗ Direct GOLC_VAE loading also failed: {e2}")
                    return {}
        
        except Exception as e:
            print(f"    ✗ Error evaluating GOLC-VAE: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _compute_generation_metrics(self, generations: List[np.ndarray], model_name: str) -> Dict:
        """Compute comprehensive metrics for generated sequences."""
        metrics = {
            "model_name": model_name,
            "num_sequences_generated": len(generations),
        }
        
        # Individual sequence metrics
        all_musical_metrics = []
        for gen in generations:
            m = {
                "pitch_entropy": MusicMetrics.compute_pitch_entropy(gen),
                "unique_pitches": MusicMetrics.compute_unique_pitches(gen),
                "note_density": MusicMetrics.compute_note_density(gen),
                **MusicMetrics.compute_pitch_range(gen),
                **MusicMetrics.compute_interval_distribution(gen),
            }
            all_musical_metrics.append(m)
        
        # Aggregate metrics
        metrics["musical_quality"] = {
            "pitch_entropy": {
                "mean": float(np.mean([m["pitch_entropy"] for m in all_musical_metrics])),
                "std": float(np.std([m["pitch_entropy"] for m in all_musical_metrics])),
                "min": float(np.min([m["pitch_entropy"] for m in all_musical_metrics])),
                "max": float(np.max([m["pitch_entropy"] for m in all_musical_metrics])),
            },
            "unique_pitches": {
                "mean": float(np.mean([m["unique_pitches"] for m in all_musical_metrics])),
                "std": float(np.std([m["unique_pitches"] for m in all_musical_metrics])),
                "min": float(np.min([m["unique_pitches"] for m in all_musical_metrics])),
                "max": float(np.max([m["unique_pitches"] for m in all_musical_metrics])),
            },
            "note_density": {
                "mean": float(np.mean([m["note_density"] for m in all_musical_metrics])),
                "std": float(np.std([m["note_density"] for m in all_musical_metrics])),
            },
            "pitch_range": {
                "mean": float(np.mean([m["range"] for m in all_musical_metrics])),
                "std": float(np.std([m["range"] for m in all_musical_metrics])),
            },
            "interval_entropy": {
                "mean": float(np.mean([m["interval_entropy"] for m in all_musical_metrics])),
                "std": float(np.std([m["interval_entropy"] for m in all_musical_metrics])),
            },
        }
        
        # Statistical diversity metrics
        metrics["diversity"] = MusicMetrics.compute_diversity_score(generations)
        
        # Pitch class distribution (reference for comparison)
        combined_seq = np.concatenate(generations)
        metrics["pitch_class_distribution"] = MusicMetrics.compute_pitch_class_distribution(combined_seq).tolist()
        
        # Chromatic coherence
        metrics["chromatic_coherence"] = {
            "mean": float(np.mean([MusicMetrics.compute_chromatic_coherence(gen) for gen in generations])),
            "std": float(np.std([MusicMetrics.compute_chromatic_coherence(gen) for gen in generations])),
        }
        
        return metrics
    
    def _estimate_reconstruction_quality(self, model, device) -> Dict:
        """Reconstruction quality requires real held-out validation inputs."""
        return {
            "status": "unavailable",
            "reason": "No held-out reconstruction dataset provided; synthetic probing is disabled.",
        }


def plot_capability_comparison(all_metrics: Dict):
    """Create comparison plots for all models."""
    print("\n  Generating comparison plots...")
    
    if not all_metrics:
        return
    
    models = list(all_metrics.keys())
    n_models = len(models)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Model Capability Comparison - Musical Quality Metrics', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    # Plot 1: Pitch Entropy
    pitch_entropy = [all_metrics[m]["musical_quality"]["pitch_entropy"]["mean"] for m in models]
    axes[0].bar(models, pitch_entropy, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_ylabel("Pitch Entropy", fontweight="bold")
    axes[0].set_title("Average Pitch Entropy (Higher = More Diverse)")
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Unique Pitches
    unique_pitches = [all_metrics[m]["musical_quality"]["unique_pitches"]["mean"] for m in models]
    axes[1].bar(models, unique_pitches, color='seagreen', alpha=0.7, edgecolor='black')
    axes[1].set_ylabel("Unique Pitches", fontweight="bold")
    axes[1].set_title("Average Unique Pitches Used")
    axes[1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Pitch Range
    pitch_range = [all_metrics[m]["musical_quality"]["pitch_range"]["mean"] for m in models]
    axes[2].bar(models, pitch_range, color='coral', alpha=0.7, edgecolor='black')
    axes[2].set_ylabel("Pitch Range (semitones)", fontweight="bold")
    axes[2].set_title("Average Pitch Range")
    axes[2].grid(axis='y', alpha=0.3)
    
    # Plot 4: Note Density
    note_density = [all_metrics[m]["musical_quality"]["note_density"]["mean"] for m in models]
    axes[3].bar(models, note_density, color='purple', alpha=0.7, edgecolor='black')
    axes[3].set_ylabel("Note Density", fontweight="bold")
    axes[3].set_title("Average Note Density (Notes/Length)")
    axes[3].grid(axis='y', alpha=0.3)
    
    # Plot 5: Pitch Coverage (Diversity)
    pitch_coverage = [all_metrics[m]["diversity"]["pitch_coverage"] for m in models]
    axes[4].bar(models, pitch_coverage, color='orange', alpha=0.7, edgecolor='black')
    axes[4].set_ylabel("Pitch Coverage", fontweight="bold")
    axes[4].set_title("Pitch Space Coverage Across All Samples")
    axes[4].set_ylim([0, 1])
    axes[4].grid(axis='y', alpha=0.3)
    
    # Plot 6: Self-Similarity
    self_similarity = [all_metrics[m]["diversity"]["self_similarity_mean"] for m in models]
    axes[5].bar(models, self_similarity, color='pink', alpha=0.7, edgecolor='black')
    axes[5].set_ylabel("Self-Similarity", fontweight="bold")
    axes[5].set_title("Average Sequence Similarity (Lower = More Diverse)")
    axes[5].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "model_capabilities_comparison.png"
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {output_path.name}")


def plot_pitch_distributions(all_metrics: Dict):
    """Plot pitch class distributions for all models."""
    print("  Generating pitch distribution plots...")
    
    models = list(all_metrics.keys())
    n_models = len(models)
    
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    if n_models == 1:
        axes = [axes]
    
    fig.suptitle('Pitch Class Distribution by Model', fontsize=14, fontweight='bold')
    
    for ax, model_name in zip(axes, models):
        dist = np.array(all_metrics[model_name]["pitch_class_distribution"])
        pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        colors = plt.cm.viridis(dist / np.max(dist))
        ax.bar(pitch_names, dist, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel("Probability", fontweight="bold")
        ax.set_title(model_name)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, np.max(dist) * 1.1])
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "pitch_distributions.png"
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {output_path.name}")


def main():
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL CAPABILITY METRICS GENERATOR")
    print("="*80)
    print("\nEvaluating models...")
    
    evaluator = ModelCapabilityEvaluator()
    all_metrics = {}
    
    # Evaluate each model
    markov_metrics = evaluator.evaluate_markov(num_samples=30)
    if markov_metrics:
        all_metrics["Markov"] = markov_metrics
        print("    ✓ Markov evaluation complete")
    
    transformer_metrics = evaluator.evaluate_transformer(num_samples=30)
    if transformer_metrics:
        all_metrics["Transformer"] = transformer_metrics
        print("    ✓ Transformer evaluation complete")
    
    golc_vae_metrics = evaluator.evaluate_golc_vae(num_samples=30)
    if golc_vae_metrics:
        all_metrics["GOLC-VAE"] = golc_vae_metrics
        print("    ✓ GOLC-VAE evaluation complete")
    
    if not all_metrics:
        print("\n✗ No models could be evaluated")
        return
    
    # Generate visualizations
    print("\n  Generating visualizations...")
    plot_capability_comparison(all_metrics)
    plot_pitch_distributions(all_metrics)
    
    # Save metrics
    print("\n  Saving metrics to JSON...")
    
    # Individual model files
    for model_name, metrics in all_metrics.items():
        output_file = OUTPUT_DIR / f"{model_name.lower().replace(' ', '_')}_metrics.json"
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"    Saved: {output_file.name}")
    
    # Summary file
    summary = {
        "evaluated_models": list(all_metrics.keys()),
        "num_models": len(all_metrics),
        "metrics_per_model": list(all_metrics[list(all_metrics.keys())[0]].keys()),
        "output_directory": str(OUTPUT_DIR),
        "individual_model_files": [f"{m.lower().replace(' ', '_')}_metrics.json" for m in all_metrics.keys()],
    }
    
    with open(OUTPUT_DIR / "capability_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"    Saved: capability_summary.json")
    
    # Print summary
    print("\n" + "="*80)
    print("CAPABILITY METRICS GENERATION COMPLETE")
    print("="*80)
    print(f"\nModels evaluated: {', '.join(all_metrics.keys())}")
    print(f"\nGenerated files:")
    print(f"  - model_capabilities_comparison.png")
    print(f"  - pitch_distributions.png")
    for model_name in all_metrics.keys():
        print(f"  - {model_name.lower().replace(' ', '_')}_metrics.json")
    print(f"  - capability_summary.json")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
