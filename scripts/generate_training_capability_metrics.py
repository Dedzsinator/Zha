#!/usr/bin/env python3
"""
COMPREHENSIVE TRAINING METRICS GENERATOR
Generates industry-standard training evaluation metrics and visualizations.

Metrics Generated:
  1. Training curves (loss, learning rate schedules)
  2. Convergence analysis (overfitting detection)
  3. Validation metrics over time
  4. Learning dynamics analysis
  5. Training summary statistics
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import torch

plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.size"] = 10
plt.rcParams['axes.facecolor'] = '#f8f9fa'

OUTPUT_DIR = Path("output/metrics/training_and_capability")
MODEL_DIR = Path("output/trained_models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class TrainingMetricsAnalyzer:
    """Analyze and visualize training metrics."""
    
    @staticmethod
    def load_history(history_path: Path) -> Dict:
        """Load training history from JSON file."""
        try:
            with open(history_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"  Error loading {history_path}: {e}")
            return {}
    
    @staticmethod
    def extract_history_from_checkpoint(checkpoint_path: Path) -> Dict:
        """Extract training history from PyTorch checkpoint file."""
        try:
            checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
            
            history = {}
            
            # Try to extract from PyTorch Lightning logs
            if isinstance(checkpoint, dict):
                # Check for pytorch_lightning specific keys
                if 'epoch' in checkpoint:
                    history['n_epochs_trained'] = int(checkpoint.get('epoch', 0))
                
                # Look for trainer state
                if 'trainer' in checkpoint:
                    trainer_state = checkpoint['trainer']
                    if 'global_step' in trainer_state:
                        history['global_step'] = int(trainer_state['global_step'])
                
                # Look for any loss information in state_dict
                if 'state_dict' in checkpoint:
                    # This won't give us full history, but at least we know the model exists
                    pass
            
            return history
        except Exception as e:
            return {}
    
    @staticmethod
    def generate_fallback_history(model_name: str, model_path: Path) -> Dict:
        """Return real metadata fallback when history logs are unavailable."""
        # Check if checkpoint file exists
        if model_path.exists():
            print(f"      Using checkpoint metadata...")
            history = TrainingMetricsAnalyzer.extract_history_from_checkpoint(model_path)
            if history:
                return history

        print(f"      No real history available for {model_name}; metrics will be marked unavailable")
        return {
            "_generated": False,
            "_note": (
                f"No tracked history found for {model_name}. "
                "Run model training with history logging enabled to generate real curves."
            )
        }
    
    @staticmethod
    def analyze_convergence(history: Dict) -> Dict:
        """
        Analyze convergence properties of training.
        Detects overfitting, underfitting, learning rate issues.
        """
        if 'loss' not in history or len(history['loss']) == 0:
            return {}
        
        losses = np.array(history['loss'])
        n_epochs = len(losses)
        
        analysis = {
            "n_epochs_trained": n_epochs,
            "final_loss": float(losses[-1]),
            "initial_loss": float(losses[0]),
            "loss_improvement": float(losses[0] - losses[-1]),
            "loss_improvement_percent": float((losses[0] - losses[-1]) / losses[0] * 100) if losses[0] != 0 else 0.0,
        }
        
        # Convergence speed (epochs to reach 90% of final loss improvement)
        improvement_90 = losses[0] - (losses[0] - losses[-1]) * 0.9
        epochs_to_90 = len([l for l in losses if l > improvement_90])
        analysis["epochs_to_90_percent_convergence"] = int(epochs_to_90)
        
        # Overfitting detection
        if 'val_loss' in history and len(history['val_loss']) > 0:
            val_losses = np.array(history['val_loss'])
            
            # Check if validation loss increases while training loss decreases (classic overfitting)
            last_train_loss = losses[-1]
            last_val_loss = val_losses[-1]
            analysis["val_train_loss_ratio"] = float(last_val_loss / last_train_loss) if last_train_loss > 0 else 1.0
            analysis["has_overfitting_indicators"] = bool(last_val_loss > last_train_loss * 1.2)
            
            # Best validation loss
            best_val_loss = np.min(val_losses)
            best_val_epoch = int(np.argmin(val_losses))
            analysis["best_val_loss"] = float(best_val_loss)
            analysis["best_val_epoch"] = best_val_epoch
            
            # Validation loss stability
            analysis["val_loss_stability"] = float(np.std(val_losses[-10:]) / np.mean(val_losses[-10:]) * 100) if len(val_losses) >= 10 else 0.0
        
        # Loss smoothness (check for oscillations)
        if len(losses) > 10:
            recent_losses = losses[-10:]
            analysis["loss_smoothness"] = float(np.std(recent_losses) / np.mean(recent_losses) * 100)
        
        # Learning rate effectiveness (if tracked)
        if 'learning_rate' in history and len(history['learning_rate']) > 0:
            lrs = np.array(history['learning_rate'])
            analysis["initial_learning_rate"] = float(lrs[0])
            analysis["final_learning_rate"] = float(lrs[-1])
            analysis["learning_rate_schedule_type"] = "dynamic" if len(set(np.round(lrs, 8))) > 1 else "static"
        
        return analysis
    
    @staticmethod
    def compute_training_statistics(history: Dict) -> Dict:
        """Compute comprehensive training statistics."""
        stats = {}
        
        for key in ['loss', 'val_loss', 'learning_rate']:
            if key in history and len(history[key]) > 0:
                values = np.array(history[key])
                stats[key] = {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "q1": float(np.percentile(values, 25)),
                    "q3": float(np.percentile(values, 75)),
                }
        
        return stats


def plot_training_curves():
    """Generate comprehensive training curve plots."""
    print("\n=== Generating Comprehensive Training Analysis ===\n")
    
    # Try to load history files - check both with and without dots
    history_files = [
        ("GOLC-VAE", MODEL_DIR / "golc_vae_history.json"),
        ("Markov", MODEL_DIR / "markov_history.json"),
        ("Transformer", MODEL_DIR / "transformer_history.json"),
    ]
    
    models_found = []
    analyzer = TrainingMetricsAnalyzer()
    convergence_analysis = {}
    training_stats = {}
    
    for model_name, history_path in history_files:
        if not history_path.exists():
            # Try alternative names for older files
            if model_name == "GOLC-VAE":
                alt_path = MODEL_DIR / ".golc_vae_history.json"
            elif model_name == "Markov":
                alt_path = MODEL_DIR / ".markov_history.json"
            elif model_name == "Transformer":
                alt_path = MODEL_DIR / ".transformer_history.json"
            else:
                alt_path = None
            
            if alt_path and alt_path.exists():
                history_path = alt_path
            else:
                print(f"  {model_name}: ✗ No history file found")
                continue
        
        history = analyzer.load_history(history_path)
        if not history or (not history.get('loss') and not history.get('train_loss')):
            print(f"  {model_name}: ✗ Empty history")
            continue
        
        # Normalize keys (some use 'loss', others use 'train_loss')
        if 'train_loss' in history and 'loss' not in history:
            history['loss'] = history['train_loss']
        if 'val_loss' not in history and 'validation_loss' in history:
            history['val_loss'] = history['validation_loss']
        
        n_epochs = len(history.get('loss', []))
        models_found.append((model_name, history))
        print(f"  {model_name}: ✓ Found history ({n_epochs} epochs)")
        
        # Analyze convergence
        convergence = analyzer.analyze_convergence(history)
        convergence_analysis[model_name] = convergence
        
        # Compute statistics
        stats = analyzer.compute_training_statistics(history)
        training_stats[model_name] = stats
        
        # Plot individual model
        _plot_single_model_comprehensive(model_name, history, convergence)
    
    if not models_found:
        print("\n  ⚠ WARNING: No training history files found!")
        print("  Expected files in output/trained_models/:")
        print("    - golc_vae_history.json")
        print("    - markov_history.json")
        print("    - transformer_history.json")
        print("  (Optional: Prefix with . for hidden files)")
        return convergence_analysis, training_stats
    
    # Plot comparison charts
    print(f"\n  Generating visualizations for {len(models_found)} models...")
    _plot_loss_comparison(models_found)
    _plot_convergence_analysis(convergence_analysis)
    _plot_learning_rate_schedules(models_found)
    
    return convergence_analysis, training_stats


def _plot_single_model_comprehensive(model_name: str, history: Dict, convergence: Dict):
    """Create a comprehensive 2x2 subplot for single model."""
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Plot 1: Loss curves
    ax1 = fig.add_subplot(gs[0, 0])
    if 'loss' in history:
        losses = history['loss']
        ax1.plot(range(len(losses)), losses, linewidth=2.5, label="Train Loss", color="steelblue", marker='o', markersize=2, markevery=max(1, len(losses)//20))
        
        if 'val_loss' in history:
            val_losses = history['val_loss']
            ax1.plot(range(len(val_losses)), val_losses, linewidth=2.5, label="Val Loss", color="coral", linestyle="--", marker='s', markersize=2, markevery=max(1, len(val_losses)//20))
        
        ax1.set_xlabel("Epoch", fontweight="bold")
        ax1.set_ylabel("Loss", fontweight="bold")
        ax1.set_title(f"{model_name}: Loss Curves")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss improvement rate
    ax2 = fig.add_subplot(gs[0, 1])
    if 'loss' in history and len(history['loss']) > 1:
        losses = np.array(history['loss'])
        loss_diff = np.abs(np.diff(losses))
        ax2.semilogy(range(len(loss_diff)), loss_diff, linewidth=2, color="seagreen", marker='x')
        ax2.set_xlabel("Epoch", fontweight="bold")
        ax2.set_ylabel("Loss Change (log scale)", fontweight="bold")
        ax2.set_title(f"{model_name}: Loss Improvement Rate")
        ax2.grid(True, alpha=0.3, which='both')
    
    # Plot 3: Learning rate schedule
    ax3 = fig.add_subplot(gs[1, 0])
    if 'learning_rate' in history and len(history['learning_rate']) > 0:
        lrs = history['learning_rate']
        ax3.plot(range(len(lrs)), lrs, linewidth=2, color="purple", marker='^', markersize=3, markevery=max(1, len(lrs)//20))
        ax3.set_xlabel("Epoch", fontweight="bold")
        ax3.set_ylabel("Learning Rate", fontweight="bold")
        ax3.set_title(f"{model_name}: Learning Rate Schedule")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "No LR schedule tracked", ha='center', va='center', transform=ax3.transAxes)
        ax3.axis('off')
    
    # Plot 4: Convergence metrics
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Print convergence info
    init_loss = convergence.get('initial_loss')
    final_loss = convergence.get('final_loss')
    best_val_loss = convergence.get('best_val_loss')
    
    init_str = f"{init_loss:.4f}" if init_loss is not None else "N/A"
    final_str = f"{final_loss:.4f}" if final_loss is not None else "N/A"
    best_str = f"{best_val_loss:.4f}" if best_val_loss is not None else "N/A"
    
    info_text = f"""
MODEL TRAINING SUMMARY: {model_name}

Total Epochs: {convergence.get('n_epochs_trained', 'N/A')}
Initial Loss: {init_str}
Final Loss: {final_str}
Loss Improvement: {convergence.get('loss_improvement_percent', 0):.1f}%

Epochs to 90% Convergence: {convergence.get('epochs_to_90_percent_convergence', 'N/A')}
Overfitting Detected: {convergence.get('has_overfitting_indicators', False)}
Val/Train Loss Ratio: {convergence.get('val_train_loss_ratio', 1.0):.2f}

Best Val Loss: {best_str}
Best Val Epoch: {convergence.get('best_val_epoch', 'N/A')}
    """
    
    ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / f"training_{model_name.lower().replace('-', '_').replace(' ', '_')}_detailed.png"
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def _plot_loss_comparison(models: List[Tuple[str, Dict]]):
    """Plot loss comparison across all models."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    # Training loss comparison
    for (model_name, history), color in zip(models, colors):
        if 'loss' in history:
            losses = history['loss']
            axes[0].plot(range(len(losses)), losses, linewidth=2.5, label=model_name, 
                        color=color, marker='o', markersize=3, markevery=max(1, len(losses)//15), alpha=0.8)
    
    axes[0].set_xlabel("Epoch", fontweight="bold", fontsize=12)
    axes[0].set_ylabel("Training Loss", fontweight="bold", fontsize=12)
    axes[0].set_title("Training Loss Comparison - All Models", fontweight="bold", fontsize=13)
    axes[0].legend(fontsize=10, loc="best")
    axes[0].grid(True, alpha=0.3)
    
    # Validation loss comparison
    has_val = False
    for (model_name, history), color in zip(models, colors):
        if 'val_loss' in history:
            has_val = True
            val_losses = history['val_loss']
            axes[1].plot(range(len(val_losses)), val_losses, linewidth=2.5, label=model_name,
                        color=color, marker='s', markersize=3, markevery=max(1, len(val_losses)//15), alpha=0.8)
    
    axes[1].set_xlabel("Epoch", fontweight="bold", fontsize=12)
    axes[1].set_ylabel("Validation Loss", fontweight="bold", fontsize=12)
    axes[1].set_title("Validation Loss Comparison - All Models", fontweight="bold", fontsize=13)
    axes[1].legend(fontsize=10, loc="best")
    axes[1].grid(True, alpha=0.3)
    
    if not has_val:
        axes[1].text(0.5, 0.5, "No validation loss data available", 
                    ha='center', va='center', transform=axes[1].transAxes)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "training_loss_comparison.png"
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def _plot_convergence_analysis(convergence_analysis: Dict):
    """Visual convergence analysis dashboard."""
    n_models = len(convergence_analysis)
    if n_models == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    models = list(convergence_analysis.keys())
    
    # Plot 1: Loss Improvement %
    improvements = [convergence_analysis[m].get('loss_improvement_percent', 0) for m in models]
    colors_imp = ['green' if x > 50 else 'orange' if x > 20 else 'red' for x in improvements]
    axes[0].barh(models, improvements, color=colors_imp, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel("Loss Improvement %", fontweight="bold")
    axes[0].set_title("Training Loss Improvement %")
    axes[0].grid(axis='x', alpha=0.3)
    for i, v in enumerate(improvements):
        axes[0].text(v, i, f" {v:.1f}%", va='center', fontweight='bold')
    
    # Plot 2: Overfitting Indicators
    overfitting = [convergence_analysis[m].get('val_train_loss_ratio', 1.0) for m in models]
    colors_overfit = ['red' if x > 1.2 else 'yellow' if x > 1.1 else 'green' for x in overfitting]
    axes[1].bar(models, overfitting, color=colors_overfit, alpha=0.7, edgecolor='black')
    axes[1].axhline(y=1.0, color='g', linestyle='--', label='Perfect (1.0)', linewidth=2)
    axes[1].axhline(y=1.2, color='r', linestyle='--', label='Overfitting threshold (1.2)', linewidth=2)
    axes[1].set_ylabel("Val/Train Loss Ratio", fontweight="bold")
    axes[1].set_title("Overfitting Detection (Lower is Better)")
    axes[1].legend(fontsize=9)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Convergence Speed
    epochs_to_90 = [convergence_analysis[m].get('epochs_to_90_percent_convergence', 0) for m in models]
    colors_speed = ['green' if x < 50 else 'orange' if x < 100 else 'red' for x in epochs_to_90]
    axes[2].bar(models, epochs_to_90, color=colors_speed, alpha=0.7, edgecolor='black')
    axes[2].set_ylabel("Epochs", fontweight="bold")
    axes[2].set_title("Convergence Speed (Epochs to 90% Improvement)")
    axes[2].grid(axis='y', alpha=0.3)
    for i, (m, v) in enumerate(zip(models, epochs_to_90)):
        axes[2].text(i, v, f" {v}", ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Summary Table
    axes[3].axis('off')
    summary_text = "CONVERGENCE SUMMARY\n" + "="*50 + "\n\n"
    for model in models:
        conv = convergence_analysis[model]
        summary_text += f"{model}:\n"
        summary_text += f"  Improvement: {conv.get('loss_improvement_percent', 0):.1f}%\n"
        summary_text += f"  Val/Train Ratio: {conv.get('val_train_loss_ratio', 1.0):.2f}\n"
        summary_text += f"  Overfitting: {'YES' if conv.get('has_overfitting_indicators', False) else 'NO'}\n\n"
    
    axes[3].text(0.05, 0.95, summary_text, transform=axes[3].transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "convergence_analysis.png"
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def _plot_learning_rate_schedules(models: List[Tuple[str, Dict]]):
    """Plot learning rate schedules if available."""
    has_lr = any('learning_rate' in history for _, history in models)
    if not has_lr:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    for (model_name, history), color in zip(models, colors):
        if 'learning_rate' in history:
            lrs = history['learning_rate']
            ax.plot(range(len(lrs)), lrs, linewidth=2.5, label=model_name, color=color, marker='o', markersize=3, alpha=0.8)
    
    ax.set_xlabel("Epoch", fontweight="bold", fontsize=12)
    ax.set_ylabel("Learning Rate", fontweight="bold", fontsize=12)
    ax.set_title("Learning Rate Schedules", fontweight="bold", fontsize=13)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "learning_rate_schedules.png"
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def report_model_stats():
    """Generate model statistics."""
    print("\n=== Model Statistics ===\n")
    
    stats = {}
    
    # Markov model
    markov_path = MODEL_DIR / "markov.npy"
    if markov_path.exists():
        size_mb = markov_path.stat().st_size / (1024 * 1024)
        stats["Markov"] = {"size_mb": size_mb, "type": "NumPy array", "framework": "NumPy"}
        print(f"Markov: {size_mb:.2f} MB (NumPy transition matrix)")
    
    # GOLC-VAE
    for golc_file in ["golc_vae_latest.pt", "golc_vae_best.pt"]:
        golc_path = MODEL_DIR / golc_file
        if golc_path.exists():
            size_mb = golc_path.stat().st_size / (1024 * 1024)
            stats["GOLC-VAE"] = {"size_mb": size_mb, "type": "PyTorch", "framework": "PyTorch Lightning"}
            print(f"GOLC-VAE: {size_mb:.2f} MB (PyTorch)")
            break
    
    # Transformer
    for tf_file in ["trained_transformer.pt", "transformer_latest.pt"]:
        tf_path = MODEL_DIR / tf_file
        if tf_path.exists():
            size_mb = tf_path.stat().st_size / (1024 * 1024)
            stats["Transformer"] = {"size_mb": size_mb, "type": "PyTorch", "framework": "PyTorch"}
            print(f"Transformer: {size_mb:.2f} MB (PyTorch)")
            break
    
    with open(OUTPUT_DIR / "model_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved: model_statistics.json")
    
    return stats


def generate_summary_report(stats: Dict, convergence_analysis: Dict, training_stats: Dict):
    """Generate comprehensive summary report."""
    print("\n" + "="*80)
    print("COMPREHENSIVE TRAINING METRICS - FINAL SUMMARY")
    print("="*80)
    
    report = {
        "generated_files": [
            "training_loss_comparison.png",
            "convergence_analysis.png",
            "learning_rate_schedules.png",
            "training_*_detailed.png",
            "model_statistics.json",
            "training_metrics_summary.json",
        ],
        "models_analyzed": list(stats.keys()),
        "convergence_analysis": convergence_analysis,
        "training_statistics": training_stats,
        "output_directory": str(OUTPUT_DIR),
        "total_model_size_mb": sum(s.get("size_mb", 0) for s in stats.values()),
    }
    
    with open(OUTPUT_DIR / "training_metrics_summary.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"\nGenerated files:")
    for f in report["generated_files"]:
        print(f"  - {f}")
    print(f"\nModels analyzed: {', '.join(report['models_analyzed'])}")
    print(f"Total model size: {report['total_model_size_mb']:.2f} MB")
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive training metrics")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("COMPREHENSIVE TRAINING METRICS GENERATOR")
    print("="*80)
    print("\nMetrics included:")
    print("  ✓ Loss convergence analysis")
    print("  ✓ Overfitting detection")
    print("  ✓ Learning rate schedules")
    print("  ✓ Training statistics (mean, std, median, etc.)")
    print("  ✓ Convergence speed analysis")
    print("  ✓ Comparative plots across models")
    
    convergence_analysis, training_stats = plot_training_curves()
    stats = report_model_stats()
    generate_summary_report(stats, convergence_analysis, training_stats)


if __name__ == "__main__":
    main()
