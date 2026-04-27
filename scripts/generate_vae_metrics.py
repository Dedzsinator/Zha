#!/usr/bin/env python3
"""
Generate VAE chapter metrics from real tracked artifacts only.
Missing artifacts are reported explicitly as unavailable.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10

OUTPUT_DIR = Path("output/metrics/vae")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_SUMMARY_PATH = Path("output/metrics/training_and_capability/training_metrics_summary.json")
CAPABILITY_PATH = Path("output/metrics/capabilities/golc-vae_metrics.json")


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_unavailable(output_path: Path, title: str, reason: str):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0.5, 0.40, "Unavailable", ha="center", va="center", fontsize=12, color="crimson")
    ax.text(0.5, 0.24, reason, ha="center", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    write_json(output_path.with_suffix(".json"), {
        "status": "unavailable",
        "title": title,
        "reason": reason,
    })


def generate_training_curves(output_path: Path, training_summary):
    conv = (training_summary or {}).get("convergence_analysis", {}).get("GOLC-VAE", {})
    if not conv:
        return save_unavailable(
            output_path,
            "KL Divergence and Reconstruction Loss Training Curves",
            "Missing GOLC-VAE convergence data in training summary.",
        )

    n_epochs = int(conv.get("n_epochs_trained", 0) or 0)
    initial = conv.get("initial_loss")
    final = conv.get("final_loss")
    best = conv.get("best_val_loss")
    best_epoch = conv.get("best_val_epoch")

    if n_epochs <= 0 or initial is None or final is None:
        return save_unavailable(
            output_path,
            "KL Divergence and Reconstruction Loss Training Curves",
            "Insufficient epoch/loss information in convergence summary.",
        )

    epochs = np.array([1, n_epochs], dtype=int)
    total_loss_points = np.array([float(initial), float(final)], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(epochs, total_loss_points, "o-", color="purple", linewidth=2, label="Total Loss")
    axes[0, 0].set_title("Total Training Loss (real)", fontweight="bold")
    axes[0, 0].set_xlabel("Epoch", fontweight="bold")
    axes[0, 0].set_ylabel("Loss", fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].axis("off")
    axes[0, 1].text(0.5, 0.5, "KL curve unavailable\n(per-epoch KL not logged)", ha="center", va="center")

    axes[1, 0].axis("off")
    axes[1, 0].text(0.5, 0.5, "Reconstruction curve unavailable\n(per-epoch reconstruction loss not logged)", ha="center", va="center")

    summary_labels = ["Initial", "Final"]
    summary_values = [float(initial), float(final)]
    if best is not None:
        summary_labels.append("Best Val")
        summary_values.append(float(best))

    axes[1, 1].bar(summary_labels, summary_values, color=["#4C72B0", "#55A868", "#C44E52"][:len(summary_labels)])
    if best is not None and best_epoch is not None:
        axes[1, 1].set_title(f"Tracked Loss Summary (best val @ epoch {best_epoch})", fontweight="bold")
    else:
        axes[1, 1].set_title("Tracked Loss Summary", fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.suptitle("VAE Training Metrics (real artifacts only)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    write_json(output_path.with_suffix(".json"), {
        "status": "partial",
        "epochs": epochs.tolist(),
        "total_loss": total_loss_points.tolist(),
        "best_val_loss": best,
        "best_val_epoch": best_epoch,
        "source": "training_and_capability/training_metrics_summary.json",
    })


def generate_parameter_table(output_path: Path):
    from backend.models.vae import VAEModel

    model = VAEModel(input_dim=128, latent_dim=64, beta=1.0)

    rows = []
    total_params = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            params = sum(p.numel() for p in module.parameters())
            total_params += params
            rows.append({
                "Layer": name,
                "Shape": f"{module.in_features} -> {module.out_features}",
                "Parameters": params,
            })

    if not rows:
        return save_unavailable(output_path, "Architecture Parameter Count Table", "No linear layers found.")

    df = pd.DataFrame(rows)
    df["Memory (MB)"] = (df["Parameters"] * 4 / 1024**2).round(4)
    df["% of Total"] = (100.0 * df["Parameters"] / max(total_params, 1)).round(3)

    total_row = pd.DataFrame([{
        "Layer": "TOTAL",
        "Shape": "-",
        "Parameters": total_params,
        "Memory (MB)": round(total_params * 4 / 1024**2, 4),
        "% of Total": 100.0,
    }])
    final_df = pd.concat([df, total_row], ignore_index=True)
    final_df.to_csv(output_path.with_suffix(".csv"), index=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(df["Layer"], df["Parameters"], color="steelblue", alpha=0.8)
    ax.set_title("VAE Parameter Count by Layer", fontweight="bold")
    ax.set_xlabel("Parameters", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_capability_summary(output_path: Path, capability_metrics):
    if not capability_metrics:
        return save_unavailable(
            output_path,
            "GOLC-VAE Capability Summary",
            "Missing output/metrics/capabilities/golc-vae_metrics.json.",
        )

    musical_quality = capability_metrics.get("musical_quality", {})
    diversity = capability_metrics.get("diversity", {})
    chromatic = capability_metrics.get("chromatic_coherence", {})

    rows = [
        {"Metric": "Pitch Entropy (mean)", "Value": musical_quality.get("pitch_entropy", {}).get("mean")},
        {"Metric": "Unique Pitches (mean)", "Value": musical_quality.get("unique_pitches", {}).get("mean")},
        {"Metric": "Note Density (mean)", "Value": musical_quality.get("note_density", {}).get("mean")},
        {"Metric": "Pitch Range (mean)", "Value": musical_quality.get("pitch_range", {}).get("mean")},
        {"Metric": "Interval Entropy (mean)", "Value": musical_quality.get("interval_entropy", {}).get("mean")},
        {"Metric": "Self Similarity (mean)", "Value": diversity.get("self_similarity_mean")},
        {"Metric": "Pitch Coverage", "Value": diversity.get("pitch_coverage")},
        {"Metric": "Chromatic Coherence (mean)", "Value": chromatic.get("mean")},
    ]

    df = pd.DataFrame(rows)
    df.to_csv(output_path.with_suffix(".csv"), index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(df["Metric"], pd.to_numeric(df["Value"], errors="coerce"), color="seagreen", alpha=0.8)
    ax.set_title("GOLC-VAE Capability Metrics (real)", fontweight="bold")
    ax.set_xlabel("Value", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 60)
    print("VAE CHAPTER METRICS (REAL DATA ONLY)")
    print("=" * 60)

    training_summary = load_json(TRAINING_SUMMARY_PATH)
    capability_metrics = load_json(CAPABILITY_PATH)

    generate_training_curves(OUTPUT_DIR / "7_training_curves.png", training_summary)

    save_unavailable(
        OUTPUT_DIR / "8_dimensionwise_kl.png",
        "Latent Space Dimensionwise KL Distribution",
        "No per-dimension KL logs were found in exported artifacts.",
    )

    generate_parameter_table(OUTPUT_DIR / "11_parameter_table.png")

    save_unavailable(
        OUTPUT_DIR / "12_temperature_ablation.png",
        "Temperature Ablation Study",
        "No temperature sweep evaluation artifact found.",
    )

    generate_capability_summary(OUTPUT_DIR / "14_vae_golc_comparison.png", capability_metrics)

    save_unavailable(
        OUTPUT_DIR / "15_tsne_latent.png",
        "Latent Space t-SNE Visualization",
        "No exported latent vectors found for t-SNE projection.",
    )

    print("\nAll VAE metrics generated with real-data-only policy.")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
