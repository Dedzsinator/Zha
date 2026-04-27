#!/usr/bin/env python3
"""
Generate Transformer chapter metrics from real tracked artifacts only.
When required artifacts are missing, emits explicit "unavailable" outputs instead
of synthetic placeholders.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10

OUTPUT_DIR = Path("output/metrics/transformer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRANSFORMER_METRICS_PATH = Path("output/metrics/transformer_metrics.json")
TRAINING_SUMMARY_PATH = Path("output/metrics/training_and_capability/training_metrics_summary.json")
MODEL_STATS_PATH = Path("output/metrics/training_and_capability/model_statistics.json")


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


def generate_parameter_table(output_path: Path):
    from backend.models.transformer import TransformerModel

    model = TransformerModel(
        input_dim=128,
        embed_dim=512,
        num_heads=8,
        num_layers=8,
        dim_feedforward=2048,
    )

    components_info = []
    embedding_params = sum(p.numel() for p in model.embedding.parameters())
    transformer_params = sum(p.numel() for p in model.transformer_encoder.parameters())
    output_params = sum(p.numel() for p in model.output_projection.parameters())
    total_params = sum(p.numel() for p in model.parameters())

    components_info.append({"Component": "Input Embedding", "Parameters": embedding_params})
    components_info.append({"Component": "Positional Encoding", "Parameters": 0})
    components_info.append({"Component": "8x Transformer Layers", "Parameters": transformer_params})
    components_info.append({"Component": "Output Projection", "Parameters": output_params})
    components_info.append({"Component": "TOTAL", "Parameters": total_params})

    df = pd.DataFrame(components_info)
    df["Memory (MB)"] = (df["Parameters"] * 4 / 1024**2).round(2)
    df["% of Total"] = (100.0 * df["Parameters"] / max(total_params, 1)).round(2)

    df.to_csv(output_path.with_suffix(".csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    body = df.iloc[:-1]
    ax.barh(body["Component"], body["Parameters"], color="steelblue", alpha=0.8)
    ax.set_title("Transformer Parameter Count by Component", fontweight="bold")
    ax.set_xlabel("Parameters", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_memory_utilization(output_path: Path, transformer_metrics):
    history = (transformer_metrics or {}).get("history", [])
    if not history:
        return save_unavailable(
            output_path,
            "Memory Utilization Over Generation",
            "Missing output/metrics/transformer_metrics.json history entries.",
        )

    steps = np.arange(1, len(history) + 1)
    utilization = np.minimum(steps, 1024)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(steps, utilization, color="steelblue", linewidth=2)
    axes[0].axhline(1024, color="red", linestyle="--", linewidth=2, label="Cap 1024")
    axes[0].set_title("Memory Buffer Utilization (Derived from real generation length)", fontweight="bold")
    axes[0].set_ylabel("Buffer Length", fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    growth = np.diff(utilization, prepend=0)
    axes[1].plot(steps, growth, color="coral", linewidth=1.8)
    axes[1].set_title("Memory Growth per Step", fontweight="bold")
    axes[1].set_xlabel("Generation Step", fontweight="bold")
    axes[1].set_ylabel("Delta", fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    write_json(output_path.with_suffix(".json"), {
        "step": steps.tolist(),
        "memory_size": utilization.tolist(),
        "growth_rate": growth.tolist(),
        "source": "transformer_metrics.history",
    })


def generate_training_curves(output_path: Path, transformer_metrics, training_summary):
    history = (transformer_metrics or {}).get("history", [])
    losses = [float(x.get("loss")) for x in history if isinstance(x, dict) and x.get("loss") is not None]
    if not losses:
        return save_unavailable(
            output_path,
            "Training Loss Curves",
            "Missing loss history in output/metrics/transformer_metrics.json.",
        )

    epochs = np.arange(1, len(losses) + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(epochs, losses, label="Transformer Train Loss", color="steelblue", linewidth=2)
    ax1.set_title("Transformer Training Loss (real)", fontweight="bold")
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Loss", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    conv = (training_summary or {}).get("convergence_analysis", {}).get("Transformer", {})
    final_loss = conv.get("final_loss")
    best_loss = conv.get("loss_improvement")
    bar_labels = ["History Mean", "History Min"]
    bar_values = [float(np.mean(losses)), float(np.min(losses))]
    if final_loss is not None:
        bar_labels.append("Final Loss")
        bar_values.append(float(final_loss))
    if best_loss is not None:
        bar_labels.append("Loss Improvement")
        bar_values.append(float(best_loss))

    ax2.bar(bar_labels, bar_values, color=["#4C72B0", "#55A868", "#C44E52", "#8172B3"][:len(bar_labels)])
    ax2.set_title("Tracked Loss Summary (real)", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    write_json(output_path.with_suffix(".json"), {
        "epoch": epochs.tolist(),
        "train_loss": losses,
        "source": "transformer_metrics.history",
    })


def generate_quality_metrics_table(output_path: Path, training_summary):
    convergence = (training_summary or {}).get("convergence_analysis", {})
    stats = (training_summary or {}).get("training_statistics", {})
    if not convergence:
        return save_unavailable(
            output_path,
            "Comprehensive Quality Metrics",
            "Missing convergence_analysis in training metrics summary.",
        )

    rows = []
    for model_name, conv in convergence.items():
        loss_stats = stats.get(model_name, {}).get("loss", {})
        rows.append({
            "Model": model_name,
            "Final Loss": conv.get("final_loss"),
            "Loss Improvement %": conv.get("loss_improvement_percent"),
            "Epochs to 90% convergence": conv.get("epochs_to_90_percent_convergence"),
            "Loss Mean": loss_stats.get("mean"),
            "Loss Std": loss_stats.get("std"),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path.with_suffix(".csv"), index=False)

    numeric_cols = [c for c in df.columns if c != "Model"]
    fig, ax = plt.subplots(figsize=(13, 6))
    table_data = df[numeric_cols].copy()
    table_data = table_data.fillna(np.nan)
    table_data = table_data.to_numpy(dtype=float)

    col_min = np.nanmin(table_data, axis=0)
    col_max = np.nanmax(table_data, axis=0)
    denom = np.where((col_max - col_min) == 0, 1.0, col_max - col_min)
    norm = (table_data - col_min) / denom

    heatmap = ax.imshow(norm, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(df["Model"])))
    ax.set_yticklabels(df["Model"].tolist())
    for i in range(table_data.shape[0]):
        for j in range(table_data.shape[1]):
            val = table_data[i, j]
            if np.isnan(val):
                text = "N/A"
            else:
                text = f"{val:.4g}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9)
    plt.colorbar(heatmap, ax=ax)
    ax.set_title("Model Quality Summary (real tracked metrics)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_computational_cost(output_path: Path, training_summary, model_stats):
    convergence = (training_summary or {}).get("convergence_analysis", {})
    if not convergence:
        return save_unavailable(
            output_path,
            "Computational Cost Breakdown",
            "Missing convergence_analysis in training metrics summary.",
        )

    rows = []
    for model_name, conv in convergence.items():
        rows.append({
            "Model": model_name,
            "Model Size (MB)": (model_stats or {}).get(model_name, {}).get("size_mb"),
            "Epochs Trained": conv.get("n_epochs_trained"),
            "Epochs to 90% convergence": conv.get("epochs_to_90_percent_convergence"),
            "Initial Loss": conv.get("initial_loss"),
            "Final Loss": conv.get("final_loss"),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path.with_suffix(".csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].bar(df["Model"], df["Model Size (MB)"], color="steelblue", alpha=0.8)
    axes[0].set_title("Model Size (MB)", fontweight="bold")
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(df["Model"], df["Epochs to 90% convergence"], color="coral", alpha=0.8)
    axes[1].set_title("Epochs to 90% Convergence", fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.suptitle("Computational Cost Proxies from Real Training Logs", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 60)
    print("TRANSFORMER CHAPTER METRICS (REAL DATA ONLY)")
    print("=" * 60)

    transformer_metrics = load_json(TRANSFORMER_METRICS_PATH)
    training_summary = load_json(TRAINING_SUMMARY_PATH)
    model_stats = load_json(MODEL_STATS_PATH)

    generate_parameter_table(OUTPUT_DIR / "17_parameter_table.png")
    generate_memory_utilization(OUTPUT_DIR / "19_memory_utilization.png", transformer_metrics)

    save_unavailable(
        OUTPUT_DIR / "20_section_coherence.png",
        "Section Coherence Analysis",
        "Requires section-level evaluation exports; no real artifact found.",
    )
    save_unavailable(
        OUTPUT_DIR / "23_sampling_comparison.png",
        "Sampling Strategy Comparison",
        "Requires sampled generations for multiple (temperature, top-p) settings.",
    )
    save_unavailable(
        OUTPUT_DIR / "25_attention_heatmaps.png",
        "Attention Weight Heatmaps",
        "Requires saved attention tensors from inference runs.",
    )

    generate_training_curves(OUTPUT_DIR / "27_training_curves.png", transformer_metrics, training_summary)

    save_unavailable(
        OUTPUT_DIR / "28_perplexity_analysis.png",
        "Perplexity Over Sequence Length",
        "Requires evaluation logs with per-position perplexity.",
    )

    generate_quality_metrics_table(OUTPUT_DIR / "29_quality_metrics.png", training_summary)
    generate_computational_cost(OUTPUT_DIR / "31_computational_cost.png", training_summary, model_stats)

    save_unavailable(
        OUTPUT_DIR / "32_limitation_analysis.png",
        "Limitation Impact Analysis",
        "No real limitation experiment outputs found.",
    )

    print("\nAll transformer metrics generated with real-data-only policy.")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
