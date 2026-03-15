#!/usr/bin/env python3
"""
Generate meaningful capability metrics and charts from TRAINED models only.
No synthetic/mock data is used.

Outputs are written to:
  output/metrics/capabilities/
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from backend.models.markov_chain import MarkovChain
from backend.models.vae import VAEModel
from backend.models.transformer import TransformerModel


plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10

OUTPUT_DIR = Path("output/metrics/capabilities")
MODEL_DIR = Path("output/trained_models")


def _safe_entropy(prob_vec: np.ndarray) -> float:
    p = np.asarray(prob_vec, dtype=np.float64)
    p = np.clip(p, 1e-12, None)
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def _interval_entropy(notes: np.ndarray) -> float:
    if len(notes) < 2:
        return 0.0
    intervals = np.diff(notes)
    values, counts = np.unique(intervals, return_counts=True)
    _ = values
    probs = counts.astype(np.float64) / counts.sum()
    return _safe_entropy(probs)


def _pairwise_diversity(vectors: np.ndarray, n_pairs: int = 2000) -> float:
    if len(vectors) < 2:
        return 0.0
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = vectors / norms

    rng = np.random.default_rng(42)
    idx_a = rng.integers(0, len(unit), size=n_pairs)
    idx_b = rng.integers(0, len(unit), size=n_pairs)
    valid = idx_a != idx_b
    idx_a = idx_a[valid]
    idx_b = idx_b[valid]
    if len(idx_a) == 0:
        return 0.0

    cos = np.sum(unit[idx_a] * unit[idx_b], axis=1)
    return float(1.0 - np.mean(cos))


def load_eval_vectors(max_samples: int = 2000):
    """Load real evaluation vectors from processed dataset."""
    for name in ("full_dataset.pt", "markov_sequences.pt"):
        path = Path("dataset/processed") / name
        if path.exists():
            data_path = path
            break
    else:
        raise FileNotFoundError("No processed dataset found in dataset/processed")

    data = torch.load(data_path)
    sequences = data.get("sequences", data) if isinstance(data, dict) else data

    vectors = []
    for item in sequences:
        seq_data = item.get("sequences", {}) if isinstance(item, dict) else {}
        notes = seq_data.get("full", seq_data.get("melody", []))
        if not notes and isinstance(item, dict):
            notes = item.get("sequence", [])
        if not notes:
            continue

        feature = np.zeros(128, dtype=np.float32)
        for n in notes:
            if isinstance(n, (int, float)) and 0 <= int(n) < 128:
                feature[int(n)] += 1.0
        if feature.sum() > 0:
            feature /= feature.sum()
            vectors.append(feature)
        if len(vectors) >= max_samples:
            break

    if not vectors:
        raise RuntimeError("Could not build evaluation vectors from processed data")

    return np.stack(vectors, axis=0)


def evaluate_markov(n_sequences: int = 256, length: int = 96):
    model_path = MODEL_DIR / "markov.npy"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing Markov model: {model_path}")

    model = MarkovChain()
    ok = model.load(str(model_path))
    if not ok:
        raise RuntimeError("Failed to load Markov model")

    rows = []
    all_notes = []

    for _ in range(n_sequences):
        out = model.generate_with_hmm(length=length, key_context="C major", use_hidden_states=True)
        notes = np.asarray(out.get("notes", []), dtype=np.int64)
        if len(notes) < 2:
            continue
        all_notes.extend(notes.tolist())

        rep_ratio = float(np.mean(notes[1:] == notes[:-1]))
        rows.append({
            "unique_notes": int(len(np.unique(notes))),
            "pitch_range": int(notes.max() - notes.min()),
            "interval_entropy": _interval_entropy(notes),
            "repetition_ratio": rep_ratio,
            "mean_pitch": float(notes.mean()),
        })

    if not rows:
        raise RuntimeError("Markov generation produced no valid sequences")

    df = pd.DataFrame(rows)
    metrics = {
        "n_sequences": int(len(df)),
        "mean_unique_notes": float(df["unique_notes"].mean()),
        "mean_pitch_range": float(df["pitch_range"].mean()),
        "mean_interval_entropy": float(df["interval_entropy"].mean()),
        "mean_repetition_ratio": float(df["repetition_ratio"].mean()),
        "active_notes_generated": int(len(set(all_notes))),
    }

    # Scatter: capability tradeoff
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(
        df["unique_notes"],
        df["interval_entropy"],
        c=df["repetition_ratio"],
        cmap="viridis_r",
        alpha=0.75,
        edgecolors="black",
    )
    plt.colorbar(sc, label="Repetition ratio (lower better)")
    plt.xlabel("Unique notes per sample")
    plt.ylabel("Interval entropy (bits)")
    plt.title("Markov Capability: Diversity vs Interval Complexity")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "markov_scatter_diversity_complexity.png", dpi=220)
    plt.close()

    # Histogram: generated pitch distribution
    plt.figure(figsize=(11, 5))
    plt.hist(all_notes, bins=np.arange(129) - 0.5, color="steelblue", alpha=0.85)
    plt.xlabel("MIDI pitch")
    plt.ylabel("Count")
    plt.title("Markov Generated Pitch Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "markov_pitch_histogram.png", dpi=220)
    plt.close()

    df.to_csv(OUTPUT_DIR / "markov_sequence_metrics.csv", index=False)
    with open(OUTPUT_DIR / "markov_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def _load_vae_state(path: Path, device):
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    return obj


def evaluate_vae(eval_vectors: np.ndarray):
    model_path = MODEL_DIR / "trained_vae.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing VAE model: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAEModel(input_dim=128, latent_dim=128, beta=0.5).to(device)
    state = _load_vae_state(model_path, device)
    model.load_state_dict(state)
    model.eval()

    x = torch.tensor(eval_vectors, dtype=torch.float32, device=device)
    with torch.no_grad():
        recon, mu, logvar = model(x)

    bce = F.binary_cross_entropy(recon, x, reduction="none").sum(dim=1).cpu().numpy()
    kl = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)).cpu().numpy()
    mu_norm = torch.norm(mu, dim=1).cpu().numpy()

    # Sampling-based capability curves
    temperatures = [0.6, 0.8, 1.0, 1.2]
    temp_rows = []
    for t in temperatures:
        with torch.no_grad():
            samples = model.sample(num_samples=512, temperature=t, device=device).cpu().numpy()
        sample_entropy = np.mean([_safe_entropy(s) for s in samples])
        sparsity = float(np.mean(samples < 0.01))
        diversity = _pairwise_diversity(samples)
        temp_rows.append({
            "temperature": t,
            "sample_entropy": float(sample_entropy),
            "sparsity": float(sparsity),
            "diversity": float(diversity),
        })

    temp_df = pd.DataFrame(temp_rows)
    recon_df = pd.DataFrame({"recon_bce": bce, "kl": kl, "mu_norm": mu_norm})

    metrics = {
        "n_eval_samples": int(len(recon_df)),
        "mean_recon_bce": float(recon_df["recon_bce"].mean()),
        "mean_kl": float(recon_df["kl"].mean()),
        "latent_norm_mean": float(recon_df["mu_norm"].mean()),
        "best_diversity_temp": float(temp_df.loc[temp_df["diversity"].idxmax(), "temperature"]),
    }

    # Scatter: recon vs KL
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(
        recon_df["recon_bce"],
        recon_df["kl"],
        c=recon_df["mu_norm"],
        cmap="plasma",
        alpha=0.75,
        edgecolors="black",
    )
    plt.colorbar(sc, label="||mu||")
    plt.xlabel("Reconstruction BCE (lower better)")
    plt.ylabel("KL divergence")
    plt.title("VAE Inference: Reconstruction-Regularization Tradeoff")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "vae_scatter_recon_vs_kl.png", dpi=220)
    plt.close()

    # Temperature curves
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(temp_df["temperature"], temp_df["sample_entropy"], "o-", label="Sample entropy")
    ax1.plot(temp_df["temperature"], temp_df["diversity"], "s-", label="Diversity")
    ax1.set_xlabel("Sampling temperature")
    ax1.set_ylabel("Entropy / Diversity")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(temp_df["temperature"], temp_df["sparsity"], "^-", color="crimson", label="Sparsity")
    ax2.set_ylabel("Sparsity (<0.01)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    plt.title("VAE Sampling Capability Across Temperatures")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "vae_temperature_tradeoffs.png", dpi=220)
    plt.close()

    recon_df.to_csv(OUTPUT_DIR / "vae_reconstruction_metrics.csv", index=False)
    temp_df.to_csv(OUTPUT_DIR / "vae_temperature_metrics.csv", index=False)
    with open(OUTPUT_DIR / "vae_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def _load_transformer_state(path: Path, device):
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    return obj


def _strip_orig_mod_prefix(state_dict):
    if not any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        return state_dict
    return {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in state_dict.items()}


def evaluate_transformer(eval_vectors: np.ndarray):
    model_path = MODEL_DIR / "trained_transformer.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing Transformer model: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel(
        input_dim=128,
        embed_dim=512,
        num_heads=8,
        num_layers=8,
        dim_feedforward=2048,
        dropout=0.1,
        enable_conditioning=True,
    ).to(device)

    state = _strip_orig_mod_prefix(_load_transformer_state(model_path, device))
    model.load_state_dict(state)
    model.eval()

    x = torch.tensor(eval_vectors, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(x)
    if logits.dim() == 3:
        logits = logits.squeeze(1)

    targets = x.argmax(dim=-1)
    ce = F.cross_entropy(logits, targets, reduction="none")
    probs = F.softmax(logits, dim=-1)
    conf, preds = probs.max(dim=-1)
    correct = (preds == targets).float()

    ce_np = ce.cpu().numpy()
    conf_np = conf.cpu().numpy()
    correct_np = correct.cpu().numpy()

    # Generation capability across temperatures
    seed_token = int(targets[0].item())
    seed = torch.zeros(1, 1, 128, device=device)
    seed[0, 0, seed_token] = 1.0

    temp_rows = []
    for temp in [0.6, 0.8, 1.0, 1.2]:
        with torch.no_grad():
            generated = model.generate(seed=seed, steps=128, temperature=temp, top_k=8, top_p=0.95)
        notes = generated.argmax(dim=-1).squeeze(0).cpu().numpy()
        repetition = float(np.mean(notes[1:] == notes[:-1])) if len(notes) > 1 else 0.0
        temp_rows.append({
            "temperature": temp,
            "unique_tokens": int(len(np.unique(notes))),
            "repetition_ratio": repetition,
            "interval_entropy": _interval_entropy(notes.astype(np.int64)),
        })

    temp_df = pd.DataFrame(temp_rows)

    # Confusion on top frequent tokens
    target_np = targets.cpu().numpy()
    pred_np = preds.cpu().numpy()
    top_tokens = pd.Series(target_np).value_counts().head(20).index.tolist()
    token_to_idx = {t: i for i, t in enumerate(top_tokens)}
    conf_mat = np.zeros((len(top_tokens), len(top_tokens)), dtype=np.int64)
    for t, p in zip(target_np, pred_np):
        if t in token_to_idx and p in token_to_idx:
            conf_mat[token_to_idx[t], token_to_idx[p]] += 1

    metrics = {
        "n_eval_samples": int(len(target_np)),
        "top1_accuracy": float(correct_np.mean()),
        "mean_cross_entropy": float(ce_np.mean()),
        "perplexity": float(math.exp(float(ce_np.mean()))),
        "mean_confidence": float(conf_np.mean()),
        "best_temp_unique_tokens": int(temp_df.loc[temp_df["unique_tokens"].idxmax(), "temperature"]),
    }

    # Scatter: confidence vs CE
    plt.figure(figsize=(10, 6))
    plt.scatter(conf_np, ce_np, c=correct_np, cmap="coolwarm", alpha=0.7, edgecolors="black")
    plt.xlabel("Prediction confidence")
    plt.ylabel("Cross-entropy loss")
    plt.title("Transformer Inference: Confidence vs Error")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "transformer_scatter_confidence_vs_error.png", dpi=220)
    plt.close()

    # Temperature tradeoff plot
    plt.figure(figsize=(10, 6))
    plt.plot(temp_df["temperature"], temp_df["unique_tokens"], "o-", label="Unique tokens")
    plt.plot(temp_df["temperature"], temp_df["interval_entropy"], "s-", label="Interval entropy")
    plt.plot(temp_df["temperature"], temp_df["repetition_ratio"], "^-", label="Repetition ratio")
    plt.xlabel("Sampling temperature")
    plt.title("Transformer Generation Capability vs Temperature")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "transformer_temperature_tradeoffs.png", dpi=220)
    plt.close()

    # Confusion heatmap
    if len(top_tokens) > 1:
        plt.figure(figsize=(10, 8))
        plt.imshow(conf_mat, cmap="Blues", aspect="auto")
        plt.colorbar(label="Count")
        plt.xlabel("Predicted token index (top-20 set)")
        plt.ylabel("True token index (top-20 set)")
        plt.title("Transformer Top-Token Confusion Heatmap")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "transformer_confusion_top20.png", dpi=220)
        plt.close()

    pd.DataFrame({
        "cross_entropy": ce_np,
        "confidence": conf_np,
        "correct": correct_np,
    }).to_csv(OUTPUT_DIR / "transformer_eval_metrics.csv", index=False)
    temp_df.to_csv(OUTPUT_DIR / "transformer_temperature_metrics.csv", index=False)
    with open(OUTPUT_DIR / "transformer_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Generate real capability metrics/charts from trained models")
    parser.add_argument("--max-samples", type=int, default=2000, help="Max evaluation samples from processed data")
    parser.add_argument("--markov-sequences", type=int, default=256, help="Number of generated Markov sequences")
    parser.add_argument("--markov-length", type=int, default=96, help="Generated length for Markov sequences")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    eval_vectors = load_eval_vectors(max_samples=args.max_samples)
    markov_metrics = evaluate_markov(n_sequences=args.markov_sequences, length=args.markov_length)
    vae_metrics = evaluate_vae(eval_vectors)
    transformer_metrics = evaluate_transformer(eval_vectors)

    summary = {
        "markov": markov_metrics,
        "vae": vae_metrics,
        "transformer": transformer_metrics,
    }
    with open(OUTPUT_DIR / "capability_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n✅ Capability metrics generated:")
    print(f"   - {OUTPUT_DIR / 'markov_metrics.json'}")
    print(f"   - {OUTPUT_DIR / 'vae_metrics.json'}")
    print(f"   - {OUTPUT_DIR / 'transformer_metrics.json'}")
    print(f"   - {OUTPUT_DIR / 'capability_summary.json'}")


if __name__ == "__main__":
    main()
