#!/usr/bin/env python3
"""
Comprehensive metrics utilities for music generation models.
Implements industry-standard metrics for evaluating generative models.

Metrics Categories:
1. Statistical Metrics: Distribution analysis, distances
2. Musical Metrics: Pitch, rhythm, harmonic analysis
3. Quality Metrics: Reconstruction, perplexity, diversity
4. Performance Metrics: Speed, memory, throughput
"""

import numpy as np
import scipy.stats as stats

# Handle scipy version compatibility
try:
    from scipy.spatial.distance import wasserstein_distance, jensenshannon
except ImportError:
    from scipy.stats import wasserstein_distance
    from scipy.spatial.distance import jensenshannon

from scipy.stats import entropy, ks_2samp
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time


class MusicMetrics:
    """Comprehensive metrics for music generation evaluation."""
    
    @staticmethod
    def compute_pitch_class_distribution(sequence: np.ndarray) -> np.ndarray:
        """
        Compute pitch class distribution (0-11 chromatic scale).
        
        Args:
            sequence: MIDI note values (0-127)
            
        Returns:
            Normalized 12-bin histogram of pitch classes
        """
        if len(sequence) == 0:
            return np.ones(12) / 12  # Uniform distribution
        
        notes = np.array([n for n in sequence if 0 < n < 128])
        if len(notes) == 0:
            return np.ones(12) / 12
        
        pitch_classes = notes % 12
        hist, _ = np.histogram(pitch_classes, bins=12, range=(0, 12))
        return hist / np.sum(hist)
    
    @staticmethod
    def compute_pitch_range(sequence: np.ndarray) -> Dict[str, int]:
        """Compute pitch range statistics."""
        notes = np.array([n for n in sequence if 0 < n < 128])
        if len(notes) == 0:
            return {"min": 0, "max": 0, "range": 0, "mean": 0, "std": 0}
        
        return {
            "min": int(np.min(notes)),
            "max": int(np.max(notes)),
            "range": int(np.max(notes) - np.min(notes)),
            "mean": float(np.mean(notes)),
            "std": float(np.std(notes)),
        }
    
    @staticmethod
    def compute_pitch_entropy(sequence: np.ndarray) -> float:
        """
        Compute Shannon entropy of pitch distribution.
        Higher entropy = more diverse pitch usage.
        """
        dist = MusicMetrics.compute_pitch_class_distribution(sequence)
        dist = dist[dist > 0]  # Remove zeros for entropy calculation
        return float(entropy(dist, base=2))
    
    @staticmethod
    def compute_unique_pitches(sequence: np.ndarray) -> int:
        """Count unique MIDI pitches."""
        notes = [n for n in sequence if 0 < n < 128]
        return len(set(notes))
    
    @staticmethod
    def compute_note_density(sequence: np.ndarray) -> float:
        """Compute note density (active notes / total length)."""
        if len(sequence) == 0:
            return 0.0
        notes = len([n for n in sequence if 0 < n < 128])
        return float(notes / len(sequence))
    
    @staticmethod
    def compute_interval_distribution(sequence: np.ndarray) -> Dict[str, float]:
        """
        Compute distribution of intervals between consecutive notes.
        """
        notes = np.array([n for n in sequence if 0 < n < 128])
        if len(notes) < 2:
            return {"mean": 0.0, "std": 0.0, "entropy": 0.0}
        
        intervals = np.abs(np.diff(notes))
        
        # Create histogram of intervals (buckets: unison, 2nd, 3rd, 4th, 5th, 6th, 7th, octave, large)
        interval_hist = np.histogram(intervals, bins=[0, 1, 2, 3, 4, 5, 6, 7, 12, 128])[0]
        interval_hist = interval_hist / np.sum(interval_hist)
        
        interval_entropy = entropy(interval_hist[interval_hist > 0], base=2)
        
        return {
            "mean_interval": float(np.mean(intervals)),
            "std_interval": float(np.std(intervals)),
            "interval_entropy": float(interval_entropy),
        }
    
    @staticmethod
    def compute_note_durations_stats(notes_list: List[Tuple[int, float, float]]) -> Dict[str, float]:
        """
        Compute statistics on note durations (for files with timing info).
        
        Args:
            notes_list: List of (pitch, start_time, end_time)
            
        Returns:
            Duration statistics
        """
        if not notes_list or len(notes_list) == 0:
            return {"mean_duration": 0.0, "std_duration": 0.0, "min_duration": 0.0, "max_duration": 0.0}
        
        durations = np.array([end - start for _, start, end in notes_list])
        durations = durations[durations > 0]
        
        if len(durations) == 0:
            return {"mean_duration": 0.0, "std_duration": 0.0, "min_duration": 0.0, "max_duration": 0.0}
        
        return {
            "mean_duration": float(np.mean(durations)),
            "std_duration": float(np.std(durations)),
            "min_duration": float(np.min(durations)),
            "max_duration": float(np.max(durations)),
        }
    
    @staticmethod
    def compute_chromatic_coherence(sequence: np.ndarray) -> float:
        """
        Measure how 'coherent' the sequence is in terms of staying within scales.
        Uses Tonal Entropy and scale conformity.
        """
        dist = MusicMetrics.compute_pitch_class_distribution(sequence)
        
        # Common scales have uneven pitch class distributions
        # Check coherence by entropy (lower = more coherent to specific scale)
        return float(entropy(dist, base=2))
    
    @staticmethod
    def compute_statistical_distance_metrics(generated: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
        """
        Compute statistical distances between generated and reference distributions.
        """
        gen_dist = MusicMetrics.compute_pitch_class_distribution(generated)
        ref_dist = MusicMetrics.compute_pitch_class_distribution(reference)
        
        return {
            "wasserstein_distance": float(wasserstein_distance(gen_dist, ref_dist)),
            "jensen_shannon_distance": float(jensenshannon(gen_dist, ref_dist)),
            "kl_divergence_gen_vs_ref": float(entropy(gen_dist, ref_dist)),
            "kl_divergence_ref_vs_gen": float(entropy(ref_dist, gen_dist)),
            "hellinger_distance": float(np.sqrt(0.5 * np.sum((np.sqrt(gen_dist) - np.sqrt(ref_dist))**2))),
        }
    
    @staticmethod
    def compute_diversity_score(sequences: List[np.ndarray]) -> Dict[str, float]:
        """
        Compute diversity metrics across multiple generated sequences.
        """
        if len(sequences) < 2:
            return {"self_similarity_mean": 0.0, "self_similarity_std": 0.0, "coverage_score": 0.0}
        
        # Compute pairwise similarities
        similarities = []
        for i in range(len(sequences)):
            for j in range(i + 1, len(sequences)):
                sim = MusicMetrics._sequence_similarity(sequences[i], sequences[j])
                similarities.append(sim)
        
        # Coverage: how much of pitch class space is covered
        all_pitches = set()
        for seq in sequences:
            all_pitches.update([n for n in seq if 0 < n < 128])
        coverage = len(all_pitches) / 128.0
        
        return {
            "self_similarity_mean": float(np.mean(similarities)) if similarities else 0.0,
            "self_similarity_std": float(np.std(similarities)) if similarities else 0.0,
            "pitch_coverage": float(coverage),
        }
    
    @staticmethod
    def _sequence_similarity(seq1: np.ndarray, seq2: np.ndarray) -> float:
        """
        Compute cosine similarity between pitch class distributions.
        """
        dist1 = MusicMetrics.compute_pitch_class_distribution(seq1)
        dist2 = MusicMetrics.compute_pitch_class_distribution(seq2)
        
        # Cosine similarity
        dot_product = np.dot(dist1, dist2)
        magnitude = np.linalg.norm(dist1) * np.linalg.norm(dist2)
        
        if magnitude == 0:
            return 0.0
        return float(dot_product / magnitude)
    
    @staticmethod
    def compute_perplexity(probabilities: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute perplexity: exp(-1/N * sum(log(P(target)))).
        Lower is better. Measures how well model predicts.
        """
        if len(probabilities) != len(targets):
            return float('inf')
        
        # Ensure probabilities are valid
        probabilities = np.clip(probabilities, 1e-10, 1.0)
        
        # Sum of log probabilities
        log_probs = np.log(probabilities[np.arange(len(targets)), targets])
        return float(np.exp(-np.mean(log_probs)))
    
    @staticmethod
    def compute_reconstruction_error(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
        """
        Compute reconstruction quality metrics (for VAE models).
        """
        if len(original) != len(reconstructed):
            return {"mse": float('inf'), "mae": float('inf'), "rmse": float('inf')}
        
        mse = float(np.mean((original - reconstructed) ** 2))
        mae = float(np.mean(np.abs(original - reconstructed)))
        rmse = float(np.sqrt(mse))
        
        return {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2_score": float(1.0 - np.sum((original - reconstructed)**2) / np.sum((original - np.mean(original))**2)) if np.var(original) > 0 else 0.0,
        }


class PerformanceMetrics:
    """Metrics for inference performance and resource usage."""
    
    @staticmethod
    def measure_inference_time(model_fn, *args, num_runs: int = 5, **kwargs) -> Dict[str, float]:
        """
        Measure inference time with warmup.
        
        Args:
            model_fn: Function to measure
            num_runs: Number of runs for averaging
            
        Returns:
            Time statistics in milliseconds
        """
        # Warmup
        try:
            model_fn(*args, **kwargs)
        except:
            pass
        
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            try:
                model_fn(*args, **kwargs)
            except:
                continue
            times.append((time.perf_counter() - start) * 1000)  # Convert to ms
        
        if not times:
            return {"mean_time_ms": 0.0, "std_time_ms": 0.0, "min_time_ms": 0.0, "max_time_ms": 0.0}
        
        return {
            "mean_time_ms": float(np.mean(times)),
            "std_time_ms": float(np.std(times)),
            "min_time_ms": float(np.min(times)),
            "max_time_ms": float(np.max(times)),
        }
    
    @staticmethod
    def measure_model_size(model_path: Path) -> Dict[str, float]:
        """Measure model file size and derived metrics."""
        if not model_path.exists():
            return {"size_bytes": 0, "size_mb": 0.0}
        
        size_bytes = model_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        
        return {
            "size_bytes": int(size_bytes),
            "size_mb": float(size_mb),
            "size_gb": float(size_mb / 1024),
        }


class MetricsReporter:
    """Generate comprehensive metrics reports with visualizations."""
    
    @staticmethod
    def create_comprehensive_report(metrics_dict: Dict) -> Dict:
        """
        Create a comprehensive report summary with all metrics organized by category.
        """
        report = {
            "timestamp": str(Path.ctime(Path('.'))),
            "categories": {
                "musical_quality": {},
                "statistical_similarity": {},
                "diversity": {},
                "reconstruction_quality": {},
                "performance": {},
            },
            "model_comparison": {},
        }
        
        return report
    
    @staticmethod
    def export_to_json(metrics: Dict, output_path: Path) -> None:
        """Export metrics to JSON with pretty formatting."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
    
    @staticmethod
    def create_metric_summary_table(metrics: Dict) -> str:
        """Create a human-readable summary table of metrics."""
        lines = []
        lines.append("=" * 80)
        lines.append("COMPREHENSIVE METRICS SUMMARY")
        lines.append("=" * 80)
        
        for category, values in metrics.items():
            if isinstance(values, dict):
                lines.append(f"\n{category.upper().replace('_', ' ')}")
                lines.append("-" * 60)
                for key, val in values.items():
                    if isinstance(val, (int, float)):
                        lines.append(f"  {key:.<45} {val:>10.4f}" if isinstance(val, float) else f"  {key:.<45} {val:>10}")
                    elif isinstance(val, dict):
                        lines.append(f"  {key}:")
                        for k2, v2 in val.items():
                            lines.append(f"    {k2:.<40} {v2:>10.4f}" if isinstance(v2, float) else f"    {k2:.<40} {v2:>10}")
        
        lines.append("\n" + "=" * 80)
        return "\n".join(lines)


# Visualization utilities
def create_distribution_comparison_plot(generated_dist: np.ndarray, reference_dist: np.ndarray, 
                                        title: str = "Pitch Class Distribution Comparison") -> str:
    """
    Returns matplotlib code snippet for distribution comparison.
    (Actual plotting handled in calling script)
    """
    return f"""
import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

# Generated
ax1.bar(range(12), {generated_dist.tolist()}, color='steelblue', alpha=0.7)
ax1.set_title('Generated Distribution')
ax1.set_xlabel('Pitch Class')
ax1.set_ylabel('Probability')

# Reference
ax2.bar(range(12), {reference_dist.tolist()}, color='coral', alpha=0.7)
ax2.set_title('Reference Distribution')
ax2.set_xlabel('Pitch Class')
ax2.set_ylabel('Probability')

# Difference
difference = {(generated_dist - reference_dist).tolist()}
ax3.bar(range(12), difference, color=['green' if x > 0 else 'red' for x in difference], alpha=0.7)
ax3.set_title('Difference (Generated - Reference)')
ax3.set_xlabel('Pitch Class')
ax3.set_ylabel('Difference')

plt.tight_layout()
plt.savefig('{title.lower().replace(" ", "_")}.png', dpi=220, bbox_inches='tight')
plt.close()
"""
