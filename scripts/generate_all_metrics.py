#!/usr/bin/env python3
"""
Master script to generate comprehensive metrics for music generation models.
"""

import subprocess
import sys
import os
from pathlib import Path


def print_metrics_overview():
    """Print comprehensive metrics overview."""
    print("\n" + "="*80)
    print("COMPREHENSIVE METRICS FRAMEWORK FOR MUSIC GENERATION MODELS")
    print("="*80)
    print("""
METRICS GENERATED:

1. TRAINING METRICS
   - Loss convergence analysis
   - Overfitting detection (val/train loss ratio)
   - Learning rate schedules
   - Training statistics (mean, std, median, percentiles)
   - Convergence speed (epochs to 90% improvement)
   - Loss smoothness and stability

2. MUSICAL QUALITY METRICS
   - Pitch Entropy (Shannon entropy of pitch distribution, 0-4 bits)
   - Unique Pitch Count (diversity of pitches used)
   - Pitch Range (semitone distance min to max)
   - Note Density (fraction of active notes, 0-1)
   - Interval Distribution (distribution of intervals between notes)
   - Chromatic Coherence (scale conformity)

3. STATISTICAL DISTANCE METRICS
   - Wasserstein Distance (optimal transport distance)
   - Jensen-Shannon Divergence (symmetric KL divergence)
   - KL Divergence (forward and reverse, bits)
   - Hellinger Distance (chi-square distance)

4. DIVERSITY METRICS
   - Self-Similarity Score (cosine similarity between sequences)
   - Pitch Coverage (% of MIDI space explored)
   - Sequence Diversity (uniqueness of generated outputs)

5. VAE-SPECIFIC METRICS (if applicable)
   - Reconstruction Error (MSE, MAE, RMSE, R²)
   - KL Divergence (latent space regularization)

6. GENERATION QUALITY
   - Pitch class distribution analysis
   - Generation success rate
   - Output validity checks
    """)
    print("="*80 + "\n")


def run_script(script_name):
    """Run a Python script and handle errors."""
    print(f"\n{'='*80}")
    print(f"RUNNING: {script_name}")
    print('='*80)
    
    try:
        project_root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(project_root)
            if not existing_pythonpath
            else f"{project_root}:{existing_pythonpath}"
        )

        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            cwd=str(project_root),
            env=env,
        )
        print(f"\nSUCCESS: {script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nFAILED: {script_name} failed with error code {e.returncode}")
        return False


def main():
    scripts_dir = Path(__file__).parent
    
    scripts = [
        scripts_dir / "generate_training_capability_metrics.py",
        scripts_dir / "generate_model_capability_metrics.py"
    ]
    
    # Print overview
    print_metrics_overview()
    
    print("="*80)
    print("METRICS GENERATION - MASTER SCRIPT")
    print("="*80)
    
    print(f"\nWill run {len(scripts)} metric generation scripts:")
    for i, script in enumerate(scripts, 1):
        print(f"  {i}. {script.name}")
    print()
    
    results = {}
    for script in scripts:
        if not script.exists():
            print(f"Script not found: {script}")
            results[script.name] = False
            continue
        
        success = run_script(str(script))
        results[script.name] = success
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    success_count = sum(results.values())
    total_count = len(results)
    
    for script_name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"{status}: {script_name}")
    
    print(f"\nTotal: {success_count}/{total_count} scripts completed successfully")
    
    if success_count == total_count:
        print("\n" + "*" * 80)
        print(" ALL METRICS GENERATED SUCCESSFULLY!")
        print("*" * 80)
        print("\nGenerated Output Files:")
        print("\n1. Training Metrics:")
        print("   - output/metrics/training_and_capability/")
        print("     * training_loss_comparison.png")
        print("     * convergence_analysis.png")
        print("     * learning_rate_schedules.png")
        print("     * training_*_detailed.png (per model)")
        print("     * training_metrics_summary.json")
        print("\n2. Capability Metrics:")
        print("   - output/metrics/capabilities/")
        print("     * model_capabilities_comparison.png")
        print("     * pitch_distributions.png")
        print("     * *_metrics.json (per model)")
        print("     * capability_summary.json")
        print("\nIndustry-Standard Metrics Computed:")
        print("  - Training convergence & overfitting detection")
        print("  - Musical quality metrics (pitch, rhythm, etc.)")
        print("  - Statistical distance metrics")
        print("  - Diversity & coverage metrics")
        print("  - Model-specific metrics")
    else:
        print("\nSome scripts failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
