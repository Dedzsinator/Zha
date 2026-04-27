#!/usr/bin/env python3
"""
Generate ALL thesis diagrams and visualizations
Creates publication-ready figures for all TODO placeholders

Requirements:
    pip install matplotlib seaborn numpy torch

Run after metrics/training artifacts exist:
    python scripts/generate_all_diagrams.py

Generates thesis diagrams using available real artifacts.
Output: output/figures/thesis/{architecture,markov,vae,transformer,multitrack,system}/
"""

import subprocess
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def run_script(script_path):
    """Run a diagram generation script"""
    print(f"\n{'='*80}")
    print(f"RUNNING: {script_path.name}")
    print('='*80)
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False,
            cwd=script_path.parent.parent  # Run from project root
        )
        print(f"{script_path.name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{script_path.name} failed with error code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"Script not found: {script_path}")
        return False

def main():
    scripts_dir = Path(__file__).parent
    
    # Diagram generation scripts in execution order
    diagram_scripts = [
        # Core architecture diagrams (6 diagrams)
        "diagrams/generate_architecture_diagrams.py",
        
        # VAE diagrams (4 diagrams)
        "diagrams/generate_vae_diagrams.py",
        
        # Markov diagrams (2 diagrams)
        "diagrams/generate_markov_diagrams.py",
        
        # Transformer diagrams (7 diagrams)
        "diagrams/generate_transformer_diagrams.py",
        
        # Multi-track diagrams (4 diagrams)
        "diagrams/generate_multitrack_diagrams.py",
        
        # System integration diagrams (3 diagrams)
        "diagrams/generate_system_diagrams.py",
    ]
    
    print("="*80)
    print("THESIS DIAGRAM GENERATION - MASTER SCRIPT")
    print("="*80)
    print("\nREAL-DATA POLICY: scripts should use tracked artifacts or report unavailable.")
    print(f"\nWill run {len(diagram_scripts)} diagram generation scripts:\n")
    for i, script in enumerate(diagram_scripts, 1):
        script_name = Path(script).stem.replace('generate_', '').replace('_diagrams', '')
        print(f"  {i}. {script_name.upper()}")
    print(f"\nTotal diagrams: 26 (6+4+2+7+4+3)")
    print()
    
    # Check prerequisites
    print("Checking prerequisites...")
    try:
        import matplotlib
        import seaborn
        import numpy
        import torch
        print("All Python packages available")
    except ImportError as e:
        print(f"Missing package: {e}")
        print("Install with: pip install matplotlib seaborn numpy torch graphviz pillow music21")
        return 1
    
    # Run all scripts
    results = {}
    for script_name in diagram_scripts:
        script_path = scripts_dir / script_name
        results[script_name] = run_script(script_path)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    successful = sum(results.values())
    total = len(results)
    
    print(f"\nCompleted: {successful}/{total} scripts")
    print("\nResults:")
    for script, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {status}: {script}")
    
    print(f"\n Output directory: output/figures/thesis/")
    print(f" Copy figures to: docs/thesis/figures/")
    
    return 0 if successful == total else 1

if __name__ == "__main__":
    sys.exit(main())
