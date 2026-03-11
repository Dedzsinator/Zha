#!/usr/bin/env python3
"""
Master script to generate ALL thesis metrics
Runs all three metric generation scripts in sequence
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_name):
    """Run a Python script and handle errors"""
    print(f"\n{'='*70}")
    print(f"RUNNING: {script_name}")
    print('='*70)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False
        )
        print(f"✅ {script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {script_name} failed with error code {e.returncode}")
        return False

def main():
    scripts_dir = Path(__file__).parent
    
    scripts = [
        scripts_dir / "generate_markov_metrics.py",
        scripts_dir / "generate_vae_metrics.py",
        scripts_dir / "generate_transformer_metrics.py"
    ]
    
    print("="*70)
    print("THESIS METRICS GENERATION - MASTER SCRIPT")
    print("="*70)
    print(f"\nWill run {len(scripts)} metric generation scripts:")
    for script in scripts:
        print(f"  - {script.name}")
    print()
    
    results = {}
    for script in scripts:
        if not script.exists():
            print(f"⚠️  Script not found: {script}")
            results[script.name] = False
            continue
        
        success = run_script(str(script))
        results[script.name] = success
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    success_count = sum(results.values())
    total_count = len(results)
    
    for script_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status}: {script_name}")
    
    print(f"\nTotal: {success_count}/{total_count} scripts completed successfully")
    
    if success_count == total_count:
        print("\n🎉 ALL METRICS GENERATED SUCCESSFULLY!")
        print("\n📁 Output directories:")
        print("   - output/metrics/markov/")
        print("   - output/metrics/vae/")
        print("   - output/metrics/transformer/")
    else:
        print("\n⚠️  Some scripts failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
