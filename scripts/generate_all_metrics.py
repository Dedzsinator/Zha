#!/usr/bin/env python3
"""
Master script to generate real inference-based capability metrics.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_script(script_name):
    """Run a Python script and handle errors"""
    print(f"\n{'='*70}")
    print(f"RUNNING: {script_name}")
    print('='*70)
    
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
        print(f"✅ {script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {script_name} failed with error code {e.returncode}")
        return False

def main():
    scripts_dir = Path(__file__).parent
    
    scripts = [
        scripts_dir / "generate_model_capability_metrics.py"
    ]
    
    print("="*70)
    print("THESIS METRICS GENERATION - MASTER SCRIPT")
    print("="*70)
    print(f"\nWill run {len(scripts)} metric generation script:")
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
        print("\n📁 Output directory:")
        print("   - output/metrics/capabilities/")
    else:
        print("\n⚠️  Some scripts failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
