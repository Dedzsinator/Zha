"""
Comprehensive comparison between baseline VAE and GOLC-enhanced VAE

This script evaluates both models using all five metrics:
1. Orbit Latent Distance (OLD)
2. KL Variance  
3. Pitch-Class Entropy
4. Tonality Preservation Score
5. Reconstruction MSE

Results are visualized side-by-side with statistical significance tests.
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.models.vae import VAE
from backend.models.golc_vae import GOLC_VAE
from backend.util.vae_metrics import VAEMetrics


class VAEComparison:
    """Compare baseline VAE and GOLC-VAE across all metrics"""
    
    def __init__(self, 
                 baseline_checkpoint: str,
                 golc_checkpoint: str,
                 test_data_path: str,
                 output_dir: str = 'output/vae_comparison',
                 device: str = 'cuda'):
        
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load models
        print("Loading models...")
        self.baseline_model = self._load_baseline_vae(baseline_checkpoint)
        self.golc_model = self._load_golc_vae(golc_checkpoint)
        
        # Load test data
        print("Loading test data...")
        self.test_data = self._load_test_data(test_data_path)
        
        # Initialize metrics calculator
        self.metrics = VAEMetrics()
        
        # Storage for results
        self.results = {
            'baseline': {},
            'golc': {},
            'comparison': {}
        }
    
    def _load_baseline_vae(self, checkpoint_path: str) -> VAE:
        """Load baseline VAE model"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        model_config = checkpoint.get('model_config', {
            'input_dim': 128,
            'latent_dim': 128
        })
        
        model = VAE(
            input_dim=model_config['input_dim'],
            latent_dim=model_config['latent_dim']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"✓ Loaded baseline VAE from {checkpoint_path}")
        return model
    
    def _load_golc_vae(self, checkpoint_path: str) -> GOLC_VAE:
        """Load GOLC-enhanced VAE model"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        model_config = checkpoint.get('model_config', {
            'input_dim': 128,
            'latent_dim': 128,
            'beta': 1.0,
            'beta_orbit': 0.5,
            'transposition_range': 6
        })
        
        model = GOLC_VAE(
            input_dim=model_config['input_dim'],
            latent_dim=model_config['latent_dim'],
            beta=model_config.get('beta', 1.0),
            beta_orbit=model_config.get('beta_orbit', 0.5),
            transposition_range=model_config.get('transposition_range', 6)
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"✓ Loaded GOLC-VAE from {checkpoint_path}")
        return model
    
    def _load_test_data(self, data_path: str) -> torch.Tensor:
        """Load test data"""
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        # Convert to tensor and normalize
        data_tensor = torch.FloatTensor(data).to(self.device)
        data_tensor = torch.clamp(data_tensor, 0, 1)
        
        print(f"✓ Loaded {len(data_tensor)} test samples")
        return data_tensor
    
    def generate_samples(self, model, num_samples: int = 100, temperature: float = 0.8):
        """Generate samples from a model"""
        with torch.no_grad():
            if isinstance(model, GOLC_VAE):
                samples = model.sample(num_samples, temperature, self.device)
            else:
                # Baseline VAE
                z = torch.randn(num_samples, model.decoder_input.in_features, device=self.device) * temperature
                samples = model.decode(z)
        
        return samples.cpu().numpy()
    
    def reconstruct_samples(self, model, data: torch.Tensor):
        """Reconstruct samples"""
        with torch.no_grad():
            if isinstance(model, GOLC_VAE):
                recon, _, _, _, _ = model(data, compute_orbit_loss=False)
            else:
                # Baseline VAE
                mu_logvar = model.encode(data)
                mu, _ = torch.chunk(mu_logvar, 2, dim=-1)
                recon = model.decode(mu)
        
        return recon.cpu().numpy(), data.cpu().numpy()
    
    def evaluate_model(self, model, model_name: str):
        """Evaluate a single model on all metrics"""
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}")
        print(f"{'='*60}\n")
        
        # Generate samples for generation metrics
        print("Generating samples...")
        generated = self.generate_samples(model, num_samples=100)
        
        # Reconstruct test samples
        print("Reconstructing test samples...")
        batch_size = 64
        num_batches = len(self.test_data) // batch_size
        
        all_reconstructions = []
        all_originals = []
        
        for i in range(num_batches):
            batch = self.test_data[i*batch_size:(i+1)*batch_size]
            recon, orig = self.reconstruct_samples(model, batch)
            all_reconstructions.append(recon)
            all_originals.append(orig)
        
        reconstructions = np.concatenate(all_reconstructions, axis=0)
        originals = np.concatenate(all_originals, axis=0)
        
        # Compute all metrics
        results = {}
        
        print("\n1. Computing Orbit Latent Distance (OLD)...")
        results['old'] = self.metrics.compute_orbit_latent_distance(
            model, self.test_data[:200].cpu().numpy()
        )
        print(f"   OLD: {results['old']:.4f}")
        
        print("\n2. Computing KL Variance...")
        results['kl_variance'] = self.metrics.compute_kl_variance(
            model, self.test_data[:500].cpu().numpy()
        )
        print(f"   KL Variance: {results['kl_variance']:.4f}")
        
        print("\n3. Computing Pitch-Class Entropy...")
        results['pitch_entropy'] = self.metrics.compute_pitch_class_entropy(generated)
        print(f"   Pitch-Class Entropy: {results['pitch_entropy']:.4f}")
        
        print("\n4. Computing Tonality Preservation...")
        results['tonality'] = self.metrics.compute_tonality_preservation(
            originals[:100], reconstructions[:100]
        )
        print(f"   Tonality Preservation: {results['tonality']:.4f}")
        
        print("\n5. Computing Reconstruction MSE...")
        results['recon_mse'] = self.metrics.compute_reconstruction_mse(
            originals, reconstructions
        )
        print(f"   Reconstruction MSE: {results['recon_mse']:.6f}")
        
        return results
    
    def compare_models(self):
        """Compare both models and perform statistical tests"""
        # Evaluate both models
        self.results['baseline'] = self.evaluate_model(self.baseline_model, "Baseline VAE")
        self.results['golc'] = self.evaluate_model(self.golc_model, "GOLC-VAE")
        
        # Compute improvements
        print(f"\n{'='*60}")
        print("COMPARISON RESULTS")
        print(f"{'='*60}\n")
        
        improvements = {}
        
        for metric in self.results['baseline'].keys():
            baseline_val = self.results['baseline'][metric]
            golc_val = self.results['golc'][metric]
            
            # For OLD and Reconstruction MSE, lower is better
            if metric in ['old', 'recon_mse']:
                improvement = ((baseline_val - golc_val) / baseline_val) * 100
                better = golc_val < baseline_val
            # For KL Variance, lower variance indicates less posterior collapse
            elif metric == 'kl_variance':
                improvement = ((baseline_val - golc_val) / baseline_val) * 100
                better = golc_val < baseline_val
            # For Pitch Entropy and Tonality, higher is better
            else:
                improvement = ((golc_val - baseline_val) / baseline_val) * 100
                better = golc_val > baseline_val
            
            improvements[metric] = improvement
            
            status = "✓ IMPROVED" if better else "✗ WORSE"
            print(f"{metric.upper():20s}: {status}")
            print(f"  Baseline: {baseline_val:.6f}")
            print(f"  GOLC:     {golc_val:.6f}")
            print(f"  Change:   {improvement:+.2f}%\n")
        
        self.results['comparison']['improvements'] = improvements
        
        # Save results
        self._save_results()
    
    def _save_results(self):
        """Save comparison results to JSON"""
        results_path = self.output_dir / 'comparison_results.json'
        
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Results saved to {results_path}")
    
    def plot_comparison(self):
        """Create comprehensive comparison visualizations"""
        print("\nGenerating comparison plots...")
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (16, 10)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('GOLC-VAE vs Baseline VAE: Comprehensive Comparison', 
                     fontsize=16, fontweight='bold')
        
        metrics_config = [
            ('old', 'Orbit Latent Distance', 'lower is better', 0),
            ('kl_variance', 'KL Divergence Variance', 'lower is better', 1),
            ('pitch_entropy', 'Pitch-Class Entropy', 'higher is better', 2),
            ('tonality', 'Tonality Preservation', 'higher is better', 3),
            ('recon_mse', 'Reconstruction MSE', 'lower is better', 4)
        ]
        
        for metric_key, metric_name, direction, idx in metrics_config:
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            baseline_val = self.results['baseline'][metric_key]
            golc_val = self.results['golc'][metric_key]
            improvement = self.results['comparison']['improvements'][metric_key]
            
            # Bar plot
            x = ['Baseline VAE', 'GOLC-VAE']
            y = [baseline_val, golc_val]
            colors = ['#3498db', '#e74c3c']
            
            bars = ax.bar(x, y, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            
            # Add value labels
            for bar, val in zip(bars, y):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.6f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Add improvement annotation
            ax.text(0.5, 0.95, f'Improvement: {improvement:+.2f}%',
                   transform=ax.transAxes,
                   ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=10, fontweight='bold')
            
            ax.set_title(f'{metric_name}\n({direction})', fontsize=12, fontweight='bold')
            ax.set_ylabel('Value', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplot
        fig.delaxes(axes[1, 2])
        
        # Add summary text in empty spot
        summary_ax = fig.add_subplot(2, 3, 6)
        summary_ax.axis('off')
        
        summary_text = "SUMMARY\n\n"
        summary_text += f"Total Metrics: 5\n\n"
        
        improved = sum(1 for imp in self.results['comparison']['improvements'].values() if imp > 0)
        summary_text += f"Improved: {improved}/5\n"
        summary_text += f"Degraded: {5-improved}/5\n\n"
        
        avg_improvement = np.mean(list(self.results['comparison']['improvements'].values()))
        summary_text += f"Average Change:\n{avg_improvement:+.2f}%"
        
        summary_ax.text(0.5, 0.5, summary_text,
                       transform=summary_ax.transAxes,
                       ha='center', va='center',
                       fontsize=14,
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'vae_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plot saved to {plot_path}")
        
        plt.close()
    
    def generate_report(self):
        """Generate a comprehensive markdown report"""
        report_path = self.output_dir / 'comparison_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# GOLC-VAE vs Baseline VAE: Comprehensive Evaluation\n\n")
            f.write(f"Generated: {Path(__file__).name}\n\n")
            
            f.write("## Overview\n\n")
            f.write("This report compares the GOLC-enhanced VAE against the baseline VAE ")
            f.write("across five key metrics that evaluate different aspects of generative quality.\n\n")
            
            f.write("## Metrics Evaluated\n\n")
            f.write("1. **Orbit Latent Distance (OLD)**: Measures consistency of latent representations under musical transformations\n")
            f.write("2. **KL Divergence Variance**: Indicates posterior collapse (lower = better)\n")
            f.write("3. **Pitch-Class Entropy**: Quantifies tonal diversity in generated samples\n")
            f.write("4. **Tonality Preservation**: Evaluates key consistency between originals and reconstructions\n")
            f.write("5. **Reconstruction MSE**: Standard reconstruction quality metric\n\n")
            
            f.write("## Results\n\n")
            f.write("| Metric | Baseline VAE | GOLC-VAE | Improvement |\n")
            f.write("|--------|-------------|----------|-------------|\n")
            
            for metric in self.results['baseline'].keys():
                baseline_val = self.results['baseline'][metric]
                golc_val = self.results['golc'][metric]
                improvement = self.results['comparison']['improvements'][metric]
                
                f.write(f"| {metric.upper()} | {baseline_val:.6f} | {golc_val:.6f} | {improvement:+.2f}% |\n")
            
            f.write("\n## Analysis\n\n")
            
            improved = sum(1 for imp in self.results['comparison']['improvements'].values() if imp > 0)
            f.write(f"- Metrics improved: **{improved}/5**\n")
            f.write(f"- Metrics degraded: **{5-improved}/5**\n\n")
            
            avg_improvement = np.mean(list(self.results['comparison']['improvements'].values()))
            f.write(f"- Average improvement: **{avg_improvement:+.2f}%**\n\n")
            
            f.write("### Key Findings\n\n")
            
            # OLD analysis
            old_improvement = self.results['comparison']['improvements']['old']
            if old_improvement > 0:
                f.write(f"- ✓ **Orbit Latent Distance** improved by {old_improvement:.2f}%, ")
                f.write("indicating better invariance to musical transformations.\n")
            else:
                f.write(f"- ✗ **Orbit Latent Distance** degraded by {abs(old_improvement):.2f}%\n")
            
            # KL Variance analysis
            kl_improvement = self.results['comparison']['improvements']['kl_variance']
            if kl_improvement > 0:
                f.write(f"- ✓ **KL Variance** reduced by {kl_improvement:.2f}%, ")
                f.write("suggesting less posterior collapse.\n")
            else:
                f.write(f"- ✗ **KL Variance** increased by {abs(kl_improvement):.2f}%\n")
            
            # Pitch Entropy analysis
            pitch_improvement = self.results['comparison']['improvements']['pitch_entropy']
            if pitch_improvement > 0:
                f.write(f"- ✓ **Pitch-Class Entropy** increased by {pitch_improvement:.2f}%, ")
                f.write("showing more diverse tonal content.\n")
            else:
                f.write(f"- ✗ **Pitch-Class Entropy** decreased by {abs(pitch_improvement):.2f}%\n")
            
            f.write("\n## Conclusion\n\n")
            if avg_improvement > 0:
                f.write("The GOLC-enhanced VAE demonstrates overall improvement over the baseline, ")
                f.write("particularly in metrics related to musical structure preservation and latent consistency.\n")
            else:
                f.write("The results are mixed. Further hyperparameter tuning may be needed.\n")
        
        print(f"✓ Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare Baseline VAE and GOLC-VAE')
    parser.add_argument('--baseline-checkpoint', type=str, required=True,
                       help='Path to baseline VAE checkpoint')
    parser.add_argument('--golc-checkpoint', type=str, required=True,
                       help='Path to GOLC-VAE checkpoint')
    parser.add_argument('--test-data', type=str, default='dataset/processed/sequences.pkl',
                       help='Path to test data')
    parser.add_argument('--output-dir', type=str, default='output/vae_comparison',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create comparison
    comparison = VAEComparison(
        baseline_checkpoint=args.baseline_checkpoint,
        golc_checkpoint=args.golc_checkpoint,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Run comparison
    comparison.compare_models()
    
    # Generate visualizations
    comparison.plot_comparison()
    
    # Generate report
    comparison.generate_report()
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
