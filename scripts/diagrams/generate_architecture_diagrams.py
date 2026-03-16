#!/usr/bin/env python3
"""
Generate core architecture diagrams (no trained models required)

Generates:
1. System architecture diagram (zha.tex placeholder 1)
2. Transformer architecture diagram (transformer.tex placeholder 1)
3. VAE encoder-decoder diagram (vae.tex placeholder 4)
4. ResidualBlock diagram (vae.tex placeholder 3)
5. Memory mechanism workflow (transformer.tex placeholder 3)
6. Structured generation flowchart (transformer.tex placeholder 5)

Output: output/figures/thesis/architecture/
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np
from pathlib import Path

# Setup
OUTPUT_DIR = Path("output/figures/thesis/architecture")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color scheme
COLORS = {
    'markov': '#FF6B6B',      # Red
    'vae': '#4ECDC4',         # Teal
    'transformer': '#45B7D1', # Blue
    'combined': '#FFA07A',    # Salmon
    'multitrack': '#98D8C8',  # Mint
    'data': '#F7DC6F',        # Yellow
    'process': '#BB8FCE',     # Purple
    'output': '#85C1E2',      # Light blue
}

def save_figure(fig, filename, dpi=300):
    """Save figure as PNG"""
    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f" Saved: {output_path}")
    plt.close(fig)


def draw_arrow(ax, start, end, label='', color='black', style='->', width=2):
    """Draw arrow with label"""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        color=color,
        linewidth=width,
        mutation_scale=20,
        zorder=1
    )
    ax.add_patch(arrow)
    
    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y, label, fontsize=9, ha='center', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='gray', alpha=0.8))


def draw_box(ax, pos, size, label, color, sublabel=''):
    """Draw colored box with label"""
    x, y = pos
    w, h = size
    
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.1",
        facecolor=color,
        edgecolor='black',
        linewidth=2,
        alpha=0.7,
        zorder=2
    )
    ax.add_patch(box)
    
    ax.text(x, y, label, fontsize=11, ha='center', va='center',
            weight='bold', zorder=3)
    
    if sublabel:
        ax.text(x, y - h/2 + 0.15, sublabel, fontsize=8, ha='center', 
                va='top', style='italic', zorder=3)


def generate_system_architecture():
    """
    Diagram 1: Complete Zha system architecture
    zha.tex line 306
    """
    print("\n=== Generating System Architecture Diagram ===")
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(8, 9.5, 'Zha System Architecture', fontsize=18, 
            weight='bold', ha='center')
    
    # Stage 1: Data preprocessing
    draw_box(ax, (2, 8), (2.5, 0.8), 'MIDI Files', COLORS['data'])
    draw_arrow(ax, (3.25, 8), (4.5, 8), 'Parse & Quantize')
    draw_box(ax, (6, 8), (2.5, 0.8), 'Token Sequences', COLORS['data'])
    
    # Stage 2: Key/Scale Analysis
    draw_box(ax, (6, 6.5), (2.5, 0.8), 'Key/Scale\nAnalysis', COLORS['process'])
    draw_arrow(ax, (6, 7.4), (6, 6.9), '')
    
    # Stage 3: Model pipeline
    y_models = 5
    
    # Markov
    draw_box(ax, (2, y_models), (2.5, 1.2), 'Markov Chain', COLORS['markov'],
             'Harmonic Structure')
    
    # VAE
    draw_box(ax, (6, y_models), (2.5, 1.2), 'VAE', COLORS['vae'],
             'Creative Variation')
    
    # Transformer
    draw_box(ax, (10, y_models), (2.5, 1.2), 'Transformer', COLORS['transformer'],
             'Coherent Refinement')
    
    # Arrows between models
    draw_arrow(ax, (3.25, y_models), (4.75, y_models), '')
    draw_arrow(ax, (7.25, y_models), (8.75, y_models), '')
    
    # Input to Markov
    draw_arrow(ax, (6, 5.9), (2, 5.6), 'Key Context', color=COLORS['process'])
    
    # Stage 4: Weighted Combination
    draw_box(ax, (6, 3), (3, 1), 'Weighted\nCombination', COLORS['combined'],
             '0.5M + 0.3V + 0.2T')
    
    # Arrows to combination
    draw_arrow(ax, (2, 4.4), (5, 3.5), '0.5×', color=COLORS['markov'])
    draw_arrow(ax, (6, 4.4), (6, 3.5), '0.3×', color=COLORS['vae'])
    draw_arrow(ax, (10, 4.4), (7, 3.5), '0.2×', color=COLORS['transformer'])
    
    # Stage 5: Filtering
    draw_box(ax, (11, 3), (2.5, 0.8), 'Scale Filter', COLORS['process'])
    draw_box(ax, (14, 3), (2.5, 0.8), 'Register Limit', COLORS['process'])
    
    draw_arrow(ax, (7.5, 3), (9.75, 3), '')
    draw_arrow(ax, (12.25, 3), (12.75, 3), '')
    
    # Stage 6: Output
    draw_box(ax, (12.5, 1.5), (3, 0.8), 'MIDI Export', COLORS['output'])
    draw_arrow(ax, (14, 2.6), (12.5, 1.9), '')
    
    # Multi-track branch
    draw_box(ax, (12.5, 5), (3, 1.2), 'Multi-track\nTransformer', 
             COLORS['multitrack'], 'Melody + Bass + Drums')
    draw_arrow(ax, (12.5, 4.4), (12.5, 1.9), 'Direct Path')
    
    # Legend
    legend_y = 0.5
    ax.text(1, legend_y, 'Legend:', fontsize=10, weight='bold')
    
    legend_items = [
        ('Markov (Structure)', COLORS['markov']),
        ('VAE (Variation)', COLORS['vae']),
        ('Transformer (Coherence)', COLORS['transformer']),
        ('Multi-track (Full)', COLORS['multitrack']),
    ]
    
    for i, (label, color) in enumerate(legend_items):
        x = 2.5 + i * 3
        ax.add_patch(Rectangle((x-0.2, legend_y-0.15), 0.4, 0.3, 
                               facecolor=color, edgecolor='black', alpha=0.7))
        ax.text(x + 0.4, legend_y, label, fontsize=9, va='center')
    
    save_figure(fig, 'system_architecture.png')
    save_figure(fig, 'system_architecture.png', dpi=150)


def generate_transformer_architecture():
    """
    Diagram 2: Transformer architecture details
    transformer.tex line 124
    """
    print("\n=== Generating Transformer Architecture Diagram ===")
    
    # Try to load real model configuration
    import json
    import torch
    
    try:
        config_path = Path("output/trained_models/training_config.json")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Extract transformer parameters
            embed_dim = config.get('embed_dim', 512)
            num_heads = config.get('num_heads', 8)
            num_layers = config.get('num_layers', 8)
            dim_feedforward = config.get('dim_feedforward', 2048)
            
            print(f" Loaded REAL transformer config: {num_layers} layers, {num_heads} heads, dim={embed_dim}")
        else:
            raise FileNotFoundError("Config not found")
            
    except Exception as e:
        print(f"  Could not load real config ({e}), using default values")
        embed_dim = 512
        num_heads = 8
        num_layers = 8
        dim_feedforward = 2048
    
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Title
    ax.text(6, 13.5, 'Zha Transformer Architecture', fontsize=16, 
            weight='bold', ha='center')
    
    # Input
    y = 12.5
    draw_box(ax, (6, y), (3, 0.6), 'Input Token Sequence', COLORS['data'])
    
    # Embedding
    y -= 1.2
    draw_arrow(ax, (6, 12.2), (6, y+0.3), '')
    draw_box(ax, (6, y), (3, 0.6), f'Token Embedding\n(128 → {embed_dim})', COLORS['process'])
    
    # Positional Encoding
    y -= 1.2
    draw_arrow(ax, (6, y+1.5), (6, y+0.3), '')
    draw_box(ax, (6, y), (3, 0.6), 'Rotary Position\nEmbedding', COLORS['process'])
    
    # Transformer Layers (x8)
    for layer_i in range(3):  # Show 3 layers + "..."
        y -= 1.5
        draw_arrow(ax, (6, y+1.2), (6, y+0.8), '')
        
        if layer_i == 1:
            ax.text(6, y, '... (8 layers total) ...', fontsize=10, 
                   ha='center', style='italic')
            continue
        
        layer_label = f'Layer {layer_i+1}' if layer_i == 0 else f'Layer 8'
        
        # Multi-head attention box
        draw_box(ax, (6, y), (4, 0.7), f'{layer_label}: Multi-Head\nAttention (8 heads)', 
                COLORS['transformer'])
        
        # Feed-forward
        y -= 1
        draw_arrow(ax, (6, y+0.65), (6, y+0.35), '')
        draw_box(ax, (6, y), (4, 0.5), 'Feed-Forward\n(512 → 2048 → 512)', 
                COLORS['process'])
        
        # Residual connections
        ax.annotate('', xy=(7.5, y-0.25), xytext=(7.5, y+1.2),
                   arrowprops=dict(arrowstyle='->', color='green', lw=1.5, ls='--'))
        ax.text(8, y+0.5, 'Residual', fontsize=8, color='green', rotation=90, va='center')
    
    # Output head
    y -= 1.2
    draw_arrow(ax, (6, y+1.2), (6, y+0.3), '')
    draw_box(ax, (6, y), (3, 0.6), 'Output Projection\n(512 → 128)', COLORS['output'])
    
    # Softmax
    y -= 1
    draw_arrow(ax, (6, y+0.7), (6, y+0.3), '')
    draw_box(ax, (6, y), (3, 0.6), 'Softmax + Sampling', COLORS['output'])
    
    # Parameters box
    param_text = (
        'Parameters:\n'
        '• Embedding: 512\n'
        '• Heads: 8\n'
        '• Layers: 8\n'
        '• FFN Hidden: 2048\n'
        '• Dropout: 0.1\n'
        '• Total Params: ~42M'
    )
    ax.text(10, 8, param_text, fontsize=9, va='top',
           bbox=dict(boxstyle='round', facecolor='lightyellow', 
                    edgecolor='black', alpha=0.8))
    
    save_figure(fig, 'transformer_architecture.png')
    save_figure(fig, 'transformer_architecture.png', dpi=150)


def generate_vae_architecture():
    """
    Diagram 3: VAE Encoder-Decoder architecture
    vae.tex line 366
    """
    print("\n=== Generating VAE Architecture Diagram ===")
    
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(8, 7.5, 'VAE Encoder-Decoder Architecture', fontsize=16, 
            weight='bold', ha='center')
    
    # Encoder path
    y = 5
    x = 1
    
    draw_box(ax, (x, y), (1.5, 0.8), 'Input\n128', COLORS['data'])
    
    x += 2
    draw_arrow(ax, (x-0.75, y), (x-0.25, y), '')
    draw_box(ax, (x, y), (1.8, 1.2), 'Linear\n128→512', COLORS['process'])
    
    x += 2.2
    draw_arrow(ax, (x-1.1, y), (x-0.25, y), '')
    draw_box(ax, (x, y), (1.8, 1.2), 'ResBlock\n512', COLORS['vae'])
    
    x += 2.2
    draw_arrow(ax, (x-1.1, y), (x-0.25, y), '')
    draw_box(ax, (x, y), (1.8, 1.2), 'Linear\n512→256', COLORS['process'])
    
    x += 2.2
    draw_arrow(ax, (x-1.1, y), (x-0.25, y), '')
    draw_box(ax, (x, y), (1.8, 1.2), 'ResBlock\n256', COLORS['vae'])
    
    # Latent space
    x += 2.2
    draw_arrow(ax, (x-1.1, y), (x-1, y), '')
    draw_box(ax, (x-0.5, y+1), (1.5, 0.7), 'μ\n(latent)', COLORS['output'])
    draw_box(ax, (x-0.5, y-1), (1.5, 0.7), 'log σ²\n(latent)', COLORS['output'])
    
    # Reparameterization
    draw_box(ax, (x+1.5, y), (1.8, 1), 'z = μ + σ·ε\nε~N(0,1)', COLORS['combined'])
    draw_arrow(ax, (x+0.25, y+1), (x+0.6, y), '', color='blue')
    draw_arrow(ax, (x+0.25, y-1), (x+0.6, y), '', color='blue')
    
    # Decoder path (mirror)
    x = 12.5
    y_dec = 3
    
    draw_box(ax, (x, y_dec), (1.8, 1.2), 'ResBlock\n256', COLORS['vae'])
    
    x += 2.2
    draw_arrow(ax, (x-1.1, y_dec), (x-0.25, y_dec), '')
    draw_box(ax, (x, y_dec), (1.8, 1.2), 'Linear\n256→512', COLORS['process'])
    
    x += 2.2
    draw_arrow(ax, (x-1.1, y_dec), (x-0.25, y_dec), '')
    draw_box(ax, (x, y_dec), (1.8, 1.2), 'ResBlock\n512', COLORS['vae'])
    
    # Output
    draw_box(ax, (15, 1), (1.5, 0.8), 'Recon\n128', COLORS['output'])
    draw_arrow(ax, (15, 2.4), (15, 1.4), '')
    
    # Connection from latent to decoder
    draw_arrow(ax, (11.3, y), (12.5, y_dec+0.6), 'Decode', 
              style='->', color='purple', width=2)
    
    # Annotations
    ax.text(5, 6.8, 'ENCODER', fontsize=12, weight='bold', ha='center',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax.text(14, 4.8, 'DECODER', fontsize=12, weight='bold', ha='center',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax.text(10.5, 5.5, 'LATENT\nSPACE', fontsize=11, weight='bold', ha='center',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    save_figure(fig, 'vae_architecture.png')
    save_figure(fig, 'vae_architecture.png', dpi=150)


def generate_residual_block():
    """
    Diagram 4: ResidualBlock detail
    vae.tex line 319
    """
    print("\n=== Generating ResidualBlock Diagram ===")
    
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(4, 9.5, 'ResidualBlock Architecture', fontsize=14, 
            weight='bold', ha='center')
    
    # Input
    y = 8.5
    draw_box(ax, (4, y), (2, 0.6), 'Input x\n(dim)', COLORS['data'])
    
    # Main path
    y -= 1.2
    draw_arrow(ax, (4, y+0.9), (4, y+0.3), '')
    draw_box(ax, (4, y), (2.5, 0.7), 'LayerNorm(dim)', COLORS['process'])
    
    y -= 1.2
    draw_arrow(ax, (4, y+0.85), (4, y+0.35), '')
    draw_box(ax, (4, y), (2.5, 0.7), 'Linear(dim → dim)', COLORS['vae'])
    
    y -= 1.2
    draw_arrow(ax, (4, y+0.85), (4, y+0.35), '')
    draw_box(ax, (4, y), (2.5, 0.7), 'SiLU Activation', COLORS['process'])
    
    y -= 1.2
    draw_arrow(ax, (4, y+0.85), (4, y+0.35), '')
    draw_box(ax, (4, y), (2.5, 0.7), 'Linear(dim → dim)', COLORS['vae'])
    
    # Skip connection
    skip_x = 1.5
    draw_arrow(ax, (3, 8.5), (skip_x, 8.5), '', style='-', color='green', width=2)
    draw_arrow(ax, (skip_x, 8.5), (skip_x, y), '', style='-', color='green', width=2)
    ax.text(skip_x-0.3, 5.5, 'Skip', fontsize=10, color='green', 
           rotation=90, weight='bold', va='center')
    
    # Addition
    y -= 1.2
    draw_arrow(ax, (4, y+0.85), (4, y+0.5), '')
    draw_arrow(ax, (skip_x, y+0.5), (3.5, y+0.5), '', color='green', width=2)
    
    circle = Circle((4, y), 0.35, facecolor='white', edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(4, y, '+', fontsize=20, ha='center', va='center', weight='bold')
    
    # Output
    y -= 1
    draw_arrow(ax, (4, y+0.65), (4, y+0.3), '')
    draw_box(ax, (4, y), (2, 0.6), 'Output\n(dim)', COLORS['output'])
    
    # Formula
    formula = r'$\mathbf{out} = \mathbf{x} + \text{MLP}(\text{LN}(\mathbf{x}))$'
    ax.text(4, 0.5, formula, fontsize=12, ha='center',
           bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black'))
    
    save_figure(fig, 'residual_block.png')
    save_figure(fig, 'residual_block.png', dpi=150)


def generate_memory_mechanism():
    """
    Diagram 5: Memory mechanism workflow
    transformer.tex line 178
    """
    print("\n=== Generating Memory Mechanism Diagram ===")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(7, 7.5, 'Memory-Based Structured Generation', fontsize=16, 
            weight='bold', ha='center')
    
    # Section markers
    sections = ['Intro (A)', 'Verse (B)', 'Chorus (C)', 'Bridge (D)', 'Outro (A)']
    section_colors = ['#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA', '#FFB3BA']
    
    y_timeline = 6
    x_start = 1
    section_width = 2.4
    
    for i, (section, color) in enumerate(zip(sections, section_colors)):
        x = x_start + i * section_width
        draw_box(ax, (x + section_width/2, y_timeline), 
                (section_width - 0.2, 0.8), section, color)
    
    # Memory storage
    y_mem = 4
    draw_box(ax, (3, y_mem), (3, 1.2), 'Section Memory\nStorage', COLORS['process'])
    
    # Arrows from sections to memory
    for i in range(4):  # First 4 sections store to memory
        x = x_start + i * section_width + section_width/2
        draw_arrow(ax, (x, y_timeline - 0.4), (4, y_mem + 0.6), 
                  '', color='blue', style='->')
    
    # Memory recall
    draw_box(ax, (10, y_mem), (3, 1.2), 'Memory Recall\n& Variation', COLORS['vae'])
    
    # Arrow from memory to recall
    draw_arrow(ax, (4.5, y_mem), (8.5, y_mem), 'Query', color='purple', width=2)
    
    # Recall to Outro
    x_outro = x_start + 4 * section_width + section_width/2
    draw_arrow(ax, (10, y_mem + 0.6), (x_outro, y_timeline - 0.4), 
              'Recall A', color='red', width=2, style='->')
    
    # Description boxes
    desc1 = 'Store section patterns\n(melody, harmony, rhythm)'
    ax.text(3, 2.5, desc1, fontsize=9, ha='center',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    desc2 = 'Retrieve & modify\nfor structural coherence'
    ax.text(10, 2.5, desc2, fontsize=9, ha='center',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Timeline axis
    ax.plot([x_start, x_start + 5*section_width], [0.8, 0.8], 
           'k-', linewidth=2)
    ax.text(x_start - 0.5, 0.8, 'Time →', fontsize=11, va='center')
    
    save_figure(fig, 'memory_mechanism.png')
    save_figure(fig, 'memory_mechanism.png', dpi=150)


def generate_structured_generation_flowchart():
    """
    Diagram 6: Structured generation flowchart
    transformer.tex line 232
    """
    print("\n=== Generating Structured Generation Flowchart ===")
    
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'Structured Music Generation Flow', fontsize=16, 
            weight='bold', ha='center')
    
    y = 10.5
    
    # Start
    draw_box(ax, (5, y), (2, 0.6), 'START', COLORS['data'])
    
    # Initialize
    y -= 1.2
    draw_arrow(ax, (5, y+0.9), (5, y+0.3), '')
    draw_box(ax, (5, y), (3.5, 0.8), 'Initialize Memory\nEmpty dict', COLORS['process'])
    
    # Section loop
    y -= 1.4
    draw_arrow(ax, (5, y+1.1), (5, y+0.5), '')
    draw_box(ax, (5, y), (4, 1), 'For each section\n(Intro, Verse, Chorus...)', 
            COLORS['transformer'])
    
    # Check memory
    y -= 1.6
    draw_arrow(ax, (5, y+1.3), (5, y+0.6), '')
    
    diamond_x = [5, 6.5, 5, 3.5]
    diamond_y = [y+0.3, y, y-0.3, y]
    ax.fill(diamond_x, diamond_y, color=COLORS['vae'], alpha=0.7, edgecolor='black', linewidth=2)
    ax.text(5, y, 'Section in\nMemory?', fontsize=9, ha='center', va='center', weight='bold')
    
    # Yes path (recall)
    y_yes = y - 1.2
    ax.text(6.8, y-0.15, 'YES', fontsize=9, weight='bold')
    draw_arrow(ax, (6.5, y), (7.5, y_yes), '', color='green')
    draw_box(ax, (7.5, y_yes), (2.5, 0.8), 'Recall from\nMemory', COLORS['vae'])
    
    # No path (generate)
    y_no = y - 1.2
    ax.text(3, y-0.15, 'NO', fontsize=9, weight='bold')
    draw_arrow(ax, (3.5, y), (2.5, y_no), '', color='red')
    draw_box(ax, (2.5, y_no), (2.5, 0.8), 'Generate\nNew', COLORS['transformer'])
    
    # Merge
    y -= 2.4
    draw_arrow(ax, (7.5, y_yes-0.4), (5, y+0.3), '', color='green')
    draw_arrow(ax, (2.5, y_no-0.4), (5, y+0.3), '', color='red')
    draw_box(ax, (5, y), (3, 0.7), 'Apply Variation', COLORS['combined'])
    
    # Store
    y -= 1.2
    draw_arrow(ax, (5, y+0.65), (5, y+0.3), '')
    draw_box(ax, (5, y), (3.5, 0.7), 'Store in Memory', COLORS['process'])
    
    # Append
    y -= 1.2
    draw_arrow(ax, (5, y+0.65), (5, y+0.3), '')
    draw_box(ax, (5, y), (3, 0.7), 'Append to Output', COLORS['output'])
    
    # Loop back
    draw_arrow(ax, (6.5, y), (8, y), '', style='-', color='blue')
    draw_arrow(ax, (8, y), (8, 8.5), '', style='-', color='blue')
    draw_arrow(ax, (8, 8.5), (6.75, 8.5), '', style='->', color='blue')
    ax.text(8.3, 5, 'Next\nSection', fontsize=8, color='blue', rotation=90, va='center')
    
    # End
    y -= 1.2
    draw_arrow(ax, (5, y+0.65), (5, y+0.3), '')
    draw_box(ax, (5, y), (2, 0.6), 'END', COLORS['output'])
    
    save_figure(fig, 'structured_generation_flowchart.png')
    save_figure(fig, 'structured_generation_flowchart.png', dpi=150)


def main():
    """Generate all architecture diagrams"""
    print("="*80)
    print("GENERATING ARCHITECTURE DIAGRAMS")
    print("="*80)
    
    generate_system_architecture()
    generate_transformer_architecture()
    generate_vae_architecture()
    generate_residual_block()
    generate_memory_mechanism()
    generate_structured_generation_flowchart()
    
    print("\n" + "="*80)
    print(f" All architecture diagrams saved to: {OUTPUT_DIR}")
    print("="*80)
    print("\nGenerated files:")
    for file in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  - {file.name}")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
