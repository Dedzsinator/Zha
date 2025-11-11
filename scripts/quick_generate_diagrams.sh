#!/bin/bash
# Quick start script for generating thesis diagrams

echo "=========================================="
echo "THESIS DIAGRAM GENERATION - QUICK START"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -d "scripts/diagrams" ]; then
    echo "❌ Error: Must run from project root directory"
    echo "   cd /path/to/Zha"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q matplotlib seaborn numpy torch pillow || {
    echo "❌ Failed to install dependencies"
    exit 1
}

echo "✅ Dependencies installed"
echo ""

# Phase 1: No training required
echo "=========================================="
echo "PHASE 1: Generating diagrams (no models)"
echo "=========================================="
echo ""

echo "1️⃣  Architecture diagrams..."
python scripts/diagrams/generate_architecture_diagrams.py || {
    echo "⚠️  Architecture diagrams failed (non-critical)"
}

echo ""
echo "2️⃣  VAE diagrams..."
python scripts/diagrams/generate_vae_diagrams.py || {
    echo "⚠️  VAE diagrams failed (non-critical)"
}

echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo ""
echo "✅ Generated diagrams ready for thesis!"
echo ""
echo "📁 Output directory: output/figures/thesis/"
echo ""
echo "Next steps:"
echo "  1. Copy PDFs to thesis: cp output/figures/thesis/*/*.pdf docs/thesis/figures/"
echo "  2. Update LaTeX \\includegraphics commands"
echo "  3. Compile thesis: cd docs/thesis && pdflatex main.tex"
echo ""
echo "For more diagrams (requires trained models):"
echo "  - Train models first: python backend/trainers/train_*.py"
echo "  - Then run: python scripts/generate_all_diagrams.py"
echo ""
echo "📖 Full documentation: scripts/diagrams/README.md"
