# Thesis Defence Presentation - README

## Overview
Created a Hungarian Beamer presentation for your thesis defence on "Music Synthesis with Neural Networks - GOLC VAE"

**File**: `thesis_defence.tex`  
**Compiled PDF**: `thesis_defence.pdf` (246 KB, 16 pages)  
**Language**: Hungarian  
**Duration**: Designed for 13-15 minutes presentation

## Presentation Structure (15 slides)

### 1. **Title Slide**
   - Your name: Dégi Nándor
   - Advisor: Dr. Csató Lehel
   - Date: Uses \today (automatically shows current date)

### 2. **Miért csináltam ezt a projektet?** (Why did I do this project?)
   - Personal motivation: You play music and wanted your own band
   - Curiosity: Can AI create something good?
   - Goal: System to help generate musical ideas

### 3. **Projekt áttekintés** (Project Overview)
   - Research questions
   - Implemented models: Markov, VAE, **GOLC-VAE**, Transformer, Diffusion
   - Dataset information

### 4. **Rendszer architektúra** (System Architecture)
   - Flow diagram: MIDI → Preprocessing → Models → Postprocessing → Output
   - Technology stack (Python, PyTorch, FastAPI, Docker)

### 5. **Markov-láncok** (Markov Chains)
   - Basic principle and mathematics
   - Implementation features (HMM, GPU acceleration)
   - Pros and cons

### 6. **VAE Alapok** (VAE Basics)
   - Architecture diagram (Encoder → Latent Space → Decoder)
   - Loss function (Reconstruction + KL divergence)
   - Advantages

### 7. **A probléma: Transzpozíciós invariancia** (The Problem)
   - Why standard VAE doesn't handle transposition well
   - Redundancy in latent space
   - Introduction to GOLC solution

### 8. **GOLC-VAE: Saját innovációm** (GOLC-VAE: My Innovation)
   - **YOUR ORIGINAL CONTRIBUTION**
   - Key concepts: Orbit, Canonical representation, Orbital consistency
   - Enhanced loss function with orbital term

### 9. **GOLC-VAE: Implementáció** (GOLC-VAE: Implementation)
   - **Code snippet** showing orbital loss computation
   - Small and focused code example (~10 lines)

### 10. **További modellek** (Other Models)
   - Transformer model overview
   - Diffusion model overview
   - Comparison of strengths/weaknesses

### 11. **Eredmények: Összehasonlítás** (Results: Comparison)
   - Table comparing all models
   - **GOLC-VAE shows 5.2× better orbital consistency**
   - Metrics: Reconstruction loss, Orbital consistency, Quality

### 12. **Eredmények: GOLC-VAE hatása** (Results: GOLC-VAE Impact)
   - Key findings:
     1. Transposition invariance learned successfully
     2. Better latent space structure
     3. Improved generation quality
     4. Better generalization

### 13. **Technikai kihívások** (Technical Challenges)
   - 4 main challenges and their solutions
   - MIDI preprocessing, GPU memory, computational cost, hyperparameter tuning

### 14. **Gyakorlati alkalmazás** (Practical Application)
   - FastAPI backend
   - Use cases: Inspiration, Variations, Completion, Style learning
   - Web interface URL

### 15. **Konklúzió** (Conclusion)
   - Achieved results (with ✓ checkmarks)
   - Future directions
   - Quote: "ML doesn't replace but complements human creativity"

### 16. **Köszönetnyilvánítás** (Acknowledgments)
   - Thanks to advisor, university, family
   - Questions slide
   - GitHub link and contact

## Key Features Met

✅ **Started with "Why"** - Slide 2 explains your personal motivation (playing music, wanting a band)  
✅ **1-2 slides per chapter** - Each topic spans 1-2 slides max  
✅ **Original contribution highlighted** - GOLC-VAE is clearly marked as your innovation (slides 7-9)  
✅ **Small code snippets** - Only one ~10 line code example showing the key orbital loss computation  
✅ **13-15 minute timing** - 15 content slides + title = ~13-15 min at 1 min/slide pace

## Compilation Instructions

### Using LuaLaTeX (Recommended):
```bash
cd /home/deginandor/Documents/Programming/Zha/docs/beamer
lualatex thesis_defence.tex
lualatex thesis_defence.tex  # Run twice for references
```

### Using XeLaTeX (Alternative):
```bash
xelatex thesis_defence.tex
xelatex thesis_defence.tex
```

## Customization Tips

1. **Update email**: Change `degi.nandor@example.com` to your real email on slide 16
2. **Adjust timing**: If running short, expand on results. If too long, combine slides 10-11
3. **Add visuals**: Replace image placeholders with actual figures from your thesis
4. **Practice**: Aim for ~50-60 seconds per slide for 13-15 minute total

## Technical Details

- **Fonts**: Uses Oswald (sans-serif) and Lato (roman) from existing font files
- **Aspect ratio**: 16:9 (modern presentation format)
- **Theme**: Uses custom CSTheme_RO theme already in the directory
- **Colors**: Blue theme with red alerts for important points
- **Mathematical notation**: Proper LaTeX math for equations

## Important Notes

1. The presentation uses `\today` for the date, which will show "2nd April, 2026"
2. Some images are referenced but may not exist - you may need to add:
   - `img/uploads/7.jpg` (for code/music visualization)
3. The presentation follows Hungarian academic presentation standards
4. All mathematical formulas are properly typeset in LaTeX

## Presentation Tips

1. **Slide 2** (Why): Speak from the heart - this is your personal story
2. **Slides 7-9** (GOLC-VAE): This is your thesis contribution - spend 3-4 minutes here
3. **Slide 11** (Results): Emphasize the 5.2× improvement clearly
4. **Slide 9** (Code): Don't read the code - explain the concept it shows
5. **Final slide**: Leave GitHub link visible during Q&A

## Files Generated

- `thesis_defence.tex` - Main LaTeX source
- `thesis_defence.pdf` - Compiled presentation (246 KB, 16 pages)
- `thesis_defence.aux`, `.log`, `.nav`, `.out`, `.snm`, `.toc` - LaTeX auxiliary files

Good luck with your defence! 🎓🎵
