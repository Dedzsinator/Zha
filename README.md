# Zha

Music synthesis with neural network

## Introduction

Zha is a neural network-based music synthesis project that allows you to generate music using machine learning models. This project combines the power of deep learning with music theory to create unique musical compositions.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- CUDA-compatible GPU (recommended for faster training, but not required)

## Getting Started

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/Zha.git
   cd Zha
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - On Windows:

     ```bash
     venv\Scripts\activate
     ```

   - On macOS/Linux:

     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

### Dataset Preparation

1. **Prepare your music dataset**
   - Place your MIDI or audio files in the `data/raw` directory
   - Supported formats: MIDI (.mid, .midi), WAV, MP3

2. **Preprocess the dataset**

   ```bash
   python scripts/preprocess.py --input data/raw --output data/processed
   ```

## Training the Model

1. **Basic training**

   ```bash
   python train.py --data data/processed --epochs 100
   ```

2. **Advanced options**

   ```bash
   python train.py --data data/processed --epochs 100 --batch-size 64 --learning-rate 0.001 --model-type lstm
   ```

## Generating Music

1. **Generate music using a trained model**

   ```bash
   python generate.py --model models/model_latest.pt --output output/my_music.mid --length 60
   ```

2. **Interactive generation**

   ```bash
   python interactive.py --model models/model_latest.pt
   ```

## Project Structure

```bash
Zha/
│
├── data/               # Data directory
│   ├── raw/            # Raw music files
│   └── processed/      # Processed datasets
│
├── models/             # Saved models
│
├── scripts/            # Utility scripts
│   └── preprocess.py   # Data preprocessing
│
├── src/                # Source code
│   ├── models/         # Neural network architectures
│   ├── utils/          # Utility functions
│   └── visualization/  # Visualization tools
│
├── output/             # Generated music output
│
├── tests/              # Unit tests
│
├── train.py            # Training script
├── generate.py         # Music generation script
├── interactive.py      # Interactive generation interface
├── requirements.txt    # Project dependencies
└── README.md           # This file
```

## Examples

### Example 1: Training on Bach compositions

```bash
python train.py --data data/processed/bach --epochs 200 --model-type transformer
```

### Example 2: Generating jazz-style music

```bash
python generate.py --model models/jazz_model.pt --style jazz --temperature 0.8 --output output/jazz_piece.mid
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Decrease batch size: `python train.py --batch-size 16`
   - Use gradient accumulation: `python train.py --gradient-accumulation 4`

2. **Poor quality generated music**
   - Try different temperatures: `python generate.py --temperature 0.7`
   - Train the model for more epochs
   - Ensure your dataset is properly preprocessed

## Contributing

Contributions to Zha are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
