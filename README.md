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
   - Place your MIDI files in the `dataset/midi` directory
   - Supported formats: MIDI (.mid, .midi)

2. **Preprocess the dataset**

   ```bash
   python scripts/preprocess_dataset.py
   ```

## Training the Model

1. **Basic training**

   ```bash
   python train.py
   ```

2. **Train individual models**

   ```bash
   # Train Markov chain model
   python backend/trainers/train_markov.py
   
   # Train VAE model
   python backend/trainers/train_vae.py
   
   # Train Transformer model
   python backend/trainers/train_transformer.py
   
   # Train Diffusion model
   python backend/trainers/train_diffusion.py
   ```

3. **Advanced training options**

   Available models: markov, vae, golc_vae, transformer, diffusion
   Use `--help` with any training script for full options.

## Generating Music

1. **Start the FastAPI server**

   ```bash
   python backend/app.py
   ```

2. **Generate music via API**

   Use the `/generate_combined` endpoint with a MIDI file upload and parameters:
   - `creativity`: Controls VAE sampling randomness (0.0-1.0)
   - `duration`: Length of generated music in seconds
   - `instrument`: Target instrument (piano, guitar, etc.)

   Example API call:
   ```python
   import requests

   files = {'midi_file': open('input.mid', 'rb')}
   data = {'creativity': 0.5, 'duration': 30, 'instrument': 'piano'}
   response = requests.post('http://localhost:8000/generate_combined', files=files, data=data)
   ```

3. **Web Interface**

   Open `http://localhost:8000/docs` in your browser for the interactive API documentation.

## Project Structure

```bash
Zha/
│
├── backend/            # FastAPI backend
│   ├── app.py          # Main API server
│   ├── models/         # Neural network architectures
│   └── trainers/       # Training scripts
│
├── dataset/            # Data directory
│   ├── midi/           # Raw MIDI files
│   └── processed/      # Processed datasets
│
├── frontend/           # Next.js frontend (optional)
│
├── output/             # Generated music and checkpoints
│   ├── trained_models/ # Saved model checkpoints
│   ├── checkpoints/    # Training checkpoints
│   └── generated/      # Generated music files
│
├── scripts/            # Utility and analysis scripts
│   ├── preprocess_dataset.py
│   ├── generate_all_metrics.py
│   └── compare_vae_models.py
│
├── docs/               # Documentation and thesis
│
├── docker-compose.yml  # Docker setup
├── Dockerfile         # Container configuration
└── requirements.txt   # Python dependencies
```

## Examples

### Example 1: Training on MIDI dataset

```bash
python train.py
```

### Example 2: Generating music via API

Start the FastAPI server and use the `/generate_combined` endpoint:

```bash
python backend/app.py
```

Then make API requests as shown in the "Generating Music" section above.

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Decrease batch size: `python train.py --batch-size 16`
   - Use gradient accumulation: `python train.py --gradient-accumulation 4`

2. **Poor quality generated music**
   - Adjust creativity parameter in API calls (0.0-1.0)
   - Try different model combinations
   - Ensure your dataset is properly preprocessed

## Contributing

Contributions to Zha are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
