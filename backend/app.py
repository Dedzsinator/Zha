# Add these imports at the top
import os
import uuid
import shutil
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Response
import torch
import os
import tempfile
import numpy as np
from pydantic import BaseModel
from backend.models.diffusion import DiffusionModel
from backend.models.transformer import TransformerModel
from backend.models.vae import VAEModel
from backend.models.markov_chain import MarkovChain
from backend.util.midi_utils import parse_midi, create_midi_file
from backend.util.audio_utils import generate_audio
from typing import List, Optional, Dict, Union

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Zha Music Generation API",
    description="API for music generation using various AI models"
)

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pretrained models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model paths
MODEL_DIR = os.environ.get("MODEL_DIR", "./trained_models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Add this after the CORS middleware setup

# Create persistent directories for generated files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATED_DIR = os.path.join(BASE_DIR, "generated_files")
MIDI_DIR = os.path.join(GENERATED_DIR, "midi")
AUDIO_DIR = os.path.join(GENERATED_DIR, "audio")

# Create directories if they don't exist
os.makedirs(MIDI_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

logger.info(f"Generated files will be stored in: {GENERATED_DIR}")
logger.info(f"MIDI files: {MIDI_DIR}")
logger.info(f"Audio files: {AUDIO_DIR}")

# Track which models are successfully loaded
models_available = {
    "transformer": False,
    "vae": False,
    "diffusion": False,
    "markov": False
}

# Initialize models
transformer_model = TransformerModel(
    input_dim=128,
    embed_dim=512,  # Increased from 256
    num_heads=8,
    num_layers=8,   # Increased from 6
    dim_feedforward=2048  # Increased from 1024
)
if os.path.exists(os.path.join(MODEL_DIR, "trained_transformer.pt")):
    try:
        # Load the state dictionary
        state_dict = torch.load(os.path.join(MODEL_DIR, "trained_transformer.pt"), map_location=device)

        # Check if this is from a compiled model (has _orig_mod prefix)
        if any(key.startswith('_orig_mod.') for key in state_dict):
            print("Detected compiled model state dict, removing _orig_mod prefix")
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('_orig_mod.'):
                    new_key = key[len('_orig_mod.'):]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict

        # Now load the cleaned state dict
        transformer_model.load_state_dict(state_dict)
        transformer_model.to(device)
        transformer_model.eval()
        models_available["transformer"] = True
        print("Transformer model loaded successfully")
    except Exception as e:
        print(f"Failed to load transformer model: {e}")

vae_model = VAEModel(input_dim=128, latent_dim=128)  # Increased from 64
if os.path.exists(os.path.join(MODEL_DIR, "trained_vae.pt")):
    try:
        state_dict = torch.load(os.path.join(MODEL_DIR, "trained_vae.pt"), map_location=device)

        # Check if this is from a compiled model (has _orig_mod prefix)
        if any(key.startswith('_orig_mod.') for key in state_dict):
            print("Detected compiled model state dict, removing _orig_mod prefix")
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('_orig_mod.'):
                    new_key = key[len('_orig_mod.'):]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict

        vae_model.load_state_dict(state_dict)
        vae_model.to(device)
        vae_model.eval()
        models_available["vae"] = True
        print("VAE model loaded successfully")
    except Exception as e:
        print(f"Failed to load VAE model: {e}")

# Load diffusion model
diffusion_model = None
if os.path.exists(os.path.join(MODEL_DIR, "trained_diffusion.pt")):
    try:
        # First try loading with architecture from training script
        from backend.trainers.train_diffusion import DiffusionModel as TrainDiffusionModel
        diffusion_model = TrainDiffusionModel()

        # Load the state dictionary with prefix handling
        state_dict = torch.load(os.path.join(MODEL_DIR, "trained_diffusion.pt"), map_location=device)

        # Remove _orig_mod prefix if present
        if any(key.startswith('_orig_mod.') for key in state_dict):
            print("Detected compiled diffusion model, removing _orig_mod prefix")
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('_orig_mod.'):
                    new_key = key[len('_orig_mod.'):]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict

        diffusion_model.load_state_dict(state_dict)
        diffusion_model.to(device)
        diffusion_model.eval()
        models_available["diffusion"] = True
        print("Diffusion model loaded successfully")
    except Exception as e:
        print(f"Error loading diffusion model: {e}")
        print("Will initialize a new diffusion model")
        diffusion_model = DiffusionModel()
        diffusion_model.to(device)

markov_model = MarkovChain()
if os.path.exists(os.path.join(MODEL_DIR, "trained_markov.npy")):
    try:
        markov_model.load(os.path.join(MODEL_DIR, "trained_markov.npy"))
        models_available["markov"] = True
        print("Markov model loaded successfully")
    except Exception as e:
        print(f"Failed to load Markov model: {e}")

class GenerationResponse(BaseModel):
    midi_url: str
    audio_url: Optional[str] = None
    message: str

@app.post("/generate/combined", response_model=GenerationResponse)
async def generate_combined(
    midi_file: UploadFile = File(...),
    instrument: str = Form("piano"),
    creativity: float = Form(0.5),
    duration: int = Form(30),
    should_generate_audio: bool = Form(True)  # Renamed from generate_audio
):
    """
    Generate music using all models combined:
    - Markov Chain for music theory structure
    - VAE for creative variations
    - Transformer for sequence structure
    - Diffusion for final synthesis
    """
    # Check if all required models are available
    required_models = ["transformer", "vae", "markov"]
    missing_models = [model for model in required_models if not models_available[model]]

    if missing_models:
        raise HTTPException(
            status_code=400,
            detail=f"Combined generation requires {', '.join(missing_models)} " +
                  f"models which are not available. Please train these models first."
        )

    try:
        # Save uploaded MIDI file temporarily
        temp_midi_path = tempfile.mktemp(suffix=".mid")
        with open(temp_midi_path, "wb") as temp_file:
            temp_file.write(await midi_file.read())

        # Parse MIDI to feature vector
        midi_features = parse_midi(temp_midi_path)
        if midi_features is None:
            raise HTTPException(status_code=400, detail="Could not parse MIDI file")

        # Convert to tensor
        midi_tensor = torch.from_numpy(midi_features).float().to(device)

        # 1. Use Markov model to generate music theory compliant sequence
        markov_sequence = markov_model.generate_sequence(
            start_note=np.argmax(midi_features) if np.max(midi_features) > 0 else 60,
            length=64
        )

        # Convert Markov sequence to feature vector
        markov_feature = np.zeros(128, dtype=np.float32)
        for note in markov_sequence:
            if 0 <= note < 128:
                markov_feature[note] += 1
        if np.sum(markov_feature) > 0:
            markov_feature = markov_feature / np.sum(markov_feature)
        markov_tensor = torch.from_numpy(markov_feature).float().to(device)

        # 2. Use VAE for creative variations
        with torch.no_grad():
            recon, mu, logvar = vae_model(midi_tensor.unsqueeze(0))

            # Sample from latent space with creativity factor
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std) * creativity
            z = mu + eps * std

            # Decode (using the decoder part of the VAE)
            vae_output = vae_model.decoder(z).squeeze(0)

        # 3. Use Transformer for structure
        with torch.no_grad():
            transformer_output = transformer_model(midi_tensor.unsqueeze(0)).squeeze(0)

        # 4. Combine outputs with weights
        combined = (
            markov_tensor * 0.3 +      # Music theory structure: 30%
            vae_output * 0.4 +         # Creative variations: 40%
            transformer_output * 0.3   # Sequence structure: 30%
        )

        # Ensure all values are non-negative (clip negative values to 0)
        combined = torch.clamp(combined, min=0.0)

        # Renormalize to ensure values sum to 1
        sum_val = combined.sum()
        if sum_val > 0:
            combined = combined / sum_val

        # Additional logging to debug issues
        logger.info(f"Combined tensor stats - Min: {combined.min().item():.6f}, Max: {combined.max().item():.6f}, Sum: {combined.sum().item():.6f}")

        # Create output MIDI file
        output_midi_path = tempfile.mktemp(suffix=".mid")
        create_midi_file(combined.cpu().numpy(), output_midi_path, duration_seconds=duration)

        # Generate unique ID and save to persistent location
        unique_id = str(uuid.uuid4())[:8]
        midi_filename = f"combined_{unique_id}.mid"
        midi_dest = os.path.join(MIDI_DIR, midi_filename)

        # Ensure temp file exists before copying
        if not os.path.exists(output_midi_path):
            raise Exception(f"Failed to create MIDI file at: {output_midi_path}")

        # Copy with error handling
        try:
            shutil.copy2(output_midi_path, midi_dest)
        except Exception as copy_err:
            logger.error(f"Failed to copy MIDI file: {copy_err}")
            raise Exception(f"Failed to save MIDI file: {copy_err}")

        # Generate audio only if requested
        audio_url = None
        if should_generate_audio:
            audio_filename = f"combined_{unique_id}.wav"
            output_audio_path = tempfile.mktemp(suffix=".wav")

            if generate_audio(output_midi_path, output_audio_path, instrument=instrument):
                audio_dest = os.path.join(AUDIO_DIR, audio_filename)

                # Ensure audio file exists before copying
                if os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 0:
                    try:
                        shutil.copy2(output_audio_path, audio_dest)
                        audio_url = f"/download/audio/{audio_filename}"
                    except Exception as audio_err:
                        logger.error(f"Failed to save audio file: {audio_err}")
                else:
                    logger.warning(f"Audio file not created or empty: {output_audio_path}")
            else:
                logger.warning("Audio generation failed")

        return {
            "midi_url": f"/download/midi/{midi_filename}",
            "audio_url": audio_url,
            "message": "Successfully generated music using combined models"
        }

    except Exception as e:
        logger.exception(f"Combined generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/generate/vae", response_model=GenerationResponse)
async def generate_vae(
    midi_file: UploadFile = File(...),
    creativity: float = Form(0.5),
    duration: int = Form(30),
    instrument: str = Form("piano"),
    generate_audio: bool = Form(True)  # Add parameter for conditional audio generation
):
    """
    Generate music using only the VAE model for creative variations.
    """
    if not models_available["vae"]:
        raise HTTPException(
            status_code=400,
            detail="VAE model is not available. Please train the model first."
        )

    try:
        # Save uploaded MIDI file temporarily
        temp_midi_path = tempfile.mktemp(suffix=".mid")
        with open(temp_midi_path, "wb") as temp_file:
            temp_file.write(await midi_file.read())

        # Parse MIDI to feature vector
        midi_features = parse_midi(temp_midi_path)
        if midi_features is None:
            raise HTTPException(status_code=400, detail="Could not parse MIDI file")

        # Convert to tensor
        midi_tensor = torch.from_numpy(midi_features).float().to(device)

        # Use VAE for generation
        with torch.no_grad():
            recon, mu, logvar = vae_model(midi_tensor.unsqueeze(0))

            # Sample from latent space with creativity factor
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std) * creativity
            z = mu + eps * std

            # Decode
            vae_output = vae_model.decoder(z).squeeze(0)

        # Create output MIDI file
        output_midi_path = tempfile.mktemp(suffix=".mid")
        create_midi_file(vae_output.cpu().numpy(), output_midi_path, duration_seconds=duration)

        # Generate unique ID for file names
        unique_id = str(uuid.uuid4())[:8]
        midi_filename = f"vae_{unique_id}.mid"
        midi_dest = os.path.join(MIDI_DIR, midi_filename)

        # Ensure temp file exists before copying
        if not os.path.exists(output_midi_path):
            raise Exception(f"Failed to create MIDI file at: {output_midi_path}")

        # Copy with error handling
        try:
            shutil.copy2(output_midi_path, midi_dest)
        except Exception as copy_err:
            logger.error(f"Failed to copy MIDI file: {copy_err}")
            raise Exception(f"Failed to save MIDI file: {copy_err}")

        # Generate audio only if requested
        audio_url = None
        if generate_audio:
            audio_filename = f"vae_{unique_id}.wav"
            output_audio_path = tempfile.mktemp(suffix=".wav")

            if generate_audio(output_midi_path, output_audio_path, instrument=instrument):
                audio_dest = os.path.join(AUDIO_DIR, audio_filename)

                # Ensure audio file exists before copying
                if os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 0:
                    try:
                        shutil.copy2(output_audio_path, audio_dest)
                        audio_url = f"/download/audio/{audio_filename}"
                    except Exception as audio_err:
                        logger.error(f"Failed to save audio file: {audio_err}")
                else:
                    logger.warning(f"Audio file not created or empty: {output_audio_path}")
            else:
                logger.warning("Audio generation failed")

        return {
            "midi_url": f"/download/midi/{midi_filename}",
            "audio_url": audio_url,
            "message": "Successfully generated music using VAE model"
        }

    except Exception as e:
        logger.exception(f"VAE generation error: {e}")
        raise HTTPException(status_code=500, detail=f"VAE generation failed: {str(e)}")

@app.post("/generate/structured_transformer", response_model=GenerationResponse)
async def generate_structured_transformer(
    midi_file: UploadFile = File(...),
    num_sections: int = Form(4),
    section_length: int = Form(16),
    transition_smoothness: float = Form(0.7),
    temperature: float = Form(0.8),
    duration: int = Form(60),
    instrument: str = Form("piano"),
    should_generate_audio: bool = Form(True)
):
    """
    Generate music with distinct sections (verse, chorus, etc.) using the Transformer model.
    """
    if not models_available["transformer"]:
        raise HTTPException(
            status_code=400,
            detail="Transformer model is not available. Please train the model first."
        )

    try:
        # Save uploaded MIDI file temporarily
        temp_midi_path = tempfile.mktemp(suffix=".mid")
        with open(temp_midi_path, "wb") as temp_file:
            temp_file.write(await midi_file.read())

        # Parse MIDI to feature vector
        midi_features = parse_midi(temp_midi_path)
        if midi_features is None:
            raise HTTPException(status_code=400, detail="Could not parse MIDI file")

        # Convert to tensor
        midi_tensor = torch.from_numpy(midi_features).float().to(device)

        # Reset any previous memory state
        transformer_model.reset_memory()

        with torch.no_grad():
            # Use the new structured generation method
            structured_output = transformer_model.generate_with_structure(
                seed=midi_tensor.unsqueeze(0),
                num_sections=num_sections,
                section_length=section_length,
                temperature=temperature,
                transition_smoothness=transition_smoothness
            )

            # Log the shape to understand what we're getting
            logger.info(f"Raw structured output shape: {structured_output.shape}")

            # Process structured output into a feature vector
            # If we get a multi-dimensional output [batch, sequence, features]
            if len(structured_output.shape) == 3:
                # Average across all sections to create a single feature vector
                transformer_output = structured_output.mean(dim=1)
            else:
                transformer_output = structured_output.squeeze(0)

            # Make sure we have the right shape [batch, features]
            if len(transformer_output.shape) == 1:
                # If we got a single vector, add batch dimension
                transformer_output = transformer_output.unsqueeze(0)

            # Final consistency check - we want a 2D tensor [batch, features]
            if len(transformer_output.shape) != 2:
                logger.error(f"Unexpected tensor shape: {transformer_output.shape}")
                # Force reshape to correct dimensions if needed
                transformer_output = transformer_output.view(1, -1)
                if transformer_output.shape[1] != 128:
                    # If not the right feature size, create default
                    logger.warning("Creating default feature vector")
                    transformer_output = torch.zeros(1, 128, device=device)
                    transformer_output[0, 60] = 1.0  # Middle C

            # Final shape check to confirm it's [1, 128] as expected
            logger.info(f"Final output shape: {transformer_output.shape}")

        # Fix any invalid values
        if torch.isnan(transformer_output).any() or torch.isinf(transformer_output).any():
            transformer_output = torch.nan_to_num(transformer_output, nan=0.0, posinf=1.0, neginf=0.0)
            logger.warning("Fixed NaN or inf values in transformer output")

        # Ensure all values are non-negative
        transformer_output = torch.clamp(transformer_output, min=0.0)

        # Renormalize to ensure values sum to 1
        sum_val = transformer_output.sum()
        if sum_val > 0:
            transformer_output = transformer_output / sum_val
        else:
            logger.warning("Sum of transformer output is 0, using default values")
            transformer_output = torch.zeros_like(transformer_output)
            transformer_output[0, 60] = 1.0  # Set middle C to 1.0

        # Create output MIDI file - ensure we pass just the features without batch dimension
        output_midi_path = tempfile.mktemp(suffix=".mid")
        create_midi_file(transformer_output[0].cpu().numpy(), output_midi_path, duration_seconds=duration)

        unique_id = str(uuid.uuid4())[:8]
        midi_filename = f"structured_{unique_id}.mid"
        midi_dest = os.path.join(MIDI_DIR, midi_filename)

        # Ensure temp file exists before copying
        if not os.path.exists(output_midi_path):
            raise Exception(f"Failed to create MIDI file at: {output_midi_path}")

        # Copy with error handling
        try:
            shutil.copy2(output_midi_path, midi_dest)
        except Exception as copy_err:
            logger.error(f"Failed to copy MIDI file: {copy_err}")
            raise Exception(f"Failed to save MIDI file: {copy_err}")

        # Check that the output file was actually created
        if not os.path.exists(output_midi_path):
            logger.error(f"MIDI file creation failed, output path doesn't exist: {output_midi_path}")
            raise Exception(f"Failed to create MIDI file at: {output_midi_path}")

        if os.path.getsize(output_midi_path) == 0:
            logger.error(f"MIDI file was created but is empty: {output_midi_path}")
            raise Exception("Created MIDI file is empty")

        # Generate unique ID for file names
        unique_id = str(uuid.uuid4())[:8]
        midi_filename = f"structured_{unique_id}.mid"
        midi_dest = os.path.join(MIDI_DIR, midi_filename)

        # Copy with error handling
        try:
            shutil.copy2(output_midi_path, midi_dest)
        except Exception as copy_err:
            logger.error(f"Failed to copy MIDI file: {copy_err}")
            raise Exception(f"Failed to save MIDI file: {copy_err}")

        # Generate audio only if requested
        audio_url = None
        if should_generate_audio:
            audio_filename = f"structured_{unique_id}.wav"
            output_audio_path = tempfile.mktemp(suffix=".wav")

            if generate_audio(output_midi_path, output_audio_path, instrument=instrument):
                audio_dest = os.path.join(AUDIO_DIR, audio_filename)

                # Ensure audio file exists before copying
                if os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 0:
                    try:
                        shutil.copy2(output_audio_path, audio_dest)
                        audio_url = f"/download/audio/{audio_filename}"
                    except Exception as audio_err:
                        logger.error(f"Failed to save audio file: {audio_err}")
                else:
                    logger.warning(f"Audio file not created or empty: {output_audio_path}")
            else:
                logger.warning("Audio generation failed")

        return {
            "midi_url": f"/download/midi/{midi_filename}",
            "audio_url": audio_url,
            "message": "Successfully generated structured music with Transformer model"
        }

    except Exception as e:
        logger.exception(f"Structured transformer generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Structured generation failed: {str(e)}")


@app.post("/generate/musical_markov", response_model=GenerationResponse)
async def generate_musical_markov(
    start_note: int = Form(60),
    key_context: str = Form("C major"),
    length: int = Form(64),
    duration: int = Form(30),
    time_signature: str = Form("4/4"),  # Added time signature parameter
    instrument: str = Form("piano"),
    should_generate_audio: bool = Form(True)
):
    """
    Generate music using the enhanced Markov Chain with music theory awareness.

    Args:
        start_note: Starting note for generation
        key_context: Musical key for generation (e.g., "C major", "A minor")
        length: Number of notes to generate
        duration: Duration in seconds of the output MIDI
        time_signature: Time signature to use (e.g. "4/4", "3/4", "6/8")
        instrument: Instrument sound to use for audio generation
        should_generate_audio: Whether to generate audio alongside the MIDI
    """
    if not models_available["markov"]:
        raise HTTPException(
            status_code=400,
            detail="Markov model is not available. Please train the model first."
        )

    try:
        # Generate with chords using the enhanced Markov model
        musical_output = markov_model.generate_with_chords(
            key_context=key_context,
            length=length,
            time_signature=time_signature  # Add time signature parameter
        )

        # Log the musical features that were generated
        logger.info(f"Generated in key: {musical_output.get('key', 'unknown')}")
        logger.info(f"Time signature: {musical_output.get('time_signature', '4/4')}")
        logger.info(f"Chords: {musical_output.get('chords', [])[:8]}")  # Show first 8 chords
        logger.info(f"Notes: {musical_output.get('notes', [])[:16]}")  # Show first 16 notes
        
        if 'durations' in musical_output:
            logger.info(f"Durations: {musical_output.get('durations', [])[:16]}")  # Show durations

        # Convert generated notes to feature vector
        feature = np.zeros(128, dtype=np.float32)
        for note in musical_output.get('notes', []):
            if 0 <= note < 128:
                feature[note] += 1

        # Normalize
        if np.sum(feature) > 0:
            feature = feature / np.sum(feature)

        # Create output MIDI file with enhanced features
        output_midi_path = tempfile.mktemp(suffix=".mid")
        
        # Check if we have durations to use
        if 'durations' in musical_output and len(musical_output['durations']) > 0:
            # Create MIDI with specific note durations
            from backend.util.midi_utils import create_midi_with_durations
            create_midi_with_durations(
                musical_output['notes'], 
                musical_output['durations'], 
                output_midi_path,
                time_signature=musical_output.get('time_signature', '4/4')
            )
        else:
            # Fallback to standard MIDI creation
            create_midi_file(feature, output_midi_path, duration_seconds=duration)

        # Rest of the function remains the same...
        # Generate unique ID and save to persistent location
        unique_id = str(uuid.uuid4())[:8]
        midi_filename = f"musical_markov_{unique_id}.mid"
        midi_dest = os.path.join(MIDI_DIR, midi_filename)

        # Ensure temp file exists before copying
        if not os.path.exists(output_midi_path):
            raise Exception(f"Failed to create MIDI file at: {output_midi_path}")

        # Create parent directory if needed
        os.makedirs(os.path.dirname(midi_dest), exist_ok=True)

        # Copy with error handling
        try:
            shutil.copy2(output_midi_path, midi_dest)
        except Exception as copy_err:
            logger.error(f"Failed to copy MIDI file: {copy_err}")
            raise Exception(f"Failed to save MIDI file: {copy_err}")

        # Generate audio only if requested
        audio_url = None
        if should_generate_audio:
            audio_filename = f"musical_markov_{unique_id}.wav"
            output_audio_path = tempfile.mktemp(suffix=".wav")

            if generate_audio(output_midi_path, output_audio_path, instrument=instrument):
                audio_dest = os.path.join(AUDIO_DIR, audio_filename)

                # Ensure temp audio file exists before copying
                if os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 0:
                    try:
                        shutil.copy2(output_audio_path, audio_dest)
                        audio_url = f"/download/audio/{audio_filename}"
                    except Exception as audio_err:
                        logger.error(f"Failed to save audio file: {audio_err}")
                else:
                    logger.warning(f"Audio file not created or empty: {output_audio_path}")
            else:
                logger.warning("Audio generation failed")

        return {
            "midi_url": f"/download/midi/{midi_filename}",
            "audio_url": audio_url,
            "message": f"Successfully generated music in {key_context} with {time_signature} time signature"
        }

    except Exception as e:
        logger.exception(f"Musical Markov generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Musical Markov generation failed: {str(e)}")

@app.post("/generate/expressive_markov", response_model=GenerationResponse)
async def generate_expressive_markov(
    key_context: str = Form("A minor"),
    length: int = Form(96),
    complexity: float = Form(0.8),
    duration: int = Form(60),
    instrument: str = Form("piano"),
    should_generate_audio: bool = Form(True)
):
    """
    Generate music using the enhanced Markov Chain with expressive rhythmic patterns.

    Args:
        key_context: Musical key for generation (e.g., "C major", "A minor")
        length: Target number of notes to generate
        complexity: Controls rhythmic complexity (0.0-1.0)
        duration: Duration in seconds of the output MIDI
        instrument: Instrument sound to use for audio generation
        should_generate_audio: Whether to generate audio alongside the MIDI
    """
    if not models_available["markov"]:
        raise HTTPException(
            status_code=400,
            detail="Markov model is not available. Please train the model first."
        )

    try:
        # Generate expressive output
        complex_output = markov_model.generate_expressive_sequence(
            key_context=key_context,
            length=length,
            complexity=complexity
        )

        # Log the musical features that were generated
        logger.info(f"Generated expressive sequence in key: {complex_output.get('key', 'unknown')}")
        logger.info(f"Time signature: {complex_output.get('time_signature', '4/4')}")
        logger.info(f"Generated {len(complex_output.get('notes', []))} notes")
        logger.info(f"First 8 notes: {complex_output.get('notes', [])[:8]}")
        logger.info(f"First 8 durations: {complex_output.get('durations', [])[:8]}")

        # Create output MIDI file with durations
        output_midi_path = tempfile.mktemp(suffix=".mid")
        
        # Use enhanced MIDI creation with durations
        from backend.util.midi_utils import create_midi_with_durations
        
        # Ensure we have valid data
        notes = complex_output.get('notes', [])
        durations = complex_output.get('durations', [])
        if not notes or len(notes) == 0:
            raise Exception("No notes were generated")
        
        # Default durations if none provided
        if not durations or len(durations) == 0:
            durations = [0.5] * len(notes)  # Default to eighth notes
        
        # Ensure lengths match
        min_length = min(len(notes), len(durations))
        notes = notes[:min_length]
        durations = durations[:min_length]
        
        # Create the MIDI file
        create_midi_with_durations(
            notes,
            durations,
            output_midi_path,
            time_signature=complex_output.get('time_signature', '4/4')
        )

        # Generate unique ID and save to persistent location
        unique_id = str(uuid.uuid4())[:8]
        midi_filename = f"expressive_{unique_id}.mid"
        midi_dest = os.path.join(MIDI_DIR, midi_filename)

        # Ensure temp file exists before copying
        if not os.path.exists(output_midi_path):
            raise Exception(f"Failed to create expressive MIDI file at: {output_midi_path}")

        # Create parent directory if needed
        os.makedirs(os.path.dirname(midi_dest), exist_ok=True)

        # Copy with error handling
        try:
            shutil.copy2(output_midi_path, midi_dest)
        except Exception as copy_err:
            logger.error(f"Failed to copy MIDI file: {copy_err}")
            raise Exception(f"Failed to save MIDI file: {copy_err}")

        # Generate audio only if requested
        audio_url = None
        if should_generate_audio:
            audio_filename = f"expressive_{unique_id}.wav"
            output_audio_path = tempfile.mktemp(suffix=".wav")

            if generate_audio(output_midi_path, output_audio_path, instrument=instrument):
                audio_dest = os.path.join(AUDIO_DIR, audio_filename)

                # Ensure temp audio file exists before copying
                if os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 0:
                    try:
                        shutil.copy2(output_audio_path, audio_dest)
                        audio_url = f"/download/audio/{audio_filename}"
                    except Exception as audio_err:
                        logger.error(f"Failed to save audio file: {audio_err}")
                else:
                    logger.warning(f"Audio file not created or empty: {output_audio_path}")
            else:
                logger.warning("Audio generation failed")

        # Create informative message
        complexity_desc = "high" if complexity > 0.7 else "medium" if complexity > 0.4 else "low"
        
        return {
            "midi_url": f"/download/midi/{midi_filename}",
            "audio_url": audio_url,
            "message": f"Successfully generated expressive music in {key_context} with {complexity_desc} complexity"
        }

    except Exception as e:
        logger.exception(f"Expressive Markov generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Expressive Markov generation failed: {str(e)}")
    
def create_midi_with_durations(notes, durations, output_path, time_signature="4/4"):
    """
    Create a MIDI file with specific note durations
    
    Args:
        notes: List of MIDI note numbers (0-127)
        durations: List of durations in quarter notes (0.5 = eighth note, 1.0 = quarter, etc.)
        output_path: Path to save the MIDI file
        time_signature: Time signature string (e.g. "4/4", "3/4")
    """
    from midiutil.MidiFile import MIDIFile
    
    # Create MIDI file with 1 track
    mf = MIDIFile(1)
    
    # Set track information
    track = 0
    channel = 0
    time = 0  # Start at beginning
    tempo = 120  # BPM
    
    mf.addTrackName(track, time, "Markov Chain Generated Track")
    mf.addTempo(track, time, tempo)
    
    # Parse time signature
    try:
        numerator, denominator = map(int, time_signature.split('/'))
        # Set time signature (numerator, denominator are encoded as bit shifts)
        denominator_bits = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4, 32: 5}.get(denominator, 2)
        mf.addTimeSignature(track, time, numerator, denominator_bits, 24, 8)
    except:
        # Default to 4/4 if parsing fails
        mf.addTimeSignature(track, time, 4, 2, 24, 8)
    
    # Add notes with their durations
    for i, (note_num, duration) in enumerate(zip(notes, durations)):
        if 0 <= note_num < 128:
            velocity = 100  # Default velocity
            
            # Add a bit of velocity variation
            if i % 4 == 0:
                velocity = 112  # Emphasize first beat
            elif i % 2 == 0:
                velocity = 96
            
            # Add the note
            mf.addNote(track, channel, note_num, time, duration, velocity)
            
            # Increment time
            time += duration
    
    # Write the MIDI file
    with open(output_path, 'wb') as outf:
        mf.writeFile(outf)
        
    return True

@app.post("/generate/transformer", response_model=GenerationResponse)
async def generate_transformer(
    midi_file: UploadFile = File(...),
    temperature: float = Form(0.8),  # Changed default from 1.0 to 0.8
    steps: int = Form(100),  # Added steps parameter
    top_k: int = Form(5),    # Added top_k parameter for better sampling
    top_p: float = Form(0.92), # Added nucleus sampling parameter
    duration: int = Form(30),
    instrument: str = Form("piano"),
    should_generate_audio: bool = Form(True)  # Changed from generate_audio
):
    """
    Generate music using only the Transformer model for sequence structure.
    """
    if not models_available["transformer"]:
        raise HTTPException(
            status_code=400,
            detail="Transformer model is not available. Please train the model first."
        )

    try:
        # Save uploaded MIDI file temporarily
        temp_midi_path = tempfile.mktemp(suffix=".mid")
        with open(temp_midi_path, "wb") as temp_file:
            temp_file.write(await midi_file.read())

        # Parse MIDI to feature vector
        midi_features = parse_midi(temp_midi_path)
        if midi_features is None:
            raise HTTPException(status_code=400, detail="Could not parse MIDI file")

        # Convert to tensor
        midi_tensor = torch.from_numpy(midi_features).float().to(device)

        # Reset memory for fresh generation
        transformer_model.reset_memory()

        # Generate using enhanced transformer with better sampling
        with torch.no_grad():
            transformer_output = transformer_model.generate(
                seed=midi_tensor.unsqueeze(0),
                steps=steps,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

            # Handle different output shapes
            if len(transformer_output.shape) == 3:
                transformer_output = transformer_output[:, -1]  # Take the last step

            transformer_output = transformer_output.squeeze(0)

            # Ensure non-negative outputs and normalize
            transformer_output = torch.clamp(transformer_output, min=0.0)
            sum_val = transformer_output.sum()
            if sum_val > 0:
                transformer_output = transformer_output / sum_val

        # Create output MIDI file
        output_midi_path = tempfile.mktemp(suffix=".mid")
        create_midi_file(transformer_output.cpu().numpy(), output_midi_path, duration_seconds=duration)

        # Generate unique ID for file names
        unique_id = str(uuid.uuid4())[:8]
        midi_filename = f"transformer_{unique_id}.mid"
        midi_dest = os.path.join(MIDI_DIR, midi_filename)

        # Ensure temp file exists before copying
        if not os.path.exists(output_midi_path):
            raise Exception(f"Failed to create MIDI file at: {output_midi_path}")

        # Copy with error handling
        try:
            shutil.copy2(output_midi_path, midi_dest)
        except Exception as copy_err:
            logger.error(f"Failed to copy MIDI file: {copy_err}")
            raise Exception(f"Failed to save MIDI file: {copy_err}")

        # Generate audio only if requested
        audio_url = None
        if should_generate_audio:  # Changed from generate_audio
            audio_filename = f"transformer_{unique_id}.wav"
            output_audio_path = tempfile.mktemp(suffix=".wav")

            if generate_audio(output_midi_path, output_audio_path, instrument=instrument):
                audio_dest = os.path.join(AUDIO_DIR, audio_filename)

                # Ensure audio file exists before copying
                if os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 0:
                    try:
                        shutil.copy2(output_audio_path, audio_dest)
                        audio_url = f"/download/audio/{audio_filename}"
                    except Exception as audio_err:
                        logger.error(f"Failed to save audio file: {audio_err}")
                else:
                    logger.warning(f"Audio file not created or empty: {output_audio_path}")
            else:
                logger.warning("Audio generation failed")

        return {
            "midi_url": f"/download/midi/{midi_filename}",
            "audio_url": audio_url,
            "message": "Successfully generated music using enhanced Transformer model"
        }

    except Exception as e:
        logger.exception(f"Transformer generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Transformer generation failed: {str(e)}")

@app.post("/generate/markov", response_model=GenerationResponse)
async def generate_markov(
    start_note: int = Form(60),
    length: int = Form(64),
    key_context: Optional[str] = Form(None),  # Added optional key context
    duration: int = Form(30),
    instrument: str = Form("piano"),
    should_generate_audio: bool = Form(True)
):
    """
    Generate music using the Markov Chain model for music theory structure.
    """
    if not models_available["markov"]:
        raise HTTPException(
            status_code=400,
            detail="Markov model is not available. Please train the model first."
        )

    try:
        # Generate sequence using Markov model with optional key context
        markov_sequence = markov_model.generate_sequence(
            start_note=start_note,
            length=length,
            key_context=key_context  # Add key context
        )

        # Convert to feature vector
        feature = np.zeros(128, dtype=np.float32)
        for note in markov_sequence:
            if 0 <= note < 128:
                feature[note] += 1

        # Normalize
        if np.sum(feature) > 0:
            feature = feature / np.sum(feature)

        # Create output MIDI file (in temp first)
        output_midi_path = tempfile.mktemp(suffix=".mid")
        create_midi_file(feature, output_midi_path, duration_seconds=duration)

        # Generate unique ID and save to persistent location
        unique_id = str(uuid.uuid4())[:8]
        midi_filename = f"markov_{unique_id}.mid"
        midi_dest = os.path.join(MIDI_DIR, midi_filename)

        # Ensure temp file exists before copying
        if not os.path.exists(output_midi_path):
            raise Exception(f"Failed to create MIDI file at: {output_midi_path}")

        # Create parent directory if needed
        os.makedirs(os.path.dirname(midi_dest), exist_ok=True)

        # Copy with error handling
        try:
            shutil.copy2(output_midi_path, midi_dest)
        except Exception as copy_err:
            logger.error(f"Failed to copy MIDI file: {copy_err}")
            raise Exception(f"Failed to save MIDI file: {copy_err}")

        # Generate audio only if requested
        audio_url = None
        if should_generate_audio:
            audio_filename = f"markov_{unique_id}.wav"
            output_audio_path = tempfile.mktemp(suffix=".wav")

            from backend.util.audio_utils import generate_audio  # Import it locally to avoid confusion

            if generate_audio(output_midi_path, output_audio_path, instrument=instrument):
                audio_dest = os.path.join(AUDIO_DIR, audio_filename)

                # Ensure temp audio file exists before copying
                if os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 0:
                    try:
                        shutil.copy2(output_audio_path, audio_dest)
                        audio_url = f"/download/audio/{audio_filename}"
                    except Exception as audio_err:
                        logger.error(f"Failed to save audio file: {audio_err}")
                else:
                    logger.warning(f"Audio file not created or empty: {output_audio_path}")
            else:
                logger.warning("Audio generation failed")

        message = "Successfully generated music using Markov model"
        if key_context:
            message += f" in {key_context}"

        return {
            "midi_url": f"/download/midi/{midi_filename}",
            "audio_url": audio_url,
            "message": message
        }

    except Exception as e:
        logger.exception(f"Markov generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Markov generation failed: {str(e)}")

@app.post("/generate/diffusion", response_model=GenerationResponse)
async def generate_diffusion(
    midi_file: UploadFile = File(None),
    steps: int = Form(100),
    duration: int = Form(30),
    instrument: str = Form("piano"),
    generate_audio: bool = Form(True)  # Add parameter for conditional audio generation
):
    """
    Generate music using only the Diffusion model.
    If no MIDI file is provided, generates from random noise.
    """
    if not models_available["diffusion"]:
        raise HTTPException(
            status_code=400,
            detail="Diffusion model is not available. Please train the model first."
        )

    try:
        if midi_file:
            # Use provided MIDI as starting point
            temp_midi_path = tempfile.mktemp(suffix=".mid")
            with open(temp_midi_path, "wb") as temp_file:
                temp_file.write(await midi_file.read())

            # Parse MIDI to feature vector
            midi_features = parse_midi(temp_midi_path)
            if midi_features is None:
                raise HTTPException(status_code=400, detail="Could not parse MIDI file")

            # Convert to tensor and add noise
            x_T = torch.from_numpy(midi_features).float().to(device)
            x_T = x_T + torch.randn_like(x_T) * 0.5
        else:
            # Generate from random noise
            x_T = torch.randn(128).to(device)

        # Generate using diffusion model
        with torch.no_grad():
            diffusion_output = diffusion_model.sample(x_T.unsqueeze(0), steps=steps).squeeze(0)

        # Create output MIDI file
        output_midi_path = tempfile.mktemp(suffix=".mid")
        create_midi_file(diffusion_output.cpu().numpy(), output_midi_path, duration_seconds=duration)

        # Generate unique ID for file names
        unique_id = str(uuid.uuid4())[:8]
        midi_filename = f"diffusion_{unique_id}.mid"
        midi_dest = os.path.join(MIDI_DIR, midi_filename)

        # Ensure temp file exists before copying
        if not os.path.exists(output_midi_path):
            raise Exception(f"Failed to create MIDI file at: {output_midi_path}")

        # Copy with error handling
        try:
            shutil.copy2(output_midi_path, midi_dest)
        except Exception as copy_err:
            logger.error(f"Failed to copy MIDI file: {copy_err}")
            raise Exception(f"Failed to save MIDI file: {copy_err}")

        # Generate audio only if requested
        audio_url = None
        if generate_audio:
            audio_filename = f"diffusion_{unique_id}.wav"
            output_audio_path = tempfile.mktemp(suffix=".wav")

            if generate_audio(output_midi_path, output_audio_path, instrument=instrument):
                audio_dest = os.path.join(AUDIO_DIR, audio_filename)

                # Ensure audio file exists before copying
                if os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 0:
                    try:
                        shutil.copy2(output_audio_path, audio_dest)
                        audio_url = f"/download/audio/{audio_filename}"
                    except Exception as audio_err:
                        logger.error(f"Failed to save audio file: {audio_err}")
                else:
                    logger.warning(f"Audio file not created or empty: {output_audio_path}")
            else:
                logger.warning("Audio generation failed")

        return {
            "midi_url": f"/download/midi/{midi_filename}",
            "audio_url": audio_url,
            "message": "Successfully generated music using Diffusion model"
        }

    except Exception as e:
        logger.exception(f"Diffusion generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Diffusion generation failed: {str(e)}")
@app.get("/download/midi/{filename}")
async def download_midi(filename: str):
    """Download a generated MIDI file"""
    file_path = os.path.join(MIDI_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"MIDI file not found: {filename}")

    return FileResponse(
        path=file_path,
        media_type="audio/midi",
        filename="generated_music.mid"
    )

@app.get("/download/audio/{filename}")
async def download_audio(filename: str):
    """Download a generated audio file"""
    file_path = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Audio file not found: {filename}")

    return FileResponse(
        path=file_path,
        media_type="audio/wav",
        filename="generated_music.wav"
    )

@app.get("/debug/{filename}")
async def debug_file(filename: str):
    """Debug endpoint to check file info"""
    file_path = tempfile.gettempdir() + "/" + filename
    if not os.path.exists(file_path):
        return {"error": "File not found", "path": file_path}


    file_info = {
        "exists": True,
        "size_bytes": os.path.getsize(file_path),
        "path": file_path,
        "temp_dir": tempfile.gettempdir(),
        "first_10_bytes": ""
    }

    try:
        with open(file_path, "rb") as f:
            file_info["first_10_bytes"] = str(f.read(10))
    except Exception as e:
        file_info["error"] = str(e)

    return file_info

@app.get("/health")
async def health_check():
    """Check if the API and models are working properly"""
    return {
        "status": "healthy",
        "device": str(device),
        "models_loaded": models_available
    }

@app.get("/list_files")
async def list_files():
    """List all generated files"""
    midi_files = os.listdir(MIDI_DIR) if os.path.exists(MIDI_DIR) else []
    audio_files = os.listdir(AUDIO_DIR) if os.path.exists(AUDIO_DIR) else []

    return {
        "midi_dir": MIDI_DIR,
        "midi_files": midi_files,
        "audio_dir": AUDIO_DIR,
        "audio_files": audio_files,
        "temp_dir": tempfile.gettempdir()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)