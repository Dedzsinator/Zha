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


@app.post("/generate/markov", response_model=GenerationResponse)
async def generate_markov(
    start_note: int = Form(60),
    length: int = Form(64),
    duration: int = Form(30),
    instrument: str = Form("piano"),
    should_generate_audio: bool = Form(True)  # Renamed from generate_audio
):
    """
    Generate music using only the Markov Chain model for music theory structure.
    """
    if not models_available["markov"]:
        raise HTTPException(
            status_code=400,
            detail="Markov model is not available. Please train the model first."
        )

    try:
        # Generate sequence using Markov model
        markov_sequence = markov_model.generate_sequence(
            start_note=start_note,
            length=length
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
        if should_generate_audio:  # Changed parameter name here
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

        return {
            "midi_url": f"/download/midi/{midi_filename}",
            "audio_url": audio_url,
            "message": "Successfully generated music using Markov model"
        }

    except Exception as e:
        logger.exception(f"Markov generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Markov generation failed: {str(e)}")

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

@app.post("/generate/transformer", response_model=GenerationResponse)
async def generate_transformer(
    midi_file: UploadFile = File(...),
    temperature: float = Form(1.0),
    duration: int = Form(30),
    instrument: str = Form("piano"),
    generate_audio: bool = Form(True)  # Add parameter for conditional audio generation
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

        # Generate using transformer
        with torch.no_grad():
            transformer_output = transformer_model.generate(
                seed=midi_tensor.unsqueeze(0),
                temperature=temperature
            ).squeeze(0)

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
        if generate_audio:
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
            "message": "Successfully generated music using Transformer model"
        }

    except Exception as e:
        logger.exception(f"Transformer generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Transformer generation failed: {str(e)}")

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