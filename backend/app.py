import os
import uuid
import shutil
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import tempfile
import numpy as np
from pydantic import BaseModel
from backend.models.transformer import TransformerModel
from backend.models.vae import VAEModel
from backend.models.markov_chain import MarkovChain
from backend.util.midi_utils import parse_midi, create_midi_file, parse_key_context
from backend.util.audio_utils import generate_audio
from typing import List, Optional, Dict, Union

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Zha Music Generation API",
    description="API for music generation using various AI models"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
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
    "markov": False
}

# Initialize models
transformer_model = TransformerModel(
    input_dim=128,
    embed_dim=512,
    num_heads=8,
    num_layers=8,
    dim_feedforward=2048
)
if os.path.exists(os.path.join(MODEL_DIR, "trained_transformer.pt")):
    try:
        state_dict = torch.load(os.path.join(MODEL_DIR, "trained_transformer.pt"), map_location=device)
        # Handle compiled model prefixes if present
        if any(key.startswith('_orig_mod.') for key in state_dict):
            new_state_dict = {key[len('_orig_mod.'):]: value if key.startswith('_orig_mod.') else value 
                             for key, value in state_dict.items()}
            state_dict = new_state_dict
        transformer_model.load_state_dict(state_dict)
        transformer_model.to(device)
        transformer_model.eval()
        models_available["transformer"] = True
        print("Transformer model loaded successfully")
    except Exception as e:
        print(f"Failed to load transformer model: {e}")

vae_model = VAEModel(input_dim=128, latent_dim=128)
if os.path.exists(os.path.join(MODEL_DIR, "trained_vae.pt")):
    try:
        state_dict = torch.load(os.path.join(MODEL_DIR, "trained_vae.pt"), map_location=device)
        if any(key.startswith('_orig_mod.') for key in state_dict):
            new_state_dict = {key[len('_orig_mod.'):]: value if key.startswith('_orig_mod.') else value 
                             for key, value in state_dict.items()}
            state_dict = new_state_dict
        vae_model.load_state_dict(state_dict)
        vae_model.to(device)
        vae_model.eval()
        models_available["vae"] = True
        print("VAE model loaded successfully")
    except Exception as e:
        print(f"Failed to load VAE model: {e}")

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
    creativity: float = Form(0.5),
    duration: int = Form(30),
    instrument: str = Form("piano"),
    should_generate_audio: bool = Form(True)
):
    """
    Generate music using combined models with enhanced musicality:
    - Markov Chain for music theory and chord structure
    - VAE for creative variations within a musical scale
    - Transformer for coherent sequence structure
    """
    # Check if all required models are available
    required_models = ["transformer", "vae", "markov"]
    missing_models = [model for model in required_models if not models_available[model]]

    if missing_models:
        raise HTTPException(
            status_code=400,
            detail=f"Combined generation requires {', '.join(missing_models)} " +
                  f"models which are not available."
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
        
        # Analyze key and scale from input MIDI
        from music21 import converter, analysis, scale
        try:
            score = converter.parse(temp_midi_path)
            key_obj = analysis.discrete.analyzeStream(score, 'key')
            detected_key = str(key_obj.tonic.name) + " " + key_obj.mode
            scale_obj = scale.ConcreteScale(pitches=scale.DiatonicScale(detected_key).getPitches())
            scale_notes = set([p.midi % 12 for p in scale_obj.getPitches()])
            logger.info(f"Detected key: {detected_key}, Scale notes: {scale_notes}")
        except Exception as e:
            logger.warning(f"Error detecting key: {e}, using C major")
            detected_key = "C major"
            scale_notes = {0, 2, 4, 5, 7, 9, 11}  # C major

        # 1. Generate chord progression with Markov model
        chord_progression = markov_model.generate_chord_progression(
            key_context=detected_key,
            num_chords=8
        )
        logger.info(f"Generated chord progression: {chord_progression}")
        
        # 2. Generate base sequence with Markov model using chord awareness
        markov_sequence = markov_model.generate_with_chords(
            key_context=detected_key,
            length=64,
            time_signature="4/4"
        )
        
        # Extract notes and create feature vector
        markov_notes = markov_sequence.get('notes', [60, 64, 67, 72])
        markov_feature = np.zeros(128, dtype=np.float32)
        for note in markov_notes:
            if 0 <= note < 128:
                markov_feature[note] += 1
        if np.sum(markov_feature) > 0:
            markov_feature = markov_feature / np.sum(markov_feature)
        markov_tensor = torch.from_numpy(markov_feature).float().to(device)

        # 3. Use VAE for creative variations
        with torch.no_grad():
            recon, mu, logvar = vae_model(midi_tensor.unsqueeze(0))

            # Sample from latent space with creativity factor
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std) * creativity
            z = mu + eps * std

            vae_output = vae_model.decode(z).squeeze(0)

        # 4. Use Transformer for structure
        with torch.no_grad():
            transformer_output = transformer_model(midi_tensor.unsqueeze(0)).squeeze(0)

        # 5. Enhanced combination with scale filtering and note range limiting
        # Start with weighted combination
        combined = torch.zeros_like(markov_tensor)
        
        # Add markov model output with higher emphasis on strong chord tones
        combined += markov_tensor * 0.5
        
        # Add VAE output with scale filtering
        for note_idx in range(128):
            note_class = note_idx % 12
            note_octave = note_idx // 12
            
            # Apply stronger scale filtering for extreme registers
            if note_octave <= 2 or note_octave >= 7:
                # Extreme low/high notes must be in scale
                if note_class in scale_notes:
                    combined[note_idx] += vae_output[note_idx] * 0.3
            else:
                # Middle register allows more creativity
                if note_class in scale_notes:
                    combined[note_idx] += vae_output[note_idx] * 0.4
                else:
                    combined[note_idx] += vae_output[note_idx] * 0.1  # Non-scale notes get reduced influence
        
        # Add transformer output with stronger structure influence
        for note_idx in range(128):
            note_octave = note_idx // 12
            
            # Limit extreme registers from transformer
            if 3 <= note_octave <= 6:  # Comfortable piano range
                combined[note_idx] += transformer_output[note_idx] * 0.3
            else:
                # Reduce influence of extreme notes
                combined[note_idx] += transformer_output[note_idx] * 0.1
        
        # Ensure all values are non-negative
        combined = torch.clamp(combined, min=0.0)
        
        # Re-normalize
        sum_val = combined.sum()
        if sum_val > 0:
            combined = combined / sum_val
        
        # Log tensor statistics for debugging
        logger.info(f"Combined tensor stats - Min: {combined.min().item():.6f}, Max: {combined.max().item():.6f}, Sum: {combined.sum().item():.6f}")

        # Create enhanced MIDI file with proper chord structure
        output_midi_path = tempfile.mktemp(suffix=".mid")
        
        # Use a more structured approach to MIDI creation with chord awareness
        from mido import MidiFile, MidiTrack, Message, MetaMessage
        
        mid = MidiFile(ticks_per_beat=480)
        
        # Melody track
        melody_track = MidiTrack()
        mid.tracks.append(melody_track)
        
        # Chord track
        chord_track = MidiTrack()
        mid.tracks.append(chord_track)
        
        # Set tempo and instruments
        melody_track.append(Message('program_change', program=INSTRUMENT_MAP.get(instrument, 0), time=0))
        chord_track.append(Message('program_change', program=INSTRUMENT_MAP.get('piano', 0), time=0))
        melody_track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(120), time=0))
        
        # Convert probability distribution to actual notes with appropriate timing
        active_notes = []
        ticks_per_quarter = mid.ticks_per_beat
        total_ticks = int(duration * ticks_per_quarter * 4)  # 4 quarters per whole note
        
        # Generate sequences with more natural timing
        note_probs = combined.cpu().numpy()
        
        # Create a pattern of note durations (in ticks)
        durations = [ticks_per_quarter//2, ticks_per_quarter//4, ticks_per_quarter//2, 
                     ticks_per_quarter, ticks_per_quarter//2, ticks_per_quarter//4]
        
        # Scale-filtered notes in descending probability order
        sorted_indices = np.argsort(-note_probs)
        filtered_notes = []
        for idx in sorted_indices:
            if idx % 12 in scale_notes and note_probs[idx] > 0.01:  # Only use notes in scale
                filtered_notes.append(idx)
                if len(filtered_notes) >= 32:  # Limit number of notes
                    break
        
        # Add chord progression (one chord every 2 measures)
        current_tick = 0
        chord_duration = ticks_per_quarter * 8  # 2 measures
        chord_idx = 0
        
        from music21 import harmony
        while current_tick < total_ticks:
            # Get current chord
            chord_name = chord_progression[chord_idx % len(chord_progression)]
            try:
                chord_obj = harmony.ChordSymbol(chord_name)
                chord_notes = [p.midi for p in chord_obj.pitches]
                
                # Add chord notes
                for note in chord_notes:
                    chord_track.append(Message('note_on', note=note, velocity=70, time=0 if note == chord_notes[0] else 0))
                
                # Make sure notes end at different times to avoid MIDI issues
                for i, note in enumerate(chord_notes):
                    end_time = chord_duration if i == 0 else 0
                    chord_track.append(Message('note_off', note=note, velocity=0, time=end_time))
                
                current_tick += chord_duration
                chord_idx += 1
                
            except Exception as e:
                logger.warning(f"Error processing chord {chord_name}: {e}")
                chord_idx += 1
        
        # Add melody using the filtered notes
        current_tick = 0
        melody_idx = 0
        
        while current_tick < total_ticks:
            # Pick note from filtered list with cycling
            note = filtered_notes[melody_idx % len(filtered_notes)]
            
            # Get note duration, cycling through duration patterns
            duration = durations[current_tick // ticks_per_quarter % len(durations)]
            
            # Add note
            melody_track.append(Message('note_on', note=note, velocity=96, time=0))
            melody_track.append(Message('note_off', note=note, velocity=0, time=duration))
            
            current_tick += duration
            melody_idx += 1
        
        # Save the MIDI file
        mid.save(output_midi_path)

        # Generate unique ID and save to persistent location
        unique_id = str(uuid.uuid4())[:8]
        midi_filename = f"combined_{unique_id}.mid"
        midi_dest = os.path.join(MIDI_DIR, midi_filename)

        # Copy file to persistent storage
        shutil.copy2(output_midi_path, midi_dest)

        # Generate audio if requested
        audio_url = None
        if should_generate_audio:
            audio_filename = f"combined_{unique_id}.wav"
            output_audio_path = tempfile.mktemp(suffix=".wav")

            if generate_audio(output_midi_path, output_audio_path, instrument=instrument):
                audio_dest = os.path.join(AUDIO_DIR, audio_filename)
                if os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 0:
                    shutil.copy2(output_audio_path, audio_dest)
                    audio_url = f"/download/audio/{audio_filename}"

        # Create a more detailed response
        return {
            "midi_url": f"/download/midi/{midi_filename}",
            "audio_url": audio_url,
            "message": f"Successfully generated enhanced music in {detected_key}",
            "key": detected_key,
            "time_signature": "4/4",
            "sections": f"Chord progression: {', '.join(chord_progression[:4])}..."
        }

    except Exception as e:
        logger.exception(f"Combined generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/generate/markov", response_model=GenerationResponse)
async def generate_markov(
    key_context: str = Form("C major"),
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
        try:
            # Parse key properly
            root, mode = parse_key_context(key_context)
            clean_key = f"{root} {mode}"
            logger.info(f"Using key: {clean_key} (parsed from {key_context})")
            
            # Generate sequence with expressive features
            complex_output = markov_model.generate_expressive_sequence(
                key_context=clean_key,
                length=length,
                complexity=complexity
            )
        except Exception as e:
            logger.warning(f"Error in key parsing: {e}, falling back to default key")
            complex_output = markov_model.generate_expressive_sequence(
                key_context="C major",  # Default to C major if parsing fails
                length=length,
                complexity=complexity
            )

        # Log generated musical features
        logger.info(f"Generated sequence in key: {complex_output.get('key', 'unknown')}")
        logger.info(f"Time signature: {complex_output.get('time_signature', '4/4')}")
        logger.info(f"Generated {len(complex_output.get('notes', []))} notes")

        # Create output MIDI file with durations
        output_midi_path = tempfile.mktemp(suffix=".mid")
        
        # Get notes and durations
        notes = complex_output.get('notes', [])
        durations = complex_output.get('durations', [])
        if not notes:
            raise Exception("No notes were generated")
            
        if not durations:
            durations = [0.5] * len(notes)  # Default to eighth notes
            
        min_length = min(len(notes), len(durations))
        notes = notes[:min_length]
        durations = durations[:min_length]
        
        # Create MIDI file
        result = create_midi_with_durations(
            notes,
            durations,
            output_midi_path,
            time_signature=complex_output.get('time_signature', '4/4')
        )

        if not result:
            raise Exception("Failed to create MIDI file")

        # Save to persistent storage
        unique_id = str(uuid.uuid4())[:8]
        midi_filename = f"markov_{unique_id}.mid"
        midi_dest = os.path.join(MIDI_DIR, midi_filename)
        os.makedirs(os.path.dirname(midi_dest), exist_ok=True)
        shutil.copy2(output_midi_path, midi_dest)

        # Generate audio if requested
        audio_url = None
        if should_generate_audio:
            audio_filename = f"markov_{unique_id}.wav"
            output_audio_path = tempfile.mktemp(suffix=".wav")

            if generate_audio(output_midi_path, output_audio_path, instrument=instrument):
                audio_dest = os.path.join(AUDIO_DIR, audio_filename)
                if os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 0:
                    shutil.copy2(output_audio_path, audio_dest)
                    audio_url = f"/download/audio/{audio_filename}"

        # Create message
        complexity_desc = "high" if complexity > 0.7 else "medium" if complexity > 0.4 else "low"
        
        return {
            "midi_url": f"/download/midi/{midi_filename}",
            "audio_url": audio_url,
            "message": f"Successfully generated expressive music in {key_context} with {complexity_desc} complexity"
        }

    except Exception as e:
        logger.exception(f"Markov generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Markov generation failed: {str(e)}")

@app.post("/generate/structured_transformer", response_model=GenerationResponse)
async def generate_structured_transformer(
    midi_file: UploadFile = File(...),
    num_sections: int = Form(3),
    section_length: int = Form(16),
    transition_smoothness: float = Form(0.7),
    temperature: float = Form(0.8),
    duration: int = Form(60),
    instrument: str = Form("piano"),
    should_generate_audio: bool = Form(True)
):
    """
    Generate structured music using the transformer model with distinct sections.
    """
    if not models_available["transformer"]:
        raise HTTPException(
            status_code=400,
            detail="Transformer model is not available. Please train the model first."
        )

    try:
        # Save uploaded file to temp location
        temp_midi_path = tempfile.mktemp(suffix=".mid")
        with open(temp_midi_path, "wb") as temp_file:
            temp_file.write(await midi_file.read())
        
        # Parse MIDI file into features
        feature_vector = parse_midi(temp_midi_path)
        
        if feature_vector is None:
            raise HTTPException(status_code=400, detail="Could not parse MIDI file")
        
        # Add batch dimension and move to device
        feature_vector = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Generate structured output
        with torch.no_grad():
            # Reset memory for fresh generation
            transformer_model.reset_memory()
            
            # Generate with structure
            output = transformer_model.generate_with_structure(
                seed=feature_vector,
                num_sections=num_sections,
                section_length=section_length,
                temperature=temperature,
                transition_smoothness=transition_smoothness
            )
            
            # Process output
            output_feature = output[0]  # Remove batch dimension
            if len(output_feature.shape) > 1:
                output_feature = output_feature.mean(dim=0)  # Average across sequence dimension
        
        # Create output MIDI file
        output_midi_path = tempfile.mktemp(suffix=".mid")
        create_midi_file(output_feature.cpu().numpy(), output_midi_path, duration_seconds=duration)
        
        # Save to persistent storage
        unique_id = str(uuid.uuid4())[:8]
        midi_filename = f"structured_{unique_id}.mid"
        midi_dest = os.path.join(MIDI_DIR, midi_filename)
        os.makedirs(os.path.dirname(midi_dest), exist_ok=True)
        shutil.copy2(output_midi_path, midi_dest)
        
        # Generate audio if requested
        audio_url = None
        if should_generate_audio:
            audio_filename = f"structured_{unique_id}.wav"
            output_audio_path = tempfile.mktemp(suffix=".wav")

            if generate_audio(output_midi_path, output_audio_path, instrument=instrument):
                audio_dest = os.path.join(AUDIO_DIR, audio_filename)
                if os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 0:
                    shutil.copy2(output_audio_path, audio_dest)
                    audio_url = f"/download/audio/{audio_filename}"

        return {
            "midi_url": f"/download/midi/{midi_filename}",
            "audio_url": audio_url,
            "message": f"Successfully generated structured music with {num_sections} sections"
        }

    except Exception as e:
        logger.exception(f"Structured transformer generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Structured transformer generation failed: {str(e)}")

@app.post("/generate/vae", response_model=GenerationResponse)
async def generate_vae(
    midi_file: UploadFile = File(...),
    creativity: float = Form(0.5),
    duration: int = Form(30),
    instrument: str = Form("piano"),
    should_generate_audio: bool = Form(True)
):
    """
    Generate music using the VAE model for creative variations.
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
            vae_output = vae_model.decode(z).squeeze(0)

        # Create output MIDI file
        output_midi_path = tempfile.mktemp(suffix=".mid")
        create_midi_file(vae_output.cpu().numpy(), output_midi_path, duration_seconds=duration)

        # Save to persistent storage
        unique_id = str(uuid.uuid4())[:8]
        midi_filename = f"vae_{unique_id}.mid"
        midi_dest = os.path.join(MIDI_DIR, midi_filename)
        shutil.copy2(output_midi_path, midi_dest)

        # Generate audio if requested
        audio_url = None
        if should_generate_audio:
            audio_filename = f"vae_{unique_id}.wav"
            output_audio_path = tempfile.mktemp(suffix=".wav")

            if generate_audio(output_midi_path, output_audio_path, instrument=instrument):
                audio_dest = os.path.join(AUDIO_DIR, audio_filename)
                if os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 0:
                    shutil.copy2(output_audio_path, audio_dest)
                    audio_url = f"/download/audio/{audio_filename}"

        return {
            "midi_url": f"/download/midi/{midi_filename}",
            "audio_url": audio_url,
            "message": f"Successfully generated music with creativity level {creativity}"
        }

    except Exception as e:
        logger.exception(f"VAE generation error: {e}")
        raise HTTPException(status_code=500, detail=f"VAE generation failed: {str(e)}")

def create_midi_with_durations(notes, durations, output_path, time_signature="4/4"):
    """
    Create a MIDI file with specific note durations
    
    Args:
        notes: List of MIDI note numbers (0-127)
        durations: List of durations in quarter notes (0.5 = eighth note, 1.0 = quarter, etc.)
        output_path: Path to save the MIDI file
        time_signature: Time signature string (e.g. "4/4", "3/4")
    """
    try:
        from midiutil.MidiFile import MIDIFile
        import os
        
        # Create MIDI file with 1 track
        mf = MIDIFile(1)
        
        # Set track information
        track = 0
        channel = 0
        time = 0  # Start at beginning
        tempo = 120  # BPM
        
        mf.addTrackName(track, time, "Markov Generated Track")
        mf.addTempo(track, time, tempo)
        
        # Parse time signature
        try:
            numerator, denominator = map(int, time_signature.split('/'))
            denominator_bits = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4, 32: 5}.get(denominator, 2)
            mf.addTimeSignature(track, time, numerator, denominator_bits, 24, 8)
        except Exception as e:
            logger.warning(f"Failed to parse time signature {time_signature}: {e}")
            mf.addTimeSignature(track, time, 4, 2, 24, 8)
        
        # Add notes with their durations
        for i, (note_num, duration) in enumerate(zip(notes, durations)):
            if 0 <= note_num < 128:
                # Add velocity variation for more natural sound
                velocity = 112 if i % 4 == 0 else (96 if i % 2 == 0 else 100)
                mf.addNote(track, channel, note_num, time, duration, velocity)
                time += duration
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        # Write the MIDI file
        with open(output_path, 'wb') as outf:
            mf.writeFile(outf)
            
        return os.path.exists(output_path)
    except Exception as e:
        logger.error(f"Error creating MIDI file: {e}")
        return False

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

@app.get("/health")
async def health_check():
    """Check if the API and models are working properly"""
    return {
        "status": "healthy",
        "device": str(device),
        "models_loaded": models_available
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)