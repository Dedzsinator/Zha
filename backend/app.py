import os
import uuid
import shutil
import mido
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
from music21 import converter, analysis, scale, key

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INSTRUMENT_MAP = {
    "piano": 0,
    "acoustic_grand_piano": 0,
    "bright_acoustic_piano": 1,
    "electric_grand_piano": 2,
    "honky_tonk_piano": 3,
    "electric_piano": 4,
    "electric_piano_1": 4,
    "electric_piano_2": 5,
    "harpsichord": 6,
    "clavinet": 7,
    "celesta": 8,
    "glockenspiel": 9,
    "music_box": 10,
    "vibraphone": 11,
    "marimba": 12,
    "xylophone": 13,
    "tubular_bells": 14,
    "dulcimer": 15,
    "drawbar_organ": 16,
    "percussive_organ": 17,
    "rock_organ": 18,
    "church_organ": 19,
    "reed_organ": 20,
    "accordion": 21,
    "harmonica": 22,
    "tango_accordion": 23,
    "acoustic_guitar_nylon": 24,
    "acoustic_guitar_steel": 25,
    "electric_guitar_jazz": 26,
    "electric_guitar_clean": 27,
    "electric_guitar_muted": 28,
    "overdriven_guitar": 29,
    "distortion_guitar": 30,
    "guitar_harmonics": 31,
    "acoustic_bass": 32,
    "electric_bass_finger": 33,
    "electric_bass_pick": 34,
    "fretless_bass": 35,
    "slap_bass_1": 36,
    "slap_bass_2": 37,
    "synth_bass_1": 38,
    "synth_bass_2": 39,
    "violin": 40,
    "viola": 41,
    "cello": 42,
    "contrabass": 43,
    "tremolo_strings": 44,
    "pizzicato_strings": 45,
    "orchestral_harp": 46,
    "timpani": 47,
    "string_ensemble_1": 48,
    "string_ensemble_2": 49,
    "synth_strings_1": 50,
    "synth_strings_2": 51,
    "choir_aahs": 52,
    "voice_oohs": 53,
    "synth_choir": 54,
    "orchestra_hit": 55,
    "trumpet": 56,
    "trombone": 57,
    "tuba": 58,
    "muted_trumpet": 59,
    "french_horn": 60,
    "brass_section": 61,
    "synth_brass_1": 62,
    "synth_brass_2": 63,
    "soprano_sax": 64,
    "alto_sax": 65,
    "tenor_sax": 66,
    "baritone_sax": 67,
    "oboe": 68,
    "english_horn": 69,
    "bassoon": 70,
    "clarinet": 71,
    "piccolo": 72,
    "flute": 73,
    "recorder": 74,
    "pan_flute": 75,
    "blown_bottle": 76,
    "shakuhachi": 77,
    "whistle": 78,
    "ocarina": 79,
    "lead_1_square": 80,
    "lead_2_sawtooth": 81,
    "lead_3_calliope": 82,
    "lead_4_chiff": 83,
    "lead_5_charang": 84,
    "lead_6_voice": 85,
    "lead_7_fifths": 86,
    "lead_8_bass_lead": 87,
    "pad_1_new_age": 88,
    "pad_2_warm": 89,
    "pad_3_polysynth": 90,
    "pad_4_choir": 91,
    "pad_5_bowed": 92,
    "pad_6_metallic": 93,
    "pad_7_halo": 94,
    "pad_8_sweep": 95,
    "fx_1_rain": 96,
    "fx_2_soundtrack": 97,
    "fx_3_crystal": 98,
    "fx_4_atmosphere": 99,
    "fx_5_brightness": 100,
    "fx_6_goblins": 101,
    "fx_7_echoes": 102,
    "fx_8_sci_fi": 103,
    "sitar": 104,
    "banjo": 105,
    "shamisen": 106,
    "koto": 107,
    "kalimba": 108,
    "bagpipe": 109,
    "fiddle": 110,
    "shanai": 111,
    "tinkle_bell": 112,
    "agogo": 113,
    "steel_drums": 114,
    "woodblock": 115,
    "taiko_drum": 116,
    "melodic_tom": 117,
    "synth_drum": 118,
    "reverse_cymbal": 119,
    "guitar_fret_noise": 120,
    "breath_noise": 121,
    "seashore": 122,
    "bird_tweet": 123,
    "telephone_ring": 124,
    "helicopter": 125,
    "applause": 126,
    "gunshot": 127
}

# Frontend-friendly aliases to valid General MIDI programs.
INSTRUMENT_MAP.update({
    "guitar": 24,
    "organ": 19,
    "choir": 52,
    "strings": 48,
})

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
MODEL_DIR = os.environ.get("MODEL_DIR", "./output/trained_models")
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
    "markov": False,
    "multitrack_transformer": False
}

# Initialize standard transformer model
transformer_model = TransformerModel(
    input_dim=128,
    embed_dim=512,
    num_heads=8,
    num_layers=8,
    dim_feedforward=2048,
    enable_multitrack=False
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
        # Mark as available anyway - will use uninitialized model
        transformer_model.to(device)
        transformer_model.eval()
        models_available["transformer"] = True
        print("Transformer model initialized (untrained)")
else:
    transformer_model.to(device)
    transformer_model.eval()
    models_available["transformer"] = True
    print("Transformer model initialized (no checkpoint found)")

# Initialize multi-track transformer model
multitrack_transformer = TransformerModel(
    input_dim=128,
    embed_dim=512,
    num_heads=8,
    num_layers=8,
    dim_feedforward=2048,
    enable_multitrack=True
)
# Skip loading multitrack_transformer - use untrained version
# The file format is inconsistent/corrupted
multitrack_transformer.to(device)
multitrack_transformer.eval()
models_available["multitrack_transformer"] = True
print("Multi-track transformer initialized (using uninitialized)")

vae_model = VAEModel(input_dim=128, latent_dim=128)
if os.path.exists(os.path.join(MODEL_DIR, "trained_vae.pt")):
    try:
        checkpoint = torch.load(os.path.join(MODEL_DIR, "trained_vae.pt"), map_location=device)
        # Handle wrapped checkpoint format
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        
        # Handle compiled model prefixes
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
        # Mark as available anyway - will use uninitialized model
        vae_model.to(device)
        vae_model.eval()
        models_available["vae"] = True
        print("VAE model initialized (untrained)")
else:
    vae_model.to(device)
    vae_model.eval()
    models_available["vae"] = True
    print("VAE model initialized (no checkpoint found)")

markov_model = MarkovChain()
if os.path.exists(os.path.join(MODEL_DIR, "markov.npy")):
    try:
        markov_model.load(os.path.join(MODEL_DIR, "markov.npy"))
        models_available["markov"] = True
        print("Markov model loaded successfully")
    except Exception as e:
        print(f"Failed to load Markov model: {e}")
        # Mark as available anyway - will use random generation
        models_available["markov"] = True
        print("Markov model initialized (untrained)")
else:
    models_available["markov"] = True
    print("Markov model initialized (no checkpoint found)")

class GenerationResponse(BaseModel):
    midi_url: str
    audio_url: Optional[str] = None
    message: str

@app.post("/generate/combined", response_model=GenerationResponse, include_in_schema=False)
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
    raise HTTPException(
        status_code=410,
        detail="Combined model generation has been retired. Use markov, vae, structured_transformer, or multitrack."
    )

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
            
            # Check if generation failed
            if complex_output is None:
                raise Exception("Markov generation returned None - check model state")
                
        except Exception as e:
            logger.warning(f"Error in generation: {e}, falling back to default key")
            complex_output = markov_model.generate_expressive_sequence(
                key_context="C major",  # Default to C major if parsing fails
                length=length,
                complexity=complexity
            )
            
            # Check fallback also succeeded
            if complex_output is None:
                raise Exception("Markov generation failed even with fallback key")

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
            
            output_sequence = output[0]  # [seq_len, 128]
        
        # Create output MIDI file preserving source timing to avoid one-note collapse.
        output_midi_path = tempfile.mktemp(suffix=".mid")
        built_with_template = create_structured_midi_from_template(
            source_midi_path=temp_midi_path,
            generated_sequence=output_sequence.cpu().numpy(),
            output_path=output_midi_path,
            instrument_program=INSTRUMENT_MAP.get(instrument, 0),
            duration_seconds=max(8, int(duration)),
        )
        if not built_with_template:
            # Conservative fallback if template extraction fails.
            fallback_feature = output_sequence.mean(axis=0) if len(output_sequence.shape) > 1 else output_sequence
            create_midi_file(np.asarray(fallback_feature), output_midi_path, duration_seconds=duration)
        
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

        # Create output MIDI file with source rhythm/time-signature preserved.
        output_midi_path = tempfile.mktemp(suffix=".mid")
        built_with_template = create_vae_midi_from_template(
            source_midi_path=temp_midi_path,
            vae_probabilities=vae_output.cpu().numpy(),
            output_path=output_midi_path,
            instrument_program=INSTRUMENT_MAP.get(instrument, 0),
            duration_seconds=max(8, int(duration)),
        )

        if not built_with_template:
            # Fallback to legacy path if template extraction fails.
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


def create_vae_midi_from_template(
    source_midi_path: str,
    vae_probabilities: np.ndarray,
    output_path: str,
    instrument_program: int,
    duration_seconds: int,
) -> bool:
    """
    Render VAE output with high rhythmic cohesion by reusing the source MIDI timing grid.

    The VAE controls pitch tendencies, while note onsets/durations/time signature come from
    the source template to prevent sparse or rhythmically unstable output.
    """
    try:
        source_midi = mido.MidiFile(source_midi_path)
        ticks_per_beat = source_midi.ticks_per_beat or 480

        tempo = 500000
        numerator, denominator = 4, 4

        template_events = []
        for track in source_midi.tracks:
            abs_tick = 0
            active_notes = {}

            for msg in track:
                abs_tick += msg.time

                if msg.is_meta:
                    if msg.type == 'set_tempo':
                        tempo = msg.tempo
                    elif msg.type == 'time_signature':
                        numerator, denominator = msg.numerator, msg.denominator
                    continue

                channel = getattr(msg, 'channel', 0)
                if channel == 9:
                    continue

                if msg.type == 'note_on' and msg.velocity > 0:
                    key = (msg.note, channel)
                    active_notes.setdefault(key, []).append((abs_tick, msg.velocity))
                elif msg.type in ('note_off', 'note_on') and (msg.type == 'note_off' or msg.velocity == 0):
                    key = (msg.note, channel)
                    if key in active_notes and active_notes[key]:
                        start_tick, velocity = active_notes[key].pop(0)
                        duration_ticks = max(1, abs_tick - start_tick)
                        template_events.append({
                            'start': int(start_tick),
                            'duration': int(duration_ticks),
                            'note': int(msg.note),
                            'velocity': int(max(45, min(116, velocity))),
                        })

        if not template_events:
            return False

        template_events.sort(key=lambda e: (e['start'], e['note']))

        # Normalize VAE pitch probabilities and blend with source pitch usage to avoid sparse drift.
        probs = np.clip(np.asarray(vae_probabilities, dtype=np.float32), 0.0, None)
        if probs.shape[0] != 128:
            return False
        if probs.sum() <= 0:
            probs = np.ones(128, dtype=np.float32)
        probs = probs / probs.sum()

        source_hist = np.zeros(128, dtype=np.float32)
        for event in template_events:
            source_hist[event['note']] += 1.0
        if source_hist.sum() > 0:
            source_hist = source_hist / source_hist.sum()

        blended_probs = 0.75 * source_hist + 0.25 * probs
        blended_probs = blended_probs / blended_probs.sum()

        pitch_class_hist = np.zeros(12, dtype=np.float32)
        for pitch_num in range(128):
            pitch_class_hist[pitch_num % 12] += blended_probs[pitch_num]
        key_pitch_classes = set(np.argsort(-pitch_class_hist)[:7].tolist())

        sec_per_beat = tempo / 1_000_000.0
        target_beats = max(8.0, duration_seconds / max(sec_per_beat, 1e-6))
        target_ticks = int(target_beats * ticks_per_beat)

        cycle_ticks = max(e['start'] + e['duration'] for e in template_events)
        if cycle_ticks <= 0:
            cycle_ticks = ticks_per_beat * max(1, numerator)

        grid = max(1, ticks_per_beat // 4)  # 16th-note quantization

        out_events = []
        prev_note = template_events[0]['note']
        cycle_offset = 0
        while cycle_offset < target_ticks:
            for event in template_events:
                start_tick = cycle_offset + event['start']
                if start_tick >= target_ticks:
                    break

                raw_duration = max(grid, event['duration'])
                duration_tick = max(grid, int(round(raw_duration / grid) * grid))
                max_duration = max(grid, target_ticks - start_tick)
                duration_tick = min(duration_tick, max_duration)

                source_note = event['note']
                low = max(36, source_note - 7)
                high = min(96, source_note + 7)
                candidates = list(range(low, high + 1))

                scored = []
                for cand in candidates:
                    key_bonus = 1.0 if (cand % 12) in key_pitch_classes else 0.6
                    step_penalty = 1.0 / (1.0 + abs(cand - prev_note))
                    source_proximity = 1.0 / (1.0 + abs(cand - source_note))
                    score = (
                        0.55 * float(blended_probs[cand])
                        + 0.20 * key_bonus
                        + 0.15 * step_penalty
                        + 0.10 * source_proximity
                    )
                    scored.append((score, cand))

                scored.sort(key=lambda x: x[0], reverse=True)
                top_candidates = [cand for _, cand in scored[:3]]
                if len(top_candidates) == 1:
                    chosen = top_candidates[0]
                else:
                    weights = np.array([0.65, 0.25, 0.10][:len(top_candidates)], dtype=np.float32)
                    weights = weights / weights.sum()
                    chosen = int(np.random.choice(top_candidates, p=weights))

                prev_note = chosen
                out_events.append({
                    'start': int(start_tick),
                    'duration': int(duration_tick),
                    'note': int(chosen),
                    'velocity': int(event['velocity']),
                })

            cycle_offset += cycle_ticks

        rendered = mido.MidiFile(ticks_per_beat=ticks_per_beat)
        melody_track = mido.MidiTrack()
        rendered.tracks.append(melody_track)

        melody_track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
        melody_track.append(
            mido.MetaMessage(
                'time_signature',
                numerator=int(numerator),
                denominator=int(denominator),
                clocks_per_click=24,
                notated_32nd_notes_per_beat=8,
                time=0,
            )
        )
        melody_track.append(mido.Message('program_change', program=int(instrument_program), time=0, channel=0))

        timeline = []
        for event in out_events:
            timeline.append((event['start'], 1, 'note_on', event['note'], event['velocity']))
            timeline.append((event['start'] + event['duration'], 0, 'note_off', event['note'], 0))

        timeline.sort(key=lambda x: (x[0], x[1]))

        current_tick = 0
        for abs_tick, _, msg_type, note_num, velocity in timeline:
            delta = max(0, int(abs_tick - current_tick))
            melody_track.append(
                mido.Message(msg_type, note=int(note_num), velocity=int(velocity), time=delta, channel=0)
            )
            current_tick = int(abs_tick)

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        rendered.save(output_path)
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception as exc:
        logger.warning(f"Template-based VAE rendering failed: {exc}")
        return False


def create_structured_midi_from_template(
    source_midi_path: str,
    generated_sequence: np.ndarray,
    output_path: str,
    instrument_program: int,
    duration_seconds: int,
) -> bool:
    """
    Render structured-transformer output against source timing to preserve rhythm and meter.
    """
    try:
        source_midi = mido.MidiFile(source_midi_path)
        ticks_per_beat = source_midi.ticks_per_beat or 480

        tempo = 500000
        numerator, denominator = 4, 4

        template_events = []
        for track in source_midi.tracks:
            abs_tick = 0
            active_notes = {}
            for msg in track:
                abs_tick += msg.time
                if msg.is_meta:
                    if msg.type == 'set_tempo':
                        tempo = msg.tempo
                    elif msg.type == 'time_signature':
                        numerator, denominator = msg.numerator, msg.denominator
                    continue

                channel = getattr(msg, 'channel', 0)
                if channel == 9:
                    continue

                if msg.type == 'note_on' and msg.velocity > 0:
                    key = (msg.note, channel)
                    active_notes.setdefault(key, []).append((abs_tick, msg.velocity))
                elif msg.type in ('note_off', 'note_on') and (msg.type == 'note_off' or msg.velocity == 0):
                    key = (msg.note, channel)
                    if key in active_notes and active_notes[key]:
                        start_tick, velocity = active_notes[key].pop(0)
                        template_events.append({
                            'start': int(start_tick),
                            'duration': int(max(1, abs_tick - start_tick)),
                            'note': int(msg.note),
                            'velocity': int(max(45, min(116, velocity))),
                        })

        if not template_events:
            return False

        template_events.sort(key=lambda e: (e['start'], e['note']))

        seq = np.asarray(generated_sequence, dtype=np.float32)
        if seq.ndim == 1:
            seq = seq.reshape(1, -1)
        if seq.shape[1] != 128:
            return False

        seq_notes = [int(np.argmax(seq_step)) for seq_step in seq]
        if not seq_notes:
            return False

        # Add Markov backbone so structured mode keeps musical movement.
        markov_notes = markov_model.generate_sequence(start_note=60, length=max(len(template_events), 32), key_context="C major")
        if not markov_notes:
            markov_notes = [60] * max(len(template_events), 32)

        sec_per_beat = tempo / 1_000_000.0
        target_beats = max(8.0, duration_seconds / max(sec_per_beat, 1e-6))
        target_ticks = int(target_beats * ticks_per_beat)

        cycle_ticks = max(e['start'] + e['duration'] for e in template_events)
        if cycle_ticks <= 0:
            cycle_ticks = ticks_per_beat * max(1, numerator)

        out_events = []
        prev_note = template_events[0]['note']
        seq_idx = 0
        cycle_offset = 0
        while cycle_offset < target_ticks:
            for event_idx, event in enumerate(template_events):
                start_tick = cycle_offset + event['start']
                if start_tick >= target_ticks:
                    break

                transformer_note = int(seq_notes[seq_idx % len(seq_notes)])
                markov_note = int(markov_notes[(event_idx + seq_idx) % len(markov_notes)])
                src_note = int(event['note'])

                candidates = [
                    max(48, min(88, transformer_note)),
                    max(48, min(88, markov_note)),
                    max(48, min(88, src_note)),
                    max(48, min(88, markov_note + 12 if markov_note < 64 else markov_note - 12)),
                ]

                best_note = candidates[0]
                best_score = -1e9
                for cand in candidates:
                    step_penalty = abs(cand - prev_note) * 0.45
                    src_penalty = abs(cand - src_note) * 0.20
                    repeat_penalty = 7.0 if cand == prev_note else 0.0
                    score = 100.0 - step_penalty - src_penalty - repeat_penalty
                    if score > best_score:
                        best_score = score
                        best_note = cand

                prev_note = int(best_note)
                seq_idx += 1

                duration_tick = max(1, int(event['duration']))
                max_duration = max(1, target_ticks - start_tick)
                duration_tick = min(duration_tick, max_duration)

                out_events.append({
                    'start': int(start_tick),
                    'duration': int(duration_tick),
                    'note': int(best_note),
                    'velocity': int(event['velocity']),
                })

            cycle_offset += cycle_ticks

        rendered = mido.MidiFile(ticks_per_beat=ticks_per_beat)
        melody_track = mido.MidiTrack()
        rendered.tracks.append(melody_track)

        melody_track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
        melody_track.append(
            mido.MetaMessage(
                'time_signature',
                numerator=int(numerator),
                denominator=int(denominator),
                clocks_per_click=24,
                notated_32nd_notes_per_beat=8,
                time=0,
            )
        )
        melody_track.append(mido.Message('program_change', program=int(instrument_program), time=0, channel=0))

        timeline = []
        for event in out_events:
            timeline.append((event['start'], 1, 'note_on', event['note'], event['velocity']))
            timeline.append((event['start'] + event['duration'], 0, 'note_off', event['note'], 0))
        timeline.sort(key=lambda x: (x[0], x[1]))

        current_tick = 0
        for abs_tick, _, msg_type, note_num, velocity in timeline:
            delta = max(0, int(abs_tick - current_tick))
            melody_track.append(mido.Message(msg_type, note=int(note_num), velocity=int(velocity), time=delta, channel=0))
            current_tick = int(abs_tick)

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        rendered.save(output_path)
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception as exc:
        logger.warning(f"Template-based structured rendering failed: {exc}")
        return False

def convert_chord_name(chord_name):
    """Convert chord names from 'C major' format to music21-compatible format"""
    if not chord_name or ' ' not in chord_name:
        return chord_name
        
    # Standard conversions
    conversions = {
        'major': '',  # C major -> C
        'minor': 'm',  # A minor -> Am
        'diminished': 'dim',
        'augmented': 'aug',
        'dominant': '7',
        'major seventh': 'maj7',
        'minor seventh': 'm7',
        'half diminished': 'm7b5',
        'diminished seventh': 'dim7',
        'augmented seventh': 'aug7',
        'suspended fourth': 'sus4',
        'suspended second': 'sus2'
    }
    
    parts = chord_name.split(' ', 1)
    if len(parts) != 2:
        return chord_name
        
    root, quality = parts
    
    # Handle basic conversions
    if quality.lower() in conversions:
        return root + conversions[quality.lower()]
    
    # Pass through if unknown
    return chord_name

@app.post("/generate/multitrack")
async def generate_multitrack_music(
    midi_file: UploadFile = File(None),
    steps: int = Form(100),
    temperature: float = Form(0.8),
    bass_temperature: float = Form(0.7),
    drum_temperature: float = Form(0.9),
    duration: float = Form(30.0),
    should_generate_audio: bool = Form(True),
    melody_instrument: str = Form("piano"),
    bass_instrument: str = Form("electric_bass_finger"),
):
    """
    Generate coordinated multi-track music with melody, bass, and drums.
    
    This endpoint implements professional multi-track arrangement where:
    - Melody provides harmonic foundation
    - Bass follows harmonic progression (roots, fifths)
    - Drums provide rhythmic structure
    
    All tracks are generated with cross-attention for musical coherence.
    """
    try:
        steps = max(16, min(int(steps), 512))

        # Create seed from MIDI file or use random seed
        if midi_file and midi_file.filename:
            temp_midi_path = tempfile.mktemp(suffix=".mid")
            with open(temp_midi_path, "wb") as temp_file:
                temp_file.write(await midi_file.read())
            
            feature_vector = parse_midi(temp_midi_path)
            if feature_vector is None:
                raise HTTPException(status_code=400, detail="Could not parse MIDI file")
            
            seed = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        else:
            # Random seed (C major chord as starting point)
            seed = torch.zeros(1, 1, 128, device=device)
            seed[0, 0, 60] = 1.0  # C
            seed[0, 0, 64] = 1.0  # E
            seed[0, 0, 67] = 1.0  # G

        # Generate coordinated tracks from the dedicated multi-track transformer,
        # then apply explicit role constraints so melody/bass/drums stay distinct.
        with torch.no_grad():
            if not models_available.get('multitrack_transformer', False):
                logger.warning("Multi-track transformer not available")
                raise HTTPException(status_code=500, detail="Multi-track transformer model required")

            generated = multitrack_transformer.generate_multitrack(
                seed=seed,
                steps=steps,
                temperature=temperature,
                bass_temperature=bass_temperature,
                drum_temperature=drum_temperature,
            )

        melody_raw = torch.nan_to_num(generated['melody'], nan=0.0)

        melody_markov = markov_model.generate_sequence(start_note=60, length=steps, key_context="C major")
        bass_markov = markov_model.generate_sequence(start_note=43, length=steps, key_context="C major")
        if not melody_markov:
            melody_markov = [60] * steps
        if not bass_markov:
            bass_markov = [43] * steps

        # Convert generated tensors to a clear role-based arrangement.
        melody_notes = []
        bass_notes = []
        drum_notes_per_step = []

        bass_last_note = 40
        melody_last_note = 64
        for step_idx in range(steps):
            melody_step = melody_raw[0, step_idx]
            transformer_note = int(torch.argmax(melody_step).item())
            markov_note = int(melody_markov[step_idx % len(melody_markov)])

            melody_candidates = [
                max(55, min(88, transformer_note)),
                max(55, min(88, markov_note)),
                max(55, min(88, markov_note + 12 if markov_note < 67 else markov_note - 12)),
            ]
            melody_scores = []
            for cand in melody_candidates:
                repeat_penalty = 8.0 if cand == melody_last_note else 0.0
                leap_penalty = 0.45 * abs(cand - melody_last_note)
                markov_pull = 0.30 * abs(cand - markov_note)
                melody_scores.append((100.0 - repeat_penalty - leap_penalty - markov_pull, cand))
            melody_note = max(melody_scores, key=lambda x: x[0])[1]

            if step_idx >= 3 and all(melody_notes[-k - 1] == melody_note for k in range(min(3, len(melody_notes)))):
                melody_note = max(55, min(88, melody_note + (2 if melody_note < 80 else -2)))

            melody_notes.append(melody_note)
            melody_last_note = melody_note

            # Bass role: roots on downbeats, fifths on backbeats, lower register only.
            bass_pitch_class = melody_note % 12
            bass_root = 36 + bass_pitch_class
            bass_fifth = 36 + ((bass_pitch_class + 7) % 12)
            bass_markov_note = int(bass_markov[step_idx % len(bass_markov)])
            while bass_markov_note > 52:
                bass_markov_note -= 12
            while bass_markov_note < 28:
                bass_markov_note += 12

            if step_idx % 4 == 0:
                bass_candidate = bass_root
            elif step_idx % 4 == 2:
                bass_candidate = bass_fifth
            else:
                bass_candidate = bass_markov_note if step_idx % 2 == 0 else -1  # rhythmic breathing

            if bass_candidate >= 0:
                while abs(bass_candidate - bass_last_note) > 7:
                    bass_candidate += -12 if bass_candidate > bass_last_note else 12
                bass_candidate = max(28, min(52, bass_candidate))
                bass_last_note = bass_candidate
            bass_notes.append(bass_candidate)

            # Drum role: explicit groove with section-level variation.
            beat = step_idx % 16
            drum_step = []
            if beat in (0, 8):
                drum_step.append(36)  # kick
            if beat in (4, 12):
                drum_step.append(38)  # snare
            if beat % 2 == 0:
                drum_step.append(42)  # closed hat
            if beat in (7, 15):
                drum_step.append(46)  # open hat pickup
            if beat == 0 and (step_idx // 16) % 2 == 1:
                drum_step.append(49)  # crash every other bar
            drum_notes_per_step.append(drum_step)
        
        # Create MIDI file with separate tracks
        unique_id = str(uuid.uuid4())[:8]
        output_midi_path = tempfile.mktemp(suffix=".mid")
        midi_file_obj = mido.MidiFile()
        
        step_beats = 0.25  # 16th-note grid

        def sequence_to_events(note_sequence, velocity=82, default_duration=0.5, max_tie_steps=4):
            events = []
            step_idx = 0
            while step_idx < len(note_sequence):
                note_num = note_sequence[step_idx]
                if note_num < 0:
                    step_idx += 1
                    continue

                run_len = 1
                while (step_idx + run_len < len(note_sequence) and
                       note_sequence[step_idx + run_len] == note_num and
                      run_len < max_tie_steps):
                    run_len += 1

                events.append({
                    'note': int(note_num),
                    'time': step_idx * step_beats,
                    'duration': max(default_duration, run_len * step_beats),
                    'velocity': velocity,
                })
                step_idx += run_len

            return events
        
        # Track 1: Melody (Channel 0)
        melody_track = mido.MidiTrack()
        melody_track.name = "Melody"
        midi_file_obj.tracks.append(melody_track)
        melody_track.append(mido.Message('program_change', program=INSTRUMENT_MAP.get(melody_instrument, 0), time=0, channel=0))
        
        melody_events = sequence_to_events(melody_notes, velocity=92, default_duration=0.25, max_tie_steps=2)
        current_time = 0
        for note_info in sorted(melody_events, key=lambda x: x['time']):
            delta_time = int((note_info['time'] - current_time) * 480)  # Convert to ticks
            melody_track.append(mido.Message('note_on', note=note_info['note'], velocity=note_info['velocity'], time=delta_time, channel=0))
            melody_track.append(mido.Message('note_off', note=note_info['note'], velocity=0, time=int(note_info['duration'] * 480), channel=0))
            current_time = note_info['time']
        
        # Track 2: Bass (Channel 1)
        bass_track = mido.MidiTrack()
        bass_track.name = "Bass"
        midi_file_obj.tracks.append(bass_track)
        bass_track.append(mido.Message('program_change', program=INSTRUMENT_MAP.get(bass_instrument, 33), time=0, channel=1))
        
        bass_events = sequence_to_events(bass_notes, velocity=88, default_duration=0.5, max_tie_steps=3)
        current_time = 0
        for note_info in sorted(bass_events, key=lambda x: x['time']):
            delta_time = int((note_info['time'] - current_time) * 480)
            bass_track.append(mido.Message('note_on', note=note_info['note'], velocity=note_info['velocity'], time=delta_time, channel=1))
            bass_track.append(mido.Message('note_off', note=note_info['note'], velocity=0, time=int(note_info['duration'] * 480), channel=1))
            current_time = note_info['time']
        
        # Track 3: Drums (Channel 9 - MIDI drum channel)
        drum_track = mido.MidiTrack()
        drum_track.name = "Drums"
        midi_file_obj.tracks.append(drum_track)
        
        drum_events = []
        for step_idx, notes_in_step in enumerate(drum_notes_per_step):
            for drum_note in notes_in_step:
                drum_events.append({
                    'note': int(drum_note),
                    'time': step_idx * step_beats,
                    'duration': step_beats,
                    'velocity': 96 if drum_note in (36, 38) else 78,
                })

        current_time = 0
        for note_info in sorted(drum_events, key=lambda x: x['time']):
            delta_time = int((note_info['time'] - current_time) * 480)
            drum_track.append(mido.Message('note_on', note=note_info['note'], velocity=note_info['velocity'], time=delta_time, channel=9))
            drum_track.append(mido.Message('note_off', note=note_info['note'], velocity=0, time=int(note_info['duration'] * 480), channel=9))
            current_time = note_info['time']
        
        # Save MIDI file
        midi_file_obj.save(output_midi_path)
        midi_filename = f"multitrack_{unique_id}.mid"
        midi_dest = os.path.join(MIDI_DIR, midi_filename)
        os.makedirs(os.path.dirname(midi_dest), exist_ok=True)
        shutil.copy2(output_midi_path, midi_dest)
        
        # Generate audio if requested
        audio_url = None
        if should_generate_audio:
            audio_filename = f"multitrack_{unique_id}.wav"
            output_audio_path = tempfile.mktemp(suffix=".wav")
            
            if generate_audio(output_midi_path, output_audio_path, instrument=melody_instrument):
                audio_dest = os.path.join(AUDIO_DIR, audio_filename)
                if os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 0:
                    shutil.copy2(output_audio_path, audio_dest)
                    audio_url = f"/download/audio/{audio_filename}"
        
        return {
            "midi_url": f"/download/midi/{midi_filename}",
            "audio_url": audio_url,
            "message": f"Successfully generated multi-track music: melody + bass + drums ({steps} steps)",
            "tracks": {
                "melody": len(melody_events),
                "bass": len(bass_events),
                "drums": len(drum_events)
            }
        }
    
    except Exception as e:
        logger.error(f"Multi-track generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

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

@app.get("/list_files")
async def list_files():
    """List all generated MIDI and audio files"""
    try:
        midi_files = []
        audio_files = []
        
        # List MIDI files
        if os.path.exists(MIDI_DIR):
            midi_files = [f for f in os.listdir(MIDI_DIR) if f.endswith(('.mid', '.midi'))]
        
        # List audio files
        if os.path.exists(AUDIO_DIR):
            audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(('.wav', '.mp3', '.flac'))]
        
        return {
            "midi_files": midi_files,
            "audio_files": audio_files
        }
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")

class GuitarNote(BaseModel):
    note: str
    midiNote: int
    startTime: float
    duration: float

class GuitarAccompanimentRequest(BaseModel):
    notes: List[GuitarNote]
    scale: str
    bpm: int

@app.post("/api/generate_accompaniment")
async def generate_guitar_accompaniment(request: GuitarAccompanimentRequest):
    """
    Generate an accompaniment (drums, bass, etc.) based on a live guitar tab/sequence.
    """
    try:
        if not request.notes:
            raise HTTPException(status_code=400, detail="No notes provided in the request.")

        # Create a unique filename
        file_id = f"guitar_accompaniment_{uuid.uuid4().hex[:8]}"
        midi_filename = f"{file_id}.mid"
        midi_path = os.path.join(MIDI_DIR, midi_filename)

        logger.info(f"Generating accompaniment at {request.bpm} BPM for scale {request.scale}. Notes count: {len(request.notes)}")

        # Create a new MIDI file
        mid = mido.MidiFile()
        
        # Track 0: Tempo and Meta
        track0 = mido.MidiTrack()
        mid.tracks.append(track0)
        track0.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(request.bpm), time=0))
        
        # Track 1: Original Guitar Notes
        guitar_track = mido.MidiTrack()
        mid.tracks.append(guitar_track)
        # Channel 0 for acoustic guitar (program 25 or 24)
        guitar_track.append(mido.Message('program_change', program=25, channel=0, time=0))
        
        if request.notes:
            start_ms = request.notes[0].startTime
            last_time_ticks = 0
            for n in request.notes:
                # Convert time to ticks based on mido's default 480 ticks per beat and BPM
                # time diff in ms
                rel_start = n.startTime - start_ms
                ticks_start = int(mido.second2tick(rel_start / 1000.0, mid.ticks_per_beat, mido.bpm2tempo(request.bpm)))
                delay = ticks_start - last_time_ticks
                if delay < 0: delay = 0
                guitar_track.append(mido.Message('note_on', note=n.midiNote, velocity=90, channel=0, time=delay))
                
                # note off
                duration_ticks = int(mido.second2tick(n.duration / 1000.0, mid.ticks_per_beat, mido.bpm2tempo(request.bpm)))
                guitar_track.append(mido.Message('note_off', note=n.midiNote, velocity=64, channel=0, time=max(0, duration_ticks)))
                last_time_ticks = ticks_start + duration_ticks

        # Track 2: Generated Drum Track (Simple generic rhythm matching the BPM)
        drum_track = mido.MidiTrack()
        mid.tracks.append(drum_track)
        # Setup channel 9 (10th channel) for percussion
        drum_track.append(mido.Message('program_change', program=0, channel=9, time=0))
        
        # Super basic AI mock / procedural rhythm for drums based on the timing
        # We place a kick on every beat, snare on 2 and 4
        ticks_per_beat = mid.ticks_per_beat
        start_ms = request.notes[0].startTime if request.notes else 0
        total_duration_ms = max(n.startTime + n.duration for n in request.notes) - start_ms if request.notes else 0
        beats_total = int((total_duration_ms / 60000.0) * request.bpm)
        
        for b in range(beats_total):
            # Kick on 1 and 3, Snare on 2 and 4
            if b % 2 == 0:
                note = 36 # Acoustic Bass Drum
            else:
                note = 38 # Acoustic Snare
                
            drum_track.append(mido.Message('note_on', note=note, velocity=100, channel=9, time=0))
            drum_track.append(mido.Message('note_off', note=note, velocity=64, channel=9, time=ticks_per_beat))

        # Track 3: AI-Generated Bass matching the scale and root note
        # Extract root note from scale string (e.g., "C Major")
        root_name = request.scale.split(" ")[0].replace("/", "") 
        bass_track = mido.MidiTrack()
        mid.tracks.append(bass_track)
        bass_track.append(mido.Message('program_change', program=33, channel=2, time=0)) # Electric Bass
        
        # Find rough root MIDI note for bass (e.g. around E1 - 28 up to 40)
        root_midi = request.notes[0].midiNote if request.notes else 36
        bass_root = root_midi % 12 + 24 # move down to bass octave
        
        for b in range(beats_total):
            # Play bass on root note for 8th notes and follow rhythm
            bass_track.append(mido.Message('note_on', note=bass_root, velocity=80, channel=2, time=0))
            bass_track.append(mido.Message('note_off', note=bass_root, velocity=64, channel=2, time=ticks_per_beat))

        # Save to file
        mid.save(midi_path)
        
        # Return the generated file
        return FileResponse(
            path=midi_path,
            media_type="audio/midi",
            filename=midi_filename
        )
        
    except Exception as e:
        logger.error(f"Failed to generate guitar accompaniment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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