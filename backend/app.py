from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import torch
import librosa
import numpy as np
from backend.models.diffusion import DiffusionModel
from backend.models.transformer import TransformerModel
from backend.models.vae import VAEModel
from util.midi_utils import parse_midi
from util.audio_utils import generate_audio

app = FastAPI()

# Load pretrained models
diffusion_model = DiffusionModel()
transformer_model = TransformerModel()
vae_model = VAEModel()

@app.post("/looper-duet/")
async def looper_duet(midi_file: UploadFile, instrument: str):
    try:
        # Parse MIDI input
        midi_notes = parse_midi(midi_file.file)
        
        # Analyze active loopers (mock data for now)
        loopers = {'tempo': 120, 'key': 'C major', 'rhythm': '4/4'}
        tempo, key, rhythm = loopers['tempo'], loopers['key'], loopers['rhythm']
        
        # Generate complementary music using the Transformer model
        context = (tempo, key, rhythm)
        generated_midi = transformer_model.generate(midi_notes, context, instrument)
        
        # Convert MIDI to audio using the Diffusion model
        generated_audio = generate_audio(diffusion_model, generated_midi)
        
        # Save the generated audio
        output_path = "generated_music.wav"
        librosa.output.write_wav(output_path, generated_audio, 22050)
        
        # Return the generated audio file
        return FileResponse(output_path, media_type='audio/wav', filename='generated_music.wav')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "_main_":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)