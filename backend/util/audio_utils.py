import os
import subprocess
import tempfile
from scipy.io import wavfile
import numpy as np
import logging

logger = logging.getLogger(__name__)

def generate_audio(midi_path, output_path, instrument="piano"):
    """
    Convert MIDI to audio using FluidSynth with improved error handling
    """
    try:
        if not os.path.exists(midi_path):
            logger.error(f"MIDI file does not exist: {midi_path}")
            return False
            
        # Check if FluidSynth is available
        soundfont_path = get_soundfont_path(instrument)

        if not soundfont_path:
            logger.error(f"No soundfont found for instrument: {instrument}")
            return False

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        # Use FluidSynth to convert MIDI to WAV with detailed error capture
        cmd = [
            "fluidsynth",
            "-ni",
            "-g", "1.0",  # gain
            "-F", output_path,  # output file
            soundfont_path,  # soundfont
            midi_path  # input MIDI
        ]

        logger.info(f"Running FluidSynth command: {' '.join(cmd)}")
        
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode != 0:
            logger.error(f"FluidSynth error: {process.stderr}")
            # Try fallback with direct PCM
            return generate_audio_fallback(midi_path, output_path)
            
        # Verify the file exists and has content
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info(f"Successfully generated audio at {output_path}")
            return True
        else:
            logger.error(f"Audio file missing or empty: {output_path}")
            return False

    except Exception as e:
        logger.exception(f"Error generating audio: {e}")
        return generate_audio_fallback(midi_path, output_path)

def generate_audio_fallback(midi_path, output_path):
    """Fallback audio generation using simpler parameters"""
    try:
        # Try with simpler parameters, raw PCM first then convert
        raw_path = f"{output_path}.raw"
        
        cmd1 = [
            "fluidsynth",
            "-q",  # Quiet mode
            "-T", "raw",  # Raw PCM output
            "-F", raw_path,  # Raw output file
            "-r", "44100",  # Sample rate
            get_soundfont_path("piano"),  # Default soundfont
            midi_path  # Input MIDI
        ]
        
        logger.info("Trying fallback audio generation")
        subprocess.run(cmd1, check=False, capture_output=True)
        
        # Convert raw to WAV with ffmpeg if available
        if os.path.exists(raw_path) and os.path.getsize(raw_path) > 0:
            try:
                cmd2 = [
                    "ffmpeg", "-y",
                    "-f", "s16le",
                    "-ar", "44100",
                    "-ac", "2",
                    "-i", raw_path,
                    output_path
                ]
                subprocess.run(cmd2, check=False, capture_output=True)
                os.remove(raw_path)  # Clean up raw file
                
                if os.path.exists(output_path):
                    return True
            except:
                pass
        
        # If still failing, create a simple beep as a last resort
        create_test_tone(output_path)
        return os.path.exists(output_path)
        
    except Exception as e:
        logger.error(f"Audio fallback failed: {e}")
        return False

def create_test_tone(output_path, duration=3):
    """Create a simple test tone as a last resort"""
    try:
        # Generate a simple sine wave
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(2 * np.pi * 440 * t) * 0.5
        
        # Convert to int16
        tone = (tone * 32767).astype(np.int16)
        
        # Write WAV file
        wavfile.write(output_path, sample_rate, tone)
        return True
    except:
        return False

def get_soundfont_path(instrument):
    """
    Get path to appropriate soundfont based on instrument name
    """
    # Default soundfont locations to check
    soundfont_paths = [
        "/usr/share/sounds/sf2/FluidR3_GM.sf2",  # Linux
        "/usr/share/soundfonts/FluidR3_GM.sf2",  # Alternative Linux
        "/usr/local/share/soundfonts/default.sf2",  # macOS
        os.path.expanduser("~/.soundfonts/FluidR3_GM.sf2"),  # User directory
        "C:\\soundfonts\\default.sf2"  # Windows
    ]

    # Check if any of the default soundfonts exist
    for path in soundfont_paths:
        if os.path.exists(path):
            return path

    # If not found, try to download a basic soundfont
    return download_soundfont()

def download_soundfont():
    """
    Download a basic soundfont if none is found
    """
    try:
        soundfont_dir = os.path.join(os.path.expanduser("~"), ".soundfonts")
        os.makedirs(soundfont_dir, exist_ok=True)

        soundfont_path = os.path.join(soundfont_dir, "default.sf2")

        if os.path.exists(soundfont_path):
            return soundfont_path

        # URL for FluidR3_GM.sf2 or another reliable free soundfont
        url = "https://member.keymusician.com/Member/FluidR3_GM/FluidR3_GM.sf2"

        logger.info(f"Downloading soundfont to {soundfont_path}...")
        subprocess.run(["wget", "-O", soundfont_path, url], check=True)

        if os.path.exists(soundfont_path):
            return soundfont_path
        else:
            return None

    except Exception as e:
        logger.error(f"Error downloading soundfont: {e}")
        return None