import mido
import numpy as np
from mido import Message, MidiFile, MidiTrack
import os
from pathlib import Path
import uuid
import subprocess
import time
import shutil
import math  # Add this import
import pretty_midi  # Make sure this is imported too
import logging

def parse_midi(midi_path):
    """
    Parse MIDI file into a feature vector suitable for model processing.

    Args:
        midi_path: Path to MIDI file

    Returns:
        numpy array of shape (128,) representing a normalized pitch histogram
    """
    try:
        # Use pretty_midi for more robust MIDI parsing
        midi_data = pretty_midi.PrettyMIDI(midi_path)

        # Create a histogram of note occurrences
        feature = np.zeros(128, dtype=np.float32)

        for instrument in midi_data.instruments:
            for note in instrument.notes:
                # Add note velocity (normalized)
                feature[note.pitch] += note.velocity / 127.0

        # Normalize if sum is greater than 0
        if np.sum(feature) > 0:
            feature = feature / np.sum(feature)

        return feature

    except Exception as e:
        print(f"Error parsing MIDI file {midi_path}: {e}")
        return None

def create_midi_file(feature_vector, output_path, bpm=120, duration_seconds=30):
    """
    Create a MIDI file from a feature vector

    Args:
        feature_vector: Numpy array of shape (128,) representing note probabilities
        output_path: Path to save the MIDI file
        bpm: Beats per minute
        duration_seconds: Length of the generated MIDI in seconds
    """
    try:
        # Normalize if needed
        if np.sum(feature_vector) > 0:
            feature_vector = feature_vector / np.sum(feature_vector)

        # Create PrettyMIDI object
        midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)

        # Create instrument (piano by default)
        piano = pretty_midi.Instrument(program=0)

        # Determine number of notes to generate based on duration
        beats_per_second = bpm / 60
        total_beats = beats_per_second * duration_seconds
        notes_per_beat = 4  # 16th notes
        total_notes = int(total_beats * notes_per_beat)

        # Generate notes based on the feature vector probabilities
        time_step = 60 / bpm / notes_per_beat  # seconds per 16th note

        for i in range(total_notes):
            # Probabilistic note selection
            if np.random.random() < 0.3:  # Control note density
                # Sample from the distribution
                note_index = np.random.choice(128, p=feature_vector)

                # Create note with random duration
                duration = np.random.choice([1, 2, 4]) * time_step  # 16th, 8th, or quarter note
                velocity = np.random.randint(60, 100)

                start_time = i * time_step
                end_time = start_time + duration

                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=note_index,
                    start=start_time,
                    end=end_time
                )

                piano.notes.append(note)

        # Add instrument to MIDI
        midi.instruments.append(piano)

        # Write MIDI file
        midi.write(output_path)
        return True

    except Exception as e:
        print(f"Error creating MIDI file: {e}")
        return False

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
        tempo = 120  # BPM - ensure this is a number, not a string
        
        # Add track name and tempo
        mf.addTrackName(track, time, "Markov Chain Generated Track")
        mf.addTempo(track, time, tempo)
        
        # Parse time signature
        try:
            numerator, denominator = map(int, time_signature.split('/'))
            # Set time signature (numerator, denominator are encoded as bit shifts)
            denominator_bits = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4, 32: 5}.get(denominator, 2)
            mf.addTimeSignature(track, time, numerator, denominator_bits, 24, 8)
        except Exception as e:
            logger.warning(f"Failed to parse time signature {time_signature}: {e}, defaulting to 4/4")
            mf.addTimeSignature(track, time, 4, 2, 24, 8)
        
        # Ensure we have valid notes to add
        if not notes:
            logger.error("No notes provided for MIDI creation")
            return False
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
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
        
        # Write the MIDI file with error handling
        try:
            with open(output_path, 'wb') as outf:
                mf.writeFile(outf)
            
            # Verify file was created
            if not os.path.exists(output_path):
                logger.error(f"MIDI file was not created at {output_path}")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error writing MIDI file: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Error in create_midi_with_durations: {e}")
        return False

def create_midi_with_chords_and_durations(notes, durations, chords=None, bpm=120, output_path=None, instrument=0):
    """
    Create a MIDI file with specific note durations and chord accompaniment.
    
    Args:
        notes: List of MIDI note numbers (0-127)
        durations: List of note durations in beats (quarter notes)
        chords: List of chord names (optional)
        bpm: Tempo in beats per minute
        output_path: Path to save the MIDI file
        instrument: MIDI instrument number (0-127)
        
    Returns:
        Path to the created MIDI file
    """
    if output_path is None:
        output_dir = Path("generated_files/midi")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"musical_markov_{uuid.uuid4().hex[:8]}.mid")
    
    # Create MIDI file with two tracks (melody + chords)
    mid = MidiFile(ticks_per_beat=480)
    melody_track = MidiTrack()
    chord_track = MidiTrack()
    mid.tracks.append(melody_track)
    mid.tracks.append(chord_track)
    
    # Set tempo
    tempo = mido.bpm2tempo(bpm)
    melody_track.append(Message('program_change', program=instrument, time=0))
    chord_track.append(Message('program_change', program=instrument, time=0))
    melody_track.append(mido.MetaMessage('set_tempo', tempo=tempo))
    
    # Add melody notes
    current_time = 0
    melody_events = []
    
    for note, duration in zip(notes, durations):
        if note is not None and 0 <= note <= 127 and duration > 0:
            duration_ticks = int(duration * mid.ticks_per_beat)
            
            # Add note on at current time
            melody_events.append((current_time, 'note_on', note))
            # Add note off after duration
            melody_events.append((current_time + duration_ticks, 'note_off', note))
            
            # Move time forward
            current_time += duration_ticks
    
    # Sort events by time
    melody_events.sort()
    
    # Add events to track with correct delta times
    last_time = 0
    for time, event_type, note in melody_events:
        delta = time - last_time
        if event_type == 'note_on':
            melody_track.append(Message('note_on', note=note, velocity=64, time=delta))
        else:
            melody_track.append(Message('note_off', note=note, velocity=64, time=delta))
        last_time = time
    
    # Add chords if provided
    if chords:
        # Implementation for chord accompaniment
        # This is simplified and would need to be expanded based on how 
        # your chord representation works
        pass
    
    # Save MIDI file
    mid.save(output_path)
    return output_path

def parse_key_context(key_context):
    """Properly parse a key context string into root and mode"""
    if not key_context or not isinstance(key_context, str):
        return "C", "major"
        
    key_str = key_context.strip()
    
    # Split string on whitespace
    parts = key_str.split()
    
    if len(parts) == 1:
        # Only tonic provided, assume major
        return parts[0], "major"
    else:
        # Get the first part as tonic, rest as mode
        tonic = parts[0]
        mode = " ".join(parts[1:]).lower()
        
        # Normalize mode
        if mode not in ['major', 'minor', 'dorian', 'phrygian', 'lydian', 'mixolydian', 'aeolian', 'locrian']:
            mode = 'major'
            
        return tonic, mode