import numpy as np
import torch
from mido import MidiFile, MidiTrack, Message, MetaMessage
import pretty_midi

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