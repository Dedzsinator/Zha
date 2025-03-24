import mido

def parse_midi(midi_file):
    """
    Parse a MIDI file and extract notes.
    """
    midi = mido.MidiFile(midi_file)
    notes = []
    for msg in midi:
        if msg.type == 'note_on':
            notes.append((msg.note, msg.velocity, msg.time))
    return notes

def save_midi(notes, output_path):
    """
    Save a list of notes as a MIDI file.
    """
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    midi.tracks.append(track)
    
    for note, velocity, time in notes:
        track.append(mido.Message('note_on', note=note, velocity=velocity, time=time))
    
    midi.save(output_path)