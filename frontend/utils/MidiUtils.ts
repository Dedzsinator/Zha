import { Midi } from '@tonejs/midi';

// WebMidi types for TypeScript
interface MidiNote {
  midi: number;
  name: string;
  time: number;
  duration: number;
  velocity: number;
}

// Create a MIDI file from recorded notes
export function createMidiFromNotes(
  notes: { note: number; velocity: number; time: number; duration: number }[],
  bpm: number = 120,
  timeSignature: [number, number] = [4, 4],
  name: string = 'Recorded MIDI',
  channel: number = 0
): Midi {
  const midi = new Midi();
  
  // Set header information
  midi.header.setTempo(bpm);
  midi.header.timeSignatures.push({
    ticks: 0,
    timeSignature: timeSignature,
    measures: 0
  });
  midi.header.name = name;
  
  // Create track
  const track = midi.addTrack();
  track.channel = channel;
  
  // Add all notes
  notes.forEach(note => {
    track.addNote({
      midi: note.note,
      time: note.time,
      duration: note.duration,
      velocity: note.velocity
    });
  });
  
  return midi;
}

// Download a MIDI file
export function downloadMidi(midi: Midi, filename: string = 'download.mid'): void {
  // Convert to array buffer
  const arrayBuffer = midi.toArray();
  
  // Create Blob
  const blob = new Blob([arrayBuffer], { type: 'audio/midi' });
  
  // Create download link
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  
  // Trigger download
  document.body.appendChild(a);
  a.click();
  
  // Clean up
  setTimeout(() => {
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, 100);
}

// Simple MIDI note analysis
export function analyzeMidiNotes(notes: MidiNote[]) {
  // Calculate pitch class distribution (0=C, 1=C#, etc.)
  const pitchClasses = Array(12).fill(0);
  
  // Track time range
  let minTime = Infinity;
  let maxTime = -Infinity;
  
  // Analyze notes
  notes.forEach(note => {
    // Update pitch class histogram
    pitchClasses[note.midi % 12]++;
    
    // Update time range
    minTime = Math.min(minTime, note.time);
    maxTime = Math.max(maxTime, note.time + note.duration);
  });
  
  // Find dominant pitch class (simple approximation of key)
  let dominantPC = 0;
  let maxCount = 0;
  pitchClasses.forEach((count, pc) => {
    if (count > maxCount) {
      maxCount = count;
      dominantPC = pc;
    }
  });
  
  // Map pitch class to note name
  const pcToName = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  
  return {
    noteCount: notes.length,
    duration: maxTime - minTime,
    pitchClasses,
    dominantPitchClass: dominantPC,
    approximateKey: pcToName[dominantPC]
  };
}

// Generate simple response notes based on input (for duet mode)
export function generateSimpleResponse(
  inputNotes: MidiNote[],
  responseStyle: 'simple' | 'chord' | 'melody' = 'simple'
): MidiNote[] {
  if (inputNotes.length === 0) return [];
  
  const analysis = analyzeMidiNotes(inputNotes);
  const dominantPC = analysis.dominantPitchClass;
  const response: MidiNote[] = [];
  
  // Get the last input note as reference
  const lastNote = inputNotes[inputNotes.length - 1];
  const lastTime = lastNote.time + lastNote.duration;
  
  // Different response based on style
  switch (responseStyle) {
    case 'simple':
      // Simple intervals (third and fifth)
      [3, 7].forEach(interval => {
        const responseMidi = ((lastNote.midi + interval) % 12) + (Math.floor(lastNote.midi / 12) * 12);
        response.push({
          midi: responseMidi,
          name: '', // Will be filled by Tone.js
          time: lastTime + 0.1,
          duration: 0.5,
          velocity: 0.7
        });
      });
      break;
      
    case 'chord':
      // Generate a chord based on dominant pitch class
      // Major chord: root, major third (4 semitones), perfect fifth (7 semitones)
      // Minor chord: root, minor third (3 semitones), perfect fifth (7 semitones)
      const isMinor = [1, 3, 6, 8, 10].includes(dominantPC); // Simple heuristic
      const intervals = isMinor ? [0, 3, 7] : [0, 4, 7];
      
      intervals.forEach(interval => {
        const chordNote = ((lastNote.midi + interval) % 12) + (Math.floor(lastNote.midi / 12) * 12);
        response.push({
          midi: chordNote,
          name: '', // Will be filled by Tone.js
          time: lastTime + 0.1,
          duration: 1.0,
          velocity: 0.7
        });
      });
      break;
      
    case 'melody':
      // Generate a short melodic response
      // Analyze input rhythm pattern
      const rhythmPattern = inputNotes.map(n => n.duration);
      const averageDuration = rhythmPattern.reduce((sum, d) => sum + d, 0) / rhythmPattern.length;
      
      // Create a simple response melody
      const melodyIntervals = [-2, 0, 3, 5, 7];
      let currentTime = lastTime + 0.1;
      
      for (let i = 0; i < 4; i++) {
        const interval = melodyIntervals[Math.floor(Math.random() * melodyIntervals.length)];
        const noteMidi = ((lastNote.midi + interval) % 12) + (Math.floor(lastNote.midi / 12) * 12);
        const noteDuration = averageDuration * (0.5 + Math.random());
        
        response.push({
          midi: noteMidi,
          name: '', // Will be filled by Tone.js
          time: currentTime,
          duration: noteDuration,
          velocity: 0.6 + Math.random() * 0.3
        });
        
        currentTime += noteDuration;
      }
      break;
  }
  
  return response;
}