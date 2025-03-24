import React, { useState } from 'react';

function LooperDuet({ onGenerate }) {
  const [midiNotes, setMidiNotes] = useState([]);

  const handleRecord = () => {
    // Start recording MIDI notes
    MIDIjs.startRecording();
  };

  const handleStop = () => {
    // Stop recording and get MIDI notes
    const notes = MIDIjs.stopRecording();
    setMidiNotes(notes);
  };

  const handleGenerate = (instrument) => {
    onGenerate(midiNotes, instrument);
  };

  return (
    <div>
      <button onClick={handleRecord}>Record MIDI</button>
      <button onClick={handleStop}>Stop Recording</button>
      <button onClick={() => handleGenerate('piano')}>Generate Piano</button>
      <button onClick={() => handleGenerate('guitar')}>Generate Guitar</button>
    </div>
  );
}

export default LooperDuet;