import React, { useState } from 'react';
import LooperDuet from './components/LooperDuet';
import MusicPlayer from './components/MusicPlayer';
import InstrumentSelector from './components/InstrumentSelector';

function App() {
  const [generatedMusic, setGeneratedMusic] = useState(null);

  const handleGenerate = async (midiNotes, instrument) => {
    const response = await fetch('http://localhost:8000/looper-duet/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        midi_notes: midiNotes,
        instrument: instrument,
      }),
    });
    const data = await response.blob();
    setGeneratedMusic(URL.createObjectURL(data));
  };

  return (
    <div className="App">
      <h1>AI Music Generator</h1>
      <LooperDuet onGenerate={handleGenerate} />
      {generatedMusic && <MusicPlayer audioUrl={generatedMusic} />}
      <InstrumentSelector />
    </div>
  );
}

export default App;