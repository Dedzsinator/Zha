import React from 'react';
import { Form, Row, Col } from 'react-bootstrap';

function ParameterControls({ parameters, onChange, activeModel }) {
  return (
    <Form>
      {/* Instrument selector - shown for all models */}
      <Form.Group className="mb-3">
        <Form.Label>Instrument</Form.Label>
        <Form.Select 
          value={parameters.instrument}
          onChange={(e) => onChange('instrument', e.target.value)}>
          <option value="piano">Piano</option>
          <option value="violin">Violin</option>
          <option value="guitar">Guitar</option>
          <option value="flute">Flute</option>
        </Form.Select>
      </Form.Group>

      {/* Duration - shown for all models */}
      <Form.Group className="mb-3">
        <Form.Label>Duration (seconds): {parameters.duration}</Form.Label>
        <Form.Range 
          min={5} 
          max={60} 
          value={parameters.duration}
          onChange={(e) => onChange('duration', parseInt(e.target.value))}
        />
      </Form.Group>

      {/* Creativity - for VAE and combined models */}
      {(activeModel === 'vae' || activeModel === 'combined') && (
        <Form.Group className="mb-3">
          <Form.Label>Creativity: {parameters.creativity.toFixed(2)}</Form.Label>
          <Form.Range 
            min={0} 
            max={1} 
            step={0.05}
            value={parameters.creativity}
            onChange={(e) => onChange('creativity', parseFloat(e.target.value))}
          />
        </Form.Group>
      )}

      {/* Temperature - for transformer model */}
      {activeModel === 'transformer' && (
        <Form.Group className="mb-3">
          <Form.Label>Temperature: {parameters.temperature.toFixed(2)}</Form.Label>
          <Form.Range 
            min={0.1} 
            max={2} 
            step={0.05}
            value={parameters.temperature}
            onChange={(e) => onChange('temperature', parseFloat(e.target.value))}
          />
        </Form.Group>
      )}

      {/* Steps - for diffusion model */}
      {activeModel === 'diffusion' && (
        <Form.Group className="mb-3">
          <Form.Label>Diffusion Steps: {parameters.steps}</Form.Label>
          <Form.Range 
            min={10} 
            max={200} 
            step={10}
            value={parameters.steps}
            onChange={(e) => onChange('steps', parseInt(e.target.value))}
          />
        </Form.Group>
      )}

      {/* Markov specific parameters */}
      {activeModel === 'markov' && (
        <>
          <Form.Group className="mb-3">
            <Form.Label>Start Note: {parameters.startNote}</Form.Label>
            <Form.Range 
              min={21} 
              max={108} 
              value={parameters.startNote}
              onChange={(e) => onChange('startNote', parseInt(e.target.value))}
            />
            <Form.Text className="text-muted">
              {getNoteNameFromMidi(parameters.startNote)}
            </Form.Text>
          </Form.Group>

          <Form.Group className="mb-3">
            <Form.Label>Sequence Length: {parameters.length}</Form.Label>
            <Form.Range 
              min={16} 
              max={128} 
              step={8}
              value={parameters.length}
              onChange={(e) => onChange('length', parseInt(e.target.value))}
            />
          </Form.Group>
        </>
      )}
    </Form>
  );
}

// Helper function to convert MIDI note numbers to note names
function getNoteNameFromMidi(midiNote) {
  const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  const noteName = noteNames[midiNote % 12];
  const octave = Math.floor(midiNote / 12) - 1;
  return `${noteName}${octave}`;
}

export default ParameterControls;