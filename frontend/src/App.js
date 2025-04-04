import React, { useState } from 'react';
import { Container, Row, Col, Card, Form, Nav } from 'react-bootstrap';
import MusicPlayer from './components/MusicPlayer';
import MIDIRecorder from './components/MIDIRecorder';
import ModelSelector from './components/ModelSelector';
import FileUploader from './components/FileUploader';
import ParameterControls from './components/ParameterControls';

function App() {
  const [activeModel, setActiveModel] = useState('combined');
  const [generatedMusic, setGeneratedMusic] = useState(null);
  const [midiFile, setMidiFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [parameters, setParameters] = useState({
    instrument: 'piano',
    creativity: 0.5,
    temperature: 1.0,
    duration: 30,
    steps: 100,
    startNote: 60,
    length: 64
  });

  const handleParameterChange = (name, value) => {
    setParameters(prev => ({ ...prev, [name]: value }));
  };

  const handleModelSelect = (model) => {
    setActiveModel(model);
    setError(null);
  };

  const handleFileUpload = (file) => {
    setMidiFile(file);
    setError(null);
  };

  const handleGenerate = async () => {
    if (needsMidiFile(activeModel) && !midiFile) {
      setError('Please upload a MIDI file first');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      
      // Add common parameters
      formData.append('instrument', parameters.instrument);
      formData.append('duration', parameters.duration);

      // Add model-specific parameters
      switch (activeModel) {
        case 'combined':
          formData.append('midi_file', midiFile);
          formData.append('creativity', parameters.creativity);
          break;
        case 'vae':
          formData.append('midi_file', midiFile);
          formData.append('creativity', parameters.creativity);
          break;
        case 'transformer':
          formData.append('midi_file', midiFile);
          formData.append('temperature', parameters.temperature);
          break;
        case 'diffusion':
          if (midiFile) {
            formData.append('midi_file', midiFile);
          }
          formData.append('steps', parameters.steps);
          break;
        case 'markov':
          formData.append('start_note', parameters.startNote);
          formData.append('length', parameters.length);
          break;
        default:
          break;
      }

      const response = await fetch(`http://localhost:8000/generate/${activeModel}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to generate music');
      }

      const data = await response.json();
      setGeneratedMusic(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const needsMidiFile = (model) => {
    return ['combined', 'vae', 'transformer'].includes(model);
  };

  return (
    <Container className="py-4">
      <Card className="mb-4 shadow-sm">
        <Card.Header as="h1" className="text-center bg-dark text-white">Zha AI Music Generator</Card.Header>
        <Card.Body>
          <Row className="mb-4">
            <Col>
              <ModelSelector activeModel={activeModel} onSelect={handleModelSelect} />
            </Col>
          </Row>
          
          <Row className="mb-4">
            <Col md={6}>
              {needsMidiFile(activeModel) ? (
                <div className="mb-3">
                  <h4>Upload MIDI Input</h4>
                  <FileUploader onFileUpload={handleFileUpload} />
                  {midiFile && <p className="text-success">File uploaded: {midiFile.name}</p>}
                </div>
              ) : (
                activeModel === 'markov' && (
                  <div className="mb-3">
                    <h4>Markov Chain Parameters</h4>
                    <p className="text-muted">No MIDI file needed for Markov generation</p>
                  </div>
                )
              )}
            </Col>
            <Col md={6}>
              <h4>Generation Parameters</h4>
              <ParameterControls 
                parameters={parameters} 
                onChange={handleParameterChange}
                activeModel={activeModel}
              />
            </Col>
          </Row>

          <Row className="mb-3">
            <Col className="text-center">
              <button 
                className="btn btn-primary btn-lg" 
                onClick={handleGenerate}
                disabled={isLoading || (needsMidiFile(activeModel) && !midiFile)}>
                {isLoading ? 'Generating...' : 'Generate Music'}
              </button>
              {error && <div className="text-danger mt-2">{error}</div>}
            </Col>
          </Row>
          
          {generatedMusic && (
            <Row className="mt-4">
              <Col>
                <MusicPlayer 
                  midiUrl={`http://localhost:8000${generatedMusic.midi_url}`}
                  audioUrl={`http://localhost:8000${generatedMusic.audio_url}`}
                  message={generatedMusic.message}
                />
              </Col>
            </Row>
          )}
        </Card.Body>
      </Card>
    </Container>
  );
}

export default App;