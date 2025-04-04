import React from 'react';
import { Nav, Card } from 'react-bootstrap';

function ModelSelector({ activeModel, onSelect }) {
  const models = [
    { id: 'combined', name: 'Combined Models', description: 'Uses all models together for best results' },
    { id: 'markov', name: 'Markov Chain', description: 'Music theory structure (no MIDI input needed)' },
    { id: 'vae', name: 'VAE', description: 'Creative variations on input' },
    { id: 'transformer', name: 'Transformer', description: 'Coherent sequence structure' },
    { id: 'diffusion', name: 'Diffusion', description: 'High-quality synthesis (optional MIDI input)' }
  ];

  return (
    <Card className="mb-4">
      <Card.Header>
        <h4>Select Generation Model</h4>
      </Card.Header>
      <Card.Body>
        <Nav variant="pills" className="flex-column flex-md-row">
          {models.map(model => (
            <Nav.Item key={model.id} className="mb-2 me-md-2">
              <Nav.Link 
                active={activeModel === model.id}
                onClick={() => onSelect(model.id)}
                className="text-center">
                {model.name}
              </Nav.Link>
            </Nav.Item>
          ))}
        </Nav>
        <div className="mt-3">
          <h5>About {models.find(m => m.id === activeModel).name}</h5>
          <p>{models.find(m => m.id === activeModel).description}</p>
        </div>
      </Card.Body>
    </Card>
  );
}

export default ModelSelector;