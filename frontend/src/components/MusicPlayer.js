import React, { useState } from 'react';
import { Card, Button, Row, Col, Alert } from 'react-bootstrap';

function MusicPlayer({ midiUrl, audioUrl, message }) {
  const [isPlaying, setIsPlaying] = useState(false);

  return (
    <Card className="shadow-sm">
      <Card.Header as="h4" className="bg-success text-white">Generated Music</Card.Header>
      <Card.Body>
        <Alert variant="success">{message}</Alert>
        
        <Row className="mb-3">
          <Col>
            <h5>Listen to Audio</h5>
            <audio 
              controls 
              className="w-100" 
              src={audioUrl}
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
              onEnded={() => setIsPlaying(false)}
            />
          </Col>
        </Row>

        <Row>
          <Col md={6} className="mb-2">
            <Button 
              href={midiUrl} 
              variant="primary" 
              className="w-100"
              download="generated_music.mid">
              Download MIDI File
            </Button>
          </Col>
          <Col md={6}>
            <Button 
              href={audioUrl} 
              variant="success" 
              className="w-100"
              download="generated_music.wav">
              Download WAV Audio
            </Button>
          </Col>
        </Row>
      </Card.Body>
    </Card>
  );
}

export default MusicPlayer;