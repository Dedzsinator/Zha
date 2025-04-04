import React from 'react';
import { Form } from 'react-bootstrap';

function InstrumentSelector({ value, onChange }) {
  return (
    <Form.Group>
      <Form.Label>Select Instrument</Form.Label>
      <Form.Select value={value} onChange={(e) => onChange(e.target.value)}>
        <option value="piano">Piano</option>
        <option value="violin">Violin</option>
        <option value="guitar">Guitar</option>
        <option value="flute">Flute</option>
      </Form.Select>
    </Form.Group>
  );
}

export default InstrumentSelector;