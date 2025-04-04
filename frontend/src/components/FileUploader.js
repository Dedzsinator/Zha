import React, { useRef } from 'react';
import { Button, Form } from 'react-bootstrap';

function FileUploader({ onFileUpload }) {
  const fileInputRef = useRef();

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file && file.name.endsWith('.mid')) {
      onFileUpload(file);
    }
  };

  const handleButtonClick = () => {
    fileInputRef.current.click();
  };

  return (
    <div>
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept=".mid,.midi"
        style={{ display: 'none' }}
      />
      <Button 
        onClick={handleButtonClick}
        variant="outline-primary"
        className="w-100">
        Select MIDI File
      </Button>
      <Form.Text className="text-muted">
        Select a .mid file to use as input for generation
      </Form.Text>
    </div>
  );
}

export default FileUploader;