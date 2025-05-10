import { useState } from 'react';
import axios, { isAxiosError } from 'axios';
import { Midi } from '@tonejs/midi';
import { GenerationResponse } from '@/types';

export function useMidiGeneration() {
  const [midiData, setMidiData] = useState<Midi | null>(null);
  const [generationResponse, setGenerationResponse] = useState<GenerationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const generateMidi = async (model: string, params: FormData) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post<GenerationResponse>(
        `http://localhost:8000/generate/${model}`,
        params,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      );
      
      setGenerationResponse(response.data);
      
      // Process response based on model type
      if (response.data.midi_url || response.data.midi_file) {
        const url = response.data.midi_url 
          ? `http://localhost:8000${response.data.midi_url}`
          : `http://localhost:8000/download/midi/${response.data.midi_file}`;
        
        const midiResponse = await axios.get(url, { responseType: 'blob' });
        const midiFile = new File(
          [midiResponse.data],
          (response.data.midi_url?.split('/').pop() || response.data.midi_file || 'output.mid'),
          { type: 'audio/midi' }
        );
        
        loadMidiFile(midiFile);
      }
      
    } catch (err: unknown) {
      if (isAxiosError(err)) {
        setError(err.response?.data?.detail || err.message || 'An error occurred');
      } else if (err instanceof Error) {
        setError(err.message || 'An error occurred');
      } else {
        setError('An unknown error occurred');
      }
      console.error(err);
    } finally {
      setLoading(false);
    }
  };
  
  const loadMidiFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        if (e.target?.result instanceof ArrayBuffer) {
          const midi = new Midi(e.target.result);
          setMidiData(midi);
        }
      } catch (err) {
        console.error('Error parsing MIDI file:', err);
        setError('Error parsing MIDI file. It may be corrupted.');
      }
    };
    reader.readAsArrayBuffer(file);
  };

  return {
    midiData,
    generationResponse,
    loading,
    error,
    generateMidi,
    loadMidiFile
  };
}