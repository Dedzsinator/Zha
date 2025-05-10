'use client';

import { useState } from 'react';
import { Midi } from '@tonejs/midi';
import { GenerationResponse } from '@/types';
import GenerationForm from '@/components/features/GenerationForm';
import PianoRoll from '@/components/features/PianoRoll';
import MidiPlayer from '@/components/features/MidiPlayer';
import Slider from '@/components/ui/Slider';
import { useAudioPlayback } from '@/hooks/useAudioPlayback';
import { formatTime } from '@/lib/midi-utils';

export default function Home() {
  const [midiData, setMidiData] = useState<Midi | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [generationResponse, setGenerationResponse] = useState<GenerationResponse | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  // Set up audio playback
  const {
    playing,
    loading: audioLoading,
    currentTime: audioTime,
    duration,
    controls: audioControls
  } = useAudioPlayback(audioUrl);

  // Update audio URL when generation response changes
  useState(() => {
    if (generationResponse?.audio_url) {
      setAudioUrl(`http://localhost:8000${generationResponse.audio_url}`);
    }
  });

  const handleMidiGenerated = async (file: File, response: GenerationResponse) => {
    setGenerationResponse(response);

    // Read the MIDI file and parse it
    const reader = new FileReader();
    reader.onload = async (e) => {
      try {
        if (e.target?.result instanceof ArrayBuffer) {
          const midi = new Midi(e.target.result);
          setMidiData(midi);
        }
      } catch (err) {
        console.error('Error parsing MIDI file:', err);
        alert('Error parsing MIDI file. It may be corrupted.');
      }
    };
    reader.readAsArrayBuffer(file);

    // Set audio URL
    if (response.audio_url) {
      setAudioUrl(`http://localhost:8000${response.audio_url}`);
    } else {
      setAudioUrl(null);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = async (e) => {
        try {
          if (e.target?.result instanceof ArrayBuffer) {
            const midi = new Midi(e.target.result);
            setMidiData(midi);
            setGenerationResponse(null);
            setAudioUrl(null);
          }
        } catch (err) {
          console.error('Error parsing MIDI file:', err);
          alert('Error parsing MIDI file. It may be corrupted.');
        }
      };
      reader.readAsArrayBuffer(file);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold text-center mb-8 bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent">
        Zha Music Generator
      </h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1">
          <GenerationForm onMidiGenerated={handleMidiGenerated} />
        </div>

        <div className="lg:col-span-2 space-y-6">
          {/* File Upload Section */}
          <div className="bg-white p-6 rounded-lg border border-gray-200">
            <h3 className="text-xl font-semibold mb-4">Upload MIDI File</h3>
            <input
              type="file"
              accept=".mid,.midi"
              onChange={handleFileUpload}
              className="w-full p-2 text-sm text-gray-500
                file:mr-4 file:py-2 file:px-4
                file:rounded-md file:border-0
                file:text-sm file:font-semibold
                file:bg-blue-50 file:text-blue-700
                hover:file:bg-blue-100"
            />
          </div>

          {/* Generated Music Info */}
          {generationResponse && (
            <div className="bg-white p-6 rounded-lg border border-gray-200">
              <h3 className="text-xl font-semibold mb-4">Generation Info</h3>
              <div className="text-sm text-gray-700">
                {generationResponse.message && (
                  <p className="mb-2">{generationResponse.message}</p>
                )}

                {/* Audio Player */}
                {audioUrl && (
                  <div className="mt-6 space-y-2">
                    <h4 className="text-lg font-medium">Generated Audio</h4>

                    {/* Player Controls */}
                    <div className="flex items-center space-x-2 mb-2">
                      <button
                        onClick={playing ? audioControls.pause : audioControls.play}
                        className="p-2 rounded-full bg-blue-100 hover:bg-blue-200 text-blue-600"
                      >
                        {playing ? (
                          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                            <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
                          </svg>
                        ) : (
                          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                            <path d="M8 5v14l11-7z" />
                          </svg>
                        )}
                      </button>

                      <button
                        onClick={audioControls.stop}
                        className="p-2 rounded-full bg-blue-100 hover:bg-blue-200 text-blue-600"
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                          <path d="M6 6h12v12H6z" />
                        </svg>
                      </button>

                      <div className="flex-grow">
                        <Slider
                          min={0}
                          max={duration || 100}
                          value={audioTime}
                          onChange={audioControls.seek}
                          disabled={audioLoading || !audioUrl}
                          color="blue"
                        />
                      </div>

                      <div className="text-sm font-mono">
                        {formatTime(audioTime)} / {formatTime(duration)}
                      </div>
                    </div>

                    {/* Volume Control */}
                    <div className="flex items-center space-x-2">
                      <button
                        onClick={audioControls.toggleMute}
                        className="p-1 text-gray-600"
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                          <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z" />
                        </svg>
                      </button>

                      <div className="w-20">
                        <Slider
                          min={0}
                          max={1}
                          step={0.01}
                          value={0.8}
                          onChange={audioControls.setVolume}
                          disabled={audioLoading || !audioUrl}
                          color="blue"
                        />
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Piano Roll Visualization */}
          <div className="bg-white p-6 rounded-lg border border-gray-200">
            <h3 className="text-xl font-semibold mb-4">Piano Roll</h3>
            {midiData ? (
              <>
                <MidiPlayer
                  midiData={midiData}
                  onTimeUpdate={setCurrentTime}
                />
                <PianoRoll
                  midiData={midiData}
                  currentTime={currentTime}
                  pixelsPerSecond={100}
                />
              </>
            ) : (
              <p className="text-center py-12 text-gray-500 italic">
                No MIDI data loaded. Please generate or upload a MIDI file.
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}