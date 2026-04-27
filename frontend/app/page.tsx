'use client';

import React, { useState } from 'react';
import MidiPlayer from '../components/features/MidiPlayer';
import MidiDeviceManager from '../components/features/MidiDeviceManager';
import GenerationForm from '../components/features/GenerationForm';
import LiveGuitarTracker from '../components/features/LiveGuitarTracker';
import { Midi } from '@tonejs/midi';

export default function Home() {
  const [midiData, setMidiData] = useState<Midi | null>(null);
  const [selectedChannel, setSelectedChannel] = useState(1);

  const handleMidiGenerated = async (file: File) => {
    try {
      // Convert File to Midi object
      const arrayBuffer = await file.arrayBuffer();
      const midi = new Midi(arrayBuffer);
      setMidiData(midi);
    } catch (error) {
      console.error('Error loading generated MIDI:', error);
    }
  };

  const handleDevicesUpdate = () => {
    // Store device information if needed for future use
  };

  const handleTimeUpdate = () => {
    // Handle time updates if needed for future features
  };

  return (
    <div className="min-h-screen bg-gray-100 p-4">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold text-center mb-8">MIDI Player & Editor</h1>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Left Panel - Device Manager and Generation */}
          <div className="lg:col-span-1 space-y-6">
            <LiveGuitarTracker onMidiGenerated={handleMidiGenerated} />
            <MidiDeviceManager
              onDevicesUpdate={handleDevicesUpdate}
              selectedChannel={selectedChannel}
              onChannelChange={setSelectedChannel}
            />
            <GenerationForm
              onMidiGenerated={handleMidiGenerated}
            />
          </div>

          {/* Right Panel - MIDI Player */}
          <div className="lg:col-span-3 bg-gray-800 rounded-lg p-4">
            <MidiPlayer
              midiData={midiData}
              onTimeUpdate={handleTimeUpdate}
              selectedChannel={selectedChannel}
              autoPlay={true}
            />
          </div>
        </div>
      </div>
    </div>
  );
}