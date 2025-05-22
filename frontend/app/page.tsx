'use client';

import { useState, useEffect } from 'react';
import { Midi } from '@tonejs/midi';
import PianoRoll from '@/components/features/PianoRoll';
import MidiPlayer from '@/components/features/MidiPlayer';
import MidiRecorder from '@/components/features/MidiRecorder';
import MidiDeviceManager from '@/components/features/MidiDeviceManager';
import DuetMode from '@/components/features/DuetMode';
import GenerationForm from '@/components/features/GenerationForm';
import { downloadMidi } from '../utils/MidiUtils';
import { FaPlus, FaDownload, FaSave, FaMagic } from 'react-icons/fa';
import { GenerationResponse } from '@/types';

export default function Home() {
  // MIDI data state
  const [midiData, setMidiData] = useState<Midi | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [activeNotes, setActiveNotes] = useState<number[]>([]);

  // MIDI devices state
  const [inputDevices, setInputDevices] = useState<WebMidi.MIDIInput[]>([]);
  const [outputDevices, setOutputDevices] = useState<WebMidi.MIDIOutput[]>([]);
  const [selectedInputId, setSelectedInputId] = useState<string>('');
  const [selectedOutputId, setSelectedOutputId] = useState<string>('');
  const [selectedChannel, setSelectedChannel] = useState<number>(1); // 1-16, 0 = all

  // Recording state
  const [isRecording, setIsRecording] = useState(false);
  const [recordedMidi, setRecordedMidi] = useState<Midi | null>(null);
  const [recordingTime, setRecordingTime] = useState(0);

  // Duet mode settings
  const [duetMode, setDuetMode] = useState(false);
  const [duetStyle, setDuetStyle] = useState<'simple' | 'chord' | 'melody'>('simple');
  const [responsiveness, setResponsiveness] = useState(0.5);

  // Generation state
  const [generationResponse, setGenerationResponse] = useState<GenerationResponse | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  // UI state
  const [activeTab, setActiveTab] = useState<'player' | 'recorder' | 'devices' | 'generator'>('player');

  // Handle MIDI device updates
  const handleDevicesUpdate = (inputs: WebMidi.MIDIInput[], outputs: WebMidi.MIDIOutput[]) => {
    setInputDevices(inputs);
    setOutputDevices(outputs);

    // Auto-select first device if none selected
    if (inputs.length > 0 && !selectedInputId) {
      setSelectedInputId(inputs[0].id);
    }
    if (outputs.length > 0 && !selectedOutputId) {
      setSelectedOutputId(outputs[0].id);
    }
  };

  // Handle active notes update (for visualizing in piano roll)
  const handleActiveNotesChange = (notes: number[]) => {
    setActiveNotes(notes);
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

  // Handle MIDI generation
  const handleMidiGenerated = async (file: File, response: GenerationResponse) => {
    setGenerationResponse(response);

    // Read the MIDI file and parse it
    const reader = new FileReader();
    reader.onload = async (e) => {
      try {
        if (e.target?.result instanceof ArrayBuffer) {
          const midi = new Midi(e.target.result);
          setMidiData(midi);
          // Switch to player tab to show the result
          setActiveTab('player');
        }
      } catch (err) {
        console.error('Error parsing MIDI file:', err);
        alert('Error parsing MIDI file. It may be corrupted.');
      }
    };
    reader.readAsArrayBuffer(file);

    // Set audio URL if available
    if (response.audio_url) {
      setAudioUrl(`http://localhost:8000${response.audio_url}`);
    } else {
      setAudioUrl(null);
    }
  };

  // Recording controls
  const startRecording = () => {
    setIsRecording(true);
    setRecordingTime(0);
    setRecordedMidi(null);

    // Start timer
    const interval = setInterval(() => {
      setRecordingTime(time => time + 0.1);
    }, 100);

    // Store interval ID for cleanup
    // @ts-ignore
    window.recordingInterval = interval;
  };

  const stopRecording = (midi: Midi | null) => {
    setIsRecording(false);
    // @ts-ignore
    clearInterval(window.recordingInterval);

    if (midi) {
      setRecordedMidi(midi);
    }
  };

  const handleSaveRecording = (name: string) => {
    if (recordedMidi) {
      downloadMidi(recordedMidi, `${name || 'recording'}.mid`);
      // Load recording into player
      setMidiData(recordedMidi);
      // Switch to player tab
      setActiveTab('player');
    }
  };

  const handleClearRecording = () => {
    setRecordedMidi(null);
  };

  // Cleanup recording timer
  useEffect(() => {
    return () => {
      if (window.recordingInterval) {
        clearInterval(window.recordingInterval);
      }
    };
  }, []);

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold text-center mb-8 bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent">
        Zha MIDI Studio
      </h1>

      {/* MIDI Device Manager */}
      <MidiDeviceManager
        onDevicesUpdate={handleDevicesUpdate}
        onActiveNotesChange={handleActiveNotesChange}
        onInputDeviceChange={setSelectedInputId}
        onOutputDeviceChange={setSelectedOutputId}
        isRecording={isRecording}
        onRecordingToggle={isRecording ? () => stopRecording(null) : startRecording}
        selectedChannel={selectedChannel}
        onChannelChange={setSelectedChannel}
      />

      {/* Tabs for Player, Recorder, File Management */}
      <div className="mb-4">
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8">
            <button
              className={`${activeTab === 'player'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                } whitespace-nowrap py-2 px-1 border-b-2 font-medium text-sm`}
              onClick={() => setActiveTab('player')}
            >
              MIDI Player
            </button>
            <button
              className={`${activeTab === 'recorder'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                } whitespace-nowrap py-2 px-1 border-b-2 font-medium text-sm`}
              onClick={() => setActiveTab('recorder')}
            >
              MIDI Recorder
            </button>
            <button
              className={`${activeTab === 'generator'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                } whitespace-nowrap py-2 px-1 border-b-2 font-medium text-sm`}
              onClick={() => setActiveTab('generator')}
            >
              <FaMagic className="inline mr-1" /> AI Generator
            </button>
            <button
              className={`${activeTab === 'devices'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                } whitespace-nowrap py-2 px-1 border-b-2 font-medium text-sm`}
              onClick={() => setActiveTab('devices')}
            >
              Device Settings
            </button>
          </nav>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="bg-white p-6 rounded-lg border border-gray-200">
        {/* Player Tab */}
        {activeTab === 'player' && (
          <div>
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold">MIDI Player</h2>

              <div className="flex items-center space-x-2">
                <DuetMode
                  enabled={duetMode}
                  onToggle={setDuetMode}
                  responsiveness={responsiveness}
                  onResponsivenessChange={setResponsiveness}
                  harmonization={duetStyle}
                  onHarmonizationChange={setDuetStyle}
                />

                <input
                  type="file"
                  accept=".mid,.midi"
                  id="upload-midi"
                  className="hidden"
                  onChange={handleFileUpload}
                />
                <label
                  htmlFor="upload-midi"
                  className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 cursor-pointer"
                >
                  <FaPlus className="mr-2 -ml-1" />
                  Load MIDI
                </label>
              </div>
            </div>

            {/* Generation Info (if available) */}
            {generationResponse && (
              <div className="mb-4 p-4 bg-blue-100 rounded-md border border-blue-200">
                <h3 className="font-medium text-blue-900 mb-2">Generated Music Info</h3>
                <p className="text-gray-800 font-medium">
                  {generationResponse.message || "AI-generated music loaded successfully"}
                </p>
                {audioUrl && (
                  <div className="mt-2">
                    <audio src={audioUrl} controls className="w-full mt-2" />
                  </div>
                )}
              </div>
            )}

            {/* MIDI Player Component */}
            {midiData ? (
              <>
                <MidiPlayer
                  midiData={midiData}
                  onTimeUpdate={setCurrentTime}
                  inputDeviceId={selectedInputId}
                  outputDeviceId={selectedOutputId}
                  selectedChannel={selectedChannel}
                  duetMode={duetMode}
                  duetStyle={duetStyle}
                  onActiveNotesChange={handleActiveNotesChange}
                />
                <PianoRoll
                  midiData={midiData}
                  currentTime={currentTime}
                  pixelsPerSecond={100}
                  activeNotes={activeNotes}
                  selectedChannel={selectedChannel}
                />
              </>
            ) : (
              <p className="text-center py-12 text-gray-500 italic">
                No MIDI data loaded. Please upload a MIDI file, record something, or generate music with AI.
              </p>
            )}
          </div>
        )}

        {/* Recorder Tab */}
        {activeTab === 'recorder' && (
          <div>
            <h2 className="text-xl font-semibold mb-4">MIDI Recorder</h2>

            <MidiRecorder
              isRecording={isRecording}
              onStartRecording={startRecording}
              onStopRecording={stopRecording}
              onSave={handleSaveRecording}
              onClear={handleClearRecording}
              inputDeviceId={selectedInputId}
              selectedChannel={selectedChannel}
              recordedMidi={recordedMidi}
              recordingTimeSeconds={recordingTime}
            />

            {recordedMidi && (
              <div className="mt-6">
                <h3 className="text-lg font-medium mb-2">Recording Preview</h3>
                <PianoRoll
                  midiData={recordedMidi}
                  currentTime={0}
                  pixelsPerSecond={100}
                  selectedChannel={selectedChannel}
                />
              </div>
            )}
          </div>
        )}

        {/* Generator Tab */}
        {activeTab === 'generator' && (
          <div>
            <div className="bg-gradient-to-r from-indigo-900 via-purple-800 to-indigo-900 p-6 rounded-lg mb-6 shadow-lg">
              <h2 className="text-2xl font-bold text-white mb-2 flex items-center">
                <FaMagic className="mr-2 text-yellow-300" /> AI Music Generator
              </h2>
              <p className="text-indigo-100 text-sm">
                Use our advanced AI to generate custom MIDI compositions based on your parameters.
                The AI can create melodies, chord progressions, and complete musical pieces in various styles.
              </p>
            </div>

            <div className="bg-gray-50 rounded-lg border border-purple-200 shadow-md p-6">
              <GenerationForm onMidiGenerated={handleMidiGenerated} />
            </div>

            {generationResponse && (
              <div className="mt-6 bg-gradient-to-r from-indigo-100 to-purple-100 rounded-lg border border-purple-200 p-6 shadow-md">
                <h3 className="font-semibold text-lg text-purple-800 mb-3 flex items-center">
                  <FaMagic className="mr-2 text-purple-600" /> Generated Music
                </h3>
                <p className="text-gray-800 mb-4 font-medium">
                  {generationResponse.message || "AI has successfully composed a new piece of music for you!"}
                </p>

                {audioUrl && (
                  <div className="bg-white p-4 rounded-md shadow-sm border border-purple-200">
                    <p className="text-gray-800 mb-2 font-medium">Listen to your AI-generated composition:</p>
                    <audio
                      src={audioUrl}
                      controls
                      className="w-full"
                      style={{
                        boxShadow: "0 3px 8px rgba(76, 29, 149, 0.15)",
                        borderRadius: "8px"
                      }}
                    />
                  </div>
                )}

                <div className="mt-4 flex justify-end">
                  <button
                    onClick={() => setActiveTab('player')}
                    className="px-4 py-2 bg-purple-700 text-white rounded-md hover:bg-purple-800 transition duration-200 flex items-center shadow-md"
                  >
                    <span className="mr-2">View in Piano Roll</span> →
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Devices Tab */}
        {activeTab === 'devices' && (
          <div>
            <h2 className="text-xl font-semibold mb-4">Device Settings</h2>

            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-medium mb-2">Input Devices</h3>
                {inputDevices.length > 0 ? (
                  <div className="bg-gray-50 rounded-md p-4">
                    <ul className="space-y-2">
                      {inputDevices.map(device => (
                        <li key={device.id} className="flex justify-between items-center">
                          <div>
                            <p className="font-medium">{device.name || 'Unknown Device'}</p>
                            <p className="text-sm text-gray-500">{device.manufacturer || 'Unknown manufacturer'}</p>
                          </div>
                          <button
                            className={`px-3 py-1 rounded-md text-sm ${selectedInputId === device.id
                              ? 'bg-blue-100 text-blue-800'
                              : 'bg-gray-100 text-gray-800 hover:bg-gray-200'
                              }`}
                            onClick={() => setSelectedInputId(device.id)}
                          >
                            {selectedInputId === device.id ? 'Selected' : 'Select'}
                          </button>
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : (
                  <p className="text-gray-500 italic">No MIDI input devices detected.</p>
                )}
              </div>

              <div>
                <h3 className="text-lg font-medium mb-2">Output Devices</h3>
                {outputDevices.length > 0 ? (
                  <div className="bg-gray-50 rounded-md p-4">
                    <ul className="space-y-2">
                      {outputDevices.map(device => (
                        <li key={device.id} className="flex justify-between items-center">
                          <div>
                            <p className="font-medium">{device.name || 'Unknown Device'}</p>
                            <p className="text-sm text-gray-500">{device.manufacturer || 'Unknown manufacturer'}</p>
                          </div>
                          <button
                            className={`px-3 py-1 rounded-md text-sm ${selectedOutputId === device.id
                              ? 'bg-blue-100 text-blue-800'
                              : 'bg-gray-100 text-gray-800 hover:bg-gray-200'
                              }`}
                            onClick={() => setSelectedOutputId(device.id)}
                          >
                            {selectedOutputId === device.id ? 'Selected' : 'Select'}
                          </button>
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : (
                  <p className="text-gray-500 italic">No MIDI output devices detected.</p>
                )}
              </div>

              <div className="bg-gradient-to-r from-blue-50 to-blue-100 p-4 rounded-md border border-blue-200">
                <h3 className="text-lg font-medium text-blue-900 mb-2">MIDI Channel Settings</h3>
                <p className="text-gray-800 mb-4">
                  Select which MIDI channel to use for both input and output. Channel 0 means &quot;all channels&quot;.
                </p>

                <div className="grid grid-cols-8 gap-2">
                  {[0, ...Array.from({ length: 16 }, (_, i) => i + 1)].map(channel => (
                    <button
                      key={channel}
                      className={`py-2 rounded-md text-center ${selectedChannel === channel
                        ? 'bg-blue-500 text-white'
                        : 'bg-white text-blue-700 hover:bg-blue-100'
                        }`}
                      onClick={() => setSelectedChannel(channel)}
                    >
                      {channel === 0 ? 'All' : channel}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}