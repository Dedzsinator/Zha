'use client';

import { useState, useEffect } from 'react';
import { FaPlug, FaUnlink, FaMicrophone, FaMicrophoneSlash } from 'react-icons/fa';

interface MidiDeviceManagerProps {
    onDevicesUpdate: (inputs: WebMidi.MIDIInput[], outputs: WebMidi.MIDIOutput[]) => void;
    onActiveNotesChange?: (notes: number[]) => void;
    onInputDeviceChange?: (deviceId: string | null) => void;
    onOutputDeviceChange?: (deviceId: string | null) => void;
    isRecording?: boolean;
    onRecordingToggle?: () => void;
    selectedChannel?: number;
    onChannelChange?: (channel: number) => void;
}

export default function MidiDeviceManager({
    onDevicesUpdate,
    onActiveNotesChange,
    onInputDeviceChange,
    onOutputDeviceChange,
    isRecording = false,
    onRecordingToggle = () => { },
    selectedChannel = 1,
    onChannelChange = () => { }
}: MidiDeviceManagerProps) {
    const [inputDevices, setInputDevices] = useState<WebMidi.MIDIInput[]>([]);
    const [outputDevices, setOutputDevices] = useState<WebMidi.MIDIOutput[]>([]);
    const [selectedInputId, setSelectedInputId] = useState<string | null>(null);
    const [selectedOutputId, setSelectedOutputId] = useState<string | null>(null);
    const [midiAccess, setMidiAccess] = useState<WebMidi.MIDIAccess | null>(null);
    const [midiError, setMidiError] = useState<string | null>(null);

    // Request MIDI access
    useEffect(() => {
        async function requestMidiAccess() {
            try {
                if (navigator.requestMIDIAccess) {
                    const access = await navigator.requestMIDIAccess({ sysex: false });
                    setMidiAccess(access);

                    // Update device lists
                    updateDeviceLists(access);

                    // Listen for device connection/disconnection
                    access.onstatechange = (e) => {
                        updateDeviceLists(access);
                    };
                } else {
                    setMidiError('WebMIDI is not supported in your browser.');
                }
            } catch (err) {
                setMidiError(`Failed to access MIDI devices: ${err}`);
                console.error(err);
            }
        }

        requestMidiAccess();
    }, []);

    // Update device lists whenever MIDI access changes
    const updateDeviceLists = (access: WebMidi.MIDIAccess) => {
        const inputs: WebMidi.MIDIInput[] = [];
        const outputs: WebMidi.MIDIOutput[] = [];

        access.inputs.forEach(input => inputs.push(input));
        access.outputs.forEach(output => outputs.push(output));

        setInputDevices(inputs);
        setOutputDevices(outputs);

        // Notify parent component
        onDevicesUpdate(inputs, outputs);
    };

    // Handle input device selection
    const handleInputChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const deviceId = e.target.value || null;
        setSelectedInputId(deviceId);
        if (onInputDeviceChange) {
            onInputDeviceChange(deviceId);
        }
    };

    // Handle output device selection
    const handleOutputChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const deviceId = e.target.value || null;
        setSelectedOutputId(deviceId);
        if (onOutputDeviceChange) {
            onOutputDeviceChange(deviceId);
        }
    };

    // Handle MIDI channel change
    const handleChannelChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const channel = parseInt(e.target.value, 10);
        onChannelChange(channel);
    };

    return (
        <div className="bg-white p-4 rounded-lg border border-gray-200 mb-4">
            <h3 className="text-lg font-semibold mb-3">MIDI Devices</h3>

            {midiError ? (
                <div className="bg-red-100 border border-red-300 text-red-700 px-4 py-3 rounded mb-4">
                    {midiError}
                </div>
            ) : null}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Input Device Selection */}
                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                        <FaPlug className="inline mr-1" /> Input Device
                    </label>
                    <select
                        className="block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                        value={selectedInputId || ""}
                        onChange={handleInputChange}
                    >
                        <option value="">No Input</option>
                        {inputDevices.map((device) => (
                            <option key={device.id} value={device.id}>
                                {device.name || `Device ${device.id}`}
                            </option>
                        ))}
                    </select>
                </div>

                {/* Output Device Selection */}
                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                        <FaUnlink className="inline mr-1" /> Output Device
                    </label>
                    <select
                        className="block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                        value={selectedOutputId || ""}
                        onChange={handleOutputChange}
                    >
                        <option value="">No Output</option>
                        {outputDevices.map((device) => (
                            <option key={device.id} value={device.id}>
                                {device.name || `Device ${device.id}`}
                            </option>
                        ))}
                    </select>
                </div>

                {/* MIDI Channel Selection */}
                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                        MIDI Channel
                    </label>
                    <select
                        className="block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                        value={selectedChannel}
                        onChange={handleChannelChange}
                    >
                        <option value="0">All</option>
                        {Array.from({ length: 16 }, (_, i) => (
                            <option key={i + 1} value={i + 1}>
                                Channel {i + 1}
                            </option>
                        ))}
                    </select>
                </div>

                {/* Recording Control */}
                <div className="flex items-end">
                    <button
                        onClick={onRecordingToggle}
                        className={`flex items-center justify-center px-4 py-2 rounded-md text-white ${isRecording
                            ? 'bg-red-600 hover:bg-red-700'
                            : 'bg-blue-600 hover:bg-blue-700'
                            }`}
                    >
                        {isRecording ? (
                            <>
                                <FaMicrophoneSlash className="mr-2" /> Stop Recording
                            </>
                        ) : (
                            <>
                                <FaMicrophone className="mr-2" /> Start Recording
                            </>
                        )}
                    </button>
                </div>
            </div>
        </div>
    );
}