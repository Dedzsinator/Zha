'use client';

import { useState, useRef } from 'react';
import { Midi } from '@tonejs/midi';
import { FaSave, FaTrash, FaClock } from 'react-icons/fa';

interface MidiRecorderProps {
    recordedMidi: Midi | null;
    isRecording: boolean;
    onStartRecording: () => void;
    onStopRecording: (midi: Midi | null) => void;
    onSave: (name: string) => void;
    onClear: () => void;
    inputDeviceId: string | null;
    selectedChannel: number;
    recordingTimeSeconds: number;
}

export default function MidiRecorder({
    recordedMidi,
    isRecording,
    onStartRecording,
    onStopRecording,
    onSave,
    onClear,
    inputDeviceId,
    selectedChannel,
    recordingTimeSeconds
}: MidiRecorderProps) {
    const [name, setName] = useState('My Recording');

    const formatTime = (seconds: number): string => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    const handleSave = () => {
        if (recordedMidi) {
            onSave(name);
        }
    };

    return (
        <div className="bg-white p-4 rounded-lg border border-gray-200">
            <h3 className="text-lg font-semibold mb-3">Recording</h3>

            <div className="flex space-x-2 mb-4">
                {!isRecording ? (
                    <button
                        onClick={onStartRecording}
                        className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
                        disabled={!inputDeviceId}
                    >
                        Start Recording
                    </button>
                ) : (
                    <button
                        onClick={() => onStopRecording(null)}
                        className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-gray-600 hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500"
                    >
                        Stop Recording
                    </button>
                )}
            </div>

            {isRecording && (
                <div className="flex items-center mb-3 text-red-600 font-medium">
                    <FaClock className="animate-pulse mr-2" />
                    <span>Recording: {formatTime(recordingTimeSeconds)}</span>
                </div>
            )}

            {recordedMidi && !isRecording && (
                <div className="space-y-3">
                    <div>
                        <p className="text-sm text-gray-600 mb-1">Recording contains {recordedMidi.tracks.reduce((sum, track) => sum + track.notes.length, 0)} notes</p>
                        <p className="text-sm text-gray-600">Duration: {formatTime(recordedMidi.duration)}</p>
                    </div>

                    <div>
                        <label htmlFor="recording-name" className="block text-sm font-medium text-gray-700 mb-1">
                            Recording Name
                        </label>
                        <input
                            type="text"
                            id="recording-name"
                            value={name}
                            onChange={(e) => setName(e.target.value)}
                            className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                        />
                    </div>

                    <div className="flex space-x-2">
                        <button
                            onClick={handleSave}
                            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                        >
                            <FaSave className="mr-2" /> Save Recording
                        </button>

                        <button
                            onClick={onClear}
                            className="inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md shadow-sm text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                        >
                            <FaTrash className="mr-2" /> Discard
                        </button>
                    </div>
                </div>
            )}

            {!recordedMidi && !isRecording && (
                <p className="text-center py-3 text-gray-500 italic">
                    No recording available. Use the &quotStart Recording&quot button to record MIDI input.
                </p>
            )}
        </div>
    );
}