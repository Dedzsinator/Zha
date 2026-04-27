import React, { useState } from 'react';
import { useGuitarTracker, DetectedNote } from '../../hooks/useGuitarTracker';
import axios from 'axios';

interface LiveGuitarTrackerProps {
    onMidiGenerated: (file: File) => void;
}

export default function LiveGuitarTracker({ onMidiGenerated }: LiveGuitarTrackerProps) {
    const {
        isRecording,
        startTracking,
        stopTracking,
        detectedNotes,
        currentBpm,
        currentScale
    } = useGuitarTracker();

    const [isGenerating, setIsGenerating] = useState(false);

    const handleGenerate = async () => {
        setIsGenerating(true);
        try {
            // Get the latest notes from the tracker
            const currentNotes = detectedNotes;

            // Create a payload of the detected sequence
            const payload = {
                notes: currentNotes.map(n => ({
                    note: n.note,
                    midiNote: n.midiNote,
                    startTime: n.timeStart,
                    duration: n.duration || 500, // ms default
                })),
                scale: currentScale,
                bpm: currentBpm,
            };

            // Ensure at least some notes were played
            if (payload.notes.length === 0) {
                alert("Please play some notes on your guitar first!");
                setIsGenerating(false);
                return;
            }

            // Also convert guitar notes to a Midi object and trigger an update so we can see them
            const { Midi } = require('@tonejs/midi');
            const guitarMidi = new Midi();
            const track = guitarMidi.addTrack();
            const startMs = payload.notes[0]?.startTime || 0;

            payload.notes.forEach(n => {
                track.addNote({
                    midi: n.midiNote,
                    time: (n.startTime - startMs) / 1000,
                    duration: n.duration / 1000,
                });
            });
            // We could call onMidiGenerated for the guitar track immediately
            const guitarBlob = new Blob([guitarMidi.toArray()], { type: 'audio/midi' });
            const guitarFile = new File([guitarBlob], 'guitar_played.mid', { type: 'audio/midi' });
            onMidiGenerated(guitarFile);

            // Call backend to generate accompaniment based on guitar tab
            const response = await axios.post('http://localhost:8000/api/generate_accompaniment', payload, {
                responseType: 'blob'
            });

            // Assuming response gives back a MIDI file
            const file = new File([response.data], 'guitar_accompaniment.mid', {
                type: 'audio/midi',
            });
            // Merge or overwrite with the new generated MIDI file
            onMidiGenerated(file);
        } catch (error) {
            console.error('Error generating accompaniment:', error);
            alert('Failed to generate accompaniment. Ensure backend is running.');
        } finally {
            setIsGenerating(false);
        }
    };

    return (
        <div className="bg-gray-800 text-white rounded-lg p-4 shadow-md space-y-4">
            <h2 className="text-xl font-bold mb-2">Live Guitar Tracker (Scarlett Input)</h2>

            <div className="flex gap-4 mb-4">
                {!isRecording ? (
                    <button
                        onClick={startTracking}
                        className="flex-1 bg-red-600 hover:bg-red-700 text-white py-2 px-4 rounded font-medium"
                    >
                        Start Listening
                    </button>
                ) : (
                    <button
                        onClick={() => {
                            stopTracking();
                            handleGenerate();
                        }}
                        className="flex-1 bg-gray-600 hover:bg-gray-700 text-white py-2 px-4 rounded font-medium"
                    >
                        Stop Listening & Generate
                    </button>
                )}

                <button
                    onClick={handleGenerate}
                    disabled={isGenerating || detectedNotes.length === 0}
                    className="flex-1 bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-700 text-white py-2 px-4 rounded font-medium"
                >
                    {isGenerating ? 'Generating...' : 'Generate Accompaniment'}
                </button>
            </div>

            <div className="grid grid-cols-2 gap-4 mb-4">
                <div className="bg-gray-900 p-3 rounded">
                    <div className="text-sm text-gray-400">Detected Scale</div>
                    <div className="text-lg font-mono">{currentScale || 'Unknown'}</div>
                </div>
                <div className="bg-gray-900 p-3 rounded">
                    <div className="text-sm text-gray-400">Estimated BPM</div>
                    <div className="text-lg font-mono">{currentBpm || 120}</div>
                </div>
            </div>

            {/* Basic Tablature / Detected Notes Display */}
            <div className="bg-gray-900 p-3 rounded h-40 overflow-y-auto font-mono text-sm leading-tight border border-gray-700 relative">
                {detectedNotes.length === 0 && (
                    <div className="absolute inset-0 flex items-center justify-center text-gray-500">
                        {isRecording ? "Listening for guitar..." : "Play guitar to see tabs!"}
                    </div>
                )}
                <div className="flex flex-wrap gap-2">
                    {detectedNotes.map((note, index) => (
                        <span key={index} className="bg-gray-800 border border-gray-600 px-2 py-1 rounded inline-block">
                            {note.note}
                            <span className="text-xs text-gray-500 ml-1">
                                ({Math.round(note.duration || 0)}ms)
                            </span>
                        </span>
                    ))}
                </div>
            </div>
        </div>
    );
}