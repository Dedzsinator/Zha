'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { Midi } from '@tonejs/midi';
import * as Tone from 'tone';
import Soundfont, { InstrumentName } from 'soundfont-player';
import { FaPlay, FaPause, FaStop, FaVolumeUp } from 'react-icons/fa';

interface MidiEditorProps {
    midiData: Midi;
    onMidiDataChange: (midi: Midi) => void;
}

interface TrackInfo {
    id: number;
    name: string;
    instrument: InstrumentName;
    volume: number;
    muted: boolean;
    solo: boolean;
    notes: Midi['tracks'][0]['notes'];
}

const INSTRUMENT_OPTIONS: { value: InstrumentName; label: string }[] = [
    { value: 'acoustic_grand_piano', label: 'Acoustic Grand Piano' },
    { value: 'bright_acoustic_piano', label: 'Bright Acoustic Piano' },
    { value: 'electric_grand_piano', label: 'Electric Grand Piano' },
    { value: 'honkytonk_piano', label: 'Honky-tonk Piano' },
    { value: 'electric_piano_1', label: 'Electric Piano 1' },
    { value: 'electric_piano_2', label: 'Electric Piano 2' },
    { value: 'harpsichord', label: 'Harpsichord' },
    { value: 'clavinet', label: 'Clavinet' },
    { value: 'celesta', label: 'Celesta' },
    { value: 'glockenspiel', label: 'Glockenspiel' },
    { value: 'music_box', label: 'Music Box' },
    { value: 'vibraphone', label: 'Vibraphone' },
    { value: 'marimba', label: 'Marimba' },
    { value: 'xylophone', label: 'Xylophone' },
    { value: 'tubular_bells', label: 'Tubular Bells' },
    { value: 'dulcimer', label: 'Dulcimer' },
    { value: 'violin', label: 'Violin' },
    { value: 'viola', label: 'Viola' },
    { value: 'cello', label: 'Cello' },
    { value: 'contrabass', label: 'Contrabass' },
    { value: 'flute', label: 'Flute' },
    { value: 'piccolo', label: 'Piccolo' },
    { value: 'recorder', label: 'Recorder' },
    { value: 'pan_flute', label: 'Pan Flute' },
    { value: 'trumpet', label: 'Trumpet' },
    { value: 'trombone', label: 'Trombone' },
    { value: 'tuba', label: 'Tuba' },
    { value: 'french_horn', label: 'French Horn' },
    { value: 'brass_section', label: 'Brass Section' },
    { value: 'soprano_sax', label: 'Soprano Sax' },
    { value: 'alto_sax', label: 'Alto Sax' },
    { value: 'tenor_sax', label: 'Tenor Sax' },
    { value: 'baritone_sax', label: 'Baritone Sax' },
    { value: 'oboe', label: 'Oboe' },
    { value: 'english_horn', label: 'English Horn' },
    { value: 'bassoon', label: 'Bassoon' },
    { value: 'clarinet', label: 'Clarinet' },
];

export default function MidiEditor({ midiData, onMidiDataChange }: MidiEditorProps) { // eslint-disable-line @typescript-eslint/no-unused-vars
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [selectedTrack, setSelectedTrack] = useState<number>(0);
    const [tracks, setTracks] = useState<TrackInfo[]>([]);
    const [zoom, setZoom] = useState(1);
    const [gridSize, setGridSize] = useState(16); // 16th notes

    // Audio components
    const audioContextRef = useRef<AudioContext | null>(null);
    const instrumentsRef = useRef<Map<number, Soundfont.Player>>(new Map());
    const playbackIntervalRef = useRef<NodeJS.Timeout | null>(null);
    const scheduledEventsRef = useRef<number[]>([]);

    // Piano roll refs
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);

    // Initialize tracks from MIDI data
    useEffect(() => {
        const trackInfos: TrackInfo[] = midiData.tracks.map((track, index) => ({
            id: index,
            name: track.name || `Track ${index + 1}`,
            instrument: 'acoustic_grand_piano',
            volume: 0.8,
            muted: false,
            solo: false,
            notes: track.notes
        }));
        setTracks(trackInfos);
    }, [midiData]);

    // Initialize audio context
    useEffect(() => {
        const initAudio = async () => {
            try {
                audioContextRef.current = new (window.AudioContext || (window as typeof window & { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
                console.log('Audio context initialized');
            } catch (error) {
                console.error('Failed to initialize audio context:', error);
            }
        };

        initAudio();

        return () => {
            if (audioContextRef.current) {
                audioContextRef.current.close();
            }
            if (playbackIntervalRef.current) {
                clearInterval(playbackIntervalRef.current);
            }
        };
    }, []);

    // Load instruments for tracks
    useEffect(() => {
        const loadInstruments = async () => {
            if (!audioContextRef.current) return;

            for (const track of tracks) {
                if (!instrumentsRef.current.has(track.id)) {
                    try {
                        const instrument = await Soundfont.instrument(audioContextRef.current, track.instrument);
                        instrumentsRef.current.set(track.id, instrument);
                        console.log(`Loaded instrument for track ${track.id}: ${track.instrument}`);
                    } catch (error) {
                        console.error(`Failed to load instrument for track ${track.id}:`, error);
                    }
                }
            }
        };

        loadInstruments();
    }, [tracks]);

    // Schedule MIDI playback
    useEffect(() => {
        if (!midiData) return;

        // Clear existing events
        scheduledEventsRef.current.forEach(eventId => Tone.Transport.clear(eventId));
        scheduledEventsRef.current = [];

        // Schedule notes for each track
        midiData.tracks.forEach((track, trackIndex) => {
            const trackInfo = tracks[trackIndex];
            if (!trackInfo || trackInfo.muted) return;

            track.notes.forEach(note => {
                const eventId = Tone.Transport.schedule((time) => {
                    const instrument = instrumentsRef.current.get(trackIndex);
                    if (instrument) {
                        const noteName = Tone.Frequency(note.midi, 'midi').toNote();
                        instrument.play(noteName, time, {
                            gain: note.velocity * trackInfo.volume,
                            duration: note.duration
                        });
                    }
                }, note.time);
                scheduledEventsRef.current.push(eventId);
            });
        });

        return () => {
            scheduledEventsRef.current.forEach(eventId => Tone.Transport.clear(eventId));
            scheduledEventsRef.current = [];
        };
    }, [midiData, tracks]);

    // Handle play/pause
    const handlePlay = useCallback(async () => {
        if (!midiData) return;

        try {
            if (audioContextRef.current && audioContextRef.current.state === 'suspended') {
                await audioContextRef.current.resume();
            }

            if (isPlaying) {
                Tone.Transport.pause();
                setIsPlaying(false);
                if (playbackIntervalRef.current) {
                    clearInterval(playbackIntervalRef.current);
                }
            } else {
                Tone.Transport.start();
                setIsPlaying(true);

                playbackIntervalRef.current = setInterval(() => {
                    const time = Tone.Transport.seconds;
                    setCurrentTime(time);
                }, 50);
            }
        } catch (error) {
            console.error('Playback error:', error);
        }
    }, [isPlaying, midiData]);

    // Handle stop
    const handleStop = useCallback(() => {
        Tone.Transport.stop();
        Tone.Transport.seconds = 0;
        setIsPlaying(false);
        setCurrentTime(0);

        if (playbackIntervalRef.current) {
            clearInterval(playbackIntervalRef.current);
        }
    }, []);

    // Update track instrument
    const updateTrackInstrument = useCallback(async (trackId: number, instrument: InstrumentName) => {
        if (!audioContextRef.current) return;

        setTracks(prev => prev.map(track =>
            track.id === trackId ? { ...track, instrument } : track
        ));

        try {
            const newInstrument = await Soundfont.instrument(audioContextRef.current, instrument);
            instrumentsRef.current.set(trackId, newInstrument);
            console.log(`Updated instrument for track ${trackId}: ${instrument}`);
        } catch (error) {
            console.error(`Failed to load instrument ${instrument} for track ${trackId}:`, error);
        }
    }, []);

    // Toggle track mute
    const toggleTrackMute = useCallback((trackId: number) => {
        setTracks(prev => prev.map(track =>
            track.id === trackId ? { ...track, muted: !track.muted } : track
        ));
    }, []);

    // Toggle track solo
    const toggleTrackSolo = useCallback((trackId: number) => {
        setTracks(prev => prev.map(track =>
            track.id === trackId ? { ...track, solo: !track.solo } : track
        ));
    }, []);

    // Draw piano roll
    useEffect(() => {
        if (!canvasRef.current || !containerRef.current) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const container = containerRef.current;
        const rect = container.getBoundingClientRect();

        canvas.width = rect.width;
        canvas.height = rect.height;

        // Clear canvas
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw grid
        ctx.strokeStyle = '#e5e7eb';
        ctx.lineWidth = 1;

        const noteHeight = 20;
        const totalNotes = 88; // 88 keys on piano

        // Draw horizontal lines (notes)
        for (let i = 0; i <= totalNotes; i++) {
            const y = i * noteHeight;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(canvas.width, y);
            ctx.stroke();

            // Highlight black keys
            const noteNumber = 21 + i; // MIDI note 21 = A0
            if ([1, 3, 6, 8, 10].includes(noteNumber % 12)) {
                ctx.fillStyle = '#f3f4f6';
                ctx.fillRect(0, y, canvas.width, noteHeight);
            }
        }

        // Draw vertical grid lines (time)
        const pixelsPerSecond = 100 * zoom;
        const totalTime = midiData.duration;

        for (let time = 0; time <= totalTime; time += 60 / gridSize) { // Grid based on gridSize
            const x = time * pixelsPerSecond;
            if (x > canvas.width) break;

            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, canvas.height);
            ctx.stroke();
        }

        // Draw notes
        const selectedTrackInfo = tracks[selectedTrack];
        if (selectedTrackInfo) {
            selectedTrackInfo.notes.forEach(note => {
                const noteNumber = note.midi - 21; // MIDI note 21 = A0, our 0
                const y = (totalNotes - 1 - noteNumber) * noteHeight;
                const x = note.time * pixelsPerSecond;
                const width = note.duration * pixelsPerSecond;
                const height = noteHeight - 2;

                // Note color based on velocity
                const velocity = note.velocity;
                const alpha = Math.max(0.3, velocity);
                ctx.fillStyle = `rgba(59, 130, 246, ${alpha})`;
                ctx.fillRect(x + 1, y + 1, width - 2, height - 2);

                // Note border
                ctx.strokeStyle = '#3b82f6';
                ctx.lineWidth = 1;
                ctx.strokeRect(x + 1, y + 1, width - 2, height - 2);
            });
        }

        // Draw playhead
        const playheadX = currentTime * pixelsPerSecond;
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(playheadX, 0);
        ctx.lineTo(playheadX, canvas.height);
        ctx.stroke();

    }, [midiData, tracks, selectedTrack, currentTime, zoom, gridSize]);

    return (
        <div className="flex h-96">
            {/* Track List */}
            <div className="w-64 bg-gray-50 border-r border-gray-200 p-4">
                <h3 className="font-semibold mb-4">Tracks</h3>
                <div className="space-y-2">
                    {tracks.map(track => (
                        <div
                            key={track.id}
                            className={`p-3 rounded border cursor-pointer ${
                                selectedTrack === track.id
                                    ? 'border-blue-500 bg-blue-50'
                                    : 'border-gray-200 bg-white'
                            }`}
                            onClick={() => setSelectedTrack(track.id)}
                        >
                            <div className="flex items-center justify-between mb-2">
                                <span className="font-medium text-sm">{track.name}</span>
                                <div className="flex items-center space-x-1">
                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            toggleTrackMute(track.id);
                                        }}
                                        className={`p-1 rounded ${
                                            track.muted ? 'bg-red-100 text-red-600' : 'bg-gray-100'
                                        }`}
                                    >
                                        <FaVolumeUp className="w-3 h-3" />
                                    </button>
                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            toggleTrackSolo(track.id);
                                        }}
                                        className={`p-1 rounded ${
                                            track.solo ? 'bg-yellow-100 text-yellow-600' : 'bg-gray-100'
                                        }`}
                                    >
                                        S
                                    </button>
                                </div>
                            </div>

                            <select
                                value={track.instrument}
                                onChange={(e) => updateTrackInstrument(track.id, e.target.value as InstrumentName)}
                                className="w-full text-xs p-1 border border-gray-300 rounded"
                                onClick={(e) => e.stopPropagation()}
                            >
                                {INSTRUMENT_OPTIONS.map(option => (
                                    <option key={option.value} value={option.value}>
                                        {option.label}
                                    </option>
                                ))}
                            </select>

                            <div className="mt-2 text-xs text-gray-600">
                                {track.notes.length} notes
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Piano Roll */}
            <div className="flex-1 flex flex-col">
                {/* Toolbar */}
                <div className="flex items-center justify-between p-4 bg-gray-50 border-b border-gray-200">
                    <div className="flex items-center space-x-4">
                        <button
                            onClick={handlePlay}
                            className="p-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                        >
                            {isPlaying ? <FaPause /> : <FaPlay />}
                        </button>
                        <button
                            onClick={handleStop}
                            className="p-2 bg-gray-500 text-white rounded hover:bg-gray-600"
                        >
                            <FaStop />
                        </button>

                        <div className="flex items-center space-x-2">
                            <label className="text-sm">Zoom:</label>
                            <input
                                type="range"
                                min="0.1"
                                max="3"
                                step="0.1"
                                value={zoom}
                                onChange={(e) => setZoom(parseFloat(e.target.value))}
                                className="w-20"
                            />
                            <span className="text-sm">{zoom.toFixed(1)}x</span>
                        </div>

                        <div className="flex items-center space-x-2">
                            <label className="text-sm">Grid:</label>
                            <select
                                value={gridSize}
                                onChange={(e) => setGridSize(parseInt(e.target.value))}
                                className="text-sm border border-gray-300 rounded px-2 py-1"
                            >
                                <option value="4">1/4</option>
                                <option value="8">1/8</option>
                                <option value="16">1/16</option>
                                <option value="32">1/32</option>
                            </select>
                        </div>
                    </div>

                    <div className="text-sm text-gray-600">
                        {currentTime.toFixed(2)}s / {midiData.duration.toFixed(2)}s
                    </div>
                </div>

                {/* Piano Roll Canvas */}
                <div ref={containerRef} className="flex-1 overflow-auto bg-white">
                    <canvas
                        ref={canvasRef}
                        className="cursor-crosshair"
                        style={{ minHeight: '600px' }}
                    />
                </div>
            </div>
        </div>
    );
}