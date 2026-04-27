'use client';

import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { FaPlay, FaPause, FaStop, FaVolumeUp, FaVolumeMute, FaSave, FaTrash, FaClock, FaMicrophone, FaMicrophoneSlash } from 'react-icons/fa';
import * as Tone from 'tone';
import { Midi } from '@tonejs/midi';
import Soundfont, { InstrumentName } from 'soundfont-player';

interface MidiPlayerProps {
    midiData: Midi | null;
    onTimeUpdate: (time: number) => void;
    inputDeviceId?: string;
    outputDeviceId?: string;
    selectedChannel?: number;
    duetMode?: boolean;
    duetStyle?: 'simple' | 'chord' | 'melody';
    onActiveNotesChange?: (notes: number[]) => void;
    autoPlay?: boolean;
}

interface TrackInfo {
    id: number;
    name: string;
    instrument: InstrumentName;
    volume: number;
    muted: boolean;
    solo: boolean;
    channel: number;
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

const NOTE_HEIGHT = 20;
const NOTES_IN_OCTAVE = 12;
const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const BLACK_KEY_INDICES = [1, 3, 6, 8, 10];
const PIANO_KEYS_WIDTH = 60;

// Reverse mapping from MIDI program to soundfont instrument name.
const INSTRUMENT_NUMBER_TO_NAME: { [key: number]: InstrumentName } = {
    0: 'acoustic_grand_piano',
    1: 'bright_acoustic_piano',
    2: 'electric_grand_piano',
    3: 'honkytonk_piano',
    4: 'electric_piano_1',
    5: 'electric_piano_2',
    6: 'harpsichord',
    7: 'clavinet',
    19: 'church_organ',
    24: 'acoustic_guitar_nylon',
    32: 'acoustic_bass',
    33: 'electric_bass_finger',
    34: 'electric_bass_pick',
    35: 'fretless_bass',
    36: 'slap_bass_1',
    37: 'slap_bass_2',
    38: 'synth_bass_1',
    39: 'synth_bass_2',
    40: 'violin',
    41: 'viola',
    42: 'cello',
    43: 'contrabass',
    48: 'string_ensemble_1',
    52: 'choir_aahs',
    56: 'trumpet',
    73: 'flute',
};

// Channel colors (16 MIDI channels)
const CHANNEL_COLORS = [
    { main: 'hsla(210, 70%, 60%, 0.75)', border: 'hsla(210, 70%, 40%, 1)' }, // Blue
    { main: 'hsla(150, 70%, 55%, 0.75)', border: 'hsla(150, 70%, 35%, 1)' }, // Green
    { main: 'hsla(30, 80%, 60%, 0.75)', border: 'hsla(30, 80%, 40%, 1)' },   // Orange
    { main: 'hsla(280, 70%, 60%, 0.75)', border: 'hsla(280, 70%, 40%, 1)' }, // Purple
    { main: 'hsla(0, 70%, 60%, 0.75)', border: 'hsla(0, 70%, 40%, 1)' },     // Red
    { main: 'hsla(180, 60%, 55%, 0.75)', border: 'hsla(180, 60%, 35%, 1)' }, // Cyan
    { main: 'hsla(50, 80%, 55%, 0.75)', border: 'hsla(50, 80%, 35%, 1)' },   // Yellow
    { main: 'hsla(320, 70%, 60%, 0.75)', border: 'hsla(320, 70%, 40%, 1)' }, // Pink
    { main: 'hsla(100, 60%, 55%, 0.75)', border: 'hsla(100, 60%, 35%, 1)' }, // Lime
    { main: 'hsla(240, 60%, 60%, 0.75)', border: 'hsla(240, 60%, 40%, 1)' }, // Indigo
    { main: 'hsla(20, 75%, 58%, 0.75)', border: 'hsla(20, 75%, 38%, 1)' },   // Coral
    { main: 'hsla(160, 65%, 55%, 0.75)', border: 'hsla(160, 65%, 35%, 1)' }, // Teal
    { main: 'hsla(300, 65%, 58%, 0.75)', border: 'hsla(300, 65%, 38%, 1)' }, // Magenta
    { main: 'hsla(75, 60%, 55%, 0.75)', border: 'hsla(75, 60%, 35%, 1)' },   // Olive
    { main: 'hsla(200, 65%, 58%, 0.75)', border: 'hsla(200, 65%, 38%, 1)' }, // Sky
    { main: 'hsla(270, 60%, 58%, 0.75)', border: 'hsla(270, 60%, 38%, 1)' }, // Violet
];

export default function MidiPlayer({
    midiData,
    onTimeUpdate,
    autoPlay
}: MidiPlayerProps) {
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const [tracks, setTracks] = useState<TrackInfo[]>([]);
    const [selectedTrack, setSelectedTrack] = useState<number>(0);

    // Recording state
    const [isRecording, setIsRecording] = useState(false);
    const [recordedMidi, setRecordedMidi] = useState<Midi | null>(null);
    const [recordingTime, setRecordingTime] = useState(0);

    // Piano roll state
    const [zoom, setZoom] = useState(1);
    const [verticalZoom, setVerticalZoom] = useState(1);
    const [viewportRange, setViewportRange] = useState({ minNote: 36, maxNote: 96 });
    const [autoScroll, setAutoScroll] = useState(true);

    // Audio components
    const audioContextRef = useRef<AudioContext | null>(null);
    const instrumentsRef = useRef<Map<number, Soundfont.Player>>(new Map());
    const playbackIntervalRef = useRef<NodeJS.Timeout | null>(null);
    const scheduledEventsRef = useRef<number[]>([]);
    const recordingIntervalRef = useRef<NodeJS.Timeout | null>(null);

    // Piano roll refs
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);

    // Calculate effective values
    const effectivePixelsPerSecond = 100 * zoom;
    const effectiveNoteHeight = NOTE_HEIGHT * verticalZoom;

    // Calculate canvas dimensions
    const canvasWidth = midiData ? Math.max(2000, midiData.duration * effectivePixelsPerSecond + 200) : 2000;
    const canvasHeight = (viewportRange.maxNote - viewportRange.minNote + 1) * effectiveNoteHeight;

    // Get visible notes for piano roll
    const visibleNotes = useMemo(() => midiData ? midiData.tracks.flatMap((track, trackIndex) =>
        track.notes
            .filter(note => note.midi >= viewportRange.minNote && note.midi <= viewportRange.maxNote)
            .map(note => ({
                midi: note.midi,
                time: note.time,
                duration: note.duration,
                velocity: note.velocity,
                channel: track.channel ?? 0,
                trackIndex
            }))
    ) : [], [midiData, viewportRange]);

    // Initialize tracks from MIDI data
    useEffect(() => {
        if (!midiData) {
            setTracks([]);
            setDuration(0);
            setCurrentTime(0);
            return;
        }

        const trackInfos: TrackInfo[] = midiData.tracks.map((track, index) => {
            // Get instrument from MIDI track or default to piano
            const instrumentNumber = track.instrument?.number ?? 0;
            const instrumentName = (track.channel === 9
                ? 'woodblock'
                : (INSTRUMENT_NUMBER_TO_NAME[instrumentNumber] || 'acoustic_grand_piano')) as InstrumentName;

            return {
                id: index,
                name: track.name || `Track ${index + 1}`,
                instrument: instrumentName,
                volume: 0.8,
                muted: false,
                solo: false,
                channel: track.channel ?? index
            };
        });
        setTracks(trackInfos);
        setDuration(midiData.duration);
        setCurrentTime(0);

        // Adjust viewport based on MIDI content
        let minNote = 127;
        let maxNote = 0;
        midiData.tracks.forEach(track => {
            track.notes.forEach(note => {
                minNote = Math.min(minNote, note.midi);
                maxNote = Math.max(maxNote, note.midi);
            });
        });

        if (minNote < 127 && maxNote > 0) {
            minNote = Math.max(0, minNote - 6);
            maxNote = Math.min(127, maxNote + 6);
            if (maxNote - minNote < 24) {
                const center = Math.floor((minNote + maxNote) / 2);
                minNote = Math.max(0, center - 12);
                maxNote = Math.min(127, center + 12);
            }
            setViewportRange({ minNote, maxNote });
        }
    }, [midiData]);

    // Load instruments for tracks
    useEffect(() => {
        const loadInstruments = async () => {
            if (!audioContextRef.current) return;

            for (const track of tracks) {
                // Check if we need to load a new instrument (either not loaded or instrument changed)
                const currentInstrument = instrumentsRef.current.get(track.id);
                const needsNewInstrument = !currentInstrument ||
                    (currentInstrument && 'name' in currentInstrument && currentInstrument.name !== track.instrument);

                if (needsNewInstrument) {
                    // Stop and remove old instrument if it exists
                    if (currentInstrument && typeof currentInstrument.stop === 'function') {
                        currentInstrument.stop();
                    }
                    instrumentsRef.current.delete(track.id);

                    try {
                        const instrument = await Soundfont.instrument(audioContextRef.current, track.instrument);
                        instrumentsRef.current.set(track.id, instrument);
                        console.log(`Loaded instrument: ${track.instrument} for track ${track.id}`);
                    } catch (error) {
                        console.error(`Failed to load instrument ${track.instrument} for track ${track.id}:`, error);
                        // Fallback to piano
                        try {
                            const fallbackInstrument = await Soundfont.instrument(audioContextRef.current, 'acoustic_grand_piano');
                            instrumentsRef.current.set(track.id, fallbackInstrument);
                        } catch (fallbackError) {
                            console.error('Failed to load fallback piano instrument:', fallbackError);
                        }
                    }
                }
            }
        };

        loadInstruments();
    }, [tracks]);

    // Schedule MIDI playback
    useEffect(() => {
        if (!midiData) return;

        // Clear any existing scheduled events
        scheduledEventsRef.current.forEach(eventId => Tone.Transport.clear(eventId));
        scheduledEventsRef.current = [];

        // Schedule notes for playback
        midiData.tracks.forEach((track, trackIndex) => {
            const trackInfo = tracks[trackIndex];
            if (!trackInfo || trackInfo.muted) return;

            // Check if solo mode is active
            const soloActive = tracks.some(t => t.solo);
            if (soloActive && !trackInfo.solo) return;

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

    // Draw piano roll
    const draw = useCallback(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d', { alpha: false });
        if (!ctx) return;

        // Clear with background
        ctx.fillStyle = '#1a1a1a';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw grid
        drawGrid(ctx, canvas.width, canvas.height);

        // Draw MIDI notes
        drawMidiNotes(ctx);

        // Draw playhead
        drawPlayhead(ctx, canvas.height);

        function drawGrid(ctx: CanvasRenderingContext2D, width: number, height: number) {
            // Draw horizontal lines for each note
            for (let note = viewportRange.minNote; note <= viewportRange.maxNote; note++) {
                const y = (viewportRange.maxNote - note) * effectiveNoteHeight;
                const noteInOctave = note % NOTES_IN_OCTAVE;

                // Alternate colors for white and black keys
                if (BLACK_KEY_INDICES.includes(noteInOctave)) {
                    ctx.fillStyle = '#252525';
                } else {
                    ctx.fillStyle = '#1f1f1f';
                }
                ctx.fillRect(0, y, width, effectiveNoteHeight);

                // Highlight C notes
                if (noteInOctave === 0) {
                    ctx.fillStyle = 'rgba(70, 130, 180, 0.1)';
                    ctx.fillRect(0, y, width, effectiveNoteHeight);
                }

                // Draw horizontal grid line
                ctx.strokeStyle = '#333';
                ctx.lineWidth = 0.5;
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(width, y);
                ctx.stroke();
            }

            // Draw vertical grid lines (beats and measures)
            ctx.strokeStyle = '#333';

            // Beat lines
            for (let beat = 0; beat * effectivePixelsPerSecond < width; beat++) {
                const x = beat * effectivePixelsPerSecond;

                if (beat % 4 === 0) {
                    // Measure line
                    ctx.strokeStyle = '#555';
                    ctx.lineWidth = 1.5;
                } else {
                    // Beat line
                    ctx.strokeStyle = '#3a3a3a';
                    ctx.lineWidth = 0.8;
                }

                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, height);
                ctx.stroke();
            }
        }

        function drawMidiNotes(ctx: CanvasRenderingContext2D) {
            visibleNotes.forEach(note => {
                const x = note.time * effectivePixelsPerSecond;
                const y = (viewportRange.maxNote - note.midi) * effectiveNoteHeight;
                const noteWidth = Math.max(2, note.duration * effectivePixelsPerSecond);

                // Get color based on channel
                const channelColors = CHANNEL_COLORS[note.channel % CHANNEL_COLORS.length];

                // Draw note rectangle
                ctx.fillStyle = channelColors.main;
                ctx.fillRect(x, y + 1, noteWidth, effectiveNoteHeight - 2);

                // Draw border
                ctx.strokeStyle = channelColors.border;
                ctx.lineWidth = 1;
                ctx.strokeRect(x, y + 1, noteWidth, effectiveNoteHeight - 2);

                // Draw velocity indicator (brightness variation)
                if (noteWidth > 5) {
                    const velocityAlpha = note.velocity * 0.5;
                    ctx.fillStyle = `rgba(255, 255, 255, ${velocityAlpha})`;
                    ctx.fillRect(x + 1, y + 2, Math.min(3, noteWidth - 2), effectiveNoteHeight - 4);
                }
            });
        }

        function drawPlayhead(ctx: CanvasRenderingContext2D, height: number) {
            const x = currentTime * effectivePixelsPerSecond;
            ctx.strokeStyle = '#ff4444';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }
    }, [visibleNotes, currentTime, effectivePixelsPerSecond, effectiveNoteHeight, viewportRange]);

    // Auto-scroll
    useEffect(() => {
        if (!autoScroll || !containerRef.current || currentTime <= 0) return;

        const container = containerRef.current;
        const playheadX = currentTime * effectivePixelsPerSecond;
        const containerWidth = container.clientWidth - PIANO_KEYS_WIDTH;
        const scrollLeft = container.scrollLeft;

        // Auto-scroll if playhead is near the right edge
        if (playheadX > scrollLeft + containerWidth * 0.8) {
            container.scrollLeft = playheadX - containerWidth * 0.3;
        } else if (playheadX < scrollLeft + containerWidth * 0.2) {
            container.scrollLeft = Math.max(0, playheadX - containerWidth * 0.7);
        }
    }, [currentTime, effectivePixelsPerSecond, autoScroll]);

    // Render piano roll
    useEffect(() => {
        draw();
    }, [draw]);

    // Handle play/pause
    const handlePlay = useCallback(async () => {
        if (!midiData) return;

        try {
            // Initialize audio context on first play
            if (!audioContextRef.current) {
                audioContextRef.current = new (window.AudioContext || (window as typeof window & { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
            }

            // Ensure audio context is running
            if (audioContextRef.current.state === 'suspended') {
                await audioContextRef.current.resume();
            }

            // Load instruments for all tracks
            for (const track of tracks) {
                if (!instrumentsRef.current.has(track.id)) {
                    const instrument = await Soundfont.instrument(audioContextRef.current, track.instrument);
                    instrumentsRef.current.set(track.id, instrument);
                }
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

                // Update time
                playbackIntervalRef.current = setInterval(() => {
                    const time = Tone.Transport.seconds;
                    setCurrentTime(time);
                    onTimeUpdate(time);
                }, 100);
            }
        } catch (error) {
            console.error('Playback error:', error);
        }
    }, [isPlaying, midiData, onTimeUpdate, tracks]);

    // Handle stop
    const handleStop = useCallback(() => {
        Tone.Transport.stop();
        Tone.Transport.position = 0;
        setIsPlaying(false);
        setCurrentTime(0);
        onTimeUpdate(0);
        if (playbackIntervalRef.current) {
            clearInterval(playbackIntervalRef.current);
        }
    }, [onTimeUpdate]);

    // Recording functions
    const startRecording = useCallback(() => {
        setIsRecording(true);
        setRecordingTime(0);
        setRecordedMidi(null);

        // Start timer
        recordingIntervalRef.current = setInterval(() => {
            setRecordingTime(time => time + 0.1);
        }, 100);
    }, []);

    const stopRecording = useCallback(() => {
        setIsRecording(false);
        if (recordingIntervalRef.current) {
            clearInterval(recordingIntervalRef.current);
        }
        // For now, just create a simple recorded MIDI
        // In a real implementation, you'd capture actual MIDI input
        if (midiData) {
            setRecordedMidi(midiData);
        }
    }, [midiData]);

    const saveRecording = useCallback(() => {
        if (recordedMidi) {
            // Create download link
            const data = recordedMidi.toArray();
            const blob = new Blob([data as unknown as BlobPart], { type: 'audio/midi' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'recording.mid';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }
    }, [recordedMidi]);

    const clearRecording = useCallback(() => {
        setRecordedMidi(null);
        setRecordingTime(0);
    }, []);

    // Track management functions
    const updateTrackInstrument = useCallback((trackId: number, instrument: InstrumentName) => {
        setTracks(prev => prev.map(track =>
            track.id === trackId ? { ...track, instrument } : track
        ));
    }, []);

    const updateTrackVolume = useCallback((trackId: number, volume: number) => {
        setTracks(prev => prev.map(track =>
            track.id === trackId ? { ...track, volume } : track
        ));
    }, []);

    const toggleTrackMute = useCallback((trackId: number) => {
        setTracks(prev => prev.map(track =>
            track.id === trackId ? { ...track, muted: !track.muted } : track
        ));
    }, []);

    const toggleTrackSolo = useCallback((trackId: number) => {
        setTracks(prev => prev.map(track =>
            track.id === trackId ? { ...track, solo: !track.solo } : track
        ));
    }, []);

    // Cleanup
    useEffect(() => {
        const instruments = instrumentsRef.current;
        const audioContext = audioContextRef.current;
        const interval = playbackIntervalRef.current;
        const recordingInterval = recordingIntervalRef.current;

        return () => {
            // Stop all instruments
            instruments.forEach(instrument => {
                instrument.stop();
            });
            instruments.clear();

            if (audioContext) {
                audioContext.close();
            }
            if (interval) {
                clearInterval(interval);
            }
            if (recordingInterval) {
                clearInterval(recordingInterval);
            }
        };
    }, []);

    useEffect(() => {
        if (autoPlay && midiData && !isPlaying) {
            handlePlay();
        }
    }, [midiData, autoPlay]);

    const formatTime = (seconds: number): string => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };
    useEffect(() => {
        if (autoPlay && midiData && !isPlaying) {
            handlePlay();
        }
    }, [midiData, autoPlay]);
    return (
        <div className="bg-white rounded-lg shadow-lg p-6 space-y-4">
            {/* Controls */}
            <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                    <button
                        onClick={handlePlay}
                        className="p-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
                        disabled={!midiData}
                    >
                        {isPlaying ? <FaPause /> : <FaPlay />}
                    </button>
                    <button
                        onClick={handleStop}
                        className="p-2 bg-gray-500 text-white rounded hover:bg-gray-600 transition-colors"
                        disabled={!midiData}
                    >
                        <FaStop />
                    </button>
                </div>

                {/* Recording Controls */}
                <div className="flex items-center space-x-2">
                    {!isRecording ? (
                        <button
                            onClick={startRecording}
                            className="p-2 bg-red-500 text-white rounded hover:bg-red-600 transition-colors"
                        >
                            <FaMicrophone />
                        </button>
                    ) : (
                        <button
                            onClick={stopRecording}
                            className="p-2 bg-red-700 text-white rounded hover:bg-red-800 transition-colors animate-pulse"
                        >
                            <FaMicrophoneSlash />
                        </button>
                    )}
                    {recordedMidi && (
                        <>
                            <button
                                onClick={saveRecording}
                                className="p-2 bg-green-500 text-white rounded hover:bg-green-600 transition-colors"
                            >
                                <FaSave />
                            </button>
                            <button
                                onClick={clearRecording}
                                className="p-2 bg-gray-500 text-white rounded hover:bg-gray-600 transition-colors"
                            >
                                <FaTrash />
                            </button>
                        </>
                    )}
                </div>

                {/* Track Selector and Instrument */}
                <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-2">
                        <label className="text-sm font-medium">Track:</label>
                        <select
                            value={selectedTrack}
                            onChange={(e) => setSelectedTrack(parseInt(e.target.value))}
                            className="px-3 py-1 border border-gray-300 rounded text-sm"
                        >
                            {tracks.map(track => (
                                <option key={track.id} value={track.id}>
                                    {track.name}
                                </option>
                            ))}
                        </select>
                    </div>

                    <div className="flex items-center space-x-2">
                        <label className="text-sm font-medium">Instrument:</label>
                        <select
                            value={tracks[selectedTrack]?.instrument || 'acoustic_grand_piano'}
                            onChange={(e) => updateTrackInstrument(selectedTrack, e.target.value as InstrumentName)}
                            className="px-3 py-1 border border-gray-300 rounded text-sm"
                        >
                            {INSTRUMENT_OPTIONS.map(option => (
                                <option key={option.value} value={option.value}>
                                    {option.label}
                                </option>
                            ))}
                        </select>
                    </div>
                </div>
            </div>

            {/* Recording Status */}
            {isRecording && (
                <div className="flex items-center space-x-2 p-2 bg-red-50 border border-red-200 rounded">
                    <FaClock className="text-red-500" />
                    <span className="text-red-700 font-medium">Recording: {formatTime(recordingTime)}</span>
                </div>
            )}

            {recordedMidi && !isRecording && (
                <div className="flex items-center space-x-2 p-2 bg-green-50 border border-green-200 rounded">
                    <FaSave className="text-green-500" />
                    <span className="text-green-700 font-medium">Recording saved: recording.mid</span>
                </div>
            )}

            {/* Progress Bar */}
            {midiData && (
                <div className="space-y-2">
                    <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                            className="bg-blue-500 h-2 rounded-full transition-all duration-100"
                            style={{ width: `${(currentTime / duration) * 100}%` }}
                        />
                    </div>
                    <div className="flex justify-between text-sm text-gray-600">
                        <span>{currentTime.toFixed(1)}s</span>
                        <span>{duration.toFixed(1)}s</span>
                    </div>
                </div>
            )}

            {/* Piano Roll */}
            <div className="border border-gray-300 rounded bg-gray-900 overflow-x-auto">
                {/* Piano Roll Controls */}
                <div className="flex items-center justify-between p-2 bg-gray-800 border-b border-gray-700 text-white text-sm">
                    <div className="flex items-center space-x-4">
                        <div className="flex items-center space-x-2">
                            <span>H-Zoom:</span>
                            <input
                                type="range"
                                min="0.5"
                                max="3"
                                step="0.1"
                                value={zoom}
                                onChange={(e) => setZoom(parseFloat(e.target.value))}
                                className="w-24"
                            />
                            <span className="w-12">{zoom.toFixed(1)}x</span>
                        </div>
                        <div className="flex items-center space-x-2">
                            <span>V-Zoom:</span>
                            <input
                                type="range"
                                min="0.5"
                                max="3"
                                step="0.1"
                                value={verticalZoom}
                                onChange={(e) => setVerticalZoom(parseFloat(e.target.value))}
                                className="w-24"
                            />
                            <span className="w-12">{verticalZoom.toFixed(1)}x</span>
                        </div>
                        <button
                            onClick={() => setAutoScroll(!autoScroll)}
                            className={`px-3 py-1 rounded text-xs ${autoScroll ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300'}`}
                        >
                            Auto-scroll
                        </button>
                        <button
                            onClick={() => { setZoom(1); setVerticalZoom(1); }}
                            className="px-3 py-1 bg-gray-700 text-gray-300 rounded text-xs hover:bg-gray-600"
                        >
                            Reset Zoom
                        </button>
                    </div>
                </div>

                {/* Piano Roll Canvas */}
                <div className="flex" style={{ height: 1000, minHeight: 600 }}>
                    {/* Piano keys sidebar */}
                    <div
                        className="flex-shrink-0 bg-gray-800 border-r border-gray-700 overflow-hidden"
                        style={{ width: PIANO_KEYS_WIDTH }}
                    >
                        <div className="flex flex-col-reverse">
                            {Array.from({ length: viewportRange.maxNote - viewportRange.minNote + 1 }).map((_, i) => {
                                const note = viewportRange.minNote + i;
                                const noteInOctave = note % NOTES_IN_OCTAVE;
                                const octave = Math.floor(note / NOTES_IN_OCTAVE) - 1;
                                const isBlackKey = BLACK_KEY_INDICES.includes(noteInOctave);
                                const noteName = NOTE_NAMES[noteInOctave];
                                const isC = noteInOctave === 0;

                                return (
                                    <div
                                        key={note}
                                        className={`flex items-center justify-end px-2 text-xs border-t border-gray-700
                                            ${isBlackKey ? 'bg-gray-700 text-gray-400' : 'bg-gray-800 text-gray-300'}
                                            ${isC ? 'font-bold' : ''}`}
                                        style={{ height: effectiveNoteHeight, minHeight: effectiveNoteHeight }}
                                    >
                                        {isC && <span>{noteName}{octave}</span>}
                                        {!isC && !isBlackKey && <span className="opacity-50">{noteName}</span>}
                                    </div>
                                );
                            })}
                        </div>
                    </div>

                    {/* Canvas container */}
                    <div
                        ref={containerRef}
                        className="flex-grow overflow-auto relative"
                    >
                        <canvas
                            ref={canvasRef}
                            width={canvasWidth}
                            height={canvasHeight}
                            className="block"
                        />
                    </div>
                </div>

                {/* Channel legend */}
                {midiData && (
                    <div className="flex flex-wrap gap-2 p-2 bg-gray-800 border-t border-gray-700 text-xs">
                        <span className="text-gray-400">Tracks:</span>
                        {midiData.tracks.map((track, index) => {
                            const channel = track.channel ?? 0;
                            const trackName = track.name || `Channel ${channel + 1}`;
                            return (
                                <div key={index} className="flex items-center gap-1 px-2 py-1 bg-gray-700 text-gray-300 rounded">
                                    <div
                                        className="w-3 h-3 rounded border border-gray-600"
                                        style={{ backgroundColor: CHANNEL_COLORS[channel % CHANNEL_COLORS.length].main }}
                                    />
                                    <span>{trackName}</span>
                                </div>
                            );
                        })}
                    </div>
                )}
            </div>

            {/* Track Controls */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {tracks.map(track => (
                    <div key={track.id} className="border border-gray-200 rounded p-3 space-y-2">
                        <div className="flex items-center justify-between">
                            <h4 className="font-medium text-sm">{track.name}</h4>
                            <div className="flex items-center space-x-1">
                                <button
                                    onClick={() => toggleTrackMute(track.id)}
                                    className={`p-1 rounded text-xs ${track.muted ? 'bg-red-100 text-red-600' : 'bg-gray-100 text-gray-600'}`}
                                >
                                    {track.muted ? <FaVolumeMute /> : <FaVolumeUp />}
                                </button>
                                <button
                                    onClick={() => toggleTrackSolo(track.id)}
                                    className={`p-1 rounded text-xs ${track.solo ? 'bg-yellow-100 text-yellow-600' : 'bg-gray-100 text-gray-600'}`}
                                >
                                    S
                                </button>
                            </div>
                        </div>

                        <div className="space-y-1">
                            <div className="flex items-center justify-between text-xs">
                                <span>Volume:</span>
                                <span>{Math.round(track.volume * 100)}%</span>
                            </div>
                            <input
                                type="range"
                                min="0"
                                max="1"
                                step="0.1"
                                value={track.volume}
                                onChange={(e) => updateTrackVolume(track.id, parseFloat(e.target.value))}
                                className="w-full"
                            />
                        </div>
                    </div>
                ))}
            </div>

            {/* Status */}
            <div className="text-sm text-gray-600 text-center">
                {midiData ? (
                    <span>Loaded: {midiData.tracks.length} tracks, {duration.toFixed(1)}s duration</span>
                ) : (
                    <span>No MIDI data loaded</span>
                )}
            </div>
        </div>
    );
}