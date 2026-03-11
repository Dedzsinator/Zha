'use client';

import { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import { Midi } from '@tonejs/midi';

interface PianoRollProps {
    midiData: Midi | null;
    currentTime: number;
    pixelsPerSecond?: number;
    activeNotes?: number[];
    selectedChannel?: number;
    editable?: boolean;
    onNoteAdd?: (note: number, time: number, duration: number, channel?: number) => void;
    onChannelSelect?: (channel: number) => void;
    showChannels?: boolean;
    height?: number;
}

const NOTE_HEIGHT = 12;
const NOTES_IN_OCTAVE = 12;
// const OCTAVES = 10; // Extended range (MIDI 0-127) - unused
// const MIN_NOTE = 0; // unused
// const MAX_NOTE = 127; // unused
// const TOTAL_NOTES = MAX_NOTE - MIN_NOTE + 1; // unused
const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const BLACK_KEY_INDICES = [1, 3, 6, 8, 10];
const PIANO_KEYS_WIDTH = 60;

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

export default function PianoRoll({
    midiData,
    currentTime = 0,
    pixelsPerSecond = 100,
    activeNotes = [],
    selectedChannel = 0,
    editable = false,
    onNoteAdd,
    onChannelSelect,
    showChannels = true,
    height = 500
}: PianoRollProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const [zoom, setZoom] = useState(1);
    const [verticalZoom, setVerticalZoom] = useState(1);
    const [isDrawing, setIsDrawing] = useState(false);
    const [drawStartPos, setDrawStartPos] = useState({ x: 0, y: 0, note: 0 });
    const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
    const [hoverNote, setHoverNote] = useState<number | null>(null);
    const [hoverTime, setHoverTime] = useState<number | null>(null);
    const [viewportRange, setViewportRange] = useState({ minNote: 36, maxNote: 96 }); // Default: C2-C7
    const [autoScroll, setAutoScroll] = useState(true);

    // Calculate actual pixels per second with zoom
    const effectivePixelsPerSecond = pixelsPerSecond * zoom;
    const effectiveNoteHeight = NOTE_HEIGHT * verticalZoom;

    // Calculate canvas dimensions
    const canvasWidth = useMemo(() => {
        if (!midiData) return 2000;
        const duration = midiData.duration;
        return Math.max(2000, duration * effectivePixelsPerSecond + 200);
    }, [midiData, effectivePixelsPerSecond]);

    const canvasHeight = useMemo(() => {
        const visibleNotes = viewportRange.maxNote - viewportRange.minNote + 1;
        return visibleNotes * effectiveNoteHeight;
    }, [viewportRange, effectiveNoteHeight]);

    // Optimize: Only process notes in visible range
    const visibleNotes = useMemo(() => {
        if (!midiData) return [];

        const notes: Array<{
            midi: number;
            time: number;
            duration: number;
            velocity: number;
            channel: number;
            trackIndex: number;
        }> = [];

        midiData.tracks.forEach((track, trackIndex) => {
            // Filter by channel if specified
            if (selectedChannel !== 0 && track.channel !== undefined && track.channel !== selectedChannel - 1) {
                return;
            }

            track.notes.forEach(note => {
                // Only include notes in visible range
                if (note.midi >= viewportRange.minNote && note.midi <= viewportRange.maxNote) {
                    notes.push({
                        midi: note.midi,
                        time: note.time,
                        duration: note.duration,
                        velocity: note.velocity,
                        channel: track.channel ?? 0,
                        trackIndex
                    });
                }
            });
        });

        return notes;
    }, [midiData, selectedChannel, viewportRange]);

    // Auto-scroll to follow playhead
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

    // Adjust viewport based on MIDI content
    useEffect(() => {
        if (!midiData || midiData.tracks.length === 0) return;

        let minNote = 127;
        let maxNote = 0;

        midiData.tracks.forEach(track => {
            track.notes.forEach(note => {
                minNote = Math.min(minNote, note.midi);
                maxNote = Math.max(maxNote, note.midi);
            });
        });

        // Add padding
        minNote = Math.max(0, minNote - 6);
        maxNote = Math.min(127, maxNote + 6);

        // Ensure minimum range
        if (maxNote - minNote < 24) {
            const center = Math.floor((minNote + maxNote) / 2);
            minNote = Math.max(0, center - 12);
            maxNote = Math.min(127, center + 12);
        }

        setViewportRange({ minNote, maxNote });
    }, [midiData]);

    // Drawing function with requestAnimationFrame for smooth rendering
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

        // Draw active notes (live input)
        drawActiveNotes(ctx);

        // Draw editing note
        if (isDrawing) {
            drawEditingNote(ctx);
        }

        // Draw hover indicator
        if (hoverNote !== null && hoverTime !== null && editable) {
            drawHoverIndicator(ctx);
        }

        // Draw playhead
        drawPlayhead(ctx, canvas.height);
    }, []); // eslint-disable-line react-hooks/exhaustive-deps

    // Render on every frame when playing, or when dependencies change
    useEffect(() => {
        draw();
    }, [draw]);

    const drawGrid = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
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
    };

    const drawMidiNotes = (ctx: CanvasRenderingContext2D) => {
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
    };

    const drawActiveNotes = (ctx: CanvasRenderingContext2D) => {
        activeNotes.forEach(note => {
            if (note < viewportRange.minNote || note > viewportRange.maxNote) return;

            const y = (viewportRange.maxNote - note) * effectiveNoteHeight;

            // Draw active note highlight
            ctx.fillStyle = 'rgba(255, 100, 100, 0.4)';
            ctx.fillRect(0, y + 1, ctx.canvas.width, effectiveNoteHeight - 2);

            // Draw border
            ctx.strokeStyle = 'rgba(255, 100, 100, 0.8)';
            ctx.lineWidth = 2;
            ctx.strokeRect(0, y + 1, ctx.canvas.width, effectiveNoteHeight - 2);
        });
    };

    const drawEditingNote = (ctx: CanvasRenderingContext2D) => {
        if (!editable) return;

        const startX = Math.min(drawStartPos.x, mousePos.x);
        const width = Math.abs(mousePos.x - drawStartPos.x);
        const y = drawStartPos.y;

        ctx.fillStyle = 'rgba(100, 255, 100, 0.5)';
        ctx.fillRect(startX, y + 1, width, effectiveNoteHeight - 2);

        ctx.strokeStyle = 'rgba(100, 255, 100, 0.9)';
        ctx.lineWidth = 2;
        ctx.strokeRect(startX, y + 1, width, effectiveNoteHeight - 2);
    };

    const drawHoverIndicator = (ctx: CanvasRenderingContext2D) => {
        if (hoverNote === null || hoverTime === null) return;
        if (hoverNote < viewportRange.minNote || hoverNote > viewportRange.maxNote) return;

        const x = hoverTime * effectivePixelsPerSecond;
        const y = (viewportRange.maxNote - hoverNote) * effectiveNoteHeight;

        ctx.fillStyle = 'rgba(150, 150, 255, 0.2)';
        ctx.fillRect(x, y, effectivePixelsPerSecond / 8, effectiveNoteHeight);

        ctx.strokeStyle = 'rgba(150, 150, 255, 0.5)';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, effectivePixelsPerSecond / 8, effectiveNoteHeight);
    };

    const drawPlayhead = (ctx: CanvasRenderingContext2D, height: number) => {
        if (currentTime >= 0) {
            const playheadX = currentTime * effectivePixelsPerSecond;

            // Draw playhead line
            ctx.strokeStyle = '#ff4444';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(playheadX, 0);
            ctx.lineTo(playheadX, height);
            ctx.stroke();

            // Draw playhead handle at top
            ctx.fillStyle = '#ff4444';
            ctx.fillRect(playheadX - 4, 0, 8, 10);
        }
    };

    // Mouse event handlers
    const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
        if (!editable || !canvasRef.current) return;

        const rect = canvasRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const note = viewportRange.maxNote - Math.floor(y / effectiveNoteHeight);
        const snappedY = Math.floor(y / effectiveNoteHeight) * effectiveNoteHeight;

        setIsDrawing(true);
        setDrawStartPos({ x, y: snappedY, note });
        setMousePos({ x, y: snappedY });
        setAutoScroll(false);
    };

    const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
        if (!canvasRef.current) return;

        const rect = canvasRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const noteIndex = viewportRange.maxNote - Math.floor(y / effectiveNoteHeight);
        const timePos = x / effectivePixelsPerSecond;

        setHoverNote(noteIndex);
        setHoverTime(timePos);

        if (isDrawing) {
            setMousePos({ x, y: drawStartPos.y });
        }
    };

    const handleMouseUp = () => {
        if (!editable || !onNoteAdd || !isDrawing) return;

        const startX = Math.min(drawStartPos.x, mousePos.x);
        const width = Math.abs(mousePos.x - drawStartPos.x);

        if (width > 5) {
            const startTime = startX / effectivePixelsPerSecond;
            const duration = width / effectivePixelsPerSecond;
            const note = drawStartPos.note;

            onNoteAdd(note, startTime, duration, selectedChannel);
        }

        setIsDrawing(false);
        setAutoScroll(true);
    };

    const handleMouseLeave = () => {
        if (isDrawing) {
            handleMouseUp();
        }
        setHoverNote(null);
        setHoverTime(null);
    };

    const handleWheel = (e: React.WheelEvent) => {
        if (e.ctrlKey || e.metaKey) {
            // Zoom with Ctrl+Wheel
            e.preventDefault();
            const delta = e.deltaY > 0 ? 0.9 : 1.1;
            setZoom(prev => Math.max(0.1, Math.min(10, prev * delta)));
        } else if (e.shiftKey) {
            // Vertical zoom with Shift+Wheel
            e.preventDefault();
            const delta = e.deltaY > 0 ? 0.9 : 1.1;
            setVerticalZoom(prev => Math.max(0.5, Math.min(3, prev * delta)));
        }
    };

    return (
        <div className="flex flex-col w-full bg-gray-900 rounded-lg overflow-hidden border border-gray-700">
            {/* Controls */}
            <div className="flex items-center gap-4 p-2 bg-gray-800 border-b border-gray-700 text-sm">
                <div className="flex items-center gap-2">
                    <label className="text-gray-300">Zoom:</label>
                    <input
                        type="range"
                        min="0.1"
                        max="5"
                        step="0.1"
                        value={zoom}
                        onChange={(e) => setZoom(parseFloat(e.target.value))}
                        className="w-24"
                    />
                    <span className="text-gray-400 w-12">{zoom.toFixed(1)}x</span>
                </div>
                <div className="flex items-center gap-2">
                    <label className="text-gray-300">V-Zoom:</label>
                    <input
                        type="range"
                        min="0.5"
                        max="3"
                        step="0.1"
                        value={verticalZoom}
                        onChange={(e) => setVerticalZoom(parseFloat(e.target.value))}
                        className="w-24"
                    />
                    <span className="text-gray-400 w-12">{verticalZoom.toFixed(1)}x</span>
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
                <div className="text-gray-400 text-xs ml-auto">
                    Ctrl+Wheel: H-Zoom | Shift+Wheel: V-Zoom
                </div>
            </div>

            {/* Piano Roll */}
            <div className="flex" style={{ height }}>
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
                    className="flex-grow overflow-auto relative bg-gray-900"
                    onWheel={handleWheel}
                >
                    <canvas
                        ref={canvasRef}
                        width={canvasWidth}
                        height={canvasHeight}
                        className="block cursor-crosshair"
                        onMouseDown={handleMouseDown}
                        onMouseMove={handleMouseMove}
                        onMouseUp={handleMouseUp}
                        onMouseLeave={handleMouseLeave}
                    />
                </div>
            </div>

            {/* Channel legend */}
            {showChannels && midiData && (
                <div className="flex flex-wrap gap-2 p-2 bg-gray-800 border-t border-gray-700 text-xs">
                    <span className="text-gray-400">Tracks:</span>
                    {midiData.tracks.map((track, index) => {
                        const channel = track.channel ?? 0;
                        const trackName = track.name || `Channel ${channel + 1}`;
                        const isSelected = selectedChannel === 0 || channel === selectedChannel - 1;
                        return (
                            <button
                                key={index}
                                onClick={() => onChannelSelect?.(channel + 1)}
                                className={`flex items-center gap-1 px-2 py-1 rounded transition-colors ${isSelected
                                        ? 'bg-blue-600 text-white'
                                        : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                                    }`}
                            >
                                <div
                                    className="w-3 h-3 rounded border border-gray-600"
                                    style={{ backgroundColor: CHANNEL_COLORS[channel % CHANNEL_COLORS.length].main }}
                                />
                                <span>{trackName}</span>
                            </button>
                        );
                    })}
                    <button
                        onClick={() => onChannelSelect?.(0)}
                        className={`px-2 py-1 rounded text-xs transition-colors ${selectedChannel === 0
                                ? 'bg-blue-600 text-white'
                                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                            }`}
                    >
                        All Tracks
                    </button>
                </div>
            )}
        </div>
    );
}