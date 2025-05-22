'use client';

import { useEffect, useRef, useState } from 'react';
import { Midi } from '@tonejs/midi';

interface PianoRollProps {
    midiData: Midi | null;
    currentTime: number;
    pixelsPerSecond?: number;
    activeNotes?: number[];
    selectedChannel?: number;
    editable?: boolean;
    onNoteAdd?: (note: number, time: number, duration: number) => void;
    onNoteRemove?: (note: number, time: number) => void;
}

const NOTE_HEIGHT = 10;
const NOTES_IN_OCTAVE = 12;
const OCTAVES = 8;
const TOTAL_NOTES = NOTES_IN_OCTAVE * OCTAVES;
const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const BLACK_KEY_INDICES = [1, 3, 6, 8, 10];

export default function PianoRoll({
    midiData,
    currentTime = 0,
    pixelsPerSecond = 100,
    activeNotes = [],
    selectedChannel = 0,
    editable = false,
    onNoteAdd,
    onNoteRemove
}: PianoRollProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const [dimensions, setDimensions] = useState({ width: 1000, height: TOTAL_NOTES * NOTE_HEIGHT });
    const [isDrawing, setIsDrawing] = useState(false);
    const [drawStartPos, setDrawStartPos] = useState({ x: 0, y: 0 });
    const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
    const [timeOffset, setTimeOffset] = useState(0);
    const [hoverNote, setHoverNote] = useState<number | null>(null);
    const [hoverTime, setHoverTime] = useState<number | null>(null);

    // Update dimensions based on container
    useEffect(() => {
        const handleResize = () => {
            if (containerRef.current && canvasRef.current) {
                setDimensions({
                    width: containerRef.current.clientWidth,
                    height: TOTAL_NOTES * NOTE_HEIGHT
                });
            }
        };

        window.addEventListener('resize', handleResize);
        handleResize();

        return () => window.removeEventListener('resize', handleResize);
    }, []);

    // Scroll horizontally to follow the playhead
    useEffect(() => {
        if (containerRef.current && currentTime > 0) {
            const playheadX = currentTime * pixelsPerSecond;
            const containerWidth = containerRef.current.clientWidth;
            const scrollLeft = containerRef.current.scrollLeft;

            // Check if playhead is outside visible area
            if (playheadX < scrollLeft || playheadX > scrollLeft + containerWidth) {
                containerRef.current.scrollLeft = playheadX - containerWidth / 2;
            }
        }
    }, [currentTime, pixelsPerSecond]);

    // Draw piano roll
    useEffect(() => {
        if (!canvasRef.current) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const { width, height } = canvas;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        // Draw grid with better styling
        drawGrid(ctx, width, height);

        // Draw MIDI notes
        if (midiData) {
            drawMidiNotes(ctx, midiData, height);
        }

        // Draw active notes (live input)
        drawActiveNotes(ctx, activeNotes, height);

        // Draw editing note (if currently drawing)
        if (isDrawing) {
            drawEditingNote(ctx, height);
        }

        // Draw hover position indicator
        if (hoverNote !== null && hoverTime !== null && editable) {
            drawHoverIndicator(ctx, height);
        }

        // Draw playhead
        drawPlayhead(ctx, height);

    }, [midiData, currentTime, pixelsPerSecond, dimensions, activeNotes, selectedChannel,
        isDrawing, drawStartPos, mousePos, hoverNote, hoverTime]);

    // Draw grid
    const drawGrid = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
        // Background
        ctx.fillStyle = '#f8f9fa';
        ctx.fillRect(0, 0, width, height);

        // Draw horizontal piano key backgrounds
        for (let i = 0; i < TOTAL_NOTES; i++) {
            // Use alternating colors for octaves for better visibility
            const octave = Math.floor(i / NOTES_IN_OCTAVE);
            const noteInOctave = i % NOTES_IN_OCTAVE;

            // Color black keys darker
            if (BLACK_KEY_INDICES.includes(noteInOctave)) {
                ctx.fillStyle = '#e9e9e9';
                ctx.fillRect(0, i * NOTE_HEIGHT, width, NOTE_HEIGHT);
            }

            // Highlight C notes for better octave visibility
            if (noteInOctave === 0) {
                ctx.fillStyle = 'rgba(200, 220, 255, 0.15)';
                ctx.fillRect(0, i * NOTE_HEIGHT, width, NOTE_HEIGHT);
            }
        }

        // Draw horizontal grid lines
        ctx.strokeStyle = '#ddd';
        ctx.lineWidth = 0.5;
        for (let i = 0; i <= TOTAL_NOTES; i++) {
            ctx.beginPath();
            ctx.moveTo(0, i * NOTE_HEIGHT);
            ctx.lineTo(width, i * NOTE_HEIGHT);
            ctx.stroke();
        }

        // Draw vertical grid lines (beats/measures)
        ctx.strokeStyle = '#e0e0e0';

        // Draw measure lines (every 4 beats)
        for (let i = 0; i < width; i += pixelsPerSecond * 4) {
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(i, 0);
            ctx.lineTo(i, height);
            ctx.stroke();
        }

        // Draw beat lines
        for (let i = 0; i < width; i += pixelsPerSecond) {
            ctx.lineWidth = 0.5;
            ctx.beginPath();
            ctx.moveTo(i, 0);
            ctx.lineTo(i, height);
            ctx.stroke();
        }
    };

    // Draw MIDI notes
    const drawMidiNotes = (ctx: CanvasRenderingContext2D, midiData: Midi, height: number) => {
        midiData.tracks.forEach(track => {
            // Filter by channel if selected
            if (selectedChannel !== 0 && track.channel !== undefined && track.channel !== selectedChannel - 1) {
                return;
            }

            // Get a unique color for this track
            const trackIndex = midiData.tracks.indexOf(track);
            const hue = (trackIndex * 40) % 360;
            const trackColor = `hsla(${hue}, 70%, 60%, 0.7)`;
            const trackBorderColor = `hsla(${hue}, 70%, 40%, 1)`;

            track.notes.forEach(note => {
                // Calculate position and dimensions
                const x = (note.time + timeOffset) * pixelsPerSecond;
                const y = height - (note.midi * NOTE_HEIGHT) - NOTE_HEIGHT;
                const noteWidth = note.duration * pixelsPerSecond;

                // Draw note rectangle with rounded corners for better aesthetics
                ctx.fillStyle = trackColor;
                ctx.beginPath();
                const radius = Math.min(4, noteWidth / 3, NOTE_HEIGHT / 2);
                ctx.roundRect(x, y, noteWidth, NOTE_HEIGHT - 1, radius);
                ctx.fill();

                // Draw border
                ctx.strokeStyle = trackBorderColor;
                ctx.lineWidth = 1;
                ctx.stroke();

                // Draw velocity indicator (small line at the beginning of note)
                const velocityHeight = NOTE_HEIGHT * 0.8 * note.velocity;
                ctx.fillStyle = trackBorderColor;
                ctx.fillRect(x, y + (NOTE_HEIGHT - velocityHeight) / 2, 3, velocityHeight);
            });
        });
    };

    // Draw active notes (from MIDI input)
    const drawActiveNotes = (ctx: CanvasRenderingContext2D, activeNotes: number[], height: number) => {
        activeNotes.forEach(note => {
            // Get y position for the note
            const y = height - (note * NOTE_HEIGHT) - NOTE_HEIGHT;

            // Draw active note marker
            ctx.fillStyle = 'rgba(255, 100, 100, 0.6)';
            ctx.fillRect(0, y, ctx.canvas.width, NOTE_HEIGHT - 1);

            // Draw border
            ctx.strokeStyle = 'rgba(220, 50, 50, 0.8)';
            ctx.lineWidth = 1;
            ctx.strokeRect(0, y, ctx.canvas.width, NOTE_HEIGHT - 1);
        });
    };

    // Draw note being created during editing
    const drawEditingNote = (ctx: CanvasRenderingContext2D, height: number) => {
        if (!editable) return;

        // Calculate note dimensions
        const startX = Math.min(drawStartPos.x, mousePos.x);
        const width = Math.abs(mousePos.x - drawStartPos.x);
        const y = drawStartPos.y;

        // Draw in-progress note
        ctx.fillStyle = 'rgba(100, 200, 100, 0.6)';
        ctx.fillRect(startX, y, width, NOTE_HEIGHT - 1);

        // Draw border
        ctx.strokeStyle = 'rgba(50, 180, 50, 0.8)';
        ctx.lineWidth = 1;
        ctx.strokeRect(startX, y, width, NOTE_HEIGHT - 1);
    };

    // Draw hover position indicator
    const drawHoverIndicator = (ctx: CanvasRenderingContext2D, height: number) => {
        if (hoverNote === null || hoverTime === null) return;

        const x = hoverTime * pixelsPerSecond;
        const y = height - (hoverNote * NOTE_HEIGHT) - NOTE_HEIGHT;

        // Draw hover box
        ctx.fillStyle = 'rgba(100, 100, 255, 0.2)';
        ctx.fillRect(x, y, pixelsPerSecond / 4, NOTE_HEIGHT);

        // Draw border
        ctx.strokeStyle = 'rgba(100, 100, 255, 0.4)';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, pixelsPerSecond / 4, NOTE_HEIGHT);
    };

    // Draw playhead
    const drawPlayhead = (ctx: CanvasRenderingContext2D, height: number) => {
        if (currentTime >= 0) {
            const playheadX = currentTime * pixelsPerSecond;
            ctx.strokeStyle = 'red';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(playheadX, 0);
            ctx.lineTo(playheadX, height);
            ctx.stroke();
        }
    };

    // Mouse event handlers for editing
    const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
        if (!editable || !canvasRef.current) return;

        const rect = canvasRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Snap y to note grid
        const snappedY = Math.floor(y / NOTE_HEIGHT) * NOTE_HEIGHT;

        setIsDrawing(true);
        setDrawStartPos({ x, y: snappedY });
        setMousePos({ x, y: snappedY });
    };

    const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
        if (!canvasRef.current) return;

        const rect = canvasRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Update hover position
        const noteIndex = TOTAL_NOTES - 1 - Math.floor(y / NOTE_HEIGHT);
        const timePos = x / pixelsPerSecond;

        setHoverNote(noteIndex);
        setHoverTime(timePos);

        if (isDrawing) {
            setMousePos({ x, y: drawStartPos.y });
        }
    };

    const handleMouseUp = () => {
        if (!editable || !onNoteAdd || !isDrawing) return;

        // Calculate note parameters
        const startX = Math.min(drawStartPos.x, mousePos.x);
        const width = Math.abs(mousePos.x - drawStartPos.x);
        const y = drawStartPos.y;

        if (width > 5) { // Minimum width to be considered a note
            const startTime = startX / pixelsPerSecond;
            const duration = width / pixelsPerSecond;
            const noteIndex = TOTAL_NOTES - 1 - Math.floor(y / NOTE_HEIGHT);

            // Add the note
            onNoteAdd(noteIndex, startTime, duration);
        }

        setIsDrawing(false);
    };

    const handleMouseLeave = () => {
        if (isDrawing) {
            handleMouseUp();
        }
        setHoverNote(null);
        setHoverTime(null);
    };

    return (
        <div className="flex w-full h-[400px] overflow-auto border border-gray-200 rounded-md bg-white">
            <div className="w-10 bg-gray-100 border-r border-gray-200 flex-shrink-0 flex flex-col">
                {Array.from({ length: OCTAVES }).map((_, octave) =>
                    NOTE_NAMES.map((note, index) => (
                        <div
                            key={`${note}-${octave}`}
                            className={`flex items-center justify-center text-xs border-b border-gray-200 text-gray-700
                ${BLACK_KEY_INDICES.includes(index) ? 'bg-gray-200' : ''}`}
                            style={{ height: NOTE_HEIGHT }}
                        >
                            {note === 'C' ? `${note}${OCTAVES - octave}` : ''}
                        </div>
                    ))
                ).flat().reverse()}
            </div>

            <div ref={containerRef} className="flex-grow overflow-auto relative">
                <canvas
                    ref={canvasRef}
                    width={dimensions.width}
                    height={dimensions.height}
                    className="block"
                    onMouseDown={handleMouseDown}
                    onMouseMove={handleMouseMove}
                    onMouseUp={handleMouseUp}
                    onMouseLeave={handleMouseLeave}
                />

                {/* Time ruler at top */}
                <div className="sticky top-0 h-6 bg-white border-b border-gray-200 z-10">
                    {/* Add time markers here */}
                </div>
            </div>
        </div>
    );
}