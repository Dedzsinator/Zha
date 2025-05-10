// components/PianoRoll.tsx
'use client';

import { useEffect, useRef, useState } from 'react';
import { Midi } from '@tonejs/midi';

interface PianoRollProps {
    midiData: Midi | null;
    currentTime: number;
    pixelsPerSecond?: number;
}

const NOTE_HEIGHT = 10;
const NOTES_IN_OCTAVE = 12;
const OCTAVES = 8;
const TOTAL_NOTES = NOTES_IN_OCTAVE * OCTAVES;
const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

export default function PianoRoll({ midiData, currentTime = 0, pixelsPerSecond = 100 }: PianoRollProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const [dimensions, setDimensions] = useState({ width: 1000, height: TOTAL_NOTES * NOTE_HEIGHT });

    useEffect(() => {
        // Update dimensions based on container
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

    useEffect(() => {
        if (!canvasRef.current || !midiData) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const { width, height } = canvas;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        // Draw grid
        ctx.fillStyle = '#f5f5f5';
        for (let i = 0; i < TOTAL_NOTES; i++) {
            // Color black keys darker
            const noteInOctave = i % NOTES_IN_OCTAVE;
            if ([1, 3, 6, 8, 10].includes(noteInOctave)) {
                ctx.fillStyle = '#e9e9e9';
                ctx.fillRect(0, i * NOTE_HEIGHT, width, NOTE_HEIGHT);
                ctx.fillStyle = '#f5f5f5';
            }

            // Draw horizontal grid lines
            ctx.strokeStyle = '#ddd';
            ctx.lineWidth = 0.5;
            ctx.beginPath();
            ctx.moveTo(0, i * NOTE_HEIGHT);
            ctx.lineTo(width, i * NOTE_HEIGHT);
            ctx.stroke();
        }

        // Draw vertical grid lines (measures)
        for (let i = 0; i < width; i += pixelsPerSecond) {
            ctx.beginPath();
            ctx.moveTo(i, 0);
            ctx.lineTo(i, height);
            ctx.stroke();
        }

        // Draw notes from MIDI data
        if (midiData && midiData.tracks) {
            midiData.tracks.forEach(track => {
                track.notes.forEach(note => {
                    // Calculate position and dimensions
                    const x = note.time * pixelsPerSecond;
                    const y = height - (note.midi * NOTE_HEIGHT) - NOTE_HEIGHT;
                    const noteWidth = note.duration * pixelsPerSecond;

                    // Draw note rectangle
                    ctx.fillStyle = 'rgba(66, 133, 244, 0.7)';
                    ctx.fillRect(x, y, noteWidth, NOTE_HEIGHT - 1);

                    // Draw border
                    ctx.strokeStyle = 'rgb(46, 103, 214)';
                    ctx.lineWidth = 1;
                    ctx.strokeRect(x, y, noteWidth, NOTE_HEIGHT - 1);
                });
            });
        }

        // Draw playhead
        if (currentTime >= 0) {
            const playheadX = currentTime * pixelsPerSecond;
            ctx.strokeStyle = 'red';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(playheadX, 0);
            ctx.lineTo(playheadX, height);
            ctx.stroke();
        }

    }, [midiData, currentTime, pixelsPerSecond, dimensions]);

    return (
        <div className="flex w-full h-[400px] overflow-auto border border-gray-200 rounded-md bg-white">
            <div className="w-10 bg-gray-100 border-r border-gray-200 flex-shrink-0 flex flex-col">
                {Array.from({ length: OCTAVES }).map((_, octave) =>
                    NOTE_NAMES.map((note, index) => (
                        <div
                            key={`${note}-${octave}`}
                            className={`flex items-center justify-center text-xs border-b border-gray-200 text-gray-700
                ${[1, 3, 6, 8, 10].includes(index) ? 'bg-gray-200' : ''}`}
                            style={{ height: NOTE_HEIGHT }}
                        >
                            {note === 'C' ? `${note}${OCTAVES - octave}` : ''}
                        </div>
                    ))
                ).flat().reverse()}
            </div>

            <div ref={containerRef} className="flex-grow overflow-auto">
                <canvas
                    ref={canvasRef}
                    width={dimensions.width}
                    height={dimensions.height}
                    className="block"
                />
            </div>
        </div>
    );
}