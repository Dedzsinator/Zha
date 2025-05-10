// components/MidiPlayer.tsx
'use client';

import { useState, useEffect, useRef } from 'react';
import * as Tone from 'tone';
import { Midi } from '@tonejs/midi';
import { FaPlay, FaPause, FaStop } from 'react-icons/fa';

interface MidiPlayerProps {
    midiData: Midi | null;
    onTimeUpdate: (time: number) => void;
}

export default function MidiPlayer({ midiData, onTimeUpdate }: MidiPlayerProps) {
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const synths = useRef<Tone.PolySynth[]>([]);
    const scheduledNotes = useRef<number[]>([]);

    useEffect(() => {
        // Clean up previous synths
        synths.current.forEach(synth => {
            synth.dispose();
        });
        synths.current = [];

        scheduledNotes.current.forEach(id => {
            Tone.Transport.clear(id);
        });
        scheduledNotes.current = [];

        if (!midiData) return;

        // Update duration
        setDuration(midiData.duration);
        setCurrentTime(0);

        // Create a synth for each track
        midiData.tracks.forEach(() => {
            const synth = new Tone.PolySynth(Tone.Synth).toDestination();
            synth.volume.value = -5; // Reduce volume a bit
            synths.current.push(synth);

            // Schedule all notes
            midiData.tracks.forEach((track, i) => {
                track.notes.forEach(note => {
                    const id = Tone.Transport.schedule((time) => {
                        synths.current[i]?.triggerAttackRelease(
                            note.name,
                            note.duration,
                            time,
                            note.velocity
                        );
                    }, note.time);

                    scheduledNotes.current.push(id);
                });
            });
        });

        // Set up loop to update current time
        const interval = setInterval(() => {
            if (isPlaying) {
                const newTime = Tone.Transport.seconds;
                setCurrentTime(newTime);
                if (onTimeUpdate) onTimeUpdate(newTime);

                // Stop playback when we reach the end
                if (newTime >= midiData.duration) {
                    handleStop();
                }
            }
        }, 16); // Update ~60fps

        return () => {
            clearInterval(interval);
            synths.current.forEach(synth => {
                synth.dispose();
            });
            scheduledNotes.current.forEach(id => {
                Tone.Transport.clear(id);
            });
        };
    }, [midiData]);

    const handlePlay = async () => {
        if (Tone.Transport.state !== "started") {
            await Tone.start();
            Tone.Transport.start();
        }
        setIsPlaying(true);
    };

    const handlePause = () => {
        Tone.Transport.pause();
        setIsPlaying(false);
    };

    const handleStop = () => {
        Tone.Transport.stop();
        Tone.Transport.seconds = 0;
        setCurrentTime(0);
        setIsPlaying(false);
        if (onTimeUpdate) onTimeUpdate(0);
    };

    const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
        const seekTime = parseFloat(e.target.value);
        Tone.Transport.seconds = seekTime;
        setCurrentTime(seekTime);
        if (onTimeUpdate) onTimeUpdate(seekTime);
    };

    const formatTime = (seconds: number): string => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    return (
        <div className="w-full p-4 bg-gray-50 border border-gray-200 rounded-md my-4">
            <div className="flex items-center mb-4">
                <button
                    className={`flex items-center justify-center w-10 h-10 rounded-full mr-2 ${!midiData ? 'bg-gray-300 cursor-not-allowed' : 'bg-blue-500 hover:bg-blue-600 text-white'
                        }`}
                    onClick={isPlaying ? handlePause : handlePlay}
                    disabled={!midiData}
                >
                    {isPlaying ? <FaPause size={14} /> : <FaPlay size={14} />}
                </button>
                <button
                    className={`flex items-center justify-center w-10 h-10 rounded-full mr-2 ${!midiData || !isPlaying ? 'bg-gray-300 cursor-not-allowed' : 'bg-blue-500 hover:bg-blue-600 text-white'
                        }`}
                    onClick={handleStop}
                    disabled={!midiData || !isPlaying}
                >
                    <FaStop size={14} />
                </button>

                <div className="ml-auto font-mono">
                    {formatTime(currentTime)} / {formatTime(duration)}
                </div>
            </div>

            <input
                type="range"
                min="0"
                max={duration || 1}
                step="0.01"
                value={currentTime}
                onChange={handleSeek}
                disabled={!midiData}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer disabled:opacity-50"
            />
        </div>
    );
}