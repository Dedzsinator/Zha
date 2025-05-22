'use client';

import { useState, useEffect, useRef } from 'react';
import { FaPlay, FaPause, FaStop } from 'react-icons/fa';
import * as Tone from 'tone';
import { Midi } from '@tonejs/midi';

interface MidiPlayerProps {
    midiData: Midi | null;
    onTimeUpdate: (time: number) => void;
    inputDeviceId: string | null;
    outputDeviceId: string | null;
    selectedChannel: number;
    duetMode: boolean;
    duetStyle?: 'simple' | 'chord' | 'melody';
    onActiveNotesChange?: (notes: number[]) => void;
}

export default function MidiPlayer({
    midiData,
    onTimeUpdate,
    inputDeviceId,
    outputDeviceId,
    selectedChannel,
    duetMode,
    duetStyle = 'simple',
    onActiveNotesChange
}: MidiPlayerProps) {
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const synths = useRef<Tone.PolySynth[]>([]);
    const scheduledNotes = useRef<number[]>([]);
    const inputDevice = useRef<WebMidi.MIDIInput | null>(null);
    const outputDevice = useRef<WebMidi.MIDIOutput | null>(null);
    const recordedNotes = useRef<{ note: number, velocity: number, time: number, duration: number }[]>([]);
    const activeNotes = useRef<Map<number, { startTime: number, velocity: number }>>(new Map());
    const startTime = useRef<number | null>(null);

    const triggerDuetResponse = (note: number, velocity: number) => {
        // Get all recently played notes (last 2 seconds)
        const recentNotes = recordedNotes.current
            .filter(n => Tone.now() - (n.time + startTime.current!) < 2)
            .map(n => n.note % 12); // Get just the pitch class

        // Add current note
        recentNotes.push(note % 12);

        // Create a response based on selected style
        setTimeout(() => {
            // Different response styles
            let responses: number[] = [];

            switch (duetStyle) {
                case 'simple':
                    // Simple responses: play a third above or a fifth above
                    responses = [3, 7]; // Minor third and perfect fifth
                    break;
                case 'chord':
                    // Chord response: play a triad
                    responses = [3, 7, 12]; // Minor triad with octave
                    break;
                case 'melody':
                    // Melody response: play a sequence
                    playMelodyResponse(note, velocity);
                    return; // Early return as we handle this separately
                default:
                    responses = [3, 7];
            }

            for (const interval of responses) {
                const responseNote = note + interval;
                if (responseNote < 108) { // Stay in reasonable MIDI range
                    playNoteOnOutput(responseNote, velocity, 0, 0.5); // Play with 0.5s duration

                    // Also play on internal synth for feedback
                    const noteName = Tone.Frequency(responseNote, "midi").toNote();
                    synths.current[1]?.triggerAttackRelease(
                        noteName,
                        0.5,
                        undefined,
                        velocity / 127
                    );
                }
            }
        }, 250); // Small delay to make it feel like a response
    };

    // Connect to MIDI input device
    useEffect(() => {
        if (!navigator.requestMIDIAccess) return;

        const connectToInputDevice = async () => {
            try {
                const access = await navigator.requestMIDIAccess();

                // Disconnect any existing input
                if (inputDevice.current) {
                    inputDevice.current.onmidimessage = null;
                }

                // Connect to new input if ID is provided
                if (inputDeviceId) {
                    const device = access.inputs.get(inputDeviceId);
                    if (device) {
                        inputDevice.current = device;
                        device.onmidimessage = handleMidiMessage;
                        console.log(`Connected to MIDI input: ${device.name}`);
                    }
                } else {
                    inputDevice.current = null;
                }
            } catch (err) {
                console.error("Failed to access MIDI devices:", err);
            }
        };

        connectToInputDevice();

        return () => {
            if (inputDevice.current) {
                inputDevice.current.onmidimessage = null;
            }
        };
    }, [inputDeviceId]);

    // Connect to MIDI output device
    useEffect(() => {
        if (!navigator.requestMIDIAccess) return;

        const connectToOutputDevice = async () => {
            try {
                const access = await navigator.requestMIDIAccess();

                // Connect to new output if ID is provided
                if (outputDeviceId) {
                    const device = access.outputs.get(outputDeviceId);
                    if (device) {
                        outputDevice.current = device;
                        console.log(`Connected to MIDI output: ${device.name}`);
                    }
                } else {
                    outputDevice.current = null;
                }
            } catch (err) {
                console.error("Failed to access MIDI devices:", err);
            }
        };

        connectToOutputDevice();
    }, [outputDeviceId]);

    const playMelodyResponse = (note: number, velocity: number) => {
        // Play a simple melodic pattern in response
        const baseNote = note;
        const pattern = [0, 2, 4, 0]; // Simple pattern

        pattern.forEach((interval, index) => {
            const responseNote = baseNote + interval;
            if (responseNote < 108) {
                // Stagger the notes with increasing delay
                const delay = 0.1 + (index * 0.15);

                // Play note on output device
                playNoteOnOutput(responseNote, velocity, delay, 0.3);

                // Also play on internal synth
                setTimeout(() => {
                    const noteName = Tone.Frequency(responseNote, "midi").toNote();
                    synths.current[1]?.triggerAttackRelease(
                        noteName,
                        0.3,
                        undefined,
                        velocity / 127
                    );
                }, delay * 1000);
            }
        });
    };

    // Handle incoming MIDI messages
    const handleMidiMessage = (message: WebMidi.MIDIMessageEvent) => {
        const data = message.data;
        const cmd = data[0] >> 4;
        const channel = (data[0] & 0xf) + 1;
        const noteNumber = data[1];
        const velocity = data[2];

        // Only process messages for our selected channel
        if (channel !== selectedChannel && selectedChannel !== 0) return;

        // Note On message (with velocity > 0)
        if (cmd === 9 && velocity > 0) {
            handleNoteOn(noteNumber, velocity);
        }
        // Note Off message or Note On with velocity 0
        else if (cmd === 8 || (cmd === 9 && velocity === 0)) {
            handleNoteOff(noteNumber);
        }
    };

    // Handle Note On message
    const handleNoteOn = (note: number, velocity: number) => {
        // Record the start time for this note
        const currentTimeInSec = Tone.now();
        activeNotes.current.set(note, {
            startTime: currentTimeInSec,
            velocity: velocity / 127  // Normalize to 0-1 range
        });

        // Play the note on internal synth
        const midiNote = note;
        const noteName = Tone.Frequency(midiNote, "midi").toNote();

        synths.current[0]?.triggerAttack(
            noteName,
            undefined,
            velocity / 127
        );

        // If in duet mode, trigger AI response
        if (duetMode) {
            // We'll implement this later
            triggerDuetResponse(note, velocity);
        }
    };

    // Handle Note Off message
    const handleNoteOff = (note: number) => {
        // Get the start info for this note
        const startInfo = activeNotes.current.get(note);
        if (startInfo) {
            // Calculate duration
            const endTime = Tone.now();
            const duration = endTime - startInfo.startTime;

            // Add to recorded notes if we're recording
            if (startTime.current !== null) {
                recordedNotes.current.push({
                    note,
                    velocity: startInfo.velocity,
                    time: startInfo.startTime - startTime.current,
                    duration
                });
            }

            // Remove from active notes
            activeNotes.current.delete(note);

            // Release the note on internal synth
            const midiNote = note;
            const noteName = Tone.Frequency(midiNote, "midi").toNote();
            synths.current[0]?.triggerRelease(noteName);
        }
    };

    // Play a note on the MIDI output device
    const playNoteOnOutput = (note: number, velocity: number, delay: number, duration: number) => {
        if (!outputDevice.current) return;

        // Create MIDI messages for note on and note off
        const noteOnMessage = [0x90 + (selectedChannel - 1), note, velocity];
        const noteOffMessage = [0x80 + (selectedChannel - 1), note, 0];

        // Send note on with delay
        setTimeout(() => {
            outputDevice.current?.send(noteOnMessage);

            // Send note off after duration
            setTimeout(() => {
                outputDevice.current?.send(noteOffMessage);
            }, duration * 1000);
        }, delay * 1000);
    };

    // Clean up and initialize midi player
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

        // Create synths for playback
        // Synth 0: for user input and primary midi playback
        // Synth 1: for duet/AI responses
        for (let i = 0; i < 2; i++) {
            const synth = new Tone.PolySynth(Tone.Synth).toDestination();
            synth.volume.value = -5; // Reduce volume a bit
            synths.current.push(synth);
        }

        // Schedule all notes for playback
        midiData.tracks.forEach((track, i) => {
            // Only schedule notes for the selected channel if specified
            if (selectedChannel !== 0 && track.channel !== undefined && track.channel !== selectedChannel - 1) {
                return;
            }

            track.notes.forEach(note => {
                const id = Tone.Transport.schedule((time) => {
                    // Play on both internal synth and MIDI output if connected
                    synths.current[0]?.triggerAttackRelease(
                        note.name,
                        note.duration,
                        time,
                        note.velocity
                    );

                    // Also send to MIDI output if connected
                    if (outputDevice.current) {
                        const noteNumber = note.midi;
                        const noteOnVelocity = Math.floor(note.velocity * 127);

                        // Note On
                        outputDevice.current.send([0x90 + (selectedChannel - 1), noteNumber, noteOnVelocity]);

                        // Schedule Note Off
                        setTimeout(() => {
                            outputDevice.current?.send([0x80 + (selectedChannel - 1), noteNumber, 0]);
                        }, note.duration * 1000);
                    }
                }, note.time);

                scheduledNotes.current.push(id);
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
    }, [midiData, selectedChannel, onTimeUpdate]);

    // Start recording
    const startRecording = () => {
        recordedNotes.current = [];
        startTime.current = Tone.now();
        console.log("Recording started");
    };

    // Stop recording and create MIDI file
    const stopRecording = () => {
        if (startTime.current === null || recordedNotes.current.length === 0) {
            console.log("No notes recorded");
            return null;
        }

        // Create a new MIDI file
        const midi = new Midi();
        const track = midi.addTrack();

        // Set channel
        track.channel = selectedChannel - 1;

        // Add recorded notes
        recordedNotes.current.forEach(noteData => {
            track.addNote({
                midi: noteData.note,
                time: noteData.time,
                duration: noteData.duration,
                velocity: noteData.velocity
            });
        });

        // Reset recording state
        startTime.current = null;

        console.log(`Recording completed with ${recordedNotes.current.length} notes`);
        return midi;
    };

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