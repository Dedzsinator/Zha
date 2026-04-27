import { useState, useEffect, useRef, useCallback } from 'react';
import { PitchDetector } from 'pitchfinder/lib/index';
const Pitchfinder = require('pitchfinder');

export interface DetectedNote {
  note: string;
  midiNote: number;
  frequency: number;
  timeStart: number;
  duration?: number;
}

interface GuitarTrackerOptions {
  threshold?: number;   // Volume threshold to trigger onset
  sampleRate?: number;
}

// Map frequency to MIDI
const freqToMidi = (freq: number) => Math.round(69 + 12 * Math.log2(freq / 440));

// Map MIDI to Note Name
const midiToNoteName = (midi: number) => {
  const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  const octave = Math.floor(midi / 12) - 1;
  const name = noteNames[midi % 12];
  return `${name}${octave}`;
};

export const useGuitarTracker = (options: GuitarTrackerOptions = {}) => {
  const [isRecording, setIsRecording] = useState(false);
  const isRecordingRef = useRef(false);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [detectedNotes, setDetectedNotesState] = useState<DetectedNote[]>([]);
  const detectedNotesRef = useRef<DetectedNote[]>([]);

  const setDetectedNotes = (updater: (prev: DetectedNote[]) => DetectedNote[]) => {
      setDetectedNotesState((prev) => {
          const newNotes = updater(prev);
          detectedNotesRef.current = newNotes;
          return newNotes;
      });
  };
  const [currentBpm, setCurrentBpm] = useState<number>(120);
  const [currentScale, setCurrentScale] = useState<string>('C Major');
  const [audioContext, setAudioContext] = useState<AudioContext | null>(null);

  const analyserRef = useRef<AnalyserNode | null>(null);
  const pitchFinderRef = useRef<PitchDetector | null>(null);
  const requestAnimationFrameRef = useRef<number>(null);
  const previousNoteRef = useRef<number | null>(null);
  const activeNoteStartRef = useRef<number | null>(null);

  const threshold = options.threshold || 0.01;

  const startTracking = async () => {
    try {
      // Get audio from the Audio Interface (e.g. Focusrite Scarlett)
      const userStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          autoGainControl: false,
          noiseSuppression: false,
          latency: 0,
        },
      });

      setStream(userStream);
      setIsRecording(true);
      isRecordingRef.current = true;

      const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
      setAudioContext(ctx);

      const source = ctx.createMediaStreamSource(userStream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);

      analyserRef.current = analyser;
      pitchFinderRef.current = Pitchfinder.YIN({ sampleRate: ctx.sampleRate });

      detectPitch();
    } catch (err) {
      console.error('Error accessing audio device:', err);
    }
  };

  const stopTracking = () => {
    setIsRecording(false);
    isRecordingRef.current = false;
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      setStream(null);
    }
    if (audioContext) {
      audioContext.close();
      setAudioContext(null);
    }
    if (requestAnimationFrameRef.current) {
      cancelAnimationFrame(requestAnimationFrameRef.current);
    }
    // Calculate final stats
    calculateScaleAndBPM();
  };

  const detectPitch = () => {
    const isActuallyRecording = isRecordingRef.current;
    if (!analyserRef.current || !pitchFinderRef.current || !isActuallyRecording) return;

    const buffer = new Float32Array(analyserRef.current.fftSize);
    analyserRef.current.getFloatTimeDomainData(buffer);

    // Calculate RMS volume for onset detection
    let rms = 0;
    for (let i = 0; i < buffer.length; i++) {
        rms += buffer[i] * buffer[i];
    }
    rms = Math.sqrt(rms / buffer.length);

    if (rms > threshold) {
      const pitch = pitchFinderRef.current(buffer);
      if (pitch && pitch > 60 && pitch < 1200) { // Guitar range roughly 80Hz - 1200Hz
        const midiNote = freqToMidi(pitch);

        // Simple debounce: If we have a new note, or starting from silence
        if (midiNote !== previousNoteRef.current) {
          const now = Date.now();
          const noteName = midiToNoteName(midiNote);
          
          console.log(`Detected pitch: ${pitch} Hz, MIDI note: ${midiNote}`);

          setDetectedNotes((prev) => {
            // End the previous note
            if (prev.length > 0 && !prev[prev.length - 1].duration) {
              prev[prev.length - 1].duration = now - prev[prev.length - 1].timeStart;
            }
            return [...prev, { note: noteName, frequency: pitch, midiNote, timeStart: now }];
          });

          previousNoteRef.current = midiNote;
        }
      }
    } else {
       // Silence
       if (previousNoteRef.current !== null) {
          setDetectedNotes(prev => {
             const newNotes = [...prev];
             if(newNotes.length > 0 && !newNotes[newNotes.length-1].duration) {
                 newNotes[newNotes.length-1].duration = Date.now() - newNotes[newNotes.length-1].timeStart;
             }
             return newNotes;
          });
       }
       previousNoteRef.current = null;
    }

    if (isRecordingRef.current) {
        requestAnimationFrameRef.current = requestAnimationFrame(detectPitch);
        
    }
  };

  // Naive Scale and BPM estimation
  const calculateScaleAndBPM = useCallback(() => {
    const currentNotes = detectedNotesRef.current;
    if (currentNotes.length < 2) return;

    // Estimate BPM from inter-onset intervals
    let intervals = [];
    for (let i = 1; i < currentNotes.length; i++) {
      intervals.push(currentNotes[i].timeStart - currentNotes[i-1].timeStart);
    }
    const avgInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length;
    // bpm = 60000 / avg_interval (assuming quarter notes)
    // Naively assume playing quarter notes for estimation
    let calcBpm = Math.round(60000 / avgInterval);
    if (!isNaN(calcBpm)) {
        if (calcBpm > 240) calcBpm = Math.round(calcBpm / 2);
        if (calcBpm < 60) calcBpm = calcBpm * 2;
        setCurrentBpm(calcBpm);
    }

    // Estimate Scale (Naive: mostly just look at the notes used)
    // A better way would be using Krumhansl-Schmuckler, but for simple MVP, pick the most common base
    const noteCounts: Record<number, number> = {};
    currentNotes.forEach(n => {
        const pc = n.midiNote % 12; // Pitch class 0-11
        noteCounts[pc] = (noteCounts[pc] || 0) + 1;
    });
    // Find most frequent note to guess root
    let root = 0;
    let maxCount = 0;
    Object.keys(noteCounts).forEach(noteStr => {
        const pc = parseInt(noteStr);
        if (noteCounts[pc] > maxCount) {
            maxCount = noteCounts[pc];
            root = pc;
        }
    });

    const rootName = midiToNoteName(root).slice(0, -1); // remove octave
    setCurrentScale(`${rootName} Major/Minor`);
  }, []);


  // Calculate periodically
  useEffect(() => {
    if (isRecording) {
      const interval = setInterval(() => {
        calculateScaleAndBPM();
      }, 3000);
      return () => clearInterval(interval);
    }
  }, [isRecording, calculateScaleAndBPM]);


  return {
    isRecording,
    startTracking,
    stopTracking,
    detectedNotes,
    currentBpm,
    currentScale
  };
};
