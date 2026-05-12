import { useState, useEffect, useRef, useCallback } from 'react';



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
  const previousNotesRef = useRef<Set<number>>(new Set());
  const requestAnimationFrameRef = useRef<number>(null);
  
  

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
      analyser.fftSize = 16384;
      analyser.smoothingTimeConstant = 0.2;
      source.connect(analyser);

      analyserRef.current = analyser;
      

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
    if (!analyserRef.current || !audioContext || !isActuallyRecording) return;

    const buffer = new Float32Array(analyserRef.current.fftSize);
    analyserRef.current.getFloatTimeDomainData(buffer);

    let rms = 0;
    for (let i = 0; i < buffer.length; i++) {
        rms += buffer[i] * buffer[i];
    }
    rms = Math.sqrt(rms / buffer.length);

    if (rms > threshold) {
      // Polyphonic frequency detection using FFT
      const bufferLength = analyserRef.current.frequencyBinCount;
      const freqData = new Float32Array(bufferLength);
      analyserRef.current.getFloatFrequencyData(freqData);
      
      const sampleRate = audioContext.sampleRate;
      const binWidth = sampleRate / analyserRef.current.fftSize;

      const peaks: { frequency: number, bin: number, magnitude: number }[] = [];
      const minDb = -60; // Noise floor for peaks
      
      for (let i = 2; i < bufferLength - 2; i++) {
        const mag = freqData[i];
        if (mag > minDb && mag > freqData[i-1] && mag > freqData[i+1] && mag > freqData[i-2] && mag > freqData[i+2]) {
          const alpha = freqData[i - 1];
          const beta = freqData[i];
          const gamma = freqData[i + 1];
          const p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma);
          const interpBin = i + p;
          const frequency = interpBin * binWidth;
          
          if (frequency > 60 && frequency < 1000) { 
             peaks.push({ frequency, bin: i, magnitude: mag });
          }
        }
      }

      peaks.sort((a, b) => b.magnitude - a.magnitude);
      
      const fundamentalPeaks = [];
      for (const peak of peaks) {
         let isHarmonic = false;
         for (const f of fundamentalPeaks) {
            const ratio = peak.frequency / f.frequency;
            // Filter out obvious integer harmonics
            if (Math.abs(ratio - Math.round(ratio)) < 0.08) {
               isHarmonic = true;
               break;
            }
         }
         if (!isHarmonic) {
             fundamentalPeaks.push(peak);
             if (fundamentalPeaks.length >= 6) break;
         }
      }

      const currentMidiNotes = new Set<number>();
      const now = Date.now();

      fundamentalPeaks.forEach((peak) => {
         const midiNote = freqToMidi(peak.frequency);
         currentMidiNotes.add(midiNote);

         if (!previousNotesRef.current.has(midiNote)) {
            const noteName = midiToNoteName(midiNote);
            setDetectedNotes((prev) => {
                return [...prev, { note: noteName, frequency: peak.frequency, midiNote, timeStart: now }];
            });
         }
      });

      previousNotesRef.current.forEach(oldNote => {
         if (!currentMidiNotes.has(oldNote)) {
            setDetectedNotes(prev => {
                const newNotes = [...prev];
                const activeIndex = newNotes.findLastIndex(n => n.midiNote === oldNote && !n.duration);
                if (activeIndex !== -1) {
                    newNotes[activeIndex] = {
                        ...newNotes[activeIndex],
                        duration: now - newNotes[activeIndex].timeStart
                    };
                }
                return newNotes;
            });
         }
      });

      previousNotesRef.current = currentMidiNotes;

    } else {
       if (previousNotesRef.current.size > 0) {
          const now = Date.now();
          setDetectedNotes(prev => {
             const newNotes = [...prev];
             for (let i = newNotes.length - 1; i >= 0; i--) {
                if (!newNotes[i].duration) {
                   newNotes[i] = {
                      ...newNotes[i],
                      duration: now - newNotes[i].timeStart
                   };
                }
             }
             return newNotes;
          });
          previousNotesRef.current.clear();
       }
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
