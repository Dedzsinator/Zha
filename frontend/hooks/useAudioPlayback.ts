'use client';

import { useState, useRef, useEffect } from 'react';

interface UseAudioPlaybackOptions {
  autoPlay?: boolean;
  loop?: boolean;
  volume?: number;
  onEnded?: () => void;
  onError?: (error: Error) => void;
}

interface AudioControls {
  play: () => Promise<void>;
  pause: () => void;
  stop: () => void;
  setVolume: (volume: number) => void;
  setPlaybackRate: (rate: number) => void;
  seek: (time: number) => void;
  toggleMute: () => void;
}

interface UseAudioPlaybackReturn {
  playing: boolean;
  loading: boolean;
  currentTime: number;
  duration: number;
  muted: boolean;
  volume: number;
  playbackRate: number;
  controls: AudioControls;
}

export function useAudioPlayback(
  audioSrc: string | null,
  options: UseAudioPlaybackOptions = {}
): UseAudioPlaybackReturn {
  const { 
    autoPlay = false, 
    loop = false, 
    volume = 1, 
    onEnded, 
    onError 
  } = options;
  
  // State variables
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [muted, setMuted] = useState(false);
  const [audioVolume, setAudioVolume] = useState(volume);
  const [playbackRate, setPlaybackRate] = useState(1);
  
  // Refs
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  // Initialize audio element
  useEffect(() => {
    if (!audioSrc) return;
    
    setLoading(true);
    
    // Create audio element if it doesn't exist
    if (!audioRef.current) {
      audioRef.current = new Audio();
      
      // Set initial properties
      audioRef.current.preload = 'metadata';
      audioRef.current.volume = audioVolume;
      audioRef.current.muted = muted;
      audioRef.current.playbackRate = playbackRate;
      audioRef.current.loop = loop;
      
      // Set up event listeners
      audioRef.current.addEventListener('loadedmetadata', () => {
        setDuration(audioRef.current?.duration || 0);
        setLoading(false);
      });
      
      audioRef.current.addEventListener('ended', () => {
        setPlaying(false);
        setCurrentTime(0);
        if (onEnded) onEnded();
      });
      
      audioRef.current.addEventListener('error', (e) => {
        setLoading(false);
        if (onError) onError(new Error('Error loading audio file'));
        console.error('Audio error:', e);
      });
    }
    
    // Update source
    audioRef.current.src = audioSrc;
    
    // Auto play if specified
    if (autoPlay) {
      audioRef.current.play().catch(err => {
        console.error('Auto play failed:', err);
      });
    }
    
    // Clean up
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.src = '';
      }
      
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [audioSrc]);
  
  // Set up time update interval
  useEffect(() => {
    if (!audioRef.current) return;
    
    if (playing) {
      // Update current time every 50ms
      intervalRef.current = setInterval(() => {
        if (audioRef.current) {
          setCurrentTime(audioRef.current.currentTime);
        }
      }, 50);
    } else if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [playing]);
  
  // Apply volume changes
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = audioVolume;
    }
  }, [audioVolume]);
  
  // Apply mute changes
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.muted = muted;
    }
  }, [muted]);
  
  // Apply playback rate changes
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.playbackRate = playbackRate;
    }
  }, [playbackRate]);
  
  // Apply loop changes
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.loop = loop;
    }
  }, [loop]);

  // Controls for the audio element
  const controls: AudioControls = {
    play: async () => {
      if (!audioRef.current) return;
      try {
        await audioRef.current.play();
        setPlaying(true);
      } catch (err) {
        console.error('Play error:', err);
        if (onError) onError(new Error('Failed to play audio'));
      }
    },
    
    pause: () => {
      if (!audioRef.current) return;
      audioRef.current.pause();
      setPlaying(false);
    },
    
    stop: () => {
      if (!audioRef.current) return;
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      setPlaying(false);
      setCurrentTime(0);
    },
    
    setVolume: (vol: number) => {
      const newVolume = Math.max(0, Math.min(1, vol));
      setAudioVolume(newVolume);
    },
    
    setPlaybackRate: (rate: number) => {
      setPlaybackRate(rate);
    },
    
    seek: (time: number) => {
      if (!audioRef.current) return;
      const newTime = Math.max(0, Math.min(time, audioRef.current.duration));
      audioRef.current.currentTime = newTime;
      setCurrentTime(newTime);
    },
    
    toggleMute: () => {
      setMuted(prev => !prev);
    }
  };
  
  return {
    playing,
    loading,
    currentTime,
    duration,
    muted,
    volume: audioVolume,
    playbackRate,
    controls
  };
}