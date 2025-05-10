import { Midi as ToneJsMidi } from '@tonejs/midi';

export interface Note {
  midi: number;
  name: string;
  time: number;
  duration: number;
  velocity: number;
}

export interface Track {
  name: string;
  notes: Note[];
}

export interface GenerationResponse {
  midi_url: string;
  audio_url?: string;
  message: string;
  key?: string;
  time_signature?: string;
  sections?: string;
}