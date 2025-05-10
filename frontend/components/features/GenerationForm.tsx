'use client';

import React, { useState } from 'react';
import axios from 'axios';
import { GenerationResponse } from '@/types';
import { FaMusic, FaRandom, FaVolumeUp, FaMagic } from 'react-icons/fa';

interface GenerationFormProps {
    onMidiGenerated: (file: File, response: GenerationResponse) => void;
}

const MODELS = [
    {
        id: 'markov',
        name: 'Markov Chain',
        description: 'Generate music with an expressive Markov model that follows music theory rules',
        icon: FaMusic,
        requiresInput: false
    },
    {
        id: 'structured_transformer',
        name: 'Structured Transformer',
        description: 'Generate structured music with distinct sections using transformer model',
        icon: FaRandom,
        requiresInput: true
    },
    {
        id: 'vae',
        name: 'Creative VAE',
        description: 'Variation and creativity with a variational autoencoder',
        icon: FaMagic,
        requiresInput: true
    },
    {
        id: 'combined',
        name: 'Combined Model',
        description: 'Best of all worlds: structure, creativity, and theory combined',
        icon: FaVolumeUp,
        requiresInput: true
    }
];

export default function GenerationForm({ onMidiGenerated }: GenerationFormProps) {
    const [selectedModel, setSelectedModel] = useState('markov');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Markov-specific parameters
    const [keyContext, setKeyContext] = useState('C major');
    const [complexity, setComplexity] = useState(0.8);
    const [length, setLength] = useState(96);

    // VAE and Combined parameters
    const [creativity, setCreativity] = useState(0.5);

    // Structured Transformer parameters
    const [numSections, setNumSections] = useState(3);
    const [sectionLength, setSectionLength] = useState(16);
    const [transitionSmoothness, setTransitionSmoothness] = useState(0.7);
    const [temperature, setTemperature] = useState(0.8);

    // Common parameters
    const [midiFile, setMidiFile] = useState<File | null>(null);
    const [duration, setDuration] = useState(30);
    const [instrument, setInstrument] = useState('piano');

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files;
        if (files && files.length > 0) {
            setMidiFile(files[0]);
        }
    };

    const getFormData = () => {
        const formData = new FormData();

        // Common parameters
        formData.append('duration', duration.toString());
        formData.append('instrument', instrument);
        formData.append('should_generate_audio', 'true');

        // Model-specific parameters
        if (selectedModel === 'markov') {
            formData.append('key_context', keyContext);
            formData.append('complexity', complexity.toString());
            formData.append('length', length.toString());
        } else if (selectedModel === 'vae') {
            if (!midiFile) throw new Error('MIDI file is required');
            formData.append('midi_file', midiFile);
            formData.append('creativity', creativity.toString());
        } else if (selectedModel === 'structured_transformer') {
            if (!midiFile) throw new Error('MIDI file is required');
            formData.append('midi_file', midiFile);
            formData.append('num_sections', numSections.toString());
            formData.append('section_length', sectionLength.toString());
            formData.append('transition_smoothness', transitionSmoothness.toString());
            formData.append('temperature', temperature.toString());
        } else if (selectedModel === 'combined') {
            if (!midiFile) throw new Error('MIDI file is required');
            formData.append('midi_file', midiFile);
            formData.append('creativity', creativity.toString());
        }

        return formData;
    };

    const handleGenerate = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        setError(null);

        try {
            const model = selectedModel;
            const requiresInput = MODELS.find(m => m.id === model)?.requiresInput;

            if (requiresInput && !midiFile) {
                throw new Error('Please upload a MIDI file');
            }

            const formData = getFormData();

            const response = await axios.post<GenerationResponse>(
                `http://localhost:8000/generate/${model}`,
                formData,
                { headers: { 'Content-Type': 'multipart/form-data' } }
            );

            // Get the MIDI file from the response
            if (response.data.midi_url) {
                const midiUrl = `http://localhost:8000${response.data.midi_url}`;
                const midiResponse = await axios.get(midiUrl, { responseType: 'blob' });
                const midiBlob = new Blob([midiResponse.data], { type: 'audio/midi' });
                const midiFileName = response.data.midi_url.split('/').pop() || 'generated.mid';
                const midiFile = new File([midiBlob], midiFileName, { type: 'audio/midi' });

                // Notify parent component
                onMidiGenerated(midiFile, response.data);
            }
        } catch (err: any) {
            console.error('Generation error:', err);
            setError(err.response?.data?.detail || err.message || 'An error occurred during generation');
        } finally {
            setLoading(false);
        }
    };

    // Get the currently selected model
    const currentModel = MODELS.find(model => model.id === selectedModel);
    const requiresInput = currentModel?.requiresInput || false;

    return (
        <div className="bg-white rounded-lg border border-gray-200 p-6">
            <h2 className="text-xl font-bold mb-4">Generate Music</h2>

            <form onSubmit={handleGenerate}>
                {/* Model Selection */}
                <div className="mb-6">
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                        Choose Generation Model
                    </label>
                    <div className="grid grid-cols-2 gap-2">
                        {MODELS.map((model) => (
                            <button
                                key={model.id}
                                type="button"
                                onClick={() => setSelectedModel(model.id)}
                                className={`p-3 rounded-lg border text-left transition-colors ${selectedModel === model.id
                                        ? 'border-blue-500 bg-blue-50 text-blue-700'
                                        : 'border-gray-200 hover:bg-gray-50'
                                    }`}
                            >
                                <div className="flex items-center">
                                    <model.icon className="h-5 w-5 mr-2" />
                                    <span className="font-medium">{model.name}</span>
                                </div>
                                <p className="text-xs mt-1 text-gray-500">{model.description}</p>
                            </button>
                        ))}
                    </div>
                </div>

                {/* MIDI File Upload (for models that need input) */}
                {requiresInput && (
                    <div className="mb-6">
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Upload MIDI File (Required)
                        </label>
                        <input
                            type="file"
                            accept=".mid,.midi"
                            onChange={handleFileChange}
                            className="w-full p-2 text-sm border rounded-md text-gray-500
                file:mr-4 file:py-2 file:px-4
                file:rounded-md file:border-0
                file:text-sm file:font-semibold
                file:bg-blue-50 file:text-blue-700
                hover:file:bg-blue-100"
                            required
                        />
                    </div>
                )}

                {/* Model-specific parameters */}
                {selectedModel === 'markov' && (
                    <div className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Musical Key
                            </label>
                            <select
                                value={keyContext}
                                onChange={(e) => setKeyContext(e.target.value)}
                                className="w-full p-2 border rounded-md"
                            >
                                <option value="C major">C major</option>
                                <option value="G major">G major</option>
                                <option value="D major">D major</option>
                                <option value="A major">A major</option>
                                <option value="E major">E major</option>
                                <option value="B major">B major</option>
                                <option value="F major">F major</option>
                                <option value="Bb major">B♭ major</option>
                                <option value="Eb major">E♭ major</option>
                                <option value="Ab major">A♭ major</option>
                                <option value="A minor">A minor</option>
                                <option value="E minor">E minor</option>
                                <option value="B minor">B minor</option>
                                <option value="F# minor">F# minor</option>
                                <option value="C# minor">C# minor</option>
                                <option value="D minor">D minor</option>
                                <option value="G minor">G minor</option>
                                <option value="C minor">C minor</option>
                            </select>
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Complexity: {complexity.toFixed(1)}
                            </label>
                            <input
                                type="range"
                                min="0"
                                max="1"
                                step="0.1"
                                value={complexity}
                                onChange={(e) => setComplexity(parseFloat(e.target.value))}
                                className="w-full"
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Length: {length} notes
                            </label>
                            <input
                                type="range"
                                min="32"
                                max="256"
                                step="16"
                                value={length}
                                onChange={(e) => setLength(parseInt(e.target.value))}
                                className="w-full"
                            />
                        </div>
                    </div>
                )}

                {(selectedModel === 'vae' || selectedModel === 'combined') && (
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Creativity: {creativity.toFixed(1)}
                        </label>
                        <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.1"
                            value={creativity}
                            onChange={(e) => setCreativity(parseFloat(e.target.value))}
                            className="w-full"
                        />
                    </div>
                )}

                {selectedModel === 'structured_transformer' && (
                    <div className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Number of Sections: {numSections}
                            </label>
                            <input
                                type="range"
                                min="1"
                                max="5"
                                step="1"
                                value={numSections}
                                onChange={(e) => setNumSections(parseInt(e.target.value))}
                                className="w-full"
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Section Length: {sectionLength} beats
                            </label>
                            <input
                                type="range"
                                min="8"
                                max="32"
                                step="4"
                                value={sectionLength}
                                onChange={(e) => setSectionLength(parseInt(e.target.value))}
                                className="w-full"
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Transition Smoothness: {transitionSmoothness.toFixed(1)}
                            </label>
                            <input
                                type="range"
                                min="0"
                                max="1"
                                step="0.1"
                                value={transitionSmoothness}
                                onChange={(e) => setTransitionSmoothness(parseFloat(e.target.value))}
                                className="w-full"
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Temperature: {temperature.toFixed(1)}
                            </label>
                            <input
                                type="range"
                                min="0.1"
                                max="1.5"
                                step="0.1"
                                value={temperature}
                                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                                className="w-full"
                            />
                        </div>
                    </div>
                )}

                {/* Common parameters */}
                <div className="mt-6 space-y-4">
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Duration: {duration} seconds
                        </label>
                        <input
                            type="range"
                            min="10"
                            max="120"
                            step="5"
                            value={duration}
                            onChange={(e) => setDuration(parseInt(e.target.value))}
                            className="w-full"
                        />
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Instrument
                        </label>
                        <select
                            value={instrument}
                            onChange={(e) => setInstrument(e.target.value)}
                            className="w-full p-2 border rounded-md"
                        >
                            <option value="piano">Piano</option>
                            <option value="guitar">Guitar</option>
                            <option value="violin">Violin</option>
                            <option value="cello">Cello</option>
                            <option value="flute">Flute</option>
                            <option value="trumpet">Trumpet</option>
                            <option value="organ">Organ</option>
                            <option value="choir">Choir</option>
                            <option value="strings">Strings</option>
                        </select>
                    </div>
                </div>

                {/* Error display */}
                {error && (
                    <div className="mt-4 p-3 bg-red-50 text-red-700 rounded-md text-sm">
                        {error}
                    </div>
                )}

                {/* Submit button */}
                <button
                    type="submit"
                    disabled={loading}
                    className={`mt-6 w-full py-2.5 px-4 rounded-md text-white font-medium 
            ${loading
                            ? 'bg-blue-300 cursor-not-allowed'
                            : 'bg-blue-600 hover:bg-blue-700'
                        }`}
                >
                    {loading ? (
                        <span className="flex items-center justify-center">
                            <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            Generating...
                        </span>
                    ) : (
                        'Generate Music'
                    )}
                </button>
            </form>
        </div>
    );
}