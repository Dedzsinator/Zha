'use client';

import { useState } from 'react';
import { FaMusic, FaRobot, FaWrench } from 'react-icons/fa';

interface DuetModeProps {
    enabled: boolean;
    onToggle: (enabled: boolean) => void;
    responsiveness: number;
    onResponsivenessChange: (value: number) => void;
    harmonization: 'simple' | 'chord' | 'melody';
    onHarmonizationChange: (mode: 'simple' | 'chord' | 'melody') => void;
}

export default function DuetMode({
    enabled,
    onToggle,
    responsiveness,
    onResponsivenessChange,
    harmonization,
    onHarmonizationChange
}: DuetModeProps) {
    return (
        <div className="bg-white p-4 rounded-lg border border-gray-200">
            <h3 className="text-lg font-semibold mb-3 flex items-center">
                <FaMusic className="mr-2" /> Duet Mode
            </h3>

            <div className="space-y-4">
                {/* Enable/Disable Toggle */}
                <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-gray-700">Enable Duet Mode</span>
                    <button
                        className={`relative inline-flex h-6 w-11 items-center rounded-full ${enabled ? 'bg-blue-600' : 'bg-gray-200'}`}
                        onClick={() => onToggle(!enabled)}
                    >
                        <span
                            className={`inline-block h-4 w-4 transform rounded-full bg-white transition ${enabled ? 'translate-x-6' : 'translate-x-1'}`}
                        />
                    </button>
                </div>

                {enabled && (
                    <>
                        {/* Responsiveness Slider */}
                        <div>
                            <label htmlFor="responsiveness" className="block text-sm font-medium text-gray-700 mb-1 flex items-center">
                                <FaRobot className="mr-1" /> Responsiveness
                            </label>
                            <input
                                id="responsiveness"
                                type="range"
                                min="0"
                                max="1"
                                step="0.01"
                                value={responsiveness}
                                onChange={(e) => onResponsivenessChange(parseFloat(e.target.value))}
                                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                            />
                            <div className="flex justify-between text-xs text-gray-500 mt-1">
                                <span>Subtle</span>
                                <span>Responsive</span>
                            </div>
                        </div>

                        {/* Harmonization Mode */}
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2 flex items-center">
                                <FaWrench className="mr-1" /> Harmonization Style
                            </label>
                            <div className="grid grid-cols-3 gap-2">
                                <button
                                    onClick={() => onHarmonizationChange('simple')}
                                    className={`px-3 py-2 text-sm font-medium rounded-md ${harmonization === 'simple'
                                            ? 'bg-blue-100 text-blue-800 border border-blue-300'
                                            : 'bg-gray-100 text-gray-800 border border-gray-300'
                                        }`}
                                >
                                    Simple
                                </button>
                                <button
                                    onClick={() => onHarmonizationChange('chord')}
                                    className={`px-3 py-2 text-sm font-medium rounded-md ${harmonization === 'chord'
                                            ? 'bg-blue-100 text-blue-800 border border-blue-300'
                                            : 'bg-gray-100 text-gray-800 border border-gray-300'
                                        }`}
                                >
                                    Chords
                                </button>
                                <button
                                    onClick={() => onHarmonizationChange('melody')}
                                    className={`px-3 py-2 text-sm font-medium rounded-md ${harmonization === 'melody'
                                            ? 'bg-blue-100 text-blue-800 border border-blue-300'
                                            : 'bg-gray-100 text-gray-800 border border-gray-300'
                                        }`}
                                >
                                    Melody
                                </button>
                            </div>
                        </div>

                        <div className="text-sm text-gray-600 bg-gray-50 p-3 rounded-md">
                            <p className="font-medium mb-1">Tips:</p>
                            <ul className="list-disc list-inside space-y-1">
                                <li>Play a few notes to trigger a response</li>
                                <li>Try different speeds and note patterns</li>
                                <li>Change harmonization style for different response types</li>
                            </ul>
                        </div>
                    </>
                )}
            </div>
        </div>
    );
}