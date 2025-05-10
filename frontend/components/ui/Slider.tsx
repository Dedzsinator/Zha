'use client';

import React, { useState, useRef, useEffect } from 'react';

interface SliderProps {
    min: number;
    max: number;
    step?: number;
    value: number;
    onChange: (value: number) => void;
    disabled?: boolean;
    className?: string;
    label?: string;
    showValue?: boolean;
    valueFormatter?: (value: number) => string;
    color?: 'blue' | 'red' | 'green' | 'purple';
}

export default function Slider({
    min,
    max,
    step = 1,
    value,
    onChange,
    disabled = false,
    className = '',
    label,
    showValue = false,
    valueFormatter = (value) => value.toString(),
    color = 'blue',
}: SliderProps) {
    const [isDragging, setIsDragging] = useState(false);
    const sliderRef = useRef<HTMLDivElement>(null);

    // Define color variants
    const colorVariants = {
        blue: 'bg-blue-500',
        red: 'bg-red-500',
        green: 'bg-green-500',
        purple: 'bg-purple-500',
    };

    // Calculate percentage for styling
    const percentage = ((value - min) / (max - min)) * 100;

    // Handle mouse/touch events for custom dragging behavior
    const handleMouseDown = (e: React.MouseEvent) => {
        if (disabled) return;
        setIsDragging(true);
        updateValueFromEvent(e);
    };

    const updateValueFromEvent = (e: React.MouseEvent | MouseEvent | React.TouchEvent | TouchEvent) => {
        if (!sliderRef.current || disabled) return;

        const rect = sliderRef.current.getBoundingClientRect();
        const clientX = 'touches' in e ? e.touches[0].clientX : 'clientX' in e ? e.clientX : 0;
        let percentage = (clientX - rect.left) / rect.width;

        // Clamp percentage between 0 and 1
        percentage = Math.max(0, Math.min(percentage, 1));

        // Calculate new value
        let newValue = min + percentage * (max - min);

        // Apply step if provided
        if (step) {
            newValue = Math.round(newValue / step) * step;
        }

        // Ensure value is within bounds
        newValue = Math.max(min, Math.min(newValue, max));

        onChange(newValue);
    };

    // Set up event listeners for mouse/touch interactions
    useEffect(() => {
        const handleMouseMove = (e: MouseEvent) => {
            if (isDragging) {
                updateValueFromEvent(e);
            }
        };

        const handleMouseUp = () => {
            setIsDragging(false);
        };

        if (isDragging) {
            document.addEventListener('mousemove', handleMouseMove);
            document.addEventListener('mouseup', handleMouseUp);
            document.addEventListener('touchmove', handleMouseMove as any);
            document.addEventListener('touchend', handleMouseUp);
        }

        return () => {
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', handleMouseUp);
            document.removeEventListener('touchmove', handleMouseMove as any);
            document.removeEventListener('touchend', handleMouseUp);
        };
    }, [isDragging]);

    return (
        <div className={`w-full ${className}`}>
            {label && (
                <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium text-gray-700">{label}</span>
                    {showValue && (
                        <span className="text-sm text-gray-500 font-medium">
                            {valueFormatter(value)}
                        </span>
                    )}
                </div>
            )}

            <div
                ref={sliderRef}
                className={`relative h-2 rounded-md bg-gray-200 ${disabled ? 'opacity-50' : 'cursor-pointer'}`}
                onMouseDown={handleMouseDown}
                onTouchStart={handleMouseDown as any}
            >
                {/* Track fill */}
                <div
                    className={`absolute h-full rounded-md ${colorVariants[color]} ${disabled ? 'opacity-40' : ''}`}
                    style={{ width: `${percentage}%` }}
                />

                {/* Thumb */}
                <div
                    className={`absolute top-1/2 -translate-y-1/2 -ml-2 w-4 h-4 rounded-full border-2 border-white ${colorVariants[color]} shadow-md ${disabled ? 'opacity-40' : ''}`}
                    style={{ left: `${percentage}%` }}
                />
            </div>
        </div>
    );
}