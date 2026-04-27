import React from 'react';

interface ButtonProps {
    children: React.ReactNode;
    onClick?: () => void;
    disabled?: boolean;
    variant?: 'primary' | 'secondary' | 'outline';
    size?: 'sm' | 'md' | 'lg';
    className?: string;
    isLoading?: boolean;
    type?: 'button' | 'submit' | 'reset';
}

export default function Button({
    children,
    onClick,
    disabled = false,
    variant = 'primary',
    size = 'md',
    className = '',
    isLoading = false,
    type = 'button',
}: ButtonProps) {
    const baseClasses = 'rounded-md font-medium flex items-center justify-center';

    const variantClasses = {
        primary: 'bg-blue-500 text-white hover:bg-blue-600',
        secondary: 'bg-gray-200 text-gray-800 hover:bg-gray-300',
        outline: 'bg-transparent border border-gray-300 text-gray-700 hover:bg-gray-100',
    };

    const sizeClasses = {
        sm: 'text-sm py-1 px-3',
        md: 'py-2 px-4',
        lg: 'text-lg py-3 px-6',
    };

    const disabledClasses = disabled ? 'opacity-60 cursor-not-allowed' : 'cursor-pointer';

    return (
        <button
            type={type}
            onClick={onClick}
            disabled={disabled || isLoading}
            className={`${baseClasses} ${variantClasses[variant]} ${sizeClasses[size]} ${disabledClasses} ${className}`}
        >
            {isLoading ? (
                <>
                    <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-current" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Loading...
                </>
            ) : children}
        </button>
    );
}