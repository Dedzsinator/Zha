import numpy as np
import torch

class MarkovChain:
    def __init__(self):
        # Transition matrix: [previous_note, next_note]
        self.transitions = np.zeros((128, 128), dtype=np.float32)
        self.trained = False

    def train(self, midi_sequences):
        """
        Train the Markov model on MIDI note sequences.

        Args:
            midi_sequences: List of note sequences, where each sequence
                            is a list of MIDI note numbers (0-127)
        """
        # Count transitions
        for sequence in midi_sequences:
            for i in range(len(sequence) - 1):
                prev_note = sequence[i]
                next_note = sequence[i + 1]
                if 0 <= prev_note < 128 and 0 <= next_note < 128:
                    self.transitions[prev_note, next_note] += 1

        # Normalize to get probabilities
        row_sums = self.transitions.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums = np.where(row_sums == 0, 1, row_sums)
        self.transitions = self.transitions / row_sums

        self.trained = True

    def save(self, filepath):
        """Save the transition matrix to a file"""
        np.save(filepath, self.transitions)

    def load(self, filepath):
        """Load the transition matrix from a file"""
        self.transitions = np.load(filepath)
        self.trained = True

    def predict_next_note(self, current_note):
        """Predict the next note given the current note"""
        if not self.trained:
            raise ValueError("Markov model is not trained yet")

        if not 0 <= current_note < 128:
            raise ValueError(f"Note value {current_note} out of range (0-127)")

        # Get probability distribution for next note
        probabilities = self.transitions[current_note]

        # Sample from distribution
        next_note = np.random.choice(128, p=probabilities)
        return next_note

    def generate_sequence(self, start_note=60, length=32):
        """Generate a sequence of notes starting with start_note"""
        if not self.trained:
            raise ValueError("Markov model is not trained yet")

        sequence = [start_note]
        current = start_note

        for _ in range(length - 1):
            current = self.predict_next_note(current)
            sequence.append(current)

        return sequence