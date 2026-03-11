import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def get_track_data(track_type, data):
    """Extract pitch and time data from a track, handling both notes and chords."""
    notes = data[track_type]
    pitches = []
    times = []
    durations = []
    velocities = []

    for n in notes:
        # n is (pitch_or_chord, duration, velocity, offset)
        pitch_data = n[0]
        duration = n[1]
        velocity = n[2]
        offset = n[3]

        # Handle pitch: if tuple (chord), take the lowest pitch
        if isinstance(pitch_data, tuple):
            pitch = min(pitch_data)  # lowest note in chord
        else:
            pitch = pitch_data  # single note

        pitches.append(pitch)
        times.append(offset)
        durations.append(duration)
        velocities.append(velocity)

    return np.array(pitches), np.array(times), np.array(durations), np.array(velocities)

# Load the processed data
data_file = '/home/deginandor/Documents/Programming/Zha/dataset/processed/test_separated.pt'
if not os.path.exists(data_file):
    print(f"Data file not found: {data_file}")
    exit(1)

data = torch.load(data_file)
print(f"Loaded data with keys: {list(data.keys())}")

# Assume data is a dict with sequences
sequences = data.get('sequences', [])
if not sequences:
    print("No sequences found in data")
    exit(1)

# Take the first sequence for visualization
sample = sequences[0]['sequences']
print(f"Sample keys: {list(sample.keys())}")

# Create figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('MIDI Preprocessing Pipeline and Markov Input Format', fontsize=16)

track_types = ['full', 'melody', 'bass', 'drums']
colors = ['blue', 'green', 'red', 'orange']

for i, track_type in enumerate(track_types):
    ax = axes[i//2, i%2]

    if track_type in sample and sample[track_type]:
        pitches, times, durations, velocities = get_track_data(track_type, sample)

        # Scatter plot: time vs pitch, size by duration, color by velocity
        scatter = ax.scatter(times, pitches,
                           s=durations*100,  # size by duration
                           c=velocities,     # color by velocity
                           alpha=0.7, cmap='viridis', edgecolors='black', linewidth=0.5)

        ax.set_title(f'{track_type.capitalize()} Track ({len(pitches)} notes)')
        ax.set_xlabel('Time (beats)')
        ax.set_ylabel('MIDI Pitch')
        ax.grid(True, alpha=0.3)

        # Add colorbar for velocity
        if i == 0:  # only for first plot
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('Velocity')

    else:
        ax.text(0.5, 0.5, f'No {track_type} data', transform=ax.transAxes, ha='center', va='center')
        ax.set_title(f'{track_type.capitalize()} Track')

plt.tight_layout()

# Save the figure
output_dir = '/home/deginandor/Documents/Programming/Zha/docs/thesis/figures'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'midi_preprocessing_pipeline.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved figure to: {output_path}")

# Also save as PDF
pdf_path = os.path.join(output_dir, 'midi_preprocessing_pipeline.pdf')
plt.savefig(pdf_path, bbox_inches='tight')
print(f"Saved PDF to: {pdf_path}")

plt.show()