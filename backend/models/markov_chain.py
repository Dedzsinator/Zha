import numpy as np
import torch
from collections import defaultdict
from music21 import analysis, chord, scale, stream, note, pitch, key, interval
from music21 import meter, duration, tempo
import random
import copy

class MarkovChain:
    def __init__(self, order=1):
        """
        Initialize a musically-aware Markov Chain model with time signature support.

        Args:
            order: The order of the Markov chain (number of previous notes to consider)
        """
        # Original single-note transition matrix
        self.transitions = np.zeros((128, 128), dtype=np.float32)

        # Advanced musical features
        self.order = order
        self.trained = False

        # Multi-order transitions for note sequences
        self.multi_order_transitions = defaultdict(lambda: defaultdict(float))

        # Chord transitions (represented as chord type to chord type)
        self.chord_transitions = defaultdict(lambda: defaultdict(float))

        # Note duration patterns (previous note duration to next note duration)
        self.duration_transitions = defaultdict(lambda: defaultdict(float))

        # Common scales/keys detected in training data
        self.common_keys = {}

        # Chord-to-scale relationships
        self.chord_to_scale = {}

        # Rhythm patterns (note duration sequences)
        self.rhythm_patterns = []

        # NEW: Time signature related data
        self.time_signatures = {}  # Store common time signatures
        self.beat_patterns = defaultdict(list)  # Beat patterns by time signature
        self.rhythmic_motifs = defaultdict(list)  # Rhythmic motifs by time signature
        self.beat_strength_transitions = defaultdict(lambda: np.zeros((5, 5)))  # Strong to weak beat transitions

        # NEW: Note density by time signature (notes per measure)
        self.note_density = defaultdict(lambda: 4)  # Default to 4 notes per measure

        # NEW: Beat positions (where notes typically fall within measures)
        self.beat_positions = defaultdict(lambda: defaultdict(int))

        # NEW: Note grouping patterns (how many consecutive notes appear)
        self.grouping_patterns = []

    def extract_musical_features(self, midi_sequences):
        """
        Extract musical features from MIDI sequences including time signatures.
        """
        # Create a music21 stream for analysis
        all_streams = []

        for sequence in midi_sequences:
            if not sequence or len(sequence) < 3:
                continue

            s = stream.Stream()
            for note_data in sequence:
                if isinstance(note_data, int):
                    # If just pitch info, create a note with default duration
                    n = note.Note(pitch=note_data)
                    n.duration.quarterLength = 0.5
                    s.append(n)
                elif isinstance(note_data, (list, tuple)) and len(note_data) >= 2:
                    # If pitch and duration info, use both
                    n = note.Note(pitch=note_data[0])
                    n.duration.quarterLength = note_data[1]
                    s.append(n)

            if len(s) > 0:
                # Add a default time signature if none exists
                if not s.getTimeSignatures():
                    ts = meter.TimeSignature('4/4')
                    s.insert(0, ts)

                # Add measures to properly analyze beat positions
                try:
                    s = s.makeMeasures()
                    all_streams.append(s)
                except Exception as e:
                    print(f"Warning: Could not create measures: {e}")
                    # Skip this stream if we can't add measures
                    continue

        # Extract musical features from all streams
        features = {
            'keys': [],
            'chords': [],
            'chord_progressions': [],
            'rhythm_patterns': [],
            'time_signatures': [],
            'beat_patterns': defaultdict(list),
            'note_density': defaultdict(list),
            'beat_positions': defaultdict(lambda: defaultdict(int)),
        }

        for s in all_streams:
            if len(s) < 4:
                continue

            # Extract time signature
            ts_objects = s.getTimeSignatures()
            if ts_objects:
                ts = ts_objects[0]  # Use first time signature
                ts_str = f"{ts.numerator}/{ts.denominator}"
                features['time_signatures'].append(ts_str)

                # Count notes per measure to determine density
                for measure in s.getElementsByClass('Measure'):
                    note_count = len(measure.notes)
                    if note_count > 0:  # Only count non-empty measures
                        features['note_density'][ts_str].append(note_count)

                # Analyze beat positions - where notes tend to fall in the measure
                for measure in s.getElementsByClass('Measure'):
                    for n in measure.notes:
                        try:
                            # FIX: Use getOffsetBySite instead of getOffsetInHierarchy
                            beat_position = n.getOffsetBySite(measure) % ts.barDuration.quarterLength
                            # Quantize to nearest 16th note
                            quantized_pos = round(beat_position * 4) / 4
                            features['beat_positions'][ts_str][quantized_pos] += 1
                        except Exception as e:
                            # If there's any error getting the offset, just continue
                            continue
                            
            # Try to detect key
            try:
                k = analysis.discrete.analyzeStream(s, 'key')
                if k:
                    key_name = str(k)
                    features['keys'].append(key_name)
            except Exception as e:
                # If key analysis fails, just continue
                pass
                
            # Extract rhythm patterns (simple implementation)
            try:
                durations = []
                for n in s.flatten().notes:
                    if hasattr(n, 'duration'):
                        durations.append(n.duration.quarterLength)
                        
                # Create rhythmic patterns from sequences of durations
                if len(durations) >= 4:
                    for i in range(len(durations) - 3):
                        pattern = tuple(durations[i:i+4])
                        features['rhythm_patterns'].append(pattern)
            except Exception as e:
                pass
                
        # THIS IS THE CRITICAL FIX - Return the features!
        return features

    def train(self, midi_sequences, progress_callback=None):
        """
        Train the enhanced Markov model including time signature analysis.
        """
        # Extract musical features for advanced analysis
        music_features = self.extract_musical_features(midi_sequences)

        # Process time signatures
        ts_counts = defaultdict(int)
        for ts in music_features['time_signatures']:
            ts_counts[ts] += 1

        # Store common time signatures with their frequencies
        total_ts = len(music_features['time_signatures']) or 1
        self.time_signatures = {ts: count/total_ts for ts, count in ts_counts.items()}

        # Process note density by time signature
        for ts, densities in music_features['note_density'].items():
            if densities:
                # Store the average number of notes per measure for this time signature
                self.note_density[ts] = sum(densities) / len(densities)

        # Process beat positions
        for ts, positions in music_features['beat_positions'].items():
            # Keep the positions normalized by total count
            total_notes = sum(positions.values()) or 1
            self.beat_positions[ts] = {pos: count/total_notes
                                     for pos, count in positions.items()}

        # Process beat patterns
        for ts, patterns in music_features['beat_patterns'].items():
            # Count occurrences of each pattern
            pattern_counts = defaultdict(int)
            for pattern in patterns:
                pattern_counts[pattern] += 1

            # Store most common patterns for each time signature (up to 10)
            sorted_patterns = sorted(pattern_counts.items(),
                                  key=lambda x: x[1], reverse=True)[:10]
            self.rhythmic_motifs[ts] = [pattern for pattern, _ in sorted_patterns]

            # Track note grouping patterns (sequences of similar durations)
            for pattern in patterns:
                if all(d == pattern[0] for d in pattern):  # All same duration
                    self.grouping_patterns.append((pattern[0], len(pattern)))

        # Train existing features
        super_train_result = self._train_base_features(midi_sequences, progress_callback)

        # Display additional statistics
        print(f"Detected {len(self.time_signatures)} common time signatures")
        for ts, freq in self.time_signatures.items():
            print(f"  - {ts}: {freq:.2%}")
            if ts in self.note_density:
                print(f"    Avg. notes per measure: {self.note_density[ts]:.1f}")

        return super_train_result

    def _train_base_features(self, midi_sequences, progress_callback=None):
        """Internal method to train original features (refactored from original train)"""
        # Existing training code from the original train method
        # Store the most common keys detected from music_features

        # Extract musical features for advanced analysis
        music_features = self.extract_musical_features(midi_sequences)

        # Store the most common keys detected
        key_counts = defaultdict(int)
        for k in music_features['keys']:
            key_counts[k] += 1

        self.common_keys = {k: count/len(music_features['keys'])
                          for k, count in key_counts.items()
                          if count > 2}  # Only keep keys appearing multiple times

        # Store common chord progressions
        chord_prog_counts = defaultdict(int)
        for prog in music_features['chord_progressions']:
            chord_prog_counts[prog] += 1

        # Keep the top 10 chord progressions
        sorted_progs = sorted(chord_prog_counts.items(),
                             key=lambda x: x[1], reverse=True)[:10]
        self.common_chord_progressions = {prog: count for prog, count in sorted_progs}

        # Store rhythm patterns
        rhythm_pattern_counts = defaultdict(int)
        for pattern in music_features['rhythm_patterns']:
            rhythm_pattern_counts[pattern] += 1

        # Keep the top rhythm patterns
        sorted_patterns = sorted(rhythm_pattern_counts.items(),
                               key=lambda x: x[1], reverse=True)[:10]
        self.rhythm_patterns = {pattern: count for pattern, count in sorted_patterns}

        # Train traditional note transition matrix
        print("Training note transitions...")
        for sequence in midi_sequences:
            if progress_callback:
                progress_callback(1)  # Report progress

            # Extract just pitches if sequence has pitch-duration pairs
            pitches = []
            for note_data in sequence:
                if isinstance(note_data, int):
                    pitches.append(note_data)
                elif isinstance(note_data, (list, tuple)) and len(note_data) >= 1:
                    pitches.append(note_data[0])

            if not pitches:
                continue

            # Single order transitions
            for i in range(len(pitches) - 1):
                prev_note = pitches[i]
                next_note = pitches[i + 1]
                if 0 <= prev_note < 128 and 0 <= next_note < 128:
                    self.transitions[prev_note, next_note] += 1

            # Multi-order transitions
            if self.order > 1:
                for i in range(len(pitches) - self.order):
                    # Create a context of n previous notes
                    context = tuple(pitches[i:i+self.order])
                    next_note = pitches[i+self.order]

                    if all(0 <= n < 128 for n in context) and 0 <= next_note < 128:
                        self.multi_order_transitions[context][next_note] += 1

        # Normalize the traditional transition matrix
        row_sums = self.transitions.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        self.transitions = self.transitions / row_sums

        # Normalize multi-order transitions
        for context, transitions in self.multi_order_transitions.items():
            total = sum(transitions.values())
            if total > 0:
                for next_note, count in transitions.items():
                    self.multi_order_transitions[context][next_note] = count / total

        self.trained = True
        print("Training complete!")

        # Display some statistics
        print(f"Model trained with order {self.order}")
        print(f"Detected {len(self.common_keys)} common keys")
        print(f"Captured {len(self.common_chord_progressions)} common chord progressions")
        print(f"Learned {len(self.rhythm_patterns)} rhythm patterns")

        return True

    def save(self, filepath):
        """Save the transition matrix and musical features to a file"""
        # Convert defaultdicts to regular dicts for saving
        data_to_save = {
            'transitions': self.transitions,
            'order': self.order,
            'common_keys': dict(self.common_keys),
            'common_chord_progressions': dict(self.common_chord_progressions),
            'rhythm_patterns': dict(self.rhythm_patterns),
            'time_signatures': dict(self.time_signatures),
            'note_density': dict(self.note_density),
            'grouping_patterns': self.grouping_patterns
        }

        # Convert nested defaultdicts to regular dicts
        beat_positions = {}
        for ts, pos_dict in self.beat_positions.items():
            beat_positions[ts] = dict(pos_dict)
        data_to_save['beat_positions'] = beat_positions

        rhythmic_motifs = {}
        for ts, motifs in self.rhythmic_motifs.items():
            rhythmic_motifs[ts] = list(motifs)
        data_to_save['rhythmic_motifs'] = rhythmic_motifs

        # Multi-order transitions
        multi_order = {}
        for context, transitions in self.multi_order_transitions.items():
            multi_order[context] = dict(transitions)
        data_to_save['multi_order_transitions'] = multi_order

        np.save(filepath, data_to_save, allow_pickle=True)

    def load(self, filepath):
        """Load the transition matrix and musical features from a file"""
        data = np.load(filepath, allow_pickle=True).item()

        # Load the traditional transition matrix
        self.transitions = data['transitions']

        # Load musical features if available
        if 'order' in data:
            self.order = data['order']
        if 'common_keys' in data:
            self.common_keys = defaultdict(float, data['common_keys'])
        if 'common_chord_progressions' in data:
            self.common_chord_progressions = defaultdict(float, data['common_chord_progressions'])
        if 'rhythm_patterns' in data:
            self.rhythm_patterns = defaultdict(float, data['rhythm_patterns'])

        # Load time signature related data
        if 'time_signatures' in data:
            self.time_signatures = data['time_signatures']
        if 'note_density' in data:
            self.note_density = defaultdict(lambda: 4, data['note_density'])
        if 'beat_positions' in data:
            self.beat_positions = defaultdict(lambda: defaultdict(int),
                                           {ts: defaultdict(int, pos)
                                            for ts, pos in data['beat_positions'].items()})
        if 'rhythmic_motifs' in data:
            self.rhythmic_motifs = defaultdict(list,
                                            {ts: motifs for ts, motifs in data['rhythmic_motifs'].items()})
        if 'grouping_patterns' in data:
            self.grouping_patterns = data['grouping_patterns']

        # Load multi-order transitions if available
        if 'multi_order_transitions' in data:
            for context, transitions in data['multi_order_transitions'].items():
                self.multi_order_transitions[context] = defaultdict(float, transitions)

        self.trained = True
        print(f"Loaded model with order {self.order}")

        # Display time signature information
        if self.time_signatures:
            print("Loaded time signatures:")
            for ts, freq in self.time_signatures.items():
                print(f"  - {ts}: {freq:.2%}")

    def generate_with_chords(self, key_context=None, length=32, time_signature="4/4"):
        """
        Generate a sequence with underlying chord progression and time signature awareness

        Args:
            key_context: Optional key to use (e.g. 'C major')
            length: Number of notes to generate
            time_signature: Time signature to use (e.g. "4/4")

        Returns:
            Dictionary with notes, durations, chords and timing
        """
        try:
            # Handle key context format issues
            cleaned_key_context = self._clean_key_context(key_context)

            # Generate a chord progression
            chords = self.generate_chord_progression(key_context=cleaned_key_context)

            # Determine start note based on key
            start_note = self._determine_start_note(cleaned_key_context)

            # Time signature handling
            if time_signature not in self.time_signatures and self.time_signatures:
                # Use the most common time signature
                time_signature = max(self.time_signatures.items(), key=lambda x: x[1])[0]

            # Get note density for this time signature
            target_density = self.note_density.get(time_signature, 4)

            # Calculate total measures needed
            try:
                numerator, denominator = map(int, time_signature.split('/'))
                beats_per_measure = numerator
            except:
                # Default to 4/4 if parsing fails
                beats_per_measure = 4

            target_measures = max(1, length // target_density)
            actual_length = int(target_measures * target_density)

            # Generate notes with time signature awareness
            notes_with_timings = self.generate_rhythmic_sequence(
                start_note=start_note,
                key_context=cleaned_key_context,
                length=actual_length,
                time_signature=time_signature,
                measures=target_measures
            )

            # Extract notes and durations
            notes = [note for note, _, _ in notes_with_timings]
            durations = [duration for _, duration, _ in notes_with_timings]
            beat_positions = [position for _, _, position in notes_with_timings]

            # Map chords to sections of the note sequence
            notes_per_chord = max(1, actual_length // len(chords))
            chord_sequence = []

            for chord_idx, chord_name in enumerate(chords):
                # Get the range of notes for this chord
                start_idx = chord_idx * notes_per_chord
                end_idx = (chord_idx + 1) * notes_per_chord
                if chord_idx == len(chords) - 1:
                    end_idx = actual_length  # Make sure we include all remaining notes

                # Assign this chord to all notes in the range
                chord_sequence.extend([chord_name] * (end_idx - start_idx))

            # Ensure all lists are the same length
            min_length = min(len(notes), len(durations), len(chord_sequence), len(beat_positions))

            return {
                'notes': notes[:min_length],
                'durations': durations[:min_length],
                'chords': chord_sequence[:min_length],
                'beat_positions': beat_positions[:min_length],
                'key': cleaned_key_context,
                'time_signature': time_signature
            }

        except Exception as e:
            print(f"Error in generate_with_chords: {e}")
            # Fallback: Generate simple sequence without chords
            notes = self.generate_sequence(
                start_note=60,
                length=length,
                key_context=None
            )

            # Create simple default durations (quarter notes)
            durations = [0.5] * len(notes)

            return {
                'notes': notes,
                'durations': durations,
                'chords': ['C major'] * length,
                'beat_positions': list(range(length)),
                'key': 'C major',
                'time_signature': '4/4'
            }

    def generate_rhythmic_sequence(self, start_note=60, key_context=None,
                                  length=32, time_signature="4/4", measures=8):
        """
        Generate a sequence with rhythmic awareness based on time signature

        Returns:
            List of (note, duration, beat_position) tuples
        """
        if not self.trained:
            raise ValueError("Markov model is not trained yet")

        # Parse time signature
        try:
            numerator, denominator = map(int, time_signature.split('/'))
            beats_per_measure = numerator
            beat_value = 4 / denominator  # Quarter note = 1.0
        except:
            beats_per_measure = 4
            beat_value = 1.0

        # Determine if we have rhythmic patterns for this time signature
        use_patterns = time_signature in self.rhythmic_motifs and self.rhythmic_motifs[time_signature]

        # Generate pitch sequence first
        pitch_sequence = self.generate_sequence(
            start_note=start_note,
            length=length,
            key_context=key_context
        )

        # Calculate total beats in the piece
        total_beats = measures * beats_per_measure

        # Generate rhythm pattern
        if use_patterns:
            # Use learned rhythmic patterns
            rhythm_pattern = []
            remaining_beats = total_beats

            while remaining_beats > 0:
                # Choose a random pattern from the learned patterns
                if self.rhythmic_motifs[time_signature]:
                    pattern = random.choice(self.rhythmic_motifs[time_signature])
                    pattern_duration = sum(pattern)

                    # Scale pattern if needed
                    if pattern_duration > remaining_beats:
                        scale_factor = remaining_beats / pattern_duration
                        pattern = tuple(d * scale_factor for d in pattern)
                        pattern_duration = remaining_beats

                    rhythm_pattern.extend(pattern)
                    remaining_beats -= pattern_duration
                else:
                    # Fallback to quarter notes
                    rhythm_pattern.append(1.0)
                    remaining_beats -= 1.0
        else:
            # Use default rhythmic pattern based on time signature
            if time_signature == "4/4":
                # Common 4/4 patterns
                patterns = [
                    [1.0, 1.0, 1.0, 1.0],  # Four quarters
                    [0.5, 0.5, 0.5, 0.5, 1.0, 1.0],  # Four eighths + two quarters
                    [1.0, 0.5, 0.5, 1.0, 1.0],  # Quarter + two eighths + two quarters
                    [2.0, 1.0, 1.0],  # Half + two quarters
                ]
            elif time_signature == "3/4":
                # Common 3/4 patterns
                patterns = [
                    [1.0, 1.0, 1.0],  # Three quarters
                    [0.5, 0.5, 0.5, 0.5, 1.0],  # Four eighths + quarter
                    [1.5, 1.5],  # Dotted quarter + dotted quarter
                ]
            elif time_signature == "6/8":
                # Common 6/8 patterns (note: 6/8 is counted in 2 beats of 3 eighth notes each)
                patterns = [
                    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # Six eighths
                    [1.0, 0.5, 0.5, 0.5, 0.5],  # Quarter + four eighths
                    [1.5, 1.5],  # Dotted quarter + dotted quarter
                ]
            else:
                # Default pattern - all quarter notes
                patterns = [[1.0] * beats_per_measure]

            # Fill with patterns
            rhythm_pattern = []
            for _ in range(measures):
                rhythm_pattern.extend(random.choice(patterns))

        # Trim or extend to match the required length
        if len(rhythm_pattern) < length:
            # Extend with quarter notes
            rhythm_pattern.extend([1.0] * (length - len(rhythm_pattern)))
        elif len(rhythm_pattern) > length:
            rhythm_pattern = rhythm_pattern[:length]

        # Create sequence with pitches, durations and beat positions
        beat_position = 0.0
        result = []

        for i in range(min(len(pitch_sequence), len(rhythm_pattern))):
            note_val = pitch_sequence[i]
            duration_val = rhythm_pattern[i]
            position = beat_position % (beats_per_measure * beat_value)

            result.append((note_val, duration_val, position))
            beat_position += duration_val

        return result

    def _clean_key_context(self, key_context):
        """Helper method to clean and standardize key context"""
        if not key_context:
            # Use a default or random key
            if self.common_keys:
                return random.choice(list(self.common_keys.keys()))
            return "C major"

        # Check if it's in proper format
        if ' ' not in key_context:
            # Assume it's just the key name like "C" and add "major" by default
            return f"{key_context} major"

        return key_context

    def _determine_start_note(self, key_context):
        """Helper method to determine start note from key context"""
        start_note = 60  # Default to middle C

        if key_context:
            try:
                # Handle key format issues
                try:
                    k = key.Key(key_context)
                except:
                    # Try alternative format
                    parts = key_context.split()
                    if len(parts) == 2:
                        tonic, mode = parts
                        k = key.Key(tonic, mode.lower())
                    else:
                        # Just use the root as key name
                        k = key.Key(parts[0])

                start_note = k.tonic.midi
            except Exception as e:
                print(f"Warning: Could not interpret key '{key_context}': {e}")

        return start_note

    def generate_expressive_sequence(self, key_context=None, length=64, complexity=0.7):
        """
        Generate a more musically expressive sequence with varied rhythms

        Args:
            key_context: Key to use (e.g. 'C major')
            length: Target length of the sequence
            complexity: How complex the rhythm should be (0.0-1.0)

        Returns:
            Dictionary with notes, durations and other musical features
        """
        # Determine time signature based on complexity
        if complexity < 0.3:
            time_sig = "4/4"  # Simpler
        elif complexity < 0.7:
            time_sig_options = ["4/4", "3/4"]
            weights = [0.7, 0.3]
            time_sig = random.choices(time_sig_options, weights=weights)[0]
        else:
            time_sig_options = ["4/4", "3/4", "6/8", "5/4", "7/8"]
            weights = [0.5, 0.2, 0.15, 0.1, 0.05]
            time_sig = random.choices(time_sig_options, weights=weights)[0]

        # Calculate a rhythmically appropriate length
        if time_sig == "4/4":
            measures = 4 + round(complexity * 8)  # 4-12 measures based on complexity
            notes_per_measure = 4 + round(complexity * 4)  # 4-8 notes per measure
        elif time_sig == "3/4":
            measures = 4 + round(complexity * 8)
            notes_per_measure = 3 + round(complexity * 3)
        else:
            measures = 4 + round(complexity * 6)
            notes_per_measure = 4 + round(complexity * 6)

        target_length = measures * notes_per_measure

        # Generate the sequence with enhanced rhythm
        return self.generate_with_chords(
            key_context=key_context,
            length=target_length,
            time_signature=time_sig
        )