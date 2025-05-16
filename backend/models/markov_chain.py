import numpy as np
import random
from collections import defaultdict, Counter
from music21 import note, pitch, scale, key, chord, stream, roman, meter
import pickle
import logging

logger = logging.getLogger(__name__)

class MarkovChain:
    """
    Efficient musical Markov Chain model with multiple transition types and music theory awareness.
    """
    def __init__(self, order=1, max_interval=12):
        # Core transition matrices
        self.transitions = np.zeros((128, 128), dtype=np.float32)
        self.max_interval = max_interval
        self.interval_range = 2 * max_interval + 1
        self.interval_transitions = {}  # Sparse dict instead of defaultdict for better memory usage
        
        # Model configuration
        self.order = order
        self.trained = False
        
        # Musical features - unified storage approach
        self.musical_features = {
            'multi_order_transitions': {},  # {context_tuple: {next_note: probability}}
            'chord_transitions': {},  # {chord1: {chord2: probability}}
            'common_chord_progressions': {},  # {(chord1, chord2): count}
            'roman_numeral_transitions': {},  # {key: {(rn1, rn2): probability}}
            'duration_transitions': {},  # {prev_duration: {next_duration: probability}}
            'common_keys': {},  # {key_name: prevalence}
            'chord_to_scale': {},  # {chord_name: {scale_notes}}
            'rhythm_patterns': {},  # {(duration, beat_strength): count}
            'time_signatures': {},  # {time_sig: prevalence}
            'beat_patterns': {},  # {time_sig: [[beat_patterns]]}
            'rhythmic_motifs': {},  # {time_sig: [[rhythmic_motifs]]}
            'beat_strength_transitions': {},  # {time_sig: numpy_matrix}
            'note_density': {},  # {time_sig: avg_density}
            'beat_positions': {},  # {time_sig: {position: frequency}}
            'grouping_patterns': []  # [[patterns]]
        }
        
    def extract_musical_features(self, midi_sequences):
        """Streamlined extraction of musical features from MIDI sequences"""
        logger.info("Extracting musical features...")
        
        all_streams = self._convert_sequences_to_streams(midi_sequences)
        if not all_streams:
            logger.warning("No valid streams created from MIDI sequences")
            return {}
            
        features = {
            'keys': [],
            'chord_progressions': [], 
            'rhythm_patterns': [], 
            'time_signatures': [],
            'note_density': defaultdict(list),
            'beat_positions': defaultdict(lambda: defaultdict(int)),
            'roman_numeral_progressions': defaultdict(list)
        }
        
        for s in all_streams:
            if len(s.notes) < 4:
                continue
                
            # Extract features from each stream concisely
            self._extract_key_features(s, features)
            self._extract_rhythm_features(s, features)
            self._extract_chord_features(s, features)
            
        return features
        
    def _convert_sequences_to_streams(self, midi_sequences):
        """Convert MIDI sequences to music21 streams efficiently"""
        all_streams = []
        logger.info(f"Converting {len(midi_sequences)} sequences to streams...")
        
        # Use at most 100 sequences to avoid excessive processing time
        sample_size = min(len(midi_sequences), 100)
        if len(midi_sequences) > sample_size:
            logger.info(f"Sampling {sample_size} sequences for feature extraction to improve performance")
            
        # Process a subset of sequences with progress reporting
        for seq_idx, sequence in enumerate(midi_sequences[:sample_size]):
            if seq_idx > 0 and seq_idx % 10 == 0:
                logger.info(f"Processed {seq_idx}/{sample_size} sequences")
                
            if not sequence or len(sequence) < 3:
                continue
                
            try:
                s = stream.Stream()
                current_offset = 0.0
                
                # Handle the format from extract_note_sequence_from_score
                for note_data in sequence:
                    # Check if note_data is a tuple of (pitch, duration)
                    if isinstance(note_data, tuple) and len(note_data) == 2:
                        pitch_val, duration_val = note_data
                        
                        # Create note
                        n = note.Note(pitch=pitch_val)
                        n.duration.quarterLength = duration_val
                        s.insert(current_offset, n)
                        current_offset += duration_val
                
                # Only process if we added notes
                if len(s.notes) > 0:
                    # Add default time signature if none exists
                    if not s.getTimeSignatures():
                        s.insert(0, meter.TimeSignature('4/4'))
                    
                    # Create measures but with a timeout in case of issues
                    try:
                        s.makeMeasures(inPlace=True)
                        all_streams.append(s)
                    except Exception as e:
                        logger.debug(f"Could not create measures: {e}")
                        
            except Exception as e:
                logger.debug(f"Error converting sequence to stream: {e}")
        
        logger.info(f"Successfully converted {len(all_streams)}/{sample_size} sequences to streams")
        return all_streams

    def _extract_key_features(self, s, features):
        """Extract key-related features from a stream"""
        try:
            key_analysis = s.analyze('key')
            if key_analysis:
                key_name = str(key_analysis)
                features['keys'].append(key_name)
                return key_analysis
        except Exception:
            pass
        return None
        
    def _extract_rhythm_features(self, s, features):
        """Extract rhythm-related features from a stream"""
        ts = s.getTimeSignatures()[0] if s.getTimeSignatures() else meter.TimeSignature('4/4')
        ts_str = f"{ts.numerator}/{ts.denominator}"
        features['time_signatures'].append(ts_str)
        
        for measure in s.getElementsByClass('Measure'):
            measure_note_count = 0
            measure_rhythm = []
            
            for n in measure.notes:
                try:
                    # Beat position and strength
                    beat_pos = n.getOffsetBySite(measure)
                    beat_strength = ts.getAccentWeight(beat_pos, forceBeatStrength=True)
                    quantized_pos = round(beat_pos % ts.barDuration.quarterLength * 4) / 4
                    features['beat_positions'][ts_str][quantized_pos] += 1
                    
                    # Store rhythm pattern
                    strength_category = 2 if beat_strength >= 0.9 else (1 if beat_strength >= 0.4 else 0)
                    measure_rhythm.append((n.duration.quarterLength, strength_category))
                    
                    measure_note_count += 1
                except Exception:
                    continue
                    
            if measure_note_count > 0:
                features['note_density'][ts_str].append(measure_note_count)
                
            if len(measure_rhythm) >= 2:
                for i in range(len(measure_rhythm) - 1):
                    features['rhythm_patterns'].append(tuple(measure_rhythm[i:i+2]))
    
    def _extract_chord_features(self, s, features):
        """Extract chord-related features from a stream"""
        current_key = self._extract_key_features(s, features)
        if not current_key:
            return
            
        try:
            chordified = s.chordify()
            progression = []
            roman_progression = []
            last_chord_symbol = None
            
            for ch in chordified.recurse().getElementsByClass('Chord'):
                chord_symbol = f"{ch.root().name} {ch.quality}"
                
                if chord_symbol != last_chord_symbol:
                    progression.append(chord_symbol)
                    last_chord_symbol = chord_symbol
                    
                    if current_key:
                        try:
                            rn = roman.romanNumeralFromChord(ch, current_key)
                            roman_progression.append(rn.figure)
                        except Exception:
                            pass
            
            if len(progression) >= 2:
                for i in range(len(progression) - 1):
                    features['chord_progressions'].append(tuple(progression[i:i+2]))
                    
            if current_key and len(roman_progression) >= 2:
                key_name = str(current_key)
                for i in range(len(roman_progression) - 1):
                    features['roman_numeral_progressions'][key_name].append(
                        tuple(roman_progression[i:i+2])
                    )
        except Exception:
            pass
            
    def train(self, midi_sequences, progress_callback=None):
        """Train the model with optimized flow and memory usage"""
        logger.info("Starting model training...")
        
        # Extract musical features efficiently with timeout handling
        logger.info("Extracting musical features...")
        
        # Skip extensive feature extraction if we have too many sequences
        # Focus on core note transitions which are more important
        skip_detailed_features = len(midi_sequences) > 500
        if skip_detailed_features:
            logger.info("Large dataset detected, using simplified feature extraction")
            
        if not skip_detailed_features:
            try:
                # Extract musical features from a subset of data
                music_features = self.extract_musical_features(midi_sequences[:min(len(midi_sequences), 200)])
                
                if music_features:
                    # Process time signatures and rhythmic features
                    self._process_time_signatures(music_features)
                    self._process_rhythm_patterns(music_features)
                    self._process_chord_progressions(music_features)
                    self._process_key_features(music_features)
                    # Process Roman numeral transitions
                    self._process_roman_numeral_transitions(music_features)
            except Exception as e:
                logger.warning(f"Feature extraction issue: {e}. Using simplified training.")
        
        # Train note transitions (the core functionality)
        logger.info("Training note transitions...")
        self._train_note_transitions(midi_sequences, progress_callback)
        
        # Train interval transitions
        logger.info("Training interval transitions...")
        self._train_interval_transitions(midi_sequences, progress_callback)
        
        self.trained = True
        logger.info("Training complete!")
        return True

    def _process_time_signatures(self, music_features):
        """Process time signature related features"""
        # Time signatures
        ts_counter = Counter(music_features['time_signatures'])
        total_ts = len(music_features['time_signatures']) or 1
        self.musical_features['time_signatures'] = {ts: count/total_ts for ts, count in ts_counter.items()}
        
        # Note density
        for ts, densities in music_features['note_density'].items():
            if densities:
                self.musical_features['note_density'][ts] = sum(densities) / len(densities)
                
        # Beat positions
        for ts, positions in music_features['beat_positions'].items():
            total_notes = sum(positions.values()) or 1
            self.musical_features['beat_positions'][ts] = {pos: count/total_notes for pos, count in positions.items()}
    
    def _process_rhythm_patterns(self, music_features):
        """Process rhythm pattern features"""
        pattern_counter = Counter(music_features['rhythm_patterns'])
        self.musical_features['rhythm_patterns'] = dict(pattern_counter.most_common(50))
        logger.info(f"Learned {len(self.musical_features['rhythm_patterns'])} common rhythm patterns")
    
    def _process_chord_progressions(self, music_features):
        """Process chord progression features"""
        chord_prog_counter = Counter(music_features['chord_progressions'])
        self.musical_features['common_chord_progressions'] = dict(chord_prog_counter.most_common(20))
        logger.info(f"Captured {len(self.musical_features['common_chord_progressions'])} common chord progressions")
    
    def _process_key_features(self, music_features):
        """Process key-related features"""
        key_counter = Counter(music_features['keys'])
        total_keys = len(music_features['keys']) or 1
        self.musical_features['common_keys'] = {k: count/total_keys for k, count in key_counter.items() if count > 2}
        logger.info(f"Detected {len(self.musical_features['common_keys'])} common keys")
    
    def _process_roman_numeral_transitions(self, music_features):
        """Process Roman numeral transitions"""
        roman_transitions = {}
        
        for key_name, progressions in music_features['roman_numeral_progressions'].items():
            key_transitions = defaultdict(Counter)
            for prog_pair in progressions:
                if len(prog_pair) == 2:
                    key_transitions[prog_pair[0]][prog_pair[1]] += 1
                    
            # Normalize for each key
            roman_transitions[key_name] = {}
            for rn1, transitions in key_transitions.items():
                total = sum(transitions.values())
                if total > 0:
                    for rn2, count in transitions.items():
                        roman_transitions[key_name][(rn1, rn2)] = count / total
                        
        self.musical_features['roman_numeral_transitions'] = roman_transitions
        logger.info(f"Learned Roman numeral transitions for {len(roman_transitions)} keys")
        
    def _train_note_transitions(self, midi_sequences, progress_callback=None):
        """Train single and multi-order note transitions efficiently"""
        logger.info("Training note transitions...")
        
        multi_order_counts = defaultdict(Counter)
        total_processed = 0
        
        for sequence in midi_sequences:
            pitches = []
            for note_data in sequence:
                if isinstance(note_data, int):
                    pitches.append(note_data)
                elif isinstance(note_data, (list, tuple)) and len(note_data) >= 1:
                    pitches.append(note_data[0])
                    
            if len(pitches) < 2:
                continue
                
            # Single-order transitions
            for i in range(len(pitches) - 1):
                prev_note, next_note = pitches[i], pitches[i + 1]
                if 0 <= prev_note < 128 and 0 <= next_note < 128:
                    self.transitions[prev_note, next_note] += 1
            
            # Multi-order transitions
            if self.order > 1 and len(pitches) > self.order:
                for i in range(len(pitches) - self.order):
                    context = tuple(pitches[i:i+self.order])
                    next_note = pitches[i+self.order]
                    if all(0 <= n < 128 for n in context) and 0 <= next_note < 128:
                        multi_order_counts[context][next_note] += 1
                        
            total_processed += 1
            if progress_callback and total_processed % 100 == 0:
                progress_callback(0.5 * total_processed / len(midi_sequences))
                
        # Normalize traditional transition matrix
        row_sums = self.transitions.sum(axis=1, keepdims=True)
        np.divide(self.transitions, np.where(row_sums == 0, 1.0, row_sums), out=self.transitions)
        
        # Normalize multi-order transitions
        multi_order_transitions = {}
        for context, transitions in multi_order_counts.items():
            total = sum(transitions.values())
            if total > 0:
                multi_order_transitions[context] = {note: count/total for note, count in transitions.items()}
                
        self.musical_features['multi_order_transitions'] = multi_order_transitions
        logger.info(f"Note transitions trained on {total_processed} sequences")
        
    def _train_interval_transitions(self, midi_sequences, progress_callback=None):
        """Train interval transitions efficiently"""
        logger.info("Training interval transitions...")
        
        interval_counts = defaultdict(lambda: np.zeros(self.interval_range, dtype=np.float32))
        total_processed = 0
        
        for sequence in midi_sequences:
            pitches = []
            for note_data in sequence:
                if isinstance(note_data, int):
                    pitches.append(note_data)
                elif isinstance(note_data, (list, tuple)) and len(note_data) >= 1:
                    pitches.append(note_data[0])
                    
            if len(pitches) < 2:
                continue
                
            for i in range(len(pitches) - 1):
                prev_note = pitches[i]
                next_note = pitches[i + 1]
                
                if 0 <= prev_note < 128 and 0 <= next_note < 128:
                    interval_val = next_note - prev_note
                    if -self.max_interval <= interval_val <= self.max_interval:
                        interval_index = interval_val + self.max_interval
                        interval_counts[prev_note][interval_index] += 1
                        
            total_processed += 1
            if progress_callback and total_processed % 100 == 0:
                progress_callback(0.5 + 0.5 * total_processed / len(midi_sequences))
                
        # Normalize and store only non-zero rows for efficiency
        for prev_note, counts in interval_counts.items():
            row_sum = counts.sum()
            if row_sum > 0:
                self.interval_transitions[prev_note] = counts / row_sum
                
        logger.info(f"Interval features trained on {total_processed} sequences")
        
    def save(self, filepath):
        """Save model with efficient serialization"""
        data_to_save = {
            'transitions': self.transitions,
            'interval_transitions': self.interval_transitions,
            'max_interval': self.max_interval,
            'order': self.order,
            'musical_features': self.musical_features
        }
        
        np.save(filepath, data_to_save, allow_pickle=True)
        logger.info(f"Model saved to {filepath}")
        
    def load(self, filepath):
        """Load model with error handling"""
        try:
            data = np.load(filepath, allow_pickle=True).item()
            
            # Load core properties
            self.transitions = data['transitions']
            self.max_interval = data.get('max_interval', 12)
            self.interval_range = 2 * self.max_interval + 1
            self.interval_transitions = data.get('interval_transitions', {})
            self.order = data.get('order', 1)
            
            # Load musical features with proper defaults
            loaded_features = data.get('musical_features', {})
            for key in self.musical_features:
                if key in loaded_features:
                    self.musical_features[key] = loaded_features[key]
            
            self.trained = True
            logger.info(f"Loaded model with order {self.order} and max interval {self.max_interval}")
            
            # Log key statistics
            log_items = [
                ('common keys', len(self.musical_features['common_keys'])),
                ('chord progressions', len(self.musical_features['common_chord_progressions'])),
                ('rhythm patterns', len(self.musical_features['rhythm_patterns'])),
                ('roman numeral transitions keys', len(self.musical_features['roman_numeral_transitions']))
            ]
            
            for name, count in log_items:
                if count:
                    logger.info(f"Loaded {count} {name}")
                    
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def generate_chord_progression(self, key_context=None, num_chords=8):
        """Generate a chord progression using learned transitions"""
        if not self.trained:
            logger.warning("Model not trained for chord progression generation")
            return ['C major', 'G major', 'A minor', 'F major'] * (num_chords // 4 + 1)
            
        cleaned_key = self._clean_key_context(key_context)
        
        # Validate the key before trying to use it
        cleaned_key = self._validate_key(cleaned_key)
        
        try:
            k = key.Key(cleaned_key)
            sc = k.getScale()
            
            progression = []
            
            # Determine generation approach based on available data
            use_roman = cleaned_key in self.musical_features['roman_numeral_transitions'] and self.musical_features['roman_numeral_transitions'][cleaned_key]
            
            if use_roman:
                # Using Roman numeral transitions
                progression = self._generate_with_roman_numerals(cleaned_key, k, num_chords)
            elif self.musical_features['common_chord_progressions']:
                # Using learned chord pairs
                progression = self._generate_with_chord_pairs(num_chords)
            else:
                # Fallback to diatonic generation
                progression = self._generate_diatonic_chords(sc, num_chords)
                
            return progression
        except Exception as e:
            # Handle any errors in chord progression generation
            logger.error(f"Error generating chord progression: {e}")
            return ['C major', 'G major', 'A minor', 'F major'] * (num_chords // 4 + 1)

    def _generate_with_roman_numerals(self, key_name, k, num_chords):
        """Generate chord progression using Roman numerals"""
        progression = []
        roman_numerals = []
        key_transitions = self.musical_features['roman_numeral_transitions'][key_name]
        
        # Start with tonic or dominant
        current_rn_figure = random.choice(['I', 'V'] if k.mode == 'major' else ['i', 'v'])
        
        try:
            start_chord = roman.RomanNumeral(current_rn_figure, k).realize()
            progression.append(f"{start_chord.root().name} {start_chord.quality}")
            roman_numerals.append(current_rn_figure)
        except Exception:
            # Fallback if Roman numeral approach fails
            return self._generate_diatonic_chords(k.getScale(), num_chords)
            
        # Generate remaining chords
        while len(progression) < num_chords:
            next_rn = None
            
            # Find possible next Roman numerals
            possible_next = []
            probs = []
            
            for (rn1, rn2), prob in key_transitions.items():
                if rn1 == current_rn_figure:
                    possible_next.append(rn2)
                    probs.append(prob)
                    
            if possible_next:
                next_rn = random.choices(possible_next, weights=probs, k=1)[0]
                
                try:
                    next_chord = roman.RomanNumeral(next_rn, k).realize()
                    progression.append(f"{next_chord.root().name} {next_chord.quality}")
                    roman_numerals.append(next_rn)
                    current_rn_figure = next_rn
                    continue
                except Exception:
                    pass
                    
            # Fallback if no valid transition
            if len(progression) >= num_chords:
                break
                
            # Use diatonic chord as fallback
            sc = k.getScale()
            diatonic_chords = self._get_diatonic_chords(sc)
            
            if diatonic_chords:
                next_chord = random.choice([c for c in diatonic_chords if c != progression[-1]] or diatonic_chords)
                progression.append(next_chord)
                
        return progression[:num_chords]
    
    def _generate_with_chord_pairs(self, num_chords):
        """Generate chord progression using learned chord pairs"""
        progression = []
        
        # Start with a common chord from learned pairs
        all_first_chords = list(set(pair[0] for pair in self.musical_features['common_chord_progressions']))
        
        if not all_first_chords:
            return ['C major', 'G major', 'A minor', 'F major'] * (num_chords // 4 + 1)
            
        current_chord = random.choice(all_first_chords)
        progression.append(current_chord)
        
        # Generate remaining chords
        while len(progression) < num_chords:
            possible_next = []
            weights = []
            
            # Find all possible next chords from learned pairs
            for (c1, c2), count in self.musical_features['common_chord_progressions'].items():
                if c1 == current_chord:
                    possible_next.append(c2)
                    weights.append(count)
                    
            if possible_next:
                # Use learned transitions
                next_chord = random.choices(possible_next, weights=weights, k=1)[0]
            else:
                # No learned transition, choose randomly from first chords
                next_chord = random.choice([c for c in all_first_chords if c != current_chord] or all_first_chords)
                
            progression.append(next_chord)
            current_chord = next_chord
            
        return progression[:num_chords]
    
    def _generate_diatonic_chords(self, scale_obj, num_chords):
        """Generate diatonic chord progression"""
        diatonic_chords = self._get_diatonic_chords(scale_obj)
        
        if not diatonic_chords:
            return ['C major', 'G major', 'A minor', 'F major'] * (num_chords // 4 + 1)
            
        progression = []
        
        # Start with I, IV, or V
        common_degrees = [1, 4, 5]
        current_degree_idx = 0
        
        while len(progression) < num_chords:
            if progression and random.random() < 0.7:
                # Choose next chord based on common progressions
                # Common chord movements: I-IV, I-V, IV-I, V-I, vi-IV, etc.
                last_degree = common_degrees[current_degree_idx % len(common_degrees)]
                
                if last_degree == 1:  # I
                    next_degree = random.choices([4, 5, 6], weights=[0.5, 0.4, 0.1])[0]
                elif last_degree == 4:  # IV
                    next_degree = random.choices([1, 5], weights=[0.7, 0.3])[0]
                elif last_degree == 5:  # V
                    next_degree = random.choices([1, 6], weights=[0.8, 0.2])[0]
                elif last_degree == 6:  # vi
                    next_degree = random.choices([2, 4], weights=[0.3, 0.7])[0]
                else:
                    next_degree = random.choice([1, 4, 5])
                    
                current_degree_idx = common_degrees.index(next_degree) if next_degree in common_degrees else 0
                
                try:
                    next_chord = scale_obj.getChord(next_degree)
                    progression.append(f"{next_chord.root().name} {next_chord.quality}")
                except Exception:
                    # Fallback to random selection
                    progression.append(random.choice(diatonic_chords))
            else:
                # Initial chord or random selection for variety
                progression.append(random.choice(diatonic_chords))
                
        return progression[:num_chords]
    
    def _get_diatonic_chords(self, scale_obj):
        """Helper to get diatonic chords from a scale"""
        diatonic_chords = []
        
        for degree in [1, 4, 5, 6, 2, 3, 7]:  # Common order of importance
            try:
                c = scale_obj.getChord(degree)
                chord_name = f"{c.root().name} {c.quality}"
                diatonic_chords.append(chord_name)
            except Exception:
                continue
                
        return diatonic_chords
        
    def generate_interval_sequence(self, start_note=60, length=32, key_context=None):
        """Generate a sequence using interval transitions with scale awareness"""
        if not self.trained:
            return [start_note] * length
            
        if not self.interval_transitions:
            return self.generate_sequence(start_note, length, key_context)
            
        sequence = [start_note]
        current_note = start_note
        
        # Get scale pitches for filtering
        scale_pitches = self._get_scale_pitches(key_context)
        
        for _ in range(length - 1):
            if current_note not in self.interval_transitions:
                # Fallback for missing note
                next_interval = random.randint(-7, 7)
                next_note = current_note + next_interval
            else:
                probs = np.array(self.interval_transitions[current_note])
                
                # Apply scale filtering if available
                if scale_pitches:
                    filtered_probs = np.zeros_like(probs)
                    for interval_idx, prob in enumerate(probs):
                        if prob > 0:
                            interval_val = interval_idx - self.max_interval
                            potential_note = current_note + interval_val
                            if potential_note in scale_pitches:
                                filtered_probs[interval_idx] = prob
                                
                    # Use filtered probabilities if any remain
                    sum_filtered = filtered_probs.sum()
                    if sum_filtered > 0:
                        probs = filtered_probs / sum_filtered
                        
                # Sample the interval
                try:
                    interval_idx = np.random.choice(self.interval_range, p=probs)
                    next_interval = interval_idx - self.max_interval
                except ValueError:
                    next_interval = 0  # Default to repeating the note
                    
                next_note = current_note + next_interval
                
            # Keep within MIDI range
            next_note = max(0, min(127, next_note))
            sequence.append(next_note)
            current_note = next_note
            
        return sequence
        
    def _get_scale_pitches(self, key_context):
        """Get scale pitches for a given key"""
        if not key_context:
            return set()
            
        try:
            k = key.Key(self._clean_key_context(key_context))
            sc = scale.AbstractScale.derive(k.getScale())
            # Get pitches across a wide range
            return set(p.midi for p in sc.getPitches(pitch.Pitch('C2'), pitch.Pitch('C7')))
        except Exception:
            return set()
            
    def generate_sequence(self, start_note=60, length=32, key_context=None):
        """Generate a sequence using note transitions"""
        if not self.trained:
            return [start_note] * length
            
        sequence = [start_note]
        current_note = start_note
        
        # Get scale pitches for filtering
        scale_pitches = self._get_scale_pitches(key_context)
        
        for _ in range(length - 1):
            if current_note >= self.transitions.shape[0]:
                current_note = random.randint(48, 72)
                
            probs = self.transitions[current_note].copy()
            
            # Apply scale filtering if available
            if scale_pitches:
                filtered_probs = np.zeros_like(probs)
                for note_idx, prob in enumerate(probs):
                    if prob > 0 and note_idx in scale_pitches:
                        filtered_probs[note_idx] = prob
                        
                if filtered_probs.sum() > 0:
                    probs = filtered_probs / filtered_probs.sum()
                    
            # Sample next note
            if probs.sum() == 0:
                # Fallback random selection
                if scale_pitches:
                    next_note = random.choice([n for n in scale_pitches if 48 <= n <= 84] or [60])
                else:
                    next_note = current_note + random.randint(-5, 5)
            else:
                try:
                    next_note = np.random.choice(128, p=probs)
                except ValueError:
                    next_note = np.argmax(probs)
                    
            next_note = max(0, min(127, next_note))
            sequence.append(next_note)
            current_note = next_note
            
        return sequence
        
    def generate_rhythmic_sequence(self, start_note=60, key_context=None,
                                   length=32, time_signature="4/4", measures=8,
                                   use_interval_generation=True):
        """Generate a sequence with expressive rhythm"""
        if not self.trained:
            pitches = [start_note] * length
            durations = [0.5] * length
            return [(p, d, 0) for p, d in zip(pitches, durations)]
            
        # Parse time signature
        try:
            numerator, denominator = map(int, time_signature.split('/'))
            beats_per_measure = numerator
            beat_value = 4 / denominator
        except Exception:
            beats_per_measure = 4
            beat_value = 1.0
            
        # Generate pitch sequence
        if use_interval_generation:
            pitch_sequence = self.generate_interval_sequence(start_note, length, key_context)
        else:
            pitch_sequence = self.generate_sequence(start_note, length, key_context)
            
        # Ensure sequence is the right length
        if len(pitch_sequence) < length:
            pitch_sequence.extend([pitch_sequence[-1]] * (length - len(pitch_sequence)))
        elif len(pitch_sequence) > length:
            pitch_sequence = pitch_sequence[:length]
            
        # Generate rhythm using learned patterns if available
        rhythm_durations = []
        current_beat = 0.0
        total_beats = measures * beats_per_measure
        
        if self.musical_features['rhythm_patterns']:
            # Use learned rhythm patterns
            rhythm_transitions = self._build_rhythm_transitions()
            
            # Start with common pattern
            possible_starts = self._get_common_rhythm_elements(first_only=True)
            if possible_starts:
                current_element = random.choice(possible_starts)
                rhythm_durations.append(current_element[0])
                current_beat += current_element[0]
                
                # Generate subsequent durations
                while len(rhythm_durations) < length and current_beat < total_beats:
                    next_element = self._get_next_rhythm_element(current_element, rhythm_transitions)
                    
                    if next_element:
                        if current_beat + next_element[0] <= total_beats:
                            rhythm_durations.append(next_element[0])
                            current_beat += next_element[0]
                            current_element = next_element
                        else:
                            break
                    else:
                        break
            
        # Fill any remaining durations with default values
        while len(rhythm_durations) < length:
            remaining = total_beats - current_beat
            if remaining <= 0:
                break
                
            dur = min(beat_value / 2, remaining)  # Default to eighth notes or smaller if needed
            rhythm_durations.append(dur)
            current_beat += dur
            
        # Combine pitch and rhythm
        result = []
        position = 0.0
        
        for i in range(min(len(pitch_sequence), len(rhythm_durations))):
            note_val = pitch_sequence[i]
            duration_val = rhythm_durations[i]
            position_in_measure = position % beats_per_measure
            result.append((note_val, duration_val, position_in_measure))
            position += duration_val
            
        return result
        
    def _build_rhythm_transitions(self):
        """Build rhythm transitions from learned patterns"""
        transitions = defaultdict(Counter)
        
        for pattern, count in self.musical_features['rhythm_patterns'].items():
            if len(pattern) == 2:
                transitions[pattern[0]][pattern[1]] += count
                
        return transitions
        
    def _get_common_rhythm_elements(self, first_only=True):
        """Get common rhythm elements from learned patterns"""
        if not self.musical_features['rhythm_patterns']:
            return []
            
        elements = set()
        for pattern in self.musical_features['rhythm_patterns']:
            if pattern and len(pattern) > 0:
                elements.add(pattern[0] if first_only else pattern[-1])
                
        return list(elements)
        
    def _get_next_rhythm_element(self, current_element, transitions):
        """Get next rhythm element based on transitions"""
        if current_element in transitions:
            options = transitions[current_element]
            if options:
                return random.choices(
                    population=list(options.keys()),
                    weights=list(options.values()),
                    k=1
                )[0]
        return None
        
    def generate_with_chords(self, key_context=None, length=32, time_signature="4/4"):
        """Generate a sequence with chord progression and timing awareness"""
        try:
            cleaned_key = self._clean_key_context(key_context)
            
            # Generate more chords than seemingly needed to ensure coverage
            num_chords = max(8, length // 4)
            chords = self.generate_chord_progression(key_context=cleaned_key, num_chords=num_chords)
            
            # Log chord progression for debugging
            logger.info(f"Generated chord progression: {chords[:8]}...")
            
            start_note = self._determine_start_note(cleaned_key)
            
            # Handle time signature
            if time_signature not in self.musical_features['time_signatures']:
                time_signature = "4/4"
                if self.musical_features['time_signatures']:
                    time_signature = max(self.musical_features['time_signatures'].items(), key=lambda x: x[1])[0]
                    
            target_density = self.musical_features['note_density'].get(time_signature, 4)
            try:
                numerator, denominator = map(int, time_signature.split('/'))
                beats_per_measure = numerator
            except Exception:
                beats_per_measure = 4
                
            # Calculate appropriate length
            target_measures = max(1, round(length / target_density))
            actual_length = int(target_measures * target_density) or length
            
            # Generate notes with rhythm
            notes_with_timings = self.generate_rhythmic_sequence(
                start_note=start_note,
                key_context=cleaned_key,
                length=actual_length,
                time_signature=time_signature,
                measures=target_measures,
                use_interval_generation=True
            )
            
            # Extract components
            notes = [n for n, _, _ in notes_with_timings]
            durations = [d for _, d, _ in notes_with_timings]
            beat_positions = [p for _, _, p in notes_with_timings]
            
            # Enhanced chord mapping
            chord_sequence = []
            # Determine beats per chord (typically 4 beats = 1 measure)
            beats_per_chord = 4  
            current_position = 0.0
            chord_index = 0
            
            # For each note, determine which chord it belongs to based on its position
            for i, (note, duration) in enumerate(zip(notes, durations)):
                # When we cross a chord boundary, advance to next chord
                if current_position >= beats_per_chord * (chord_index + 1):
                    chord_index = min(len(chords) - 1, int(current_position / beats_per_chord))
                
                chord_sequence.append(chords[chord_index])
                current_position += duration
                
            # Log chord usage statistics
            chord_counts = {}
            for chord in chord_sequence:
                chord_counts[chord] = chord_counts.get(chord, 0) + 1
            logger.info(f"Chord distribution in output: {chord_counts}")
            
            # Ensure everything is same length
            final_length = len(notes)
            
            # Create harmonized notes based on chord information
            chord_notes = []
            for i, chord_name in enumerate(chord_sequence[:final_length]):
                try:
                    # Extract chord notes from the chord name
                    ch = self._parse_chord_name(chord_name)
                    if ch and ch.pitches:
                        # Store the chord pitches for potential harmonization
                        chord_notes.append([p.midi % 12 for p in ch.pitches])
                    else:
                        chord_notes.append([0, 4, 7])  # C major fallback
                except Exception as e:
                    logger.debug(f"Error parsing chord {chord_name}: {e}")
                    chord_notes.append([0, 4, 7])  # C major fallback
            
            # Apply chord-aware pitch correction (ensure notes fit with chords)
            adjusted_notes = []
            for i, note in enumerate(notes[:final_length]):
                chord_pitches = chord_notes[i]
                # If the current note doesn't fit the chord, adjust it
                if (note % 12) not in chord_pitches:
                    # Find the nearest chord pitch
                    note_class = note % 12
                    distances = [(p - note_class) % 12 for p in chord_pitches]
                    min_dist_idx = distances.index(min(distances))
                    octave = note // 12
                    adjusted = chord_pitches[min_dist_idx] + (octave * 12)
                    adjusted_notes.append(adjusted)
                else:
                    adjusted_notes.append(note)
            
            # Return the generated sequence with chord information
            return {
                'notes': adjusted_notes,  # Use pitch-corrected notes
                'durations': durations[:final_length],
                'chords': chord_sequence[:final_length],
                'beat_positions': beat_positions[:final_length],
                'key': cleaned_key,
                'time_signature': time_signature
            }
            
        except Exception as e:
            logger.error(f"Error in generate_with_chords: {e}")
            # Fallback generation
            notes = self.generate_sequence(start_note=60, length=length)
            durations = [0.5] * len(notes)
            return {
                'notes': notes,
                'durations': durations,
                'chords': ['C major'] * len(notes),
                'beat_positions': list(np.arange(0, len(notes)*0.5, 0.5)),
                'key': 'C major',
                'time_signature': '4/4'
            }

    def _parse_chord_name(self, chord_name):
        """Convert chord name string to music21 chord object"""
        try:
            # Extract root and quality
            parts = chord_name.split(' ', 1)
            if len(parts) != 2:
                return None
                
            root_name, quality = parts
            
            # Handle common quality naming inconsistencies
            quality_map = {
                'major': '',
                'minor': 'm',
                'm': 'm',
                'diminished': 'dim',
                'dim': 'dim',
                'augmented': 'aug',
                'aug': 'aug',
                '7': '7',
                'maj7': 'maj7',
                'm7': 'm7',
                'dim7': 'dim7'
            }
            
            # Map the quality to something music21 understands
            mapped_quality = quality_map.get(quality.lower(), quality)
            
            # Create the chord
            return chord.Chord(f"{root_name}{mapped_quality}")
            
        except Exception as e:
            logger.debug(f"Chord parsing error: {e}")
            return None

    def generate_expressive_sequence(self, key_context=None, length=64, complexity=0.7):
        """Generate a musically expressive sequence with varied rhythms"""
        # Select time signature based on complexity
        if complexity < 0.3:
            time_sig = "4/4"
        elif complexity < 0.7:
            time_sig = random.choices(["4/4", "3/4"], weights=[0.7, 0.3])[0]
        else:
            time_sig = random.choices(
                ["4/4", "3/4", "6/8", "5/4", "7/8"],
                weights=[0.5, 0.2, 0.15, 0.1, 0.05]
            )[0]
            
        # Generate sequence with enhanced features
        return self.generate_with_chords(
            key_context=key_context,
            length=length,
            time_signature=time_sig
        )
        
    def _clean_key_context(self, key_context):
        """Standardize key context format and validate it"""
        if not key_context:
            # Use default or random key
            if self.musical_features['common_keys']:
                return random.choice(list(self.musical_features['common_keys'].keys()))
            return "C major"
            
        try:
            # Clean spaces and handle special characters
            key_context = key_context.strip()
            
            # Check for missing spaces between note and mode
            if key_context.lower().endswith("major") and " " not in key_context:
                note_name = key_context[:-5].strip().capitalize()
                return f"{note_name} major"
            elif key_context.lower().endswith("minor") and " " not in key_context:
                note_name = key_context[:-5].strip().capitalize()
                return f"{note_name} minor"
            
            # If it contains the word "major" or "minor" but is incorrectly formatted
            if "major" in key_context.lower():
                parts = key_context.lower().split("major")
                tonic = parts[0].strip().capitalize()
                # Ensure tonic is just a note name
                if len(tonic) > 2:  # Most note names are 1-2 chars
                    tonic = tonic[0]  # Take just the first letter as a fallback
                return f"{tonic} major"
                
            if "minor" in key_context.lower():
                parts = key_context.lower().split("minor")
                tonic = parts[0].strip().capitalize()
                if len(tonic) > 2:
                    tonic = tonic[0]
                return f"{tonic} minor"
            
            # Standard format parsing
            parts = key_context.split()
            
            if len(parts) == 1:
                # Just a note name, assume major
                return f"{parts[0].capitalize()} major"
            elif len(parts) == 2:
                # Note name and mode
                if parts[1].lower() not in ["major", "minor"]:
                    return "C major"  # Invalid mode
                return f"{parts[0].capitalize()} {parts[1].lower()}"
            else:
                return "C major"  # Invalid format
                
        except Exception as e:
            logger.warning(f"Error cleaning key context '{key_context}': {e}", exc_info=True)
            return "C major"  # Safe fallback

    def _validate_key(self, key_str):
        """Validate if a key string can be used with music21"""
        try:
            k = key.Key(key_str)
            return key_str
        except Exception as e:
            logger.warning(f"Invalid key '{key_str}': {e}")
            return "C major"  # Safe fallback

    def _determine_start_note(self, key_context):
        """Determine appropriate start note for a key"""
        start_note = 60  # Default to middle C
        
        try:
            k = key.Key(key_context)
            start_note = k.tonic.midi
            
            # Adjust to comfortable range
            while start_note < 48:
                start_note += 12
            while start_note > 72:
                start_note -= 12
                
        except Exception:
            pass
            
        return start_note