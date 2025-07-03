import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, Counter
from music21 import note, pitch, scale, key, chord, stream, roman, meter
import logging
from sklearn.cluster import KMeans
from hmmlearn import hmm
import cupy as cp
import warnings

logger = logging.getLogger(__name__)

# Try to use CUDA if available with better error handling
HAS_CUPY = False
try:
    import cupy as cp
    # Test CuPy functionality
    test_array = cp.array([1, 2, 3])
    _ = test_array + 1  # Simple operation to verify it works
    HAS_CUPY = True
    logger.info("✅ CuPy detected and working - GPU acceleration enabled")
except ImportError:
    logger.warning("⚠️ CuPy not found - falling back to CPU computation")
except Exception as e:
    logger.warning(f"⚠️ CuPy found but not working ({e}) - falling back to CPU computation")
    HAS_CUPY = False

class CUDAOptimizer:
    """GPU-accelerated tensor operations for Markov chains with robust error handling"""
    
    @staticmethod
    def to_gpu(array):
        """Convert numpy array to GPU array if possible"""
        if HAS_CUPY and isinstance(array, np.ndarray):
            try:
                return cp.asarray(array)
            except Exception as e:
                logger.warning(f"GPU conversion failed: {e}")
                return array
        return array
    
    @staticmethod
    def to_cpu(array):
        """Convert GPU array back to numpy if needed"""
        if HAS_CUPY and hasattr(array, 'get'):
            try:
                return cp.asnumpy(array)
            except Exception as e:
                logger.warning(f"CPU conversion failed: {e}")
                return array
        return array
    
    @staticmethod
    def zeros_like_gpu(shape, dtype=np.float32):
        """Create zeros array on GPU if available"""
        if HAS_CUPY:
            try:
                return cp.zeros(shape, dtype=dtype)
            except Exception as e:
                logger.warning(f"GPU zeros creation failed: {e}, using CPU")
                return np.zeros(shape, dtype=dtype)
        return np.zeros(shape, dtype=dtype)
    
    @staticmethod
    def normalize_gpu(matrix, axis=1):
        """GPU-accelerated matrix normalization with fallback"""
        try:
            if HAS_CUPY and hasattr(matrix, 'sum'):
                sums = cp.sum(matrix, axis=axis, keepdims=True)
                sums = cp.where(sums == 0, 1.0, sums)
                return matrix / sums
            else:
                sums = np.sum(matrix, axis=axis, keepdims=True)
                sums = np.where(sums == 0, 1.0, sums)
                return matrix / sums
        except Exception as e:
            logger.warning(f"GPU normalization failed: {e}, using CPU")
            if hasattr(matrix, 'get'):
                matrix = cp.asnumpy(matrix)
            sums = np.sum(matrix, axis=axis, keepdims=True)
            sums = np.where(sums == 0, 1.0, sums)
            return matrix / sums

class MarkovChain:
    """
    Enhanced musical Markov Chain model with HMM capabilities, GPU acceleration, and high-order transitions.
    """
    def __init__(self, order=3, max_interval=12, n_hidden_states=16, use_gpu=True):
        # Enhanced model configuration
        self.order = max(order, 3)  # Increase minimum order to 3
        self.max_interval = max_interval
        self.interval_range = 2 * max_interval + 1
        self.n_hidden_states = n_hidden_states
        self.use_gpu = use_gpu and HAS_CUPY
        self.trained = False
        
        # Initialize GPU optimizer
        self.gpu_opt = CUDAOptimizer()
        
        # Enhanced transition matrices with GPU support
        if self.use_gpu:
            self.transitions = self.gpu_opt.zeros_like_gpu((128, 128), dtype=np.float32)
            self.higher_order_transitions = {}  # Will store GPU arrays
        else:
            self.transitions = np.zeros((128, 128), dtype=np.float32)
            self.higher_order_transitions = {}
            
        # Sparse interval transitions for memory efficiency
        self.interval_transitions = {}
        
        # HMM Components
        self.hmm_model = None
        self.hidden_states = None
        self.emission_probs = None
        self.transition_probs = None
        
        # Enhanced musical features with GPU acceleration
        self.musical_features = {
            'multi_order_transitions': {},  # Now supports orders 2-6
            'chord_transitions': {},
            'common_chord_progressions': {},
            'roman_numeral_transitions': {},
            'duration_transitions': {},
            'common_keys': {},
            'chord_to_scale': {},
            'rhythm_patterns': {},
            'time_signatures': {},
            'beat_patterns': {},
            'rhythmic_motifs': {},
            'beat_strength_transitions': {},
            'note_density': {},
            'beat_positions': {},
            'grouping_patterns': [],
            'harmonic_progressions': {},  # New: Enhanced harmonic analysis
            'melodic_contours': {},       # New: Melodic shape patterns
            'dynamic_patterns': {},       # New: Velocity/dynamics
            'rhythmic_syncopation': {},   # New: Syncopation patterns
            'phrase_boundaries': {},      # New: Musical phrase structure
        }
        
        # State tracking for HMM
        self.current_hidden_state = 0
        self.state_history = []
        
        # HMM Algorithm components
        self.alpha = None  # Forward probabilities
        self.beta = None   # Backward probabilities
        self.gamma = None  # State posterior probabilities
        self.xi = None     # Transition posterior probabilities
        self.log_likelihood = None  # Model likelihood
        
    def forward_algorithm(self, observations):
        """
        Forward algorithm implementation using hmmlearn
        
        Args:
            observations: Sequence of observed musical features
            
        Returns:
            Forward probabilities (alpha) and log likelihood
        """
        if self.hmm_model is None:
            logger.warning("HMM model not initialized. Cannot run forward algorithm.")
            return None, None
            
        try:
            # Ensure observations are in correct format
            obs_array = np.array(observations).reshape(-1, 1) if np.array(observations).ndim == 1 else np.array(observations)
            
            # Compute forward probabilities using hmmlearn
            log_likelihood, self.alpha = self.hmm_model._do_forward_pass(obs_array)
            
            logger.info(f"Forward algorithm completed. Log likelihood: {log_likelihood:.4f}")
            return self.alpha, log_likelihood
            
        except Exception as e:
            logger.error(f"Forward algorithm failed: {e}")
            return None, None
    
    def backward_algorithm(self, observations):
        """
        Backward algorithm implementation using hmmlearn
        
        Args:
            observations: Sequence of observed musical features
            
        Returns:
            Backward probabilities (beta)
        """
        if self.hmm_model is None:
            logger.warning("HMM model not initialized. Cannot run backward algorithm.")
            return None
            
        try:
            # Ensure observations are in correct format
            obs_array = np.array(observations).reshape(-1, 1) if np.array(observations).ndim == 1 else np.array(observations)
            
            # Compute backward probabilities using hmmlearn
            self.beta = self.hmm_model._do_backward_pass(obs_array)
            
            logger.info("Backward algorithm completed")
            return self.beta
            
        except Exception as e:
            logger.error(f"Backward algorithm failed: {e}")
            return None
    
    def forward_backward_algorithm(self, observations):
        """
        Combined forward-backward algorithm for computing state posteriors
        
        Args:
            observations: Sequence of observed musical features
            
        Returns:
            State posteriors (gamma) and transition posteriors (xi)
        """
        if self.hmm_model is None:
            logger.warning("HMM model not initialized. Cannot run forward-backward algorithm.")
            return None, None
            
        try:
            # Ensure observations are in correct format
            obs_array = np.array(observations).reshape(-1, 1) if np.array(observations).ndim == 1 else np.array(observations)
            
            # Run forward algorithm
            log_likelihood, alpha = self.hmm_model._do_forward_pass(obs_array)
            
            # Run backward algorithm
            beta = self.hmm_model._do_backward_pass(obs_array)
            
            # Compute state posteriors (gamma)
            self.gamma = alpha + beta
            # Normalize to get probabilities
            self.gamma = np.exp(self.gamma - np.logaddexp.reduce(self.gamma, axis=1, keepdims=True))
            
            # Compute transition posteriors (xi)
            T = len(obs_array)
            self.xi = np.zeros((T-1, self.n_hidden_states, self.n_hidden_states))
            
            for t in range(T-1):
                for i in range(self.n_hidden_states):
                    for j in range(self.n_hidden_states):
                        self.xi[t, i, j] = (alpha[t, i] + 
                                          np.log(self.hmm_model.transmat_[i, j]) + 
                                          self.hmm_model._compute_log_likelihood(obs_array[t+1:t+2])[j] + 
                                          beta[t+1, j])
                
                # Normalize xi for each time step
                log_xi_sum = np.logaddexp.reduce(np.logaddexp.reduce(self.xi[t], axis=1), axis=0)
                self.xi[t] = np.exp(self.xi[t] - log_xi_sum)
            
            logger.info("Forward-backward algorithm completed")
            return self.gamma, self.xi
            
        except Exception as e:
            logger.error(f"Forward-backward algorithm failed: {e}")
            return None, None
    
    def viterbi_algorithm(self, observations):
        """
        Viterbi algorithm for finding the most likely hidden state sequence
        
        Args:
            observations: Sequence of observed musical features
            
        Returns:
            Most likely state sequence and its log probability
        """
        if self.hmm_model is None:
            logger.warning("HMM model not initialized. Cannot run Viterbi algorithm.")
            return None, None
            
        try:
            # Ensure observations are in correct format
            obs_array = np.array(observations).reshape(-1, 1) if np.array(observations).ndim == 1 else np.array(observations)
            
            # Use hmmlearn's decode method which implements Viterbi
            log_prob, state_sequence = self.hmm_model.decode(obs_array, algorithm="viterbi")
            
            logger.info(f"Viterbi algorithm completed. Log probability: {log_prob:.4f}")
            
            # Store for analysis
            self.state_history = list(state_sequence)
            
            return state_sequence, log_prob
            
        except Exception as e:
            logger.error(f"Viterbi algorithm failed: {e}")
            return None, None
    
    def baum_welch_algorithm(self, observations, max_iterations=100, tolerance=1e-6):
        """
        Baum-Welch algorithm for HMM parameter estimation
        
        Args:
            observations: Sequence of observed musical features
            max_iterations: Maximum number of EM iterations
            tolerance: Convergence tolerance
            
        Returns:
            Trained HMM model and convergence history
        """
        if self.hmm_model is None:
            logger.warning("HMM model not initialized. Initializing with default parameters.")
            self.hmm_model = hmm.GaussianHMM(n_components=self.n_hidden_states, covariance_type="full")
            
        try:
            # Ensure observations are in correct format
            obs_array = np.array(observations).reshape(-1, 1) if np.array(observations).ndim == 1 else np.array(observations)
            
            # Store original parameters for monitoring
            original_logprob = self.hmm_model.score(obs_array)
            
            # Train model using Baum-Welch (EM algorithm)
            convergence_history = []
            
            for iteration in range(max_iterations):
                # Store current likelihood
                current_logprob = self.hmm_model.score(obs_array)
                convergence_history.append(current_logprob)
                
                # Perform one EM iteration
                self.hmm_model.fit(obs_array)
                
                # Check convergence
                new_logprob = self.hmm_model.score(obs_array)
                improvement = new_logprob - current_logprob
                
                logger.info(f"Baum-Welch iteration {iteration+1}: Log-likelihood = {new_logprob:.6f}, "
                           f"Improvement = {improvement:.6f}")
                
                if abs(improvement) < tolerance:
                    logger.info(f"Converged after {iteration+1} iterations")
                    break
            
            final_logprob = self.hmm_model.score(obs_array)
            improvement = final_logprob - original_logprob
            
            logger.info(f"Baum-Welch training completed. "
                       f"Final log-likelihood: {final_logprob:.6f}, "
                       f"Total improvement: {improvement:.6f}")
            
            return self.hmm_model, convergence_history
            
        except Exception as e:
            logger.error(f"Baum-Welch algorithm failed: {e}")
            return None, None
    
    def predict_next_note_hmm(self, note_sequence, use_viterbi=True):
        """
        Predict next note using HMM algorithms
        
        Args:
            note_sequence: Current sequence of notes
            use_viterbi: Whether to use Viterbi for state estimation
            
        Returns:
            Predicted next note and confidence
        """
        if self.hmm_model is None or not self.trained:
            logger.warning("HMM model not trained. Using fallback prediction.")
            return self._fallback_note_prediction(note_sequence)
            
        try:
            # Extract features from note sequence
            features = self._sequence_to_features(note_sequence)
            
            if use_viterbi:
                # Use Viterbi to find most likely state sequence
                state_sequence, log_prob = self.viterbi_algorithm(features)
                if state_sequence is not None:
                    current_state = state_sequence[-1]
                else:
                    current_state = 0
            else:
                # Use forward algorithm for state prediction
                alpha, log_likelihood = self.forward_algorithm(features)
                if alpha is not None:
                    # Get most likely current state
                    current_state = np.argmax(alpha[-1])
                else:
                    current_state = 0
            
            # Predict next note based on current state and transitions
            next_note = self._predict_from_state(current_state, note_sequence)
            
            # Calculate confidence based on state probability
            confidence = self._calculate_prediction_confidence(current_state, features)
            
            return next_note, confidence
            
        except Exception as e:
            logger.error(f"HMM prediction failed: {e}")
            return self._fallback_note_prediction(note_sequence)
    
    def _sequence_to_features(self, note_sequence):
        """Convert note sequence to feature vectors for HMM"""
        if len(note_sequence) < 2:
            return [[60.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.5]]  # Default feature
            
        features = []
        window_size = min(4, len(note_sequence))
        
        for i in range(len(note_sequence) - window_size + 1):
            window = note_sequence[i:i+window_size]
            
            # Extract note values
            notes = []
            for note_data in window:
                if isinstance(note_data, (list, tuple)):
                    notes.append(note_data[0])
                else:
                    notes.append(note_data)
            
            # Calculate features
            pitch_mean = np.mean(notes)
            pitch_std = np.std(notes) if len(notes) > 1 else 0
            pitch_range = max(notes) - min(notes) if len(notes) > 1 else 0
            
            # Interval features
            intervals = [notes[j+1] - notes[j] for j in range(len(notes)-1)]
            interval_mean = np.mean(intervals) if intervals else 0
            interval_std = np.std(intervals) if len(intervals) > 1 else 0
            
            # Contour
            ascending = sum(1 for x in intervals if x > 0)
            contour_ratio = ascending / max(len(intervals), 1)
            
            feature_vector = [
                pitch_mean, pitch_std, pitch_range, 0.25,  # duration placeholder
                interval_mean, interval_std, contour_ratio
            ]
            features.append(feature_vector)
        
        return features if features else [[60.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.5]]
    
    def _predict_from_state(self, state, note_sequence):
        """Predict next note based on HMM state and Markov transitions"""
        try:
            # Get last few notes for context
            context = note_sequence[-min(self.order, len(note_sequence)):]
            
            # Extract note values
            context_notes = []
            for note_data in context:
                if isinstance(note_data, (list, tuple)):
                    context_notes.append(note_data[0])
                else:
                    context_notes.append(note_data)
            
            if not context_notes:
                return 60  # Default middle C
                
            last_note = context_notes[-1]
            
            # Weight transition probabilities by HMM state
            if last_note in range(128):
                transitions = self.transitions[last_note] if hasattr(self.transitions, '__getitem__') else None
                
                if transitions is not None:
                    # Convert to CPU if needed
                    if hasattr(transitions, 'get'):
                        transitions = self.gpu_opt.to_cpu(transitions)
                    
                    # Apply HMM state weighting
                    weighted_transitions = self._weight_transitions_by_state(transitions, state)
                    
                    # Sample next note
                    if np.sum(weighted_transitions) > 0:
                        probabilities = weighted_transitions / np.sum(weighted_transitions)
                        next_note = np.random.choice(128, p=probabilities)
                        return int(next_note)
            
            # Fallback to interval-based prediction
            return self._interval_based_prediction(context_notes, state)
            
        except Exception as e:
            logger.warning(f"State-based prediction failed: {e}")
            return int(note_sequence[-1]) if note_sequence else 60
    
    def _weight_transitions_by_state(self, transitions, state):
        """Weight transition probabilities based on HMM state"""
        try:
            # Simple state-based weighting
            # In practice, you might learn these weights during training
            state_weights = np.ones_like(transitions)
            
            # Example: different states prefer different note ranges
            if state < self.n_hidden_states // 3:  # Lower states - prefer lower notes
                state_weights[60:] *= 0.7
                state_weights[:60] *= 1.3
            elif state < 2 * self.n_hidden_states // 3:  # Middle states - balanced
                pass  # No modification
            else:  # Higher states - prefer higher notes
                state_weights[:60] *= 0.7
                state_weights[60:] *= 1.3
            
            return transitions * state_weights
            
        except Exception:
            return transitions
    
    def _interval_based_prediction(self, context_notes, state):
        """Fallback prediction using interval transitions"""
        try:
            if len(context_notes) >= 2:
                last_interval = context_notes[-1] - context_notes[-2]
                
                # State-based interval preference
                if state < self.n_hidden_states // 2:
                    # Lower states prefer smaller intervals
                    preferred_intervals = [-2, -1, 0, 1, 2]
                else:
                    # Higher states allow larger intervals
                    preferred_intervals = [-5, -3, -2, -1, 0, 1, 2, 3, 5]
                
                # Choose interval based on state preference
                next_interval = np.random.choice(preferred_intervals)
                next_note = context_notes[-1] + next_interval
                
                # Clamp to MIDI range
                return max(0, min(127, next_note))
            
            return context_notes[-1] if context_notes else 60
            
        except Exception:
            return 60
    
    def _calculate_prediction_confidence(self, state, features):
        """Calculate confidence score for prediction"""
        try:
            if self.hmm_model is None or not features:
                return 0.5
                
            # Use state probability as confidence measure
            obs_array = np.array(features[-1]).reshape(1, -1)
            log_prob = self.hmm_model.score(obs_array)
            
            # Convert log probability to confidence score (0-1)
            confidence = min(1.0, max(0.1, np.exp(log_prob / 10)))
            return confidence
            
        except Exception:
            return 0.5
    
    def _fallback_note_prediction(self, note_sequence):
        """Simple fallback prediction when HMM fails"""
        if not note_sequence:
            return 60, 0.3
            
        last_note = note_sequence[-1]
        if isinstance(last_note, (list, tuple)):
            last_note = last_note[0]
            
        # Simple random walk
        interval = np.random.choice([-2, -1, 0, 1, 2], p=[0.1, 0.2, 0.4, 0.2, 0.1])
        next_note = max(0, min(127, last_note + interval))
        
        return next_note, 0.3
    
    def _initialize_hmm(self, sequences):
        """Initialize Hidden Markov Model with musical structure awareness"""
        logger.info("Initializing HMM with musical structure awareness...")
        
        # Extract feature vectors from sequences
        features = self._extract_hmm_features(sequences)
        
        if len(features) == 0:
            logger.warning("No features extracted for HMM initialization")
            return
        
        # Use KMeans to initialize hidden states
        if len(features) >= self.n_hidden_states:
            features_array = np.array(features)
            
            # GPU-accelerated clustering if available
            if self.use_gpu:
                try:
                    from cuml.cluster import KMeans as cuKMeans
                    kmeans = cuKMeans(n_clusters=self.n_hidden_states, random_state=42)
                    cluster_labels = kmeans.fit_predict(features_array)
                    cluster_labels = cp.asnumpy(cluster_labels) if hasattr(cluster_labels, 'get') else cluster_labels
                except ImportError:
                    logger.warning("cuML not available, using sklearn KMeans")
                    kmeans = KMeans(n_clusters=self.n_hidden_states, random_state=42)
                    cluster_labels = kmeans.fit_predict(features_array)
            else:
                kmeans = KMeans(n_clusters=self.n_hidden_states, random_state=42)
                cluster_labels = kmeans.fit_predict(features_array)
            
            # Initialize HMM
            self.hmm_model = hmm.GaussianHMM(n_components=self.n_hidden_states, covariance_type="full")
            
            try:
                self.hmm_model.fit(features_array.reshape(-1, 1) if features_array.ndim == 1 else features_array)
                logger.info(f"HMM initialized with {self.n_hidden_states} hidden states")
            except Exception as e:
                logger.warning(f"HMM fitting failed: {e}, using simpler model")
                self.hmm_model = None
        else:
            logger.warning(f"Insufficient data for HMM (need >= {self.n_hidden_states} samples)")
            
    def _extract_hmm_features(self, sequences):
        """Extract features for HMM training"""
        features = []
        
        for sequence in sequences[:min(len(sequences), 1000)]:  # Limit for performance
            pitches = []
            durations = []
            
            for note_data in sequence:
                if isinstance(note_data, (list, tuple)) and len(note_data) >= 2:
                    pitches.append(note_data[0])
                    durations.append(note_data[1])
                elif isinstance(note_data, int):
                    pitches.append(note_data)
                    durations.append(0.25)  # Default duration
                    
            if len(pitches) >= 4:  # Minimum sequence length
                # Extract statistical features
                pitch_mean = np.mean(pitches)
                pitch_std = np.std(pitches)
                pitch_range = max(pitches) - min(pitches)
                duration_mean = np.mean(durations) if durations else 0.25
                
                # Interval features
                intervals = [pitches[i+1] - pitches[i] for i in range(len(pitches)-1)]
                interval_mean = np.mean(intervals) if intervals else 0
                interval_std = np.std(intervals) if intervals else 0
                
                # Contour features
                ascending = sum(1 for i in intervals if i > 0)
                descending = sum(1 for i in intervals if i < 0)
                contour_ratio = ascending / max(len(intervals), 1)
                
                feature_vector = [
                    pitch_mean, pitch_std, pitch_range,
                    duration_mean, interval_mean, interval_std,
                    contour_ratio
                ]
                features.append(feature_vector)
                
        return features
        
    def train(self, midi_sequences, progress_callback=None):
        """Enhanced training with HMM initialization and GPU acceleration"""
        logger.info("Starting enhanced model training with HMM and GPU acceleration...")
        
        # Initialize HMM first
        logger.info("Initializing HMM structure...")
        self._initialize_hmm(midi_sequences)
        
        # Extract musical features with enhanced analysis
        if len(midi_sequences) <= 500:
            try:
                logger.info("Extracting enhanced musical features...")
                music_features = self.extract_musical_features(midi_sequences[:min(len(midi_sequences), 200)])
                
                if music_features:
                    # Process extracted features
                    self._process_time_signatures(music_features)
                    self._process_rhythm_patterns(music_features)
                    self._process_chord_progressions(music_features)
                    self._process_key_features(music_features)
                    self._process_roman_numeral_transitions(music_features)
                    
                    # Enhanced processing for new features
                    self._process_enhanced_features(music_features)
            except Exception as e:
                logger.warning(f"Feature extraction issue: {e}. Using simplified training.")
        else:
            logger.info("Large dataset detected, using simplified feature extraction")
        
        # Train core transition matrices with GPU acceleration
        logger.info("Training enhanced note transitions...")
        self._train_enhanced_note_transitions(midi_sequences, progress_callback)
        
        logger.info("Training enhanced interval transitions...")
        self._train_enhanced_interval_transitions(midi_sequences, progress_callback)
        
        # Train higher-order transitions (up to order 6)
        logger.info("Training higher-order transitions...")
        self._train_higher_order_transitions(midi_sequences, progress_callback)
        
        # Finalize GPU matrices
        if self.use_gpu:
            logger.info("Finalizing GPU matrices...")
            self._finalize_gpu_matrices()
        
        self.trained = True
        logger.info("Enhanced training complete!")
        return True
        
    def _process_enhanced_features(self, music_features):
        """Process enhanced musical features"""
        # Melodic contours
        if 'melodic_contours' in music_features:
            contour_counter = Counter(music_features['melodic_contours'])
            self.musical_features['melodic_contours'] = dict(contour_counter.most_common(30))
            
        # Dynamic patterns
        if 'dynamic_patterns' in music_features:
            dynamic_counter = Counter(music_features['dynamic_patterns'])
            self.musical_features['dynamic_patterns'] = dict(dynamic_counter.most_common(20))
            
        # Phrase boundaries
        if 'phrase_boundaries' in music_features:
            phrase_counter = Counter(music_features['phrase_boundaries'])
            self.musical_features['phrase_boundaries'] = dict(phrase_counter.most_common(15))
            
        logger.info("Enhanced musical features processed")
        
    def _train_enhanced_note_transitions(self, midi_sequences, progress_callback=None):
        """GPU-accelerated note transition training with HMM awareness"""
        logger.info("Training enhanced note transitions with GPU acceleration...")
        
        # Initialize GPU arrays if available and working
        gpu_transitions = None
        use_gpu_for_this = self.use_gpu and HAS_CUPY
        
        if use_gpu_for_this:
            try:
                gpu_transitions = self.gpu_opt.zeros_like_gpu((128, 128), dtype=np.float32)
                logger.info("🚀 GPU arrays initialized successfully")
            except Exception as e:
                logger.warning(f"⚠️ GPU array initialization failed: {e}, falling back to CPU")
                use_gpu_for_this = False
                gpu_transitions = None
        
        if not use_gpu_for_this:
            gpu_transitions = self.transitions.copy()
            
        multi_order_counts = defaultdict(Counter)
        total_processed = 0
        
        logger.info(f"Processing {len(midi_sequences)} sequences...")
        
        for sequence in midi_sequences:
            pitches = []
            for note_data in sequence:
                if isinstance(note_data, int):
                    pitches.append(note_data)
                elif isinstance(note_data, (list, tuple)) and len(note_data) >= 1:
                    pitches.append(note_data[0])
                    
            if len(pitches) < 2:
                continue
                
            # Single-order transitions with error handling
            try:
                for i in range(len(pitches) - 1):
                    prev_note, next_note = pitches[i], pitches[i + 1]
                    if 0 <= prev_note < 128 and 0 <= next_note < 128:
                        if use_gpu_for_this and gpu_transitions is not None:
                            try:
                                gpu_transitions[prev_note, next_note] += 1
                            except Exception as e:
                                # If GPU operation fails, switch to CPU
                                logger.warning(f"GPU operation failed, switching to CPU: {e}")
                                use_gpu_for_this = False
                                # Convert back to CPU and continue
                                if hasattr(gpu_transitions, 'get'):
                                    self.transitions = self.gpu_opt.to_cpu(gpu_transitions)
                                gpu_transitions = self.transitions
                                gpu_transitions[prev_note, next_note] += 1
                        else:
                            gpu_transitions[prev_note, next_note] += 1
            except Exception as e:
                logger.warning(f"Error processing sequence: {e}")
                continue
            
            # Multi-order transitions (up to order 6)
            max_order = min(6, self.order)
            for order in range(2, max_order + 1):
                if len(pitches) > order:
                    for i in range(len(pitches) - order):
                        context = tuple(pitches[i:i+order])
                        next_note = pitches[i+order]
                        if all(0 <= n < 128 for n in context) and 0 <= next_note < 128:
                            multi_order_counts[(order, context)][next_note] += 1
                        
            total_processed += 1
            if progress_callback and total_processed % 100 == 0:
                progress_callback(0.3 * total_processed / len(midi_sequences))
                
        # Normalize transition matrix with GPU acceleration if possible
        try:
            if use_gpu_for_this and HAS_CUPY and hasattr(gpu_transitions, 'get'):
                logger.info("🚀 Normalizing matrices on GPU...")
                gpu_transitions = self.gpu_opt.normalize_gpu(gpu_transitions, axis=1)
                self.transitions = self.gpu_opt.to_cpu(gpu_transitions)
            else:
                logger.info("💻 Normalizing matrices on CPU...")
                row_sums = gpu_transitions.sum(axis=1, keepdims=True)
                np.divide(gpu_transitions, np.where(row_sums == 0, 1.0, row_sums), out=gpu_transitions)
                self.transitions = gpu_transitions
        except Exception as e:
            logger.warning(f"GPU normalization failed: {e}, using CPU")
            if hasattr(gpu_transitions, 'get'):
                gpu_transitions = self.gpu_opt.to_cpu(gpu_transitions)
            row_sums = gpu_transitions.sum(axis=1, keepdims=True)
            np.divide(gpu_transitions, np.where(row_sums == 0, 1.0, row_sums), out=gpu_transitions)
            self.transitions = gpu_transitions
        
        # Process multi-order transitions
        for (order, context), transitions in multi_order_counts.items():
            total = sum(transitions.values())
            if total > 0:
                if order not in self.musical_features['multi_order_transitions']:
                    self.musical_features['multi_order_transitions'][order] = {}
                self.musical_features['multi_order_transitions'][order][context] = {
                    note: count/total for note, count in transitions.items()
                }
                
        logger.info(f"✅ Enhanced note transitions trained on {total_processed} sequences")
        
    def _train_enhanced_interval_transitions(self, midi_sequences, progress_callback=None):
        """Enhanced interval transition training with GPU support"""
        logger.info("Training enhanced interval transitions...")
        
        # Use defaultdict with GPU arrays if available
        if self.use_gpu:
            interval_counts = defaultdict(lambda: self.gpu_opt.zeros_like_gpu(self.interval_range, dtype=np.float32))
        else:
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
                progress_callback(0.3 + 0.3 * total_processed / len(midi_sequences))
                
        # Normalize and store with GPU optimization
        for prev_note, counts in interval_counts.items():
            if self.use_gpu:
                counts_cpu = self.gpu_opt.to_cpu(counts)
                row_sum = counts_cpu.sum()
            else:
                row_sum = counts.sum()
                counts_cpu = counts
                
            if row_sum > 0:
                self.interval_transitions[prev_note] = counts_cpu / row_sum
                
        logger.info(f"Enhanced interval features trained on {total_processed} sequences")
        
    def _train_higher_order_transitions(self, midi_sequences, progress_callback=None):
        """Train higher-order transitions with GPU acceleration"""
        logger.info("Training higher-order transitions (orders 2-6)...")
        
        for order in range(2, min(7, self.order + 1)):
            logger.info(f"Training order-{order} transitions...")
            
            if self.use_gpu:
                # Use sparse representation for higher orders due to memory constraints
                transition_counts = defaultdict(Counter)
            else:
                transition_counts = defaultdict(Counter)
            
            total_processed = 0
            
            for sequence in midi_sequences:
                pitches = []
                for note_data in sequence:
                    if isinstance(note_data, int):
                        pitches.append(note_data)
                    elif isinstance(note_data, (list, tuple)) and len(note_data) >= 1:
                        pitches.append(note_data[0])
                        
                if len(pitches) <= order:
                    continue
                    
                for i in range(len(pitches) - order):
                    context = tuple(pitches[i:i+order])
                    next_note = pitches[i+order]
                    
                    if all(0 <= n < 128 for n in context) and 0 <= next_note < 128:
                        transition_counts[context][next_note] += 1
                        
                total_processed += 1
                
            # Normalize and store
            normalized_transitions = {}
            for context, transitions in transition_counts.items():
                total = sum(transitions.values())
                if total > 3:  # Only keep contexts with sufficient data
                    normalized_transitions[context] = {
                        note: count/total for note, count in transitions.items()
                    }
            
            if normalized_transitions:
                self.higher_order_transitions[order] = normalized_transitions
                logger.info(f"Stored {len(normalized_transitions)} order-{order} contexts")
            
            if progress_callback:
                progress_callback(0.6 + 0.3 * (order - 1) / 5)
                
    def _finalize_gpu_matrices(self):
        """Finalize and optimize GPU matrices"""
        if not self.use_gpu:
            return
            
        logger.info("Finalizing GPU matrices...")
        
        # Convert main transition matrix back to CPU for saving
        if hasattr(self.transitions, 'get'):  # CuPy array
            self.transitions = self.gpu_opt.to_cpu(self.transitions)
            
        # Clean up GPU memory
        try:
            if HAS_CUPY:
                cp.get_default_memory_pool().free_all_blocks()
        except:
            pass
            
        logger.info("GPU matrices finalized")

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
        
    def _map_notes_to_chords(self, notes, durations, chords):
        """Map each note to the appropriate chord in the progression"""
        if not notes or not chords:
            return []
            
        chord_sequence = []
        total_notes = len(notes)
        notes_per_chord = max(1, total_notes // len(chords))
        
        for i in range(total_notes):
            chord_idx = min(i // notes_per_chord, len(chords) - 1)
            chord_sequence.append(chords[chord_idx])
            
        return chord_sequence

    def _adjust_notes_to_chords(self, notes, chord_sequence):
        """Adjust notes to better fit with their corresponding chords"""
        if not notes or not chord_sequence:
            return notes
            
        adjusted_notes = []
        
        try:
            from music21 import harmony
            
            for i, (note_value, chord_name) in enumerate(zip(notes, chord_sequence)):
                # Try to create a chord object
                try:
                    chord_obj = harmony.ChordSymbol(chord_name)
                    chord_pitches = set([p.midi % 12 for p in chord_obj.pitches])
                    
                    note_pitch_class = note_value % 12
                    
                    # If note is already in chord, keep it
                    if note_pitch_class in chord_pitches:
                        adjusted_notes.append(note_value)
                        continue
                        
                    # Otherwise, find the nearest chord tone
                    distance_to_chord = [(abs((pc - note_pitch_class) % 12), pc) for pc in chord_pitches]
                    nearest_pitch_class = min(distance_to_chord, key=lambda x: x[0])[1]
                    
                    # Adjust the note to the nearest chord tone while preserving octave
                    octave = note_value // 12
                    adjusted_note = (octave * 12) + nearest_pitch_class
                    
                    # Ensure the note is in valid MIDI range
                    if 0 <= adjusted_note < 128:
                        adjusted_notes.append(adjusted_note)
                    else:
                        adjusted_notes.append(note_value)  # Keep original if out of range
                        
                except Exception:
                    # If chord parsing fails, keep the original note
                    adjusted_notes.append(note_value)
                    
        except ImportError:
            # If music21 is not available, return the original notes
            return notes
            
        return adjusted_notes

    def _generate_fallback_sequence(self, length):
        """Generate a simple sequence when other methods fail"""
        # Create a C major scale
        c_major = [60, 62, 64, 65, 67, 69, 71, 72]
        
        # Generate random notes from the scale
        import random
        notes = [random.choice(c_major) for _ in range(length)]
        
        # Generate simple durations (quarter and eighth notes)
        durations = [0.5 if random.random() > 0.5 else 0.25 for _ in range(length)]
        
        # Create simple output
        return {
            'notes': notes,
            'durations': durations,
            'chords': ['C', 'G', 'Am', 'F'] * (length // 4 + 1),
            'beat_positions': [i % 4 for i in range(length)],
            'key': 'C major',
            'time_signature': '4/4'
        }

    def generate_expressive_sequence(self, key_context=None, length=64, complexity=0.7):
        """Generate an expressive musical sequence with proper rhythm and harmony"""
        try:
            # Clean the key context
            cleaned_key = self._clean_key_context(key_context)
            logger.info(f"Generating expressive sequence in key: {cleaned_key}")
            
            # Generate with chords for more musical structure
            return self.generate_with_chords(
                key_context=cleaned_key,
                length=length,
                time_signature="4/4"
            )
        except Exception as e:
            logger.error(f"Error in generate_expressive_sequence: {e}")
            return self._generate_fallback_sequence(length)
        
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
            
            # Generate chord progression
            num_chords = max(8, length // 4)
            chords = self.generate_chord_progression(key_context=cleaned_key, num_chords=num_chords)
            
            # Determine starting note
            start_note = self._determine_start_note(cleaned_key)
            
            # Handle time signature
            if time_signature not in self.musical_features['time_signatures']:
                time_signature = "4/4"
                if self.musical_features['time_signatures']:
                    time_signature = max(self.musical_features['time_signatures'].items(), key=lambda x: x[1])[0]
                    
            # Calculate appropriate measures and note density
            target_density = self.musical_features['note_density'].get(time_signature, 4)
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
            
            # Process and harmonize the sequence
            notes = [n for n, _, _ in notes_with_timings]
            durations = [d for _, d, _ in notes_with_timings]
            beat_positions = [p for _, _, p in notes_with_timings]
            
            # Map notes to appropriate chords
            chord_sequence = self._map_notes_to_chords(notes, durations, chords)
            
            # Apply chord-aware pitch correction
            adjusted_notes = self._adjust_notes_to_chords(notes, chord_sequence)
            
            # Return the complete sequence information
            return {
                'notes': adjusted_notes,
                'durations': durations[:len(adjusted_notes)],
                'chords': chord_sequence[:len(adjusted_notes)],
                'beat_positions': beat_positions[:len(adjusted_notes)],
                'key': cleaned_key,
                'time_signature': time_signature
            }
            
        except Exception as e:
            logger.error(f"Error in generate_with_chords: {e}")

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
        """Properly clean and format key string for music21 compatibility"""
        if key_context is None or not isinstance(key_context, str):
            return "C major"  # Default key
            
        try:
            # Handle common issues with key parsing
            key_str = key_context.strip()
            
            # Split into tonic and mode
            if ' ' in key_str:
                parts = key_str.split(' ', 1)
                tonic, mode = parts[0], parts[1]
                
                # Normalize tonic name
                tonic = tonic.capitalize()
                
                # Normalize mode name
                mode = mode.lower()
                if mode not in ['major', 'minor', 'dorian', 'phrygian', 'lydian', 'mixolydian', 'aeolian', 'locrian']:
                    mode = 'major'  # Default to major if invalid mode
                    
                # Format properly for music21
                from music21 import key
                key_obj = key.Key(tonic, mode)
                return f"{tonic} {mode}"
            else:
                # Assume major if only tonic is provided
                return f"{key_str} major"
                
        except Exception as e:
            logger.warning(f"Failed to parse key '{key_context}': {e}, defaulting to C major")
            return "C major"

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
    
    # Enhanced generation methods with HMM and higher-order transitions
    
    def generate_with_hmm(self, length=64, key_context=None, use_hidden_states=True):
        """Generate sequence using HMM and higher-order transitions"""
        if not self.trained or not self.hmm_model:
            logger.warning("HMM not available, falling back to standard generation")
            return self.generate_expressive_sequence(key_context, length)
        
        try:
            # Generate hidden state sequence
            if use_hidden_states:
                hidden_states = self._generate_hidden_state_sequence(length)
            else:
                hidden_states = [0] * length
                
            # Generate notes based on hidden states and higher-order context
            sequence = []
            
            # Start with a musically appropriate note
            start_note = self._determine_start_note(key_context)
            sequence.append(start_note)
            
            for i in range(1, length):
                # Use the highest available order for context
                context_notes = sequence[-min(self.order, len(sequence)):]
                next_note = self._predict_next_note_hmm(
                    context_notes, 
                    hidden_states[i] if i < len(hidden_states) else 0,
                    key_context
                )
                sequence.append(next_note)
                
            # Generate timing and rhythm
            durations = self._generate_hmm_durations(sequence, hidden_states[:len(sequence)])
            
            return {
                'notes': sequence,
                'durations': durations,
                'hidden_states': hidden_states[:len(sequence)],
                'key': key_context or 'C major'
            }
            
        except Exception as e:
            logger.error(f"HMM generation failed: {e}")
            return self.generate_expressive_sequence(key_context, length)
    
    def _generate_hidden_state_sequence(self, length):
        """Generate sequence of hidden states using HMM"""
        try:
            # Sample from HMM
            states, _ = self.hmm_model.sample(length)
            return states.flatten().tolist()
        except Exception as e:
            logger.warning(f"HMM state generation failed: {e}")
            # Fallback to simple state pattern
            return [i % self.n_hidden_states for i in range(length)]
    
    def _predict_next_note_hmm(self, context_notes, hidden_state, key_context):
        """Predict next note using HMM state and higher-order context"""
        try:
            # Try highest order first
            for order in range(min(len(context_notes), 6), 0, -1):
                if order in self.higher_order_transitions:
                    context = tuple(context_notes[-order:])
                    if context in self.higher_order_transitions[order]:
                        transitions = self.higher_order_transitions[order][context]
                        
                        # Weight by hidden state preference
                        weighted_transitions = self._weight_by_hidden_state(
                            transitions, hidden_state, key_context
                        )
                        
                        if weighted_transitions:
                            notes = list(weighted_transitions.keys())
                            probs = list(weighted_transitions.values())
                            return random.choices(notes, weights=probs)[0]
            
            # Fallback to standard transition
            if len(context_notes) > 0:
                current_note = context_notes[-1]
                if 0 <= current_note < 128:
                    probs = self.transitions[current_note]
                    if probs.sum() > 0:
                        return np.random.choice(128, p=probs)
            
            # Final fallback
            scale_pitches = self._get_scale_pitches(key_context)
            if scale_pitches:
                return random.choice([n for n in scale_pitches if 48 <= n <= 84])
            else:
                return random.randint(48, 84)
                
        except Exception as e:
            logger.warning(f"HMM note prediction failed: {e}")
            return random.randint(48, 84)
    
    def _weight_by_hidden_state(self, transitions, hidden_state, key_context):
        """Weight note transitions by hidden state preferences"""
        try:
            # Get scale pitches for the key
            scale_pitches = self._get_scale_pitches(key_context)
            
            weighted = {}
            for note, prob in transitions.items():
                weight = prob
                
                # Boost notes in the current key
                if scale_pitches and note in scale_pitches:
                    weight *= 1.5
                
                # Hidden state preferences (simplified model)
                state_mod = hidden_state % 4
                if state_mod == 0:  # Stable state - prefer consonant intervals
                    if note % 12 in [0, 4, 7]:  # Root, third, fifth
                        weight *= 1.3
                elif state_mod == 1:  # Ascending state
                    weight *= (1.2 if note > max(transitions.keys()) * 0.5 else 0.8)
                elif state_mod == 2:  # Descending state
                    weight *= (1.2 if note < max(transitions.keys()) * 0.5 else 0.8)
                else:  # Transitional state - prefer stepwise motion
                    prev_notes = list(transitions.keys())
                    if prev_notes:
                        avg_note = sum(prev_notes) / len(prev_notes)
                        if abs(note - avg_note) <= 2:  # Within step
                            weight *= 1.4
                
                if weight > 0:
                    weighted[note] = weight
            
            # Normalize
            total_weight = sum(weighted.values())
            if total_weight > 0:
                return {note: weight/total_weight for note, weight in weighted.items()}
            
            return transitions
            
        except Exception:
            return transitions
    
    def _generate_hmm_durations(self, notes, hidden_states):
        """Generate durations based on HMM states"""
        durations = []
        
        for i, (note, state) in enumerate(zip(notes, hidden_states)):
            # Duration based on hidden state
            state_mod = state % 4
            
            if state_mod == 0:  # Stable - longer notes
                duration = random.choices([0.5, 1.0, 1.5], weights=[0.5, 0.4, 0.1])[0]
            elif state_mod == 1:  # Active - varied durations
                duration = random.choices([0.25, 0.5, 0.75], weights=[0.4, 0.5, 0.1])[0]
            elif state_mod == 2:  # Flowing - medium durations
                duration = random.choices([0.375, 0.5, 0.75], weights=[0.3, 0.6, 0.1])[0]
            else:  # Transitional - short notes
                duration = random.choices([0.125, 0.25, 0.5], weights=[0.3, 0.5, 0.2])[0]
            
            durations.append(duration)
        
        return durations
    
    def generate_with_adaptive_order(self, length=64, key_context=None, complexity=0.7):
        """Generate sequence with adaptive order selection based on context"""
        if not self.trained:
            return self.generate_expressive_sequence(key_context, length)
        
        sequence = []
        start_note = self._determine_start_note(key_context)
        sequence.append(start_note)
        
        scale_pitches = self._get_scale_pitches(key_context)
        
        for i in range(1, length):
            # Adaptively choose order based on position and complexity
            max_available_order = min(len(sequence), 6)
            
            # Higher complexity prefers higher order
            if complexity > 0.8:
                preferred_order = max_available_order
            elif complexity > 0.5:
                preferred_order = max(3, max_available_order - 1)
            else:
                preferred_order = max(2, max_available_order - 2)
            
            # Try orders from high to low
            next_note = None
            for order in range(min(preferred_order, max_available_order), 0, -1):
                context = tuple(sequence[-order:])
                
                # Check multiple transition sources
                transitions = None
                if order in self.higher_order_transitions and context in self.higher_order_transitions[order]:
                    transitions = self.higher_order_transitions[order][context]
                elif order <= 3 and order in self.musical_features.get('multi_order_transitions', {}):
                    if context in self.musical_features['multi_order_transitions'][order]:
                        transitions = self.musical_features['multi_order_transitions'][order][context]
                
                if transitions:
                    # Apply scale filtering
                    if scale_pitches:
                        filtered_transitions = {
                            note: prob for note, prob in transitions.items()
                            if note in scale_pitches
                        }
                        if filtered_transitions:
                            total = sum(filtered_transitions.values())
                            filtered_transitions = {
                                note: prob/total for note, prob in filtered_transitions.items()
                            }
                            transitions = filtered_transitions
                    
                    # Sample from transitions
                    if transitions:
                        notes_list = list(transitions.keys())
                        probs_list = list(transitions.values())
                        next_note = random.choices(notes_list, weights=probs_list)[0]
                        break
            
            # Fallback to interval-based generation
            if next_note is None:
                if sequence[-1] in self.interval_transitions:
                    interval_probs = self.interval_transitions[sequence[-1]]
                    interval_idx = np.random.choice(len(interval_probs), p=interval_probs)
                    interval = interval_idx - self.max_interval
                    next_note = max(0, min(127, sequence[-1] + interval))
                else:
                    # Final fallback
                    if scale_pitches:
                        valid_notes = [n for n in scale_pitches if 48 <= n <= 84]
                        if valid_notes:
                            next_note = random.choice(valid_notes)
                        else:
                            next_note = random.randint(48, 84)
                    else:
                        next_note = sequence[-1] + random.randint(-5, 5)
                        next_note = max(0, min(127, next_note))
            
            sequence.append(next_note)
        
        return sequence
    
    def generate_structured_piece(self, length=128, key_context=None, sections=4):
        """Generate a structured musical piece with multiple sections"""
        if not self.trained:
            return self.generate_expressive_sequence(key_context, length)
        
        section_length = length // sections
        piece = {'notes': [], 'durations': [], 'sections': [], 'key': key_context or 'C major'}
        
        current_key = key_context
        
        for section_idx in range(sections):
            logger.info(f"Generating section {section_idx + 1}/{sections}")
            
            # Vary complexity and key for different sections
            if section_idx == 0:  # Introduction
                complexity = 0.5
                section_key = current_key
            elif section_idx == sections - 1:  # Conclusion
                complexity = 0.6
                section_key = current_key  # Return to original key
            else:  # Development sections
                complexity = min(0.9, 0.6 + section_idx * 0.1)
                # Optionally modulate to related keys
                if random.random() < 0.3:
                    section_key = self._get_related_key(current_key)
                else:
                    section_key = current_key
            
            # Generate section with HMM if available
            if self.hmm_model:
                section_data = self.generate_with_hmm(
                    length=section_length,
                    key_context=section_key,
                    use_hidden_states=True
                )
            else:
                section_notes = self.generate_with_adaptive_order(
                    length=section_length,
                    key_context=section_key,
                    complexity=complexity
                )
                section_data = {
                    'notes': section_notes,
                    'durations': [0.5] * len(section_notes)
                }
            
            # Add section to piece
            piece['notes'].extend(section_data['notes'])
            piece['durations'].extend(section_data['durations'])
            piece['sections'].append({
                'start': len(piece['notes']) - len(section_data['notes']),
                'end': len(piece['notes']),
                'key': section_key,
                'complexity': complexity
            })
        
        return piece
    
    def _get_related_key(self, key_context):
        """Get a musically related key for modulation"""
        if not key_context:
            return "C major"
        
        try:
            k = key.Key(key_context)
            
            # Get relative and closely related keys
            related_keys = []
            
            if k.mode == 'major':
                # Relative minor
                relative_minor = k.relative
                related_keys.append(f"{relative_minor.tonic.name} minor")
                
                # Dominant and subdominant
                dominant = k.tonic.transpose(7)  # Perfect fifth up
                subdominant = k.tonic.transpose(-5)  # Perfect fourth down
                related_keys.extend([
                    f"{dominant.name} major",
                    f"{subdominant.name} major"
                ])
            else:  # minor
                # Relative major
                relative_major = k.relative
                related_keys.append(f"{relative_major.tonic.name} major")
                
                # Mediant and submediant
                mediant = k.tonic.transpose(3)  # Minor third up
                submediant = k.tonic.transpose(-4)  # Major sixth down
                related_keys.extend([
                    f"{mediant.name} major",
                    f"{submediant.name} major"
                ])
            
            return random.choice(related_keys) if related_keys else key_context
            
        except Exception:
            return key_context