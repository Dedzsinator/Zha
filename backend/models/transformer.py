import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    Positional encoding for helping the model understand sequence position
    
    Args:
        embed_dim: Embedding dimension
        max_len: Maximum sequence length to pre-compute encodings for
    """
    def __init__(self, embed_dim, max_len=2048):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        Add positional encoding to input embeddings
        
        Args:
            x: Input embeddings [batch, seq_len, embed_dim]
            
        Returns:
            Embeddings with positional information added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiTrackAttention(nn.Module):
    """
    Multi-track attention mechanism for coordinating melody, bass, and drums.
    
    This module allows different musical tracks to attend to each other,
    ensuring harmonic and rhythmic coherence across the arrangement.
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(MultiTrackAttention, self).__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, 
            num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query_track, key_value_track):
        """
        Apply cross-attention between two tracks.
        
        Args:
            query_track: The track attending to another [batch, seq_len, embed_dim]
            key_value_track: The track being attended to [batch, seq_len, embed_dim]
        
        Returns:
            Attended representation of query_track
        """
        # Cross-attention
        attn_output, _ = self.cross_attention(
            query_track, 
            key_value_track, 
            key_value_track
        )
        
        # Residual connection and normalization
        output = self.norm(query_track + self.dropout(attn_output))
        return output

class TransformerModel(nn.Module):
    def __init__(self, input_dim=128, embed_dim=512, num_heads=8,
                num_layers=8, dim_feedforward=2048, dropout=0.1, 
                enable_multitrack=False, enable_conditioning=True):
        """
        Enhanced Music Transformer model for sequence generation with optional multi-track support.
        
        When enable_multitrack=True, the model generates coordinated melody, bass, and drum tracks
        using specialized cross-attention mechanisms for harmonic and rhythmic coherence.
        
        When enable_conditioning=True, the model incorporates chord progressions and tempo changes
        as conditioning inputs for more musically coherent generation.
        
        Args:
            input_dim: Input dimension (typically MIDI notes = 128)
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            enable_multitrack: Enable multi-track generation (melody + bass + drums)
            enable_conditioning: Enable chord/tempo conditioning
        """
        super(TransformerModel, self).__init__()
        
        self.enable_multitrack = enable_multitrack
        self.input_dim = input_dim
        self.enable_conditioning = enable_conditioning
        
        # Input projection (shared across tracks for parameter efficiency)
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        # Conditioning components
        if enable_conditioning:
            # Chord conditioning: embed chord root, quality, and pitches
            self.chord_embedding = nn.Linear(12 + 7 + 12, embed_dim // 4)  # root(12) + quality(7) + pitches(12)
            # Tempo conditioning: embed tempo as a scalar
            self.tempo_embedding = nn.Linear(1, embed_dim // 4)
            # Combine conditioning embeddings
            self.conditioning_projection = nn.Linear(embed_dim // 2, embed_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim)
        
        # Main transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )
        
        # Multi-track components (only if enabled)
        if enable_multitrack:
            # Separate encoders for bass and drums to capture their unique characteristics
            self.bass_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=embed_dim, 
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=True
                ),
                num_layers=max(2, num_layers // 2)  # Lighter model for bass
            )
            
            self.drum_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=embed_dim, 
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=True
                ),
                num_layers=max(2, num_layers // 2)  # Lighter model for drums
            )
            
            # Cross-attention modules for inter-track coordination
            self.bass_melody_attention = MultiTrackAttention(embed_dim, num_heads, dropout)
            self.drum_melody_attention = MultiTrackAttention(embed_dim, num_heads, dropout)
            self.drum_bass_attention = MultiTrackAttention(embed_dim, num_heads, dropout)
            
            # Track-specific output projections
            self.melody_output = nn.Linear(embed_dim, input_dim)
            self.bass_output = nn.Linear(embed_dim, input_dim)
            
            # Drums use a subset of MIDI notes (typically 35-81 for General MIDI drums)
            # But we keep full input_dim and apply constraints during sampling
            self.drum_output = nn.Linear(embed_dim, input_dim)
            
            # Learnable track type embeddings (added to positional encoding)
            self.melody_type_emb = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            self.bass_type_emb = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            self.drum_type_emb = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        else:
            # Single-track output projection
            self.output_projection = nn.Linear(embed_dim, input_dim)
        
        # Initialize memory structures
        self.memory = None
        self.memory_length = 0
        self.section_memories = {}
        
        # Multi-track memory (if enabled)
        if enable_multitrack:
            self.bass_memory = None
            self.drum_memory = None
        
        # Initialize mask for auto-regressive generation
        self._generate_square_subsequent_mask = self._generate_mask

    def forward(self, x, use_memory=False, return_all_tracks=False, 
                chord_conditioning=None, tempo_conditioning=None):
        """
        Forward pass through the transformer.
        
        For multi-track models, generates melody, bass, and drums simultaneously
        with cross-attention for harmonic and rhythmic coherence.
        
        When conditioning is enabled, incorporates chord progressions and tempo changes
        as additional inputs for more musically coherent generation.
        
        Args:
            x: Input tensor [batch_size, input_dim] or [batch_size, seq_len, input_dim]
            use_memory: Whether to use stored memory for generation
            return_all_tracks: If multi-track is enabled, return dict with all tracks
            chord_conditioning: Chord conditioning tensor [batch_size, seq_len, chord_dim] or None
            tempo_conditioning: Tempo conditioning tensor [batch_size, seq_len, 1] or None
            
        Returns:
            Single-track: Output tensor [batch_size, input_dim] or [batch_size, seq_len, input_dim]
            Multi-track: Dictionary {'melody': ..., 'bass': ..., 'drums': ...}
        """
        # Handle input shape - add sequence dimension if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        if not self.enable_multitrack:
            # Standard single-track processing
            if use_memory and hasattr(self, 'memory') and self.memory is not None:
                x = torch.cat([self.memory, x], dim=1)
                # Extend conditioning if provided
                if chord_conditioning is not None and self.enable_conditioning:
                    chord_conditioning = torch.cat([torch.zeros_like(chord_conditioning[:, :self.memory.size(1), :]), chord_conditioning], dim=1)
                if tempo_conditioning is not None and self.enable_conditioning:
                    tempo_conditioning = torch.cat([torch.zeros_like(tempo_conditioning[:, :self.memory.size(1), :]), tempo_conditioning], dim=1)
            
            x_emb = self.embedding(x)
            
            # Add conditioning if enabled
            if self.enable_conditioning and (chord_conditioning is not None or tempo_conditioning is not None):
                conditioning_emb = self._get_conditioning_embedding(x_emb.size(0), x_emb.size(1), 
                                                                  chord_conditioning, tempo_conditioning, x.device)
                x_emb = x_emb + conditioning_emb
            
            x_emb = self.pos_encoder(x_emb)
            output = self.transformer_encoder(x_emb)
            output = self.output_projection(output)
            output = torch.nan_to_num(output, nan=0.0)
            
            if use_memory:
                max_memory_length = 1024
                self.memory = output.detach()
                if self.memory.size(1) > max_memory_length:
                    self.memory = self.memory[:, -max_memory_length:, :]
                self.memory_length = self.memory.size(1)
            
            # Don't squeeze for generation - keep 3D shape
            # if output.size(1) == 1:
            #     output = output.squeeze(1)
            
            return output
        
        else:
            # Multi-track processing with cross-attention
            batch_size, seq_len, _ = x.shape
            
            # Embed input for all tracks (shared embedding)
            x_emb = self.embedding(x)
            x_pos = self.pos_encoder(x_emb)
            
            # Add track-specific type embeddings
            melody_emb = x_pos + self.melody_type_emb.expand(batch_size, seq_len, -1)
            bass_emb = x_pos + self.bass_type_emb.expand(batch_size, seq_len, -1)
            drum_emb = x_pos + self.drum_type_emb.expand(batch_size, seq_len, -1)
            
            # Process melody track (main track)
            melody_hidden = self.transformer_encoder(melody_emb)
            
            # Process bass track with attention to melody (for harmonic coherence)
            bass_hidden = self.bass_encoder(bass_emb)
            bass_hidden = self.bass_melody_attention(bass_hidden, melody_hidden)
            
            # Process drum track with attention to both melody and bass
            drum_hidden = self.drum_encoder(drum_emb)
            drum_hidden = self.drum_melody_attention(drum_hidden, melody_hidden)
            drum_hidden = self.drum_bass_attention(drum_hidden, bass_hidden)
            
            # Project to output space
            melody_out = torch.nan_to_num(self.melody_output(melody_hidden), nan=0.0)
            bass_out = torch.nan_to_num(self.bass_output(bass_hidden), nan=0.0)
            drum_out = torch.nan_to_num(self.drum_output(drum_hidden), nan=0.0)
            
            # Update memory for all tracks if requested
            if use_memory:
                max_memory_length = 1024
                self.memory = melody_hidden.detach()
                self.bass_memory = bass_hidden.detach()
                self.drum_memory = drum_hidden.detach()
                
                if self.memory.size(1) > max_memory_length:
                    self.memory = self.memory[:, -max_memory_length:, :]
                    self.bass_memory = self.bass_memory[:, -max_memory_length:, :]
                    self.drum_memory = self.drum_memory[:, -max_memory_length:, :]
            
            if return_all_tracks:
                return {
                    'melody': melody_out,
                    'bass': bass_out,
                    'drums': drum_out
                }
            else:
                # Return melody by default for backward compatibility
                if melody_out.size(1) == 1:
                    melody_out = melody_out.squeeze(1)
                return melody_out

    def _get_conditioning_embedding(self, batch_size, seq_len, chord_conditioning, tempo_conditioning, device):
        """
        Create conditioning embeddings from chord and tempo information.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            chord_conditioning: Chord conditioning tensor [batch_size, seq_len, chord_dim] or None
            tempo_conditioning: Tempo conditioning tensor [batch_size, seq_len, 1] or None
            device: Target device
            
        Returns:
            Conditioning embedding tensor [batch_size, seq_len, embed_dim]
        """
        conditioning_parts = []
        
        if chord_conditioning is not None:
            # chord_conditioning should be [batch_size, seq_len, 31] where 31 = 12 (root one-hot) + 7 (quality one-hot) + 12 (pitch classes)
            chord_emb = self.chord_embedding(chord_conditioning)  # [batch_size, seq_len, embed_dim//4]
            conditioning_parts.append(chord_emb)
        else:
            # Default chord embedding (no conditioning)
            chord_emb = torch.zeros(batch_size, seq_len, self.chord_embedding.out_features, device=device)
            conditioning_parts.append(chord_emb)
            
        if tempo_conditioning is not None:
            # tempo_conditioning should be [batch_size, seq_len, 1] with normalized tempo values
            tempo_emb = self.tempo_embedding(tempo_conditioning)  # [batch_size, seq_len, embed_dim//4]
            conditioning_parts.append(tempo_emb)
        else:
            # Default tempo embedding (medium tempo)
            tempo_emb = torch.zeros(batch_size, seq_len, self.tempo_embedding.out_features, device=device)
            conditioning_parts.append(tempo_emb)
        
        # Concatenate conditioning embeddings
        combined_conditioning = torch.cat(conditioning_parts, dim=-1)  # [batch_size, seq_len, embed_dim//2]
        
        # Project to full embedding dimension
        conditioning_emb = self.conditioning_projection(combined_conditioning)  # [batch_size, seq_len, embed_dim]
        
        return conditioning_emb

    def _prepare_conditioning(self, chord_progression, tempo_curve, batch_size, total_length, device):
        """
        Prepare conditioning tensors from chord progression and tempo curve.
        
        Args:
            chord_progression: List of chord dicts or None
            tempo_curve: List of tempo values or None
            batch_size: Batch size
            total_length: Total sequence length (seed + generated)
            device: Target device
            
        Returns:
            Tuple of (chord_conditioning, tempo_conditioning) tensors
        """
        chord_conditioning = None
        tempo_conditioning = None
        
        if chord_progression is not None and len(chord_progression) > 0:
            # Convert chord progression to conditioning tensor
            chord_tensor = torch.zeros(batch_size, total_length, 31, device=device)  # 12 root + 7 quality + 12 pitches
            
            for i, chord in enumerate(chord_progression):
                if i >= total_length:
                    break
                    
                # Root (one-hot, 12 notes)
                if 'root' in chord and chord['root'] is not None:
                    root_idx = chord['root'] % 12
                    chord_tensor[:, i, root_idx] = 1.0
                
                # Quality (simplified: major, minor, dim, aug, sus4, 7, maj7)
                if 'quality' in chord and chord['quality'] is not None:
                    quality = chord['quality'].lower()
                    if 'major' in quality or quality == 'maj':
                        chord_tensor[:, i, 12] = 1.0  # major
                    elif 'minor' in quality or quality == 'min':
                        chord_tensor[:, i, 13] = 1.0  # minor
                    elif 'dim' in quality:
                        chord_tensor[:, i, 14] = 1.0  # diminished
                    elif 'aug' in quality:
                        chord_tensor[:, i, 15] = 1.0  # augmented
                    elif 'sus4' in quality:
                        chord_tensor[:, i, 16] = 1.0  # suspended 4th
                    elif '7' in quality and 'maj' not in quality:
                        chord_tensor[:, i, 17] = 1.0  # dominant 7th
                    elif 'maj7' in quality or '7' in quality:
                        chord_tensor[:, i, 18] = 1.0  # major 7th
                
                # Pitch classes (12 notes)
                if 'pitches' in chord and chord['pitches'] is not None:
                    for pitch in chord['pitches']:
                        pitch_class = pitch % 12
                        chord_tensor[:, i, 19 + pitch_class] = 1.0
            
            chord_conditioning = chord_tensor
        
        if tempo_curve is not None and len(tempo_curve) > 0:
            # Convert tempo curve to conditioning tensor
            tempo_tensor = torch.zeros(batch_size, total_length, 1, device=device)
            
            for i, tempo in enumerate(tempo_curve):
                if i >= total_length:
                    break
                # Normalize tempo (assuming 60-200 BPM range)
                normalized_tempo = (tempo - 60) / (200 - 60)  # Scale to [0, 1]
                normalized_tempo = torch.clamp(torch.tensor(normalized_tempo), 0.0, 1.0)
                tempo_tensor[:, i, 0] = normalized_tempo
            
            tempo_conditioning = tempo_tensor
        
        return chord_conditioning, tempo_conditioning

    def _generate_mask(self, sz):
        """Generate a square mask for the sequence"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def reset_memory(self):
        """Reset memory for a fresh generation"""
        self.memory = None
        self.memory_length = 0
        self.section_memories = {}
        
        # Reset multi-track memory if enabled
        if self.enable_multitrack:
            self.bass_memory = None
            self.drum_memory = None
    
    def generate(self, seed, steps=100, temperature=0.8, top_k=5, top_p=0.92,
                 chord_progression=None, tempo_curve=None):
        """
        Generate a sequence using the transformer
        
        Args:
            seed: Initial seed tensor [batch_size, seq_len, input_dim]
            steps: Number of steps to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Number of top probabilities to sample from
            top_p: Nucleus sampling probability threshold
            chord_progression: List of chord dicts or None for conditioning
            tempo_curve: List of tempo values or None for conditioning
            
        Returns:
            Generated sequence
        """
        self.eval()  # Set to evaluation mode
        
        # Initialize with seed
        x = seed
        
        # Reset memory
        self.reset_memory()
        
        # Prepare conditioning tensors if enabled
        chord_conditioning = None
        tempo_conditioning = None
        if self.enable_conditioning:
            chord_conditioning, tempo_conditioning = self._prepare_conditioning(
                chord_progression, tempo_curve, seed.size(0), steps + seed.size(1), seed.device)
        
        # Generate steps
        for i in range(steps):
            # Prepare conditioning for current sequence length
            current_chord = None
            current_tempo = None
            current_seq_len = x.size(1)
            
            if chord_conditioning is not None:
                # Slice conditioning to match current sequence length
                current_chord = chord_conditioning[:, :current_seq_len, :]
            if tempo_conditioning is not None:
                current_tempo = tempo_conditioning[:, :current_seq_len, :]
            
            # Forward pass with memory usage
            with torch.no_grad():
                output = self.forward(x, use_memory=(i > 0), 
                                    chord_conditioning=current_chord, 
                                    tempo_conditioning=current_tempo)
            
            # Get the last step's output
            next_token_logits = output[:, -1, :] / temperature
            next_token_logits = torch.nan_to_num(next_token_logits, nan=0.0)
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('inf')
            
            # Apply nucleus (top-p) filtering
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('inf')
            
            # Sample from the filtered distribution
            probs = F.softmax(next_token_logits, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0)
            probs = torch.clamp(probs, min=1e-8)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            
            if torch.isnan(probs).any() or (probs.sum(dim=-1) == 0).any():
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, num_samples=1)
            
            # Create one-hot vector for the next token
            next_token_one_hot = torch.zeros(
                seed.size(0), 1, seed.size(-1), 
                device=seed.device
            )
            
            # Fill the one-hot vector
            for batch_idx in range(seed.size(0)):
                next_token_one_hot[batch_idx, 0, next_token[batch_idx]] = 1.0
            
            # Concatenate with the input sequence
            x = torch.cat([x, next_token_one_hot], dim=1)
        
        # Return the full generated sequence
        return x
    
    def generate_with_structure(self, seed, num_sections=4, section_length=16, 
                              temperature=0.8, transition_smoothness=0.7):
        """
        Generate music with multiple distinct sections (verse, chorus, bridge, etc.)
        
        Args:
            seed: Initial seed tensor [batch_size, input_dim]
            num_sections: Number of sections to generate
            section_length: Length of each section
            temperature: Sampling temperature
            transition_smoothness: How smoothly to transition between sections (0-1)
            
        Returns:
            Generated sequence with distinct sections
        """
        self.eval()  # Set to evaluation mode
        batch_size = seed.size(0)
        
        # Reset memory
        self.reset_memory()
        
        # Initialize the output
        all_outputs = []
        
        # Generate each section
        for section_id in range(num_sections):
            # Use the seed for the first section, generated output for others
            current_seed = seed if section_id == 0 else last_section_output[:, -1:, :]
            
            # If we've seen this section before, load its memory
            if section_id in self.section_memories:
                self.memory = self.section_memories[section_id]
            else:
                # For new sections, optionally mix with previous section's memory
                if section_id > 0 and transition_smoothness > 0:
                    # Create a blended memory from previous section
                    prev_memory = self.section_memories.get(section_id - 1, None)
                    if prev_memory is not None:
                        # Initialize with some memory from previous section for smoother transitions
                        blend_len = int(prev_memory.size(1) * transition_smoothness)
                        if blend_len > 0:
                            self.memory = prev_memory[:, -blend_len:, :]
            
            # Generate this section
            section_output = self._generate_section(
                current_seed, 
                steps=section_length,
                temperature=temperature
            )
            
            # Store this section's memory for future reference
            self.section_memories[section_id] = self.memory
            
            # Save this section's output
            all_outputs.append(section_output)
            last_section_output = section_output
        
        # Concatenate all sections
        full_output = torch.cat(all_outputs, dim=1)
        return full_output
    
    def _generate_section(self, seed, steps=16, temperature=0.8):
        """Helper method to generate a single section"""
        # Ensure seed has proper dimensions [batch, seq_len, features] or [batch, features]
        x = seed
        if len(x.shape) == 2 and x.shape[1] != self.embedding.in_features:
            # Already has sequence dimension [batch, seq_len]
            pass
        elif len(x.shape) == 2:
            # Shape is [batch, features], add sequence dimension
            x = x.unsqueeze(1)  # [batch, 1, features]
        
        outputs = [x]
        
        # Generate steps
        for i in range(steps):
            # Forward pass with memory usage
            with torch.no_grad():
                output = self.forward(x, use_memory=(i > 0 or self.memory is not None))
            
            # Apply temperature
            # Ensure we're accessing the last token properly
            logits = output[:, -1, :] if output.dim() == 3 else output
            logits = logits / temperature
            
            # Sample from the output distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Create one-hot vector for the next token
            next_token_one_hot = torch.zeros(
                x.size(0), 1, self.embedding.in_features,  # Use model's input features 
                device=x.device
            )
            
            # Fill the one-hot vector
            for batch_idx in range(x.size(0)):
                if next_token[batch_idx].item() < self.embedding.in_features:
                    next_token_one_hot[batch_idx, 0, next_token[batch_idx]] = 1.0
            
            # Store this output
            outputs.append(next_token_one_hot)
            
            # Update x for the next iteration
            x = next_token_one_hot
        
        # Concatenate all outputs
        return torch.cat(outputs, dim=1)
    
    def generate_multitrack(self, seed, steps=100, temperature=0.8, 
                           bass_temperature=0.7, drum_temperature=0.9):
        """
        Generate coordinated melody, bass, and drum tracks.
        
        This method implements the multi-track generation algorithm where:
        1. Melody provides the harmonic foundation
        2. Bass follows the melody's harmonic progression (root notes, fifths)
        3. Drums provide rhythmic structure synchronized with bass
        
        Args:
            seed: Initial seed tensor [batch_size, input_dim]
            steps: Number of steps to generate
            temperature: Sampling temperature for melody (higher = more random)
            bass_temperature: Sampling temperature for bass (typically lower for stability)
            drum_temperature: Sampling temperature for drums (typically higher for variety)
            
        Returns:
            Dictionary containing:
                - 'melody': Generated melody track [batch_size, steps, input_dim]
                - 'bass': Generated bass track [batch_size, steps, input_dim]
                - 'drums': Generated drum track [batch_size, steps, input_dim]
                - 'combined': Combined multi-track representation
        """
        if not self.enable_multitrack:
            raise ValueError("Multi-track generation requires enable_multitrack=True")
        
        self.eval()  # Set to evaluation mode
        batch_size = seed.size(0)
        device = seed.device
        
        # Reset memory
        self.reset_memory()
        
        # Initialize with seed
        x = seed.unsqueeze(1) if len(seed.shape) == 2 else seed
        
        # Storage for generated tracks
        melody_sequence = []
        bass_sequence = []
        drum_sequence = []
        
        # MIDI drum note mappings (General MIDI standard)
        DRUM_NOTES = {
            'kick': 36,      # Bass Drum 1
            'snare': 38,     # Acoustic Snare
            'hihat_closed': 42,  # Closed Hi-Hat
            'hihat_open': 46,    # Open Hi-Hat
            'crash': 49,     # Crash Cymbal 1
            'ride': 51,      # Ride Cymbal 1
            'tom_low': 43,   # Low Tom
            'tom_mid': 47,   # Mid Tom
            'tom_high': 50   # High Tom
        }
        
        # Generate step by step
        for step in range(steps):
            with torch.no_grad():
                # Forward pass through multi-track model
                outputs = self.forward(x, use_memory=(step > 0), return_all_tracks=True)
                
                melody_logits = outputs['melody'][:, -1, :] / temperature
                bass_logits = outputs['bass'][:, -1, :] / bass_temperature
                drum_logits = outputs['drums'][:, -1, :] / drum_temperature
                
                # === MELODY GENERATION ===
                melody_probs = F.softmax(melody_logits, dim=-1)
                melody_note = torch.multinomial(melody_probs, num_samples=1)
                
                # === BASS GENERATION (Harmonically Constrained) ===
                # Bass follows melody's harmonic implications
                # Typically plays root notes, octaves, or fifths
                bass_probs = F.softmax(bass_logits, dim=-1)
                melody_pitch_class = melody_note % 12  # Get pitch class (C=0, C#=1, etc.)
                
                # Define harmonic bass notes (root, fifth, octave below)
                # Bass typically plays in lower register (MIDI 28-52, roughly E1-E3)
                bass_root = 28 + melody_pitch_class.item()  # Root in bass register
                bass_fifth = bass_root + 7  # Perfect fifth
                bass_octave_up = bass_root + 12  # Octave
                
                # Create weighted distribution favoring harmonic bass notes
                bass_probs_adjusted = bass_probs.clone()
                
                # Boost probability of harmonically relevant notes
                harmonic_boost = 10.0  # Strong bias toward harmonic notes
                if bass_root < self.input_dim:
                    bass_probs_adjusted[0, bass_root] *= harmonic_boost
                if bass_fifth < self.input_dim:
                    bass_probs_adjusted[0, bass_fifth] *= harmonic_boost / 2
                if bass_octave_up < self.input_dim:
                    bass_probs_adjusted[0, bass_octave_up] *= harmonic_boost / 3
                
                # Suppress notes outside bass range (keep only MIDI 28-52)
                bass_mask = torch.ones_like(bass_probs_adjusted)
                bass_mask[:, :28] = 0.01  # Very low probability for notes too low
                bass_mask[:, 53:] = 0.01  # Very low probability for notes too high
                bass_probs_adjusted *= bass_mask
                
                # Renormalize and sample
                bass_probs_adjusted = bass_probs_adjusted / bass_probs_adjusted.sum(dim=-1, keepdim=True)
                bass_note = torch.multinomial(bass_probs_adjusted, num_samples=1)
                
                # === DRUM GENERATION (Rhythmically Constrained) ===
                # Drums constrained to General MIDI drum notes (35-81)
                # Apply rhythmic patterns based on step position
                
                # Create drum-specific probability distribution
                drum_probs = F.softmax(drum_logits, dim=-1)
                drum_probs_adjusted = torch.zeros_like(drum_probs)
                
                # Define rhythmic pattern (4/4 time, 16th note resolution)
                beat_position = step % 16  # Position within a measure
                
                # Kick drum (strong beats: 0, 8) 
                if beat_position % 8 == 0:
                    drum_probs_adjusted[0, DRUM_NOTES['kick']] = 0.8
                elif beat_position % 4 == 0:
                    drum_probs_adjusted[0, DRUM_NOTES['kick']] = 0.3
                
                # Snare (backbeats: 4, 12)
                if beat_position in [4, 12]:
                    drum_probs_adjusted[0, DRUM_NOTES['snare']] = 0.7
                
                # Hi-hat (every 2 steps for constant rhythm)
                if beat_position % 2 == 0:
                    drum_probs_adjusted[0, DRUM_NOTES['hihat_closed']] = 0.6
                else:
                    drum_probs_adjusted[0, DRUM_NOTES['hihat_closed']] = 0.4
                
                # Occasional open hi-hat or crash for variation
                if beat_position == 0 or (beat_position == 8 and step % 32 == 0):
                    drum_probs_adjusted[0, DRUM_NOTES['crash']] = 0.2
                
                # Add some of the model's learned probabilities (weighted by drum_temperature)
                model_drum_contribution = drum_temperature * 0.1
                for note_idx in DRUM_NOTES.values():
                    if note_idx < self.input_dim:
                        drum_probs_adjusted[0, note_idx] += model_drum_contribution * drum_probs[0, note_idx]
                
                # Renormalize
                drum_probs_sum = drum_probs_adjusted.sum(dim=-1, keepdim=True)
                if drum_probs_sum > 0:
                    drum_probs_adjusted = drum_probs_adjusted / drum_probs_sum
                else:
                    # Fallback: uniform over drum notes
                    for note_idx in DRUM_NOTES.values():
                        if note_idx < self.input_dim:
                            drum_probs_adjusted[0, note_idx] = 1.0 / len(DRUM_NOTES)
                
                drum_note = torch.multinomial(drum_probs_adjusted, num_samples=1)
                
                # === CREATE ONE-HOT VECTORS ===
                melody_one_hot = torch.zeros(batch_size, 1, self.input_dim, device=device)
                bass_one_hot = torch.zeros(batch_size, 1, self.input_dim, device=device)
                drum_one_hot = torch.zeros(batch_size, 1, self.input_dim, device=device)
                
                melody_one_hot[0, 0, melody_note[0]] = 1.0
                bass_one_hot[0, 0, bass_note[0]] = 1.0
                drum_one_hot[0, 0, drum_note[0]] = 1.0
                
                # Store generated notes
                melody_sequence.append(melody_one_hot)
                bass_sequence.append(bass_one_hot)
                drum_sequence.append(drum_one_hot)
                
                # Update input for next iteration (use melody as primary guide)
                x = melody_one_hot
        
        # Concatenate all generated steps
        melody_output = torch.cat(melody_sequence, dim=1)
        bass_output = torch.cat(bass_sequence, dim=1)
        drum_output = torch.cat(drum_sequence, dim=1)
        
        # Create combined representation (sum of all tracks)
        # Note: In practice, you may want to assign to different MIDI channels
        combined = {
            'melody': melody_output,
            'bass': bass_output,
            'drums': drum_output,
            'combined': melody_output + bass_output + drum_output  # Simple mix
        }
        
        return combined