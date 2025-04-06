import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    Positional encoding for helping the model understand sequence position
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

class MusicTransformerModel(nn.Module):
    def __init__(self, input_dim=128, embed_dim=512, num_heads=8,
                 num_layers=8, dim_feedforward=2048, dropout=0.1):
        """
        Enhanced Music Transformer model for sequence generation
        
        Args:
            input_dim: Input dimension (typically MIDI notes = 128)
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
        """
        super(MusicTransformerModel, self).__init__()
        
        # Input projection
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim)
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Use batch-first convention [batch, seq, features]
        )
        
        # Full transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(embed_dim, input_dim)
        
        # Initialize memory for long-term memory in generation
        self.memory = None
        self.memory_key = None
        self.memory_length = 0
        
        # For section-based generation
        self.section_memories = {}
        
        # Initialize mask for auto-regressive generation
        self._generate_square_subsequent_mask = self._generate_mask

    def forward(self, x, sections=None, use_memory=False):
        """
        Forward pass through the transformer
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            sections: Optional section IDs for structured generation
            use_memory: Whether to use past memory for generation
            
        Returns:
            Output tensor [batch_size, seq_len, input_dim]
        """
        # Project input to embedding dimension
        x = self.embedding(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # If using memory from previous generations
        if use_memory and self.memory is not None:
            # Concatenate memory with current input along sequence dimension
            x = torch.cat([self.memory, x], dim=1)
        
        # Pass through transformer encoder
        output = self.transformer_encoder(x)
        
        # If we're using memory, store the current output as memory
        if use_memory:
            # Store the last sequence as memory for next generation
            self.memory = output.detach()  # Detach to prevent backprop through memory
            self.memory_length = output.size(1)  # Keep track of memory length
            
        # Project output back to input dimension
        output = self.output_projection(output)
        
        # Apply softmax to get probability distribution over notes
        return F.softmax(output, dim=-1)

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
    
    def generate(self, seed, steps=100, temperature=0.8, top_k=5, top_p=0.92):
        """
        Generate a sequence using the transformer
        
        Args:
            seed: Initial seed tensor [batch_size, seq_len, input_dim]
            steps: Number of steps to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Number of top probabilities to sample from
            top_p: Nucleus sampling probability threshold
            
        Returns:
            Generated sequence
        """
        self.eval()  # Set to evaluation mode
        
        # Initialize with seed
        x = seed
        
        # Reset memory
        self.reset_memory()
        
        # Generate steps
        for i in range(steps):
            # Forward pass with memory usage
            with torch.no_grad():
                output = self.forward(x, use_memory=(i > 0))
            
            # Get the last step's output
            next_token_logits = output[:, -1, :] / temperature
            
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
            x = next_token_one_hot
        
        # Return the full generated sequence
        return output
    
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
        
        # Concatenate all sections or return them separately based on needs
        # For now, we'll concatenate them along the sequence dimension
        full_output = torch.cat(all_outputs, dim=1)
        return full_output
    
    def _generate_section(self, seed, steps=16, temperature=0.8):
        """Helper method to generate a single section"""
        x = seed
        outputs = [x]
        
        # Generate steps
        for i in range(steps):
            # Forward pass with memory usage
            with torch.no_grad():
                output = self.forward(x, use_memory=(i > 0 or self.memory is not None))
            
            # Apply temperature
            logits = output[:, -1, :] / temperature
            
            # Sample from the output distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Create one-hot vector for the next token
            next_token_one_hot = torch.zeros(
                seed.size(0), 1, seed.size(-1), 
                device=seed.device
            )
            
            # Fill the one-hot vector
            for batch_idx in range(seed.size(0)):
                next_token_one_hot[batch_idx, 0, next_token[batch_idx]] = 1.0
            
            # Store this output
            outputs.append(next_token_one_hot)
            
            # Update x for the next iteration
            x = next_token_one_hot
        
        # Concatenate all outputs
        return torch.cat(outputs, dim=1)

# For backward compatibility
TransformerModel = MusicTransformerModel
