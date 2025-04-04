import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim=128, embed_dim=256, num_heads=8,
                 num_layers=6, dim_feedforward=1024, dropout=0.1):
        super().__init__()

        # Embedding layer to convert note features to embeddings
        self.embedding = nn.Linear(input_dim, embed_dim)

        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # Output projection
        self.output = nn.Linear(embed_dim, input_dim)

    def forward(self, x):
        # x shape: [batch_size, input_dim]

        # Embedding
        x = self.embedding(x)  # [batch_size, embed_dim]

        # Add positional dimension for transformer (treat as sequence length 1)
        x = x.unsqueeze(1)  # [batch_size, 1, embed_dim]

        # Apply transformer
        x = self.transformer_encoder(x)  # [batch_size, 1, embed_dim]

        # Remove positional dimension
        x = x.squeeze(1)  # [batch_size, embed_dim]

        # Project to output space
        output = self.output(x)  # [batch_size, input_dim]

        return output

    def generate(self, seed, steps=100, temperature=1.0):
        """
        Generate a sequence of notes from a seed.

        Args:
            seed: Initial seed tensor [batch_size, input_dim]
            steps: Number of generation steps
            temperature: Controls randomness (lower = more deterministic)

        Returns:
            Generated sequence tensor [batch_size, input_dim]
        """
        with torch.no_grad():
            self.eval()
            batch_size = seed.shape[0]
            device = seed.device

            # Initialize with seed
            current = seed

            for _ in range(steps):
                # Forward pass through model
                output = self.forward(current)

                # Apply temperature
                if temperature != 1.0:
                    output = output / temperature

                # Sample from distribution
                probs = torch.softmax(output, dim=1)
                next_notes = torch.zeros_like(probs)

                # Get indices of top-k values
                _, top_indices = torch.topk(probs, k=5, dim=1)

                # Randomly select from top-k for each item in batch
                for i in range(batch_size):
                    idx = top_indices[i, torch.randint(0, 5, (1,))]
                    next_notes[i, idx] = 1.0

                # Update current with newly generated notes
                current = next_notes

            return current