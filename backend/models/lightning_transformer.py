import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math

class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d_model, max_seq_len=5000):
        super().__init__()
        self.d_model = d_model
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class EnhancedMultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_head = d_model // num_heads
        self.num_heads = num_heads
        self.scale = self.d_head ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbeddings(self.d_head)

    def forward(self, x, mask=None, key_value=None):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        
        if key_value is not None:
            k_src = key_value
            v_src = key_value
            L_kv = key_value.shape[1]
        else:
            k_src = x
            v_src = x
            L_kv = L
            
        k = self.k_proj(k_src).view(B, L_kv, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(v_src).view(B, L_kv, self.num_heads, self.d_head).transpose(1, 2)
        
        cos_q, sin_q = self.rope(x, seq_len=L)
        cos_k, sin_k = self.rope(k_src, seq_len=L_kv)
        
        cos_q = cos_q.unsqueeze(0).unsqueeze(0)
        sin_q = sin_q.unsqueeze(0).unsqueeze(0)
        cos_k = cos_k.unsqueeze(0).unsqueeze(0)
        sin_k = sin_k.unsqueeze(0).unsqueeze(0)

        q = (q * cos_q) + (rotate_half(q) * sin_q)
        k = (k * cos_k) + (rotate_half(k) * sin_k)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            if mask.dim() == 2:
                attn = attn.masked_fill(mask == float('-inf'), float('-inf'))
            elif mask.dim() == 3:
                 attn = attn.masked_fill(mask.unsqueeze(1) == float('-inf'), float('-inf'))
            
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)

class GatedFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.wg = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        residual = x
        hidden = F.silu(self.w1(x)) * torch.sigmoid(self.wg(x))
        output = self.w2(self.dropout(hidden))
        return self.norm(residual + output)

class EnhancedTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = EnhancedMultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.gated_ffn = GatedFeedForward(d_model, dim_feedforward, dropout)

    def forward(self, src, src_mask=None):
        src2 = self.self_attn(src, mask=src_mask)
        src = self.norm1(src + self.dropout1(src2))
        src = self.gated_ffn(src)
        return src

class MultiTrackAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attention = EnhancedMultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query_track, key_value_track):
        attn_output = self.cross_attention(query_track, key_value=key_value_track)
        output = self.norm(query_track + self.dropout(attn_output))
        return output

class LightningTransformer(pl.LightningModule):
    def __init__(self, 
                 input_dim: int = 128,
                 embed_dim: int = 256,
                 num_heads: int = 4,
                 num_layers: int = 4,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 learning_rate: float = 1e-4,
                 enable_multitrack: bool = False,
                 enable_conditioning: bool = True):
        super().__init__()
        self.save_hyperparameters()
        
        self.enable_multitrack = enable_multitrack
        self.enable_conditioning = enable_conditioning
        self.input_dim = input_dim
        
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        if enable_conditioning:
            self.chord_embedding = nn.Linear(12 + 7 + 12, embed_dim // 4)
            self.tempo_embedding = nn.Linear(1, embed_dim // 4)
            self.conditioning_projection = nn.Linear(embed_dim // 2, embed_dim)

        self.layers = nn.ModuleList([
            EnhancedTransformerEncoderLayer(embed_dim, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        if enable_multitrack:
            self.bass_layers = nn.ModuleList([
                EnhancedTransformerEncoderLayer(embed_dim, num_heads, dim_feedforward, dropout)
                for _ in range(max(2, num_layers // 2))
            ])
            
            self.drum_layers = nn.ModuleList([
                EnhancedTransformerEncoderLayer(embed_dim, num_heads, dim_feedforward, dropout)
                for _ in range(max(2, num_layers // 2))
            ])
            
            self.bass_melody_attention = MultiTrackAttention(embed_dim, num_heads, dropout)
            self.drum_melody_attention = MultiTrackAttention(embed_dim, num_heads, dropout)
            self.drum_bass_attention = MultiTrackAttention(embed_dim, num_heads, dropout)
            
            self.melody_output = nn.Linear(embed_dim, input_dim)
            self.bass_output = nn.Linear(embed_dim, input_dim)
            self.drum_output = nn.Linear(embed_dim, input_dim)
            
            self.melody_type_emb = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            self.bass_type_emb = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            self.drum_type_emb = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        else:
            self.output_projection = nn.Linear(embed_dim, input_dim)
        
        self.learning_rate = learning_rate
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _get_conditioning_embedding(self, batch_size, seq_len, chord_cond, tempo_cond, device):
        conditioning_parts = []
        
        if chord_cond is not None:
            chord_emb = self.chord_embedding(chord_cond)
            conditioning_parts.append(chord_emb)
        else:
            chord_emb = torch.zeros(batch_size, seq_len, self.chord_embedding.out_features, device=device)
            conditioning_parts.append(chord_emb)
            
        if tempo_cond is not None:
            tempo_emb = self.tempo_embedding(tempo_cond)
            conditioning_parts.append(tempo_emb)
        else:
            tempo_emb = torch.zeros(batch_size, seq_len, self.tempo_embedding.out_features, device=device)
            conditioning_parts.append(tempo_emb)
        
        combined_conditioning = torch.cat(conditioning_parts, dim=-1)
        conditioning_emb = self.conditioning_projection(combined_conditioning)
        return conditioning_emb

    def forward(self, x, mask=None, chord_cond=None, tempo_cond=None):
        if not self.enable_multitrack:
            x_emb = self.embedding(x)
            
            if self.enable_conditioning and (chord_cond is not None or tempo_cond is not None):
                conditioning_emb = self._get_conditioning_embedding(x_emb.size(0), x_emb.size(1), 
                                                                  chord_cond, tempo_cond, x.device)
                x_emb = x_emb + conditioning_emb

            for layer in self.layers:
                x_emb = layer(x_emb, src_mask=mask)
            return self.output_projection(x_emb)
        else:
            batch_size, seq_len, _ = x.shape
            x_emb = self.embedding(x)
            
            if self.enable_conditioning and (chord_cond is not None or tempo_cond is not None):
                conditioning_emb = self._get_conditioning_embedding(batch_size, seq_len, 
                                                                  chord_cond, tempo_cond, x.device)
                x_emb = x_emb + conditioning_emb
            
            melody_emb = x_emb + self.melody_type_emb
            bass_emb = x_emb + self.bass_type_emb
            drum_emb = x_emb + self.drum_type_emb
            
            melody_hidden = melody_emb
            for layer in self.layers:
                melody_hidden = layer(melody_hidden, src_mask=mask)
                
            bass_hidden = bass_emb
            for layer in self.bass_layers:
                bass_hidden = layer(bass_hidden, src_mask=mask)
            bass_hidden = self.bass_melody_attention(bass_hidden, melody_hidden)
            
            drum_hidden = drum_emb
            for layer in self.drum_layers:
                drum_hidden = layer(drum_hidden, src_mask=mask)
            drum_hidden = self.drum_melody_attention(drum_hidden, melody_hidden)
            drum_hidden = self.drum_bass_attention(drum_hidden, bass_hidden)
            
            melody_out = self.melody_output(melody_hidden)
            bass_out = self.bass_output(bass_hidden)
            drum_out = self.drum_output(drum_hidden)
            
            return {
                'melody': melody_out,
                'bass': bass_out,
                'drums': drum_out
            }

    def training_step(self, batch, batch_idx):
        if isinstance(batch, list) or isinstance(batch, tuple):
            x = batch[0]
        else:
            x = batch
            
        inputs = x[:, :-1, :]
        targets = x[:, 1:, :]
        
        seq_len = inputs.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(self.device)
        
        if self.enable_multitrack:
            outputs = self(inputs, mask=mask)
            loss_melody = F.mse_loss(outputs['melody'], targets)
            loss_bass = F.mse_loss(outputs['bass'], targets)
            loss_drums = F.mse_loss(outputs['drums'], targets)
            loss = loss_melody + loss_bass + loss_drums
        else:
            outputs = self(inputs, mask=mask)
            loss = F.mse_loss(outputs, targets)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, list) or isinstance(batch, tuple):
            x = batch[0]
        else:
            x = batch
        
        inputs = x[:, :-1, :]
        targets = x[:, 1:, :]
        
        seq_len = inputs.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(self.device)
        
        if self.enable_multitrack:
            outputs = self(inputs, mask=mask)
            loss = F.mse_loss(outputs['melody'], targets)
        else:
            outputs = self(inputs, mask=mask)
            loss = F.mse_loss(outputs, targets)
        
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]
