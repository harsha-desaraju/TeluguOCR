"""
Build a GPT style model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers.modeling_outputs import CausalLMOutput

@dataclass
class GPTConfig:
    vocab_size: int = 2048
    embed_dim: int = 512
    hidden_dim: int = 2048
    num_heads: int = 8
    num_layers: int = 12
    ctx_len: int = 1024
    dropout: float = 0.1


def calculate_positional_encodings(positions: torch.Tensor, embed_dim: int):
    i = torch.arange(embed_dim // 2, dtype=torch.float32)
    div_term = 10000 ** (2 * i / embed_dim)          # (D/2,)
    pos = positions.float().unsqueeze(1)              # (T, 1)
    args = pos / div_term                             # (T, D/2)
    enc = torch.zeros(len(positions), embed_dim)
    enc[:, 0::2] = torch.sin(args)
    enc[:, 1::2] = torch.cos(args)
    return enc


class SwiGLU(nn.Module):
    """Implement the SwiGLU activation function"""
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, embed_dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class MultiHeadAttention(nn.Module):
    """Implement multi head attention"""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.dropout   = dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask=None):
        # attn_mask: (B, 1, T, T) boolean — True means KEEP, False means MASK OUT
        B, T, _ = query.shape
        _, S, _ = key.shape

        queries = self.q_proj(query).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        keys    = self.k_proj(key).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        values  = self.v_proj(value).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        dropout_p = self.dropout if self.training else 0.0
        ctx_embeds = F.scaled_dot_product_attention(
            queries, keys, values,
            dropout_p=dropout_p,
            attn_mask=attn_mask,
            is_causal=False
        )

        ctx_embeds = ctx_embeds.transpose(1, 2).reshape(B, T, self.embed_dim)
        return self.out_proj(ctx_embeds)


class GPTTransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attention_layer = MultiHeadAttention(config.embed_dim, config.num_heads, config.dropout)
        self.mlp = nn.Sequential(
            SwiGLU(config.embed_dim, config.hidden_dim),
            nn.Dropout(config.dropout)
        )
        self.layer_norm1 = nn.LayerNorm(config.embed_dim)
        self.layer_norm2 = nn.LayerNorm(config.embed_dim)

    def forward(self, x: torch.Tensor, attn_mask = None):
        # x -> B, T, D
        normed = self.layer_norm1(x)
        x = x + self.attention_layer(normed, normed, normed, attn_mask=attn_mask)
        x = x + self.mlp(self.layer_norm2(x))
        return x



class GPTModel(nn.Module):
    _keys_to_ignore_on_save = None
    def __init__(self, config: GPTConfig, pad_index: int):
        super().__init__()
        self.pad_index = pad_index
        self.embedding_layer = nn.Embedding(config.vocab_size, config.embed_dim)
        self.register_buffer("positional_encodings", calculate_positional_encodings(torch.arange(config.ctx_len), config.embed_dim))
        self.transformer_blocks = nn.ModuleList([
            GPTTransformerBlock(config) for _ in range(config.num_layers)
        ])
        self.layer_norm = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear( config.embed_dim, config.vocab_size, bias=False)

    def _build_attn_mask(self, input_ids, attention_mask):
        """
        Builds a combined boolean causal + padding mask.
        SDPA expects: True = attend, False = ignore.
        Shape: (B, 1, T, T)
        """
        B, T = input_ids.shape
        device = input_ids.device
        # Causal mask: upper triangle is False (masked), lower triangle True
        causal = torch.ones(T, T, dtype=torch.bool, device=device).tril()  # (T, T)
        if attention_mask is not None:
            # attention_mask: (B, T), 1=real token, 0=pad
            # Expand to (B, 1, 1, T) so it broadcasts over query positions
            pad_mask = attention_mask.bool().unsqueeze(1).unsqueeze(2)      # (B, 1, 1, T)
            combined = causal.unsqueeze(0).unsqueeze(0) & pad_mask          # (B, 1, T, T)
        else:
            combined = causal.unsqueeze(0).unsqueeze(0)                     # (1, 1, T, T)
        return combined

    def forward(self, input_ids, attention_mask=None, labels = None):
        # input_ids -> (B, T)
        embeds = self.embedding_layer(input_ids)
        T = input_ids.shape[1]
        embeds = embeds + self.positional_encodings[:T].unsqueeze(0)
        attn_mask = self._build_attn_mask(input_ids, attention_mask)
        for block in self.transformer_blocks:
            embeds = block(embeds, attn_mask=attn_mask)
        embeds = self.layer_norm(embeds)
        logits = self.lm_head(embeds)
        loss = None

        # Always calculate the loss
        # shift for causal LM
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.pad_index
        )
        return CausalLMOutput(
            loss=loss,
            logits=logits
        )



if __name__ == '__main__':

    model_config = GPTConfig(
        vocab_size=2048,
        embed_dim=512,
        hidden_dim=1368,  # 2.67 * 512 = 2/3 * 4 * hidden_dim
        num_heads=8,
        num_layers=16,
        ctx_len=256,
        dropout=0.1
    )

    model = GPTModel(model_config, pad_index=3)

    inp = torch.randint(0, 2048, (4, 10))

    out = model(inp)
    # print(out.shape)

    params = 0
    for layer in model.parameters():
        params += layer.numel()

    print(f"No. of parameters in the model is: {params}")