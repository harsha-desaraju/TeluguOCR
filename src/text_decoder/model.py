"""
Build a GPT style model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

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
    def __init__(self, embed_dim: int, num_heads: int, dropout: float, is_causal: bool):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.dropout   = dropout
        self.is_causal = is_causal

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        B, T, _ = query.shape
        _, S, _ = key.shape

        queries = self.q_proj(query).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        keys    = self.k_proj(key).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        values  = self.v_proj(value).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        dropout_p = self.dropout if self.training else 0.0
        ctx_embeds = F.scaled_dot_product_attention(
            queries, keys, values,
            dropout_p=dropout_p,
            is_causal=self.is_causal
        )

        ctx_embeds = ctx_embeds.transpose(1, 2).reshape(B, T, self.embed_dim)
        return self.out_proj(ctx_embeds)


class GPTTransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attention_layer = MultiHeadAttention(config.embed_dim, config.num_heads, config.dropout, is_causal=True)
        self.mlp = nn.Sequential(
            SwiGLU(config.embed_dim, config.hidden_dim),
            nn.Dropout(config.dropout)
        )
        self.layer_norm1 = nn.LayerNorm(config.embed_dim)
        self.layer_norm2 = nn.LayerNorm(config.embed_dim)

    def forward(self, x):
        # x -> B, T, D
        normed = self.layer_norm1(x)
        x = x + self.attention_layer(normed, normed, normed)
        x = x + self.mlp(self.layer_norm2(x))
        return x



class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.embedding_layer = nn.Embedding(config.vocab_size, config.embed_dim)
        self.register_buffer("positional_encodings", calculate_positional_encodings(torch.arange(config.ctx_len), config.embed_dim))
        self.transformer_blocks = nn.ModuleList([
            GPTTransformerBlock(config) for _ in range(config.num_layers)
        ])
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size)
        self.lm_head.weight = self.embedding_layer.weight

    def forward(self, x):
        # x -> B, T, D
        embeds = self.embedding_layer(x)

        num_tokens = x.shape[1]
        embeds = embeds + self.positional_encodings[:num_tokens, :].unsqueeze(0)

        for block in self.transformer_blocks:
            embeds = block(embeds)

        logits = self.lm_head(embeds)

        return logits



if __name__ == '__main__':

    model_config = GPTConfig(
        vocab_size=2048,
        embed_dim=512,
        hidden_dim=1368,  # 2.67 * 512 = 2/3 * 4 * hidden_dim
        num_heads=8,
        num_layers=24,
        ctx_len=1024,
        dropout=0.1
    )

    model = GPTModel(model_config)

    inp = torch.randint(0, 2048, (4, 10))

    out = model(inp)
    print(out.shape)

    params = 0
    for layer in model.parameters():
        params += layer.numel()

    print(f"No. of parameters in the model is: {params}")