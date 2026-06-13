"""
Build a GPT style model
"""

import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 2048
    embed_dim: int = 512
    hidden_dim: int = 2048
    num_heads: int = 8
    num_layers: int = 12
    ctx_len: int = 1024


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
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

class GPTTransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attention_layer = nn.MultiheadAttention(config.embed_dim, config.num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, config.hidden_dim),
            SwiGLU(config.embed_dim, config.hidden_dim),
            nn.Linear(config.hidden_dim, config.embed_dim)
        )
        self.layer_norm1 = nn.LayerNorm(config.embed_dim)
        self.layer_norm2 = nn.LayerNorm(config.embed_dim)

        causal_mask = torch.triu(torch.ones(config.ctx_len, config.ctx_len), diagonal=1)
        self.register_buffer("attention_mask", causal_mask)

    def forward(self, x):
        # x -> B, T, D
        T = x.shape[1]
        normed = self.layer_norm1(x)
        attn_mask = self.attention_mask[:T, :T]
        attn_out, _ = self.attention_layer(normed, normed, normed, attn_mask=attn_mask)
        x = x + attn_out
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

    model = GPTModel(GPTConfig())
    print(model)

    inp = torch.randint(0, 2048, (4, 10))

    out = model(inp)
    print(out.shape)

    params = 0
    for layer in model.parameters():
        params += layer.numel()

    print(f"No. of parameters in the model is: {params}")


    """
    Change the activation function from ReLU to something better
    Add dropout where required
    """