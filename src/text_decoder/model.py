"""
Build a GPT style model
"""

import torch
import torch.nn as nn


class GPTConfig:
    vocab_size: int = 2048
    embed_dim: int = 512
    hidden_dim: int = 2048
    num_heads: int = 8
    num_layers: int = 12
    ctx_len: int = 1024


def calculate_positional_encodings(positions: torch.Tensor, embed_dim: int):
    """Calculate the 1d sinusoidal positional encodings"""
    encodings = []
    for pos in positions:
        sin_enc = torch.sin(torch.tensor([pos/10000**((2*(i//2))/embed_dim) for i in range(embed_dim)]))
        cos_enc = torch.cos(torch.tensor([pos/10000**((2*(i//2))/embed_dim) for i in range(embed_dim)]))
        enc = torch.where(torch.arange(embed_dim)%2==0, sin_enc, cos_enc)
        encodings.append(enc.unsqueeze(0))
    encodings = torch.concat(encodings, dim=0)
    return encodings


class GPTTransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attention_layer = nn.MultiheadAttention(config.embed_dim, config.num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embed_dim)
        )
        self.layer_norm1 = nn.LayerNorm(config.embed_dim)
        self.layer_norm2 = nn.LayerNorm(config.embed_dim)

    def forward(self, x):
        # x -> B, T, D
        ctx_embed, _ = self.attention_layer(x, x, x)
        ctx_embed = self.layer_norm1(ctx_embed)
        ctx_embed = self.mlp(ctx_embed)
        ctx_embed = self.layer_norm2(ctx_embed)
        return ctx_embed



class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.embedding_layer = nn.Embedding(config.vocab_size, config.embed_dim)
        self.register_buffer("positional_encodings", calculate_positional_encodings(torch.arange(config.ctx_len), config.embed_dim))
        self.transformer_blocks = nn.ModuleList([
            GPTTransformerBlock(config) for _ in range(config.num_layers)
        ])
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size)

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

    """
    Change the activation function from ReLU to something better
    Add dropout where required
    """