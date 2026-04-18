
"""
Encoder - Decoder Architecture

Encoder:
    Embedding Layer
    Positional Encoding layer
    Transformer Block
        Attention Block
            Multi-Head Attention
                Self Attention Block
                Feed Forward Block
        Feed Forward Block

Decoder:
    Embedding Layer
    Positional Encoding Layer
    Transformer Block
        Attention Block
            Multi-Head Attention
                Causal Self Attention
                Feed Forward Block
            Multi-Head Attention
                Cross-Attention
                Feed Forward Block
        Feed Forward Block
    Feed Forward Block
    Softmax

Base Components:
    1) Attention Block (Bi & Causal)
    2) Multi-Head Attention Block
    3) Transformer Block
    4) Positional Encoding Block

Existing Components:
    1) Embedding Block
    2) Feed Forward Block

** Try both native python implementation and using pytorch components ** - Compare speed
"""

import math
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """Implements self-attention"""
    def __init__(self, embed_dim: int, proj_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, proj_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, proj_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, proj_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.proj_dim = proj_dim

    def forward(self, x):
        # x - (B, T, D)
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        att_weights = (queries @ torch.transpose(keys, 1, 2)/math.sqrt(self.proj_dim))
        scaled_att_weights = self.softmax(att_weights)
        embeds = scaled_att_weights @ values
        return embeds



class MultiHeadAttention(nn.Module):
    """Implements multi-head attention"""
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()

        assert embed_dim % num_heads == 0, f"embed_dim should be divisible by num_heads"

        self.q_proj = nn.Parameter(
            nn.init.kaiming_uniform_(torch.randn((num_heads, embed_dim, embed_dim // num_heads)), a=math.sqrt(5)), requires_grad=True)
        self.k_proj = nn.Parameter(
            nn.init.kaiming_uniform_(torch.randn((num_heads, embed_dim, embed_dim // num_heads)), a=math.sqrt(5)), requires_grad=True)
        self.v_proj = nn.Parameter(
            nn.init.kaiming_uniform_(torch.randn((num_heads, embed_dim, embed_dim // num_heads)), a=math.sqrt(5)), requires_grad=True)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.proj_dim = embed_dim // num_heads

    def forward(self, x: torch.Tensor):
        # x - (B, T, D)
        # D - embed_dim
        # p - proj_dim (embed_dim // num_heads)

        x = x.unsqueeze(1)     # (B, 1, T, D)
        queries = x @ self.q_proj       # (B, num_heads, T, p)
        keys = x @ self.k_proj          # (B, num_heads, T, p)
        values = x @ self.v_proj        # (B, num_heads, T, p)

        att_weights = (queries @ torch.transpose(keys, 2, 3)/math.sqrt(self.proj_dim))                 # (B, num_heads, T, T)
        scaled_att_weights = self.softmax(att_weights)        # (B, num_heads, T, T)
        embeds = scaled_att_weights @ values                                           # (B, num_heads, T, p)
        B, num_heads, T, p = embeds.shape
        embeds = embeds.permute(0, 2, 1, 3)
        embeds = embeds.reshape(B, T, num_heads*p)
        embeds = self.out_proj(embeds)
        return embeds



class MultiHeadCrossAttention(nn.Module):
    """Implements multi-head attention"""
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()

        assert embed_dim % num_heads == 0, f"embed_dim should be divisible by num_heads"

        self.q_proj = nn.Parameter(
            nn.init.kaiming_uniform_(torch.randn((num_heads, embed_dim, embed_dim // num_heads)), a=math.sqrt(5)),
            requires_grad=True)
        self.k_proj = nn.Parameter(
            nn.init.kaiming_uniform_(torch.randn((num_heads, embed_dim, embed_dim // num_heads)), a=math.sqrt(5)),
            requires_grad=True)
        self.v_proj = nn.Parameter(
            nn.init.kaiming_uniform_(torch.randn((num_heads, embed_dim, embed_dim // num_heads)), a=math.sqrt(5)),
            requires_grad=True)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.proj_dim = embed_dim // num_heads

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor):
        # x - (B, T, D)
        # D - embed_dim
        # p - proj_dim (embed_dim // num_heads)

        x = x.unsqueeze(1)     # (B, 1, T, D)
        encoder_output = encoder_output.unsqueeze(1)
        queries = x @ self.q_proj       # (B, num_heads, T, p)
        keys = encoder_output @ self.k_proj          # (B, num_heads, T, p)
        values = encoder_output @ self.v_proj        # (B, num_heads, T, p)

        att_weights = (queries @ torch.transpose(keys, 2, 3)/math.sqrt(self.proj_dim))                 # (B, num_heads, T, T)
        scaled_att_weights = self.softmax(att_weights)                                                             # (B, num_heads, T, T)
        embeds = scaled_att_weights @ values                                                                       # (B, num_heads, T, p)
        B, num_heads, T, p = embeds.shape
        embeds = embeds.permute(0, 2, 1, 3)
        embeds = embeds.reshape(B, T, num_heads*p)
        embeds = self.out_proj(embeds)
        return embeds




class CausalSelfAttention(SelfAttention):
    def __init__(self, embed_dim: int, proj_dim: int):
        super().__init__(embed_dim, proj_dim)

    def forward(self, x):
        # x - (B, T, D)

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        att_weights = queries @ torch.transpose(keys, 1, 2)
        causal_mask = torch.triu(torch.full((att_weights.shape[-1], att_weights.shape[-1]), fill_value=float('-inf'), device=att_weights.device), diagonal=1)
        att_weights = (att_weights + causal_mask)/math.sqrt(self.proj_dim)
        scaled_att_weights = self.softmax(att_weights)
        embeds = scaled_att_weights @ values
        return embeds


class CausalMultiHeadAttention(MultiHeadAttention):
    """Implements Causal Multi Head Attention"""
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__(embed_dim, num_heads)

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)     # B, 1, T, D

        queries = x @ self.q_proj
        keys = x @ self.k_proj
        values = x @ self.v_proj

        att_weights = queries @ torch.transpose(keys, 2, 3)

        causal_mask = torch.triu(torch.full((att_weights.shape[-1], att_weights.shape[-1]), fill_value=float('-inf'), device=att_weights.device), diagonal=1)
        att_weights = att_weights + causal_mask.unsqueeze(0).unsqueeze(0)
        att_weights = att_weights/math.sqrt(self.proj_dim)
        scaled_att_weights = self.softmax(att_weights)
        embeds = scaled_att_weights @ values
        B, num_heads, T, p = embeds.shape
        embeds = embeds.permute(0, 2, 1, 3)
        embeds = embeds.reshape(B, T, num_heads * p)
        embeds = self.out_proj(embeds)
        return embeds





class EncoderTransformerBlock(nn.Module):
    """Implements a transformer block"""
    def __init__(self, embed_dim: int, num_heads: int, ffn_hidden_dim: int):
        super().__init__()
        self.mha_block = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Linear(ffn_hidden_dim, embed_dim)
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        att_out = self.mha_block(x)
        att_out = self.layer_norm1(x + att_out)
        cont_embed = self.ffn(att_out)
        cont_embed = self.layer_norm2(att_out + cont_embed)
        return cont_embed


class DecoderTransformerBlock(nn.Module):
    """Implements decoder transformer block"""
    def __init__(self, embed_dim: int, num_heads: int, ffn_hidden_dim: int):
        super().__init__()
        self.self_mha_block = CausalMultiHeadAttention(embed_dim, num_heads)
        self.cross_mha_block = MultiHeadCrossAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Linear(ffn_hidden_dim, embed_dim)
        )

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.layer_norm3 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor):

        self_att_out = self.self_mha_block(x)
        self_att_out = self.layer_norm1(x + self_att_out)

        cross_att_out = self.cross_mha_block(self_att_out, encoder_output)
        cross_att_out = self.layer_norm2(self_att_out + cross_att_out)

        cont_embed = self.ffn(cross_att_out)
        cont_embed = self.layer_norm3(cross_att_out + cont_embed)
        return cont_embed






class Encoder(nn.Module):
    """Encoder block of the transformer"""
    def __init__(self, vocab_size: int, embed_dim: int, ctx_len: int, num_blocks: int, num_heads: int, ffn_hidden_dim: int):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(
            nn.init.kaiming_uniform_(torch.randn((ctx_len, embed_dim)), a=math.sqrt(5)), requires_grad=True
        )
        self.encoder_transformer_blocks = nn.Sequential(*[
            EncoderTransformerBlock(embed_dim, num_heads, ffn_hidden_dim)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        embeds = self.embedding_layer(x)
        embeds = embeds + self.positional_encoding[:embeds.shape[-2]]
        cont_embeds = self.encoder_transformer_blocks(embeds)
        return cont_embeds



class Decoder(nn.Module):
    """Decoder block of a transformer"""

    def __init__(self, vocab_size: int, embed_dim: int, ctx_len: int, num_blocks: int, num_heads: int,ffn_hidden_dim: int):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(
            nn.init.kaiming_uniform_(torch.randn((ctx_len, embed_dim)), a=math.sqrt(5)), requires_grad=True
        )
        self.decoder_transformer_blocks = nn.ModuleList([
            DecoderTransformerBlock(embed_dim, num_heads, ffn_hidden_dim)
            for _ in range(num_blocks)
        ])

        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, encoder_output):
        embeds = self.embedding_layer(x)
        embeds = embeds + self.positional_encoding[:embeds.shape[-2]]
        for module in self.decoder_transformer_blocks:
            embeds = module(embeds, encoder_output)
        logits = self.linear(embeds)
        return logits




if __name__ == '__main__':

    from src.model_training.transformer_models.benchmarking import benchmark_model

    DEVICE = "mps"

    vocab_size = 16384
    embed_dim = 512
    ctx_len = 256
    num_encoder_blocks = 12
    num_decoder_blocks = 12
    num_heads = 8
    ffn_hidden_size = 4 * embed_dim

    batch_size, num_tokens = 16, ctx_len

    encoder = Encoder(vocab_size, embed_dim, ctx_len, num_encoder_blocks, num_heads, ffn_hidden_size)
    decoder = Decoder(vocab_size, embed_dim, ctx_len, num_decoder_blocks, num_heads, ffn_hidden_size)

    print(f"Device: {DEVICE}")

    num_params = 0
    for layer in encoder.parameters():
        num_params += layer.numel()
    print(f"No. of parameters in Encoder model: {num_params}")

    num_params = 0
    for layer in decoder.parameters():
        num_params += layer.numel()
    print(f"No. of parameters in Decoder model: {num_params}")

    print("\nEncoder Benchmark:")
    benchmark_model(encoder, batch_size, seq_len=num_tokens, embed_dim=embed_dim, vocab_size=vocab_size)

    print("\nDecoder Benchmark:")
    benchmark_model(decoder, batch_size, seq_len=num_tokens, embed_dim=embed_dim, vocab_size=vocab_size)