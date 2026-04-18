
"""
Implementation of transformer encoder decoder architecture using inbuilt pytorch functions
"""

import torch
import torch.nn as nn


class EncoderTransformerBlock(nn.Module):
    """Implement transformer block"""
    def __init__(self, embed_dim: int, num_heads: int, hidden_layer_size: int, dropout: float = 0.0):
        super().__init__()
        self.attention_block = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, embed_dim)
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query_in: torch.Tensor):
        """
        Forward pass through the transformer block
        Args:
            query_in: Input to the Query projection matrix
        Returns:
            Context embeddings of the input vectors after the forward pass through the transformer block
        """

        att_out, _ = self.attention_block(query_in, query_in, query_in, need_weights=False)
        att_out = self.layer_norm1(att_out + query_in)

        mlp_out = self.mlp(att_out)
        mlp_out = self.layer_norm2(mlp_out + att_out)
        return mlp_out



class DecoderTransformerBlock(nn.Module):
    """Implement transformer block"""
    def __init__(self, embed_dim: int, num_heads: int, ctx_len: int, hidden_layer_size: int, encoder_embed_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        self.self_attention_block = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        if encoder_embed_dim:
            self.cross_attention_block = nn.MultiheadAttention(embed_dim, num_heads, kdim=encoder_embed_dim, vdim=encoder_embed_dim, dropout=dropout, batch_first=True)
        else:
            self.cross_attention_block = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, embed_dim)
        )

        causal_att_mask = torch.triu(torch.ones(ctx_len, ctx_len), diagonal=1)
        causal_att_mask = causal_att_mask.masked_fill(causal_att_mask == 1, float('-inf'))
        self.register_buffer("causal_att_mask", causal_att_mask)

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.layer_norm3 = nn.LayerNorm(embed_dim)

    def forward(self, query_in: torch.Tensor, encoder_output: torch.Tensor):
        """
        Forward pass through the transformer block
        Args:
            query_in: Input to the Query projection matrix
            encoder_output: Output of the encoder for cross attention
        Returns:
            Context embeddings of the input vectors after the forward pass through the transformer block
        """
        B, T, D = query_in.shape
        att_mask = self.causal_att_mask[:T, :T]

        att_out, _ = self.self_attention_block(query_in, query_in, query_in, attn_mask=att_mask, need_weights=False, is_causal=True)
        att_out = self.layer_norm1(att_out + query_in)

        cross_att_out, _ = self.cross_attention_block(att_out, encoder_output, encoder_output, need_weights=False)
        cross_att_out = self.layer_norm2(cross_att_out + att_out)

        mlp_out = self.mlp(cross_att_out)
        mlp_out = self.layer_norm3(mlp_out + cross_att_out)
        return mlp_out


class Encoder(nn.Module):
    """Implements Encoder module of transformers"""
    def __init__(self, vocab_size: int, embed_dim: int, ctx_len: int, num_blocks: int, num_heads: int, hidden_layer_size: int | None = None, dropout: float = 0.0):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Embedding(ctx_len, embed_dim)

        self.register_buffer("positional_indices", torch.arange(ctx_len))
        self.ctx_len = ctx_len

        hidden_layer_size = hidden_layer_size if hidden_layer_size else 4*embed_dim

        self.encoder_transformer_blocks = nn.Sequential(*[
            EncoderTransformerBlock(embed_dim, num_heads, hidden_layer_size, dropout=dropout)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        """Forward pass through the encoder"""
        B, T = x.shape
        T = min(T, self.ctx_len)
        embeddings = self.embedding_layer(x)
        # truncate the number fo tokens to ctx len
        embeddings = embeddings[:, -T:, :]
        pos_encodings = self.positional_encoding(self.positional_indices[:T].expand(B, -1))
        embeddings = embeddings + pos_encodings
        embeddings = self.encoder_transformer_blocks(embeddings)
        return embeddings



class Decoder(nn.Module):
    """Implements Decoder module of transformers"""
    def __init__(self, vocab_size: int, embed_dim: int, ctx_len: int, num_blocks: int, num_heads: int, hidden_layer_size: int | None = None, encoder_embed_dim: int | None = None, dropout: float = 0.0):
        # Use encoder_embed_dim in case the embedding dimension of encoder and decoder are different, else keep it None
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Embedding(ctx_len, embed_dim)

        self.register_buffer("positional_indices", torch.arange(ctx_len))
        self.ctx_len = ctx_len

        hidden_layer_size = hidden_layer_size if hidden_layer_size else 4*embed_dim

        self.decoder_transformer_blocks = nn.ModuleList([
            DecoderTransformerBlock(embed_dim, num_heads, ctx_len, hidden_layer_size, encoder_embed_dim, dropout=dropout)
            for _ in range(num_blocks)
        ])

        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, encoder_output):
        """Forward pass through the encoder"""
        B, T = x.shape
        T = min(T, self.ctx_len)
        embeddings = self.embedding_layer(x)
        # truncate the number fo tokens to ctx len
        embeddings = embeddings[:, -T:, :]
        pos_encodings = self.positional_encoding(self.positional_indices[:T].expand(B, -1))
        embeddings = embeddings + pos_encodings

        for block in self.decoder_transformer_blocks:
            embeddings = block(embeddings, encoder_output)

        logits = self.linear(embeddings)
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

    """
    Device: mps
    No. of parameters in Encoder model: 46348288
    No. of parameters in Decoder model: 67373056
    
    Encoder Benchmark:
    Avg forward time : 0.14980 sec
    Avg backward time: 0.27210 sec
    Tokens/sec: 27343.53
    
    Decoder Benchmark:
    Avg forward time : 0.26579 sec
    Avg backward time: 0.45597 sec
    Tokens/sec: 15410.87
    """