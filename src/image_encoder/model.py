"""
Vision Transformer based Image Encoder model
"""

import torch
import torch.nn as nn
from utils import ImagePreprocessor
from PIL import Image
from dataclasses import dataclass
from utils import random_masking, patchify, get_2d_sinusoidal_encoding

@dataclass
class ViTConfig:
    embed_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.2
    hidden_layer_size: int = 1024
    num_blocks: int = 8
    patch_size: int = 16
    image_height: int = 64
    max_image_width:int = 1024




class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, hidden_layer_size: int, dropout: float):
        super().__init__()
        self.multi_head_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_layer_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_layer_size, embed_dim),
            nn.Dropout(dropout)
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, key_padding_mask = None):
        # query, key, value - B, T, D
        norm_x = self.layer_norm1(x)
        ctx_embed, _ = self.multi_head_attention(
            norm_x, norm_x, norm_x,
            key_padding_mask=key_padding_mask,
            need_weights=False)
        ctx_embed = ctx_embed + x

        norm_ctx_embed = self.layer_norm2(ctx_embed)
        norm_ctx_embed = self.mlp(norm_ctx_embed)
        norm_ctx_embed = norm_ctx_embed + ctx_embed
        return norm_ctx_embed



class ViTEncoder(nn.Module):
    def __init__(self, vit_config: ViTConfig):
        super().__init__()
        self.image_embedding = nn.Conv2d(in_channels=1, out_channels=vit_config.embed_dim, kernel_size=vit_config.patch_size, stride=vit_config.patch_size)

        enc = get_2d_sinusoidal_encoding(
            vit_config.image_height // vit_config.patch_size,
            vit_config.max_image_width // vit_config.patch_size,
            vit_config.embed_dim
        )
        self.register_buffer("positional_encoding", enc)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(vit_config.embed_dim, vit_config.num_heads, vit_config.hidden_layer_size, vit_config.dropout)
              for _ in range(vit_config.num_blocks)]
        )
        self.layer_norm = nn.LayerNorm(vit_config.embed_dim)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor, mask_ratio: float | None = None):
        # x -> B, C, H, W
        embeds = self.image_embedding(x)
        embeds = embeds.flatten(2).transpose(1, 2)
        pos_encodings = self.positional_encoding[:embeds.shape[1]]
        embeds = embeds + pos_encodings

        if mask_ratio is not None:
            embeds, mask, restore_ids = random_masking(embeds, mask_ratio)
        else:
            mask, restore_ids = None, None

        if restore_ids is not None:
            visible_padding_mask = torch.gather(
                padding_mask, dim=1,
                index=restore_ids[:, :embeds.shape[1]]
            )
        else:
            visible_padding_mask = padding_mask

        visible_padding_mask = visible_padding_mask.bool()

        ctx_embeds = embeds
        for block in self.transformer_blocks:
            ctx_embeds = block(ctx_embeds, visible_padding_mask)

        ctx_embeds = self.layer_norm(ctx_embeds)
        return ctx_embeds, mask, restore_ids




class ViTDecoder(nn.Module):
    def __init__(self, config: ViTConfig, encoder_dim: int):
        super().__init__()
        self.embedding_layer = nn.Linear(encoder_dim, config.embed_dim)

        enc = get_2d_sinusoidal_encoding(
            config.image_height // config.patch_size,
            config.max_image_width // config.patch_size,
            config.embed_dim
        )
        self.register_buffer("positional_encoding", enc)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.num_heads, config.hidden_layer_size, config.dropout)
            for _ in range(config.num_blocks)
        ])
        self.decoder_norm = nn.LayerNorm(config.embed_dim)

        self.output_projection = nn.Linear(config.embed_dim, config.patch_size ** 2)


    def forward(self, latent: torch.Tensor, restore_ids: torch.Tensor, padding_mask: torch.Tensor):
        x = self.embedding_layer(latent)

        B, L, D = x.shape
        N = restore_ids.shape[1]

        mask_tokens = self.mask_token.repeat(B, N-L, 1)

        _x = torch.cat([x, mask_tokens], dim=1)
        _x = torch.gather(
            _x, dim=1,
            index=restore_ids.unsqueeze(-1).repeat(1, 1, D)
        )
        _x = _x + self.positional_encoding[:N]
        for block in self.transformer_blocks:
            _x = block(_x, padding_mask.bool())
        _x = self.decoder_norm(_x)
        proj = self.output_projection(_x)
        return proj




class MaskedAutoEncoder(nn.Module):
    def __init__(self, encoder_config: ViTConfig, decoder_config: ViTConfig, mask_ratio: float):
        super().__init__()

        self.mask_ratio = mask_ratio
        self.patch_size = encoder_config.patch_size
        self.encoder_model = ViTEncoder(encoder_config)
        self.decoder_model = ViTDecoder(decoder_config, encoder_config.embed_dim)


    def forward(self, images: torch.Tensor, padding_mask: torch.Tensor):
        latent, mask, restore_ids = self.encoder_model(images, padding_mask,  self.mask_ratio)

        pred = self.decoder_model(latent, restore_ids, padding_mask)

        target = patchify(images, self.patch_size)

        loss = ((target - pred)**2).mean(dim=-1)

        valid_patch_mask = (~padding_mask.bool()).float()
        effective_mask = mask * valid_patch_mask
        loss = (loss * effective_mask).sum() / effective_mask.sum()

        return {"loss": loss, "logits": pred}







if __name__ == '__main__':
    from pathlib import Path

    img_path = Path.cwd().parents[1] / 'data/images/sanatanadharm/5 కర్మసన్యాసయోగము/1_10.jpeg'

    # See whether your code is working

    img = Image.open(img_path)
    image_transformer = ImagePreprocessor(image_height=64, max_image_width=1024, patch_size=16)
    transformed_img = image_transformer(img)
    print(img.size, '->', transformed_img.shape)

    transformed_img = torch.concat([transformed_img.unsqueeze(0), transformed_img.unsqueeze(0)], dim=0)
    # transformed_img = transformed_img.unsqueeze(1)
    print(transformed_img.shape)

    auto_encoder = MaskedAutoEncoder(
        encoder_config=ViTConfig(),
        decoder_config=ViTConfig(),
        mask_ratio=0.75
    )
    # print(auto_encoder)

    pad_mask = torch.zeros((2, 256))

    output = auto_encoder(transformed_img, pad_mask)
    print(output)


