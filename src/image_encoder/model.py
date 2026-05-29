"""
Vision Transformer based Image Encoder model
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from dataclasses import dataclass


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
        self.multi_head_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_layer_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_layer_size, embed_dim),
            nn.Dropout(dropout)
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # query, key, value - B, T, D
        norm_x = self.layer_norm1(x)
        ctx_embed, _ = self.multi_head_attention(norm_x, norm_x, norm_x, need_weights=False)
        ctx_embed = ctx_embed + x

        norm_ctx_embed = self.layer_norm2(ctx_embed)
        norm_ctx_embed = self.mlp(norm_ctx_embed)
        norm_ctx_embed = norm_ctx_embed + ctx_embed
        return norm_ctx_embed



class ViTEncoder(nn.Module):
    def __init__(self, vit_config: ViTConfig):
        super().__init__()
        ctx_len = (vit_config.image_height//vit_config.patch_size) * (vit_config.max_image_width//vit_config.patch_size)
        self.image_embedding = nn.Conv2d(in_channels=1, out_channels=vit_config.embed_dim, kernel_size=vit_config.patch_size, stride=vit_config.patch_size)
        self.positional_encoding = nn.Embedding(ctx_len, vit_config.embed_dim)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(vit_config.embed_dim, vit_config.num_heads, vit_config.hidden_layer_size, vit_config.dropout)
              for _ in range(vit_config.num_blocks)]
        )
        self.layer_norm = nn.LayerNorm(vit_config.embed_dim)
        self.register_buffer("positions", torch.arange(ctx_len))

    def forward(self, x: torch.Tensor):
        # x -> B, C, H, W
        embeds = self.image_embedding(x)
        embeds = embeds.flatten(2)
        embeds = embeds.transpose(1, 2)
        pos_encodings = self.positional_encoding(self.positions[:embeds.shape[1]])
        embeds = embeds + pos_encodings
        ctx_embeds = self.transformer_blocks(embeds)
        ctx_embeds = self.layer_norm(ctx_embeds)
        return ctx_embeds


# Creating the patches of the input image
# Masking the tokens of the patched input
# Adding back the masked tokens
# Creating the decoder
# Thinking and changing the Positional encoding function - Change this
# Think of patch_size of 8
# Use high mask ratio
# Think of quantization aware training



class ImagePreprocessor:
    """
    Preprocesses the image before encoding the image
    1) Change the image to gray scale
    2) Resize the image
    3) Pad the image to the nearest multiple of patch size
    4) Normalize the image
    """
    def __init__(self, image_height: int, max_image_width: int, patch_size: int):
        assert image_height % patch_size == 0, "Image height should be a multiple of patch size"
        self.image_height = image_height
        self.max_image_width = max_image_width
        self.patch_size = patch_size
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def _transform(self, img: Image.Image) -> torch.Tensor:
        # Convert to GrayScale
        img = img.convert('L')

        # Calculate the resize target for the image while preserving the aspect ratio
        img_w, img_h = img.size
        scale_factor = self.image_height/img_h
        if scale_factor * img_w > self.max_image_width:
            scale_factor = self.max_image_width/img_w
            target_size = (int(scale_factor * img_h), self.max_image_width)
            diff = self.image_height - target_size[0]
            pad_t, pad_b = diff//2, diff - diff//2
            pad_l, pad_r = 0, 0
        else:
            target_size = (self.image_height, int(scale_factor*img_w))
            # Find the nearest multiple of patch size for padding
            diff = (-target_size[1]) % self.patch_size
            pad_l, pad_r = 0, diff
            pad_t, pad_b = 0, 0

        img = transforms.Resize(target_size)(img)
        img = transforms.Pad((pad_l, pad_t, pad_r, pad_b), fill=255)(img)

        return self.to_tensor(img)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self._transform(img)






if __name__ == '__main__':
    from pathlib import Path

    img_path = Path.cwd().parents[1] / 'data/images/sanatanadharm/5 కర్మసన్యాసయోగము/1_10.jpeg'

    # See whether your code is working

    img = Image.open(img_path)
    image_transformer = ImagePreprocessor(image_height=64, max_image_width=1024, patch_size=16)
    transformed_img = image_transformer(img)
    # transformed_img = transformed_img.squeeze(0)
    print(img.size, '->', transformed_img.shape)

    transformed_img = torch.concat([transformed_img, transformed_img], dim=0)
    transformed_img = transformed_img.unsqueeze(1)
    print(transformed_img.shape)


    encoder_model = ViTEncoder(ViTConfig())
    out = encoder_model(transformed_img)
    print(out.shape)
