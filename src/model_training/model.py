from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class MAEConfig:
    # Image
    img_height: int = 64
    img_width: int = 256
    patch_size: int = 16
    in_channels: int = 3

    # Encoder
    enc_embed_dim: int = 768
    enc_depth: int = 12
    enc_num_heads: int = 12
    enc_mlp_ratio: float = 4.0

    # Decoder (lighter)
    dec_embed_dim: int = 512
    dec_depth: int = 4
    dec_num_heads: int = 8
    dec_mlp_ratio: float = 4.0

    # MAE
    mask_ratio: float = 0.75

    # Training
    dropout: float = 0.0


class PatchEmbed(nn.Module):
    def __init__(self, cfg: MAEConfig):
        super().__init__()
        self.patch_size = cfg.patch_size
        self.proj = nn.Conv2d(
            cfg.in_channels,
            cfg.enc_embed_dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size
        )

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)               # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    def __init__(self, cfg: MAEConfig, num_patches):
        super().__init__()
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, cfg.enc_embed_dim)
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                cfg.enc_embed_dim,
                cfg.enc_num_heads,
                cfg.enc_mlp_ratio,
                cfg.dropout
            )
            for _ in range(cfg.enc_depth)
        ])

        self.norm = nn.LayerNorm(cfg.enc_embed_dim)

    def forward(self, x):
        # print("Input vector: ", x.shape)
        # print("Embed vector: ", self.pos_embed.shape)
        # x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

class MAEDecoder(nn.Module):
    def __init__(self, cfg: MAEConfig, num_patches):
        super().__init__()

        self.embed = nn.Linear(cfg.enc_embed_dim, cfg.dec_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.dec_embed_dim))

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, cfg.dec_embed_dim)
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                cfg.dec_embed_dim,
                cfg.dec_num_heads,
                cfg.dec_mlp_ratio,
                cfg.dropout
            )
            for _ in range(cfg.dec_depth)
        ])

        self.norm = nn.LayerNorm(cfg.dec_embed_dim)

        patch_dim = cfg.patch_size * cfg.patch_size * cfg.in_channels
        self.head = nn.Linear(cfg.dec_embed_dim, patch_dim)

    def forward(self, x, ids_restore):
        x = self.embed(x)

        B, L, D = x.shape
        N = ids_restore.shape[1]

        mask_tokens = self.mask_token.repeat(B, N - L, 1)
        x_ = torch.cat([x, mask_tokens], dim=1)

        x_ = torch.gather(
            x_, dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, D)
        )

        x_ = x_ + self.pos_embed

        for blk in self.blocks:
            x_ = blk(x_)

        x_ = self.norm(x_)
        return self.head(x_)


class MaskedAutoencoderViT(nn.Module):
    def __init__(self, cfg: MAEConfig):
        super().__init__()
        self.cfg = cfg

        self.patch_embed = PatchEmbed(cfg)

        num_patches = (cfg.img_height // cfg.patch_size) * \
                      (cfg.img_width // cfg.patch_size)

        self.encoder = ViTEncoder(cfg, num_patches)
        self.decoder = MAEDecoder(cfg, num_patches)

    def forward(self, imgs):
        # print(f"1: {imgs.shape}")
        patches = self.patch_embed(imgs)
        # print(f"2: {patches.shape}")

        # Add positional embedding before masking
        patches = patches + self.encoder.pos_embed

        x_masked, mask, ids_restore = random_masking(
            patches, self.cfg.mask_ratio
        )
        # print(f"3: {x_masked.shape}")

        latent = self.encoder(x_masked)
        pred = self.decoder(latent, ids_restore)

        return pred, mask





def random_masking(x, mask_ratio):
    """
    x: [B, N, D]
    """
    B, N, D = x.shape
    len_keep = int(N * (1 - mask_ratio))

    noise = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(
        x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
    )

    mask = torch.ones([B, N], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore



def mae_loss(imgs, pred, mask, cfg: MAEConfig):
    B, C, H, W = imgs.shape
    P = cfg.patch_size

    target = imgs.unfold(2, P, P).unfold(3, P, P)
    target = target.permute(0, 2, 3, 1, 4, 5)
    target = target.reshape(B, -1, P * P * C)

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)

    return (loss * mask).sum() / mask.sum()


if __name__ == '__main__':

    model_config = MAEConfig(img_height=32, img_width=128, enc_depth=8, enc_num_heads=4, in_channels=3, enc_embed_dim=256, dec_embed_dim=128, dec_num_heads=4)

    model = MaskedAutoencoderViT(model_config).to("mps")

    num_params = 0
    for layer in model.encoder.parameters():
        num_params += layer.numel()

    print(f"No. of elements in the encoder is: {num_params}")

    num_params = 0
    for layer in model.decoder.parameters():
        num_params += layer.numel()

    print(f"No. of elements in the decoder is: {num_params}")
