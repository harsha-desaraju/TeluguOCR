"""
Convolutional-stem CTC image encoder (Stage A of the Telugu OCR spec).
======================================================================

Implements the image encoder described in §2 of ``telugu_ocr_training_spec.md``:
a convolutional stem that tokenizes a grayscale line image, a pre-LN
transformer encoder, and a linear CTC head over the grapheme vocabulary.

This replaced an earlier ViT Masked Auto-Encoder, which no longer exists anywhere in
the repo. The spec (§11 decision log) rejected MAE pretraining and frozen-encoder CTC
probing for this task; the encoder here is always trained end-to-end against the CTC
objective.

Shapes (input 1×64×W grayscale, W a multiple of 8, W ≤ max_image_width)::

    conv stem : (B, 1, 64, W)  ->  (B, 384, 1, W/8)  ->  (B, T=W/8, 384)
    transformer: (B, T, 384)   ->  (B, T, 384)
    CTC head  : (B, T, 384)    ->  (B, T, 2049)          (2048 graphemes + blank)

Height collapses to 1 architecturally inside the stem, so ``T = W/8`` tokens is
the CTC "time" axis (max 128 at W=1024). Every conv layer strides the height by
2 (64→32→16→8→4→2→1) and only strides the width on layers 1/2/4 (→ W/8 total).

The model computes the CTC loss internally when ``labels`` are supplied and
returns ``{"loss", "logits"}`` (HuggingFace ``Trainer``-compatible, mirroring
``train_ctc_probe.py``); ``logits`` are the greedy per-frame token ids with
padded frames forced to blank, cheap to gather for CER computation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


# ============================================================================
# Config
# ============================================================================
@dataclass
class CTCEncoderConfig:
    # ---- input ----
    image_height: int = 64          # fixed by the positional design (§2)
    max_image_width: int = 1024     # W ≤ 1024, multiple of 8
    downsample: int = 8             # conv-stem width reduction: T = W // downsample
    max_frames: int = 128           # = max_image_width // downsample (pos-emb length)

    # ---- conv stem ----
    # Per-layer output channels for the 6 conv blocks (§2.1 table).
    stem_channels: tuple = (32, 64, 128, 256, 320, 384)
    num_groups: int = 32            # GroupNorm groups (all stem widths divide 32)

    # ---- transformer encoder ----
    embed_dim: int = 384            # d_model (= last stem channel)
    num_layers: int = 10
    num_heads: int = 8
    mlp_dim: int = 1536
    dropout: float = 0.1
    drop_path_rate: float = 0.1     # stochastic depth (max rate, linearly scaled)

    # ---- CTC head ----
    vocab_size: int = 2048          # grapheme classes; blank appended at this index
    # blank_id and num_classes are derived: blank_id = vocab_size,
    # num_classes = vocab_size + 1.


# ============================================================================
# Building blocks
# ============================================================================
class ConvBlock(nn.Module):
    """Conv → GroupNorm → GELU (one row of the §2.1 stem table)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple,
        padding: tuple,
        num_groups: int,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        # GroupNorm groups must divide the channel count; fall back gracefully.
        groups = num_groups if out_channels % num_groups == 0 else 1
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ConvStem(nn.Module):
    """Six-block convolutional tokenizer: (B,1,64,W) -> (B, T=W/8, 384).

    Strides follow §2.1 exactly: height is halved every block (64→1) while the
    width is only halved on blocks 1, 2 and 4, giving a net width factor of 8.
    """

    def __init__(self, cfg: CTCEncoderConfig):
        super().__init__()
        c = cfg.stem_channels
        assert len(c) == 6, "stem expects 6 conv blocks (§2.1)"
        assert c[-1] == cfg.embed_dim, "last stem channel must equal embed_dim"

        # (kernel, stride, padding) per block. 3×3 blocks pad 1 to keep exact
        # /2 (stride 2) or same-size (stride 1) spatial arithmetic; the final
        # 2×1 block uses no padding to fold height 2 → 1.
        specs = [
            ((3, 3), (2, 2), (1, 1)),  # Conv1: 32×32×W/2
            ((3, 3), (2, 2), (1, 1)),  # Conv2: 64×16×W/4
            ((3, 3), (2, 1), (1, 1)),  # Conv3: 128×8×W/4
            ((3, 3), (2, 2), (1, 1)),  # Conv4: 256×4×W/8
            ((3, 3), (2, 1), (1, 1)),  # Conv5: 320×2×W/8
            ((2, 1), (2, 1), (0, 0)),  # Conv6: 384×1×W/8
        ]

        in_ch = 1
        blocks = []
        for out_ch, (k, s, p) in zip(c, specs):
            blocks.append(ConvBlock(in_ch, out_ch, k, s, p, cfg.num_groups))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H=64, W)
        for block in self.blocks:
            x = block(x)
        B, D, H, T = x.shape
        assert H == 1, f"conv stem did not collapse height to 1 (got {H})"
        return x.squeeze(2).transpose(1, 2)  # (B, T, D)


class DropPath(nn.Module):
    """Stochastic depth: randomly drop the residual branch per-sample at train time."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # broadcast a per-sample mask over all non-batch dims
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x / keep_prob * mask


class TransformerBlock(nn.Module):
    """Pre-LN transformer encoder block with stochastic depth (§2.1)."""

    def __init__(self, cfg: CTCEncoderConfig, drop_path: float):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(cfg.embed_dim)
        self.attention = nn.MultiheadAttention(
            cfg.embed_dim, cfg.num_heads, dropout=cfg.dropout, batch_first=True
        )
        self.layer_norm2 = nn.LayerNorm(cfg.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.mlp_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_dim, cfg.embed_dim),
            nn.Dropout(cfg.dropout),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        norm_x = self.layer_norm1(x)
        attn_out, _ = self.attention(
            norm_x, norm_x, norm_x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.layer_norm2(x)))
        return x


# ============================================================================
# Image encoder + CTC head
# ============================================================================
class ImageEncoderCTC(nn.Module):
    """Conv stem → transformer encoder → CTC head (~20M params).

    forward(images, input_lengths=None, labels=None, label_lengths=None)

      images        (B, 1, 64, W)   grayscale line images, W a multiple of 8
      input_lengths (B,)            valid frame count per sample = W_real // 8.
                                    None -> every sample assumed full width (T).
      labels        (B, S)          grapheme target ids, right-padded with
                                    ``label_pad_id``. Passing labels triggers the
                                    internal CTC loss.
      label_lengths (B,)            optional explicit target lengths; if None they
                                    are derived from ``labels != label_pad_id``.

    Returns ``{"loss", "logits", "log_probs"}`` where
      loss      scalar CTC loss (None if labels not given)
      logits    (B, T) greedy per-frame token ids, padded frames forced to blank
      log_probs (B, T, C) fp32 frame log-probabilities (for beam search / fusion)
    """

    def __init__(self, cfg: CTCEncoderConfig, label_pad_id: int = -100):
        super().__init__()
        self.cfg = cfg
        self.blank_id = cfg.vocab_size
        self.num_classes = cfg.vocab_size + 1
        self.label_pad_id = label_pad_id

        self.stem = ConvStem(cfg)

        # Learned 1D positional embeddings, length max_frames (§2.1).
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.max_frames, cfg.embed_dim))
        self.dropout = nn.Dropout(cfg.dropout)

        # Stochastic depth: linearly increasing drop rate 0 -> drop_path_rate.
        dpr = torch.linspace(0.0, cfg.drop_path_rate, cfg.num_layers).tolist()
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, drop_path=dpr[i]) for i in range(cfg.num_layers)]
        )
        self.layer_norm = nn.LayerNorm(cfg.embed_dim)

        self.ctc_head = nn.Linear(cfg.embed_dim, self.num_classes)

        self.ctc_loss = nn.CTCLoss(blank=self.blank_id, zero_infinity=True)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, images: torch.Tensor, input_lengths: torch.Tensor | None):
        """Run the stem + transformer, returning frame features and the frame
        padding mask (True = padded frame)."""
        feats = self.stem(images)                       # (B, T, D)
        B, T, D = feats.shape
        assert T <= self.cfg.max_frames, (
            f"T={T} exceeds max_frames={self.cfg.max_frames}; widen the pos-emb"
        )
        feats = feats + self.pos_embed[:, :T, :]
        feats = self.dropout(feats)

        if input_lengths is not None:
            frame_idx = torch.arange(T, device=feats.device).unsqueeze(0)  # (1, T)
            key_padding_mask = frame_idx >= input_lengths.unsqueeze(1)     # (B, T)
        else:
            key_padding_mask = None

        for block in self.blocks:
            feats = block(feats, key_padding_mask)
        feats = self.layer_norm(feats)
        return feats, key_padding_mask

    def forward(
        self,
        images: torch.Tensor,
        input_lengths: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        label_lengths: torch.Tensor | None = None,
    ):
        feats, _ = self.encode(images, input_lengths)   # (B, T, D)
        logits = self.ctc_head(feats)                   # (B, T, C)
        B, T, C = logits.shape

        if input_lengths is None:
            input_lengths = torch.full((B,), T, dtype=torch.long, device=logits.device)

        # log_softmax kept in fp32 for CTC numerical stability under fp16 autocast.
        log_probs = logits.float().log_softmax(dim=-1)  # (B, T, C)

        loss = None
        if labels is not None:
            if label_lengths is None:
                target_mask = labels != self.label_pad_id
                label_lengths = target_mask.sum(dim=1)
                targets = labels[target_mask]           # flattened, pad removed
            else:
                targets = labels
            # nn.CTCLoss expects (T, B, C).
            loss = self.ctc_loss(
                log_probs.permute(1, 0, 2),
                targets,
                input_lengths,
                label_lengths,
            )

        # Greedy per-frame ids with padded frames forced to blank (clean decode).
        pred_ids = log_probs.argmax(dim=-1)             # (B, T)
        beyond = torch.arange(T, device=logits.device).unsqueeze(0) >= input_lengths.unsqueeze(1)
        pred_ids = pred_ids.masked_fill(beyond, self.blank_id)

        return {"loss": loss, "logits": pred_ids, "log_probs": log_probs}

    @staticmethod
    def frames_from_width(width: int, downsample: int = 8) -> int:
        """Number of CTC frames a line of pixel ``width`` produces (= width/8)."""
        assert width % downsample == 0, "width must be a multiple of the downsample factor"
        return width // downsample


# ============================================================================
# Sanity check
# ============================================================================
if __name__ == "__main__":
    cfg = CTCEncoderConfig()
    model = ImageEncoderCTC(cfg)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,} (~{num_params / 1e6:.1f}M)")

    # Two samples of different widths, batched to the wider one.
    B, H, W = 2, cfg.image_height, 512
    images = torch.randn(B, 1, H, W)
    # sample 0 uses full width; sample 1 only the first 384px (48 frames).
    input_lengths = torch.tensor(
        [W // cfg.downsample, 384 // cfg.downsample], dtype=torch.long
    )

    # Fake CTC targets (grapheme ids in [0, vocab_size)), padded with -100.
    labels = torch.full((B, 10), -100, dtype=torch.long)
    labels[0, :8] = torch.randint(0, cfg.vocab_size, (8,))
    labels[1, :5] = torch.randint(0, cfg.vocab_size, (5,))

    out = model(images, input_lengths=input_lengths, labels=labels)
    T = W // cfg.downsample
    print(f"blank id: {model.blank_id} | num classes: {model.num_classes}")
    print(f"log_probs: {tuple(out['log_probs'].shape)} (expected ({B}, {T}, {model.num_classes}))")
    print(f"greedy logits: {tuple(out['logits'].shape)} (expected ({B}, {T}))")
    print(f"CTC loss: {out['loss'].item():.4f}")

    # T ≥ 2L + 1 CTC feasibility check (§2.2).
    for i, L in enumerate([8, 5]):
        assert int(input_lengths[i]) >= 2 * L + 1, "CTC T ≥ 2L+1 violated"
    print("CTC feasibility (T >= 2L+1): OK")
