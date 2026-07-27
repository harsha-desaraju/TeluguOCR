"""
Stage-A CONTINUATION: extend the CTC image encoder 1024px -> 2048px and resume.
===============================================================================

Resumes an existing ``ImageEncoderCTC`` (conv stem + 10-layer transformer, d=384,
learned 1D positional embeddings) from a checkpoint and extends the maximum line
width 1024->2048px, i.e. the frame axis T 128->256. Self-contained / Kaggle-ready
(inlines model, tokenizer, augmentation, preprocessing, collator) like the other
training scripts. Derived from ``train_ctc_encoder.py``; changes vs that file are
marked with ``# CHANGED(2048): ...`` comments (what / where / why).

Explicitly UNCHANGED (per spec): vocab/blank index, conv-stem strides, CTC loss,
dropout / stochastic depth, and the augmentation pipeline (inlined verbatim).

Key design points:
  * Positional embedding is SPLIT into two Parameters — ``pos_embed`` (rows 0-127,
    loaded bit-identical from the checkpoint) and ``pos_embed_ext`` (rows 128-255,
    fresh trunc_normal init) — so the new rows can take a higher LR (PyTorch LR
    param-groups act per whole Parameter). No interpolation.
  * Plain HF Trainer: it handles DDP sharding, the sampler, checkpoint/resume and
    step counting. No custom batch sampler / width buckets (one per-device batch
    size for all widths; lower it if memory is tight). The 2-tier warmup->cosine
    schedule is built in a ``create_scheduler`` override so Trainer supplies the
    correct ``num_training_steps`` (= full epochs).
  * Length-stratified eval every 2k steps: short (T<=128) vs long (129-256) crossed
    with natural vs random text = 4 CER slices, plus non-blank emission rate.
  * This variant uses a NORMAL (downloaded, map-style) dataset: the whole ``train``
    split is fetched to disk and transformed on the fly via ``with_transform``.
    Training is epoch-based (``num_train_epochs``) — Trainer knows the length and
    computes the total optimizer steps (the scheduler T_max) itself. See
    ``train_ctc_encoder_2048.py`` for the streaming variant (use that when the pool
    doesn't fit on disk).

Run (single GPU / CPU):   python3 -m src.image_encoder.train_ctc_encoder_2048_normal
Run (multi-GPU DDP):      torchrun --nproc_per_node=<N> -m src.image_encoder.train_ctc_encoder_2048_normal
"""

from __future__ import annotations

import os
import sys
import math
import json
import regex
import random

from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from datasets import load_dataset, concatenate_datasets
from transformers import Trainer, TrainingArguments, PreTrainedTokenizer, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint


# ============================================================================
# Model config (CHANGED(2048) fields flagged)
# ============================================================================
@dataclass
class CTCEncoderConfig:
    # ---- input ----
    image_height: int = 64
    max_image_width: int = 2048     # CHANGED(2048): 1024 -> 2048 (W <= 2048)
    downsample: int = 8             # T = W // downsample
    base_frames: int = 128          # CHANGED(2048): rows loaded from the old ckpt (old max)
    max_frames: int = 256           # CHANGED(2048): 128 -> 256 pos-emb length (T up to 256)

    # ---- conv stem (UNCHANGED strides/channels) ----
    stem_channels: tuple = (32, 64, 128, 256, 320, 384)
    num_groups: int = 32

    # ---- transformer encoder (UNCHANGED) ----
    embed_dim: int = 384
    num_layers: int = 10
    num_heads: int = 8
    mlp_dim: int = 1536
    dropout: float = 0.1            # UNCHANGED (spec)
    drop_path_rate: float = 0.1     # UNCHANGED (spec)

    # ---- CTC head (UNCHANGED vocab/blank) ----
    vocab_size: int = 2048          # blank appended at index vocab_size


# ============================================================================
# Building blocks (UNCHANGED)
# ============================================================================
class ConvBlock(nn.Module):
    """Conv -> GroupNorm -> GELU."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, num_groups):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        groups = num_groups if out_channels % num_groups == 0 else 1
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ConvStem(nn.Module):
    """Six-block convolutional tokenizer: (B,1,64,W) -> (B, T=W/8, 384). UNCHANGED."""

    def __init__(self, cfg: CTCEncoderConfig):
        super().__init__()
        c = cfg.stem_channels
        assert len(c) == 6 and c[-1] == cfg.embed_dim
        specs = [
            ((3, 3), (2, 2), (1, 1)),  # 32×32×W/2
            ((3, 3), (2, 2), (1, 1)),  # 64×16×W/4
            ((3, 3), (2, 1), (1, 1)),  # 128×8×W/4
            ((3, 3), (2, 2), (1, 1)),  # 256×4×W/8
            ((3, 3), (2, 1), (1, 1)),  # 320×2×W/8
            ((2, 1), (2, 1), (0, 0)),  # 384×1×W/8
        ]
        in_ch, blocks = 1, []
        for out_ch, (k, s, p) in zip(c, specs):
            blocks.append(ConvBlock(in_ch, out_ch, k, s, p, cfg.num_groups))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        B, D, H, T = x.shape
        assert H == 1, f"conv stem did not collapse height to 1 (got {H})"
        return x.squeeze(2).transpose(1, 2)  # (B, T, D)


class DropPath(nn.Module):
    """Stochastic depth (UNCHANGED)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x / keep * mask


class TransformerBlock(nn.Module):
    """Pre-LN transformer block (UNCHANGED)."""

    def __init__(self, cfg: CTCEncoderConfig, drop_path: float):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(cfg.embed_dim)
        self.attention = nn.MultiheadAttention(
            cfg.embed_dim, cfg.num_heads, dropout=cfg.dropout, batch_first=True
        )
        self.layer_norm2 = nn.LayerNorm(cfg.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.mlp_dim), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_dim, cfg.embed_dim), nn.Dropout(cfg.dropout),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x, key_padding_mask=None):
        norm_x = self.layer_norm1(x)
        attn_out, _ = self.attention(norm_x, norm_x, norm_x,
                                     key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.layer_norm2(x)))
        return x


LABEL_PAD_ID = -100


class ImageEncoderCTC(nn.Module):
    """Conv stem -> transformer -> CTC head. Returns {"loss","logits"} for HF Trainer.

    CHANGED(2048): the learned positional embedding is SPLIT into two Parameters so
    the extended rows can take a different LR:
      * ``pos_embed``     (1, base_frames, D)            <- loaded from ckpt
      * ``pos_embed_ext`` (1, max_frames-base_frames, D) <- fresh trunc_normal
    forward concatenates them; short lines (T<=base_frames) never touch pos_embed_ext,
    so those rows only receive gradient from the new long lines.
    """

    def __init__(self, cfg: CTCEncoderConfig, label_pad_id: int = LABEL_PAD_ID):
        super().__init__()
        self.cfg = cfg
        self.blank_id = cfg.vocab_size           # UNCHANGED blank index
        self.num_classes = cfg.vocab_size + 1
        self.label_pad_id = label_pad_id

        self.stem = ConvStem(cfg)

        # CHANGED(2048): split positional embedding into base + extension.
        n_ext = cfg.max_frames - cfg.base_frames
        assert n_ext > 0, "max_frames must exceed base_frames"
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.base_frames, cfg.embed_dim))
        self.pos_embed_ext = nn.Parameter(torch.zeros(1, n_ext, cfg.embed_dim))

        self.dropout = nn.Dropout(cfg.dropout)
        dpr = torch.linspace(0.0, cfg.drop_path_rate, cfg.num_layers).tolist()
        self.blocks = nn.ModuleList([TransformerBlock(cfg, dpr[i]) for i in range(cfg.num_layers)])
        self.layer_norm = nn.LayerNorm(cfg.embed_dim)
        self.ctc_head = nn.Linear(cfg.embed_dim, self.num_classes)
        self.ctc_loss = nn.CTCLoss(blank=self.blank_id, zero_infinity=True)  # UNCHANGED

        self._init_weights()

    def _init_weights(self):
        # Same std as the original init; the base table is overwritten by the
        # checkpoint load in extend_from_checkpoint (the ext rows stay at this init).
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_ext, std=0.02)  # CHANGED(2048): init new rows
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _pos(self, T):
        # CHANGED(2048): concat the two tables, then slice to T.
        full = torch.cat([self.pos_embed, self.pos_embed_ext], dim=1)  # (1, max_frames, D)
        return full[:, :T, :]

    def forward(self, images, input_lengths=None, labels=None, label_lengths=None):
        feats = self.stem(images)                       # (B, T, D)
        B, T, D = feats.shape
        # fail-fast tripwire if a stray line wider than max_frames*downsample slips in.
        assert T <= self.cfg.max_frames, f"T={T} exceeds max_frames={self.cfg.max_frames}"
        feats = feats + self._pos(T)
        feats = self.dropout(feats)

        if input_lengths is None:
            input_lengths = torch.full((B,), T, dtype=torch.long, device=feats.device)

        frame_idx = torch.arange(T, device=feats.device).unsqueeze(0)
        key_padding_mask = frame_idx >= input_lengths.unsqueeze(1)   # (B, T) True = pad

        for block in self.blocks:
            feats = block(feats, key_padding_mask)
        feats = self.layer_norm(feats)
        logits = self.ctc_head(feats)                   # (B, T, C)

        log_probs = logits.float().log_softmax(dim=-1)  # fp32 for CTC stability

        loss = None
        if labels is not None:
            if label_lengths is None:
                target_mask = labels != self.label_pad_id
                label_lengths = target_mask.sum(dim=1)
                targets = labels[target_mask]
            else:
                targets = labels
            loss = self.ctc_loss(log_probs.permute(1, 0, 2), targets, input_lengths, label_lengths)

        pred_ids = log_probs.argmax(dim=-1)
        pred_ids = pred_ids.masked_fill(key_padding_mask, self.blank_id)
        return {"loss": loss, "logits": pred_ids}


# ============================================================================
# Checkpoint loader — CHANGED(2048): extend pos_embed from an OLD (128-frame) ckpt
# ============================================================================
def extend_from_checkpoint(model: ImageEncoderCTC, ckpt_path: str):
    """Load an OLD checkpoint into the extended model. Every param loads strictly
    EXCEPT pos_embed_ext (the new rows), which stays at its fresh trunc_normal init.
    The old ``pos_embed`` (rows 0-127) is copied in bit-identically; no interpolation.
    """
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(ckpt_path)
    else:
        state = torch.load(ckpt_path, map_location="cpu")
    if any(k.startswith("model.") for k in state) and "pos_embed" not in state:
        state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}

    if "pos_embed" not in state:
        raise KeyError("checkpoint has no 'pos_embed'; is this an ImageEncoderCTC checkpoint?")
    exp = (1, model.cfg.base_frames, model.cfg.embed_dim)
    assert tuple(state["pos_embed"].shape) == exp, (
        f"old pos_embed {tuple(state['pos_embed'].shape)} != expected {exp}; "
        f"set base_frames to the old max_frames.")

    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [m for m in missing if m != "pos_embed_ext"]   # ext kept at fresh init
    assert not missing and not unexpected, f"load mismatch: missing={missing} unexpected={unexpected}"
    print(f"[extend] loaded old ckpt; pos_embed[0:{model.cfg.base_frames}] copied, "
          f"pos_embed_ext[{model.cfg.base_frames}:{model.cfg.max_frames}] fresh-init.")
    return model


# ============================================================================
# Grapheme tokenizer (inlined, UNCHANGED)
# ============================================================================
class TeluguGraphemeTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "vocab.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file=None, vocab_list=None, add_bos_token=True, add_eos_token=True,
                 pad_token="[PAD]", unk_token="[UNK]", bos_token="[BOS]", eos_token="[EOS]",
                 mask_token="[MASK]", **kwargs):
        vocab = {}
        if vocab_file is not None:
            with open(vocab_file, encoding="utf-8") as f:
                vocab = json.load(f)
        if not vocab and vocab_list is not None:
            for g in vocab_list:
                vocab[g] = len(vocab)
        if not vocab:
            raise AssertionError("Either `vocab_file` or `vocab_list` has to be given.")
        self.SPECIAL_TOKENS_LIST = [pad_token, unk_token, bos_token, eos_token, mask_token]
        self.grapheme_pattern = regex.compile(r"\X")
        for tok in self.SPECIAL_TOKENS_LIST:
            if tok not in vocab:
                vocab[tok] = len(vocab)
        self.vocab = vocab
        self._inv_vocab = {v: k for k, v in vocab.items()}
        self.add_bos_token, self.add_eos_token, self.UNK = add_bos_token, add_eos_token, unk_token
        super().__init__(pad_token=pad_token, unk_token=unk_token, bos_token=bos_token,
                         eos_token=eos_token, mask_token=mask_token, add_bos_token=add_bos_token,
                         add_eos_token=add_eos_token, padding_side="right", model_max_length=4096, **kwargs)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab)

    def _tokenize(self, text, **kwargs):
        out = []
        for g in self.grapheme_pattern.findall(text):
            if g in self.vocab:
                out.append(g)
            else:
                out.extend(list(g))
        return out

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.UNK, 1))

    def _convert_id_to_token(self, index):
        return self._inv_vocab.get(index, self.UNK)

    def convert_tokens_to_string(self, tokens):
        return "".join(t for t in tokens if t not in self.SPECIAL_TOKENS_LIST)

    def build_inputs_with_special_tokens(self, a, b=None):
        bos = [self.bos_token_id] if self.add_bos_token else []
        eos = [self.eos_token_id] if self.add_eos_token else []
        out = bos + a + eos
        if b is not None:
            out += bos + b + eos
        return out

    def save_vocabulary(self, save_directory, filename_prefix=None):
        os.makedirs(save_directory, exist_ok=True)
        fname = (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        path = os.path.join(save_directory, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        return (path,)


# ============================================================================
# Image augmentation — INLINED VERBATIM from train_ctc_encoder.py (UNCHANGED per spec)
# ============================================================================
# CHANGED(2048): inlined here (was a cross-file import) so this training file is
# self-contained. The pipeline itself is identical to train_ctc_encoder.py.
import cv2  # noqa: E402
from augraphy import (  # noqa: E402
    InkBleed, InkMottling, BleedThrough,
    LowInkRandomLines, BrightnessTexturize,
    DirtyDrum, DirtyRollers, SubtleNoise, Jpeg,
)


def set_seed(seed):
    """Seed `random` (combo choice) and `np.random` (custom effects + augraphy)."""
    random.seed(seed)
    np.random.seed(seed)


def motion_blur(img: np.ndarray, degree: int = 7) -> np.ndarray:
    kernel = np.zeros((degree, degree), dtype=np.float32)
    kernel[(degree - 1) // 2, :] = 1.0
    kernel /= degree
    return cv2.filter2D(img, -1, kernel)


def gaussian_noise(img: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    noise = np.random.normal(0.0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def salt_pepper(img: np.ndarray, amount: float = 0.003) -> np.ndarray:
    out = img.copy()
    h, w = img.shape[:2]
    num = int(amount * h * w)
    rows = np.random.randint(0, h, num); cols = np.random.randint(0, w, num)
    out[rows, cols] = 255
    rows = np.random.randint(0, h, num); cols = np.random.randint(0, w, num)
    out[rows, cols] = 0
    return out


# augraphy objects are PRE-BUILT once (construction is the dominant per-sample cost).
COMBOS = {
    "InkBleed": [
        ("int .2 / sev .1", InkBleed(intensity_range=(0.2, 0.2), severity=(0.1, 0.1), kernel_size=(5, 5), p=1)),
        ("int .4 / sev .2", InkBleed(intensity_range=(0.4, 0.4), severity=(0.2, 0.2), kernel_size=(5, 5), p=1)),
        ("int .7 / sev .3", InkBleed(intensity_range=(0.7, 0.7), severity=(0.3, 0.3), kernel_size=(5, 5), p=1)),
        ("int .95 / sev .4", InkBleed(intensity_range=(0.95, 0.95), severity=(0.4, 0.4), kernel_size=(5, 5), p=1)),
    ],
    "InkMottling": [
        ("alpha .10 / k3", InkMottling(ink_mottling_alpha_range=(0.10, 0.10), ink_mottling_gaussian_kernel_range=(3, 3), p=1)),
        ("alpha .20 / k3", InkMottling(ink_mottling_alpha_range=(0.20, 0.20), ink_mottling_gaussian_kernel_range=(3, 3), p=1)),
        ("alpha .35 / k5", InkMottling(ink_mottling_alpha_range=(0.35, 0.35), ink_mottling_gaussian_kernel_range=(5, 5), p=1)),
        ("alpha .50 / k7", InkMottling(ink_mottling_alpha_range=(0.50, 0.50), ink_mottling_gaussian_kernel_range=(7, 7), p=1)),
    ],
    "BleedThrough": [
        ("int .1 / a .1 / off10", BleedThrough(intensity_range=(0.1, 0.1), alpha=0.1, offsets=(10, 10), p=1)),
        ("int .3 / a .2 / off20", BleedThrough(intensity_range=(0.3, 0.3), alpha=0.2, offsets=(20, 20), p=1)),
        ("int .6 / a .3 / off30", BleedThrough(intensity_range=(0.6, 0.6), alpha=0.3, offsets=(30, 30), p=1)),
        ("int .9 / a .4 / off40", BleedThrough(intensity_range=(0.9, 0.9), alpha=0.4, offsets=(40, 40), p=1)),
    ],
    "LowInkRandomLines": [
        ("count 3", LowInkRandomLines(count_range=(3, 3), use_consistent_lines=True, p=1)),
        ("count 8", LowInkRandomLines(count_range=(8, 8), use_consistent_lines=True, p=1)),
        ("count 15", LowInkRandomLines(count_range=(15, 15), use_consistent_lines=True, p=1)),
        ("count 25", LowInkRandomLines(count_range=(25, 25), use_consistent_lines=True, p=1)),
    ],
    "BrightnessTexturize": [
        ("tex .97 / dev .03", BrightnessTexturize(texturize_range=(0.95, 0.99), deviation=0.03, p=1)),
        ("tex .92 / dev .08", BrightnessTexturize(texturize_range=(0.90, 0.94), deviation=0.08, p=1)),
        ("tex .82 / dev .15", BrightnessTexturize(texturize_range=(0.80, 0.84), deviation=0.15, p=1)),
        ("tex .72 / dev .25", BrightnessTexturize(texturize_range=(0.70, 0.74), deviation=0.25, p=1)),
    ],
    "DirtyDrum": [
        ("lw1 / conc .05 / ni .2", DirtyDrum(line_width_range=(1, 1), line_concentration=0.05, noise_intensity=0.2, p=1)),
        ("lw1-2 / conc .1 / ni .4", DirtyDrum(line_width_range=(1, 2), line_concentration=0.10, noise_intensity=0.4, p=1)),
        ("lw2-3 / conc .2 / ni .6", DirtyDrum(line_width_range=(2, 3), line_concentration=0.20, noise_intensity=0.6, p=1)),
        ("lw3-5 / conc .3 / ni .8", DirtyDrum(line_width_range=(3, 5), line_concentration=0.30, noise_intensity=0.8, p=1)),
    ],
    "DirtyRollers": [
        ("lw 4-6", DirtyRollers(line_width_range=(4, 6), p=1)),
        ("lw 8-12", DirtyRollers(line_width_range=(8, 12), p=1)),
        ("lw 16-20", DirtyRollers(line_width_range=(16, 20), p=1)),
        ("lw 28-36", DirtyRollers(line_width_range=(28, 36), p=1)),
    ],
    "SubtleNoise": [
        ("range 5", SubtleNoise(subtle_range=5, p=1)),
        ("range 15", SubtleNoise(subtle_range=15, p=1)),
        ("range 30", SubtleNoise(subtle_range=30, p=1)),
        ("range 50", SubtleNoise(subtle_range=50, p=1)),
    ],
    "Jpeg": [
        ("quality 90", Jpeg(quality_range=(90, 90), p=1)),
        ("quality 70", Jpeg(quality_range=(70, 70), p=1)),
        ("quality 45", Jpeg(quality_range=(45, 45), p=1)),
        ("quality 20", Jpeg(quality_range=(20, 20), p=1)),
    ],
    "MotionBlur": [
        ("deg 4", lambda im: motion_blur(im, degree=4)),
        ("deg 5", lambda im: motion_blur(im, degree=5)),
        ("deg 6", lambda im: motion_blur(im, degree=6)),
        ("deg 7", lambda im: motion_blur(im, degree=7)),
    ],
    "GaussianNoise": [
        ("sig 10", lambda im: gaussian_noise(im, sigma=10)),
        ("sig 11", lambda im: gaussian_noise(im, sigma=11)),
        ("sig 12", lambda im: gaussian_noise(im, sigma=12)),
        ("sig 13", lambda im: gaussian_noise(im, sigma=13)),
    ],
    "SaltPepper": [
        ("amt .003", lambda im: salt_pepper(im, amount=0.003)),
        ("amt .005", lambda im: salt_pepper(im, amount=0.005)),
        ("amt .007", lambda im: salt_pepper(im, amount=0.007)),
        ("amt .009", lambda im: salt_pepper(im, amount=0.009)),
    ],
}

AUG_UTILITY = {
    "DirtyRollers": 1.0, "DirtyDrum": 1.0, "BleedThrough": 0.9, "InkBleed": 0.9,
    "LowInkRandomLines": 0.8, "Jpeg": 0.7, "BrightnessTexturize": 0.6,
    "InkMottling": 0.6, "SubtleNoise": 0.5, "GaussianNoise": 0.5,
    "MotionBlur": 0.3, "SaltPepper": 0.3,
}
AUG_TIME_MS = {
    "InkBleed": 1.3, "InkMottling": 0.7, "BleedThrough": 5.4, "LowInkRandomLines": 0.2,
    "BrightnessTexturize": 0.8, "DirtyDrum": 3.9, "DirtyRollers": 16.6, "SubtleNoise": 0.2,
    "Jpeg": 0.3, "MotionBlur": 0.1, "GaussianNoise": 0.6, "SaltPepper": 0.0,
}
assert set(AUG_UTILITY) == set(COMBOS) == set(AUG_TIME_MS)

P_CLEAN = 0.25   # fraction of samples left clean
LAM = 0.0        # cost-penalty per ms; 0 = utility-only


def _build_aug_distribution(utility, times, lam):
    names = list(COMBOS.keys())
    raw = np.array([utility[n] / (1.0 + lam * times[n]) for n in names], dtype=np.float64)
    return names, list(raw / raw.sum())


_AUG_NAMES, _AUG_PROBS = _build_aug_distribution(AUG_UTILITY, AUG_TIME_MS, LAM)


def make_weighted_augmenter(p_clean=P_CLEAN):
    """Return augment(np_uint8 HxWx3) -> np_uint8 HxWx3. Never raises."""
    names, weights = _AUG_NAMES, _AUG_PROBS

    def augment(img):
        if random.random() < p_clean:
            return img
        name = random.choices(names, weights=weights, k=1)[0]
        _label, applier = random.choice(COMBOS[name])
        try:
            out = applier(img)
            if isinstance(out, dict):
                out = out.get("output", img)
            return out
        except Exception as exc:
            print(f"[warn] aug {name} failed at this scale: {exc}; using clean")
            return img

    return augment


# ============================================================================
# Preprocessing + collator
# ============================================================================
class ImagePreprocessor:
    """grayscale + to-tensor + normalize; optional augment on the RGB crop (train)."""

    def __init__(self, tokenizer, image_col, text_col, augment_fn=None):
        self.tokenizer = tokenizer
        self.image_col = image_col
        self.text_col = text_col
        self.augment_fn = augment_fn  # train split only; augmentation pipeline UNCHANGED
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def _img(self, img):
        if self.augment_fn is not None:
            arr = self.augment_fn(np.array(img.convert("RGB")))
            img = Image.fromarray(np.asarray(arr, dtype=np.uint8))
        return self.to_tensor(img.convert("L"))

    def __call__(self, batch):
        return {
            "line_image": [self._img(im) for im in batch[self.image_col]],
            "target_ids": [self.tokenizer(t, add_special_tokens=False)["input_ids"]
                           for t in batch[self.text_col]],
        }


class CTCCollator:
    def __init__(self, downsample: int):
        self.downsample = downsample

    def __call__(self, batch):
        imgs = [s["line_image"] for s in batch]
        B = len(imgs)
        C, H = imgs[0].shape[0], imgs[0].shape[1]
        widths = [im.shape[2] for im in imgs]
        # round padded width up to a multiple of downsample -> integer conv T.
        max_w = ((max(widths) + self.downsample - 1) // self.downsample) * self.downsample
        images = imgs[0].new_full((B, C, H, max_w), 1.0)  # white padding
        input_lengths = torch.empty(B, dtype=torch.long)
        for i, im in enumerate(imgs):
            w = im.shape[2]
            images[i, :, :, :w] = im
            input_lengths[i] = (w + self.downsample - 1) // self.downsample
        max_s = max(max(len(s["target_ids"]) for s in batch), 1)
        labels = torch.full((B, max_s), LABEL_PAD_ID, dtype=torch.long)
        for i, s in enumerate(batch):
            ids = s["target_ids"]
            if ids:
                labels[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        return {"images": images, "input_lengths": input_lengths, "labels": labels}


# ============================================================================
# Metrics — grapheme-level CER + non-blank emission rate (per eval slice)
# ============================================================================
_GRAPHEME = regex.compile(r"\X")


def _edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def make_compute_metrics(tokenizer, blank_id):
    def _collapse(row):
        out, prev = [], None
        for t in row:
            t = int(t)
            if t != prev and t != blank_id and t != LABEL_PAD_ID:
                out.append(t)
            prev = t
        return out

    def compute_metrics(eval_pred):
        preds = eval_pred.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        labels = eval_pred.label_ids
        tot_edits = tot_ref = 0
        for prow, lrow in zip(preds, labels):
            pred = tokenizer.decode(_collapse(prow), skip_special_tokens=True)
            ref = tokenizer.decode([int(t) for t in lrow if int(t) != LABEL_PAD_ID],
                                   skip_special_tokens=True)
            pg, rg = _GRAPHEME.findall(pred), _GRAPHEME.findall(ref)
            tot_edits += _edit_distance(pg, rg)
            tot_ref += len(rg)
        nb = sum(int(t) != blank_id for prow in preds for t in prow)  # non-blank emission
        total = sum(len(prow) for prow in preds) or 1
        return {"cer": tot_edits / max(tot_ref, 1), "nonblank": nb / total}

    return compute_metrics


# ============================================================================
# Optimizer / scheduler — CHANGED(2048): 2 LR tiers + warmup->cosine to 1e-5
# ============================================================================
def build_optimizer(model, lr_rest, lr_ext, weight_decay, betas):
    """Two LR tiers (new pos rows @lr_ext, everything else @lr_rest). Within the
    'rest' tier, biases / norm weights / positional embeddings get no weight decay
    (matches the prior Trainer default); other weights get `weight_decay`."""
    ext, rest_decay, rest_nodecay = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name == "pos_embed_ext":                       # CHANGED(2048): new rows -> lr_ext
            ext.append(p)
        elif name == "pos_embed" or p.ndim < 2 or "norm" in name.lower():
            rest_nodecay.append(p)                        # embeddings/bias/norm -> no WD
        else:
            rest_decay.append(p)
    groups = [
        {"params": rest_decay, "lr": lr_rest, "weight_decay": weight_decay},
        {"params": rest_nodecay, "lr": lr_rest, "weight_decay": 0.0},
        {"params": ext, "lr": lr_ext, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, betas=betas)


def build_scheduler(optimizer, warmup_steps, total_steps, min_lr):
    """Per-group linear warmup -> cosine decay to an ABSOLUTE floor `min_lr`.
    Each group decays from its own peak (group 'lr') down to `min_lr`."""
    from torch.optim.lr_scheduler import LambdaLR

    def make(peak):
        def f(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            prog = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
            cos = 0.5 * (1.0 + math.cos(math.pi * prog))
            return (min_lr + (peak - min_lr) * cos) / peak    # absolute floor min_lr
        return f

    return LambdaLR(optimizer, [make(g["lr"]) for g in optimizer.param_groups])


class CTCTrainer(Trainer):
    """Trainer that builds the 2-tier warmup->cosine scheduler with the step count
    Trainer computes (so we don't reason about DDP/epochs ourselves)."""

    def __init__(self, *args, warmup_steps=1500, min_lr=1e-5, **kwargs):
        self._warmup_steps = warmup_steps
        self._min_lr = min_lr
        super().__init__(*args, **kwargs)

    def create_scheduler(self, num_training_steps, optimizer=None):  # CHANGED(2048)
        if self.lr_scheduler is None:
            self.lr_scheduler = build_scheduler(
                optimizer or self.optimizer, self._warmup_steps, num_training_steps, self._min_lr)
        return self.lr_scheduler


class GradNormAlert(TrainerCallback):
    """CHANGED(2048): alert on grad-norm spikes > threshold (default 10)."""

    def __init__(self, threshold=10.0):
        self.threshold = threshold

    def on_log(self, args, state, control, logs=None, **kwargs):
        gn = (logs or {}).get("grad_norm")
        if gn is not None and gn > self.threshold:
            print(f"[ALERT] grad_norm={gn:.2f} > {self.threshold} at step {state.global_step}")


# ============================================================================
# Eval slices — CHANGED(stream): built from streamed-and-materialized rows so the
# eval path never triggers a full-dataset download (see __main__).
# ============================================================================
class ListEvalDataset(torch.utils.data.Dataset):
    """In-memory map-style eval dataset over a list of raw row dicts. Applies the
    (no-aug) eval preprocessing per item — Trainer accepts this as an eval dataset."""

    def __init__(self, rows, preprocessor, image_col, text_col):
        self.rows = rows
        self.prep = preprocessor
        self.image_col = image_col
        self.text_col = text_col

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return {
            "line_image": self.prep._img(r[self.image_col]),
            "target_ids": self.prep.tokenizer(r[self.text_col], add_special_tokens=False)["input_ids"],
        }


def _row_width(row, image_col, width_col):
    """Pixel width for a raw row: width column if present, else the image header."""
    if width_col and width_col in row and row[width_col] is not None:
        return int(row[width_col])
    return int(row[image_col].width)


def build_eval_slices(rows, image_col, text_col, source_col, width_col, preprocessor,
                      base_width, slice_cap, seed):
    """Partition materialized eval rows into 4 (or 2) slices by width and text source."""
    widths = [_row_width(r, image_col, width_col) for r in rows]
    has_src = bool(rows) and source_col in rows[0]

    def subset(predicate):
        idx = [i for i in range(len(rows)) if predicate(i)]
        if not idx:
            return None
        if slice_cap and len(idx) > slice_cap:
            g = torch.Generator().manual_seed(seed)
            idx = [idx[j] for j in torch.randperm(len(idx), generator=g)[:slice_cap].tolist()]
        return ListEvalDataset([rows[i] for i in idx], preprocessor, image_col, text_col)

    slices = {}
    if has_src:
        for lbl, longp in (("short", False), ("long", True)):
            for s in ("natural", "random"):
                sub = subset(lambda i, lp=longp, ss=s: (widths[i] > base_width) == lp and rows[i].get(source_col) == ss)
                if sub is not None:
                    slices[f"{lbl}_{s}"] = sub
    else:
        for lbl, longp in (("short", False), ("long", True)):
            sub = subset(lambda i, lp=longp: (widths[i] > base_width) == lp)
            if sub is not None:
                slices[lbl] = sub
    print(f"[eval] slices: { {k: len(v) for k, v in slices.items()} }")
    return slices


def find_last_checkpoint(output_dir, prev_run_dir):
    for d in (output_dir, prev_run_dir):
        if d and os.path.isdir(d):
            ckpt = get_last_checkpoint(d)
            if ckpt is not None:
                return ckpt
    return None


# ============================================================================
if __name__ == "__main__":
    # ---------------------------- CONFIG (edit me) ----------------------------
    OLD_CHECKPOINT = "/kaggle/input/models/harshadesaraju99/telugu-image-ctc-encoder/transformers/default/1/model.safetensors"  # OLD 128-frame ckpt (weights)
    OUTPUT_DIR = "/kaggle/working/ctc-encoder-2048"
    PREV_RUN_DIR = None            # prior 2048-run output re-mounted read-only, for 12h resume

    DATASET_NAME = "harsha-desaraju/sample-dataset"
    DATASET_SPLIT = "train"          # NORMAL (downloaded, map-style) dataset
    EVAL_SPLIT = "validation"        # small held-out split for the eval slices
    IMAGE_COLUMN = "image"
    TEXT_COLUMN = "text"
    SOURCE_COLUMN = "text_source"  # natural / random, for eval slices
    WIDTH_COLUMN = "image_width"   # used to split short/long eval slices (falls back to measuring)
    VOCAB_FILE = "/kaggle/input/datasets/harshadesaraju99/telugu-tokenizer-vocab/telugu-vocab.json"

    # frames / widths
    DOWNSAMPLE = 8
    BASE_FRAMES = 128              # old max (rows loaded from ckpt)
    MAX_FRAMES = 256               # new max (T up to 256 -> 2048px)
    MAX_IMAGE_WIDTH = 2048
    BASE_WIDTH = BASE_FRAMES * DOWNSAMPLE  # 1024 (short/long boundary)

    # batching — one per-device batch for all widths (memory is sufficient; lower if not)
    PER_DEVICE_BATCH = 64
    GRAD_ACCUM = 2                 # global batch = PER_DEVICE_BATCH * GRAD_ACCUM * #GPUs

    # optimizer / schedule
    LR_REST = 1e-4                 # everything except the new pos rows
    LR_EXT = 3e-4                  # new pos rows (128-255)
    MIN_LR = 1e-5
    WARMUP_STEPS = 1500
    WEIGHT_DECAY = 0.05
    BETAS = (0.9, 0.98)
    # NORMAL dataset has a known length, so training is epoch-based: Trainer computes
    # the total optimizer steps (accounting for #GPUs / grad-accum) and hands them to
    # the cosine scheduler (T_max) via create_scheduler. No manual step math needed.
    EPOCHS = 5                     # complete passes over the train split

    # eval / io
    EVAL_SLICE_CAP = 1500          # cap each eval slice for speed
    EVAL_BATCH = 32                # smaller eval batch (long slices can be up to 2048px)
    SAVE_EVAL_STEPS = 2000
    SEED = 42
    # --------------------------------------------------------------------------

    set_seed(SEED)  # reproducible augmentation
    print(f"[cfg] epochs={EPOCHS} per_device_batch={PER_DEVICE_BATCH} grad_accum={GRAD_ACCUM}")

    tokenizer = TeluguGraphemeTokenizer(vocab_file=VOCAB_FILE)
    blank_id = len(tokenizer)
    print(f"[tokenizer] vocab {len(tokenizer)} | blank {blank_id} | classes {len(tokenizer)+1}")

    cfg = CTCEncoderConfig(
        image_height=64, max_image_width=MAX_IMAGE_WIDTH, downsample=DOWNSAMPLE,
        base_frames=BASE_FRAMES, max_frames=MAX_FRAMES, vocab_size=len(tokenizer),
    )
    model = ImageEncoderCTC(cfg)

    # ---- Resume logic: Trainer checkpoint (this run) vs first-launch old-ckpt ----
    last_checkpoint = find_last_checkpoint(OUTPUT_DIR, PREV_RUN_DIR)
    if last_checkpoint is None:
        extend_from_checkpoint(model, OLD_CHECKPOINT)   # first launch: seed from old 128-frame ckpt
    else:
        print(f"[resume] Trainer checkpoint -> {last_checkpoint} (state restored by Trainer)")
        if PREV_RUN_DIR and last_checkpoint.startswith(str(PREV_RUN_DIR)):
            import shutil
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            dst = os.path.join(OUTPUT_DIR, os.path.basename(last_checkpoint))
            if not os.path.isdir(dst):
                shutil.copytree(last_checkpoint, dst)
            last_checkpoint = dst
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params {n_params:,} (~{n_params/1e6:.1f}M) | max_frames {cfg.max_frames}")

    # ---- Data: NORMAL (downloaded, map-style) train + validation ----
    train_preprocessor = ImagePreprocessor(tokenizer, IMAGE_COLUMN, TEXT_COLUMN,
                                           augment_fn=make_weighted_augmenter())  # train aug
    eval_preprocessor = ImagePreprocessor(tokenizer, IMAGE_COLUMN, TEXT_COLUMN, augment_fn=None)

    # Train: whole split downloaded to disk, transformed lazily via with_transform.
    # Trainer's sampler shuffles each epoch, so no manual shuffle is needed.
    configs = ['train_0000', 'train_0001', 'train_0002', 'train_0003', 'train_0004', 'train_0005', 'train_0006', 'train_0007',
               # 'train_0008', 'train_0009'
            ]
    train_hf = []
    for config in configs:
        tds = load_dataset(DATASET_NAME, config)[DATASET_SPLIT]
        train_hf.append(tds)
    train_hf = concatenate_datasets(train_hf, axis=0)

    # train_hf = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    train_ds = train_hf.with_transform(train_preprocessor)
    print(f"[data] {DATASET_NAME}:{DATASET_SPLIT} -> {len(train_hf)} rows; cols {train_hf.column_names}")

    # Validation: small; materialize rows and build the 4 eval slices.
    val_ds = load_dataset(DATASET_NAME, EVAL_SPLIT)[EVAL_SPLIT]
    val_rows = list(val_ds)
    print(f"[data] validation {DATASET_NAME}:{EVAL_SPLIT} -> {len(val_rows)} rows")
    eval_slices = build_eval_slices(val_rows, IMAGE_COLUMN, TEXT_COLUMN, SOURCE_COLUMN,
                                    WIDTH_COLUMN, eval_preprocessor, BASE_WIDTH, EVAL_SLICE_CAP, SEED)

    # ---- Optimizer (fresh, 2 LR tiers); scheduler built by CTCTrainer ----
    optimizer = build_optimizer(model, LR_REST, LR_EXT, WEIGHT_DECAY, BETAS)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,             # NORMAL dataset -> epoch-based; Trainer computes total steps
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=EVAL_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_grad_norm=1.0,                   # UNCHANGED grad clip
        fp16=torch.cuda.is_available(),      # T4 -> fp16 + GradScaler (UNCHANGED)
        save_strategy="steps",
        save_steps=SAVE_EVAL_STEPS,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=SAVE_EVAL_STEPS,
        load_best_model_at_end=False,
        logging_strategy="steps",
        logging_steps=50,
        logging_first_step=True,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=4,
        dataloader_persistent_workers=True,
        remove_unused_columns=False,
        label_names=["labels"],
        ddp_find_unused_parameters=False,
        # On resume, Trainer replays/skips already-seen batches to restore data order.
        # For a very large train split that skip is slow; set ignore_data_skip=True to
        # jump straight to the saved step (at the cost of exact data-order replay).
        report_to="wandb",
        run_name="ctc-encoder-stage-2",
        seed=SEED,
    )

    trainer = CTCTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_slices,            # dict -> Trainer logs eval_<slice>_cer / _nonblank
        data_collator=CTCCollator(DOWNSAMPLE),
        compute_metrics=make_compute_metrics(tokenizer, blank_id),
        optimizers=(optimizer, None),        # fresh optimizer; scheduler via create_scheduler
        callbacks=[GradNormAlert(threshold=10.0)],
        warmup_steps=WARMUP_STEPS,
        min_lr=MIN_LR,
    )

    trainer.train(resume_from_checkpoint=last_checkpoint)

    trainer.save_model(OUTPUT_DIR)
    if trainer.is_world_process_zero():
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "final_model.pt"))
        print(f"[done] saved -> {OUTPUT_DIR}")
