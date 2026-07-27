"""
Stage A training: conv-stem CTC image encoder (self-contained / Kaggle-ready).
==============================================================================

Trains the ``ImageEncoderCTC`` model (see ``ctc_encoder.py``, §2 of
``telugu_ocr_training_spec.md``) from scratch with the CTC objective on
(line-image, text) pairs, orchestrated by the HuggingFace ``Trainer`` for free
fp16 / DDP / checkpointing / logging.

Like the other training scripts in this repo (``single_train_file.py``,
``train_stage_*.py``) this file **inlines** the model, tokenizer, preprocessing,
collator and metric so it can be uploaded and run standalone on Kaggle / Colab.
When you change core model logic in ``ctc_encoder.py``, update the inlined copy
below too.

Data assumptions (from the setup discussion):
  * Dataset ``harsha-desaraju/telugu-line-text-image`` with columns
    ``image`` (PIL) + ``text`` (str).
  * Images are already normalized to height 64 and width a multiple of 8, so the
    preprocessor only grayscales + tensorizes + normalizes (no resize/pad).
  * CTC frames per sample: T = width // 8; passed as ``input_lengths``.

Eval reports greedy CTC CER (§5 convergence criteria) plus ``eval_loss``.

Run (single GPU / CPU):
    python3 -m src.image_encoder.train_ctc_encoder

Run (multi-GPU DDP):
    torchrun --nproc_per_node=<NUM_GPUS> -m src.image_encoder.train_ctc_encoder
"""

from __future__ import annotations

import os
import json
import regex
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, PreTrainedTokenizer
from transformers.trainer_utils import get_last_checkpoint


# ============================================================================
# Model (inlined copy of ctc_encoder.py — keep in sync)
# ============================================================================
@dataclass
class CTCEncoderConfig:
    # ---- input ----
    image_height: int = 64
    max_image_width: int = 1024
    downsample: int = 8             # conv-stem width reduction: T = W // downsample
    max_frames: int = 128           # = max_image_width // downsample (pos-emb length)

    # ---- conv stem ----
    stem_channels: tuple = (32, 64, 128, 256, 320, 384)
    num_groups: int = 32

    # ---- transformer encoder ----
    embed_dim: int = 384
    num_layers: int = 10
    num_heads: int = 8
    mlp_dim: int = 1536
    dropout: float = 0.1
    drop_path_rate: float = 0.1

    # ---- CTC head ----
    vocab_size: int = 2048          # grapheme classes; blank appended at this index


class ConvBlock(nn.Module):
    """Conv → GroupNorm → GELU (one row of the §2.1 stem table)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, num_groups):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        groups = num_groups if out_channels % num_groups == 0 else 1
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ConvStem(nn.Module):
    """Six-block convolutional tokenizer: (B,1,64,W) -> (B, T=W/8, 384)."""

    def __init__(self, cfg: CTCEncoderConfig):
        super().__init__()
        c = cfg.stem_channels
        assert len(c) == 6, "stem expects 6 conv blocks (§2.1)"
        assert c[-1] == cfg.embed_dim, "last stem channel must equal embed_dim"

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

    def forward(self, x):
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

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
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

    def forward(self, x, key_padding_mask=None):
        norm_x = self.layer_norm1(x)
        attn_out, _ = self.attention(
            norm_x, norm_x, norm_x, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.layer_norm2(x)))
        return x


LABEL_PAD_ID = -100  # ignore marker for padded target positions (HF convention)


class ImageEncoderCTC(nn.Module):
    """Conv stem → transformer encoder → CTC head (~20M params).

    HF ``Trainer``-compatible: computes the CTC loss internally when ``labels``
    are given and returns ``{"loss", "logits"}`` where ``logits`` are the greedy
    per-frame token ids (padded frames forced to blank) — small ints, cheap to
    gather for CER. NOTE: unlike ctc_encoder.py it does NOT return the full
    (B,T,C) log-probs, so eval doesn't OOM accumulating them.
    """

    def __init__(self, cfg: CTCEncoderConfig, label_pad_id: int = LABEL_PAD_ID):
        super().__init__()
        self.cfg = cfg
        self.blank_id = cfg.vocab_size
        self.num_classes = cfg.vocab_size + 1
        self.label_pad_id = label_pad_id

        self.stem = ConvStem(cfg)
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.max_frames, cfg.embed_dim))
        self.dropout = nn.Dropout(cfg.dropout)

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

    def forward(self, images, input_lengths=None, labels=None, label_lengths=None):
        feats = self.stem(images)                       # (B, T, D)
        B, T, D = feats.shape
        assert T <= self.cfg.max_frames, f"T={T} exceeds max_frames={self.cfg.max_frames}"
        feats = feats + self.pos_embed[:, :T, :]
        feats = self.dropout(feats)

        if input_lengths is None:
            input_lengths = torch.full((B,), T, dtype=torch.long, device=feats.device)

        frame_idx = torch.arange(T, device=feats.device).unsqueeze(0)   # (1, T)
        key_padding_mask = frame_idx >= input_lengths.unsqueeze(1)       # (B, T) True = pad

        for block in self.blocks:
            feats = block(feats, key_padding_mask)
        feats = self.layer_norm(feats)
        logits = self.ctc_head(feats)                   # (B, T, C)

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
            loss = self.ctc_loss(
                log_probs.permute(1, 0, 2), targets, input_lengths, label_lengths
            )

        # Greedy per-frame ids with padded frames forced to blank (clean decode).
        pred_ids = log_probs.argmax(dim=-1)             # (B, T)
        pred_ids = pred_ids.masked_fill(key_padding_mask, self.blank_id)

        return {"loss": loss, "logits": pred_ids}


# ============================================================================
# Grapheme tokenizer (inlined copy — see text_decoder/grapheme_tokenizer)
# ============================================================================
class TeluguGraphemeTokenizer(PreTrainedTokenizer):
    """HuggingFace-compatible grapheme-cluster tokenizer for Telugu."""

    vocab_files_names = {"vocab_file": "vocab.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        vocab_list: Optional[list[str]] = None,
        add_bos_token: bool = True,
        add_eos_token: bool = True,
        pad_token: str = "[PAD]",
        unk_token: str = "[UNK]",
        bos_token: str = "[BOS]",
        eos_token: str = "[EOS]",
        mask_token: str = "[MASK]",
        **kwargs,
    ):
        vocab = {}
        if vocab_file is not None:
            with open(vocab_file, encoding="utf-8") as f:
                vocab = json.load(f)
        if not vocab and vocab_list is not None:
            for grapheme in vocab_list:
                vocab[grapheme] = len(vocab)
        if not vocab:
            raise AssertionError("Either `vocab_file` or `vocab_list` has to be given.")

        self.SPECIAL_TOKENS_LIST = [pad_token, unk_token, bos_token, eos_token, mask_token]
        self.grapheme_pattern = regex.compile(r"\X")

        for tok in self.SPECIAL_TOKENS_LIST:
            if tok not in vocab:
                vocab[tok] = len(vocab)

        self.vocab = vocab  # must be BEFORE super().__init__()
        self._inv_vocab = {v: k for k, v in vocab.items()}
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.UNK = unk_token

        super().__init__(
            pad_token=pad_token,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            mask_token=mask_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            padding_side="right",
            model_max_length=4096,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.vocab)

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        tokens = []
        for grapheme in self.grapheme_pattern.findall(text):
            if grapheme in self.vocab:
                tokens.append(grapheme)
            else:
                for codepoint in grapheme:
                    tokens.append(codepoint)
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab.get(self.UNK, 1))

    def _convert_id_to_token(self, index: int) -> str:
        return self._inv_vocab.get(index, self.UNK)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        cleaned = [t for t in tokens if t not in self.SPECIAL_TOKENS_LIST]
        return "".join(cleaned)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos = [self.bos_token_id] if self.add_bos_token else []
        eos = [self.eos_token_id] if self.add_eos_token else []
        out = bos + token_ids_0 + eos
        if token_ids_1 is not None:
            out += bos + token_ids_1 + eos
        return out

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        if already_has_special_tokens:
            return super().get_special_tokens_mask(token_ids_0, token_ids_1, already_has_special_tokens=True)
        bos = [1] if self.add_bos_token else []
        eos = [1] if self.add_eos_token else []
        res = bos + [0] * len(token_ids_0) + eos
        if token_ids_1 is not None:
            res += bos + [0] * len(token_ids_1) + eos
        return res

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        os.makedirs(save_directory, exist_ok=True)
        fname = (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        vocab_path = os.path.join(save_directory, fname)
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        return (vocab_path,)


# ============================================================================
# Image augmentation (train split only) — ported from train_stage_2.py
# ============================================================================
# Target-aware, cost-aware augmentation sampler. Each combo family maps a real
# print-degradation (roller/drum streaks, bleed-through, ink bleed, low-ink
# lines, jpeg, noise, motion blur) to 4 severities. A sample is left CLEAN with
# probability P_CLEAN; otherwise a family is drawn ∝ AUG_UTILITY / (1+LAM·time)
# and a uniform severity is applied. Augmentation runs on the 64px crop (cheap)
# and never raises — a bad combo falls back to the clean image.
#
# NOTE (spec §4.1) prefers cheap load-time augs with heavy degradations baked in
# at render time; this ports train_stage_2.py's full pipeline as requested — the
# 64px crop + P_CLEAN keep the CPU cost in check. Tune P_CLEAN / LAM if the
# dataloader becomes the bottleneck (spec §10: GPU util < ~70%).
import random
import numpy as np
import cv2

from augraphy import (
    InkBleed, InkMottling, BleedThrough,
    LowInkRandomLines, BrightnessTexturize,
    DirtyDrum, DirtyRollers, SubtleNoise, Jpeg,
)


def set_seed(seed):
    """Seed the two PRNGs used here: `random` (combo choice) and `np.random`
    (custom effects + augraphy internals). Some augraphy effects still carry
    internal randomness, so identical combos may still differ run-to-run."""
    random.seed(seed)
    np.random.seed(seed)


# ---- custom effects (NumPy / OpenCV) --------------------------------------
def motion_blur(img: np.ndarray, degree: int = 7) -> np.ndarray:
    """Horizontal motion blur via a normalized 1-row averaging kernel."""
    kernel = np.zeros((degree, degree), dtype=np.float32)
    kernel[(degree - 1) // 2, :] = 1.0
    kernel /= degree
    return cv2.filter2D(img, -1, kernel)


def gaussian_noise(img: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """Additive zero-mean Gaussian noise with standard deviation `sigma`."""
    noise = np.random.normal(0.0, sigma, img.shape)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def salt_pepper(img: np.ndarray, amount: float = 0.003) -> np.ndarray:
    """Salt-and-pepper noise; `amount` = fraction of pixels hit per polarity."""
    out = img.copy()
    h, w = img.shape[:2]
    num = int(amount * h * w)
    rows = np.random.randint(0, h, num); cols = np.random.randint(0, w, num)
    out[rows, cols] = 255  # salt
    rows = np.random.randint(0, h, num); cols = np.random.randint(0, w, num)
    out[rows, cols] = 0    # pepper
    return out


# ---- parameter combos: subtle (col 1) -> heavy (col 4) --------------------
# augraphy objects are PRE-BUILT once (construction, esp. DirtyRollers, is the
# dominant per-sample cost); reuse still re-samples internal randomness.
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

# Target-relevance weights in [0, 1]. Higher = looks more like degraded print.
AUG_UTILITY = {
    "DirtyDrum": 1.0, "BleedThrough": 0.9, "InkBleed": 0.9,
    "LowInkRandomLines": 0.8, "Jpeg": 0.7, "BrightnessTexturize": 0.6,
    "InkMottling": 0.6, "SubtleNoise": 0.5, "GaussianNoise": 0.5,
    "MotionBlur": 0.3, "SaltPepper": 0.3, "DirtyRollers": 5.0
}
# Avg ms/image at FULL res; re-measure at 64px if you set LAM > 0.
AUG_TIME_MS = {
    "InkBleed": 1.3, "InkMottling": 0.7, "BleedThrough": 5.4, "LowInkRandomLines": 0.2,
    "BrightnessTexturize": 0.8, "DirtyDrum": 3.9, "DirtyRollers": 16.6, "SubtleNoise": 0.2,
    "Jpeg": 0.3, "MotionBlur": 0.1, "GaussianNoise": 0.6, "SaltPepper": 0.0,
}
assert set(AUG_UTILITY) == set(COMBOS) == set(AUG_TIME_MS), "keys must match COMBOS"

P_CLEAN = 0.50   # fraction of samples left clean (crisp-text regularizer + CPU trim)
LAM = 1.0        # cost-penalty strength per ms; 0 = utility-only. Raise if CPU-bound.


def _build_aug_distribution(utility, times, lam):
    names = list(COMBOS.keys())
    raw = np.array([utility[n] / (1.0 + lam * times[n]) for n in names], dtype=np.float64)
    return names, list(raw / raw.sum())


_AUG_NAMES, _AUG_PROBS = _build_aug_distribution(AUG_UTILITY, AUG_TIME_MS, LAM)


def make_weighted_augmenter(p_clean=P_CLEAN):
    """Return augment(np_uint8 HxWx3) -> np_uint8 HxWx3. Never raises.

    Uses python `random` (torch seeds it per DataLoader worker) so combo/severity
    choices decorrelate across workers. If augraphy's internal numpy randomness
    correlates across workers, add a worker_init_fn that reseeds numpy:
        def _wi(wid): np.random.seed(torch.initial_seed() % 2**32)
    """
    names, weights = _AUG_NAMES, _AUG_PROBS

    def augment(img):
        if random.random() < p_clean:
            return img
        name = random.choices(names, weights=weights, k=1)[0]
        _label, applier = random.choice(COMBOS[name])  # uniform over the 4 severities
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
    """`datasets.with_transform` hook: grayscale + to-tensor + normalize.

    No resize/pad — the source images are already height 64 with width a
    multiple of 8. Returns already-tokenized grapheme ids so tokenization runs
    in the dataloader workers.
    """

    def __init__(self, tokenizer, image_col: str, text_col: str, augment_fn=None):
        self.tokenizer = tokenizer
        self.image_col = image_col
        self.text_col = text_col
        # augment_fn (train split only): runs the augraphy + custom-effect
        # sampler on the RGB 64px crop before grayscale + tensor. eval passes
        # None so the held-out split stays fixed/clean.
        self.augment_fn = augment_fn
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),                       # (1,H,W) float in [0,1]
            transforms.Normalize(mean=[0.5], std=[0.5]),  # -> [-1, 1]
        ])

    def _transform_image(self, img: Image.Image) -> torch.Tensor:
        if self.augment_fn is not None:
            arr = np.array(img.convert("RGB"))           # (H, W, 3) uint8 for augraphy
            arr = self.augment_fn(arr)
            img = Image.fromarray(np.asarray(arr, dtype=np.uint8))
        return self.to_tensor(img.convert("L"))          # grayscale for the model

    def __call__(self, batch: dict) -> dict:
        images = [self._transform_image(img) for img in batch[self.image_col]]
        target_ids = [
            self.tokenizer(t, add_special_tokens=False)["input_ids"]
            for t in batch[self.text_col]
        ]
        return {"line_image": images, "target_ids": target_ids}


class CTCCollator:
    """Right-pads variable-width line images and builds Trainer inputs.

    Returns:
      images        (B, 1, 64, Wmax)  padded line images (fill 1.0 = white)
      input_lengths (B,)              valid frame count = W_i // downsample (CTC T)
      labels        (B, Smax)         grapheme ids, padded with LABEL_PAD_ID
    """

    def __init__(self, downsample: int):
        self.downsample = downsample

    def __call__(self, batch):
        imgs = [s["line_image"] for s in batch]
        B = len(imgs)
        C, H = imgs[0].shape[0], imgs[0].shape[1]
        widths = [im.shape[2] for im in imgs]
        W_max = max(widths)

        # Padded background is white; normalized (255/255 - 0.5)/0.5 = 1.0.
        images = imgs[0].new_full((B, C, H, W_max), 1.0)
        input_lengths = torch.empty(B, dtype=torch.long)
        for i, im in enumerate(imgs):
            w = im.shape[2]
            images[i, :, :, :w] = im
            input_lengths[i] = w // self.downsample

        max_s = max(max(len(s["target_ids"]) for s in batch), 1)
        labels = torch.full((B, max_s), LABEL_PAD_ID, dtype=torch.long)
        for i, s in enumerate(batch):
            ids = s["target_ids"]
            if ids:
                labels[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

        return {"images": images, "input_lengths": input_lengths, "labels": labels}


# ============================================================================
# CER metric
# ============================================================================
def levenshtein(a: list, b: list) -> int:
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def corpus_cer(preds: list[str], refs: list[str]) -> float:
    total_edits, total_chars = 0, 0
    for pred, ref in zip(preds, refs):
        total_edits += levenshtein(list(pred), list(ref))
        total_chars += len(ref)
    return total_edits / max(total_chars, 1)


def make_compute_metrics(tokenizer, blank_id: int):
    """Greedy-CTC-collapse the gathered per-frame ids and compute corpus CER."""

    def _collapse(row) -> list[int]:
        ids, prev = [], None
        for t in row:
            t = int(t)
            if t != prev and t != blank_id and t != LABEL_PAD_ID:
                ids.append(t)
            prev = t
        return ids

    def compute_metrics(eval_pred):
        preds = eval_pred.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        labels = eval_pred.label_ids

        pred_texts = [tokenizer.decode(_collapse(row), skip_special_tokens=True) for row in preds]
        ref_texts = [
            tokenizer.decode([int(t) for t in row if int(t) != LABEL_PAD_ID], skip_special_tokens=True)
            for row in labels
        ]
        return {"cer": corpus_cer(pred_texts, ref_texts)}

    return compute_metrics


# ============================================================================
# Train
# ============================================================================
if __name__ == "__main__":
    # ---- Stage A recipe (§5) ---------------------------------------------
    PER_DEVICE_BATCH = 64  # 64/GPU × 2 T4 (DDP) = global 128
    GRAD_ACCUM = 1
    MAX_STEPS = 300_000  # ~15 epochs over 2.5M lines @ global batch 128
    WARMUP_STEPS = 5_000
    LEARNING_RATE = 3e-4
    MIN_LR = 3e-5                   # cosine floor (§5: cosine to 3e-5, not to 0)
    WEIGHT_DECAY = 0.05
    print(f"Per-device batch: {PER_DEVICE_BATCH} | max steps: {MAX_STEPS}")

    # TEST_SIZE = 0.01
    EVAL_SUBSET_SIZE = 3000  # keep greedy-CER eval fast; None = full split

    DOWNSAMPLE = 8
    IMAGE_COLUMN = "image"
    TEXT_COLUMN = "text"
    DATASET_NAME = "harsha-desaraju/sample-dataset"

    VOCAB_FILE = "/kaggle/input/datasets/harshadesaraju1999/telugu-tokenizer-vocab/telugu-vocab.json"

    # New checkpoints written here (writable on Kaggle: /kaggle/working/...).
    OUTPUT_DIR = "/kaggle/working/ctc-encoder"
    # A previous session's committed output re-added as a read-only input.
    # None for the first run; set to the prior session's path for continuations.
    PREV_RUN_DIR = None  # e.g. "/kaggle/input/ctc-encoder-prev/ctc-encoder"

    # ---- Tokenizer -------------------------------------------------------
    tokenizer = TeluguGraphemeTokenizer(vocab_file=VOCAB_FILE)
    blank_id = len(tokenizer)  # dedicated CTC blank, just outside the vocab
    print(f"Vocab size: {len(tokenizer)} | blank id: {blank_id} | classes: {len(tokenizer) + 1}")

    # ---- Data ------------------------------------------------------------
    train_ds = load_dataset(DATASET_NAME, columns=[IMAGE_COLUMN, TEXT_COLUMN], split="train")
    test_ds = load_dataset(DATASET_NAME, columns=[IMAGE_COLUMN, TEXT_COLUMN], split="validation")

    if EVAL_SUBSET_SIZE is not None and EVAL_SUBSET_SIZE < len(test_ds):
        test_ds = test_ds.shuffle(seed=42).select(range(EVAL_SUBSET_SIZE))

    # Reproducible augmentation choices (seeds `random` + `np.random`).
    set_seed(42)
    # Train split is augmented on the fly; eval split stays clean/fixed.
    train_preprocessor = ImagePreprocessor(
        tokenizer, IMAGE_COLUMN, TEXT_COLUMN, augment_fn=make_weighted_augmenter()
    )
    eval_preprocessor = ImagePreprocessor(
        tokenizer, IMAGE_COLUMN, TEXT_COLUMN, augment_fn=None
    )
    train_ds = train_ds.with_transform(train_preprocessor)
    test_ds = test_ds.with_transform(eval_preprocessor)

    # ---- Model (FROM SCRATCH) -------------------------------------------
    # Blank lands at index len(tokenizer): set vocab_size to the tokenizer length
    # so num_classes = len(tokenizer) + 1.
    cfg = CTCEncoderConfig(
        image_height=64,
        max_image_width=1024,
        downsample=DOWNSAMPLE,
        max_frames=1024 // DOWNSAMPLE,
        vocab_size=len(tokenizer),
    )
    model = ImageEncoderCTC(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,} (~{n_params / 1e6:.1f}M)")


    # ---- Resume logic ----------------------------------------------------
    def find_last_checkpoint():
        for d in (OUTPUT_DIR, PREV_RUN_DIR):
            if d and os.path.isdir(d):
                ckpt = get_last_checkpoint(d)
                if ckpt is not None:
                    return ckpt
        return None


    last_checkpoint = find_last_checkpoint()
    print(f"Resuming from: {last_checkpoint}" if last_checkpoint else "No checkpoint — starting fresh.")

    # If the checkpoint is in the read-only PREV_RUN_DIR, copy it into the
    # writable OUTPUT_DIR so the Trainer can keep writing / continue its state.
    if last_checkpoint is not None and last_checkpoint.startswith(str(PREV_RUN_DIR or "")):
        import shutil

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        dst = os.path.join(OUTPUT_DIR, os.path.basename(last_checkpoint))
        if not os.path.isdir(dst):
            print(f"Copying checkpoint into writable dir: {dst}")
            shutil.copytree(last_checkpoint, dst)
        last_checkpoint = dst

    # ---- TrainingArguments ----------------------------------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        max_steps=MAX_STEPS,

        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,

        optim="adamw_torch_fused",
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        adam_beta1=0.9,
        adam_beta2=0.98,
        warmup_steps=WARMUP_STEPS,
        # Cosine decay from LEARNING_RATE down to a MIN_LR floor (not to 0).
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": MIN_LR},
        max_grad_norm=1.0,            # fp16 + CTC stability

        # --- Checkpointing: frequent, to survive the 12h Kaggle wall (§9) ---
        save_strategy="steps",
        save_steps=2000,
        save_total_limit=2,
        load_best_model_at_end=False,  # pure continuation; don't reload "best"

        # --- Eval: greedy CER + eval_loss ---
        eval_strategy="steps",
        eval_steps=2000,

        # Precision / perf (T4 = Turing -> fp16, no bf16)
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=4,
        dataloader_persistent_workers=True,

        remove_unused_columns=False,   # our collator builds custom tensors
        label_names=["labels"],        # so Trainer gathers labels for the metric
        ddp_find_unused_parameters=False,

        logging_strategy="steps",
        logging_steps=50,
        logging_first_step=True,
        report_to="wandb",              # "wandb" to enable W&B
        run_name="ctc-encoder-stage-1",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=CTCCollator(DOWNSAMPLE),
        compute_metrics=make_compute_metrics(tokenizer, blank_id),
    )

    trainer.train(resume_from_checkpoint=last_checkpoint)

    # Final weights (only reached if the full run completes in one session).
    trainer.save_model(OUTPUT_DIR)                              # HF-style, resumable
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/final_model.pt")  # raw state_dict
