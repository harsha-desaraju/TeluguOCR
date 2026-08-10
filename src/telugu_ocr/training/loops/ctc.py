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
  * Positional embedding is a SINGLE ``pos_embed`` Parameter (1, max_frames, D),
    sliced to T in forward. It was split into ``pos_embed`` + ``pos_embed_ext`` during
    the 1024->2048 extension so the new rows could take a higher LR; now that training
    is done it is one table, and ``load_checkpoint`` merges an old split checkpoint into
    it row-for-row (no weight changes, no interpolation).
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

Run (single GPU / CPU):   python3 -m src.telugu_ocr.models.image_encoder_normal
Run (multi-GPU DDP):      torchrun --nproc_per_node=<N> -m src.telugu_ocr.models.image_encoder_normal
"""

from __future__ import annotations

import os
import sys
import math
import json
import regex
import random

# ============================================================================
# CPU-thread limiting for the augmentation DataLoader workers.
# ============================================================================
# Under a multi-worker DataLoader — and especially under DDP, where it is
# (#ranks x dataloader_num_workers) loader processes sharing a few vCPUs — OpenCV,
# OpenMP/BLAS and numba EACH spin up their own thread pool per worker. The workers then
# oversubscribe the cores many-to-many and augmentation gets SLOWER, not faster. Pinning
# every worker's math libraries to a single thread removes that contention so the speed
# comes from running many workers in parallel, one core each.
#
# These env vars size their thread pools at import time, so they MUST be set before numpy
# / torch / cv2 are imported below (cv2 itself is pinned at runtime in the augmentation
# section). Kaggle is Linux, so the DataLoader forks its workers and they inherit this.
#
# Flip LIMIT_AUG_THREADS to False to disable the whole thing if you measure it making
# throughput WORSE (e.g. on a box with many idle cores where per-op multithreading helped).
LIMIT_AUG_THREADS = True
if LIMIT_AUG_THREADS:
    for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[_thread_var] = "1"

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
    dropout: float = 0.05            # UNCHANGED (spec)
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

    The learned positional embedding is a SINGLE ``pos_embed`` (1, max_frames, D),
    sliced to T in forward. (It was previously split into ``pos_embed`` + ``pos_embed_ext``
    so the extension rows could take a higher LR while extending the context window; now
    that training is finished it is one table. ``load_checkpoint`` merges an old split
    checkpoint into it row-for-row, so no weights change.)
    """

    def __init__(self, cfg: CTCEncoderConfig, label_pad_id: int = LABEL_PAD_ID):
        super().__init__()
        self.cfg = cfg
        self.blank_id = cfg.vocab_size           # UNCHANGED blank index
        self.num_classes = cfg.vocab_size + 1
        self.label_pad_id = label_pad_id

        self.stem = ConvStem(cfg)

        # Single learned positional embedding of the full length. Training is done, so
        # the base/extension split (two Parameters, concatenated in forward) has served
        # its purpose and is collapsed into one table. `load_checkpoint` merges an old
        # split checkpoint into this table, so the trained weights transfer exactly.
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.max_frames, cfg.embed_dim))

        self.dropout = nn.Dropout(cfg.dropout)
        dpr = torch.linspace(0.0, cfg.drop_path_rate, cfg.num_layers).tolist()
        self.blocks = nn.ModuleList([TransformerBlock(cfg, dpr[i]) for i in range(cfg.num_layers)])
        self.layer_norm = nn.LayerNorm(cfg.embed_dim)
        self.ctc_head = nn.Linear(cfg.embed_dim, self.num_classes)
        self.ctc_loss = nn.CTCLoss(blank=self.blank_id, zero_infinity=True)  # UNCHANGED

        self._init_weights()

    def _init_weights(self):
        # Same std as the original init; the whole table is overwritten by the
        # checkpoint load in load_checkpoint.
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _pos(self, T):
        return self.pos_embed[:, :T, :]

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
# Checkpoint loader — merges a split (pos_embed + pos_embed_ext) checkpoint into the
# single-table model, and also accepts an already-merged or an old 128-frame checkpoint.
# ============================================================================
def load_checkpoint(model: ImageEncoderCTC, ckpt_path: str):
    """Load a checkpoint into the single-``pos_embed`` model. Handles three shapes:

      * split      ``pos_embed`` (1, base, D) + ``pos_embed_ext`` (1, max-base, D)
                   -> concatenated into ``pos_embed`` (1, max, D). This is the trained
                   2048 checkpoint; the concat reproduces the old forward exactly, so no
                   weight changes.
      * merged     ``pos_embed`` (1, max_frames, D)                -> loaded directly.
      * old 128    ``pos_embed`` (1, base_frames, D), no extension -> copied into rows
                   0:base_frames; rows base_frames:max_frames stay at fresh init (the
                   pre-training extension-seed path).

    Everything else loads strictly.
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

    D, max_frames, base = model.cfg.embed_dim, model.cfg.max_frames, model.cfg.base_frames
    pos = state["pos_embed"]
    ext = state.pop("pos_embed_ext", None)
    seeded_from_old = False

    if ext is not None:                                   # split checkpoint -> merge
        assert tuple(pos.shape)[1] + tuple(ext.shape)[1] == max_frames, (
            f"split rows {tuple(pos.shape)[1]}+{tuple(ext.shape)[1]} != max_frames={max_frames}")
        state["pos_embed"] = torch.cat([pos, ext], dim=1)
    elif tuple(pos.shape) == (1, max_frames, D):          # already merged
        pass
    elif tuple(pos.shape) == (1, base, D):                # old pre-extension ckpt
        merged = model.pos_embed.detach().clone()         # keep fresh init on rows base:
        merged[:, :base, :] = pos
        state["pos_embed"] = merged
        seeded_from_old = True
    else:
        raise AssertionError(
            f"pos_embed {tuple(pos.shape)} matches neither merged (1,{max_frames},{D}) "
            f"nor split-base / old (1,{base},{D})")

    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing and not unexpected, f"load mismatch: missing={missing} unexpected={unexpected}"
    if seeded_from_old:
        print(f"[load] old 128-frame ckpt: pos_embed[0:{base}] copied, "
              f"[{base}:{max_frames}] fresh-init.")
    else:
        print(f"[load] {'merged split' if ext is not None else 'single-table'} "
              f"checkpoint into pos_embed (1,{max_frames},{D}).")
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
# Image augmentation — INLINED from src/telugu_ocr/data/augment.py
# ============================================================================
# CHANGED(2048): the sweep-style augmenter (one effect per image, chosen by a
# utility/time weighting) is replaced by the calibrated COMPOSED degradation pipeline
# from src/telugu_ocr/data/augment.py — tuned so the augmented
# distribution CONTAINS the real PDF-crop distribution measured at h=64. It composes
# many effects per sample (bilevel -> ink -> paper/grime -> tone -> crop artefacts ->
# geometry -> noise -> resample -> compression -> auto-levels) with a two-sided
# legibility backstop. Entry point: make_augmenter(p_clean) -> augment(uint8 HxW|HxWx3)
# -> same shape. Fed the stored 64px-tall crop by ImagePreprocessor (the scale-dependent
# params — blur sigma, band widths, speckle size — are tuned for that height). The
# preview __main__ and its matplotlib/LADDERS helpers are intentionally not inlined.
import cv2  # noqa: E402
from augraphy import (  # noqa: E402
    BadPhotoCopy,
    BleedThrough,
    BrightnessTexturize,
    DirtyDrum,
    InkBleed,
    InkMottling,
    Jpeg,
    Letterpress,
    NoiseTexturize,
    SubtleNoise,
)

# Pin OpenCV to one thread (runtime companion to the env vars set at the top of the file).
# Forked DataLoader workers inherit this, so each augments on a single core instead of
# every worker's cv2 fighting for all of them. Gated by the same LIMIT_AUG_THREADS toggle.
if LIMIT_AUG_THREADS:
    cv2.setNumThreads(1)


def set_seed(seed):
    """Seed both PRNGs used here: `random` (which effects fire, and how hard) and
    `np.random` (the custom effects plus Augraphy's internals). Some Augraphy
    augmentations carry additional internal randomness, so identical draws may still
    differ run-to-run unless seeded here.

    Under a PyTorch DataLoader, torch seeds `random` per worker but NOT numpy, so pass a
    worker_init_fn if you see repeated grain patterns across workers:
        def _wi(wid): np.random.seed(torch.initial_seed() % 2**32)
    """
    random.seed(seed)
    np.random.seed(seed)


# ======================================================================================
# Custom effects
#
# These cover the failure modes visible in the real crops that Augraphy has no direct
# equivalent for. All operate on a single-channel uint8 image (the pipeline converts
# once at entry, see `degrade`).
# ======================================================================================

def resolution_loss(img, scale=0.5, down=cv2.INTER_AREA, up=cv2.INTER_LINEAR):
    """Round-trip through a lower resolution: the dominant synthetic/real gap.

    Treat this as a PROBABILITY knob, not a strength knob. Even scale=0.95 takes the
    Laplacian variance of a clean render from ~3900 to ~1400, because the round trip
    itself destroys the pixel-exact edges. `scale` only shapes the blurry tail; whether
    it fires at all is what controls the median.
    """
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(4, int(w * scale)), max(4, int(h * scale))),
                       interpolation=down)
    return cv2.resize(small, (w, h), interpolation=up)


def gaussian_blur(img, sigma=0.8):
    """Optical/scanner defocus. Beyond sigma 1.5 at h=64 the glyphs stop being legible."""
    k = max(3, int(2 * round(3 * sigma) + 1))
    return cv2.GaussianBlur(img, (k, k), sigma)


def motion_blur(img, degree=5, angle=0):
    """Scan drag / camera shake. `degree` is the streak length in pixels; `angle` rotates
    the streak (0 = horizontal, 90 = vertical)."""
    k = np.zeros((degree, degree), np.float32)
    k[degree // 2, :] = 1.0 / degree
    if angle:
        rot = cv2.getRotationMatrix2D((degree / 2 - 0.5, degree / 2 - 0.5), angle, 1.0)
        k = cv2.warpAffine(k, rot, (degree, degree))
        k /= max(1e-6, k.sum())
    return cv2.filter2D(img, -1, k)


def stroke_weight(img, amount=1.0):
    """Thicken (amount > 0) or thin (amount < 0) the ink.

    Real book scans run heavier than a clean render; faded pages run lighter. Fractional
    amounts are blended rather than snapped, because one whole pixel is a large change at
    h=64. Measured ceilings: +1.5 is the most that keeps vowel-sign counters open (+2.2
    closes them), and -1.8 shatters strokes into disconnected fragments.
    """
    if abs(amount) < 1e-3:
        return img
    n = int(np.ceil(abs(amount)))
    frac = abs(amount) / n
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    out = img
    for _ in range(n):
        # ink is dark, so thickening ink means eroding the grayscale image
        moved = cv2.erode(out, k) if amount > 0 else cv2.dilate(out, k)
        out = cv2.addWeighted(moved, frac, out, 1 - frac, 0)
    return out


_K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def _random_field(h, w, scale, octaves=2, decay=0.6):
    """Smooth multi-scale random field in [0,1], used as a spatial mask.

    Coarse octaves make whole groups of characters fade together; fine octaves bite into
    individual strokes. Both scales are present in the real crops, which is why this is
    multi-octave rather than a single blob size.
    """
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        # Floor at 4px: below that the mask is pixel noise rather than a blob, and it
        # dapples the INSIDE of a stroke into a dotted outline -- the same unusable look
        # that got LowInkRandomLines dropped.
        sc = max(4, int(scale / (2 ** o)))
        gh, gw = max(2, h // sc), max(2, w // sc)
        n = np.random.rand(gh, gw).astype(np.float32)
        out += amp * cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= decay
    return np.clip(out / tot, 0.0, 1.0)


def ink_dropout(img, coverage=0.25, scale=12, strength=1.0, octaves=2, ramp=0.8):
    """Erase ink in smooth random patches: some strokes lose a piece, their neighbours
    stay intact.

    This is the "partly missing character" look that dominates the real crops and that
    nothing else here reproduces. Global thinning (`stroke_weight` with a negative amount)
    thins EVERY character equally, and Letterpress fades the whole line; neither gives the
    within-line variation the real scans have -- in data/actual_samples the loop of a `ర`
    is broken open while the `ప` beside it is solid black, and one end of a line is heavy
    while the other is fragmented.

    Two spatial scales matter, hence `octaves`: coarse patches (scale ~30-60px) fade
    several characters together, fine ones (~5-15px) cut a single stroke.

    coverage : fraction of the crop the mask touches
    scale    : coarsest blob size in px; finer octaves are scale/2, scale/4, ...
    strength : 0..1; at 1.0 the deepest parts of the mask clear a thin stroke completely
    """
    h, w = img.shape[:2]
    f = _random_field(h, w, scale, octaves)
    thr = float(np.quantile(f, 1.0 - coverage))
    span = max(1e-6, float(f.max()) - thr)
    m = (np.clip((f - thr) / span, 0.0, 1.0) ** ramp) * strength

    # ink is dark, so DILATING the grayscale image is what eats the ink away
    t1 = cv2.dilate(img, _K3)
    t2 = cv2.dilate(t1, _K3)

    # piecewise ramp img -> t1 -> t2 as the mask deepens, so shallow patches only thin
    # the stroke while deep ones remove it outright
    lo = np.clip(m * 2.0, 0.0, 1.0)
    hi = np.clip((m - 0.5) * 2.0, 0.0, 1.0)
    stage = img.astype(np.float32) * (1 - lo) + t1.astype(np.float32) * lo
    out = stage * (1 - hi) + t2.astype(np.float32) * hi

    # dropout must only ever REMOVE ink, never add it
    return np.maximum(img, np.clip(out, 0, 255).astype(np.uint8))


def speckle(img, amount=0.004, blob_frac=0.15, dark=(30, 120), light_amount=0.0):
    """Dust and print-through specks, plus optional white ink drop-outs.

    Specks are single pixels, `blob_frac` of them grown to 2x2. Anything larger reads as
    snow rather than dust: a radius-2 circle is 5px on a 64px line, wider than a Telugu
    stroke, and it measured far outside the real samples' component density.
    """
    out = img.copy()
    h, w = out.shape[:2]
    n = int(amount * h * w)
    if n <= 0:
        return out
    ys = np.random.randint(0, h, n)
    xs = np.random.randint(0, w, n)
    vals = np.random.randint(dark[0], dark[1] + 1, n).astype(np.uint8)
    out[ys, xs] = np.minimum(out[ys, xs], vals)
    nb = int(n * blob_frac)
    if nb:
        for dy, dx in ((0, 1), (1, 0), (1, 1)):
            yy = np.clip(ys[:nb] + dy, 0, h - 1)
            xx = np.clip(xs[:nb] + dx, 0, w - 1)
            out[yy, xx] = np.minimum(out[yy, xx], vals[:nb])
    if light_amount > 0:
        m = int(light_amount * h * w)
        out[np.random.randint(0, h, m), np.random.randint(0, w, m)] = 255
    return out


def neighbour_line(img, top_frac=0.12, bottom_frac=0.12, alpha=0.75, shift=37):
    """Stamp faded fragments of adjacent text into the top and/or bottom rows.

    A line detector rarely crops tightly enough to exclude the descenders of the line
    above or the ascenders of the line below. Uses a squashed, shifted slice of the crop's
    own content as the fragment, so it looks like text without being readable text.
    """
    out = img.copy()
    h, w = out.shape[:2]
    src = np.roll(img, shift, axis=1)
    for frac, at_top in ((top_frac, True), (bottom_frac, False)):
        if frac <= 0:
            continue
        band_h = max(1, int(h * frac))
        if at_top:
            sl = src[int(h * 0.55):int(h * 0.95)]      # descenders of the line above
        else:
            sl = src[int(h * 0.05):int(h * 0.45)]      # ascenders of the line below
        band = cv2.resize(sl, (w, band_h), interpolation=cv2.INTER_AREA)
        y0, y1 = (0, band_h) if at_top else (h - band_h, h)
        out[y0:y1] = cv2.addWeighted(band, alpha, out[y0:y1], 1 - alpha, 0)
    return out


def ruled_line(img, thickness=1, value=90, at="bottom", offset=2):
    """A printed rule or the underline of the neighbouring line clipping the crop edge.
    Given a slight waviness so it is not a perfect raster row."""
    out = img.copy()
    h, w = out.shape[:2]
    y = offset if at == "top" else h - 1 - offset
    xs = np.arange(w)
    ys = np.clip(y + (np.sin(xs / max(40, w / 6.0)) * 1.2).astype(int), 0, h - 1)
    for dy in range(thickness):
        yy = np.clip(ys + dy, 0, h - 1)
        out[yy, xs] = np.minimum(out[yy, xs], value)
    return out


def skew(img, angle=1.0, max_drift=3.0, border=255):
    """Rotate, but clamp the angle so the far end of the line drifts at most `max_drift`
    pixels vertically.

    A line crop is very wide relative to its height, so an innocent-looking 2 degrees
    moves the end of a 900px crop by over 30px -- half the image height -- and slices the
    glyphs off. Real crops follow the text, so the residual skew *inside* a crop stays
    small even on a badly skewed page.
    """
    h, w = img.shape[:2]
    cap = np.degrees(np.arctan(max_drift / max(1, w / 2)))
    angle = float(np.clip(angle, -cap, cap))
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def wave(img, amplitude=1.5, period=160, border=255):
    """Baseline waviness from a curved page. Safer than rotation on a wide crop, since
    the displacement does not grow with width."""
    h, w = img.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    return cv2.remap(img, xs, ys + amplitude * np.sin(2 * np.pi * xs / period),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=border)


def pad_jitter(img, top=0.10, bottom=0.06, border=255):
    """Loosen the crop: shrink the glyph band and re-pad it unevenly, the way a real line
    detector leaves the text off-centre with uneven headroom. Height is preserved."""
    h, w = img.shape[:2]
    pt, pb = int(h * top), int(h * bottom)
    inner = max(8, h - pt - pb)
    small = cv2.resize(img, (w, inner), interpolation=cv2.INTER_AREA)
    return cv2.copyMakeBorder(small, pt, h - inner - pt, 0, 0,
                              cv2.BORDER_CONSTANT, value=border)


def binarize(img, offset=0, blur_first=0.0, adaptive=False, block=31):
    """Hard threshold, as many source PDFs are bilevel.

    Blurring BEFORE the threshold is what produces the blobby, ragged, partly-merged
    strokes of the harshest real samples -- a straight Otsu on a clean render just gives
    crisp edges. Never leave this as the final step: the resample later in the pipeline
    is what puts the soft grey edges of a real scan back.
    """
    src = gaussian_blur(img, blur_first) if blur_first > 0 else img
    if adaptive:
        return cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, block | 1, offset)
    t, _ = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ((src > (t + offset)) * 255).astype(np.uint8)


def gamma(img, g=1.0):
    """g < 1 darkens the ink, g > 1 fades it."""
    lut = np.clip(((np.arange(256) / 255.0) ** g) * 255, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def contrast_shift(img, gain=1.0, bias=0.0):
    return np.clip(img.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)


def illumination(img, strength=30, axis="x"):
    """Smooth lighting gradient from an unevenly lit page scan."""
    h, w = img.shape[:2]
    ramp = np.linspace(-1, 1, w if axis == "x" else h, dtype=np.float32)
    ramp = np.sin(ramp * np.pi / 2) * strength
    field = np.tile(ramp, (h, 1)) if axis == "x" else np.tile(ramp[:, None], (1, w))
    return np.clip(img.astype(np.float32) - field, 0, 255).astype(np.uint8)


def paper_normalize(img, target=248, pct=97, max_gain=2.6, black_pct=0.0):
    """Auto-levels, the way a scanner's own processing does it. Run this LAST.

    Stacked gamma / contrast / illumination / texture leaves the paper a flat mid-grey,
    which no real crop shows: every sample in data/actual_samples has a background mean
    of 247-252. Maps the `pct`th percentile onto `target`.

    `max_gain` has to be generous (2.6): the grime and tone effects above can leave the
    paper well under 200, and a tighter cap silently strands those samples at a grey the
    real crops never show.

    `black_pct` sets a black point as well, which matters more than it looks: a pure
    multiply lightens the INK along with the paper, so contrast stays capped around 200
    and the crisp bold real crops (215-230) stay unreachable. Pulling the black point up
    lets the paper go white while the ink stays dark, which is what a real scanner's
    levels do.
    """
    # Percentiles are measured on the MIDDLE rows only, then applied to the whole crop.
    # pad_jitter's pure-white padding and the stamped neighbour-line bands live at the
    # top/bottom edges; including them puts the 85th percentile at 255 straight away, the
    # gain comes out as 1.0, and a genuinely grey paper is left untouched.
    h = img.shape[0]
    core = img[int(h * 0.2):int(h * 0.8)]
    if core.size < 16:
        core = img
    p = float(np.percentile(core, pct))
    if p <= 1:
        return img
    lo = float(np.percentile(core, black_pct)) if black_pct > 0 else 0.0
    lo = min(lo, p - 1)                      # keep the span positive
    gain = min(max_gain, target / max(1.0, p - lo))
    # Re-derive the black point from the (possibly capped) gain so that `p` lands exactly
    # on `target`. Without this, a low-contrast crop asks for a gain above max_gain, the
    # cap silently denies it, and subtracting the full `lo` anyway leaves EVERYTHING darker
    # than it started -- paper at 255 came out at 169 with the ink crushed to 0, i.e. an
    # inverted-looking image out of a function whose job is to whiten the paper.
    lo = max(0.0, p - target / gain)
    if gain <= 1.001 and lo <= 0:
        return img
    out = (img.astype(np.float32) - lo) * gain
    return np.clip(out, 0, 255).astype(np.uint8)


# ======================================================================================
# Pre-built Augraphy severity ladders
#
# Four levels per augmentation, level 0 ~ the mildest real crop and level 3 ~ the worst
# still-legible one. Objects are built once at import and reused; each __call__ re-samples
# the object's internal randomness, so reuse still varies the output. Building a ladder
# (rather than one object with a wide range) is what lets the severity draw in `degrade`
# actually control strength while keeping the objects pre-built.
#
# Per-call cost at h=64 on augraphy 8.2.6 -- these drive the probabilities in `degrade`:
#   SubtleNoise/Jpeg 0.2ms, InkMottling 0.7, BrightnessTexturize 1.0, InkBleed 1.1,
#   BleedThrough 1.4, DirtyDrum 3.6, NoiseTexturize 5.8, BadPhotoCopy 33, Letterpress 79.
# ======================================================================================

_INK_BLEED = [
    # kernel_size must stay 3-5: at 7+ on a 64px line the glyphs merge into a black blob
    InkBleed(intensity_range=(0.4, 0.6), severity=(0.20, 0.30), kernel_size=(3, 3), p=1),
    InkBleed(intensity_range=(0.6, 0.8), severity=(0.30, 0.45), kernel_size=(5, 5), p=1),
    InkBleed(intensity_range=(0.85, 0.95), severity=(0.40, 0.45), kernel_size=(5, 5), p=1),
    InkBleed(intensity_range=(0.90, 1.00), severity=(0.45, 0.50), kernel_size=(5, 5), p=1),
]

_INK_MOTTLING = [
    InkMottling(ink_mottling_alpha_range=(0.20, 0.30), ink_mottling_gaussian_kernel_range=(3, 3), p=1),
    InkMottling(ink_mottling_alpha_range=(0.35, 0.45), ink_mottling_gaussian_kernel_range=(3, 3), p=1),
    InkMottling(ink_mottling_alpha_range=(0.50, 0.60), ink_mottling_gaussian_kernel_range=(3, 3), p=1),
    InkMottling(ink_mottling_alpha_range=(0.65, 0.75), ink_mottling_gaussian_kernel_range=(5, 5), p=1),
]

# Fades ink hard (drops contrast to 66-94 against the real band of 145-216) and costs
# ~79ms/call, so it fires rarely and is paired with a dilate by the ordering in `degrade`.
_LETTERPRESS = [
    Letterpress(n_samples=(100, 200), std_range=(400, 600), value_range=(200, 255), p=1),
    Letterpress(n_samples=(200, 300), std_range=(700, 1100), value_range=(200, 255), p=1),
    Letterpress(n_samples=(300, 500), std_range=(1100, 1600), value_range=(210, 255), p=1),
    Letterpress(n_samples=(400, 600), std_range=(1600, 2200), value_range=(220, 255), p=1),
]

_BLEED_THROUGH = [
    BleedThrough(intensity_range=(0.2, 0.4), alpha=0.15, offsets=(14, 14), p=1),
    BleedThrough(intensity_range=(0.4, 0.6), alpha=0.22, offsets=(18, 18), p=1),
    BleedThrough(intensity_range=(0.6, 0.8), alpha=0.30, offsets=(24, 24), p=1),
    BleedThrough(intensity_range=(0.8, 0.9), alpha=0.38, offsets=(30, 30), p=1),
]

_NOISE_TEXTURIZE = [
    NoiseTexturize(sigma_range=(2, 6), turbulence_range=(2, 5), p=1),
    NoiseTexturize(sigma_range=(4, 10), turbulence_range=(2, 5), p=1),
    NoiseTexturize(sigma_range=(6, 14), turbulence_range=(3, 6), p=1),
    NoiseTexturize(sigma_range=(10, 20), turbulence_range=(3, 7), p=1),
]

_BRIGHTNESS_TEXTURIZE = [
    BrightnessTexturize(texturize_range=(0.94, 0.98), deviation=0.05, p=1),
    BrightnessTexturize(texturize_range=(0.91, 0.96), deviation=0.08, p=1),
    BrightnessTexturize(texturize_range=(0.88, 0.94), deviation=0.11, p=1),
    BrightnessTexturize(texturize_range=(0.85, 0.92), deviation=0.14, p=1),
]

# line_width stays 1-2 and direction is horizontal (1) or both (2). Wide vertical streaks
# smear straight across whole glyphs at this height.
_DIRTY_DRUM = [
    DirtyDrum(line_width_range=(1, 1), line_concentration=0.05, direction=1, noise_intensity=0.3, p=1),
    DirtyDrum(line_width_range=(1, 1), line_concentration=0.10, direction=1, noise_intensity=0.5, p=1),
    DirtyDrum(line_width_range=(1, 2), line_concentration=0.18, direction=1, noise_intensity=0.65, p=1),
    DirtyDrum(line_width_range=(1, 2), line_concentration=0.25, direction=2, noise_intensity=0.80, p=1),
]

# noise_type=1 ONLY. Types 4-7 route through a worley-noise path that raises an
# AssertionError inside numba's JIT type inference. Being page-scale, on a line crop this
# lands as an edge grime band, which is exactly what several real samples show. ~33ms.
_BAD_PHOTOCOPY = [
    BadPhotoCopy(noise_type=1, noise_concentration=(0.10, 0.20), noise_sparsity=(0.6, 0.8), p=1),
    BadPhotoCopy(noise_type=1, noise_concentration=(0.10, 0.30), noise_sparsity=(0.5, 0.7), p=1),
    BadPhotoCopy(noise_type=1, noise_concentration=(0.20, 0.40), noise_sparsity=(0.4, 0.6), p=1),
    BadPhotoCopy(noise_type=1, noise_concentration=(0.30, 0.50), noise_sparsity=(0.3, 0.5), p=1),
]

_SUBTLE_NOISE = [
    SubtleNoise(subtle_range=6, p=1),
    SubtleNoise(subtle_range=10, p=1),
    SubtleNoise(subtle_range=14, p=1),
    SubtleNoise(subtle_range=18, p=1),
]

_JPEG = [
    Jpeg(quality_range=(60, 80), p=1),
    Jpeg(quality_range=(45, 65), p=1),
    Jpeg(quality_range=(30, 50), p=1),
    Jpeg(quality_range=(20, 35), p=1),
]

# ======================================================================================
# Composition
# ======================================================================================

def _p(prob):
    return random.random() < prob


def _scaled_p(prob, s):
    """Probability scaled by severity, so `severity` controls HOW MANY effects fire as
    well as how hard each one hits.

    Without this, a severity-0 sample still draws every effect at full probability and
    just gets mild versions of all of them, which stacks into something heavier than any
    mild real crop -- the opposite of what the parameter promises.
    """
    return random.random() < prob * (0.45 + 0.55 * s)


def _pick(ladder, s, jitter=0.25):
    """Choose a severity level from the ladder given the sample's severity `s` in [0, 1].

    The jitter keeps neighbouring samples from all landing on the same level, so the
    augmented distribution stays smooth instead of clustering at four points.
    """
    x = s + random.uniform(-jitter, jitter)
    return ladder[int(np.clip(x, 0.0, 0.999) * len(ladder))]


def _ink_fraction(g):
    """Cheap proxy for how much of the crop is ink.

    Measured over the middle rows only, so the neighbouring-line and ruled-line bands
    stamped into the top/bottom edges do not count as glyph ink.

    The threshold is RELATIVE to the crop's own paper level, not a fixed 128. With a fixed
    threshold, any global brightness change fakes a huge ink change -- an illumination
    gradient that lifts the ink from 100 to 130 reads as "94% of the ink vanished" and
    sends the backstop off chasing a problem that isn't there.
    """
    h = g.shape[0]
    core = g[int(h * 0.2):int(h * 0.8)]
    if core.size < 16:
        core = g
    paper = float(np.percentile(core, 90))
    thr = max(24.0, paper * 0.62)
    return float((core < thr).mean())


def _limit_ink_change(g, ref_frac, max_ratio=1.6, min_ratio=0.55, min_dark=110,
                      max_steps=3):
    """Pull the ink back toward the input's if the composition went too far either way.

    Per-effect ceilings do NOT bound a composition, and both directions can destroy the
    label rather than making the example hard:
      * too much ink -- a +0.8 dilate, a severity-0.5 InkBleed and a blurred binarisation
        are each individually safe, but stacked they close every counter and leave a solid
        blob. Erosion cannot undo that.
      * too little ink -- ink_dropout, a negative stroke_weight, Letterpress and a gamma
        fade stacked can erase so much of a glyph that it is no longer identifiable.
    This backstop is what lets the effect probabilities above stay aggressive.

    For reference: the real crops sit at 0.105-0.152 ink fraction against 0.115 for the
    clean render -- at most ~1.3x and never below ~0.8x -- so 1.6x / 0.55x is generous
    headroom on both sides and only the extreme tail is touched.

    `min_dark` is a separate floor on ink DARKNESS rather than ink area. The fade, dropout
    and brightening effects can stack into a crop whose darkest pixel is ~150 on white
    paper: still technically "inked" by area, but a blank image as far as the model is
    concerned, carrying a full text label. Stretching the levels is the fix, and doing it
    here rather than forcing a black point on every sample keeps genuinely low-contrast
    crops (the real range reaches down to 145) in the distribution.
    """
    # The darkness test uses the SAME middle-rows region as _ink_fraction. Testing the
    # whole crop instead lets a blank glyph band pass whenever the stamped neighbour-line
    # or ruled-line bands at the edges happen to be dark.
    h = g.shape[0]
    core = g[int(h * 0.2):int(h * 0.8)]
    if core.size < 16:
        core = g
    if float(np.percentile(core, 1)) > min_dark:
        g = paper_normalize(g, target=252, pct=85, black_pct=1.0)
    for _ in range(max_steps):
        frac = _ink_fraction(g)
        if frac > ref_frac * max_ratio:
            g = stroke_weight(g, -0.5)
        elif frac < ref_frac * min_ratio:
            g = stroke_weight(g, 0.5)
        else:
            break
    return g


def _limit_ink_loss(g, pre, min_ratio=0.55, max_steps=3):
    """Fade the ink-stage damage back toward `pre` if it erased too much of the glyph.

    This has to happen HERE, in the ink stage, and not only in the final backstop: the
    dropout/fade/Letterpress stack can leave a crop with literally no ink left (observed:
    ink fraction 0.000, darkest pixel 235 on white paper), and once the strokes are gone no
    end-of-pipeline guard can bring them back -- stretching or dilating a blank image just
    gives a blank image. Blending back toward the pre-damage copy restores both the
    coverage and the darkness while keeping the *character* of the dropout, just weaker.
    """
    ref = _ink_fraction(pre)
    if ref <= 0:
        return g
    for _ in range(max_steps):
        if _ink_fraction(g) >= ref * min_ratio:
            break
        g = cv2.addWeighted(pre, 0.5, g, 0.5, 0)
    return g


def _safe(applier, g):
    """Apply one Augraphy augmentation, falling back to the input on failure.

    Augraphy's randomised internals occasionally throw on a 64px-tall crop, and
    BadPhotoCopy can blow up inside numba's JIT. One bad draw must not kill a
    multi-hundred-thousand-image generation run or a training epoch.
    """
    try:
        out = applier(g)
    except Exception:
        return g
    if isinstance(out, dict):
        out = out.get("output", g)
    if out is None:
        return g
    out = np.asarray(out)
    if out.ndim == 3:                      # some augraphy paths promote to 3 channels
        out = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    return np.clip(out, 0, 255).astype(np.uint8)


def degrade(img, severity=None):
    """Apply the full composed degradation to one line crop.

    img      : uint8, HxW or HxWx3. Should already be the 64px-tall crop.
    severity : 0.0 (mildest real crop) to 1.0 (worst still-legible one). None draws it
               uniformly, which is what you want for training -- the model needs the whole
               range, not the average of it.
    Returns the same shape and dtype as `img`.

    No single effect here makes a synthetic crop look real: blur alone leaves the ink too
    light, thickening alone stays too clean, noise alone sits on razor-sharp glyphs. The
    real crops are a composition whose effects offset one another, which is why this is
    one function rather than a menu to pick one item from.
    """
    was_colour = img.ndim == 3
    # Work in grayscale throughout: the model consumes grayscale anyway, several ops
    # (Otsu, constant borders) are single-channel only, and it keeps every effect
    # channel-consistent. Converted back at the end.
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if was_colour else img
    g = np.ascontiguousarray(g, dtype=np.uint8)

    s = random.random() if severity is None else float(np.clip(severity, 0.0, 1.0))
    ref_ink = _ink_fraction(g)     # baseline for the legibility backstop near the end

    # ---- bilevel source --------------------------------------------------------------
    # Many source PDFs are bilevel, and they were thresholded at print/scan time -- i.e.
    # BEFORE the ink spread and the resample, not after. Order matters here beyond
    # realism: thresholding already-thickened ink fuses the strokes into a solid blob and
    # erodes cannot recover the counters, because binarisation threw the information away.
    # Applied first, it instead gives the crisp bilevel text that the thickening and
    # resample below turn into a convincing scan.
    if _scaled_p(0.18, s):
        g = binarize(g, offset=random.randint(-5, 10),
                     blur_first=random.uniform(0.4, 1.2))

    # ---- ink -------------------------------------------------------------------------
    ink_pre = g                    # snapshot for _limit_ink_loss at the end of this stage

    # Decided up front because it changes the thickening below. A page that failed to
    # transfer its ink is not also over-inked: letting a heavy dilate run first simply
    # fattens the dropout bites back over, and the two effects cancel into ordinary
    # slightly-blurry text. Coupling them is both more realistic and what keeps the broken
    # strokes actually visible in the output.
    do_dropout = _scaled_p(0.50, s)

    if _p(0.80):
        if do_dropout:
            amount = random.uniform(-0.35, 0.25)
        else:
            # heavier is the common case; the fading branch covers worn/low-toner pages
            amount = (random.uniform(0.10, 0.20 + 0.60 * s) if _p(0.8)
                      else -random.uniform(0.10, 0.15 + 0.35 * s))
        g = stroke_weight(g, amount)
    if _scaled_p(0.50, s):
        g = _safe(_pick(_INK_BLEED, s), g)
    if do_dropout:
        # High probability on purpose: partly-missing strokes are a defining feature of
        # the real crops, not an edge case. Coarse and fine blob scales are both drawn so
        # the model sees whole-word fading and single-stroke breaks.
        g = ink_dropout(g,
                        coverage=random.uniform(0.12, 0.22 + 0.23 * s),
                        scale=random.choice([6, 9, 12, 18, 30, 45]),
                        strength=random.uniform(0.60, 0.80 + 0.20 * s),
                        octaves=random.choice([2, 2, 3]))
    if _scaled_p(0.30, s):
        g = _safe(_pick(_INK_MOTTLING, s), g)
    if _scaled_p(0.03, s):                            # CHANGED(2048): 0.06 -> 0.03 (speed)
        g = _safe(_pick(_LETTERPRESS, s), g)          # ~79ms/call: halved, the biggest cost
    # Catch over-erasure while the strokes still exist to be restored.
    g = _limit_ink_loss(g, ink_pre)

    # ---- paper / grime ---------------------------------------------------------------
    if _scaled_p(0.25, s):
        g = _safe(_pick(_BLEED_THROUGH, s), g)
    if _scaled_p(0.25, s):
        g = _safe(_pick(_NOISE_TEXTURIZE, s), g)
    if _scaled_p(0.20, s):
        g = _safe(_pick(_BRIGHTNESS_TEXTURIZE, s), g)
    if _scaled_p(0.20, s):
        g = _safe(_pick(_DIRTY_DRUM, s), g)
    if _scaled_p(0.025, s):                           # CHANGED(2048): 0.05 -> 0.025 (speed)
        g = _safe(_pick(_BAD_PHOTOCOPY, s), g)        # ~33ms/call: halved
    if _scaled_p(0.25, s):
        g = illumination(g, random.uniform(8, 12 + 16 * s), random.choice(["x", "y"]))

    # ---- tone ------------------------------------------------------------------------
    if _scaled_p(0.45, s):
        # Weighted toward darkening: the real crops have darker ink than the render
        # (ink mean median 62 vs 61 clean but with far heavier strokes), and fading is
        # the rarer failure mode.
        g = gamma(g, random.uniform(0.90 - 0.45 * s, 0.95) if _p(0.7)
                     else random.uniform(1.10, 1.15 + 0.35 * s))
    if _scaled_p(0.35, s):
        # gain > 1 with a negative bias is what reaches the crisp high-contrast end of
        # the real range (the bold samples measure 215-230); without it the pipeline tops
        # out near 196 and those crops fall outside the augmented distribution.
        g = contrast_shift(g, random.uniform(0.95, 1.10 + 0.30 * s),
                           random.uniform(-8 - 20 * s, 4 + 6 * s))

    # ---- crop artefacts --------------------------------------------------------------
    if _scaled_p(0.35, s):
        g = neighbour_line(g, 0.12 if _p(0.5) else 0.0, 0.09 + 0.06 * s,
                           alpha=random.uniform(0.5, 0.9),
                           shift=random.randint(20, 120))
    if _scaled_p(0.15, s):
        g = ruled_line(g, random.choice([1, 2]), random.randint(50, 140),
                       random.choice(["top", "bottom"]), random.randint(0, 3))
    if _p(0.30):
        g = pad_jitter(g, random.uniform(0.02, 0.14), random.uniform(0.02, 0.10))

    # ---- geometry --------------------------------------------------------------------
    if _p(0.40):
        g = skew(g, random.choice([-3.0, 3.0]), max_drift=random.uniform(1.0, 4.0))
    if _p(0.30):
        g = wave(g, random.uniform(0.8, 1.0 + 1.6 * s), random.uniform(80, 220))

    # ---- sensor noise (before the resample, so the resample averages it down) --------
    grainy = False
    if _scaled_p(0.35, s):
        g = _safe(_pick(_SUBTLE_NOISE, s), g)
        grainy = True
    if _scaled_p(0.30, s):
        g = speckle(g, random.uniform(0.0008, 0.0015 + 0.004 * s), blob_frac=0.15,
                    dark=(20, 130), light_amount=0.002 if _p(0.3) else 0.0)
        grainy = True

    # ---- resample --------------------------------------------------------------------
    # Grain that was never resampled reads as video static rather than paper, so a grainy
    # sample takes the resample that would have produced it far more often. Most samples
    # skip this entirely -- see the note on resolution_loss about why it is a probability
    # knob and not a strength knob.
    if _p(0.55 if grainy else 0.28):
        g = resolution_loss(g, random.uniform(0.78 - 0.25 * s, 1.0),
                            up=random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC]))
    if _scaled_p(0.15, s):
        g = gaussian_blur(g, random.uniform(0.30, 0.35 + 0.35 * s))
    if _scaled_p(0.10, s):
        g = motion_blur(g, 3 if _p(0.7) else 5, random.choice([0, 90]))

    # ---- compression -----------------------------------------------------------------
    if _scaled_p(0.40, s):
        g = _safe(_pick(_JPEG, s), g)

    # ---- auto-levels -----------------------------------------------------------------
    # pct=85 with a target at/near 255 clips the paper bulk to white, which is how a real
    # scan looks; mapping a high percentile instead leaves the paper a few levels grey.
    # The black point stays optional so low-contrast crops remain possible (the real range
    # goes down to 145) -- forcing it on every sample stretches the paper variance well past
    # the real 13-16. The blank-image case it used to cover is handled by the darkness floor
    # in the backstop below instead, which only fires when it is actually needed.
    g = paper_normalize(g, target=random.randint(250, 255), pct=85,
                        black_pct=random.uniform(0.5, 3.0) if _p(0.7) else 0.0)

    # ---- legibility backstop, truly last ---------------------------------------------
    # It has to run AFTER auto-levels, not before: auto-levels rescales tone, which moves
    # the ink measurement, so a backstop placed earlier bounds an intermediate image rather
    # than the one the model is trained on. Two-sided, because the ink stage can now remove
    # ink (ink_dropout) as well as add it.
    g = _limit_ink_change(g, ref_ink)

    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR) if was_colour else g


P_CLEAN = 0.50

def make_augmenter(p_clean=P_CLEAN, severity=None):
    """Build the per-sample callable: augment(uint8 HxW or HxWx3) -> same shape.

    p_clean  : fraction of samples passed through untouched. Keeps the model able to read
               crisp text (it already does that well -- don't regress it) and trims CPU.
    severity : pin the severity for every sample, or None to draw per sample. Pinning is
               for previews and ablations, not training.

    Never raises: a failing effect falls back to the image as it was.
    """
    def augment(img):
        if random.random() < p_clean:
            return img
        try:
            return degrade(img, severity)
        except Exception as exc:
            print(f"[warn] augmentation failed, using clean image: {exc}")
            return img

    return augment


# ======================================================================================
# Preview: augmented synthetic crops next to the real ones they are imitating.
# ======================================================================================


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
def build_optimizer(model, lr, weight_decay, betas):
    """Single LR tier. Biases / norm weights / the positional embedding get no weight
    decay (matches the prior Trainer default); other weights get `weight_decay`.

    (The extension-only LR tier is gone: with the split ``pos_embed_ext`` merged into a
    single trained ``pos_embed``, there are no longer any 'new' rows to speed up.)"""
    decay, nodecay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name == "pos_embed" or p.ndim < 2 or "norm" in name.lower():
            nodecay.append(p)                             # embeddings/bias/norm -> no WD
        else:
            decay.append(p)
    groups = [
        {"params": decay, "lr": lr, "weight_decay": weight_decay},
        {"params": nodecay, "lr": lr, "weight_decay": 0.0},
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

    DATASET_NAME1 = "harsha-desaraju/sample-dataset-new"
    DATASET_NAME2 = "harsha-desaraju/telugu-pdf-line-image-text"
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
    LR = 1e-4                      # single LR tier (pos_embed is now one trained table)
    MIN_LR = 1e-5
    WARMUP_STEPS = 1000
    WEIGHT_DECAY = 0.05
    BETAS = (0.9, 0.98)
    # NORMAL dataset has a known length, so training is epoch-based: Trainer computes
    # the total optimizer steps (accounting for #GPUs / grad-accum) and hands them to
    # the cosine scheduler (T_max) via create_scheduler. No manual step math needed.
    EPOCHS = 5                     # complete passes over the train split

    # eval / io
    EVAL_SLICE_CAP = 1500          # cap each eval slice for speed
    EVAL_BATCH = 32                # smaller eval batch (long slices can be up to 2048px)
    SAVE_EVAL_STEPS = 1000
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
        load_checkpoint(model, OLD_CHECKPOINT)          # first launch: seed from a checkpoint
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
                                           augment_fn=make_augmenter())  # train aug
    eval_preprocessor = ImagePreprocessor(tokenizer, IMAGE_COLUMN, TEXT_COLUMN, augment_fn=None)

    # Train: whole split downloaded to disk, transformed lazily via with_transform.
    # Trainer's sampler shuffles each epoch, so no manual shuffle is needed.
    configs = ['train_0000', 'train_0001', 'train_0002',
               # 'train_0003', 'train_0004', 'train_0005', 'train_0006', 'train_0007',
               # 'train_0008', 'train_0009'
            ]
    train_hf = []
    for config in configs:
        tds = load_dataset(DATASET_NAME1, config, columns=[IMAGE_COLUMN, TEXT_COLUMN, SOURCE_COLUMN])[DATASET_SPLIT]
        train_hf.append(tds)

    # Load the 2nd dataset - The real images
    tds = load_dataset(DATASET_NAME2, split='train', columns=[IMAGE_COLUMN, TEXT_COLUMN])
    tds = tds.add_column(SOURCE_COLUMN, ['natural']*len(tds))
    tds = train_hf.append(tds)

    train_hf = concatenate_datasets(train_hf, axis=0)

    train_ds = train_hf.with_transform(train_preprocessor)
    print(f"[data] {DATASET_NAME1}:{DATASET_SPLIT} -> {len(train_hf)} rows; cols {train_hf.column_names}")

    # Validation: small; materialize rows and build the 4 eval slices.
    val_ds = load_dataset(DATASET_NAME1, EVAL_SPLIT)[EVAL_SPLIT]
    val_rows = list(val_ds)
    print(f"[data] validation {DATASET_NAME1}:{EVAL_SPLIT} -> {len(val_rows)} rows")
    eval_slices = build_eval_slices(val_rows, IMAGE_COLUMN, TEXT_COLUMN, SOURCE_COLUMN,
                                    WIDTH_COLUMN, eval_preprocessor, BASE_WIDTH, EVAL_SLICE_CAP, SEED)

    # ---- Optimizer (fresh, single LR tier); scheduler built by CTCTrainer ----
    optimizer = build_optimizer(model, LR, WEIGHT_DECAY, BETAS)

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
        dataloader_num_workers=4,
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
        run_name="ctc-encoder-stage-3",
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
