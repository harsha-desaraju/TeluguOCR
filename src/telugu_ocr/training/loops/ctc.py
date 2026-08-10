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
import regex
import json
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
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint


# ============================================================================
# Model config (CHANGED(2048) fields flagged)
# ============================================================================
from src.telugu_ocr.models.image_encoder import (CTCEncoderConfig, ConvBlock, ConvStem,
                                                DropPath, ImageEncoderCTC, TransformerBlock)


# ============================================================================
# Building blocks (UNCHANGED)
# ============================================================================








LABEL_PAD_ID = -100




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
# Grapheme tokenizer
# ============================================================================
from src.telugu_ocr.tokenizer.grapheme import TeluguGraphemeTokenizer


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

# Pin OpenCV to one thread (runtime companion to the env vars set at the top of the file).
# Forked DataLoader workers inherit this, so each augments on a single core instead of
# every worker's cv2 fighting for all of them. Gated by the same LIMIT_AUG_THREADS toggle.
if LIMIT_AUG_THREADS:
    cv2.setNumThreads(1)


# ============================================================================
# Image augmentation
# ============================================================================
from src.telugu_ocr.data.augment import make_augmenter, set_seed


# ======================================================================================
# Custom effects
#
# These cover the failure modes visible in the real crops that Augraphy has no direct
# equivalent for. All operate on a single-channel uint8 image (the pipeline converts
# once at entry, see `degrade`).
# ======================================================================================





































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



# Fades ink hard (drops contrast to 66-94 against the real band of 145-216) and costs
# ~79ms/call, so it fires rarely and is paired with a dilate by the ordering in `degrade`.




# line_width stays 1-2 and direction is horizontal (1) or both (2). Wide vertical streaks
# smear straight across whole glyphs at this height.

# noise_type=1 ONLY. Types 4-7 route through a worley-noise path that raises an
# AssertionError inside numba's JIT type inference. Being page-scale, on a line crop this
# lands as an edge grime band, which is exactly what several real samples show. ~33ms.



# ======================================================================================
# Composition
# ======================================================================================

















P_CLEAN = 0.50



# ======================================================================================
# Preview: augmented synthetic crops next to the real ones they are imitating.
# ======================================================================================


# ============================================================================
# Preprocessing + collator
# ============================================================================
from src.telugu_ocr.data.collators import CTCBatchMapper


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
    train_preprocessor = CTCBatchMapper(tokenizer, IMAGE_COLUMN, TEXT_COLUMN,
                                           augment_fn=make_augmenter(p_clean=P_CLEAN))  # train aug
    eval_preprocessor = CTCBatchMapper(tokenizer, IMAGE_COLUMN, TEXT_COLUMN, augment_fn=None)

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
