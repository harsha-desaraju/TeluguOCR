from __future__ import annotations

# ============================================================================
# CPU-thread limiting for the augmentation DataLoader workers (PORTED from
# train_ctc_encoder.py). These env vars size the math-library thread pools at
# import time, so they MUST be set BEFORE numpy / torch / cv2 are imported.
# Under a multi-worker (and DDP) DataLoader, pinning each worker's OpenCV/OMP/BLAS
# to one thread stops them oversubscribing the cores and makes augmentation faster.
# Flip LIMIT_AUG_THREADS to False if you measure it hurting throughput.
# ============================================================================
import os

LIMIT_AUG_THREADS = True
if LIMIT_AUG_THREADS:
    for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[_thread_var] = "1"

# Keep wandb from intercepting rank-0's stdout/stderr (console mirroring). A stalled
# console stream blocks rank-0's print() while rank 1 waits at the post-eval save
# collective, and after ddp_timeout the NCCL watchdog kills the run — this is what
# ended the 2xT4 run at step 16000. Metric logging (report_to="wandb") is unaffected.
os.environ.setdefault("WANDB_CONSOLE", "off")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutput

from transformers import Trainer, TrainingArguments
from transformers import TrainerCallback  # CER callback + grad-norm alert
from transformers.trainer_utils import get_last_checkpoint  # resume helper

from PIL import Image
from dataclasses import dataclass

from datasets import load_dataset, concatenate_datasets
from torchvision import transforms

import contextlib
import gc
import json
import math
import random
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple



# Encoder model utils

from src.telugu_ocr.data.collators import LineTensorizer


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



from src.telugu_ocr.models.encoder_decoder import (DecoderTransformerBlock, EncoderDecoder,
                                                   TextDecoder)
from src.telugu_ocr.models.image_encoder import (CTCEncoderConfig, ConvBlock, ConvStem, DropPath,
                                                 ImageEncoderCTC, TransformerBlock)
from src.telugu_ocr.models.text_decoder import (GPTConfig, GPTModel, GPTTransformerBlock,
                                                MultiHeadAttention, SwiGLU,
                                                calculate_positional_encodings)












# ============================================================================
# Grapheme tokenizer
# ============================================================================
from src.telugu_ocr.tokenizer.grapheme import TeluguGraphemeTokenizer


















# ============================================================================
# CTC auxiliary-loss weight (STAGE-2). The combined training objective is
#     loss = CE_loss + CTC_LOSS_WEIGHT * CTC_loss
# where CE is the decoder LM cross-entropy and CTC comes from the encoder's CTC
# head. Exposed as a config knob so the weight can be tuned; set to 0.0 to disable
# the CTC term entirely (pure LM cross-entropy).
# ============================================================================
CTC_LOSS_WEIGHT = 0.3




# ============================================================================
# ADDED (stage-1): CER evaluation utilities + callback.
# ============================================================================
def _edit_distance(a, b):
    # Levenshtein, dependency-free
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[n]


def compute_cer(preds, refs):
    tot_e = tot_c = 0
    for p, r in zip(preds, refs):
        tot_e += _edit_distance(p, r)
        tot_c += max(len(r), 1)
    return tot_e / max(tot_c, 1)


# B=1 greedy generation with no KV cache is SLOW (~seconds/sequence). This runs on
# rank 0 only while the other ranks wait at the next barrier, so keep the TOTAL small:
# at 48/slice x 2 slices (natural / random) that is ~96 sequences, a few minutes —
# comfortably inside the raised ddp_timeout. Raise for a sharper CER estimate only if
# you also raise ddp_timeout.
CER_EVAL_SAMPLES = 48   # eval images decoded per slice per CER eval (B=1 greedy)
CER_PRINT_K = 2  # print this many ref/hyp pairs each eval (kept small: stdout volume
                 # is what backed up the notebook pipe / wandb console and hung rank 0)


class CEREvalCallback(TrainerCallback):
    """CER during evaluate(), reported PER SLICE, for the model's TWO decoders:

      TEXT decoder (autoregressive LM), in two flavours:
        * generation CER     (``eval_<slice>_cer``)     — free-running B=1 greedy decode
        * teacher-forced CER (``eval_<slice>_tf_cer``)  — one forward with the ground-truth
                                                          tokens fed in, argmax next-token
      CTC decoder (the encoder's CTC head):
        * CTC CER            (``eval_<slice>_ctc_cer``) — per-frame argmax -> collapse
                                                          repeats -> drop blanks (no ref)

    All three are computed on the SAME up-to-CER_EVAL_SAMPLES images per slice, so they are
    directly comparable. Macro-averages ``eval_cer`` / ``eval_tf_cer`` / ``eval_ctc_cer``
    over slices are added too. All metrics go into the metrics dict and to wandb.

    Early stopping was removed, so these are purely for monitoring; everything runs on
    rank 0 only (see on_evaluate).
    """

    def __init__(self, model, eval_slices, preprocessor, tokenizer,
                 image_col="image", text_col="text"):
        self.model = model
        self.slices = eval_slices  # {name: list of raw rows with image/text cols}
        self.prep = preprocessor   # clean preprocessor (no augmentation)
        self.tok = tokenizer
        self.image_col = image_col
        self.text_col = text_col

    @torch.no_grad()
    def _teacher_forced_pred(self, bridged, cross_key_mask, ref_ids, device):
        """Teacher-forced through the TEXT decoder, REUSING a precomputed (bridged) encoder
        output — no re-encode. logits[t] predicts token t+1, so argmax(logits[:-1]) are the
        predictions for ref positions 1..T-1. Returns the decoded hypothesis string."""
        ids = torch.as_tensor([ref_ids], dtype=torch.long, device=device)  # (1, T)
        out = self.model.decoder_model(ids, bridged, text_padding_mask=None,
                                       img_text_padding_mask=cross_key_mask, labels=None)
        pred = out.logits[0, :-1].argmax(dim=-1)         # (T-1,)
        return self.tok.decode(pred.tolist(), skip_special_tokens=True)

    @torch.no_grad()
    def _ctc_pred(self, enc_raw):
        """CTC greedy decode from the encoder's CTC head (the second decoder), REUSING the
        precomputed RAW (pre-bridge) encoder output: per-frame argmax -> collapse consecutive
        repeats -> drop blanks -> decode. Collapse matches the standalone CTC encoder's eval."""
        logits = self.model.encoder_model.ctc_head(enc_raw)       # (1, T, C=vocab+1)
        frame_ids = logits[0].argmax(dim=-1).tolist()
        blank = self.model.encoder_model.blank_id
        collapsed, prev = [], None
        for t in frame_ids:
            if t != prev and t != blank:
                collapsed.append(t)
            prev = t
        return self.tok.decode(collapsed, skip_special_tokens=True)

    def _slice_cer(self, rows, device, ctx, image_col, text_col):
        n = min(CER_EVAL_SAMPLES, len(rows))
        gen_preds, tf_preds, ctc_preds, refs = [], [], [], []
        for i in range(n):
            ex = rows[i]
            pix = self.prep(ex[image_col]).unsqueeze(0).to(device)  # (1, 1, H, W)
            ref_ids = self.tok.encode(ex[text_col])
            cap = min(ctx - 1, int(1.5 * len(ref_ids)) + 10)        # length-aware cap

            # ---- ONE encoder forward per image, shared by all three decoders ----
            # B=1 un-padded -> input_lengths=None, so key_padding_mask (and cross_key_mask)
            # is None. enc_raw feeds the CTC head; bridged (enc_raw -> enc_to_dec) feeds the
            # text decoder's cross-attention. Mirrors EncoderDecoder.forward exactly.
            with torch.no_grad():
                enc_raw, key_padding_mask = self.model.encoder_model.encode(pix, None)  # (1, T, D)
                bridged = self.model.enc_to_dec(enc_raw)                                # (1, T, dec_dim)
            cross_key_mask = (None if key_padding_mask is None
                              else (~key_padding_mask).unsqueeze(1).unsqueeze(2))

            gen_ids = self.model.generate(                                    # text decoder (generation)
                pix, self.tok.bos_token_id, self.tok.eos_token_id,
                max_new_tokens=cap, enc_out=bridged, cross_key_mask=cross_key_mask)
            gen_preds.append(self.tok.decode(gen_ids, skip_special_tokens=True))
            tf_preds.append(                                                  # text decoder (teacher-forced)
                self._teacher_forced_pred(bridged, cross_key_mask, ref_ids, device))
            ctc_preds.append(self._ctc_pred(enc_raw))                         # CTC decoder
            refs.append(ex[text_col])
        return {
            "gen_cer": compute_cer(gen_preds, refs),
            "tf_cer": compute_cer(tf_preds, refs),
            "ctc_cer": compute_cer(ctc_preds, refs),
            "gen": gen_preds, "tf": tf_preds, "ctc": ctc_preds, "refs": refs,
        }

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # RANK-0 ONLY. This CER is monitoring-only (no early stopping / best-model
        # selection reads it), and the generation is a long, collective-free stretch of
        # work. Running it independently on every rank lets the ranks desync — one
        # finishes and hits the next NCCL barrier while the other is still decoding — and
        # blows past ddp_timeout (the 30-min ALLGATHER watchdog that killed the run).
        # Doing it on rank 0 alone keeps control flow deterministic; the other ranks skip
        # to the following save/step barrier and wait (ddp_timeout is raised to cover it).
        if not state.is_world_process_zero:
            return
        try:
            self.model.eval()
            device = next(self.model.parameters()).device
            ctx = self.model.decoder_model.positional_encodings.shape[0]
            image_col, text_col = self.image_col, self.text_col
            per_gen, per_tf, per_ctc = {}, {}, {}
            for name, rows in self.slices.items():
                if not rows:
                    continue
                r = self._slice_cer(rows, device, ctx, image_col, text_col)
                per_gen[name] = r["gen_cer"]
                per_tf[name] = r["tf_cer"]
                per_ctc[name] = r["ctc_cer"]
                if metrics is not None:
                    metrics[f"eval_{name}_cer"] = r["gen_cer"]          # text decoder (generation)
                    metrics[f"eval_{name}_tf_cer"] = r["tf_cer"]        # text decoder (teacher-forced)
                    metrics[f"eval_{name}_ctc_cer"] = r["ctc_cer"]      # CTC decoder
                print(f"[eval] step {state.global_step}  {name} genCER={r['gen_cer']:.4f} "
                      f"tfCER={r['tf_cer']:.4f} ctcCER={r['ctc_cer']:.4f} "
                      f"(n={min(CER_EVAL_SAMPLES, len(rows))})", flush=True)
                for gp, tp, cp, rf in list(zip(r["gen"], r["tf"], r["ctc"], r["refs"]))[:CER_PRINT_K]:
                    print(f"    [{name}] ref: {rf!r}")
                    print(f"    [{name}] gen: {gp!r}")
                    print(f"    [{name}] tf : {tp!r}")
                    print(f"    [{name}] ctc: {cp!r}")
            if per_gen:
                macro = sum(per_gen.values()) / len(per_gen)          # macro-avg text-decoder generation CER
                macro_tf = sum(per_tf.values()) / len(per_tf)         # macro-avg text-decoder teacher-forced CER
                macro_ctc = sum(per_ctc.values()) / len(per_ctc)      # macro-avg CTC-decoder CER
                if metrics is not None:
                    metrics["eval_cer"] = macro
                    metrics["eval_tf_cer"] = macro_tf
                    metrics["eval_ctc_cer"] = macro_ctc
                print(f"[eval] step {state.global_step}  macro genCER={macro:.4f} "
                      f"tfCER={macro_tf:.4f} ctcCER={macro_ctc:.4f}", flush=True)
                try:
                    import wandb
                    wandb.log({**{f"eval_{k}_cer": v for k, v in per_gen.items()},
                               **{f"eval_{k}_tf_cer": v for k, v in per_tf.items()},
                               **{f"eval_{k}_ctc_cer": v for k, v in per_ctc.items()},
                               "eval_cer": macro, "eval_tf_cer": macro_tf,
                               "eval_ctc_cer": macro_ctc}, step=state.global_step)
                except Exception:
                    pass
        except Exception as exc:
            print(f"[CER] failed at step {state.global_step}: {exc}", flush=True)
        finally:
            self.model.train()  # restore train mode for the rest of training


class OCRCollator:
    """
    Each dataset example is expected to be a dict with:
      - 'pixel_values': float tensor (1, H, W_i)   # H fixed (image_height=64), W_i variable
      - 'input_ids'   : 1D long tensor / list      # variable length, NO padding yet

    Produces a batch dict whose keys match EncoderDecoder.forward exactly:
      pixel_values      (B, 1, H, W_max)   float, right-padded on width with zeros
      input_ids         (B, T_max)         long, right-padded with pad_token_id
      input_lengths     (B,)               long, valid CTC frames = W_i // downsample
      text_padding_mask (B, T_max)         float, 1 = REAL token  (decoder convention)

    The CTC encoder collapses image height to 1, so each image contributes
    W_i // downsample frame tokens. ``input_lengths`` tells encode() how many of
    the padded-batch frames are real; it builds the cross-attention key mask from it.

    Assumes every image width W_i is already a multiple of ``downsample`` (your
    ImagePreprocessor pads to that). If not, the batch is padded up to a multiple.
    """

    def __init__(self, pad_token_id: int, downsample: int = 8, ctc_strip_ids=()):
        self.pad_token_id = pad_token_id
        self.downsample = downsample
        # Token ids removed from input_ids to form the CTC grapheme targets. Pass the
        # BOS / EOS ids: input_ids are [BOS, graphemes..., EOS], and the CTC head is
        # trained on the bare grapheme sequence (same convention as the CTC encoder).
        self.ctc_strip_ids = set(ctc_strip_ids)

    def __call__(self, batch):
        images = [ex["pixel_values"] for ex in batch]
        token_seqs = [torch.as_tensor(ex["input_ids"], dtype=torch.long) for ex in batch]
        B = len(batch)

        # ---- Images ----
        widths = [img.shape[-1] for img in images]
        max_w = max(widths)
        if max_w % self.downsample != 0:  # safety; should already be a multiple
            max_w += self.downsample - (max_w % self.downsample)

        padded_imgs = [F.pad(img, (0, max_w - w)) for img, w in zip(images, widths)]
        pixel_values = torch.stack(padded_imgs, dim=0)  # (B, 1, H, max_w)

        # Valid frame count per sample = real_width // downsample (ceil so a partial
        # edge frame is counted as real, not padded).
        input_lengths = torch.tensor(
            [min((w + self.downsample - 1) // self.downsample, max_w // self.downsample) for w in widths],
            dtype=torch.long,
        )

        # ---- Text (right-pad) ----
        max_t = max(seq.shape[0] for seq in token_seqs)
        input_ids = torch.full((B, max_t), self.pad_token_id, dtype=torch.long)
        text_padding_mask = torch.zeros((B, max_t), dtype=torch.float)  # 1 = real
        for i, seq in enumerate(token_seqs):
            n = seq.shape[0]
            input_ids[i, :n] = seq
            text_padding_mask[i, :n] = 1.0

        # ---- CTC targets: bare grapheme ids (BOS/EOS stripped), right-padded ----
        # Padded with 0; only the first ctc_label_lengths[i] entries per row are read by
        # nn.CTCLoss, so the pad value is irrelevant.
        ctc_seqs = [
            torch.as_tensor([t for t in seq.tolist() if t not in self.ctc_strip_ids],
                            dtype=torch.long)
            for seq in token_seqs
        ]
        ctc_label_lengths = torch.tensor([s.shape[0] for s in ctc_seqs], dtype=torch.long)
        max_s = max(int(ctc_label_lengths.max().item()), 1)
        ctc_labels = torch.zeros((B, max_s), dtype=torch.long)
        for i, s in enumerate(ctc_seqs):
            ctc_labels[i, :s.shape[0]] = s

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "text_padding_mask": text_padding_mask,
            "ctc_labels": ctc_labels,
            "ctc_label_lengths": ctc_label_lengths,
        }


# ============================================================================
# Optimizer / scheduler — PORTED from train_ctc_encoder.py.
# ============================================================================
# The three LR tiers for STAGE-2 full training. Each parameter is assigned to exactly
# one tier by name (see _param_tier); tiers get independent peak (max) and floor (min)
# learning rates.
#   'encoder' -> the image encoder backbone AND its CTC head (encoder_model.*)
#   'cross'   -> the cross-attention adapters (cross_attention_layer, layer_norm1_5,
#                cross_attn_gate) PLUS the enc_to_dec bridge that feeds the cross-attention
#                keys/values
#   'lm'      -> everything else in the decoder (token embeddings, self-attention, MLPs,
#                layer norms, lm_head) — the pretrained language model
LR_TIERS = ("encoder", "cross", "lm")
_CROSS_KEYS = ("cross_attention_layer", "layer_norm1_5", "cross_attn_gate")


def _param_tier(name: str) -> str:
    if name.startswith("encoder_model."):
        return "encoder"                                  # backbone + ctc_head
    if name.startswith("enc_to_dec."):
        return "cross"                                    # bridge feeds cross-attn K/V
    if any(k in name for k in _CROSS_KEYS):
        return "cross"                                    # cross-attention adapters
    return "lm"                                           # rest of the decoder / LM


def build_optimizer(model, lrs, min_lrs, weight_decay, betas):
    """Three LR tiers (encoder / cross-attention / LM), each with its own peak LR.

    `lrs` and `min_lrs` are dicts keyed by LR_TIERS giving each tier's peak (max) and
    cosine floor (min) LR. The per-tier floor is attached to each param group as
    'min_lr' so build_scheduler can decay every group from its own peak to its own min.

    Biases / norm weights / positional embeddings get NO weight decay (matches the
    prior convention); other 2-D weights get `weight_decay`."""
    tiers = {t: {"decay": [], "nodecay": []} for t in LR_TIERS}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        t = _param_tier(name)
        if "pos_embed" in name or p.ndim < 2 or "norm" in name.lower():
            tiers[t]["nodecay"].append(p)                 # embeddings/bias/norm -> no WD
        else:
            tiers[t]["decay"].append(p)

    groups = []
    for t in LR_TIERS:
        if tiers[t]["decay"]:
            groups.append({"params": tiers[t]["decay"], "lr": lrs[t],
                           "min_lr": min_lrs[t], "weight_decay": weight_decay})
        if tiers[t]["nodecay"]:
            groups.append({"params": tiers[t]["nodecay"], "lr": lrs[t],
                           "min_lr": min_lrs[t], "weight_decay": 0.0})
    return torch.optim.AdamW(groups, betas=betas)


# Scheduler: per-group linear warmup -> cosine decay from the group's own peak (group
# 'lr') to the group's own floor (group 'min_lr'). Each of the three LR tiers therefore
# rides its own cosine curve between its configured max and min.
def build_scheduler(optimizer, warmup_steps, total_steps):
    from torch.optim.lr_scheduler import LambdaLR

    def make(peak, floor):
        def f(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            prog = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
            cos = 0.5 * (1.0 + math.cos(math.pi * prog))
            return (floor + (peak - floor) * cos) / peak  # peak -> floor, this group's own
        return f

    return LambdaLR(optimizer, [make(g["lr"], g["min_lr"]) for g in optimizer.param_groups])


class EncoderDecoderTrainer(Trainer):
    """Trainer that builds the per-tier warmup->cosine-to-min_lr scheduler with the
    step count Trainer computes (so we don't reason about DDP/epochs ourselves).

    STAGE-2 (full end-to-end): the whole network trains, so the training forward runs
    in the normal ``.train()`` mode — dropout / stochastic-depth (DropPath) stay ON for
    regularization. (Stage-1 forced ``model.eval()`` inside ``compute_loss`` to keep the
    frozen backbone deterministic while only the adapters warmed up; that mechanism is
    removed here since nothing is frozen.)
    """

    def __init__(self, *args, warmup_steps=1500, **kwargs):
        self._warmup_steps = warmup_steps
        super().__init__(*args, **kwargs)

    def create_scheduler(self, num_training_steps, optimizer=None):
        if self.lr_scheduler is None:
            self.lr_scheduler = build_scheduler(
                optimizer or self.optimizer, self._warmup_steps, num_training_steps)
        return self.lr_scheduler

    def log(self, logs, *args, **kwargs):
        # Surface the CE / CTC breakdown behind the combined training loss so it reaches
        # the console and wandb alongside `loss`. Values are the last micro-batch's
        # components (a close proxy for the logged running mean) — enough to watch how the
        # two terms trade off while tuning ctc_loss_weight. Only added to TRAIN logs
        # (gated on 'loss'), never eval logs.
        if "loss" in logs:
            core = self.model.module if hasattr(self.model, "module") else self.model
            ce = getattr(core, "_ce_loss", None)
            ctc = getattr(core, "_ctc_loss", None)
            if ce is not None:
                logs["ce_loss"] = float(ce)
            if ctc is not None:
                logs["ctc_loss"] = float(ctc)
        return super().log(logs, *args, **kwargs)


class GradNormAlert(TrainerCallback):
    """PORTED from train_ctc_encoder.py: alert on grad-norm spikes > threshold."""

    def __init__(self, threshold=10.0):
        self.threshold = threshold

    def on_log(self, args, state, control, logs=None, **kwargs):
        gn = (logs or {}).get("grad_norm")
        if gn is not None and gn > self.threshold:
            print(f"[ALERT] grad_norm={gn:.2f} > {self.threshold} at step {state.global_step}")


# ============================================================================
# Eval slices — PORTED from train_ctc_encoder.py, adapted to the encoder-decoder
# data format (each item is {pixel_values, input_ids} for OCRCollator).
# ============================================================================
class ListEvalDataset(torch.utils.data.Dataset):
    """In-memory map-style eval dataset over raw row dicts. Applies the (no-aug)
    eval preprocessing per item and produces the encoder-decoder input format."""

    def __init__(self, rows, preprocessor, tokenizer, image_col, text_col):
        self.rows = rows
        self.prep = preprocessor       # clean ImagePreprocessor (augment_fn=None)
        self.tok = tokenizer
        self.image_col = image_col
        self.text_col = text_col

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return {
            "pixel_values": self.prep(r[self.image_col]),          # (1, H, W_i)
            "input_ids": self.tok.encode(r[self.text_col]),        # BOS ... EOS
        }


def build_eval_slices(rows, source_col, slice_cap, seed):
    """Partition materialized eval rows by TEXT SOURCE (natural / random) when the
    source column is present; otherwise a single 'all' slice. Returns a dict
    ``{slice_name: [raw_row, ...]}`` (raw rows, for the generate-based CER).

    (Width-based short/long slicing was dropped: two source slices keep the total
    generation count small enough for the rank-0-only CER eval.)"""
    has_src = bool(rows) and source_col in rows[0]

    def subset(predicate):
        idx = [i for i in range(len(rows)) if predicate(i)]
        if not idx:
            return None
        if slice_cap and len(idx) > slice_cap:
            g = torch.Generator().manual_seed(seed)
            idx = [idx[j] for j in torch.randperm(len(idx), generator=g)[:slice_cap].tolist()]
        return [rows[i] for i in idx]

    slices = {}
    if has_src:
        for s in ("natural", "random"):
            sub = subset(lambda i, ss=s: rows[i].get(source_col) == ss)
            if sub is not None:
                slices[s] = sub
    else:
        sub = subset(lambda i: True)
        if sub is not None:
            slices["all"] = sub
    print(f"[eval] slices: { {k: len(v) for k, v in slices.items()} }")
    return slices


def find_last_checkpoint(output_dir, prev_run_dir):
    for d in (output_dir, prev_run_dir):
        if d and os.path.isdir(d):
            ckpt = get_last_checkpoint(d)
            if ckpt is not None:
                return ckpt
    return None


if __name__ == '__main__':

    # DDP topology sanity: this line prints ONCE PER PROCESS running the script as
    # __main__ (i.e. per rank). If you see it more times than --nproc_per_node, extra
    # processes are running. Read the fields to tell WHICH:
    #   * The 2 real ranks share one ppid (the torchrun agent) and one MASTER_PORT,
    #     with LOCAL_RANK 0 and 1.
    #   * A straggler from an EARLIER torchrun has a DIFFERENT MASTER_PORT (and usually
    #     ppid=1, reparented to init after its launcher died) -> kill it / restart kernel.
    print(f"[proc] pid={os.getpid()} ppid={os.getppid()} "
          f"RANK={os.environ.get('RANK')} LOCAL_RANK={os.environ.get('LOCAL_RANK')} "
          f"WORLD_SIZE={os.environ.get('WORLD_SIZE')} MASTER_PORT={os.environ.get('MASTER_PORT')}",
          flush=True)

    tokenizer = TeluguGraphemeTokenizer(
        vocab_file="/kaggle/input/datasets/harshadesaraju99/telugu-tokenizer-vocab/telugu-vocab.json")
    print(tokenizer.pad_token_id)

    # -------------- Step-1: Model configs (must match the STAGE-1 checkpoint) --------------
    # Decoder (GPT) config — the language model stage-1 fine-tuned.
    decoder_config = GPTConfig(
        vocab_size=len(tokenizer),
        embed_dim=512,
        hidden_dim=1368,  # 2.67 * 512 = 2/3 * 4 * hidden_dim
        num_heads=8,
        num_layers=16,
        ctx_len=256,
        dropout=0.05
    )
    # Image encoder (CTC) config — must match the checkpoint.
    img_encoder_cfg = CTCEncoderConfig(
        max_image_width=2048,
        max_frames=256,
    )

    # -------------- Step-2: Build the encoder-decoder model --------------
    # Combined objective: loss = CE + CTC_LOSS_WEIGHT * CTC (tune / set 0.0 to disable CTC).
    model = EncoderDecoder(
        encoder_config=img_encoder_cfg,
        decoder_config=decoder_config,
        pad_index=tokenizer.pad_token_id,
        ctc_loss_weight=CTC_LOSS_WEIGHT,
        # STAGE-2 trains end-to-end: gradients MUST reach the encoder.
        encoder_no_grad=False,
    )

    # -------------- Step-3: Load the STAGE-1 weights (ONE checkpoint) --------------
    # Stage-1 already trained the image encoder, the cross-attention layers and the
    # language model TOGETHER, so the whole EncoderDecoder loads from a single state dict
    # (no separate GPT / CTC-encoder loading + weight transfer + verification any more).
    STAGE1_MODEL_PATH = "/kaggle/input/models/harshadesaraju99/telugu-ocr-model-stage-1/pytorch/default/1/final_model.pt"  # <-- set to your stage-1 EncoderDecoder state dict
    load_result = model.load_state_dict(
        torch.load(STAGE1_MODEL_PATH, map_location="cpu"),
        strict=False,
    )
    # The stage-2 model has the SAME parameters as stage-1 — including the per-block
    # `cross_attn_gate` (the trained tanh gate is loaded, not re-initialised) — so the load
    # must be EXACT: neither missing nor unexpected keys. This guarantees the model starts
    # identical to stage-1 before end-to-end refinement.
    print(f"[load] stage-1 -> encoder-decoder: "
          f"missing={list(load_result.missing_keys)} "
          f"unexpected={list(load_result.unexpected_keys)}", flush=True)
    assert not load_result.missing_keys, \
        f"stage-1 checkpoint is missing weights for: {load_result.missing_keys}"
    assert not load_result.unexpected_keys, \
        f"stage-1 checkpoint has unexpected weights: {load_result.unexpected_keys}"

    # -------------- Step-4: FULL / STAGE-2 TRAINING — no freezing --------------
    # The entire network trains end-to-end: the image encoder (+ its CTC head), the new
    # cross-attention adapters, and the pretrained language model. Nothing is frozen; each
    # group is instead trained at its own LR tier (see build_optimizer / LR / MIN_LR below).
    # The encoder must receive gradients, which EncoderDecoder handles via
    # encoder_no_grad=False (set in __init__ for stage-2).
    assert not model.encoder_no_grad, "STAGE-2 needs gradients through the encoder"

    # -------------- Step-5: Sanity check on the trainable layers --------------
    trainable_layers = []
    for name, params in model.named_parameters():
        if params.requires_grad:
            trainable_layers.append(name)

    # print("List of trainable layers in the model:")
    # for name in trainable_layers:
    #     print(name)

    # -------------- Print model sizes --------------
    encoder_params = 0
    for params in model.encoder_model.parameters():
        encoder_params += params.numel()
    print(f"No. of parameters in encoder: {encoder_params}")

    decoder_params = 0
    for params in model.decoder_model.parameters():
        decoder_params += params.numel()
    print(f"No. of parameters in decoder: {decoder_params}")

    trainable_params = 0
    for params in model.parameters():
        if params.requires_grad:
            trainable_params += params.numel()
    print(f"No. of trainable parameters: {trainable_params}")
    print(f"Percentage of trainable parameters: {((trainable_params / (encoder_params + decoder_params)) * 100):.2f}%")

    # ---------------------------- Data config (PORTED: same datasets as train_ctc_encoder.py) ----
    DATASET_NAME1 = "harsha-desaraju/sample-dataset-new"          # synthetic (has text_source)
    DATASET_NAME2 = "harsha-desaraju/telugu-wikisource-text-images"  # real PDF-line crops
    DATASET_SPLIT = "train"          # NORMAL (downloaded, map-style) dataset
    EVAL_SPLIT = "validation"        # small held-out split for the eval slices
    IMAGE_COLUMN = "image"
    TEXT_COLUMN = "text"
    SOURCE_COLUMN = "text_source"    # natural / random / real; drives augmentation + eval slices
    TRAIN_CONFIGS = [
                    # 'train_0000', 'train_0001', 'train_0002', 'train_0003',
                     'train_0004', 'train_0005',
                    # 'train_0006', 'train_0007',
                     ]
    RND_SAM_FRAC = 0.05

    # frames / widths
    DOWNSAMPLE = 8                   # conv-stem width reduction (T = W // DOWNSAMPLE)
    IMAGE_HEIGHT = 64
    MAX_IMAGE_WIDTH = 2048           # matches the CTC encoder (max_frames=256)

    # batching — T4 (16 GB): stage-1 trains only cross-attn + gates + bridge, but the
    # full 16-layer decoder forward is retained for backprop, so keep the per-device
    # batch modest and recover global batch via grad-accum. Raise if you have headroom.
    PER_DEVICE_BATCH = 32
    GRAD_ACCUM = 2                   # global batch = PER_DEVICE_BATCH * GRAD_ACCUM * #GPUs
    EVAL_BATCH = 32                  # smaller (long slices can be up to 2048px)

    # optimizer / schedule — THREE LR tiers (encoder / cross-attention / LM), each with
    # its OWN cosine peak (max, LR) and floor (min, MIN_LR). warmup -> per-tier cosine
    # from LR[tier] down to MIN_LR[tier]. Defaults: the freshly-added cross-attention
    # adapters ride the highest LR, the pretrained LM the lowest (light refinement), the
    # encoder in between. Tune per your run.
    LR = {                           # per-tier PEAK (max) LR
        "encoder": 3e-5,
        "cross":   5e-5,
        "lm":      3e-5,
    }
    MIN_LR = {                       # per-tier cosine FLOOR (min) LR
        "encoder": 3e-6,
        "cross":   5e-6,
        "lm":      3e-6,
    }
    WARMUP_STEPS = 2000
    WEIGHT_DECAY = 0.05
    BETAS = (0.9, 0.98)
    EPOCHS = 5

    # eval / io
    EVAL_SLICE_CAP = 1500            # cap each eval slice for speed
    EVAL_LOSS_CAP = 512              # rows used for the (teacher-forced) eval_loss set
    SAVE_EVAL_STEPS = 1000
    OUTPUT_DIR = "/kaggle/working/telugu-ocr-stage2"
    PREV_RUN_DIR = None             # prior run dir re-mounted read-only, for 12h resume

    SEED = 42
    set_seed(SEED)                   # reproducible augmentation (random + np.random)

    # ---- Preprocessors: train augmented (composed degrade pipeline), eval clean ----
    train_preprocessor = LineTensorizer(augment_fn=make_augmenter(p_clean=P_CLEAN))
    eval_preprocessor = LineTensorizer()


    # Sources whose images are NOT augmented: the real PDF crops already carry real-world
    # degradation, so the synthetic augmentation pipeline is applied only to 'natural' /
    # 'random'. Everything not in this set gets the augmenting preprocessor.
    NO_AUG_SOURCES = {"real"}

    def make_sample_transformer(aug_preprocessor, clean_preprocessor):
        def sample_transformer(batch):
            sources = batch[SOURCE_COLUMN]
            # Per-image: real -> clean preprocessing; natural / random -> augmented.
            images = [
                (clean_preprocessor if src in NO_AUG_SOURCES else aug_preprocessor)(img)
                for img, src in zip(batch[IMAGE_COLUMN], sources)
            ]                                                               # list of (1, H, W_i)
            input_ids = [tokenizer.encode(t) for t in batch[TEXT_COLUMN]]   # BOS ... EOS
            return {"pixel_values": images, "input_ids": input_ids}
        return sample_transformer


    # ---- Train: synthetic configs (DATASET_NAME1) + real PDF-line crops (DATASET_NAME2) ----
    train_parts = [
        load_dataset(DATASET_NAME1, c, columns=[IMAGE_COLUMN, TEXT_COLUMN, SOURCE_COLUMN])[DATASET_SPLIT]
        for c in TRAIN_CONFIGS
    ]
    # 2nd dataset — the real PDF crops; tag them 'real' so they are (a) kept in full and
    # (b) NOT augmented (they already carry real degradation). See NO_AUG_SOURCES above.
    real_ds = load_dataset(DATASET_NAME2, split=DATASET_SPLIT, columns=[IMAGE_COLUMN, TEXT_COLUMN])
    real_ds = real_ds.add_column(SOURCE_COLUMN, ['real'] * len(real_ds))
    train_parts.append(real_ds)

    train_hf = concatenate_datasets(train_parts, axis=0)

    # Make the random samples only a given percentage
    indices = defaultdict(list)

    for i, value in enumerate(train_hf["text_source"]):
        indices[value].append(i)

    nat_ds = train_hf.select(indices["natural"])
    real_ds = train_hf.select(indices['real'])
    rnd_ds = train_hf.select(indices["random"])

    num_rnd = int((RND_SAM_FRAC/(1+RND_SAM_FRAC))*(len(nat_ds)+len(real_ds)))
    rnd_ds = rnd_ds.shuffle(seed=SEED).select(range(num_rnd))

    train_hf = concatenate_datasets([nat_ds, real_ds, rnd_ds], axis=0)
    train_hf = train_hf.shuffle(seed=SEED)

    del nat_ds, rnd_ds, real_ds
    gc.collect()

    # Augmenting preprocessor for natural/random, clean (eval) preprocessor for real.
    train_dataset = train_hf.with_transform(
        make_sample_transformer(train_preprocessor, eval_preprocessor))
    print(f"[data] train -> {len(train_hf)} rows; cols {train_hf.column_names}")

    # ---- Validation: materialize rows, build width/source eval slices ----
    val_ds = load_dataset(DATASET_NAME1, EVAL_SPLIT)[EVAL_SPLIT]
    val_rows = list(val_ds)
    print(f"[data] validation -> {len(val_rows)} rows")
    eval_slices = build_eval_slices(val_rows, SOURCE_COLUMN, EVAL_SLICE_CAP, SEED)

    # A single small transformed eval set drives Trainer's eval_loss and fires
    # on_evaluate exactly ONCE; per-slice generate-based CER is added by CEREvalCallback.
    eval_ds = ListEvalDataset(val_rows[:EVAL_LOSS_CAP], eval_preprocessor, tokenizer,
                              IMAGE_COLUMN, TEXT_COLUMN)

    # BOS/EOS are stripped from input_ids to form the CTC grapheme targets.
    data_collator = OCRCollator(pad_token_id=tokenizer.pad_token_id, downsample=DOWNSAMPLE,
                                ctc_strip_ids=(tokenizer.bos_token_id, tokenizer.eos_token_id))

    # ---- Optimizer (fresh, three LR tiers); scheduler built by EncoderDecoderTrainer ----
    optimizer = build_optimizer(model, LR, MIN_LR, WEIGHT_DECAY, BETAS)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=EVAL_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_grad_norm=1.0,                       # PORTED: grad clip
        # NCCL collective timeout (default 1800s = 30min). The rank-0-only CER eval makes
        # the other ranks idle-wait at the next barrier; raise this so a slow eval / save
        # can never trip the watchdog. This is the timeout that killed the 2-GPU run.
        ddp_timeout=3600,                        # 2h

        # Optimizer + LR are supplied explicitly via `optimizers=` below (build_optimizer),
        # and the warmup->cosine-to-min_lr scheduler is built by create_scheduler, so no
        # optim / learning_rate / weight_decay / betas / lr_scheduler_type are set here.

        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        ignore_data_skip=True,                   # fast resume (don't replay skipped data)

        fp16=torch.cuda.is_available(),          # T4 -> fp16 + GradScaler
        gradient_checkpointing=False,
        # CPU budget: Kaggle 2x T4 = 4 vCPUs, and this runs 2-rank DDP (one process per
        # GPU), so dataloader_num_workers is PER RANK -> total workers = 2 * num_workers.
        # 2 keeps it at 4 worker processes = 4 vCPUs (one per core); 4 would spawn 8 on 4
        # cores and thrash. Each worker is pinned to 1 math-lib thread (LIMIT_AUG_THREADS
        # at the top of the file), so single-threaded-per-worker + no oversubscription is
        # the fastest split for the CPU-bound augmentation pipeline.
        dataloader_num_workers=2,
        dataloader_pin_memory=True,              # faster H2D copy; pinning thread overlaps compute
        dataloader_prefetch_factor=4,            # 4 batches buffered/worker to smooth augment bursts
        dataloader_persistent_workers=True,      # don't re-spawn workers each epoch (startup cost)

        eval_strategy="steps",
        eval_steps=SAVE_EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_EVAL_STEPS,
        save_total_limit=2,
        load_best_model_at_end=False,            # PORTED: no best-model tracking / no early stopping

        # No per-step tqdm bar: 36k \r-updates on stderr flood the notebook pipe (the
        # other half of the stdout-backpressure hang). Progress is visible via the
        # 50-step log lines instead.
        disable_tqdm=True,

        logging_strategy="steps",
        logging_steps=50,
        logging_first_step=True,
        report_to="wandb",
        run_name="stage-2-finetuning",
        seed=SEED,
    )

    trainer = EncoderDecoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_ds,                    # drives eval_loss (fires on_evaluate once)
        data_collator=data_collator,
        optimizers=(optimizer, None),            # scheduler built by create_scheduler
        callbacks=[
            # Per-slice generate-based CER -> eval_<slice>_cer + macro eval_cer.
            CEREvalCallback(model, eval_slices, eval_preprocessor, tokenizer,
                            image_col=IMAGE_COLUMN, text_col=TEXT_COLUMN),
            GradNormAlert(threshold=10.0),
        ],
        warmup_steps=WARMUP_STEPS,
    )

    # ---- Resume: this run's checkpoints, else a prior-run dir (PORTED) ----
    if PREV_RUN_DIR:
        last_ckpt = PREV_RUN_DIR
    else:
        last_ckpt = find_last_checkpoint(OUTPUT_DIR, PREV_RUN_DIR)
    print(f"Resuming from: {last_ckpt}" if last_ckpt else "Starting fresh (no checkpoint found)")

    trainer.train(resume_from_checkpoint=last_ckpt)

    # Save final artifacts (HF checkpoint + stage-1-style .pt)
    trainer.save_model(OUTPUT_DIR)
    if trainer.is_world_process_zero():
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "final_model.pt"))
        print("Finished stage-1 training!")