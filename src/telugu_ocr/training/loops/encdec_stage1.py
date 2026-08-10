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
# ADDED (stage-1): CER evaluation utilities + callback.
# ============================================================================
from src.telugu_ocr.metrics.errors import _edit_distance, compute_cer
from src.telugu_ocr.training.callbacks import GradNormAlert
from src.telugu_ocr.training.checkpoint import find_last_checkpoint
from src.telugu_ocr.training.eval_slices import ListEvalDataset, build_eval_slices
from src.telugu_ocr.training.optim import build_optimizer, build_scheduler




# B=1 greedy generation with no KV cache is SLOW (~seconds/sequence). This runs on
# rank 0 only while the other ranks wait at the next barrier, so keep the TOTAL small:
# at 48/slice x 2 slices (natural / random) that is ~96 sequences, a few minutes —
# comfortably inside the raised ddp_timeout. Raise for a sharper CER estimate only if
# you also raise ddp_timeout.
CER_EVAL_SAMPLES = 48   # eval images decoded per slice per CER eval (B=1 greedy)
CER_PRINT_K = 5  # print this many ref/hyp pairs each eval


class CEREvalCallback(TrainerCallback):
    """Generate-based CER during evaluate(), reported PER WIDTH SLICE.

    ``eval_slices`` is a dict ``{slice_name: [raw_row, ...]}`` where each raw row
    has the image / text columns. For each slice we B=1 greedy-decode up to
    CER_EVAL_SAMPLES images and inject ``eval_<slice>_cer`` into the metrics dict
    (also logged to wandb). A macro-average ``eval_cer`` over slices is added too.

    Early stopping was removed, so this metric is purely for monitoring; CER is
    still computed on every rank (identical subset -> identical value) while
    printing / wandb logging happen on rank 0 only.
    """

    def __init__(self, model, eval_slices, preprocessor, tokenizer,
                 image_col="image", text_col="text"):
        self.model = model
        self.slices = eval_slices  # {name: list of raw rows with image/text cols}
        self.prep = preprocessor   # clean preprocessor (no augmentation)
        self.tok = tokenizer
        self.image_col = image_col
        self.text_col = text_col

    def _slice_cer(self, rows, device, ctx, image_col, text_col):
        n = min(CER_EVAL_SAMPLES, len(rows))
        preds, refs = [], []
        for i in range(n):
            ex = rows[i]
            pix = self.prep(ex[image_col]).unsqueeze(0).to(device)  # (1, 1, H, W)
            ref_ids = self.tok.encode(ex[text_col])
            cap = min(ctx - 1, int(1.5 * len(ref_ids)) + 10)        # length-aware cap
            ids = self.model.generate(pix, self.tok.bos_token_id, self.tok.eos_token_id,
                                      max_new_tokens=cap)
            preds.append(self.tok.decode(ids, skip_special_tokens=True))
            refs.append(ex[text_col])
        return compute_cer(preds, refs), preds, refs

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
            per_slice = {}
            for name, rows in self.slices.items():
                if not rows:
                    continue
                cer, preds, refs = self._slice_cer(rows, device, ctx, image_col, text_col)
                per_slice[name] = cer
                if metrics is not None:
                    metrics[f"eval_{name}_cer"] = cer
                print(f"[eval] step {state.global_step}  {name} CER={cer:.4f} (n={min(CER_EVAL_SAMPLES, len(rows))})", flush=True)
                for p, r in list(zip(preds, refs))[:CER_PRINT_K]:
                    print(f"    [{name}] ref: {r!r}")
                    print(f"    [{name}] hyp: {p!r}")
            if per_slice:
                macro = sum(per_slice.values()) / len(per_slice)  # macro-average across slices
                if metrics is not None:
                    metrics["eval_cer"] = macro
                print(f"[eval] step {state.global_step}  macro CER={macro:.4f}", flush=True)
                try:
                    import wandb
                    wandb.log({**{f"eval_{k}_cer": v for k, v in per_slice.items()},
                               "eval_cer": macro}, step=state.global_step)
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

    def __init__(self, pad_token_id: int, downsample: int = 8):
        self.pad_token_id = pad_token_id
        self.downsample = downsample

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

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "text_padding_mask": text_padding_mask
        }


# ============================================================================
# Optimizer / scheduler — PORTED from train_ctc_encoder.py.
# ============================================================================


# Scheduler: warmup -> cosine decay to an ABSOLUTE floor `min_lr`. Single LR tier.


class EncoderDecoderTrainer(Trainer):
    """Trainer that builds the warmup->cosine-to-min_lr scheduler with the step
    count Trainer computes (so we don't reason about DDP/epochs ourselves).

    STAGE-1 ONLY — trains with the model held in ``.eval()`` mode
    =========================================================================
    In stage-1 the image encoder and the pretrained decoder layers are FROZEN and
    only the newly-added adapters (cross-attention + zero-init gates + the 384->512
    bridge) train. We do NOT want the frozen backbone's dropout / stochastic-depth
    (DropPath) injecting noise into the features the adapters learn from, so every
    TRAINING forward is run in eval mode (see ``compute_loss``).

    HF ``Trainer.training_step`` calls ``model.train()` at the start of every step,
    so a one-off ``model.eval()`` before ``trainer.train()`` would not stick; forcing
    it inside ``compute_loss`` (right before the forward) is what makes it hold. Note
    ``eval()`` only flips the dropout/DropPath flags — it does NOT stop gradients, so
    the adapters still train normally.

    This ALSO disables dropout inside the trainable cross-attention adapters, which is
    acceptable for stage-1 adapter warm-up.

    ⚠️  CHANGE FOR FULL / STAGE-2 TRAINING: once the backbone is unfrozen and trained
    end-to-end, set ``FORCE_EVAL_DURING_TRAIN = False`` (or drop the ``compute_loss``
    override) so dropout / stochastic depth are active again for regularization.
    """

    # Stage-1 default: run the training forward in eval mode. Flip to False for
    # full / end-to-end (stage-2) training where dropout should be ON.
    FORCE_EVAL_DURING_TRAIN = True

    def __init__(self, *args, warmup_steps=1500, min_lr=1e-5, **kwargs):
        self._warmup_steps = warmup_steps
        self._min_lr = min_lr
        super().__init__(*args, **kwargs)

    def create_scheduler(self, num_training_steps, optimizer=None):
        if self.lr_scheduler is None:
            self.lr_scheduler = build_scheduler(
                optimizer or self.optimizer, self._warmup_steps, num_training_steps, self._min_lr)
        return self.lr_scheduler

    def compute_loss(self, model, inputs, *args, **kwargs):
        # STAGE-1: keep the frozen backbone (and, acceptably, the adapters) deterministic
        # for the training forward — dropout / DropPath off. Trainer.training_step has
        # already called model.train(); we override it here, per step, right before the
        # forward. Gradients still flow, so the adapter layers train.
        # ⚠️ Remove / gate off (FORCE_EVAL_DURING_TRAIN=False) for full/stage-2 training.
        if self.FORCE_EVAL_DURING_TRAIN:
            model.eval()
        return super().compute_loss(model, inputs, *args, **kwargs)




# ============================================================================
# Eval slices — PORTED from train_ctc_encoder.py, adapted to the encoder-decoder
# data format (each item is {pixel_values, input_ids} for OCRCollator).
# ============================================================================






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
        vocab_file="/kaggle/input/datasets/harshadesaraju1999/telugu-tokenizer-vocab/telugu-vocab.json")
    print(tokenizer.pad_token_id)

    # -------------- Step-1: Load the pretrained models --------------
    # Pretrained text decoder model
    pretrained_config = GPTConfig(
        vocab_size=len(tokenizer),
        embed_dim=512,
        hidden_dim=1368,  # 2.67 * 512 = 2/3 * 4 * hidden_dim
        num_heads=8,
        num_layers=16,
        ctx_len=256,
        dropout=0.1
    )

    pretrained_text_model = GPTModel(pretrained_config, pad_index=tokenizer.pad_token_id)
    pretrained_text_model.load_state_dict(
        torch.load("/kaggle/input/models/harshadesaraju1999/telugugpt/pytorch/default/1/final_model.pt")
    )

    # Load the pretrained CTC image encoder (config must match the checkpoint).
    img_encoder_cfg = CTCEncoderConfig(
        max_image_width=2048,
        max_frames=256,
    )

    pretrained_encoder_model = ImageEncoderCTC(img_encoder_cfg)
    pretrained_encoder_model.load_state_dict(
        torch.load(
            '/kaggle/input/models/harshadesaraju1999/telugu-ctc-image-encoder/pytorch/default/1/final_model.pt',
        )
    )

    # -------------- Step-2: Initialize the new encoder-decoder model --------------
    model = EncoderDecoder(
        encoder_config=img_encoder_cfg,
        decoder_config=pretrained_config,
        pad_index=tokenizer.pad_token_id,
        # STAGE-1: the encoder is frozen, so keep it out of the autograd graph entirely
        # rather than relying on requires_grad -- that saves the encoder activations.
        # This was hardcoded inside stage-1's own copy of EncoderDecoder before phase 3.
        encoder_no_grad=True,
        # No CTC auxiliary term in stage 1; only the cross-attention is training.
        ctc_loss_weight=0.0,
    )

    # -------------- Step-3: Replace the random weights with pretrained weights --------------
    # Replace the text decoder weights
    match_result = model.decoder_model.load_state_dict(
        pretrained_text_model.state_dict(),
        strict=False
    )

    # Replace the image encoder weights (full CTC encoder, incl. the CTC head we won't use)
    model.encoder_model.load_state_dict(pretrained_encoder_model.state_dict())

    # -------------- Step-4: Verify the weight transfer for decoder model --------------
    # Verify the layers not matched are the newly added layers (cross-attention +
    # its layer norm + the zero-init tanh gate; present on even blocks only).
    newly_added_layers = ['cross_attention', 'layer_norm1_5', 'cross_attn_gate']
    all_matched = True
    for layer_name in match_result.missing_keys:
        is_new = any([new_layer in layer_name for new_layer in newly_added_layers])
        if not is_new:
            all_matched = False

    assert all_matched, "Some of the old layers' weights did not match"

    # -------------- Step-4.5: image contribution starts at zero --------------
    # The zero-init tanh gate inside each cross-attention block (cross_attn_gate,
    # tanh(0) == 0) already makes the image contribute NOTHING at init, so the model
    # starts EXACTLY as the pretrained LM and phases the image in gradually
    # (Flamingo/adapter style). No manual out_proj zeroing is needed anymore, and the
    # old per-block loop would AttributeError on the odd blocks that have no cross-attn.

    # -------------- Step-5: Freeze the weights --------------
    # Decoder
    # Now, freeze the old pretrained layers and allow only new layers to train
    for name, params in model.decoder_model.named_parameters():
        if name not in match_result.missing_keys:
            params.requires_grad = False

    # Do some sanity checks
    # 1 - Check if the newly added layers are the unfrozen layers
    unfrozen_layers = []
    for name, params in model.decoder_model.named_parameters():
        if params.requires_grad:
            unfrozen_layers.append(name)

    # 2 - Check the missing_keys layers and unfrozen layers match
    all_matched = True
    for name1, name2 in zip(match_result.missing_keys, unfrozen_layers):
        if name1 != name2:
            print(f"{name1} --- {name2}", flush=True)
            all_matched = False

    assert all_matched, "Some pretrained layers are not frozen!"

    # Encoder
    # Load and freeze the image encoder also
    for name, parameters in model.encoder_model.named_parameters():
        parameters.requires_grad = False

    # No need for any checks for encoder as the model configuration is the same

    # -------------- Step-6: Sanity check on the unfrozen layers --------------
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
    DATASET_NAME2 = "harsha-desaraju/telugu-pdf-line-image-text"  # real PDF-line crops
    DATASET_SPLIT = "train"          # NORMAL (downloaded, map-style) dataset
    EVAL_SPLIT = "validation"        # small held-out split for the eval slices
    IMAGE_COLUMN = "image"
    TEXT_COLUMN = "text"
    SOURCE_COLUMN = "text_source"    # natural / random -> eval slices (optional column)
    TRAIN_CONFIGS = ['train_0000', 'train_0001', 'train_0002', 'train_0003',
                     #'train_0004', 'train_0005', 'train_0006', 'train_0007',
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

    # optimizer / schedule (PORTED: single-tier build_optimizer + warmup -> cosine to min_lr)
    LR = 1e-4                        # single LR tier (matches train_ctc_encoder.py)
    MIN_LR = 1e-5
    WARMUP_STEPS = 2000
    WEIGHT_DECAY = 0.05
    BETAS = (0.9, 0.98)
    EPOCHS = 4

    # eval / io
    EVAL_SLICE_CAP = 1500            # cap each eval slice for speed
    EVAL_LOSS_CAP = 512              # rows used for the (teacher-forced) eval_loss set
    SAVE_EVAL_STEPS = 1000
    OUTPUT_DIR = "/kaggle/working/telugu-ocr-stage1"
    PREV_RUN_DIR = "/kaggle/input/models/harshadesaraju1999/telugu-ocr-checkpoint/transformers/default/1"              # prior run dir re-mounted read-only, for 12h resume

    SEED = 42
    set_seed(SEED)                   # reproducible augmentation (random + np.random)

    # ---- Preprocessors: train augmented (composed degrade pipeline), eval clean ----
    train_preprocessor = LineTensorizer(augment_fn=make_augmenter(p_clean=P_CLEAN))
    eval_preprocessor = LineTensorizer()


    def make_sample_transformer(preprocessor):
        def sample_transformer(batch):
            images = [preprocessor(img) for img in batch[IMAGE_COLUMN]]     # list of (1, H, W_i)
            input_ids = [tokenizer.encode(t) for t in batch[TEXT_COLUMN]]   # BOS ... EOS
            return {"pixel_values": images, "input_ids": input_ids}
        return sample_transformer


    # ---- Train: synthetic configs (DATASET_NAME1) + real PDF-line crops (DATASET_NAME2) ----
    train_parts = [
        load_dataset(DATASET_NAME1, c, columns=[IMAGE_COLUMN, TEXT_COLUMN, SOURCE_COLUMN])[DATASET_SPLIT]
        for c in TRAIN_CONFIGS
    ]
    # 2nd dataset — the real images; tag them all as 'natural' for the source column.
    real_ds = load_dataset(DATASET_NAME2, split=DATASET_SPLIT, columns=[IMAGE_COLUMN, TEXT_COLUMN])
    real_ds = real_ds.add_column(SOURCE_COLUMN, ['natural'] * len(real_ds))
    train_parts.append(real_ds)

    train_hf = concatenate_datasets(train_parts, axis=0)

    # Make the random samples only a given percentage
    indices = defaultdict(list)

    for i, value in enumerate(train_hf["text_source"]):
        indices[value].append(i)

    nat_ds = train_hf.select(indices["natural"])
    rnd_ds = train_hf.select(indices["random"])

    num_rnd = int((RND_SAM_FRAC/(1+RND_SAM_FRAC))*len(nat_ds))
    rnd_ds = rnd_ds.shuffle(seed=SEED).select(range(num_rnd))

    train_hf = concatenate_datasets([nat_ds, rnd_ds], axis=0)
    train_hf = train_hf.shuffle(seed=SEED)

    del nat_ds, rnd_ds
    gc.collect()

    train_dataset = train_hf.with_transform(make_sample_transformer(train_preprocessor))
    print(f"[data] train -> {len(train_hf)} rows; cols {train_hf.column_names}")

    # ---- Validation: materialize rows, build width/source eval slices ----
    val_ds = load_dataset(DATASET_NAME1, EVAL_SPLIT)[EVAL_SPLIT]
    val_rows = list(val_ds)
    print(f"[data] validation -> {len(val_rows)} rows")
    eval_slices = build_eval_slices(val_rows, SOURCE_COLUMN,
                                    EVAL_SLICE_CAP, SEED)

    # A single small transformed eval set drives Trainer's eval_loss and fires
    # on_evaluate exactly ONCE; per-slice generate-based CER is added by CEREvalCallback.
    eval_ds = ListEvalDataset(val_rows[:EVAL_LOSS_CAP], eval_preprocessor, tokenizer,
                              IMAGE_COLUMN, TEXT_COLUMN)

    data_collator = OCRCollator(pad_token_id=tokenizer.pad_token_id, downsample=DOWNSAMPLE)

    # ---- Optimizer (fresh, single LR tier); scheduler built by EncoderDecoderTrainer ----
    # Single LR tier: only the new adapter params are trainable in stage 1.
    optimizer = build_optimizer(model, lrs=LR, weight_decay=WEIGHT_DECAY, betas=BETAS)

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
        ddp_timeout=7200,                        # 2h

        # Optimizer + LR are supplied explicitly via `optimizers=` below (build_optimizer),
        # and the warmup->cosine-to-min_lr scheduler is built by create_scheduler, so no
        # optim / learning_rate / weight_decay / betas / lr_scheduler_type are set here.

        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        ignore_data_skip=True,                   # fast resume (don't replay skipped data)

        fp16=torch.cuda.is_available(),          # T4 -> fp16 + GradScaler
        gradient_checkpointing=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=4,
        dataloader_persistent_workers=True,

        eval_strategy="steps",
        eval_steps=SAVE_EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_EVAL_STEPS,
        save_total_limit=2,
        load_best_model_at_end=False,            # PORTED: no best-model tracking / no early stopping

        logging_strategy="steps",
        logging_steps=50,
        logging_first_step=True,
        report_to="wandb",
        run_name="stage-1-finetuning",
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
        min_lr=MIN_LR,
    )

    # ---- Resume: this run's checkpoints, else a prior-run dir (PORTED) ----
    last_ckpt = find_last_checkpoint(OUTPUT_DIR, PREV_RUN_DIR)
    print(f"Resuming from: {last_ckpt}" if last_ckpt else "Starting fresh (no checkpoint found)")

    trainer.train(resume_from_checkpoint=last_ckpt)

    # Save final artifacts (HF checkpoint + stage-1-style .pt)
    trainer.save_model(OUTPUT_DIR)
    if trainer.is_world_process_zero():
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "final_model.pt"))
        print("Finished stage-1 training!")