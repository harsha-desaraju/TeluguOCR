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
from src.telugu_ocr.data.collators import OCRCollator
from src.telugu_ocr.training.callbacks import CEREvalCallback
from src.telugu_ocr.training.optim import LR_TIERS, _param_tier
from src.telugu_ocr.training.trainer import EncoderDecoderTrainer
                 # is what backed up the notebook pipe / wandb console and hung rank 0)






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






# Scheduler: per-group linear warmup -> cosine decay from the group's own peak (group
# 'lr') to the group's own floor (group 'min_lr'). Each of the three LR tiers therefore
# rides its own cosine curve between its configured max and min.






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
    data_collator = OCRCollator(emit_ctc=True, pad_token_id=tokenizer.pad_token_id, downsample=DOWNSAMPLE,
                                ctc_strip_ids=(tokenizer.bos_token_id, tokenizer.eos_token_id))

    # ---- Optimizer (fresh, three LR tiers); scheduler built by EncoderDecoderTrainer ----
    # Three LR tiers (encoder / cross-attention / LM), each with its own peak and floor.
    optimizer = build_optimizer(model, lrs=LR, min_lrs=MIN_LR, weight_decay=WEIGHT_DECAY,
                                betas=BETAS, tier_fn=_param_tier, tiers=LR_TIERS)

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
            # STAGE-2 also reports the teacher-forced and CTC decoders, on the same
            # images, so the three can be compared directly.
            CEREvalCallback(model, eval_slices, eval_preprocessor, tokenizer,
                            image_col=IMAGE_COLUMN, text_col=TEXT_COLUMN,
                            report_tf=True, report_ctc=True),
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