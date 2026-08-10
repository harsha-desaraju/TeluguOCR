"""Encoder-decoder fine-tuning: ONE loop, two stages, two configs.

    STAGE 1   Freeze the pretrained CTC encoder and the pretrained GPT decoder; train
              only the newly-added cross-attention adapters, their layer norms, the
              zero-init tanh gates and the 384->512 bridge. Starts from TWO checkpoints
              (the CTC encoder and the grapheme LM) and transfers their weights in.

    STAGE 2   Unfreeze everything and refine end-to-end from ONE checkpoint (stage 1's
              output), on three LR tiers, with the encoder's CTC head added back as an
              auxiliary objective: loss = CE + ctc_loss_weight * CTC.

WHY ONE FILE
    These were two ~2200-line scripts that shared roughly 60% of their content by
    hand-copy. Phase 3 moved everything they share into the package -- the models, the
    tokenizer, augmentation, the collator, the callbacks, the trainer, the optimizer --
    and what was left was two orchestration blocks whose differences turned out to be
    almost entirely CONFIGURATION.

    The one place the logic genuinely differs is model construction (two checkpoints and
    a freeze, versus one checkpoint and none), so that is a branch. Everything else is
    driven by STAGE_CONFIGS below. Notably the training-data path needed no branch at
    all: stage 1 is stage 2 with an empty `no_aug_sources` and the real crops tagged
    'natural' instead of 'real', which makes the real bucket empty and the augmentation
    unconditional -- exactly what stage 1 did.

RUNNING IT
    Set STAGE at the bottom, then:
        python3 -m src.telugu_ocr.training.loops.encdec
        torchrun --nproc_per_node=<N> -m src.telugu_ocr.training.loops.encdec

    For Kaggle, bundle it into a standalone file first (phase 4):
        python3 scripts/bundle.py src/telugu_ocr/training/loops/encdec.py -o encdec_kaggle.py
"""

from __future__ import annotations

import os

# Keep OpenCV / BLAS single-threaded inside dataloader workers. MUST run before cv2 and
# numpy are imported anywhere, hence before the package imports below.
LIMIT_AUG_THREADS = True
if LIMIT_AUG_THREADS:
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_v, "1")

import gc  # noqa: E402
from collections import defaultdict  # noqa: E402

import torch  # noqa: E402
from datasets import concatenate_datasets, load_dataset  # noqa: E402
from transformers import TrainingArguments  # noqa: E402

from src.telugu_ocr.data.augment import make_augmenter, set_seed  # noqa: E402
from src.telugu_ocr.data.collators import LineTensorizer, OCRCollator  # noqa: E402
from src.telugu_ocr.models.encoder_decoder import EncoderDecoder  # noqa: E402
from src.telugu_ocr.models.image_encoder import CTCEncoderConfig, ImageEncoderCTC  # noqa: E402
from src.telugu_ocr.models.text_decoder import GPTConfig, GPTModel  # noqa: E402
from src.telugu_ocr.tokenizer.grapheme import TeluguGraphemeTokenizer  # noqa: E402
from src.telugu_ocr.training.callbacks import CEREvalCallback, GradNormAlert  # noqa: E402
from src.telugu_ocr.training.checkpoint import find_last_checkpoint  # noqa: E402
from src.telugu_ocr.training.eval_slices import ListEvalDataset, build_eval_slices  # noqa: E402
from src.telugu_ocr.training.optim import LR_TIERS, _param_tier, build_optimizer  # noqa: E402
from src.telugu_ocr.training.trainer import EncoderDecoderTrainer  # noqa: E402

if LIMIT_AUG_THREADS:
    try:
        import cv2

        cv2.setNumThreads(0)
    except Exception:
        pass


# ======================================================================================
# Configs — one per stage. Everything that differs between the two runs lives here.
# ======================================================================================
_COMMON = dict(
    # model shape (must match the checkpoints; see configs/checkpoints.yaml)
    decoder=dict(embed_dim=512, hidden_dim=1368, num_heads=8, num_layers=16, ctx_len=256),
    encoder=dict(max_image_width=2048, max_frames=256),

    # data
    dataset_synth="harsha-desaraju/sample-dataset-new",
    dataset_split="train",
    eval_split="validation",
    image_column="image",
    text_column="text",
    source_column="text_source",
    rnd_sample_frac=0.05,

    # frames / widths
    downsample=8,
    image_height=64,
    max_image_width=2048,

    # batching
    per_device_batch=32,
    grad_accum=2,
    eval_batch=32,

    # schedule
    warmup_steps=2000,
    weight_decay=0.05,
    betas=(0.9, 0.98),

    # eval / io
    eval_slice_cap=1500,
    eval_loss_cap=512,
    save_eval_steps=1000,
    seed=42,
    p_clean=0.50,
    grad_norm_alert=10.0,
)

STAGE_CONFIGS = {
    1: {
        **_COMMON,
        "vocab_file": "/kaggle/input/datasets/harshadesaraju1999/telugu-tokenizer-vocab/telugu-vocab.json",
        "decoder": {**_COMMON["decoder"], "dropout": 0.1},

        # STAGE 1 starts from two separate pretrained checkpoints.
        "gpt_checkpoint": "/kaggle/input/models/harshadesaraju1999/telugugpt/pytorch/default/1/final_model.pt",
        "ctc_checkpoint": "/kaggle/input/models/harshadesaraju1999/telugu-ctc-image-encoder/pytorch/default/1/final_model.pt",

        "dataset_real": "harsha-desaraju/telugu-pdf-line-image-text",
        "train_configs": ["train_0000", "train_0001", "train_0002", "train_0003"],
        # Stage 1 tagged the real crops 'natural' and augmented everything, so the 'real'
        # bucket is empty and no_aug_sources is empty -- see the module docstring.
        "real_source_tag": "natural",
        "no_aug_sources": set(),

        # one LR tier: only the adapters train
        "lr": 1e-4,
        "min_lr": 1e-5,
        "tiered_lr": False,
        "epochs": 4,

        # the frozen backbone must not inject dropout/DropPath noise into the adapters
        "force_eval_during_train": True,
        "encoder_no_grad": True,
        "ctc_loss_weight": 0.0,
        "report_tf": False,
        "report_ctc": False,

        "output_dir": "/kaggle/working/telugu-ocr-stage1",
        "prev_run_dir": "/kaggle/input/models/harshadesaraju1999/telugu-ocr-checkpoint/transformers/default/1",
        "resume_prev_dir_directly": False,
        "run_name": "stage-1-finetuning",
        "dataloader_workers": 4,
        "ddp_timeout": 7200,
        "disable_tqdm": False,
    },
    2: {
        **_COMMON,
        "vocab_file": "/kaggle/input/datasets/harshadesaraju99/telugu-tokenizer-vocab/telugu-vocab.json",
        "decoder": {**_COMMON["decoder"], "dropout": 0.05},

        # STAGE 2 starts from ONE checkpoint: stage 1's whole EncoderDecoder.
        "stage1_checkpoint": "/kaggle/input/models/harshadesaraju99/telugu-ocr-model-stage-1/pytorch/default/1/final_model.pt",

        "dataset_real": "harsha-desaraju/telugu-wikisource-text-images",
        "train_configs": ["train_0004", "train_0005"],
        # The real crops already carry real-world degradation, so they are kept as their
        # own source and NOT put through the synthetic augmentation pipeline.
        "real_source_tag": "real",
        "no_aug_sources": {"real"},

        # three LR tiers: encoder lowest, the still-young cross-attention highest
        "lr": {"encoder": 3e-5, "cross": 5e-5, "lm": 3e-5},
        "min_lr": {"encoder": 3e-6, "cross": 5e-6, "lm": 3e-6},
        "tiered_lr": True,
        "epochs": 5,

        "force_eval_during_train": False,
        "encoder_no_grad": False,
        "ctc_loss_weight": 0.3,
        "report_tf": True,
        "report_ctc": True,

        "output_dir": "/kaggle/working/telugu-ocr-stage2",
        "prev_run_dir": None,
        "resume_prev_dir_directly": True,
        "run_name": "stage-2-finetuning",
        # Kaggle 2x T4 = 4 vCPUs and this is 2-rank DDP, so workers are PER RANK: 2 here
        # means 4 processes on 4 cores. 4 would spawn 8 and thrash.
        "dataloader_workers": 2,
        "ddp_timeout": 3600,
        # 36k \r-updates on stderr flood the notebook pipe; the 50-step logs suffice.
        "disable_tqdm": True,
    },
}


# ======================================================================================
# Model construction — the one place the two stages genuinely differ
# ======================================================================================
def build_model_stage1(cfg, tokenizer):
    """Two pretrained checkpoints -> one adapter-trainable EncoderDecoder."""
    decoder_config = GPTConfig(vocab_size=len(tokenizer), **cfg["decoder"])
    encoder_config = CTCEncoderConfig(**cfg["encoder"])

    pretrained_text_model = GPTModel(decoder_config, pad_index=tokenizer.pad_token_id)
    pretrained_text_model.load_state_dict(torch.load(cfg["gpt_checkpoint"]))

    pretrained_encoder_model = ImageEncoderCTC(encoder_config)
    pretrained_encoder_model.load_state_dict(torch.load(cfg["ctc_checkpoint"]))

    model = EncoderDecoder(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        pad_index=tokenizer.pad_token_id,
        encoder_no_grad=cfg["encoder_no_grad"],
        ctc_loss_weight=cfg["ctc_loss_weight"],
    )

    # Transfer. strict=False on the decoder: the cross-attention adapters do not exist in
    # the pretrained LM, so they are exactly the keys allowed to be missing.
    match_result = model.decoder_model.load_state_dict(
        pretrained_text_model.state_dict(), strict=False)
    model.encoder_model.load_state_dict(pretrained_encoder_model.state_dict())

    newly_added_layers = ["cross_attention", "layer_norm1_5", "cross_attn_gate"]
    for layer_name in match_result.missing_keys:
        assert any(new in layer_name for new in newly_added_layers), \
            f"pretrained layer did not match: {layer_name}"

    # The image contributes NOTHING at init: each block's cross_attn_gate is zero-init and
    # tanh(0) == 0, so the model starts exactly as the pretrained LM and phases the image
    # in as the gate trains (Flamingo/adapter style).

    # Freeze everything that came from a checkpoint; only the adapters keep requires_grad.
    for name, params in model.decoder_model.named_parameters():
        if name not in match_result.missing_keys:
            params.requires_grad = False
    unfrozen = [n for n, p in model.decoder_model.named_parameters() if p.requires_grad]
    assert unfrozen == list(match_result.missing_keys), \
        "the unfrozen decoder layers are not exactly the newly-added ones"

    for parameters in model.encoder_model.parameters():
        parameters.requires_grad = False

    return model


def build_model_stage2(cfg, tokenizer):
    """ONE stage-1 checkpoint -> a fully trainable EncoderDecoder."""
    decoder_config = GPTConfig(vocab_size=len(tokenizer), **cfg["decoder"])
    encoder_config = CTCEncoderConfig(**cfg["encoder"])

    model = EncoderDecoder(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        pad_index=tokenizer.pad_token_id,
        ctc_loss_weight=cfg["ctc_loss_weight"],
        encoder_no_grad=cfg["encoder_no_grad"],
    )

    # Stage 1 already trained the encoder, the cross-attention and the LM TOGETHER, so the
    # whole model loads from one state dict. The load must be EXACT -- including each
    # block's trained cross_attn_gate -- so stage 2 starts precisely where stage 1 stopped.
    load_result = model.load_state_dict(
        torch.load(cfg["stage1_checkpoint"], map_location="cpu"), strict=False)
    print(f"[load] stage-1 -> encoder-decoder: missing={list(load_result.missing_keys)} "
          f"unexpected={list(load_result.unexpected_keys)}", flush=True)
    assert not load_result.missing_keys, \
        f"stage-1 checkpoint is missing weights for: {load_result.missing_keys}"
    assert not load_result.unexpected_keys, \
        f"stage-1 checkpoint has unexpected weights: {load_result.unexpected_keys}"

    # Nothing is frozen; each group is instead trained at its own LR tier.
    assert not model.encoder_no_grad, "STAGE-2 needs gradients through the encoder"
    return model


BUILDERS = {1: build_model_stage1, 2: build_model_stage2}


# ======================================================================================
# The loop — identical for both stages
# ======================================================================================
def run(stage: int):
    cfg = STAGE_CONFIGS[stage]

    # DDP topology sanity: prints ONCE PER RANK. More lines than --nproc_per_node means
    # extra processes; a straggler from an earlier torchrun has a DIFFERENT MASTER_PORT.
    print(f"[proc] pid={os.getpid()} ppid={os.getppid()} "
          f"RANK={os.environ.get('RANK')} LOCAL_RANK={os.environ.get('LOCAL_RANK')} "
          f"WORLD_SIZE={os.environ.get('WORLD_SIZE')} MASTER_PORT={os.environ.get('MASTER_PORT')}",
          flush=True)

    tokenizer = TeluguGraphemeTokenizer(vocab_file=cfg["vocab_file"])
    model = BUILDERS[stage](cfg, tokenizer)

    encoder_params = sum(p.numel() for p in model.encoder_model.parameters())
    decoder_params = sum(p.numel() for p in model.decoder_model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"No. of parameters in encoder: {encoder_params}")
    print(f"No. of parameters in decoder: {decoder_params}")
    print(f"No. of trainable parameters: {trainable_params}")
    print("Percentage of trainable parameters: "
          f"{((trainable_params / (encoder_params + decoder_params)) * 100):.2f}%")

    IMAGE_COLUMN, TEXT_COLUMN = cfg["image_column"], cfg["text_column"]
    SOURCE_COLUMN, SEED = cfg["source_column"], cfg["seed"]
    set_seed(SEED)                   # reproducible augmentation (random + np.random)

    # ---- Preprocessors: train augmented (composed degrade pipeline), eval clean ----
    train_preprocessor = LineTensorizer(augment_fn=make_augmenter(p_clean=cfg["p_clean"]))
    eval_preprocessor = LineTensorizer()
    no_aug = cfg["no_aug_sources"]

    def make_sample_transformer(aug_preprocessor, clean_preprocessor):
        def sample_transformer(batch):
            sources = batch[SOURCE_COLUMN]
            # Per-image: a source in no_aug_sources is preprocessed clean (it already
            # carries real degradation); everything else goes through augmentation.
            images = [
                (clean_preprocessor if src in no_aug else aug_preprocessor)(img)
                for img, src in zip(batch[IMAGE_COLUMN], sources)
            ]                                                          # list of (1, H, W_i)
            input_ids = [tokenizer.encode(t) for t in batch[TEXT_COLUMN]]   # BOS ... EOS
            return {"pixel_values": images, "input_ids": input_ids}
        return sample_transformer

    # ---- Train: synthetic configs + real line crops ----
    train_parts = [
        load_dataset(cfg["dataset_synth"], c,
                     columns=[IMAGE_COLUMN, TEXT_COLUMN, SOURCE_COLUMN])[cfg["dataset_split"]]
        for c in cfg["train_configs"]
    ]
    real_ds = load_dataset(cfg["dataset_real"], split=cfg["dataset_split"],
                           columns=[IMAGE_COLUMN, TEXT_COLUMN])
    real_ds = real_ds.add_column(SOURCE_COLUMN, [cfg["real_source_tag"]] * len(real_ds))
    train_parts.append(real_ds)
    train_hf = concatenate_datasets(train_parts, axis=0)

    # Cap the random-text share of the training mix.
    indices = defaultdict(list)
    for i, value in enumerate(train_hf[SOURCE_COLUMN]):
        indices[value].append(i)
    nat_ds = train_hf.select(indices["natural"])
    rnd_ds = train_hf.select(indices["random"])
    real_ds = train_hf.select(indices["real"])          # empty when real_source_tag=natural

    frac = cfg["rnd_sample_frac"]
    num_rnd = int((frac / (1 + frac)) * (len(nat_ds) + len(real_ds)))
    rnd_ds = rnd_ds.shuffle(seed=SEED).select(range(num_rnd))

    train_hf = concatenate_datasets([nat_ds, real_ds, rnd_ds], axis=0).shuffle(seed=SEED)
    del nat_ds, rnd_ds, real_ds
    gc.collect()

    train_dataset = train_hf.with_transform(
        make_sample_transformer(train_preprocessor, eval_preprocessor))
    print(f"[data] train -> {len(train_hf)} rows; cols {train_hf.column_names}")

    # ---- Validation: materialize rows, build source-based eval slices ----
    val_ds = load_dataset(cfg["dataset_synth"], cfg["eval_split"])[cfg["eval_split"]]
    val_rows = list(val_ds)
    print(f"[data] validation -> {len(val_rows)} rows")
    eval_slices = build_eval_slices(val_rows, SOURCE_COLUMN, cfg["eval_slice_cap"], SEED)

    # One small transformed eval set drives Trainer's eval_loss and fires on_evaluate
    # exactly ONCE; per-slice CER is added by CEREvalCallback.
    eval_ds = ListEvalDataset(val_rows[:cfg["eval_loss_cap"]], eval_preprocessor, tokenizer,
                              IMAGE_COLUMN, TEXT_COLUMN)

    collator_kwargs = {}
    if cfg["ctc_loss_weight"] > 0:
        # BOS/EOS are stripped from input_ids to form the CTC grapheme targets.
        collator_kwargs = dict(emit_ctc=True,
                               ctc_strip_ids=(tokenizer.bos_token_id, tokenizer.eos_token_id))
    data_collator = OCRCollator(pad_token_id=tokenizer.pad_token_id,
                                downsample=cfg["downsample"], **collator_kwargs)

    # ---- Optimizer; scheduler is built by EncoderDecoderTrainer.create_scheduler ----
    if cfg["tiered_lr"]:
        optimizer = build_optimizer(model, lrs=cfg["lr"], min_lrs=cfg["min_lr"],
                                    weight_decay=cfg["weight_decay"], betas=cfg["betas"],
                                    tier_fn=_param_tier, tiers=LR_TIERS)
        trainer_min_lr = None            # each group carries its own floor
    else:
        optimizer = build_optimizer(model, lrs=cfg["lr"], weight_decay=cfg["weight_decay"],
                                    betas=cfg["betas"])
        trainer_min_lr = cfg["min_lr"]   # one absolute floor

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["per_device_batch"],
        per_device_eval_batch_size=cfg["eval_batch"],
        gradient_accumulation_steps=cfg["grad_accum"],
        max_grad_norm=1.0,
        # NCCL collective timeout. The rank-0-only CER eval leaves the other ranks waiting
        # at the next barrier; raise this so a slow eval / save cannot trip the watchdog.
        ddp_timeout=cfg["ddp_timeout"],

        # Optimizer and LR come from `optimizers=` below, and the warmup->cosine scheduler
        # from create_scheduler, so no optim / learning_rate / lr_scheduler_type here.
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        ignore_data_skip=True,                   # fast resume (don't replay skipped data)

        fp16=torch.cuda.is_available(),
        gradient_checkpointing=False,
        dataloader_num_workers=cfg["dataloader_workers"],
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=4,
        dataloader_persistent_workers=True,
        disable_tqdm=cfg["disable_tqdm"],

        eval_strategy="steps",
        eval_steps=cfg["save_eval_steps"],
        save_strategy="steps",
        save_steps=cfg["save_eval_steps"],
        save_total_limit=2,
        load_best_model_at_end=False,            # no best-model tracking / early stopping

        logging_strategy="steps",
        logging_steps=50,
        logging_first_step=True,
        report_to="wandb",
        run_name=cfg["run_name"],
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
            CEREvalCallback(model, eval_slices, eval_preprocessor, tokenizer,
                            image_col=IMAGE_COLUMN, text_col=TEXT_COLUMN,
                            report_tf=cfg["report_tf"], report_ctc=cfg["report_ctc"]),
            GradNormAlert(threshold=cfg["grad_norm_alert"]),
        ],
        warmup_steps=cfg["warmup_steps"],
        min_lr=trainer_min_lr,
        force_eval_during_train=cfg["force_eval_during_train"],
    )

    # ---- Resume: this run's checkpoints, else a prior-run dir ----
    if cfg["resume_prev_dir_directly"] and cfg["prev_run_dir"]:
        last_ckpt = cfg["prev_run_dir"]
    else:
        last_ckpt = find_last_checkpoint(cfg["output_dir"], cfg["prev_run_dir"])
    print(f"Resuming from: {last_ckpt}" if last_ckpt else "Starting fresh (no checkpoint found)")

    trainer.train(resume_from_checkpoint=last_ckpt)

    trainer.save_model(cfg["output_dir"])
    if trainer.is_world_process_zero():
        torch.save(model.state_dict(), os.path.join(cfg["output_dir"], "final_model.pt"))
        print(f"Finished stage-{stage} training!")


if __name__ == "__main__":
    STAGE = 1          # 1 = adapters only, from two checkpoints; 2 = end-to-end refine
    run(STAGE)
