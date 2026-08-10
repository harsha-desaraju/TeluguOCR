# Phase 3 — what was collapsed, what it cost, and what is left

Phase 3 was "collapse duplicates, one component at a time". The plan assumed the copies
were the same code that had drifted, and that the job was picking the right one. That was
true for the models and the tokenizer. It was **not** true for preprocessing or for the
optimizers, and the two exceptions are the interesting part of this document.

## Result

| | before phase 3 | after |
|---|---|---|
| duplicated top-level definitions | 69 | 23 |
| of those, drifted | 41 | 23 |
| `training/loops/ctc.py` | 1601 | 662 |
| `training/loops/decoder_lm.py` | 295 | 152 |
| `training/loops/encdec_stage1.py` | 2162 | 778 |
| `training/loops/encdec_stage2.py` | 2405 | 892 |

The 23 that remain are the OCR engines and the per-script `main` functions; see *What is
left* below. `main` appearing in four files is not duplication — four scripts each have
their own entry point.

## The oracles, and why each one exists

Every step was gated on a check that could actually fail. They live in `tests/` where they
are worth keeping:

| step | oracle | what it would have caught |
|---|---|---|
| tokenizer | vocab + corpus hashes over 11 edge-case strings | a changed label space, which corrupts training data silently |
| preprocessing | pixel hashes over 6 inputs covering every branch | a resize/resample change — "the model just reads badly", no error |
| augmentation | seeded-RNG output hashes, 2 widths x 4 severities x 8 seeds | a changed augmentation distribution |
| models | strict load + forward probe against the real checkpoints | an architecture change |
| training | `tests/micro_train_curve.py` — a 12-step CPU run | a changed loss, optimizer grouping or LR schedule |

`tests/test_checkpoint_compat.py` ran after every single step and never went red.

## The two places the plan was wrong

### Preprocessing was 7 implementations, not 4 — and collapsing them would have corrupted training

The plan said `ImagePreprocessor` 4 -> 1. Reduced to a common form and run over the same
inputs, the copies fell into three groups, not one:

- **four pixel-identical geometry implementations** that differed only in RETURN TYPE
  (PIL, uint8 ndarray, normalized tensor). That is why they drifted unnoticed for so
  long: nothing compares a PIL image to a tensor.
- **one divergent geometry implementation** — the eval script's, resampling with LANCZOS
  and with no over-wide cap. `benchmark/engines.py` imports that one, so the benchmark was
  scoring the model on pixels no training path ever produced.
- **two that do no geometry at all**, on purpose, because training datasets already hold
  height-64 crops.

The trap was the third group. The encdec copy took `(image_height, max_image_width,
downsample)`, stored all three, and used none of them — it read as the geometry class and
behaved as a tensorizer. Folding it onto the geometry version would have added a second
resample to every training step: no error, no shape change, just a slow bleed in accuracy.

So the collapse was 7 -> 3, by ROLE: `resize_line_image` (one geometry, `out=` picks the
return type), `LineTensorizer` and `CTCBatchMapper` (no geometry, honestly named).

### The optimizers differed because the stages differ

Three copies of `build_optimizer`, three genuinely different, and none of them wrong.
Stage 1 trains one tier at a single LR; stage 2 trains three, each on its own cosine curve.
The shared version is the tiered form with single-tier as a special case — not a winner
picked from three.

## Behaviour that changed, deliberately

Two changes move numbers. Both were the point rather than a side effect.

1. **`scripts/eval/encoder_decoder.py::preprocess_image` now resamples BILINEAR, not
   LANCZOS**, and applies the over-wide cap. It is the function `benchmark/engines.py`
   uses, so published benchmark CER will shift slightly. That shift is measurement error
   being removed: eval now sees what training produced.
2. **`data/augment.py` adopted the training probabilities** — letterpress 0.06 -> 0.03,
   bad-photocopy 0.05 -> 0.025, now named `P_LETTERPRESS` / `P_BAD_PHOTOCOPY`. The three
   training loops had halved them for speed and were the values behind every checkpoint;
   this module had kept the originals and was consumed only by its own preview.

Everything else was verified byte-identical.

## Bugs found on the way

- **`consensus_labelling.py`'s CTC engine has been broken since before the refactor.** Its
  "2048" branch imported `src.image_encoder.train_ctc_encoder_2048`, a module with no
  history in git and no file on disk, so the branch that fires for every real checkpoint
  raised `ModuleNotFoundError`. The import is fixed; the engine is not, because it still
  builds `CTCEncoderConfig()` from dataclass defaults (1024px/128 frames) that no trained
  artifact uses. `FIXME(phase-3)` marks it. Fixing it means building from
  `configs/models/ctc_encoder_2048.yaml`.
- **`model_diagnosis.ipynb` called `make_weighted_augmenter`**, which exists
  nowhere in the repo and never has. Repointed at `make_augmenter`.
- **`make_augmenter()` was called bare in all three loops**, resolving `p_clean` from each
  loop's own `P_CLEAN = 0.50` while the canonical default is `0.20`. Importing the shared
  function without touching the call sites would have quietly cut the clean-sample fraction
  by 2.5x. The call sites now pass it explicitly.

## What is left

### 1. The OCR-engine collapse

`OCREngine` / `TesseractEngine` / `PaddleOCREngine` still exist in `benchmark/engines.py`,
`pipelines/label/consensus_labelling.py` and `pipelines/label/single_engine_labelling.py`,
at three versions each.

This is **not** mechanical, which is why it was not rushed. The two base classes are
different abstractions wearing one name:

| | `benchmark/engines.py` | `pipelines/label/consensus_labelling.py` |
|---|---|---|
| entry point | `transcribe(images)`, normalises single-vs-batch | `run(images) -> list[str \| None]` |
| scheduling | none | `device_kind` serialises GPU engines against each other |
| progress | none | `_tick(k)` callback so the bar advances mid-chunk |
| confidence | not supported | may return `(texts, scores)` |

A shared base has to serve both contracts, and picking one silently changes how the other's
callers behave. Do it as its own step, with an oracle: `tesseract` and `paddleocr` both run
in this environment (verified), so the check is to transcribe a fixed image set through
each copy before and after and diff the strings.

### 2. The stage-1 / stage-2 loop-body merge

`encdec_stage1.py` (778) and `encdec_stage2.py` (892) still share roughly their whole
shape. Everything above the `__main__` block is now imported rather than duplicated; what
remains is the two orchestration blocks, which differ substantially — stage 1 loads two
pretrained checkpoints and freezes, stage 2 loads one stage-1 checkpoint and unfreezes.

Merging them into one `loops/encdec.py` plus two configs is the plan's target and is still
right. It was not done here because it is ~400 lines of careful authoring, and doing it
half-way is worse than not starting. The groundwork is in place:

- `tests/micro_train_curve.py` already drives both stages through their collator,
  optimizer, scheduler and model forward, and has a committed baseline. It already
  supports the merged layout — set `MERGED=1` and it imports `loops.encdec` instead.
- The stage differences are now explicit parameters rather than hardcoded constants:
  `encoder_no_grad`, `ctc_loss_weight`, the optimizer's `tier_fn` / `tiers`.

The recipe: capture with `python3 -m tests.micro_train_curve capture`, merge, then
`check`. Both curves must come back with max delta 0.00e+00.
