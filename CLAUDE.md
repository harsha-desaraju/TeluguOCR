# CLAUDE.md

Guidance for working in this repository.

## What this is

A research project for building a **Telugu OCR** system from scratch. It trains a
vision-encoder → text-decoder model that reads an image of a line of Telugu text and
outputs the transcribed text. Everything is custom PyTorch (no off-the-shelf OCR),
trained in stages and orchestrated with the HuggingFace `Trainer`.

## Architecture (the three trained components)

The final model is assembled in [src/telugu_ocr/models/encoder_decoder.py](src/telugu_ocr/models/encoder_decoder.py)
(`EncoderDecoder`) from two independently pretrained models.

1. **Image encoder** — a **conv-stem CTC model** (`ImageEncoderCTC`) in
   [src/telugu_ocr/models/image_encoder.py](src/telugu_ocr/models/image_encoder.py).
   A six-stage convolutional stem collapses a grayscale line image to a sequence of
   frames (height → 1, width ÷ 8), a 10-layer pre-LN transformer encodes them, and a
   linear CTC head predicts a grapheme per frame. `embed_dim=384`, learned position
   table of 256 frames, so **input is height 64, width up to 2048**.
   *There is no ViT and no masked auto-encoder.* An earlier version of this project used
   ViT-MAE pretraining; it was rejected (spec §11) and no MAE code remains.
2. **Text decoder** — a GPT-style causal LM (`GPTModel`) in
   [src/telugu_ocr/models/text_decoder.py](src/telugu_ocr/models/text_decoder.py):
   16 layers, `embed_dim=512`, SwiGLU MLPs, pretrained on Telugu text. It uses a custom
   **grapheme (akshara) tokenizer**,
   [src/telugu_ocr/tokenizer/grapheme.py](src/telugu_ocr/tokenizer/grapheme.py) —
   `regex.\X` splits text into Unicode grapheme clusters, one cluster → one token, no
   BPE. Vocabulary is 2048.
3. **Encoder–decoder** — `TextDecoder` adds **cross-attention** on every second block of
   the pretrained GPT, behind a zero-init tanh gate so the image contributes nothing at
   initialisation and is phased in as the gate trains. A `enc_to_dec` linear bridges the
   encoder's 384 dims to the decoder's 512.

Fine-tuning runs in two stages, both in
[src/telugu_ocr/training/loops/encdec.py](src/telugu_ocr/training/loops/encdec.py),
selected by `STAGE`:

| | starts from | trains | objective |
|---|---|---|---|
| **stage 1** | the two pretrained checkpoints | the cross-attention adapters, their norms, the gates, the bridge | cross-entropy |
| **stage 2** | stage 1's single checkpoint | everything, on three LR tiers | `CE + 0.3 × CTC` |

Stage 1 runs its training forward in `.eval()` mode on purpose: the backbone is frozen,
so its dropout and DropPath would inject noise into the features the adapters learn
from. Stage 2 unfreezes and re-enables both.

**`configs/checkpoints.yaml` is the registry** that says which config reproduces which
checkpoint. The model dataclass defaults describe *no* trained artifact — build from the
configs, never from the defaults.

## Repository layout

- `src/telugu_ocr/models/` — `image_encoder.py` (conv-stem CTC), `text_decoder.py` (GPT),
  `encoder_decoder.py` (the combined OCR model).
- `src/telugu_ocr/tokenizer/` — `grapheme.py` + `vocab.py`, with the vocab and grapheme
  distributions under `assets/`.
- `src/telugu_ocr/data/` — `preprocess.py` (the one line-geometry implementation),
  `augment.py` (the degradation pipeline), `collators.py` (`LineTensorizer`,
  `CTCBatchMapper`, `OCRCollator` — all of which do NO resizing, see below).
- `src/telugu_ocr/training/loops/` — `ctc.py`, `decoder_lm.py`, and `encdec.py` (both
  fine-tuning stages, selected by `STAGE` / `STAGE_CONFIGS`). They import from the
  package; `scripts/bundle.py` re-inlines one into a standalone file for Kaggle.
- `src/telugu_ocr/training/` — `optim.py`, `trainer.py`, `callbacks.py`, `checkpoint.py`,
  `eval_slices.py`.
- `src/telugu_ocr/metrics/` — `errors.py` (edit distance, CER) and `normalize.py` (the
  shared grapheme splitter; note the two `normalize` functions are deliberately NOT
  merged, see that module's docstring).
- `benchmark/` — `engines.py` holds the whole engine stack in one file: the shared
  `OCREngine` contract, the Tesseract / PaddleOCR / Surya adapters and the wrapper around
  this repo's own model. Plus `metrics.py` (alignment + confusion tables) and
  `run_benchmark.py`. The adapters are NOT part of the model — `src/telugu_ocr/` imports
  nothing from them and builds without pytesseract or paddleocr installed. But
  `pipelines/label/` DOES import them for consensus labelling, so the labelling pipeline
  depends on `benchmark/`, which is the wrong direction for a data pipeline and is a
  deliberate trade recorded in `benchmark/engines.py`.
- `pipelines/` — data pipelines, by stage: `acquire/` (PDF scraping/downloading),
  `pages/` (PDF→images), `synth/` (synthetic text-line generation + fonts),
  `label/` (pseudo-labelling), `publish/` (push datasets to HF Hub), and
  `wikisource/` (the Wikisource scrape→align→build→push pipeline, kept intact).
- `configs/` — model/train configs and `checkpoints.yaml`, the registry saying which
  config reproduces which checkpoint.
- `scripts/` — `bundle.py` (generates the standalone single-file training script),
  `token_fertility.py`, and `eval/` (ad-hoc diagnostic runners, not a test suite).
- `tests/` — the regression oracles plus `model_registry.py` (builds any model from a
  config) and `make_fingerprints.py` (re-records the checkpoint baseline).
- `misc/` — one-off scripts and the exploratory notebooks (`model_diagnosis.ipynb`,
  `syn-data-gen.ipynb`, `calculate-token-distribution.ipynb`). Per the coding rules
  below, miscellaneous code belongs here.
- `tools/annotation/` (the correction UI) and `tests/`.
- `models/` — training outputs and checkpoints (gitignored-ish; large, not source).
- `data/` — corpora, generated images, `temp_test/` sample images. Gitignored.

## Conventions and gotchas

- **Single-file training scripts are GENERATED, not hand-copied.** Kaggle and Colab take
  one uploaded file, not a package. That constraint used to be met by inlining copies of
  the model, tokenizer and utils into each training script — which produced 69 duplicated
  top-level definitions, 41 of them silently drifted apart. The loops now import from the
  package like normal code, and `scripts/bundle.py` generates the standalone file:

      python3 scripts/bundle.py src/telugu_ocr/training/loops/encdec.py -o encdec_kaggle.py

  **Never hand-edit a bundled file** — regenerate it. There is exactly one definition of
  everything, in `src/telugu_ocr/`.
- **Build models from `configs/`, never from the dataclass defaults.** The defaults
  describe no trained artifact: `CTCEncoderConfig` defaults to 1024px/128 frames while
  every checkpoint is 2048/256, and `GPTConfig` defaults shallower than the trained 16
  layers. Building from defaults fails on one tensor's shape (encoder) or silently
  ignores 44 tensors (decoder).
- **Image preprocessing height must be 64,** and there is ONE implementation of the
  geometry: `resize_line_image` in `src/telugu_ocr/data/preprocess.py`. Getting it wrong
  is silent — no exception, no shape error, the model just reads badly. Note that the
  training loops deliberately do NO resizing (`LineTensorizer`, `CTCBatchMapper`),
  because their datasets already hold height-64 crops; resizing there would resample
  twice.
- **Hardcoded absolute paths** — many scripts contain paths like
  `/kaggle/input/...` or `/Users/.../TeluguOCR/models/...`. Expect to adjust per machine.

## Environment & commands

- **Python 3.12**, dependency management via **`uv`** (`pyproject.toml` + `uv.lock`).
  Install deps with `uv sync`. A `.venv/` is present.
- Use **`python3`**, not `python` — `python` is not on PATH in this environment.
- Run scripts from the repo root so `src...` imports resolve, e.g.
  `python3 -m src.telugu_ocr.training.loops.encdec` (or bundle it first for a GPU host).
- **Regression checks** — no pytest runner, but these exist and should stay green:

      python3 -m tests.test_checkpoint_compat   # models still match the checkpoints
      python3 -m tests.micro_train_curve check  # training loss + CER unchanged
      python3 -m tests.engine_equivalence check # OCR engines transcribe identically

  Re-record a baseline only when a change is intentional and explained
  (`... capture`, or `python3 -m tests.make_fingerprints`). Never regenerate one to
  silence a failure you have not explained.
- `scripts/eval/*.py` are ad-hoc diagnostic/visualisation runners, not tests.
- Training relies on **`.env`** for secrets: `HF_TOKEN` (HuggingFace Hub) and
  `WANDB_API_KEY` (Weights & Biases logging). Never commit real values or print them.

## Data pipeline (high level)

Scrape/download Telugu PDFs → convert to page/line images → generate synthetic text-line
images (rendered across 30 fonts × 6 sizes, then degraded by `data/augment.py`) → push
datasets to HF Hub → consume via `datasets.load_dataset` in training.

Real labelled lines come from two further routes: Wikisource proofread scans, cut into
lines and aligned against the known page transcript (`pipelines/wikisource/`), and
consensus pseudo-labelling, where several engines transcribe the same crop and only
agreement is kept (`pipelines/label/`).

The language mix of the text corpus (Telugu / Sanskrit / English) was fixed when
`telugu-sanskrit-english-text-1024` was built; it is a property of that published
dataset, not a constant you will find in this repo.

## General Coding Instructions
1) Do not make the inputs command line arguments unless I explicitly ask for it. Use inline arguments only in the `__main__` block.
2) Ask if you have any questions. Do not assume anything.
3) Keep all the miscellaneous code in the misc/ folder.