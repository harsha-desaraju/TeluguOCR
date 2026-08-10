# CLAUDE.md

Guidance for working in this repository.

## What this is

A research project for building a **Telugu OCR** system from scratch. It trains a
vision-encoder → text-decoder model that reads an image of a line of Telugu text and
outputs the transcribed text. Everything is custom PyTorch (no off-the-shelf OCR),
trained in stages and orchestrated with the HuggingFace `Trainer`.

## Architecture (the three trained components)

The final model is assembled in [src/telugu_ocr/models/encoder_decoder.py](src/telugu_ocr/models/encoder_decoder.py)
(`EncoderDecoder`) by transferring weights from two independently pretrained models:

1. **Image encoder** — a ViT Masked Auto-Encoder (`MaskedAutoEncoder` /
   `ViTEncoder`) in [src/telugu_ocr/models/image_encoder.py](src/telugu_ocr/models/image_encoder.py).
   Self-supervised pretraining reconstructs masked image patches. Input: grayscale
   line images, height 64, patch size 8, variable width up to 1024.
2. **Text decoder** — a GPT-style causal LM (`GPTModel`) in
   [src/telugu_ocr/models/text_decoder.py](src/telugu_ocr/models/text_decoder.py) with SwiGLU MLPs, pretrained
   on Telugu text. Uses a custom **grapheme (akshara) tokenizer**,
   [src/telugu_ocr/tokenizer/grapheme.py](src/telugu_ocr/tokenizer/grapheme.py) —
   splits text into Unicode grapheme clusters via `regex.\X`, one cluster → one token
   (no BPE).
3. **Encoder-decoder** — `TextDecoder` adds **cross-attention** layers on top of the
   pretrained GPT. Fine-tuning happens in two stages (`train_stage_1.py`,
   `train_stage_2.py`): stage 1 typically freezes the pretrained encoder + decoder
   weights and trains only the new cross-attention (`cross_attention`, `layer_norm1_5`);
   stage 2 unfreezes more for end-to-end refinement.

The `if __name__ == '__main__'` block in `encoder_decoder/model.py` documents the exact
load → transfer → freeze → sanity-check recipe.

## Repository layout

> **Phase 2 note.** The paths below are current; the *architecture* section above still
> describes a ViT-MAE encoder and is wrong (the trained encoder is a conv-stem CTC model).
> Phase 5 of the package refactor rewrites it. See `docs/refactor/component_inventory.md`.

- `src/telugu_ocr/models/` — `image_encoder.py` (conv-stem CTC), `text_decoder.py` (GPT),
  `encoder_decoder.py` (the combined OCR model).
- `src/telugu_ocr/tokenizer/` — `grapheme.py` + `vocab.py`, with the vocab and grapheme
  distributions under `assets/`.
- `src/telugu_ocr/data/` — `preprocess.py` (`ImagePreprocessor`) and `augment.py`.
- `src/telugu_ocr/training/loops/` — `ctc.py`, `decoder_lm.py`, `encdec_stage1.py`,
  `encdec_stage2.py`. Self-contained scripts (see the duplication note below).
- `pipelines/` — data pipelines, by stage: `acquire/` (PDF scraping/downloading),
  `pages/` (PDF→images), `synth/` (synthetic text-line generation + fonts),
  `label/` (pseudo-labelling), `publish/` (push datasets to HF Hub), and
  `wikisource/` (the Wikisource scrape→align→build→push pipeline, kept intact).
- `configs/` — model/train configs and `checkpoints.yaml`, the registry saying which
  config reproduces which checkpoint.
- `scripts/eval/` — ad-hoc diagnostic/eval runners (not a test suite).
- `benchmark/`, `tools/annotation/`, `notebooks/`, `misc/`, `tests/`, `docs/`.
- `models/` — training outputs and checkpoints (gitignored-ish; large, not source).
- `data/` — corpora, generated images, `temp_test/` sample images. Gitignored.

## Conventions and gotchas

- **Duplicated code is intentional.** Training scripts like
  `encoder_decoder/train_stage_1.py`, `train_stage_2.py`, and
  `image_encoder/single_train_file.py` **inline copies** of the model, tokenizer, and
  utils rather than importing from `src/`. This is so each file is a self-contained
  script that can be uploaded and run standalone on **Kaggle / Colab / remote GPUs**.
  When you change core model logic, check whether the inlined copies also need updating.
- **`.py` vs `.ipynb` pairs** — several components have both a script and a notebook
  (`stage-1-finetuning.ipynb`, etc.); notebooks are the interactive/remote counterparts.
- **Hardcoded absolute paths** — many scripts contain absolute paths like
  `/Users/xai/Personal/Projects/TeluguOCR/models/...` and reference checkpoint dirs such
  as `models/image_encoder/results (1)/telugu-vitmae/final_model.pt`. Expect to adjust
  these per machine.
- **Image preprocessing height must be 64.** The encoder's positional encoding is built
  for `image_height // patch_size`, and collators assume a uniform batch height. Images
  that aren't resized to height 64 will break batching/inference.

## Environment & commands

- **Python 3.12**, dependency management via **`uv`** (`pyproject.toml` + `uv.lock`).
  Install deps with `uv sync`. A `.venv/` is present.
- Use **`python3`**, not `python` — `python` is not on PATH in this environment.
- Run scripts from the repo root so `src...` imports resolve, e.g.
  `python3 -m src.telugu_ocr.training.loops.encdec_stage1` (or run the self-contained inlined
  scripts directly on a GPU host).
- There is **no configured test runner or linter**. `test_model.py` files are ad-hoc
  eval/visualization scripts, not a test suite.
- Training relies on **`.env`** for secrets: `HF_TOKEN` (HuggingFace Hub) and
  `WANDB_API_KEY` (Weights & Biases logging). Never commit real values or print them.

## Data pipeline (high level)

Scrape/download Telugu PDFs → convert to page/line images → generate synthetic text-line
images (Telugu 70% / Sanskrit 15% / English 15%, with grapheme-length distribution and
augraphy augmentation) → push datasets to HF Hub → consume via `datasets.load_dataset`
in training.

## General Coding Instructions
1) Do not make the inputs command line arguments unless I explicitly ask for it. Use inline arguments only in the `__main__` block.
2) Ask if you have any questions. Do not assume anything.
3) Keep all the miscellaneous code in the misc/ folder.