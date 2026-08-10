"""One OCR engine over a HuggingFace image column -- single pass, predictions as columns.

The stripped-down cousin of consensus_labelling.py: it runs exactly ONE engine over one
image column of a HuggingFace dataset, attaches that engine's predictions as two new
columns, and optionally pushes the result to the Hub. No consensus, no tiering, no resume
machinery -- just a bulk pass, which is what you want when a dataset is too big to justify
three engines, or when you are topping up an existing set with one more engine's opinion.

Pick the engine with ENGINE in the config block at the bottom:

    ENGINE = "tesseract"   CPU-only, runs anywhere (Kaggle, Colab, a laptop)
    ENGINE = "paddle"      GPU recommended

OUTPUT
    The input dataset, unchanged, plus two columns named after the engine:
        pred_tesseract / tesseract_score      confidence is mean per-WORD, 0-100
        pred_paddle    / paddle_score         confidence is per-LINE, 0-1
    A score of -1.0 means unavailable, or the row failed. NOTE the two scales differ and
    are NOT comparable; that is how each engine reports, and rescaling one would only
    disguise which engine a column came from.

    Columns are attached with add_column at the very end, so the image column is never
    decoded and re-encoded (mapping over the dataset would re-encode every image). The
    flip side is that the run is held in memory and has no resume: a disconnect loses it.
    Use consensus_labelling.py when you need crash-safety.

This file is SELF-CONTAINED on purpose (the repo convention): both engines and the
batching are inlined, no `src...` imports, so it can be uploaded to Kaggle / Colab / any
host and run standalone.

WHY PADDLE IS RECOGNITION-ONLY (no detection)
    The inputs are already single-line crops. Letting the full PaddleOCR pipeline
    re-detect boxes inside a 64px strip only invents sub-boxes, drops text, and returns
    the fragments OUT OF READING ORDER, so naively joining them scrambles the line. So
    paddleocr 3.x `TextRecognition` is loaded with the Telugu recognition model directly.

    `lang="te"` resolves to `te_PP-OCRv5_mobile_rec` -- the only Telugu recognition model
    (there is no `te_PP-OCRv5_server_rec`). paddleocr's `_utils/langs.py` lists only the
    script GROUPS, so grepping it wrongly suggests Telugu is absent; it is not.

TWO TESSERACT SETTINGS THAT MATTER FOR LINE CROPS
    * psm 7 ("treat the image as a single text line"). The default psm 3 runs full page
      layout analysis on a 64px strip and frequently returns nothing at all.
    * 2x upscale. Tesseract wants ~30-35px of x-height; a 64px line crop is at the bottom
      of its comfortable range, and a 2x cubic upscale measurably helps.

======================================================================================
RUNNING
======================================================================================
    python3 -m pipelines.label.single_engine_labelling

Tesseract needs its BINARY and the Telugu language data, not just the pip wheel:
    Kaggle / Debian / Ubuntu:  !apt-get -qq install -y tesseract-ocr tesseract-ocr-tel
    macOS:                     brew install tesseract tesseract-lang

PaddleOCR on a Kaggle GPU needs the GPU wheel from Paddle's own index, and the install
order matters -- see the RUNNING ON KAGGLE section of consensus_labelling.py, which
documents the paddle/paddleocr version mismatch and the torch NCCL repair in full.

Auth for push_to_hub: HF_TOKEN as an environment variable, in .env, or as a Kaggle secret
named HF_TOKEN (Add-ons -> Secrets). Never hard-code a token in this file.

Internet must be ON to load the dataset from the Hub and to push. To go offline, point
INPUT_REPO at a local save_to_disk path -- load_from_disk is used automatically.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

try:                                     # progress bar with ETA; degrades to prints
    from tqdm.auto import tqdm

    _HAVE_TQDM = True
except ImportError:
    _HAVE_TQDM = False


# ======================================================================================
# Tesseract (recognition + per-word confidence), one thread per in-flight crop
# ======================================================================================
from benchmark.engines import PaddleOCREngine, TesseractEngine


# ======================================================================================
# PaddleOCR recognition engine (recognition-only)
# ======================================================================================


ENGINES = {"tesseract": TesseractEngine, "paddle": PaddleOCREngine}


# ======================================================================================
# Labelling: run the engine over the whole dataset in one pass, attach two columns
# ======================================================================================
def label_dataset(ds, engine, image_col: str = "image", chunk_size: int = 512,
                  show_progress: bool = True):
    """Run `engine` over every row of `ds`, returning it with pred_<engine> / <engine>_score
    added.

    Single pass, held in memory: predictions accumulate in two lists and are attached with
    add_column at the end. Work is chunked only so images are decoded a chunk at a time
    (peak memory stays bounded) -- there is no per-chunk saving or resume here; a
    disconnect loses the run.
    """
    n = len(ds)
    all_texts: list[str] = []
    all_scores: list[float] = []

    bar = tqdm(total=n, unit="row", desc=engine.name, dynamic_ncols=True,
               smoothing=0.05) if (show_progress and _HAVE_TQDM) else None
    tick = (lambda k: bar.update(k)) if bar is not None else None

    for start in range(0, n, chunk_size):
        idx = range(start, min(start + chunk_size, n))
        images = ds.select(idx)[image_col]
        images = [im if isinstance(im, Image.Image) else Image.fromarray(np.asarray(im))
                  for im in images]

        texts, scores = engine.run(images, tick=tick)

        # None (a failed row/batch) -> "" text and -1.0 score; a genuine empty read stays "".
        all_texts.extend(t if t is not None else "" for t in texts)
        all_scores.extend(round(float(s), 4) if s is not None else -1.0 for s in scores)

        if bar is None and show_progress:
            print(f"  {min(start + chunk_size, n)}/{n} rows", flush=True)

    if bar is not None:
        bar.close()

    out = ds
    for name, col in ((f"pred_{engine.name}", all_texts),
                      (f"{engine.name}_score", all_scores)):
        if name in out.column_names:
            out = out.remove_columns(name)
        out = out.add_column(name, col)
    return out


def hf_login() -> None:
    """Log in to the Hub from HF_TOKEN: env var, .env file, or Kaggle secret."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from dotenv import load_dotenv

            load_dotenv()
            token = os.environ.get("HF_TOKEN")
        except Exception:
            pass
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient

            token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            token = None
    if not token:
        raise SystemExit(
            "No HuggingFace token found. Set the HF_TOKEN environment variable, put it in "
            ".env, or add a Kaggle secret named HF_TOKEN (Add-ons -> Secrets).")
    from huggingface_hub import login

    login(token=token)


def load_split(input_repo: str, config: str | None, split: str):
    """One split from the Hub, or from a local save_to_disk path if `input_repo` exists."""
    if os.path.exists(input_repo):
        from datasets import load_from_disk
        ds = load_from_disk(input_repo)
        if not hasattr(ds, "column_names") or isinstance(ds.column_names, dict):
            ds = ds[split]
        return ds
    from datasets import load_dataset
    return load_dataset(input_repo, config, split=split)


# ======================================================================================
# Inline configuration
# ======================================================================================
if __name__ == "__main__":
    # ---- which engine ----
    ENGINE = "tesseract"             # "tesseract" (CPU, anywhere) | "paddle" (GPU)

    # ---- input ----
    INPUT_REPO = "harsha-desaraju/telugu-book-line-images"   # HF repo id or local path
    # Each entry is labelled and pushed separately, so a multi-config dataset can be
    # worked through in one run. [None] for a dataset with no configs.
    INPUT_CONFIGS = ["set_1"]
    INPUT_SPLIT = "train"
    IMAGE_COL = "line_image"         # the column holding the images
    LIMIT = None                     # set to an int for a smoke test, None for all rows
    CHUNK_SIZE = 512                 # images decoded per chunk (memory only, not a save unit)

    # ---- tesseract ----
    TESSERACT_LANG = "tel"           # Telugu only
    TESSERACT_THREADS = None         # None -> os.cpu_count() (~4 on Kaggle 2x T4)
    UPSCALE = 2.0

    # ---- paddle ----
    PADDLE_LANG = "te"               # -> te_PP-OCRv5_mobile_rec (the only Telugu rec model)
    PADDLE_BATCH = 128
    PADDLE_USE_GPU = True
    PADDLE_DEVICE = None             # e.g. "gpu:0" to pin a card on a multi-GPU host

    # ---- output ----
    PUSH_TO_HUB = None               # e.g. "harsha-desaraju/telugu-book-line-tesseract"
    PRIVATE = False                  # create the Hub repo private
    OUTPUT_DIR = None                # local save_to_disk dir, or None to skip
    # --------------------------------------------------------------------------------

    if ENGINE not in ENGINES:
        raise SystemExit(f"ENGINE must be one of {sorted(ENGINES)}, not {ENGINE!r}")

    if PUSH_TO_HUB:
        hf_login()

    for config in INPUT_CONFIGS:
        print(f"\nLoading {INPUT_REPO} (config={config}, split={INPUT_SPLIT}) ...",
              flush=True)
        ds = load_split(INPUT_REPO, config, INPUT_SPLIT)
        if LIMIT:
            ds = ds.select(range(min(LIMIT, len(ds))))
        print(f"  {len(ds)} rows | columns: {ds.column_names}", flush=True)
        if IMAGE_COL not in ds.column_names:
            raise SystemExit(f"image column {IMAGE_COL!r} not in dataset; "
                             f"columns are {ds.column_names}")

        if ENGINE == "tesseract":
            engine = TesseractEngine(lang=TESSERACT_LANG, num_threads=TESSERACT_THREADS,
                                     upscale=UPSCALE)
        else:
            engine = PaddleOCREngine(lang=PADDLE_LANG, batch_size=PADDLE_BATCH,
                                     use_gpu=PADDLE_USE_GPU, device=PADDLE_DEVICE)

        labelled = label_dataset(ds, engine, image_col=IMAGE_COL, chunk_size=CHUNK_SIZE)
        print(f"\nDone: {len(labelled)} rows labelled | columns now: "
              f"{labelled.column_names}", flush=True)

        if OUTPUT_DIR:
            out_dir = Path(OUTPUT_DIR) / config if config else Path(OUTPUT_DIR)
            out_dir.parent.mkdir(parents=True, exist_ok=True)
            labelled.save_to_disk(str(out_dir))
            print(f"Saved  -> {out_dir}")

        if PUSH_TO_HUB:
            print(f"Pushing -> {PUSH_TO_HUB} (config={config}, private={PRIVATE}) ...",
                  flush=True)
            if config:
                labelled.push_to_hub(PUSH_TO_HUB, config, private=PRIVATE)
            else:
                labelled.push_to_hub(PUSH_TO_HUB, private=PRIVATE)
            print(f"Pushed  -> {PUSH_TO_HUB}")
        else:
            print("PUSH_TO_HUB is None; nothing pushed.")
