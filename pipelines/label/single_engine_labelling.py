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
class TesseractEngine:
    """Tesseract via pytesseract over a list of PIL images.

    Contract: `run(images)` takes a list of PIL images and returns a (texts, scores) pair,
    each the same length as `images`. A text of None means Tesseract FAILED on that row
    (the caller writes "" / -1.0 for those); "" is a genuine empty read.

    pytesseract shells out to the `tesseract` binary, which releases the GIL, so a thread
    pool gives real concurrency without pickling images.
    """

    name = "tesseract"

    def __init__(self, lang: str = "tel", psm: int = 7, oem: int = 1,
                 upscale: float = 2.0, num_threads: int | None = None, timeout: int = 20):
        import pytesseract

        self._pt = pytesseract
        self._Output = pytesseract.Output
        self.lang = lang
        self.config = f"--psm {psm} --oem {oem}"
        self.upscale = upscale
        self.num_threads = num_threads or (os.cpu_count() or 4)
        self.timeout = timeout

        # Fail loudly and early if the binary or the language data is missing, rather than
        # returning empty strings for every row.
        try:
            available = set(self._pt.get_languages(config=""))
        except Exception as exc:
            raise RuntimeError(
                f"Tesseract binary not found or not runnable ({exc}). Install it, e.g. "
                f"`apt-get install -y tesseract-ocr tesseract-ocr-tel`."
            ) from exc
        missing = [l for l in self.lang.split("+") if l not in available]
        if missing:
            raise RuntimeError(
                f"Tesseract is installed but language data {missing} is missing "
                f"(have: {sorted(available)}). Install e.g. `apt-get install -y "
                f"tesseract-ocr-{missing[0]}`.")
        print(f"[tesseract] lang={self.lang} {self.config} upscale={self.upscale} "
              f"threads={self.num_threads}")

    def _one(self, img: Image.Image) -> tuple[str | None, float | None]:
        """Return (line_text, mean_word_confidence) or (None, None) on failure."""
        try:
            im = img.convert("L")
            if self.upscale and self.upscale != 1.0:
                im = im.resize((max(1, int(im.width * self.upscale)),
                                max(1, int(im.height * self.upscale))),
                               Image.BICUBIC)
            data = self._pt.image_to_data(im, lang=self.lang, config=self.config,
                                          timeout=self.timeout,
                                          output_type=self._Output.DICT)
            words, confs = [], []
            for word, conf in zip(data.get("text", []), data.get("conf", [])):
                word = (word or "").strip()
                if not word:
                    continue
                words.append(word)
                try:
                    c = float(conf)
                except (TypeError, ValueError):
                    c = -1.0
                if c >= 0:
                    confs.append(c)
            text = " ".join(words)
            score = (sum(confs) / len(confs)) if confs else -1.0
            return text, score
        except Exception:
            return None, None

    def run(self, images: list[Image.Image], tick=None):
        lock = threading.Lock()

        def work(img):
            out = self._one(img)
            if tick is not None:                 # tqdm.update is not thread-safe
                with lock:
                    tick(1)
            return out

        with ThreadPoolExecutor(max_workers=self.num_threads) as pool:
            results = list(pool.map(work, images))
        texts = [r[0] for r in results]
        scores = [r[1] for r in results]
        return texts, scores


# ======================================================================================
# PaddleOCR recognition engine (recognition-only)
# ======================================================================================
class PaddleOCREngine:
    """PaddleOCR recognition-only over a list of PIL images.

    Detection is switched OFF on purpose (see module docstring): the inputs are already
    line crops. Uses paddleocr 3.x `TextRecognition` with the Telugu recognition model
    loaded by name, and reads `predict`'s per-line confidence alongside the text.

    Contract: `run(images)` takes a list of PIL images and returns a (texts, scores) pair,
    each the same length as `images`. A text of None means Paddle FAILED on that row (a
    whole batch errored) and must not be mistaken for an empty reading -- the caller
    writes "" / -1.0 for those.
    """

    name = "paddle"

    DEFAULT_MODEL = "te_PP-OCRv5_mobile_rec"

    def __init__(self, model_name: str | None = None, lang: str = "te",
                 batch_size: int = 128, use_gpu: bool = True, device: str | None = None,
                 **kwargs):
        from paddleocr import TextRecognition

        self.model_name = model_name or (self.DEFAULT_MODEL if lang == "te"
                                         else f"{lang}_PP-OCRv5_mobile_rec")
        # Pin paddle to a specific GPU on multi-GPU hosts, e.g. device="gpu:1".
        self.device = device
        if device is not None:
            kwargs.setdefault("device", device)
        try:
            self._rec = TextRecognition(model_name=self.model_name, **kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"could not load PaddleOCR recognition model {self.model_name!r}: {exc}. "
                f"Pass model_name= explicitly if this release names it differently."
            ) from exc
        self.batch_size = batch_size

        # Report where paddle actually landed, so a silent CPU fallback on a GPU box is
        # visible rather than just slow.
        try:
            import paddle
            on_gpu = (paddle.device.is_compiled_with_cuda()
                      and paddle.device.cuda.device_count() > 0)
        except Exception:
            on_gpu = False
        if use_gpu and not on_gpu:
            print("[paddle] WARNING: use_gpu=True but no CUDA device is visible to paddle; "
                  "running on CPU. Install paddlepaddle-gpu and enable the Kaggle GPU.")
        print(f"[paddle] model={self.model_name} "
              f"device={device or ('gpu' if on_gpu else 'cpu')} "
              f"batch_size={self.batch_size}")

    @staticmethod
    def _extract(res) -> tuple[str, float]:
        """Pull (text, score) out of a paddleocr 3.x result object/dict."""
        if res is None:
            return "", 0.0
        if isinstance(res, str):
            return res, 0.0
        get = res.get if hasattr(res, "get") else (lambda k, d=None: getattr(res, k, d))
        text = get("rec_text", None)
        if text is None:
            texts = get("rec_texts", None)
            text = texts[0] if isinstance(texts, (list, tuple)) and texts else ""
        score = get("rec_score", None)
        if score is None:
            scores = get("rec_scores", None)
            score = scores[0] if isinstance(scores, (list, tuple)) and scores else 0.0
        return (text or ""), float(score or 0.0)

    def run(self, images: list[Image.Image], tick=None):
        texts: list[str | None] = []
        scores: list[float | None] = []
        for i in range(0, len(images), self.batch_size):
            batch = [np.array(im.convert("RGB")) for im in images[i:i + self.batch_size]]
            try:
                out = list(self._rec.predict(batch))
            except Exception as exc:
                print(f"[paddle] batch of {len(batch)} failed: {exc}")
                texts.extend([None] * len(batch))
                scores.extend([None] * len(batch))
                if tick:
                    tick(len(batch))
                continue
            if len(out) != len(batch):
                # never silently misalign predictions with rows
                print(f"[paddle] returned {len(out)} results for {len(batch)} inputs; "
                      f"marking the batch failed")
                texts.extend([None] * len(batch))
                scores.extend([None] * len(batch))
                if tick:
                    tick(len(batch))
                continue
            for o in out:
                t, sc = self._extract(o)
                texts.append(t)
                scores.append(sc)
            if tick:
                tick(len(batch))
        return texts, scores


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
