"""Swappable OCR engines behind one fixed contract.

THE CONTRACT
    Every engine is a callable that takes an image or a list of images and gives back
    a string or a list of strings:

        engine = TesseractEngine()
        text  = engine(img)            # PIL.Image | path | ndarray  -> str
        texts = engine([img1, img2])   # list of the above           -> list[str]

    Swapping the engine is the only thing that changes; nothing downstream of it moves.
    An engine NEVER returns None and never raises for a single bad image -- a failure is
    an empty string, counted in `n_failed` -- so a benchmark loop cannot silently die on
    one corrupt crop.

    Subclasses implement `run(list[Image.Image]) -> list[str]` only. The
    single-vs-list handling and the loading of paths/arrays live in the base class, so
    the contract is written once and cannot drift between engines.

WHY NOT REUSE consensus_labelling.OCREngine
    That class already exists and is good, but its `run()` returns `list[str | None]`
    and sometimes a `(texts, scores)` tuple, which is exactly the variability the
    benchmark wants gone. These wrap the same underlying libraries with the narrower
    contract above. The tricky per-engine settings are carried over deliberately (see
    each class) rather than rediscovered.

ENGINES
    TesseractEngine   CPU, needs the tesseract binary + `tel` traineddata.
    PaddleOCREngine   recognition-only (the inputs are already line crops).
    SuryaEngine       surya-ocr 0.14.x, recognition-only, with its document-model
                      markup and neighbouring-line output undone -- see the class.
    TeluguOCREngine   our encoder-decoder checkpoint. `decode` selects the head:
                        "ctc"   -- CTC greedy off the encoder's CTC head
                        "beam"  -- LM beam search over the text decoder
                        "joint" -- beam n-best rescored by
                                   lam*logP_ctc + (1-lam)*logP_attn
                      The decoding itself is imported from
                      scripts/eval/encoder_decoder.py rather than re-implemented, so
                      this benchmark and that script cannot drift apart.

WHAT IS IN HERE
    The shared `OCREngine` contract, the third-party adapters (Tesseract, PaddleOCR,
    Surya) and the wrapper around this repo's own model. One file, as it was before the
    phase-3 refactor -- every heavy import (torch, paddleocr, surya, pytesseract) is
    lazy, inside the method that needs it, so importing this module is cheap and costs
    nothing if a given backend is not installed.

    NOTE ON LAYERING. `pipelines/label/` imports TesseractEngine / PaddleOCREngine from
    here for consensus pseudo-labelling, so the labelling pipeline depends on
    benchmark/. That is the wrong direction for a data pipeline and it is a deliberate
    trade: these adapters are not part of the model (src/telugu_ocr/ imports nothing
    from them) and keeping them in one place beat spreading them over a third top-level
    package. If pipelines/ ever has to run somewhere benchmark/ is not available, this
    is the seam to cut.
"""

from __future__ import annotations

import html
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
import numpy as np


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Tesseract
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# PaddleOCR
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Surya
# ---------------------------------------------------------------------------



# --------------------------------------------------------------------------
# The shared contract
# --------------------------------------------------------------------------
def to_pil(image) -> Image.Image:
    """Accept a PIL image, a filesystem path, or a numpy array."""
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (str, Path, os.PathLike)):
        img = Image.open(image)
        img.load()                      # decode now; the file handle is released
        return img
    if hasattr(image, "__array_interface__") or hasattr(image, "shape"):
        return Image.fromarray(image)
    raise TypeError(f"cannot interpret {type(image).__name__} as an image")


_to_pil = to_pil          # the name benchmark/engines.py used


class OCREngine:
    """Base class. Implement `run`; `transcribe` is derived from it.

    `device_kind` decides scheduling: 'gpu' engines are serialised against each other,
    'cpu' engines all run concurrently.
    """

    name: str = "base"
    device_kind: str = "cpu"
    # Set by run_engines; call self._tick(k) after finishing k images so the progress bar
    # advances DURING a chunk instead of jumping once at the end of it.
    _on_items = None

    def run(self, images: list[Image.Image]):
        """list[PIL] -> list[str | None], or (list[str | None], list[float]).

        None means this engine failed on that image, which is NOT the same as reading it
        as empty -- consensus must be able to tell those apart.
        """
        raise NotImplementedError

    def transcribe(self, images):
        """image | list[image] -> str | list[str]. Failures become "".

        The benchmark's contract: accepts a single image or a batch, accepts paths and
        arrays as well as PIL, and always returns plain strings.
        """
        single = isinstance(images, (Image.Image, str, Path, os.PathLike)) or not (
            isinstance(images, (list, tuple)))
        batch = [images] if single else list(images)
        if not batch:
            return "" if single else []

        texts = self.run([to_pil(im) for im in batch])
        if isinstance(texts, tuple):        # (texts, scores) -> drop the scores
            texts = texts[0]

        if len(texts) != len(batch):
            raise RuntimeError(
                f"{self.name}: got {len(texts)} texts for {len(batch)} images")
        texts = ["" if t is None else str(t) for t in texts]
        return texts[0] if single else texts

    # Callable form, so an engine can be passed anywhere a function is expected.
    __call__ = transcribe

    def _tick(self, k: int = 1) -> None:
        """Report k finished images. Safe to call from worker threads, and a no-op when
        no progress sink is attached, so engines can call it unconditionally."""
        cb = self._on_items
        if cb is not None:
            cb(k)

    def close(self) -> None:
        """Release whatever the backend is holding. Default: nothing to release.

        benchmark/run_benchmark.py calls this on every engine after scoring it, so that
        a GPU-backed engine frees its device memory before the next one loads. Engines
        with nothing to free inherit the no-op -- which is the point of defining it here
        rather than making the caller guess with hasattr.
        """
        return None


# --------------------------------------------------------------------------
# Tesseract
# --------------------------------------------------------------------------
class TesseractEngine(OCREngine):
    """Tesseract on line crops, one thread per in-flight crop.

    Two settings matter, and the second one is measured rather than assumed:

    * upscaling. Tesseract wants roughly 30-35px of x-height; a 64px line crop sits at
      the bottom of its comfortable range, and a 2x cubic upscale measurably helps.
      Beyond that it barely moves (0.128-0.139 CER across 1x-3x), so 2x is the default.

    * `psm`, and THE DEFAULT HERE IS NOT THE BEST ONE. Measured over 40 human-reviewed
      lines (tesseract 5.5.0, normalized CER, lower is better):

          tel      psm 13  up 2.0   0.130
          tel+eng  psm 13  up 2.0   0.205     adding `eng` hallucinates Latin onto Telugu
          tel      psm  7  up 2.0   0.301     <- the default below
          tel+eng  psm  7  up 2.0   0.370
          tel      psm  6  up 2.0   0.341

      psm 7 ("single text line") is the documented choice for line crops, but on this
      build it frequently returns NOTHING for a Telugu line -- 7 of those 40 came back
      empty -- and is 2.3x worse than psm 13 ("raw line"), which bypasses the
      Tesseract-specific layout hacks.

      The default stays 7 because the pseudo-labelling pipelines relied on it and every
      label they have already written was produced with it; changing it silently would
      make new labels inconsistent with old ones. The benchmark passes psm=13
      explicitly, so its baseline is not crippled. Consider moving the labellers to 13
      as a deliberate step, with a re-label.
    """

    name = "tesseract"
    device_kind = "cpu"

    def __init__(self, lang: str = "tel", psm: int = 7, oem: int = 1,
                 upscale: float = 2.0, num_threads: int = 8, timeout: int = 20):
        import pytesseract

        self._pt = pytesseract
        self.lang = lang
        self.config = f"--psm {psm} --oem {oem}"
        self.upscale = upscale
        self.num_threads = num_threads
        self.timeout = timeout
        self.n_failed = 0

    def _one(self, img: Image.Image) -> str | None:
        try:
            im = img.convert("L")
            if self.upscale and self.upscale != 1.0:
                im = im.resize((max(1, int(im.width * self.upscale)),
                                max(1, int(im.height * self.upscale))),
                               Image.BICUBIC)
            return self._pt.image_to_string(im, lang=self.lang, config=self.config,
                                            timeout=self.timeout)
        except Exception:
            # None, not "": the labeller must be able to tell a failed engine apart from
            # one that genuinely read nothing. transcribe() coerces it to "" for the
            # benchmark, which does not make that distinction.
            self.n_failed += 1
            return None
        finally:
            self._tick(1)

    def run(self, images: list[Image.Image]) -> list[str | None]:
        with ThreadPoolExecutor(max_workers=self.num_threads) as pool:
            return list(pool.map(self._one, images))


# --------------------------------------------------------------------------
# PaddleOCR
# --------------------------------------------------------------------------
class PaddleOCREngine(OCREngine):
    """PaddleOCR recognition-only.

    Detection is switched OFF on purpose: the inputs are already line crops, so letting
    Paddle re-detect boxes inside a 64px strip only invents sub-boxes and drops text.

    Uses paddleocr 3.x `TextRecognition` with the Telugu recognition model directly,
    rather than the full `PaddleOCR` pipeline. That skips detection entirely, which is what
    you want here: the inputs are already line crops, and letting the pipeline re-detect
    boxes inside one strip invents sub-boxes and returns the fragments OUT OF READING
    ORDER, so naively joining them silently scrambles the text.

    Telugu support is real but easy to miss: paddleocr's `_utils/langs.py` only lists the
    script GROUPS (latin/arabic/cyrillic/devanagari), so grepping it suggests Telugu is
    absent. It is not -- `lang="te"` resolves through the model registry to
    `te_PP-OCRv5_mobile_rec`, which is what this engine loads by name. Only the mobile
    variant exists; there is no `te_PP-OCRv5_server_rec`.

    `predict` also returns a per-line confidence, which is recorded alongside the text --
    a cheap extra filter for pseudo-labels.

    Verified against paddleocr 3.7.0 / paddle 3.3.1.
    """

    name = "paddle"
    device_kind = "gpu"          # assume GPU; harmless if it falls back to CPU

    DEFAULT_MODEL = "te_PP-OCRv5_mobile_rec"

    def __init__(self, model_name: str | None = None, lang: str = "te",
                 batch_size: int = 32, device: str | None = None, **kwargs):
        from paddleocr import TextRecognition

        self.model_name = model_name or (self.DEFAULT_MODEL if lang == "te"
                                         else f"{lang}_PP-OCRv5_mobile_rec")
        # Pin paddle to a specific GPU on multi-GPU hosts, e.g. device="gpu:0" while our
        # torch model takes cuda:1. Passed straight through to TextRecognition, and also
        # kept on the instance so run_engines can tell whether two GPU engines collide.
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
        # Scheduling honesty: the GPU lock exists to stop two GPU engines fighting over the
        # device. When paddle is running on CPU (no CUDA build / no device) holding that lock
        # only serializes it against our model for no reason, so declare what it really is.
        try:
            import paddle
            self.device_kind = ("gpu" if paddle.device.is_compiled_with_cuda()
                                and paddle.device.cuda.device_count() > 0 else "cpu")
        except Exception:
            self.device_kind = "cpu"

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

    def run(self, images: list[Image.Image]):
        texts: list[str | None] = []
        scores: list[float | None] = []
        for i in range(0, len(images), self.batch_size):
            batch = [np.array(im.convert("RGB")) for im in images[i:i + self.batch_size]]
            try:
                out = list(self._rec.predict(batch))
            except Exception as exc:
                print(f"[{self.name}] batch of {len(batch)} failed: {exc}")
                texts.extend([None] * len(batch))
                scores.extend([None] * len(batch))
                self._tick(len(batch))
                continue
            if len(out) != len(batch):
                # never silently misalign predictions with rows
                print(f"[{self.name}] returned {len(out)} results for {len(batch)} inputs; "
                      f"marking the batch failed")
                texts.extend([None] * len(batch))
                scores.extend([None] * len(batch))
                self._tick(len(batch))
                continue
            for o in out:
                t, sc = self._extract(o)
                texts.append(t)
                scores.append(sc)
            self._tick(len(batch))
        return texts, scores

class SuryaEngine(OCREngine):
    """Surya 1 (surya-ocr 0.14.x), recognition only.

    Detection is skipped the way Surya supports natively: pass `bboxes` covering the
    whole crop and `det_predictor=None`. Same reasoning as PaddleOCREngine -- the input
    is already one line.

    Surya is a DOCUMENT model, so two things must be undone before its output is
    comparable to a line recogniser's, and both cost a lot of accuracy if ignored.
    Measured over the first 40 benchmark lines (akshara error rate, lower is better):

        math_mode=False, strip tags, longest segment   0.187   <- defaults here
        math_mode=False, strip tags, first segment     0.246
        math_mode=False, strip tags, join segments     0.300
        math_mode=False, raw output                    0.413
        math_mode=True,  raw output                    0.523

    1. MARKUP. Surya emits inline HTML -- <br>, <b>, <i>, and with math_mode=True a lot
       of <math> -- which is formatting, not a reading error. Tags are stripped and
       entities unescaped. math_mode=False also makes the underlying reading better
       here, not just less marked up.
    Batches are kept small and MPS memory is returned after each one (see `_free`):
    a 1044-line pass segfaulted at batch 19 of 33 without it.

    2. THE NEIGHBOURING LINE. These crops carry a sliver of the line above or below,
       and a document model reads it, returning two <br>-separated segments (6 of those
       40). Which segment is the real line varies -- sometimes first, sometimes second
       -- so `take="longest"` picks the longest, the full line being longer than a
       partial sliver. It is a heuristic, but a reference-free one: it never looks at
       the ground truth. "join" (score everything Surya read) and "first" are available
       for comparison; the table above is the argument for the default.
    """

    name = "surya"
    device_kind = "gpu"

    _TAG = re.compile(r"<[^>]+>")
    _BR = re.compile(r"<br\s*/?>", re.IGNORECASE)

    def __init__(self, math_mode: bool = False, take: str = "longest",
                 batch_size: int = 16, recognition_batch_size: int | None = None,
                 free_every_batch: bool = True):
        from surya.common.surya.schema import TaskNames
        from surya.recognition import RecognitionPredictor

        if take not in ("longest", "first", "join"):
            raise ValueError(f"take must be longest|first|join, got {take!r}")
        self._rec = RecognitionPredictor()
        self._task = TaskNames.ocr_with_boxes
        self.math_mode = math_mode
        self.take = take
        self.batch_size = batch_size
        self.recognition_batch_size = recognition_batch_size
        self.free_every_batch = free_every_batch
        self.n_failed = 0
        self.n_multi_segment = 0

    @staticmethod
    def _free():
        """Hand MPS memory back between batches.

        Surya segfaulted partway through a 1044-line pass (batch 19 of 33) with
        everything else already released, which is the signature of memory growing
        across batches rather than of a conflict with another engine. torch does not
        return MPS blocks on its own, so they are dropped explicitly here.
        """
        import gc

        gc.collect()
        try:
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _segments(self, text: str) -> list[str]:
        text = self._BR.sub("\n", str(text))
        text = self._TAG.sub("", text)
        text = html.unescape(text)
        return [seg.strip() for seg in text.split("\n") if seg.strip()]

    def _pick(self, text: str) -> str:
        segs = self._segments(text)
        if not segs:
            return ""
        if len(segs) > 1:
            self.n_multi_segment += 1
        if self.take == "first":
            return segs[0]
        if self.take == "join":
            return " ".join(segs)
        return max(segs, key=len)

    def run(self, images: list[Image.Image]) -> list[str]:
        out: list[str] = []
        for i in range(0, len(images), self.batch_size):
            chunk = [im.convert("RGB") for im in images[i:i + self.batch_size]]
            # One box per image covering all of it: "read this whole crop as one line".
            bboxes = [[[0, 0, im.width, im.height]] for im in chunk]
            try:
                results = self._rec(
                    chunk, task_names=[self._task] * len(chunk), bboxes=bboxes,
                    det_predictor=None, math_mode=self.math_mode,
                    recognition_batch_size=self.recognition_batch_size)
            except Exception as exc:
                print(f"[{self.name}] batch of {len(chunk)} failed: {exc}")
                self.n_failed += len(chunk)
                out.extend([""] * len(chunk))
                continue
            for res in results:
                lines = getattr(res, "text_lines", None) or []
                out.append(self._pick(" ".join(ln.text for ln in lines)))
            if self.free_every_batch:
                self._free()
        return out


# ---------------------------------------------------------------------------
# Our model
# ---------------------------------------------------------------------------
class TeluguOCREngine(OCREngine):
    """The stage-2 encoder-decoder checkpoint.

    `decode` picks which head produces the string:
      "ctc"    per-frame argmax on the CTC head, collapse repeats, drop blanks. No
               language knowledge; cheap (no beam search runs at all).
      "beam"   KV-cached beam search over the cross-attention text decoder.
      "joint"  the beam's n-best re-ranked by lam*logP_ctc + (1-lam)*logP_attn, with
               logP_ctc from the CTC forward algorithm over the SAME encoder frames.

    One encoder forward feeds whichever decoders are needed, exactly as in
    scripts/eval/encoder_decoder.py -- whose `preprocess_image`, `ctc_greedy_ids`,
    `beam_search` and `ctc_hyp_logprobs` are imported rather than copied, so this
    benchmark reports the same numbers that script does.

    Images wider than the encoder's max width (2048px once scaled to height 64) cannot
    be encoded. Rather than silently truncating them, `too_wide` records them and the
    engine returns "". The harness drops those samples for EVERY engine so the
    comparison stays like-for-like.
    """

    name = "model"
    device_kind = "gpu"

    def __init__(self, checkpoint: str, vocab_file: str, decode: str = "joint",
                 lam: float = 0.4, beam_width: int = 5, max_new_tokens: int = 200,
                 beam_len_alpha: float = 0.0, device: str | None = None,
                 embed_dim: int = 512, hidden_dim: int = 1368, num_heads: int = 8,
                 num_layers: int = 16, ctx_len: int = 256,
                 max_image_width: int = 2048, max_frames: int = 256):
        import torch

        from src.telugu_ocr.models.encoder_decoder import EncoderDecoder
        from src.telugu_ocr.models.image_encoder import CTCEncoderConfig
        from src.telugu_ocr.models.text_decoder import GPTConfig
        from src.telugu_ocr.tokenizer.grapheme import TeluguGraphemeTokenizer

        if decode not in ("ctc", "beam", "joint"):
            raise ValueError(f"decode must be ctc|beam|joint, got {decode!r}")
        self.decode = decode
        self.lam = lam
        self.beam_width = beam_width
        self.max_new_tokens = max_new_tokens
        self.beam_len_alpha = beam_len_alpha
        self.name = f"model-{decode}" + (f"@{lam}" if decode == "joint" else "")

        self._torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available()
                                 else "cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = TeluguGraphemeTokenizer(vocab_file=vocab_file)

        # Configs must match what stage-2 trained with or the strict load below rejects
        # the checkpoint (see src/telugu_ocr/training/loops/encdec.py (STAGE=2)).
        decoder_config = GPTConfig(
            vocab_size=len(self.tokenizer), embed_dim=embed_dim, hidden_dim=hidden_dim,
            num_heads=num_heads, num_layers=num_layers, ctx_len=ctx_len, dropout=0.0)
        self.encoder_config = CTCEncoderConfig(max_image_width=max_image_width,
                                               max_frames=max_frames)

        model = EncoderDecoder(self.encoder_config, decoder_config,
                               self.tokenizer.pad_token_id)
        if str(checkpoint).endswith(".safetensors"):
            from safetensors.torch import load_file
            state = load_file(checkpoint)
        else:
            state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=True)
        self.model = model.eval().to(self.device)
        self.blank_id = model.encoder_model.blank_id

        self.too_wide: list[int] = []
        self.n_failed = 0

    def max_source_width(self) -> int:
        return self.encoder_config.max_image_width

    def fits(self, img: Image.Image) -> bool:
        """False when the crop, scaled to height 64, exceeds the encoder's max width."""
        return img.width / max(img.height, 1) * 64 <= self.encoder_config.max_image_width

    def run(self, images: list[Image.Image]) -> list[str]:
        import torch

        from scripts.eval.encoder_decoder import (
            beam_search, ctc_greedy_ids, ctc_hyp_logprobs, preprocess_image,
        )

        out: list[str] = []
        for i, img in enumerate(images):
            if not self.fits(img):
                self.too_wide.append(i)
                out.append("")
                continue
            try:
                pix = preprocess_image(img).unsqueeze(0).to(self.device)

                # One encoder forward feeds every decoder below.
                with torch.no_grad():
                    enc_raw, _ = self.model.encoder_model.encode(pix, None)
                    ctc_lp = (self.model.encoder_model.ctc_head(enc_raw)[0]
                              .float().log_softmax(-1).cpu())

                if self.decode == "ctc":
                    # No beam search at all -- this is why "ctc" is the cheap engine.
                    ids = ctc_greedy_ids(ctc_lp, self.blank_id)
                    out.append(self.tokenizer.decode(
                        ids, skip_special_tokens=True).strip())
                    continue

                with torch.no_grad():
                    bridged = self.model.enc_to_dec(enc_raw)
                nbest = beam_search(self.model, bridged, self.tokenizer.bos_token_id,
                                    self.tokenizer.eos_token_id, self.beam_width,
                                    self.max_new_tokens)
                texts = [self.tokenizer.decode(h["ids"], skip_special_tokens=True).strip()
                         for h in nbest]
                attn_lp = torch.tensor([h["logp"] for h in nbest])

                if self.decode == "beam":
                    norm = attn_lp / torch.tensor(
                        [(len(h["ids"]) + 1.0) ** self.beam_len_alpha for h in nbest])
                    out.append(texts[int(norm.argmax())])
                    continue

                # joint: rescore the same n-best with the CTC forward algorithm
                ctc_scores = ctc_hyp_logprobs(ctc_lp, [h["ids"] for h in nbest],
                                              self.blank_id)
                combined = self.lam * ctc_scores + (1.0 - self.lam) * attn_lp
                if torch.isinf(combined).all():      # CTC rejected every hypothesis
                    combined = attn_lp
                out.append(texts[int(combined.argmax())])
            except Exception as exc:
                print(f"[{self.name}] image {i} failed: {exc}")
                self.n_failed += 1
                out.append("")
        return out


# ---------------------------------------------------------------------------
# Registry — add an engine here and the benchmark can name it.
# ---------------------------------------------------------------------------
def build_engine(spec: dict) -> OCREngine:
    """{"engine": "tesseract", ...kwargs} -> a ready OCREngine."""
    spec = dict(spec)
    kind = spec.pop("engine")
    builders = {
        "tesseract": TesseractEngine,
        "paddle": PaddleOCREngine,
        "surya": SuryaEngine,
        "model": TeluguOCREngine,
    }
    if kind not in builders:
        raise ValueError(f"unknown engine {kind!r}; have {sorted(builders)}")
    return builders[kind](**spec)
