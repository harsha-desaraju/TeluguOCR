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
"""

from __future__ import annotations

import html
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------
from benchmark.engines_ext.base import OCREngine, to_pil as _to_pil
from benchmark.engines_ext.paddle import PaddleOCREngine
from benchmark.engines_ext.tesseract import TesseractEngine




# ---------------------------------------------------------------------------
# Tesseract
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# PaddleOCR
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Surya
# ---------------------------------------------------------------------------


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
