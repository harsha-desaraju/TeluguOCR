"""PaddleOCR recognition, via paddleocr 3.x TextRecognition.

One implementation, previously three. The pseudo-labeller's version was the superset --
it pins a specific GPU (so paddle and our torch model can occupy different cards) and
returns the per-line confidence paddle already computes. The benchmark's copy had
neither; it now gets both, and `transcribe` drops the scores it does not want.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from benchmark.engines_ext.base import OCREngine

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
