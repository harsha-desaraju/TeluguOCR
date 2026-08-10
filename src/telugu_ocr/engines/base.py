"""The OCR engine contract, shared by benchmarking and pseudo-labelling.

WHY THERE WERE TWO BASE CLASSES
    `OCREngine` existed in benchmark/engines.py and again in
    pipelines/label/consensus_labelling.py, and they were not drifted copies of one
    idea -- they were two different contracts wearing one name:

      benchmark    transcribe(images) normalised single-vs-batch, coerced None to "",
                   and validated that the backend returned one text per input.
      labeller     run(images) -> list[str | None], plus `device_kind` so GPU engines
                   could be serialised against each other, a `_tick` progress callback,
                   and permission to return (texts, scores) when the backend exposes a
                   confidence.

    Neither is redundant. The labeller needs None to mean "this engine failed on this
    line" so consensus can ignore it rather than score an empty string against the
    reference; the benchmark needs a plain list of strings.

    So `run` is the PRIMITIVE every engine implements -- the labeller's contract,
    because it carries strictly more information -- and `transcribe` is a wrapper over
    it providing the benchmark's contract. An engine implements one method; both
    callers keep the semantics they had.
"""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image


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
