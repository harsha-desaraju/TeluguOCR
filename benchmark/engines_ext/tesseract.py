"""Tesseract, via pytesseract. One implementation, previously three."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from benchmark.engines_ext.base import OCREngine


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
