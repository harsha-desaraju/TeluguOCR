"""Third-party OCR engine adapters, with one shared contract.

Tesseract and PaddleOCR wrappers. These are NOT part of the model: nothing in
src/telugu_ocr/ imports them, and the package builds and trains fine without
pytesseract or paddleocr installed.

NOTE ON LAYERING. They live under benchmark/ but have TWO consumers -- benchmark/ and
pipelines/label/, which needs them for consensus pseudo-labelling. So the labelling
pipeline imports from benchmark/, which is the wrong way round for a data pipeline to
depend. It is a deliberate, known trade: keeping them here rather than in a third
top-level package. If pipelines/ ever needs to run somewhere benchmark/ is not
available, this is the seam to cut.


Collapsed in phase 3 from three parallel sets: benchmark/engines.py and the two
pseudo-labelling scripts each carried their own OCREngine / TesseractEngine /
PaddleOCREngine. See base.py for why the two base classes were not the same thing.

Surya and the repo's own encoder-decoder engine stay where they are: surya-ocr cannot be
a declared dependency (every release pins pillow>=10.2,<11 against the rest of the
stack), and the model engine is bound up with checkpoint loading.
"""

from benchmark.engines_ext.base import OCREngine, to_pil
from benchmark.engines_ext.paddle import PaddleOCREngine
from benchmark.engines_ext.tesseract import TesseractEngine

__all__ = ["OCREngine", "PaddleOCREngine", "TesseractEngine", "to_pil"]
