"""OCR engines with one shared contract.

Collapsed in phase 3 from three parallel sets: benchmark/engines.py and the two
pseudo-labelling scripts each carried their own OCREngine / TesseractEngine /
PaddleOCREngine. See base.py for why the two base classes were not the same thing.

Surya and the repo's own encoder-decoder engine stay where they are: surya-ocr cannot be
a declared dependency (every release pins pillow>=10.2,<11 against the rest of the
stack), and the model engine is bound up with checkpoint loading.
"""

from src.telugu_ocr.engines.base import OCREngine, to_pil
from src.telugu_ocr.engines.paddle import PaddleOCREngine
from src.telugu_ocr.engines.tesseract import TesseractEngine

__all__ = ["OCREngine", "PaddleOCREngine", "TesseractEngine", "to_pil"]
