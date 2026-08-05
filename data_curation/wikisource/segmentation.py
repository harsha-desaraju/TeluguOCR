"""Cut a Wikisource page scan into ordered text-line boxes.

Tesseract's page layout analysis does the cutting. Two alternatives were built and
measured against it on 163 pages drawn from 138 different books -- a horizontal
ink-projection segmenter, and PP-OCR text detection with the word boxes merged back
into lines -- and this is the one that won:

    segmenter          corpus CER   struct%   yield@0.10   yield@0.15
    tesseract_layout       0.556      11.2       40.3%        49.6%
    paddle_det             0.553      10.1       34.1%        46.2%
    projection             0.593      23.1       26.6%        34.4%

`paddle_det` is level with it on reading accuracy and actually a shade better at
loose thresholds, but the gap opens up as the threshold tightens (+18% relative
lines at a 0.10 cut, +29% at 0.05), and this pipeline is run strict. The likely
reason is that tesseract emits real textline boxes that hug the printed line, while
merging detection fragments into a bounding box leaves it slightly loose -- a small
penalty that only bites when you are demanding close agreement. The projection
segmenter was the fastest by 3x but over-segmented badly, and its junk boxes
poisoned the concatenated hypothesis that the aligner runs on.

Both losing segmenters have been deleted rather than kept behind a flag, and this
package has never been committed, so they are not recoverable from anywhere -- if the
comparison needs redoing after the recogniser is retrained, they have to be rewritten.
The projection one is the fiddly half: an ink profile alone merges touching lines on
letterpress and is defeated outright by a page border, which lifts the profile off
zero so the whole text block reads as one band.

CONTRACT
    segmenter.segment(page) -> list[LineBox], in reading order.
    `page` is a PreparedPage from prepare_page(), not a raw array: deskewing is done
    once and shared, both because it is the slow part and because the OCR engine has
    to crop from the same deskewed image the boxes were measured on.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


# ======================================================================================
# Shared page preparation
# ======================================================================================
@dataclass
class PreparedPage:
    """One page, prepared once and reused by the segmenter and the OCR engine.

    gray    deskewed grayscale, uint8. Crops for the recogniser come from HERE, not
            from `binary` -- the models were trained on grayscale, and a
            hard-thresholded crop measurably hurts on faint scans.
    binary  deskewed ink mask. Used only to measure how much ink is inside a box.
    """

    gray: np.ndarray
    binary: np.ndarray
    angle: float

    @property
    def shape(self):
        return self.gray.shape


def prepare_page(img, deskew: bool = True, max_skew: float = 8.0) -> PreparedPage:
    """Grayscale + deskew + binarise a page image.

    The skew estimate is clamped to +/-max_skew degrees. `determine_skew` on a page
    whose dominant straight lines are a decorative rule or a plate border
    occasionally returns something like 45 degrees; rotating by that turns a readable
    page into confetti, and nothing downstream can recover from it. A scan with
    genuinely more than 8 degrees of skew is rare enough that declining to correct it
    is the better failure.
    """
    arr = np.asarray(img.convert("L") if isinstance(img, Image.Image) else img)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = np.ascontiguousarray(arr, dtype=np.uint8)

    angle = 0.0
    if deskew:
        try:
            from deskew import determine_skew

            est = determine_skew(gray)
            if est is not None and abs(est) <= max_skew:
                angle = float(est)
        except Exception:
            angle = 0.0
        if angle:
            h, w = gray.shape
            rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            gray = cv2.warpAffine(gray, rot, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    # Adaptive rather than Otsu: these scans have page-scale illumination gradients
    # (gutter shadow, uneven flatbed lighting) that a single global threshold either
    # blows out on one side or floods on the other.
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 15) > 0
    return PreparedPage(gray=gray, binary=binary, angle=angle)


@dataclass
class LineBox:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def crop(self, page: PreparedPage, pad: int = 3) -> Image.Image:
        """Grayscale crop with a little padding, clipped to the page."""
        h, w = page.gray.shape
        return Image.fromarray(page.gray[max(0, self.y0 - pad):min(h, self.y1 + pad),
                                         max(0, self.x0 - pad):min(w, self.x1 + pad)])


# ======================================================================================
# Filtering
# ======================================================================================
def preprocess_line(img: Image.Image, height: int = 64, max_width: int = 2048,
                    downsample: int = 8) -> np.ndarray:
    """A line crop in the form the encoder consumes: grayscale uint8, fixed height,
    width padded to a multiple of `downsample`.

    Mirrors ImagePreprocessor in src/image_encoder/utils.py, including the over-wide
    branch: past `max_width` the scale is driven by width instead and the height
    shortfall is padded, rather than squashing the glyphs horizontally. Getting that
    wrong is silent -- the model just reads badly -- so keep it in sync.

    Storing crops in this form rather than as cut has two payoffs: the dataset is a
    quarter the size, and the stored bytes are provably the pixels the recogniser saw
    when it judged the label. It is also idempotent, so running the encoder's own
    preprocessing over a stored crop is a no-op rather than a second resample.
    """
    im = img.convert("L")
    w, h = im.size
    scale = height / h
    if scale * w > max_width:
        scale = max_width / w
        target_h = max(1, int(scale * h))
        im = im.resize((max_width, target_h), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8)
        pad_top = (height - target_h) // 2
        arr = np.pad(arr, ((pad_top, height - target_h - pad_top), (0, 0)),
                     constant_values=255)
    else:
        im = im.resize((max(1, int(scale * w)), height), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8)

    pad_w = (-arr.shape[1]) % downsample
    if pad_w:
        arr = np.pad(arr, ((0, 0), (0, pad_w)), constant_values=255)
    return arr


def encode_jpeg(arr: np.ndarray, quality: int = 90) -> bytes:
    """JPEG-encode a preprocessed crop. ~8 KB/line at height 64 quality 90, against
    ~26 KB as PNG. The page scans are JPEG to begin with, so the crops already carry
    those artefacts and re-encoding costs little (measured mean pixel delta ~1/255)."""
    import io

    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue()


@dataclass
class BoxFilter:
    """Geometry filters applied to the segmenter's output.

    `max_ink_frac` is the one that is not obvious: a box whose pixels are more than
    ~55% ink is not text, it is a photograph, a plate, or a solid decorative rule.
    Wikisource books are full of them -- cover pages, portrait plates, ornamental
    dividers -- and without this they reach the recogniser, which dutifully invents a
    line of text that the aligner then has to charge against the real ground truth.
    """

    min_height: int = 12
    min_width: int = 40
    max_height_frac: float = 0.25   # of page height
    max_width_frac: float = 1.0     # of page width
    min_ink_frac: float = 0.005     # blank strip
    max_ink_frac: float = 0.55      # photo / plate / solid rule

    def keep(self, box: LineBox, page: PreparedPage) -> bool:
        h, w = page.binary.shape
        if box.height < self.min_height or box.width < self.min_width:
            return False
        if box.height > self.max_height_frac * h or box.width > self.max_width_frac * w:
            return False
        patch = page.binary[box.y0:box.y1, box.x0:box.x1]
        if patch.size == 0:
            return False
        return self.min_ink_frac <= float(patch.mean()) <= self.max_ink_frac


# ======================================================================================
# Segmenter
# ======================================================================================
class TesseractLayoutSegmenter:
    """Tesseract's page layout analysis, taking its level-4 (textline) boxes.

    psm 3 runs full automatic page segmentation, which is what is wanted here -- the
    input is a whole scanned page, not a line crop, and tesseract's block/paragraph/
    line hierarchy handles indentation, centred headings and mixed Telugu/Latin
    without any of it being hand-coded. Only the geometry is used; whatever text
    tesseract reads while doing this is discarded, because the recogniser that
    actually reads the crops is chosen separately.

    Boxes come back in tesseract's reading order already; the re-sort by (top, left)
    is just a stable tiebreak.
    """

    name = "tesseract_layout"

    def __init__(self, lang: str = "tel+eng", psm: int = 3,
                 box_filter: BoxFilter | None = None, timeout: int = 60):
        import pytesseract

        self._pt = pytesseract
        self.lang = lang
        self.config = f"--oem 3 --psm {psm}"
        self.box_filter = box_filter or BoxFilter()
        self.timeout = timeout

    def segment(self, page: PreparedPage) -> list[LineBox]:
        try:
            data = self._pt.image_to_data(
                Image.fromarray(page.gray),
                output_type=self._pt.Output.DATAFRAME,
                config=self.config,
                lang=self.lang,
                timeout=self.timeout,
            )
        except Exception:
            return []
        if data is None or len(data) == 0:
            return []

        boxes = []
        for _, row in data[data["level"] == 4].iterrows():
            box = LineBox(int(row["left"]), int(row["top"]),
                          int(row["left"] + row["width"]), int(row["top"] + row["height"]))
            if self.box_filter.keep(box, page):
                boxes.append(box)
        boxes.sort(key=lambda b: (b.y0, b.x0))
        return boxes
