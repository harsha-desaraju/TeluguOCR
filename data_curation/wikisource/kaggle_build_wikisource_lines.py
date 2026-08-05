"""Build a line-level Telugu OCR dataset from Wikisource, end to end, on Kaggle.

Self-contained on purpose (the repo convention): everything needed -- scraping, page
segmentation, the CTC recogniser, the forced aligner, preprocessing and the Hub upload
-- is inlined here so this one file can be pasted into a Kaggle notebook and run. When
the versions in data_curation/wikisource/ or src/image_encoder/ change, this copy has
to be updated to match.

WHAT IT PRODUCES
    One row per text LINE. No page-level images are kept -- pages are downloaded,
    diced, and thrown away without ever touching disk. Crops are stored exactly as the
    encoder consumes them: grayscale, height 64, width padded to a multiple of 8,
    JPEG. Storing them pre-processed means training does no geometry work and the
    stored bytes are provably what the recogniser saw when the label was judged.

    line_image      the crop, ready for the encoder
    text            the label: human-proofread Wikisource text, never OCR output
    line_cer        agreement between the crop's reading and the assigned span. NOT an
                    error rate on the label -- it measures whether the span was cut in
                    the right place.
    page_cer        the same for the whole page. A line can score well by luck on a
                    page whose transcript does not match the scan, and that cannot be
                    reconstructed from line_cer later.
    page_yield      fraction of that page's lines that passed
    accepted        verdict of the full strict policy, line and page guards together
    reject_reason   why not, when not accepted
    n_graphemes, digits_converted, starts_mid_word, ends_mid_word,
    slug, page_no, source_file, line_no, checkpoint

    Lines are NOT hard-filtered at the strict threshold. Everything clearing a loose
    BASE_MAX_CER floor is stored with its scores, so strictness is a query at training
    time rather than a decision frozen into a 30-hour rebuild. It also avoids biasing
    the set: a hard CER filter removes the lines the recogniser reads WORST -- odd
    typefaces, degraded letterpress, dense conjuncts -- which are the examples the
    model most needs. Only two things are dropped: lines whose aligned span is empty
    (no label exists for them), and lines above BASE_MAX_CER (noise by any reading).

REQUEST PACING
    There is no artificial sleep between requests, and none is needed. Downloading a
    page takes about a second; segmenting and reading it takes several. Pages are
    fetched by a small thread pool one mini-batch ahead of the processing loop, so the
    processing rate IS the rate limiter -- the average works out under one request per
    second even though the pool is 8 wide. What remains is short bursts of concurrent
    fetches, which `fetch_image` already handles: 429 and 503 are waited out with
    exponential backoff honouring Retry-After. See PACING below.

RUNTIME -- THIS WILL NOT FINISH IN ONE SESSION
    Roughly 2 seconds per page, dominated by tesseract layout analysis on the CPU (the
    GPU recogniser is the cheap part). Against ~53.5k usable pages that is close to
    30 hours, and Kaggle caps a session at 12. So the run is built to be resumed:
    pages are processed in shards, each finished shard is pushed to the Hub as its own
    config, and a restart asks the Hub which configs already exist and skips them. The
    category listing order is stable, so shard k always holds the same pages.

    MAX_RUNTIME_HOURS stops the loop cleanly before Kaggle kills the session, so the
    shard in flight is never lost. Expect to run this three or four times.

SETUP (first cell of the notebook)
    !apt-get -qq update && apt-get -qq install -y tesseract-ocr tesseract-ocr-tel
    !pip -q install rapidfuzz deskew regex
    Turn Internet ON in the notebook settings, add your HF_TOKEN as a Kaggle secret,
    and attach the checkpoint + vocab as a Kaggle dataset (see CHECKPOINT / VOCAB).
"""

import io
import json
import os
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from urllib.parse import unquote, urlparse

import cv2
import numpy as np
import regex
import requests
import torch
import torch.nn as nn
from bs4 import BeautifulSoup
from PIL import Image
from rapidfuzz.distance import Levenshtein
from tqdm.auto import tqdm

# ======================================================================================
# 1. Text normalization, graphemes, CER
# ======================================================================================
_ZERO_WIDTH_RE = re.compile(r"[​‌‍⁠﻿]")
_WS_RE = re.compile(r"\s+")
_GRAPHEME_RE = regex.compile(r"\X")
_LATIN_RE = re.compile(r"[A-Za-z]")
_PUNCT_FOLD = str.maketrans({
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
    "―": "-", "−": "-", "­": "",
    "‘": "'", "’": "'", "‚": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "″": '"',
    "⁄": "/", "…": "...",
})


def normalize(text: str) -> str:
    """NFKC, strip zero-width, fold equivalent punctuation, collapse whitespace.

    The punctuation fold is the step that matters on this corpus: NFKC passes en-dash
    and curly quotes straight through, and the recognisers disagree on those for
    identical glyphs, which would otherwise show up as CER that is really formatting.
    Zero-width characters matter twice over -- the tokenizer vocab has no entry
    containing one, and ZWNJ changes grapheme clustering outright.
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = _ZERO_WIDTH_RE.sub("", t)
    t = t.translate(_PUNCT_FOLD)
    return _WS_RE.sub(" ", t).strip()


def graphemes(text: str) -> list:
    return _GRAPHEME_RE.findall(text) if text else []


def cer(ref: str, hyp: str) -> float:
    """Grapheme-level CER, reference length as the denominator. Not clipped at 1.0."""
    r, h = graphemes(ref), graphemes(hyp)
    if not r:
        return 0.0 if not h else 1.0
    return Levenshtein.distance(r, h) / len(r)


def english_frac(text: str) -> float:
    letters = [c for c in (text or "") if c.isalpha()]
    if not letters:
        return 0.0
    return sum(bool(_LATIN_RE.match(c)) for c in letters) / len(letters)


# ======================================================================================
# 2. Forced alignment
#
# Wikisource gives one block of text per page and proofreaders reflow prose, so its
# newlines are paragraph breaks, not the printer's line breaks. Concatenate the noisy
# per-line readings into one string, align that against the page text once, and push
# each line's boundary through the alignment to find where to cut. The readings only
# locate the boundaries; the stored label is always the human text.
# ======================================================================================
ARABIC_DIGITS = "0123456789"
TELUGU_DIGITS = "౦౧౨౩౪౫౬౭౮౯"
_ARABIC_TO_TELUGU = str.maketrans(ARABIC_DIGITS, TELUGU_DIGITS)


def _prepare_ground_truth(text: str):
    """Graphemes of the page text, plus where the transcript's own line breaks fell.

    Usually those are paragraph breaks, but on title pages, tables of contents and
    real verse the source carries <br> and they line up with the printed lines exactly.
    Collapsing them to spaces throws away a free hint, so their positions are kept.
    """
    gt, newline_at = [], set()
    for i, raw in enumerate(ln for ln in (normalize(x) for x in text.split("\n")) if ln):
        if i:
            gt.append(" ")
            newline_at.add(len(gt))
        gt.extend(graphemes(raw))
    return gt, newline_at


def _index_map(src: list, dest: list) -> np.ndarray:
    """Map every position in `src` to the corresponding position in `dest`.

    Linear across each equal/replace block (so blocks of unequal length still map
    sensibly); a delete block collapses onto one dest position; an insert block
    contributes no src positions. Monotone, so cut points can never cross.
    """
    n_src, n_dest = len(src), len(dest)
    mapping = np.zeros(n_src + 1, dtype=np.int64)
    for op in Levenshtein.opcodes(src, dest):
        n = op.src_end - op.src_start
        if n == 0:
            continue
        m = op.dest_end - op.dest_start
        for k in range(n):
            mapping[op.src_start + k] = op.dest_start + (k * m) // n
    mapping[n_src] = n_dest
    return np.maximum.accumulate(mapping)


def _word_boundaries(gt: list) -> np.ndarray:
    edges = {0, len(gt)}
    edges.update(i for i in range(1, len(gt)) if gt[i - 1] == " ")
    return np.array(sorted(edges), dtype=np.int64)


def _refine_cuts(raw_cuts, gt, hypotheses, boundaries, newline_at, newline_bonus):
    """Put each boundary where it best explains the two line images that share it.

    Always snapping the cut to the nearest space is wrong: printers break words across
    lines and the transcript has already reflowed them back together, so for a broken
    word the correct cut falls INSIDE a word. Snapping hands the whole word to the
    upper line and steals the lower line's opening, corrupting both labels. Measured on
    this corpus 17% of raw cuts land inside a word and every one is within 5 graphemes
    of a boundary, so a fixed snap window always fires and cannot tell a real split
    from alignment jitter.

    Distance cannot separate the two cases; evidence can. Try the raw position and the
    boundaries either side, keep the lowest combined CER. Transcript line breaks get a
    small bonus, since where the source kept real line structure it beats the
    recogniser's opinion.
    """
    cuts = list(raw_cuts)
    for i in range(1, len(cuts) - 1):
        lo, hi = cuts[i - 1], cuts[i + 1]
        if lo >= hi:
            cuts[i] = lo
            continue
        idx = int(np.searchsorted(boundaries, cuts[i]))
        candidates = {min(max(cuts[i], lo), hi)}
        for j in (idx - 1, idx):
            if 0 <= j < len(boundaries) and lo <= int(boundaries[j]) <= hi:
                candidates.add(int(boundaries[j]))

        best, best_score = None, None
        for cand in candidates:
            score = (cer("".join(gt[lo:cand]).strip(), hypotheses[i - 1])
                     + cer("".join(gt[cand:hi]).strip(), hypotheses[i]))
            if cand in newline_at:
                score -= newline_bonus
            if best_score is None or score < best_score:
                best, best_score = cand, score
        cuts[i] = best
    return cuts


@dataclass
class LineAlignment:
    index: int
    text: str
    hypothesis: str
    cer: float
    n_graphemes: int
    accepted: bool = False
    reject_reason: str = None
    digits_converted: bool = False
    starts_mid_word: bool = False
    ends_mid_word: bool = False


@dataclass
class PageAlignment:
    lines: list = field(default_factory=list)
    page_cer: float = 1.0
    page_accepted: bool = False
    page_reject_reason: str = None
    # Fraction of detected lines explainable at a FIXED loose cut -- the page guard's
    # input, and the number worth storing. Unlike the strict yield it does not move
    # when max_line_cer moves, so it stays comparable across runs.
    broad_yield: float = 0.0

    @property
    def n_lines(self):
        return len(self.lines)

    @property
    def n_accepted(self):
        return sum(l.accepted for l in self.lines)

    @property
    def strict_yield(self):
        """Reporting only. Read after a rejected page has had its lines flipped, so it
        is always 0 there -- never store it as the page's yield."""
        return self.n_accepted / self.n_lines if self.lines else 0.0


@dataclass
class AcceptPolicy:
    """What counts as a usable line. See the module docstring for how these are used.

    max_line_cer        the main filter; a corroboration test, not an error rate on the
                        label. 0.10 because the recogniser reads this corpus at ~0.19
                        CER, so this demands closer agreement than it manages on
                        average. Measured yields: 0.05 keeps 28% of detected lines,
                        0.10 keeps 40%, 0.15 keeps 50%. Re-measure after a retrain.
    edge_max_line_cer   tighter bar for the first and last text lines of a page.
                        Proofreaders adjust page boundaries for continuity, completing
                        a split word on one page and dropping it from the other, so
                        those two lines are the least likely to match what is printed.
    min_graphemes       a 2-akshara span passes a CER test on one correct akshara.
    max_grapheme_ratio  a span several times the page median means the aligner dumped
                        an unmatched region onto this line.
    convert_digits      transcribers type 1893 where the page prints ౧౮౯౩; rewriting
                        keeps the label matching the image. Both digit sets are in the
                        vocab, so either is learnable.
    require_digit_evidence
                        only convert when the reading actually contains a Telugu
                        numeral, not merely when it lacks an Arabic one. Without this
                        the rewrite fires overwhelmingly on folio and verse numbers
                        that leaked into a span from elsewhere on the page -- measured
                        over 40 pages, every conversion under the looser rule sat at
                        the label's leading or trailing edge and none in the interior.
    reject_script_mismatch
                        mojibake filter. Some transcripts were pasted from documents
                        typed in a legacy 8-bit Telugu font, so Latin passages arrive
                        as Telugu-shaped nonsense. One-directional: Latin crop, Telugu
                        transcript with no Latin at all.
    min_page_yield / page_yield_cer / max_page_cer
                        page guards. Where most lines fail, the few that passed are as
                        likely lucky as correct, so the whole page is dropped. The
                        yield read here is measured at page_yield_cer, a FIXED loose
                        cut, NOT at max_line_cer -- tying it to the strict cut made the
                        knobs fight, discarding 37 of 163 pages that had read perfectly
                        well (page CER <= 0.25) purely because a tighter line threshold
                        had pushed their yield under the guard.
    """

    max_line_cer: float = 0.10
    edge_max_line_cer: float = 0.05
    min_graphemes: int = 5
    max_grapheme_ratio: float = 3.0
    convert_digits: bool = True
    require_digit_evidence: bool = True
    reject_script_mismatch: bool = True
    newline_bonus: float = 0.02
    min_page_yield: float = 0.35
    page_yield_cer: float = 0.40
    max_page_cer: float = 0.5


def _script_mismatch(gt: str, hyp: str) -> bool:
    if not gt or not hyp:
        return False
    if english_frac(hyp) < 0.6 or english_frac(gt) > 0.1:
        return False
    return any("ఀ" <= ch <= "౿" for ch in gt)


def _convert_digits(span: str, hyp: str, require_evidence: bool = True):
    if not any(ch in ARABIC_DIGITS for ch in span):
        return span, False
    if any(ch in ARABIC_DIGITS for ch in hyp):
        return span, False
    if require_evidence and not any(ch in TELUGU_DIGITS for ch in hyp):
        return span, False
    return span.translate(_ARABIC_TO_TELUGU), True


def align_page(hypotheses, page_text, policy):
    """Assign each line hypothesis a span of `page_text`, and judge the result.

    The cuts partition the page text exactly: every grapheme lands in one line and no
    line overlaps another. That is deliberate -- when the page holds something the
    transcript omits (running header, folio number, plate caption) the orphaned text
    has to go somewhere, and gluing it onto a neighbour blows up that neighbour's CER
    where the filter can catch it. Dropping unmatched text silently would instead leave
    a clean-looking line whose label is missing a chunk.
    """
    result = PageAlignment()
    gt, newline_at = _prepare_ground_truth(page_text)
    if not gt:
        result.page_reject_reason = "empty_page_text"
        return result

    clean = [normalize(h or "") for h in hypotheses]
    joined, starts = [], []
    for i, text in enumerate(clean):
        if i:
            joined.append(" ")      # mirrors how the transcript joins two printed lines
        starts.append(len(joined))
        joined.extend(graphemes(text))
    starts.append(len(joined))

    result.page_cer = cer("".join(gt), "".join(joined))
    mapping = _index_map(joined, gt)
    boundaries = _word_boundaries(gt)

    raw_cuts = [int(mapping[s]) for s in starts]
    raw_cuts[0], raw_cuts[-1] = 0, len(gt)
    for i in range(1, len(raw_cuts)):
        raw_cuts[i] = max(raw_cuts[i], raw_cuts[i - 1])
    cuts = _refine_cuts(raw_cuts, gt, clean, boundaries, newline_at, policy.newline_bonus)

    spans = ["".join(gt[a:b]).strip() for a, b in zip(cuts, cuts[1:])]
    if policy.convert_digits:
        pairs = [_convert_digits(s, h, policy.require_digit_evidence)
                 for s, h in zip(spans, clean)]
        spans = [p[0] for p in pairs]
        digit_flags = [p[1] for p in pairs]
    else:
        digit_flags = [False] * len(spans)

    lengths = [len(graphemes(s)) for s in spans]
    median_len = float(np.median([n for n in lengths if n])) if any(lengths) else 0.0

    # First and last lines CARRYING TEXT, which is not line 0 and line N-1: a running
    # header or folio number usually takes one of those boxes and gets an empty span.
    with_text = [i for i, n in enumerate(lengths) if n]
    edges = {with_text[0], with_text[-1]} if with_text else set()
    boundary_set = set(boundaries.tolist())

    for i, (span, hyp, n) in enumerate(zip(spans, clean, lengths)):
        line = LineAlignment(index=i, text=span, hypothesis=hyp, cer=cer(span, hyp),
                             n_graphemes=n, digits_converted=digit_flags[i],
                             starts_mid_word=cuts[i] not in boundary_set,
                             ends_mid_word=cuts[i + 1] not in boundary_set)
        limit = policy.edge_max_line_cer if i in edges else policy.max_line_cer
        if n == 0:
            line.reject_reason = "empty_span"
        elif n < policy.min_graphemes:
            line.reject_reason = "too_short"
        elif median_len and n > policy.max_grapheme_ratio * median_len:
            line.reject_reason = "span_too_long"
        elif policy.reject_script_mismatch and _script_mismatch(span, hyp):
            line.reject_reason = "script_mismatch"
        elif line.cer > limit:
            line.reject_reason = "edge_cer" if i in edges else "cer"
        else:
            line.accepted = True
        result.lines.append(line)

    explainable = sum(1 for line in result.lines
                      if line.n_graphemes and line.cer <= policy.page_yield_cer)
    result.broad_yield = explainable / len(result.lines) if result.lines else 0.0

    if result.page_cer > policy.max_page_cer:
        result.page_reject_reason = "page_cer"
    elif result.broad_yield < policy.min_page_yield:
        result.page_reject_reason = "page_yield"
    else:
        result.page_accepted = True

    if not result.page_accepted:
        for line in result.lines:
            if line.accepted:
                line.accepted = False
                line.reject_reason = result.page_reject_reason
    return result


# ======================================================================================
# 3. Page preparation and line segmentation
#
# Tesseract layout analysis, chosen by measurement over 163 pages from 138 books: it
# ties PP-OCR detection on reading accuracy but pulls ahead as the CER threshold
# tightens (+18% lines at a 0.10 cut, +29% at 0.05), and an ink-projection segmenter
# was 3x faster but over-segmented badly.
# ======================================================================================
@dataclass
class PreparedPage:
    gray: np.ndarray      # crops come from here; the models were trained on grayscale
    binary: np.ndarray    # only for measuring how much ink is inside a box


def prepare_page(img, deskew=True, max_skew=8.0) -> PreparedPage:
    """Grayscale, deskew, binarise.

    The skew estimate is clamped: `determine_skew` on a page whose dominant straight
    lines are a decorative rule or a plate border occasionally returns ~45 degrees, and
    rotating by that turns a readable page into confetti with nothing downstream able
    to recover. Adaptive threshold rather than Otsu because these scans have page-scale
    illumination gradients that a global threshold blows out on one side.
    """
    arr = np.asarray(img.convert("L") if isinstance(img, Image.Image) else img)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = np.ascontiguousarray(arr, dtype=np.uint8)

    if deskew:
        try:
            from deskew import determine_skew

            est = determine_skew(gray)
            angle = float(est) if est is not None and abs(est) <= max_skew else 0.0
        except Exception:
            angle = 0.0
        if angle:
            h, w = gray.shape
            rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            gray = cv2.warpAffine(gray, rot, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 15) > 0
    return PreparedPage(gray=gray, binary=binary)


@dataclass
class BoxFilter:
    """max_ink_frac is the non-obvious one: a box more than ~55% ink is a photograph,
    a plate or a solid rule, not text. These books are full of them, and without this
    they reach the recogniser, which invents a line of text the aligner then has to
    charge against the real ground truth."""

    min_height: int = 12
    min_width: int = 40
    max_height_frac: float = 0.25
    max_width_frac: float = 1.0
    min_ink_frac: float = 0.005
    max_ink_frac: float = 0.55


def segment_lines(page: PreparedPage, box_filter: BoxFilter, lang="tel+eng",
                  psm=3, timeout=60):
    """Tesseract's level-4 (textline) boxes, in reading order.

    Only the geometry is used; whatever text tesseract reads on the way is discarded,
    because the recogniser that actually reads the crops is a different model.
    """
    import pytesseract

    try:
        data = pytesseract.image_to_data(
            Image.fromarray(page.gray), output_type=pytesseract.Output.DATAFRAME,
            config=f"--oem 3 --psm {psm}", lang=lang, timeout=timeout)
    except Exception:
        return []
    if data is None or len(data) == 0:
        return []

    h, w = page.binary.shape
    boxes = []
    for _, row in data[data["level"] == 4].iterrows():
        x0, y0 = int(row["left"]), int(row["top"])
        x1, y1 = x0 + int(row["width"]), y0 + int(row["height"])
        bh, bw = y1 - y0, x1 - x0
        if bh < box_filter.min_height or bw < box_filter.min_width:
            continue
        if bh > box_filter.max_height_frac * h or bw > box_filter.max_width_frac * w:
            continue
        patch = page.binary[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        if not (box_filter.min_ink_frac <= float(patch.mean()) <= box_filter.max_ink_frac):
            continue
        boxes.append((x0, y0, x1, y1))
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def crop_line(page: PreparedPage, box, pad=3) -> Image.Image:
    x0, y0, x1, y1 = box
    h, w = page.gray.shape
    return Image.fromarray(page.gray[max(0, y0 - pad):min(h, y1 + pad),
                                     max(0, x0 - pad):min(w, x1 + pad)])


# ======================================================================================
# 4. Line preprocessing -- the stored form AND the model input
#
# Done once per crop and used for both, so the bytes in the dataset are exactly what
# the recogniser saw when it judged the label. Mirrors ImagePreprocessor in
# src/image_encoder/utils.py, including the over-wide branch: past max_width the scale
# is driven by width instead and the height shortfall is padded, rather than squashing
# the glyphs horizontally. Getting this wrong is silent -- the model just reads badly.
# ======================================================================================
def preprocess_line(img: Image.Image, height=64, max_width=2048, downsample=8) -> np.ndarray:
    """Grayscale uint8, height `height`, width padded to a multiple of `downsample`."""
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


def encode_jpeg(arr: np.ndarray, quality=90) -> bytes:
    """JPEG-encode a preprocessed crop. Measured ~13 KB/line at height 64, quality 90,
    against ~26 KB as PNG -- roughly 12 GB versus 24 GB over the whole corpus. The page
    scans are JPEG to begin with, so the crops already carry those artefacts."""
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue()


# ======================================================================================
# 5. The CTC recogniser (inlined from src/image_encoder/train_ctc_encoder_2048.py)
# ======================================================================================
@dataclass
class CTCEncoderConfig:
    image_height: int = 64
    max_image_width: int = 2048
    downsample: int = 8
    base_frames: int = 128
    max_frames: int = 256
    stem_channels: tuple = (32, 64, 128, 256, 320, 384)
    num_groups: int = 32
    embed_dim: int = 384
    num_layers: int = 10
    num_heads: int = 8
    mlp_dim: int = 1536
    dropout: float = 0.05
    drop_path_rate: float = 0.1
    vocab_size: int = 2048


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, num_groups):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding)
        groups = num_groups if out_channels % num_groups == 0 else 1
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ConvStem(nn.Module):
    """Six-block convolutional tokenizer: (B,1,64,W) -> (B, T=W/8, 384)."""

    def __init__(self, cfg):
        super().__init__()
        c = cfg.stem_channels
        specs = [((3, 3), (2, 2), (1, 1)), ((3, 3), (2, 2), (1, 1)),
                 ((3, 3), (2, 1), (1, 1)), ((3, 3), (2, 2), (1, 1)),
                 ((3, 3), (2, 1), (1, 1)), ((2, 1), (2, 1), (0, 0))]
        in_ch, blocks = 1, []
        for out_ch, (k, s, p) in zip(c, specs):
            blocks.append(ConvBlock(in_ch, out_ch, k, s, p, cfg.num_groups))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        B, D, H, T = x.shape
        assert H == 1, f"conv stem did not collapse height to 1 (got {H})"
        return x.squeeze(2).transpose(1, 2)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return x  # inference only


class TransformerBlock(nn.Module):
    def __init__(self, cfg, drop_path):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(cfg.embed_dim)
        self.attention = nn.MultiheadAttention(cfg.embed_dim, cfg.num_heads,
                                               dropout=cfg.dropout, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(cfg.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.mlp_dim), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_dim, cfg.embed_dim), nn.Dropout(cfg.dropout))
        self.drop_path = DropPath(drop_path)

    def forward(self, x, key_padding_mask=None):
        norm_x = self.layer_norm1(x)
        attn_out, _ = self.attention(norm_x, norm_x, norm_x,
                                     key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop_path(attn_out)
        return x + self.drop_path(self.mlp(self.layer_norm2(x)))


class ImageEncoderCTC(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.blank_id = cfg.vocab_size
        self.num_classes = cfg.vocab_size + 1
        self.stem = ConvStem(cfg)
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.max_frames, cfg.embed_dim))
        self.dropout = nn.Dropout(cfg.dropout)
        dpr = torch.linspace(0.0, cfg.drop_path_rate, cfg.num_layers).tolist()
        self.blocks = nn.ModuleList([TransformerBlock(cfg, dpr[i])
                                     for i in range(cfg.num_layers)])
        self.layer_norm = nn.LayerNorm(cfg.embed_dim)
        self.ctc_head = nn.Linear(cfg.embed_dim, self.num_classes)

    @torch.no_grad()
    def predict_ids(self, images, input_lengths):
        feats = self.stem(images)
        B, T, D = feats.shape
        assert T <= self.cfg.max_frames, f"T={T} exceeds max_frames={self.cfg.max_frames}"
        feats = feats + self.pos_embed[:, :T, :]
        frame_idx = torch.arange(T, device=feats.device).unsqueeze(0)
        key_padding_mask = frame_idx >= input_lengths.unsqueeze(1)
        for block in self.blocks:
            feats = block(feats, key_padding_mask)
        logits = self.ctc_head(self.layer_norm(feats))
        pred = logits.float().log_softmax(dim=-1).argmax(dim=-1)
        return pred.masked_fill(key_padding_mask, self.blank_id)


def load_ctc_checkpoint(model, ckpt_path):
    """Load a checkpoint, merging a split position table if it has one.

    Two shapes exist in this repo and nothing in the path says which is which: stage-2
    stores `pos_embed` (1,128,D) plus `pos_embed_ext` (1,128,D), stage-3 stores them
    already merged as (1,256,D). Both are 2048px-wide models. Concatenating the split
    pair reproduces the trained forward pass exactly, so no weights change.

    The load is strict on purpose. A tolerant load against the wrong architecture
    leaves weights at their random init, the model emits fluent-looking garbage, and
    that garbage becomes the CER that decides which labels to keep.
    """
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    for prefix in ("module.", "model."):
        if any(k.startswith(prefix) for k in state) and "pos_embed" not in state:
            state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            break

    ext = state.pop("pos_embed_ext", None)
    if ext is not None:
        state["pos_embed"] = torch.cat([state["pos_embed"], ext], dim=1)
    state = {k: v for k, v in state.items() if not k.startswith("ctc_loss")}

    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [k for k in unexpected if "ctc_loss" not in k]
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint does not match the model: {len(missing)} missing, "
            f"{len(unexpected)} unexpected.\n  missing: {list(missing)[:5]}\n"
            f"  unexpected: {list(unexpected)[:5]}\nRefusing to run on partial weights.")
    return model


class Recogniser:
    """Batched greedy CTC decoding, batched under a PIXEL budget rather than a count.

    With a fixed batch size the widest crop sets the padded width for the whole batch
    and most of the compute goes into padding, so crops are sorted by width and packed
    until the padded batch would exceed max_pixels.
    """

    def __init__(self, checkpoint, vocab_file, device=None,
                 max_pixels_per_batch=64 * 2048 * 24):
        self.cfg = CTCEncoderConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_ctc_checkpoint(ImageEncoderCTC(self.cfg), checkpoint)
        self.model.eval().to(self.device)
        self.blank_id = self.model.blank_id
        self.max_pixels = max_pixels_per_batch

        with open(vocab_file, encoding="utf-8") as fh:
            vocab = json.load(fh)
        if isinstance(vocab, dict):
            self.id_to_token = {int(i): t for t, i in vocab.items()}
        else:
            self.id_to_token = {i: t for i, t in enumerate(vocab)}

    def _decode(self, row):
        """Greedy CTC collapse: drop repeats, then drop blanks."""
        out, prev = [], None
        for t in row:
            t = int(t)
            if t != prev and t != self.blank_id and t >= 0:
                out.append(self.id_to_token.get(t, ""))
            prev = t
        return "".join(out)

    def read(self, arrays):
        """arrays: list of preprocessed uint8 (64, W) crops -> list of strings."""
        if not arrays:
            return []
        texts = [None] * len(arrays)
        order = sorted(range(len(arrays)), key=lambda i: arrays[i].shape[1])

        batch, widest = [], 0
        def flush():
            if not batch:
                return
            imgs = torch.full((len(batch), 1, self.cfg.image_height, widest), 1.0,
                              dtype=torch.float32)
            lengths = []
            for slot, idx in enumerate(batch):
                arr = arrays[idx]
                x = (arr.astype(np.float32) / 255.0 - 0.5) / 0.5
                imgs[slot, 0, :, :arr.shape[1]] = torch.from_numpy(x)
                lengths.append(arr.shape[1] // self.cfg.downsample)
            pred = self.model.predict_ids(
                imgs.to(self.device),
                torch.tensor(lengths, dtype=torch.long, device=self.device))
            for slot, idx in enumerate(batch):
                texts[idx] = self._decode(pred[slot].tolist())

        for idx in order:
            w = max(widest, arrays[idx].shape[1])
            if batch and (len(batch) + 1) * self.cfg.image_height * w > self.max_pixels:
                flush()
                batch, widest = [], 0
                w = arrays[idx].shape[1]
            batch.append(idx)
            widest = w
        flush()
        return texts


# ======================================================================================
# 6. Scraping
# ======================================================================================
API = "https://te.wikisource.org/w/api.php"
HEADERS = {"User-Agent": "TeluguOCR-dataset/0.1 (your@email.com)"}
IMAGE_PROPS = "filename|size|fullsize|url|responsiveimages"
_THUMB_WIDTH_RE = re.compile(r"-(\d+)px-")
_local = threading.local()


def session():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers.update(HEADERS)
    return _local.s


def api_get(params, retries=3, delay=1.0):
    params = {**params, "format": "json", "formatversion": 2}
    for attempt in range(retries):
        try:
            r = session().get(API, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            tqdm.write(f"  api error ({e}), retrying...")
            time.sleep(delay * (attempt + 1))
    return None


def title_from_url(url):
    return unquote(urlparse(url).path.split("/wiki/", 1)[1]).replace("_", " ")


def image_url_candidates(info, target_width=1280):
    """Scan URLs for one page, best first.

    The base URL must come from the API, not be assembled from the file name: these
    books are hosted per-wiki (upload.wikimedia.org/wikisource/te/...) rather than on
    Commons, and the width in a thumbnail path has to be one the server will serve.
    Hand-built Commons URLs 404'd on roughly two thirds of a sample.

    THE WIDTH IS WHAT MATTERS FOR OCR. What the API offers is the wiki's default
    thumbnail size, and for a large share of this category that is 500px -- ~20px line
    heights, which upscale to the encoder's 64px as mush. In a 163-page sample 47% came
    back at 500px and their page CER was far worse than the 1280px pages'. MediaWiki
    renders PDF and DjVu on demand at whatever width the path asks for, so the width
    token is rewritten to `target_width` and tried first; asking above the source's
    native resolution answers HTTP 400, which is why the API's own URL is kept as the
    fallback.
    """
    seen, scored = set(), []
    for url in [info.get("thumbnail"), info.get("fullsize"),
                *(info.get("responsiveimages") or {}).values()]:
        if not url:
            continue
        if url.startswith("//"):
            url = "https:" + url
        url = url.split("?", 1)[0]
        if url in seen:
            continue
        seen.add(url)
        m = _THUMB_WIDTH_RE.search(url)
        scored.append((int(m.group(1)) if m else 0, url))

    if not scored:
        return []
    big_enough = sorted(s for s in scored if s[0] >= target_width)
    if big_enough:
        return [big_enough[0][1]]
    _, best = max(scored)
    upscaled = _THUMB_WIDTH_RE.sub(f"-{target_width}px-", best, count=1)
    return [upscaled, best] if upscaled != best else [best]


def iter_category_pages(category, limit=None):
    """Yield {title, filename, page_no, image_urls} for every Page: in the category.

    `generator=categorymembers` makes the members the input to a prop query, so one
    call returns 50 members together with their scan info. gcmlimit is 50 rather than
    500 because prop queries cap there. Pages with no scan behind them, and titles
    without a /<number> suffix, are skipped -- 208 of the category's 53,714 members.
    """
    params = {"action": "query", "generator": "categorymembers", "gcmtitle": category,
              "gcmnamespace": 104, "gcmlimit": 50, "prop": "imageforpage",
              "prppifpprop": IMAGE_PROPS}
    seen = 0
    while True:
        data = api_get(params)
        if data is None:
            return
        for page in data.get("query", {}).get("pages", []):
            info = page.get("imagesforpage") or {}
            if not info.get("filename"):
                continue
            match = re.search(r"/(\d+)$", page["title"])
            if not match:
                continue
            yield {"title": page["title"], "filename": info["filename"],
                   "page_no": int(match.group(1)),
                   "image_urls": image_url_candidates(info)}
            seen += 1
            if limit and seen >= limit:
                return
        if "continue" not in data:
            return
        params.update(data["continue"])


def fetch_text(title):
    """Rendered page text, without header/footer/notes.

    Rendered rather than raw wikitext on purpose: templates like {{Center|...}} expand
    to real <br> breaks, so the pages that ARE line-per-line in the source come out
    line-per-line here.
    """
    data = api_get({"action": "parse", "page": title, "prop": "text",
                    "disablelimitreport": 1, "disableeditsection": 1})
    if data is None or "parse" not in data:
        return ""
    soup = BeautifulSoup(data["parse"]["text"], "html.parser")
    body = soup.select_one("div.pagetext")
    if body is None:
        return ""
    for tag in body.select("sup.reference, div.reflist, ol.references"):
        tag.decompose()
    return "\n".join(ln for ln in (x.strip() for x in body.get_text("\n").split("\n")) if ln)


def fetch_image_bytes(urls, retries=4, timeout=90):
    """Download a scan into memory, trying each candidate URL in turn.

    Throttling and a missing rendering need opposite responses. 429/503 is waited out
    with exponential backoff honouring Retry-After -- a plain retry loop treats it as
    transient and hammers straight back in, which is how an early run stalled at 13
    files with every worker spinning. 400 (width above native) and 404 will not become
    correct on a retry, so they move straight to the next candidate.
    """
    for url in urls or []:
        delay = 2.0
        for _ in range(retries):
            try:
                r = session().get(url, timeout=timeout)
                if r.status_code in (429, 503):
                    wait = float(r.headers.get("Retry-After") or delay)
                    tqdm.write(f"  throttled ({r.status_code}), waiting {wait:.0f}s")
                    time.sleep(wait)
                    delay = min(delay * 2, 60.0)
                    continue
                if r.status_code in (400, 404):
                    break
                r.raise_for_status()
                return r.content
            except Exception as e:
                tqdm.write(f"  image error ({e}), retrying...")
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
    return None


def slugify(page):
    stem = re.sub(r"\.(pdf|djvu)$", "", page["filename"], flags=re.I)
    return f"{re.sub(r'[^A-Za-z0-9]+', '_', stem).strip('_')}_p{page['page_no']:04d}"


# ======================================================================================
# 7. Pipeline
#
# PACING. Fetch-and-segment runs in a thread pool one mini-batch ahead of the
# recogniser, so downloads are naturally paced by how fast pages can be processed
# rather than by a sleep. A mini-batch of 32 pages fetches in a few seconds and then
# takes a minute or so to segment and read, which averages under one request per
# second even with 8 threads. pytesseract shells out to the tesseract binary, so it
# releases the GIL and the pool parallelises segmentation as well as I/O.
# ======================================================================================
def fetch_and_segment(page, box_filter, deskew, crop_pad, store_height,
                      max_width, downsample, fetch_gate):
    """Download one page and reduce it to preprocessed line crops. Runs in a worker.

    The download is gated separately from the worker count. Workers are sized for CPU
    work; letting all of them fetch at once fires far more concurrent requests than
    intended and is counterproductive -- measured on cold pages, 2 concurrent fetches
    gave 0.90 img/s with no throttling, 4 gave 0.65 with eight 429s, and 8 gave 0.58
    with nineteen. Past two, Wikimedia throttles and the backoff waits swamp the gain.
    """
    try:
        with fetch_gate:
            raw = fetch_image_bytes(page["image_urls"])
            if raw is None:
                return None
            text = fetch_text(page["title"])
        if not text.strip():
            return None
        with Image.open(io.BytesIO(raw)) as img:
            prepared = prepare_page(img, deskew=deskew)
        boxes = segment_lines(prepared, box_filter)
        arrays = [preprocess_line(crop_line(prepared, b, crop_pad), store_height,
                                  max_width, downsample) for b in boxes]
        return {"page": page, "slug": slugify(page), "text": text, "arrays": arrays}
    except Exception as exc:
        tqdm.write(f"  [fetch] {page['title'][:50]}: {type(exc).__name__}: {exc}")
        return None


def process_shard(pages, recogniser, policy, cfg):
    """Fetch, read, align and encode one shard. Returns the dataset rows."""
    rows, stats = [], {}
    fetch_gate = threading.Semaphore(cfg["fetch_concurrency"])

    def bump(key, n=1):
        stats[key] = stats.get(key, 0) + n

    with ThreadPoolExecutor(max_workers=cfg["num_workers"]) as pool:
        bar = tqdm(total=len(pages), desc="pages", unit="pg", smoothing=0.05, leave=False)
        for start in range(0, len(pages), cfg["batch_pages"]):
            chunk = pages[start:start + cfg["batch_pages"]]
            for prepared in pool.map(
                    lambda p: fetch_and_segment(
                        p, cfg["box_filter"], cfg["deskew"], cfg["crop_pad"],
                        cfg["store_height"], cfg["max_width"], cfg["downsample"],
                        fetch_gate),
                    chunk):
                bar.update(1)
                if prepared is None:
                    bump("pages_failed")
                    continue

                arrays = prepared["arrays"]
                hypotheses = recogniser.read(arrays)
                result = align_page(hypotheses, prepared["text"], policy)
                bump("pages")
                bump("pages_accepted", int(result.page_accepted))
                bump("lines_detected", len(result.lines))

                source_file = re.sub(r"\.(pdf|djvu)$", "", prepared["page"]["filename"],
                                     flags=re.I)
                for line in result.lines:
                    if not line.text:
                        bump("dropped_empty_span")
                        continue
                    if line.cer > cfg["base_max_cer"]:
                        bump("dropped_above_base_cer")
                        continue
                    jpeg = encode_jpeg(arrays[line.index], cfg["jpeg_quality"])
                    rows.append({
                        "line_image": {"bytes": jpeg,
                                       "path": f"{prepared['slug']}_l{line.index:03d}.jpg"},
                        "text": line.text,
                        "line_cer": round(line.cer, 4),
                        "page_cer": round(result.page_cer, 4),
                        "page_yield": round(result.broad_yield, 4),
                        "accepted": line.accepted,
                        "reject_reason": line.reject_reason or "",
                        "n_graphemes": line.n_graphemes,
                        "digits_converted": line.digits_converted,
                        "starts_mid_word": line.starts_mid_word,
                        "ends_mid_word": line.ends_mid_word,
                        "slug": prepared["slug"],
                        "page_no": prepared["page"]["page_no"],
                        "source_file": source_file,
                        "line_no": line.index,
                        "checkpoint": cfg["checkpoint_name"],
                    })
                    bump("lines_stored")
                    bump("lines_accepted", int(line.accepted))
        bar.close()
    return rows, stats


def push_shard(rows, repo, config_name, private=False, max_shard_size="500MB"):
    from datasets import Dataset, Features, Value
    from datasets import Image as HFImage

    features = Features({
        "line_image": HFImage(), "text": Value("string"),
        "line_cer": Value("float32"), "page_cer": Value("float32"),
        "page_yield": Value("float32"), "accepted": Value("bool"),
        "reject_reason": Value("string"), "n_graphemes": Value("int32"),
        "digits_converted": Value("bool"), "starts_mid_word": Value("bool"),
        "ends_mid_word": Value("bool"), "slug": Value("string"),
        "page_no": Value("int32"), "source_file": Value("string"),
        "line_no": Value("int32"), "checkpoint": Value("string"),
    })
    ds = Dataset.from_dict({k: [r[k] for r in rows] for k in rows[0]}, features=features)
    ds.push_to_hub(repo, config_name=config_name, split="train",
                   private=private, max_shard_size=max_shard_size)


def hf_login():
    """HF_TOKEN from the environment, a .env file, or a Kaggle secret."""
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
        raise SystemExit("no HF_TOKEN found (env, .env, or Kaggle secret) -- uploads "
                         "would fail after hours of work, so stopping now")
    from huggingface_hub import login

    login(token=token)


def check_dependencies():
    """Fail immediately rather than hours in, with the exact commands to fix it."""
    problems = []
    try:
        import pytesseract

        langs = pytesseract.get_languages(config="")
        if "tel" not in langs:
            problems.append("tesseract has no 'tel' language data  ->  "
                            "!apt-get install -y tesseract-ocr-tel")
    except Exception as exc:
        problems.append(f"pytesseract/tesseract unavailable ({exc})  ->  "
                        "!apt-get install -y tesseract-ocr tesseract-ocr-tel && "
                        "pip install pytesseract")
    try:
        import deskew  # noqa: F401
    except ImportError:
        print("[warn] deskew not installed; skew correction will be skipped "
              "(pip install deskew)")
    if problems:
        raise SystemExit("setup incomplete:\n  " + "\n  ".join(problems))


def chunked(iterable, size):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def run(cfg, policy):
    check_dependencies()
    hf_login()

    from datasets import get_dataset_config_names

    try:
        existing = set(get_dataset_config_names(cfg["repo"]))
    except Exception:
        existing = set()
    print(f"repo {cfg['repo']} | {len(existing)} shards already uploaded")

    recogniser = Recogniser(cfg["checkpoint"], cfg["vocab"])
    print(f"[model] {recogniser.device}, vocab {len(recogniser.id_to_token)}")

    category = title_from_url(cfg["start_url"])
    deadline = time.time() + cfg["max_runtime_hours"] * 3600
    totals = {}

    for shard_idx, chunk in enumerate(
            chunked(iter_category_pages(category, cfg["limit_pages"]), cfg["shard_size"])):
        name = f"train_{shard_idx:04d}"
        if name in existing:
            continue
        if time.time() > deadline:
            print(f"\n[stop] runtime budget reached; next run resumes at {name}")
            break

        print(f"\n[shard] {name}: {len(chunk)} pages")
        rows, stats = process_shard(chunk, recogniser, policy, cfg)
        for k, v in stats.items():
            totals[k] = totals.get(k, 0) + v
        if not rows:
            print(f"[shard] {name}: nothing usable, skipping upload")
            continue

        mb = sum(len(r["line_image"]["bytes"]) for r in rows) / 1e6
        print(f"[shard] {name}: {len(rows)} lines "
              f"({stats.get('lines_accepted', 0)} strict), {mb:.0f} MB -> uploading")
        push_shard(rows, cfg["repo"], name, private=cfg["private"])
        del rows

    print("\n" + "=" * 68)
    for key in ("pages", "pages_failed", "pages_accepted", "lines_detected",
                "lines_stored", "lines_accepted", "dropped_empty_span",
                "dropped_above_base_cer"):
        print(f"  {key:<26}{totals.get(key, 0):>10,}")
    print("=" * 68)


# ======================================================================================
# Inline configuration
# ======================================================================================
if __name__ == "__main__":
    CONFIG = {
        # ---- source ----
        "start_url": "https://te.wikisource.org/wiki/%E0%B0%B5%E0%B0%B0%E0%B1%8D%E0%B0%97%E0%B0%82:%E0%B0%86%E0%B0%AE%E0%B1%8B%E0%B0%A6%E0%B0%BF%E0%B0%82%E0%B0%9A%E0%B0%AC%E0%B0%A1%E0%B1%8D%E0%B0%A1%E0%B0%B5%E0%B0%BF",
        "limit_pages": None,        # int for a smoke test, None for the whole category

        # ---- destination ----
        "repo": "harsha-desaraju/telugu-wikisource-lines",
        "private": False,

        # ---- model (attach as a Kaggle dataset, or point at local paths) ----
        "checkpoint": "/kaggle/input/telugu-ocr-ctc/final_model.pt",
        "vocab": "/kaggle/input/telugu-ocr-ctc/telugu-vocab.json",
        "checkpoint_name": "ctc_encoder_stage-3",

        # ---- sharding and runtime ----
        # A shard is the resume unit: a session that dies mid-shard loses that shard's
        # work, so keep it well under what fits in the time budget. Whole pipeline is
        # ~2s/page, so 1000 pages is roughly 35 minutes.
        "shard_size": 1000,
        "max_runtime_hours": 11.0,  # stop cleanly before Kaggle's 12h cap

        # ---- pacing ----
        # No sleep between requests: processing is slower than downloading, so the
        # processing rate is the rate limiter. See PACING above.
        "num_workers": 8,           # threads fetching + segmenting
        "fetch_concurrency": 2,     # concurrent HTTP requests -- the measured optimum
        "batch_pages": 32,          # pages fetched ahead of the recogniser

        # ---- crops ----
        "store_height": 64,         # the encoder's input height
        "max_width": 2048,          # the recogniser's cap; wider lines are squashed
        "downsample": 8,            # width padded to a multiple of this
        "jpeg_quality": 90,         # ~13 KB/line; ~12 GB over the whole corpus
        "crop_pad": 3,
        "deskew": True,
        "box_filter": BoxFilter(),

        # ---- what to keep ----
        # The storage floor, NOT the training threshold. Everything below it is written
        # out with its scores so strictness stays a query at training time.
        "base_max_cer": 0.40,
    }

    POLICY = AcceptPolicy(
        max_line_cer=0.10,
        edge_max_line_cer=0.05,
        min_graphemes=5,
        max_grapheme_ratio=3.0,
        convert_digits=True,
        require_digit_evidence=True,
        reject_script_mismatch=True,
        min_page_yield=0.35,
        page_yield_cer=0.40,
        max_page_cer=0.5,
    )

    run(CONFIG, POLICY)
