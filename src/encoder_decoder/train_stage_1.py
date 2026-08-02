from __future__ import annotations

# ============================================================================
# CPU-thread limiting for the augmentation DataLoader workers (PORTED from
# train_ctc_encoder.py). These env vars size the math-library thread pools at
# import time, so they MUST be set BEFORE numpy / torch / cv2 are imported.
# Under a multi-worker (and DDP) DataLoader, pinning each worker's OpenCV/OMP/BLAS
# to one thread stops them oversubscribing the cores and makes augmentation faster.
# Flip LIMIT_AUG_THREADS to False if you measure it hurting throughput.
# ============================================================================
import os

LIMIT_AUG_THREADS = True
if LIMIT_AUG_THREADS:
    for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[_thread_var] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutput

from transformers import Trainer, TrainingArguments
from transformers import TrainerCallback  # CER callback + grad-norm alert
from transformers.trainer_utils import get_last_checkpoint  # resume helper

from PIL import Image
from dataclasses import dataclass

from datasets import load_dataset, concatenate_datasets
from torchvision import transforms

import contextlib
import gc
import json
import math
import random
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import regex
from transformers import PreTrainedTokenizer


# Encoder model utils

class ImagePreprocessor:
    """
    Preprocesses the image before encoding the image
    1) Change the image to gray scale
    2) Resize the image
    3) Pad the image to the nearest multiple of patch size
    4) Normalize the image
    """

    def __init__(self, image_height: int, max_image_width: int, patch_size: int, augment_fn=None):
        assert image_height % patch_size == 0, "Image height should be a multiple of patch size"
        self.image_height = image_height
        self.max_image_width = max_image_width
        self.patch_size = patch_size
        # augment_fn: train split only; runs on the RGB crop before grayscale. The
        # augmentation pipeline is inlined verbatim from the CTC encoder training file.
        self.augment_fn = augment_fn
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def _transform(self, img: Image.Image) -> torch.Tensor:
        # Optional augmentation on the RGB crop (train split only).
        if self.augment_fn is not None:
            arr = self.augment_fn(np.array(img.convert("RGB")))
            img = Image.fromarray(np.asarray(arr, dtype=np.uint8))

        # Convert to GrayScale
        img = img.convert('L')

        return self.to_tensor(img)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self._transform(img)


# ============================================================================
# Image augmentation — INLINED from data_curation/text_line_images/image_augmentation.py
# ============================================================================
# CHANGED(2048): the sweep-style augmenter (one effect per image, chosen by a
# utility/time weighting) is replaced by the calibrated COMPOSED degradation pipeline
# from data_curation/text_line_images/image_augmentation.py — tuned so the augmented
# distribution CONTAINS the real PDF-crop distribution measured at h=64. It composes
# many effects per sample (bilevel -> ink -> paper/grime -> tone -> crop artefacts ->
# geometry -> noise -> resample -> compression -> auto-levels) with a two-sided
# legibility backstop. Entry point: make_augmenter(p_clean) -> augment(uint8 HxW|HxWx3)
# -> same shape. Fed the stored 64px-tall crop by ImagePreprocessor (the scale-dependent
# params — blur sigma, band widths, speckle size — are tuned for that height). The
# preview __main__ and its matplotlib/LADDERS helpers are intentionally not inlined.
import cv2  # noqa: E402
from augraphy import (  # noqa: E402
    BadPhotoCopy,
    BleedThrough,
    BrightnessTexturize,
    DirtyDrum,
    InkBleed,
    InkMottling,
    Jpeg,
    Letterpress,
    NoiseTexturize,
    SubtleNoise,
)

# Pin OpenCV to one thread (runtime companion to the env vars set at the top of the file).
# Forked DataLoader workers inherit this, so each augments on a single core instead of
# every worker's cv2 fighting for all of them. Gated by the same LIMIT_AUG_THREADS toggle.
if LIMIT_AUG_THREADS:
    cv2.setNumThreads(1)


def set_seed(seed):
    """Seed both PRNGs used here: `random` (which effects fire, and how hard) and
    `np.random` (the custom effects plus Augraphy's internals). Some Augraphy
    augmentations carry additional internal randomness, so identical draws may still
    differ run-to-run unless seeded here.

    Under a PyTorch DataLoader, torch seeds `random` per worker but NOT numpy, so pass a
    worker_init_fn if you see repeated grain patterns across workers:
        def _wi(wid): np.random.seed(torch.initial_seed() % 2**32)
    """
    random.seed(seed)
    np.random.seed(seed)


# ======================================================================================
# Custom effects
#
# These cover the failure modes visible in the real crops that Augraphy has no direct
# equivalent for. All operate on a single-channel uint8 image (the pipeline converts
# once at entry, see `degrade`).
# ======================================================================================

def resolution_loss(img, scale=0.5, down=cv2.INTER_AREA, up=cv2.INTER_LINEAR):
    """Round-trip through a lower resolution: the dominant synthetic/real gap.

    Treat this as a PROBABILITY knob, not a strength knob. Even scale=0.95 takes the
    Laplacian variance of a clean render from ~3900 to ~1400, because the round trip
    itself destroys the pixel-exact edges. `scale` only shapes the blurry tail; whether
    it fires at all is what controls the median.
    """
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(4, int(w * scale)), max(4, int(h * scale))),
                       interpolation=down)
    return cv2.resize(small, (w, h), interpolation=up)


def gaussian_blur(img, sigma=0.8):
    """Optical/scanner defocus. Beyond sigma 1.5 at h=64 the glyphs stop being legible."""
    k = max(3, int(2 * round(3 * sigma) + 1))
    return cv2.GaussianBlur(img, (k, k), sigma)


def motion_blur(img, degree=5, angle=0):
    """Scan drag / camera shake. `degree` is the streak length in pixels; `angle` rotates
    the streak (0 = horizontal, 90 = vertical)."""
    k = np.zeros((degree, degree), np.float32)
    k[degree // 2, :] = 1.0 / degree
    if angle:
        rot = cv2.getRotationMatrix2D((degree / 2 - 0.5, degree / 2 - 0.5), angle, 1.0)
        k = cv2.warpAffine(k, rot, (degree, degree))
        k /= max(1e-6, k.sum())
    return cv2.filter2D(img, -1, k)


def stroke_weight(img, amount=1.0):
    """Thicken (amount > 0) or thin (amount < 0) the ink.

    Real book scans run heavier than a clean render; faded pages run lighter. Fractional
    amounts are blended rather than snapped, because one whole pixel is a large change at
    h=64. Measured ceilings: +1.5 is the most that keeps vowel-sign counters open (+2.2
    closes them), and -1.8 shatters strokes into disconnected fragments.
    """
    if abs(amount) < 1e-3:
        return img
    n = int(np.ceil(abs(amount)))
    frac = abs(amount) / n
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    out = img
    for _ in range(n):
        # ink is dark, so thickening ink means eroding the grayscale image
        moved = cv2.erode(out, k) if amount > 0 else cv2.dilate(out, k)
        out = cv2.addWeighted(moved, frac, out, 1 - frac, 0)
    return out


_K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def _random_field(h, w, scale, octaves=2, decay=0.6):
    """Smooth multi-scale random field in [0,1], used as a spatial mask.

    Coarse octaves make whole groups of characters fade together; fine octaves bite into
    individual strokes. Both scales are present in the real crops, which is why this is
    multi-octave rather than a single blob size.
    """
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        # Floor at 4px: below that the mask is pixel noise rather than a blob, and it
        # dapples the INSIDE of a stroke into a dotted outline -- the same unusable look
        # that got LowInkRandomLines dropped.
        sc = max(4, int(scale / (2 ** o)))
        gh, gw = max(2, h // sc), max(2, w // sc)
        n = np.random.rand(gh, gw).astype(np.float32)
        out += amp * cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= decay
    return np.clip(out / tot, 0.0, 1.0)


def ink_dropout(img, coverage=0.25, scale=12, strength=1.0, octaves=2, ramp=0.8):
    """Erase ink in smooth random patches: some strokes lose a piece, their neighbours
    stay intact.

    This is the "partly missing character" look that dominates the real crops and that
    nothing else here reproduces. Global thinning (`stroke_weight` with a negative amount)
    thins EVERY character equally, and Letterpress fades the whole line; neither gives the
    within-line variation the real scans have -- in data/actual_samples the loop of a `ర`
    is broken open while the `ప` beside it is solid black, and one end of a line is heavy
    while the other is fragmented.

    Two spatial scales matter, hence `octaves`: coarse patches (scale ~30-60px) fade
    several characters together, fine ones (~5-15px) cut a single stroke.

    coverage : fraction of the crop the mask touches
    scale    : coarsest blob size in px; finer octaves are scale/2, scale/4, ...
    strength : 0..1; at 1.0 the deepest parts of the mask clear a thin stroke completely
    """
    h, w = img.shape[:2]
    f = _random_field(h, w, scale, octaves)
    thr = float(np.quantile(f, 1.0 - coverage))
    span = max(1e-6, float(f.max()) - thr)
    m = (np.clip((f - thr) / span, 0.0, 1.0) ** ramp) * strength

    # ink is dark, so DILATING the grayscale image is what eats the ink away
    t1 = cv2.dilate(img, _K3)
    t2 = cv2.dilate(t1, _K3)

    # piecewise ramp img -> t1 -> t2 as the mask deepens, so shallow patches only thin
    # the stroke while deep ones remove it outright
    lo = np.clip(m * 2.0, 0.0, 1.0)
    hi = np.clip((m - 0.5) * 2.0, 0.0, 1.0)
    stage = img.astype(np.float32) * (1 - lo) + t1.astype(np.float32) * lo
    out = stage * (1 - hi) + t2.astype(np.float32) * hi

    # dropout must only ever REMOVE ink, never add it
    return np.maximum(img, np.clip(out, 0, 255).astype(np.uint8))


def speckle(img, amount=0.004, blob_frac=0.15, dark=(30, 120), light_amount=0.0):
    """Dust and print-through specks, plus optional white ink drop-outs.

    Specks are single pixels, `blob_frac` of them grown to 2x2. Anything larger reads as
    snow rather than dust: a radius-2 circle is 5px on a 64px line, wider than a Telugu
    stroke, and it measured far outside the real samples' component density.
    """
    out = img.copy()
    h, w = out.shape[:2]
    n = int(amount * h * w)
    if n <= 0:
        return out
    ys = np.random.randint(0, h, n)
    xs = np.random.randint(0, w, n)
    vals = np.random.randint(dark[0], dark[1] + 1, n).astype(np.uint8)
    out[ys, xs] = np.minimum(out[ys, xs], vals)
    nb = int(n * blob_frac)
    if nb:
        for dy, dx in ((0, 1), (1, 0), (1, 1)):
            yy = np.clip(ys[:nb] + dy, 0, h - 1)
            xx = np.clip(xs[:nb] + dx, 0, w - 1)
            out[yy, xx] = np.minimum(out[yy, xx], vals[:nb])
    if light_amount > 0:
        m = int(light_amount * h * w)
        out[np.random.randint(0, h, m), np.random.randint(0, w, m)] = 255
    return out


def neighbour_line(img, top_frac=0.12, bottom_frac=0.12, alpha=0.75, shift=37):
    """Stamp faded fragments of adjacent text into the top and/or bottom rows.

    A line detector rarely crops tightly enough to exclude the descenders of the line
    above or the ascenders of the line below. Uses a squashed, shifted slice of the crop's
    own content as the fragment, so it looks like text without being readable text.
    """
    out = img.copy()
    h, w = out.shape[:2]
    src = np.roll(img, shift, axis=1)
    for frac, at_top in ((top_frac, True), (bottom_frac, False)):
        if frac <= 0:
            continue
        band_h = max(1, int(h * frac))
        if at_top:
            sl = src[int(h * 0.55):int(h * 0.95)]      # descenders of the line above
        else:
            sl = src[int(h * 0.05):int(h * 0.45)]      # ascenders of the line below
        band = cv2.resize(sl, (w, band_h), interpolation=cv2.INTER_AREA)
        y0, y1 = (0, band_h) if at_top else (h - band_h, h)
        out[y0:y1] = cv2.addWeighted(band, alpha, out[y0:y1], 1 - alpha, 0)
    return out


def ruled_line(img, thickness=1, value=90, at="bottom", offset=2):
    """A printed rule or the underline of the neighbouring line clipping the crop edge.
    Given a slight waviness so it is not a perfect raster row."""
    out = img.copy()
    h, w = out.shape[:2]
    y = offset if at == "top" else h - 1 - offset
    xs = np.arange(w)
    ys = np.clip(y + (np.sin(xs / max(40, w / 6.0)) * 1.2).astype(int), 0, h - 1)
    for dy in range(thickness):
        yy = np.clip(ys + dy, 0, h - 1)
        out[yy, xs] = np.minimum(out[yy, xs], value)
    return out


def skew(img, angle=1.0, max_drift=3.0, border=255):
    """Rotate, but clamp the angle so the far end of the line drifts at most `max_drift`
    pixels vertically.

    A line crop is very wide relative to its height, so an innocent-looking 2 degrees
    moves the end of a 900px crop by over 30px -- half the image height -- and slices the
    glyphs off. Real crops follow the text, so the residual skew *inside* a crop stays
    small even on a badly skewed page.
    """
    h, w = img.shape[:2]
    cap = np.degrees(np.arctan(max_drift / max(1, w / 2)))
    angle = float(np.clip(angle, -cap, cap))
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def wave(img, amplitude=1.5, period=160, border=255):
    """Baseline waviness from a curved page. Safer than rotation on a wide crop, since
    the displacement does not grow with width."""
    h, w = img.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    return cv2.remap(img, xs, ys + amplitude * np.sin(2 * np.pi * xs / period),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=border)


def pad_jitter(img, top=0.10, bottom=0.06, border=255):
    """Loosen the crop: shrink the glyph band and re-pad it unevenly, the way a real line
    detector leaves the text off-centre with uneven headroom. Height is preserved."""
    h, w = img.shape[:2]
    pt, pb = int(h * top), int(h * bottom)
    inner = max(8, h - pt - pb)
    small = cv2.resize(img, (w, inner), interpolation=cv2.INTER_AREA)
    return cv2.copyMakeBorder(small, pt, h - inner - pt, 0, 0,
                              cv2.BORDER_CONSTANT, value=border)


def binarize(img, offset=0, blur_first=0.0, adaptive=False, block=31):
    """Hard threshold, as many source PDFs are bilevel.

    Blurring BEFORE the threshold is what produces the blobby, ragged, partly-merged
    strokes of the harshest real samples -- a straight Otsu on a clean render just gives
    crisp edges. Never leave this as the final step: the resample later in the pipeline
    is what puts the soft grey edges of a real scan back.
    """
    src = gaussian_blur(img, blur_first) if blur_first > 0 else img
    if adaptive:
        return cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, block | 1, offset)
    t, _ = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ((src > (t + offset)) * 255).astype(np.uint8)


def gamma(img, g=1.0):
    """g < 1 darkens the ink, g > 1 fades it."""
    lut = np.clip(((np.arange(256) / 255.0) ** g) * 255, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def contrast_shift(img, gain=1.0, bias=0.0):
    return np.clip(img.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)


def illumination(img, strength=30, axis="x"):
    """Smooth lighting gradient from an unevenly lit page scan."""
    h, w = img.shape[:2]
    ramp = np.linspace(-1, 1, w if axis == "x" else h, dtype=np.float32)
    ramp = np.sin(ramp * np.pi / 2) * strength
    field = np.tile(ramp, (h, 1)) if axis == "x" else np.tile(ramp[:, None], (1, w))
    return np.clip(img.astype(np.float32) - field, 0, 255).astype(np.uint8)


def paper_normalize(img, target=248, pct=97, max_gain=2.6, black_pct=0.0):
    """Auto-levels, the way a scanner's own processing does it. Run this LAST.

    Stacked gamma / contrast / illumination / texture leaves the paper a flat mid-grey,
    which no real crop shows: every sample in data/actual_samples has a background mean
    of 247-252. Maps the `pct`th percentile onto `target`.

    `max_gain` has to be generous (2.6): the grime and tone effects above can leave the
    paper well under 200, and a tighter cap silently strands those samples at a grey the
    real crops never show.

    `black_pct` sets a black point as well, which matters more than it looks: a pure
    multiply lightens the INK along with the paper, so contrast stays capped around 200
    and the crisp bold real crops (215-230) stay unreachable. Pulling the black point up
    lets the paper go white while the ink stays dark, which is what a real scanner's
    levels do.
    """
    # Percentiles are measured on the MIDDLE rows only, then applied to the whole crop.
    # pad_jitter's pure-white padding and the stamped neighbour-line bands live at the
    # top/bottom edges; including them puts the 85th percentile at 255 straight away, the
    # gain comes out as 1.0, and a genuinely grey paper is left untouched.
    h = img.shape[0]
    core = img[int(h * 0.2):int(h * 0.8)]
    if core.size < 16:
        core = img
    p = float(np.percentile(core, pct))
    if p <= 1:
        return img
    lo = float(np.percentile(core, black_pct)) if black_pct > 0 else 0.0
    lo = min(lo, p - 1)                      # keep the span positive
    gain = min(max_gain, target / max(1.0, p - lo))
    # Re-derive the black point from the (possibly capped) gain so that `p` lands exactly
    # on `target`. Without this, a low-contrast crop asks for a gain above max_gain, the
    # cap silently denies it, and subtracting the full `lo` anyway leaves EVERYTHING darker
    # than it started -- paper at 255 came out at 169 with the ink crushed to 0, i.e. an
    # inverted-looking image out of a function whose job is to whiten the paper.
    lo = max(0.0, p - target / gain)
    if gain <= 1.001 and lo <= 0:
        return img
    out = (img.astype(np.float32) - lo) * gain
    return np.clip(out, 0, 255).astype(np.uint8)


# ======================================================================================
# Pre-built Augraphy severity ladders
#
# Four levels per augmentation, level 0 ~ the mildest real crop and level 3 ~ the worst
# still-legible one. Objects are built once at import and reused; each __call__ re-samples
# the object's internal randomness, so reuse still varies the output. Building a ladder
# (rather than one object with a wide range) is what lets the severity draw in `degrade`
# actually control strength while keeping the objects pre-built.
#
# Per-call cost at h=64 on augraphy 8.2.6 -- these drive the probabilities in `degrade`:
#   SubtleNoise/Jpeg 0.2ms, InkMottling 0.7, BrightnessTexturize 1.0, InkBleed 1.1,
#   BleedThrough 1.4, DirtyDrum 3.6, NoiseTexturize 5.8, BadPhotoCopy 33, Letterpress 79.
# ======================================================================================

_INK_BLEED = [
    # kernel_size must stay 3-5: at 7+ on a 64px line the glyphs merge into a black blob
    InkBleed(intensity_range=(0.4, 0.6), severity=(0.20, 0.30), kernel_size=(3, 3), p=1),
    InkBleed(intensity_range=(0.6, 0.8), severity=(0.30, 0.45), kernel_size=(5, 5), p=1),
    InkBleed(intensity_range=(0.85, 0.95), severity=(0.40, 0.45), kernel_size=(5, 5), p=1),
    InkBleed(intensity_range=(0.90, 1.00), severity=(0.45, 0.50), kernel_size=(5, 5), p=1),
]

_INK_MOTTLING = [
    InkMottling(ink_mottling_alpha_range=(0.20, 0.30), ink_mottling_gaussian_kernel_range=(3, 3), p=1),
    InkMottling(ink_mottling_alpha_range=(0.35, 0.45), ink_mottling_gaussian_kernel_range=(3, 3), p=1),
    InkMottling(ink_mottling_alpha_range=(0.50, 0.60), ink_mottling_gaussian_kernel_range=(3, 3), p=1),
    InkMottling(ink_mottling_alpha_range=(0.65, 0.75), ink_mottling_gaussian_kernel_range=(5, 5), p=1),
]

# Fades ink hard (drops contrast to 66-94 against the real band of 145-216) and costs
# ~79ms/call, so it fires rarely and is paired with a dilate by the ordering in `degrade`.
_LETTERPRESS = [
    Letterpress(n_samples=(100, 200), std_range=(400, 600), value_range=(200, 255), p=1),
    Letterpress(n_samples=(200, 300), std_range=(700, 1100), value_range=(200, 255), p=1),
    Letterpress(n_samples=(300, 500), std_range=(1100, 1600), value_range=(210, 255), p=1),
    Letterpress(n_samples=(400, 600), std_range=(1600, 2200), value_range=(220, 255), p=1),
]

_BLEED_THROUGH = [
    BleedThrough(intensity_range=(0.2, 0.4), alpha=0.15, offsets=(14, 14), p=1),
    BleedThrough(intensity_range=(0.4, 0.6), alpha=0.22, offsets=(18, 18), p=1),
    BleedThrough(intensity_range=(0.6, 0.8), alpha=0.30, offsets=(24, 24), p=1),
    BleedThrough(intensity_range=(0.8, 0.9), alpha=0.38, offsets=(30, 30), p=1),
]

_NOISE_TEXTURIZE = [
    NoiseTexturize(sigma_range=(2, 6), turbulence_range=(2, 5), p=1),
    NoiseTexturize(sigma_range=(4, 10), turbulence_range=(2, 5), p=1),
    NoiseTexturize(sigma_range=(6, 14), turbulence_range=(3, 6), p=1),
    NoiseTexturize(sigma_range=(10, 20), turbulence_range=(3, 7), p=1),
]

_BRIGHTNESS_TEXTURIZE = [
    BrightnessTexturize(texturize_range=(0.94, 0.98), deviation=0.05, p=1),
    BrightnessTexturize(texturize_range=(0.91, 0.96), deviation=0.08, p=1),
    BrightnessTexturize(texturize_range=(0.88, 0.94), deviation=0.11, p=1),
    BrightnessTexturize(texturize_range=(0.85, 0.92), deviation=0.14, p=1),
]

# line_width stays 1-2 and direction is horizontal (1) or both (2). Wide vertical streaks
# smear straight across whole glyphs at this height.
_DIRTY_DRUM = [
    DirtyDrum(line_width_range=(1, 1), line_concentration=0.05, direction=1, noise_intensity=0.3, p=1),
    DirtyDrum(line_width_range=(1, 1), line_concentration=0.10, direction=1, noise_intensity=0.5, p=1),
    DirtyDrum(line_width_range=(1, 2), line_concentration=0.18, direction=1, noise_intensity=0.65, p=1),
    DirtyDrum(line_width_range=(1, 2), line_concentration=0.25, direction=2, noise_intensity=0.80, p=1),
]

# noise_type=1 ONLY. Types 4-7 route through a worley-noise path that raises an
# AssertionError inside numba's JIT type inference. Being page-scale, on a line crop this
# lands as an edge grime band, which is exactly what several real samples show. ~33ms.
_BAD_PHOTOCOPY = [
    BadPhotoCopy(noise_type=1, noise_concentration=(0.10, 0.20), noise_sparsity=(0.6, 0.8), p=1),
    BadPhotoCopy(noise_type=1, noise_concentration=(0.10, 0.30), noise_sparsity=(0.5, 0.7), p=1),
    BadPhotoCopy(noise_type=1, noise_concentration=(0.20, 0.40), noise_sparsity=(0.4, 0.6), p=1),
    BadPhotoCopy(noise_type=1, noise_concentration=(0.30, 0.50), noise_sparsity=(0.3, 0.5), p=1),
]

_SUBTLE_NOISE = [
    SubtleNoise(subtle_range=6, p=1),
    SubtleNoise(subtle_range=10, p=1),
    SubtleNoise(subtle_range=14, p=1),
    SubtleNoise(subtle_range=18, p=1),
]

_JPEG = [
    Jpeg(quality_range=(60, 80), p=1),
    Jpeg(quality_range=(45, 65), p=1),
    Jpeg(quality_range=(30, 50), p=1),
    Jpeg(quality_range=(20, 35), p=1),
]

# ======================================================================================
# Composition
# ======================================================================================

def _p(prob):
    return random.random() < prob


def _scaled_p(prob, s):
    """Probability scaled by severity, so `severity` controls HOW MANY effects fire as
    well as how hard each one hits.

    Without this, a severity-0 sample still draws every effect at full probability and
    just gets mild versions of all of them, which stacks into something heavier than any
    mild real crop -- the opposite of what the parameter promises.
    """
    return random.random() < prob * (0.45 + 0.55 * s)


def _pick(ladder, s, jitter=0.25):
    """Choose a severity level from the ladder given the sample's severity `s` in [0, 1].

    The jitter keeps neighbouring samples from all landing on the same level, so the
    augmented distribution stays smooth instead of clustering at four points.
    """
    x = s + random.uniform(-jitter, jitter)
    return ladder[int(np.clip(x, 0.0, 0.999) * len(ladder))]


def _ink_fraction(g):
    """Cheap proxy for how much of the crop is ink.

    Measured over the middle rows only, so the neighbouring-line and ruled-line bands
    stamped into the top/bottom edges do not count as glyph ink.

    The threshold is RELATIVE to the crop's own paper level, not a fixed 128. With a fixed
    threshold, any global brightness change fakes a huge ink change -- an illumination
    gradient that lifts the ink from 100 to 130 reads as "94% of the ink vanished" and
    sends the backstop off chasing a problem that isn't there.
    """
    h = g.shape[0]
    core = g[int(h * 0.2):int(h * 0.8)]
    if core.size < 16:
        core = g
    paper = float(np.percentile(core, 90))
    thr = max(24.0, paper * 0.62)
    return float((core < thr).mean())


def _limit_ink_change(g, ref_frac, max_ratio=1.6, min_ratio=0.55, min_dark=110,
                      max_steps=3):
    """Pull the ink back toward the input's if the composition went too far either way.

    Per-effect ceilings do NOT bound a composition, and both directions can destroy the
    label rather than making the example hard:
      * too much ink -- a +0.8 dilate, a severity-0.5 InkBleed and a blurred binarisation
        are each individually safe, but stacked they close every counter and leave a solid
        blob. Erosion cannot undo that.
      * too little ink -- ink_dropout, a negative stroke_weight, Letterpress and a gamma
        fade stacked can erase so much of a glyph that it is no longer identifiable.
    This backstop is what lets the effect probabilities above stay aggressive.

    For reference: the real crops sit at 0.105-0.152 ink fraction against 0.115 for the
    clean render -- at most ~1.3x and never below ~0.8x -- so 1.6x / 0.55x is generous
    headroom on both sides and only the extreme tail is touched.

    `min_dark` is a separate floor on ink DARKNESS rather than ink area. The fade, dropout
    and brightening effects can stack into a crop whose darkest pixel is ~150 on white
    paper: still technically "inked" by area, but a blank image as far as the model is
    concerned, carrying a full text label. Stretching the levels is the fix, and doing it
    here rather than forcing a black point on every sample keeps genuinely low-contrast
    crops (the real range reaches down to 145) in the distribution.
    """
    # The darkness test uses the SAME middle-rows region as _ink_fraction. Testing the
    # whole crop instead lets a blank glyph band pass whenever the stamped neighbour-line
    # or ruled-line bands at the edges happen to be dark.
    h = g.shape[0]
    core = g[int(h * 0.2):int(h * 0.8)]
    if core.size < 16:
        core = g
    if float(np.percentile(core, 1)) > min_dark:
        g = paper_normalize(g, target=252, pct=85, black_pct=1.0)
    for _ in range(max_steps):
        frac = _ink_fraction(g)
        if frac > ref_frac * max_ratio:
            g = stroke_weight(g, -0.5)
        elif frac < ref_frac * min_ratio:
            g = stroke_weight(g, 0.5)
        else:
            break
    return g


def _limit_ink_loss(g, pre, min_ratio=0.55, max_steps=3):
    """Fade the ink-stage damage back toward `pre` if it erased too much of the glyph.

    This has to happen HERE, in the ink stage, and not only in the final backstop: the
    dropout/fade/Letterpress stack can leave a crop with literally no ink left (observed:
    ink fraction 0.000, darkest pixel 235 on white paper), and once the strokes are gone no
    end-of-pipeline guard can bring them back -- stretching or dilating a blank image just
    gives a blank image. Blending back toward the pre-damage copy restores both the
    coverage and the darkness while keeping the *character* of the dropout, just weaker.
    """
    ref = _ink_fraction(pre)
    if ref <= 0:
        return g
    for _ in range(max_steps):
        if _ink_fraction(g) >= ref * min_ratio:
            break
        g = cv2.addWeighted(pre, 0.5, g, 0.5, 0)
    return g


def _safe(applier, g):
    """Apply one Augraphy augmentation, falling back to the input on failure.

    Augraphy's randomised internals occasionally throw on a 64px-tall crop, and
    BadPhotoCopy can blow up inside numba's JIT. One bad draw must not kill a
    multi-hundred-thousand-image generation run or a training epoch.
    """
    try:
        out = applier(g)
    except Exception:
        return g
    if isinstance(out, dict):
        out = out.get("output", g)
    if out is None:
        return g
    out = np.asarray(out)
    if out.ndim == 3:                      # some augraphy paths promote to 3 channels
        out = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    return np.clip(out, 0, 255).astype(np.uint8)


def degrade(img, severity=None):
    """Apply the full composed degradation to one line crop.

    img      : uint8, HxW or HxWx3. Should already be the 64px-tall crop.
    severity : 0.0 (mildest real crop) to 1.0 (worst still-legible one). None draws it
               uniformly, which is what you want for training -- the model needs the whole
               range, not the average of it.
    Returns the same shape and dtype as `img`.

    No single effect here makes a synthetic crop look real: blur alone leaves the ink too
    light, thickening alone stays too clean, noise alone sits on razor-sharp glyphs. The
    real crops are a composition whose effects offset one another, which is why this is
    one function rather than a menu to pick one item from.
    """
    was_colour = img.ndim == 3
    # Work in grayscale throughout: the model consumes grayscale anyway, several ops
    # (Otsu, constant borders) are single-channel only, and it keeps every effect
    # channel-consistent. Converted back at the end.
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if was_colour else img
    g = np.ascontiguousarray(g, dtype=np.uint8)

    s = random.random() if severity is None else float(np.clip(severity, 0.0, 1.0))
    ref_ink = _ink_fraction(g)     # baseline for the legibility backstop near the end

    # ---- bilevel source --------------------------------------------------------------
    # Many source PDFs are bilevel, and they were thresholded at print/scan time -- i.e.
    # BEFORE the ink spread and the resample, not after. Order matters here beyond
    # realism: thresholding already-thickened ink fuses the strokes into a solid blob and
    # erodes cannot recover the counters, because binarisation threw the information away.
    # Applied first, it instead gives the crisp bilevel text that the thickening and
    # resample below turn into a convincing scan.
    if _scaled_p(0.18, s):
        g = binarize(g, offset=random.randint(-5, 10),
                     blur_first=random.uniform(0.4, 1.2))

    # ---- ink -------------------------------------------------------------------------
    ink_pre = g                    # snapshot for _limit_ink_loss at the end of this stage

    # Decided up front because it changes the thickening below. A page that failed to
    # transfer its ink is not also over-inked: letting a heavy dilate run first simply
    # fattens the dropout bites back over, and the two effects cancel into ordinary
    # slightly-blurry text. Coupling them is both more realistic and what keeps the broken
    # strokes actually visible in the output.
    do_dropout = _scaled_p(0.50, s)

    if _p(0.80):
        if do_dropout:
            amount = random.uniform(-0.35, 0.25)
        else:
            # heavier is the common case; the fading branch covers worn/low-toner pages
            amount = (random.uniform(0.10, 0.20 + 0.60 * s) if _p(0.8)
                      else -random.uniform(0.10, 0.15 + 0.35 * s))
        g = stroke_weight(g, amount)
    if _scaled_p(0.50, s):
        g = _safe(_pick(_INK_BLEED, s), g)
    if do_dropout:
        # High probability on purpose: partly-missing strokes are a defining feature of
        # the real crops, not an edge case. Coarse and fine blob scales are both drawn so
        # the model sees whole-word fading and single-stroke breaks.
        g = ink_dropout(g,
                        coverage=random.uniform(0.12, 0.22 + 0.23 * s),
                        scale=random.choice([6, 9, 12, 18, 30, 45]),
                        strength=random.uniform(0.60, 0.80 + 0.20 * s),
                        octaves=random.choice([2, 2, 3]))
    if _scaled_p(0.30, s):
        g = _safe(_pick(_INK_MOTTLING, s), g)
    if _scaled_p(0.03, s):                            # CHANGED(2048): 0.06 -> 0.03 (speed)
        g = _safe(_pick(_LETTERPRESS, s), g)          # ~79ms/call: halved, the biggest cost
    # Catch over-erasure while the strokes still exist to be restored.
    g = _limit_ink_loss(g, ink_pre)

    # ---- paper / grime ---------------------------------------------------------------
    if _scaled_p(0.25, s):
        g = _safe(_pick(_BLEED_THROUGH, s), g)
    if _scaled_p(0.25, s):
        g = _safe(_pick(_NOISE_TEXTURIZE, s), g)
    if _scaled_p(0.20, s):
        g = _safe(_pick(_BRIGHTNESS_TEXTURIZE, s), g)
    if _scaled_p(0.20, s):
        g = _safe(_pick(_DIRTY_DRUM, s), g)
    if _scaled_p(0.025, s):                           # CHANGED(2048): 0.05 -> 0.025 (speed)
        g = _safe(_pick(_BAD_PHOTOCOPY, s), g)        # ~33ms/call: halved
    if _scaled_p(0.25, s):
        g = illumination(g, random.uniform(8, 12 + 16 * s), random.choice(["x", "y"]))

    # ---- tone ------------------------------------------------------------------------
    if _scaled_p(0.45, s):
        # Weighted toward darkening: the real crops have darker ink than the render
        # (ink mean median 62 vs 61 clean but with far heavier strokes), and fading is
        # the rarer failure mode.
        g = gamma(g, random.uniform(0.90 - 0.45 * s, 0.95) if _p(0.7)
                     else random.uniform(1.10, 1.15 + 0.35 * s))
    if _scaled_p(0.35, s):
        # gain > 1 with a negative bias is what reaches the crisp high-contrast end of
        # the real range (the bold samples measure 215-230); without it the pipeline tops
        # out near 196 and those crops fall outside the augmented distribution.
        g = contrast_shift(g, random.uniform(0.95, 1.10 + 0.30 * s),
                           random.uniform(-8 - 20 * s, 4 + 6 * s))

    # ---- crop artefacts --------------------------------------------------------------
    if _scaled_p(0.35, s):
        g = neighbour_line(g, 0.12 if _p(0.5) else 0.0, 0.09 + 0.06 * s,
                           alpha=random.uniform(0.5, 0.9),
                           shift=random.randint(20, 120))
    if _scaled_p(0.15, s):
        g = ruled_line(g, random.choice([1, 2]), random.randint(50, 140),
                       random.choice(["top", "bottom"]), random.randint(0, 3))
    if _p(0.30):
        g = pad_jitter(g, random.uniform(0.02, 0.14), random.uniform(0.02, 0.10))

    # ---- geometry --------------------------------------------------------------------
    if _p(0.40):
        g = skew(g, random.choice([-3.0, 3.0]), max_drift=random.uniform(1.0, 4.0))
    if _p(0.30):
        g = wave(g, random.uniform(0.8, 1.0 + 1.6 * s), random.uniform(80, 220))

    # ---- sensor noise (before the resample, so the resample averages it down) --------
    grainy = False
    if _scaled_p(0.35, s):
        g = _safe(_pick(_SUBTLE_NOISE, s), g)
        grainy = True
    if _scaled_p(0.30, s):
        g = speckle(g, random.uniform(0.0008, 0.0015 + 0.004 * s), blob_frac=0.15,
                    dark=(20, 130), light_amount=0.002 if _p(0.3) else 0.0)
        grainy = True

    # ---- resample --------------------------------------------------------------------
    # Grain that was never resampled reads as video static rather than paper, so a grainy
    # sample takes the resample that would have produced it far more often. Most samples
    # skip this entirely -- see the note on resolution_loss about why it is a probability
    # knob and not a strength knob.
    if _p(0.55 if grainy else 0.28):
        g = resolution_loss(g, random.uniform(0.78 - 0.25 * s, 1.0),
                            up=random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC]))
    if _scaled_p(0.15, s):
        g = gaussian_blur(g, random.uniform(0.30, 0.35 + 0.35 * s))
    if _scaled_p(0.10, s):
        g = motion_blur(g, 3 if _p(0.7) else 5, random.choice([0, 90]))

    # ---- compression -----------------------------------------------------------------
    if _scaled_p(0.40, s):
        g = _safe(_pick(_JPEG, s), g)

    # ---- auto-levels -----------------------------------------------------------------
    # pct=85 with a target at/near 255 clips the paper bulk to white, which is how a real
    # scan looks; mapping a high percentile instead leaves the paper a few levels grey.
    # The black point stays optional so low-contrast crops remain possible (the real range
    # goes down to 145) -- forcing it on every sample stretches the paper variance well past
    # the real 13-16. The blank-image case it used to cover is handled by the darkness floor
    # in the backstop below instead, which only fires when it is actually needed.
    g = paper_normalize(g, target=random.randint(250, 255), pct=85,
                        black_pct=random.uniform(0.5, 3.0) if _p(0.7) else 0.0)

    # ---- legibility backstop, truly last ---------------------------------------------
    # It has to run AFTER auto-levels, not before: auto-levels rescales tone, which moves
    # the ink measurement, so a backstop placed earlier bounds an intermediate image rather
    # than the one the model is trained on. Two-sided, because the ink stage can now remove
    # ink (ink_dropout) as well as add it.
    g = _limit_ink_change(g, ref_ink)

    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR) if was_colour else g


P_CLEAN = 0.50

def make_augmenter(p_clean=P_CLEAN, severity=None):
    """Build the per-sample callable: augment(uint8 HxW or HxWx3) -> same shape.

    p_clean  : fraction of samples passed through untouched. Keeps the model able to read
               crisp text (it already does that well -- don't regress it) and trims CPU.
    severity : pin the severity for every sample, or None to draw per sample. Pinning is
               for previews and ablations, not training.

    Never raises: a failing effect falls back to the image as it was.
    """
    def augment(img):
        if random.random() < p_clean:
            return img
        try:
            return degrade(img, severity)
        except Exception as exc:
            print(f"[warn] augmentation failed, using clean image: {exc}")
            return img

    return augment


@dataclass
class CTCEncoderConfig:
    # ---- input ----
    image_height: int = 64          # fixed by the positional design (§2)
    max_image_width: int = 1024     # W ≤ max_image_width, multiple of 8
    downsample: int = 8             # conv-stem width reduction: T = W // downsample
    max_frames: int = 128           # = max_image_width // downsample (pos-emb length)

    # ---- conv stem ----
    stem_channels: tuple = (32, 64, 128, 256, 320, 384)
    num_groups: int = 32            # GroupNorm groups (all stem widths divide 32)

    # ---- transformer encoder ----
    embed_dim: int = 384            # d_model (= last stem channel)
    num_layers: int = 10
    num_heads: int = 8
    mlp_dim: int = 1536
    dropout: float = 0.05           # CHANGED: match train_ctc_encoder.py (was 0.1)
    drop_path_rate: float = 0.1     # stochastic depth (max rate, linearly scaled)

    # ---- CTC head ----
    vocab_size: int = 2048          # grapheme classes; blank appended at this index


class ConvBlock(nn.Module):
    """Conv → GroupNorm → GELU (one row of the §2.1 stem table)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, num_groups):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        groups = num_groups if out_channels % num_groups == 0 else 1
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ConvStem(nn.Module):
    """Six-block convolutional tokenizer: (B,1,64,W) -> (B, T=W/8, 384)."""

    def __init__(self, cfg: CTCEncoderConfig):
        super().__init__()
        c = cfg.stem_channels
        assert len(c) == 6, "stem expects 6 conv blocks (§2.1)"
        assert c[-1] == cfg.embed_dim, "last stem channel must equal embed_dim"

        specs = [
            ((3, 3), (2, 2), (1, 1)),  # Conv1: 32×32×W/2
            ((3, 3), (2, 2), (1, 1)),  # Conv2: 64×16×W/4
            ((3, 3), (2, 1), (1, 1)),  # Conv3: 128×8×W/4
            ((3, 3), (2, 2), (1, 1)),  # Conv4: 256×4×W/8
            ((3, 3), (2, 1), (1, 1)),  # Conv5: 320×2×W/8
            ((2, 1), (2, 1), (0, 0)),  # Conv6: 384×1×W/8
        ]

        in_ch = 1
        blocks = []
        for out_ch, (k, s, p) in zip(c, specs):
            blocks.append(ConvBlock(in_ch, out_ch, k, s, p, cfg.num_groups))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H=64, W)
        for block in self.blocks:
            x = block(x)
        B, D, H, T = x.shape
        assert H == 1, f"conv stem did not collapse height to 1 (got {H})"
        return x.squeeze(2).transpose(1, 2)  # (B, T, D)


class DropPath(nn.Module):
    """Stochastic depth: randomly drop the residual branch per-sample at train time."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x / keep_prob * mask


class TransformerBlock(nn.Module):
    """Pre-LN transformer encoder block with stochastic depth (§2.1)."""

    def __init__(self, cfg: CTCEncoderConfig, drop_path: float):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(cfg.embed_dim)
        self.attention = nn.MultiheadAttention(
            cfg.embed_dim, cfg.num_heads, dropout=cfg.dropout, batch_first=True
        )
        self.layer_norm2 = nn.LayerNorm(cfg.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.mlp_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_dim, cfg.embed_dim),
            nn.Dropout(cfg.dropout),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        norm_x = self.layer_norm1(x)
        attn_out, _ = self.attention(
            norm_x, norm_x, norm_x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.layer_norm2(x)))
        return x


class ImageEncoderCTC(nn.Module):
    """Conv stem → transformer encoder → CTC head (~20M params).

    Only ``encode()`` is used by the encoder-decoder (frame features + a frame
    padding mask); the CTC head / loss path is kept for standalone eval parity.
    """

    def __init__(self, cfg: CTCEncoderConfig, label_pad_id: int = -100):
        super().__init__()
        self.cfg = cfg
        self.blank_id = cfg.vocab_size
        self.num_classes = cfg.vocab_size + 1
        self.label_pad_id = label_pad_id

        self.stem = ConvStem(cfg)

        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.max_frames, cfg.embed_dim))
        self.dropout = nn.Dropout(cfg.dropout)

        dpr = torch.linspace(0.0, cfg.drop_path_rate, cfg.num_layers).tolist()
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, drop_path=dpr[i]) for i in range(cfg.num_layers)]
        )
        self.layer_norm = nn.LayerNorm(cfg.embed_dim)

        self.ctc_head = nn.Linear(cfg.embed_dim, self.num_classes)
        self.ctc_loss = nn.CTCLoss(blank=self.blank_id, zero_infinity=True)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, images: torch.Tensor, input_lengths: torch.Tensor | None):
        """Run the stem + transformer, returning frame features and the frame
        padding mask (True = padded frame)."""
        feats = self.stem(images)                       # (B, T, D)
        B, T, D = feats.shape
        assert T <= self.cfg.max_frames, (
            f"T={T} exceeds max_frames={self.cfg.max_frames}; widen the pos-emb"
        )
        feats = feats + self.pos_embed[:, :T, :]
        feats = self.dropout(feats)

        if input_lengths is not None:
            frame_idx = torch.arange(T, device=feats.device).unsqueeze(0)  # (1, T)
            key_padding_mask = frame_idx >= input_lengths.unsqueeze(1)     # (B, T)
        else:
            key_padding_mask = None

        for block in self.blocks:
            feats = block(feats, key_padding_mask)
        feats = self.layer_norm(feats)
        return feats, key_padding_mask

    @staticmethod
    def frames_from_width(width: int, downsample: int = 8) -> int:
        """Number of CTC frames a line of pixel ``width`` produces (= width/8)."""
        assert width % downsample == 0, "width must be a multiple of the downsample factor"
        return width // downsample


class TeluguGraphemeTokenizer(PreTrainedTokenizer):
    """HuggingFace-compatible grapheme-cluster tokenizer for Telugu.

    Parameters
    ----------
    vocab : dict[str, int]
        Mapping from grapheme string → token ID.  Must include all
        ``SPECIAL_TOKENS_LIST`` entries.
    add_bos_token : bool
        Automatically prepend ``[BOS]`` on ``encode()``.
    add_eos_token : bool
        Automatically append ``[EOS]`` on ``encode()``.
    """

    vocab_files_names = {"vocab_file": "vocab.json"}  # ← fix 1
    model_input_names = ["input_ids", "attention_mask"]  # ← fix 2

    def __init__(
            self,
            vocab_file: Optional[str] = None,
            vocab_list: Optional[list[str]] = None,
            add_bos_token: bool = True,
            add_eos_token: bool = True,
            pad_token: str = "[PAD]",
            unk_token: str = "[UNK]",
            bos_token: str = "[BOS]",
            eos_token: str = "[EOS]",
            mask_token: str = "[MASK]",
            **kwargs,
    ):
        vocab = {}

        if vocab_file is not None:
            with open(vocab_file, encoding="utf-8") as f:
                vocab = json.load(f)

        if not vocab and vocab_list is not None:
            for grapheme in vocab_list:
                vocab[grapheme] = len(vocab)

        if not vocab:
            raise AssertionError("Either `vocab_file` or `vocab_list` has to be given.")

        self.SPECIAL_TOKENS_LIST = [pad_token, unk_token, bos_token, eos_token, mask_token]

        self.grapheme_pattern = regex.compile(r'\X')

        for tok in self.SPECIAL_TOKENS_LIST:
            if tok not in vocab:
                vocab[tok] = len(vocab)

        self.vocab = vocab  # ← must be BEFORE super().__init__()
        self._inv_vocab = {v: k for k, v in vocab.items()}
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.UNK = unk_token

        super().__init__(
            pad_token=pad_token,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            mask_token=mask_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            padding_side="right",
            model_max_length=4096,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.vocab)

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        tokens = []
        for grapheme in self.grapheme_pattern.findall(text):
            if grapheme in self.vocab:
                tokens.append(grapheme)
            else:
                for codepoint in grapheme:
                    tokens.append(codepoint)
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab.get(self.UNK, 1))

    def _convert_id_to_token(self, index: int) -> str:
        return self._inv_vocab.get(index, self.UNK)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        cleaned = [t for t in tokens if t not in self.SPECIAL_TOKENS_LIST]
        return "".join(cleaned)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos = [self.bos_token_id] if self.add_bos_token else []
        eos = [self.eos_token_id] if self.add_eos_token else []
        out = bos + token_ids_0 + eos
        if token_ids_1 is not None:
            out += bos + token_ids_1 + eos
        return out

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        if already_has_special_tokens:
            return super().get_special_tokens_mask(token_ids_0, token_ids_1, already_has_special_tokens=True)
        bos = [1] if self.add_bos_token else []
        eos = [1] if self.add_eos_token else []
        res = bos + [0] * len(token_ids_0) + eos
        if token_ids_1 is not None:
            res += bos + [0] * len(token_ids_1) + eos
        return res

    def save_vocabulary(
            self,
            save_directory: str,
            filename_prefix: Optional[str] = None,
    ) -> Tuple[str]:
        os.makedirs(save_directory, exist_ok=True)
        fname = (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        vocab_path = os.path.join(save_directory, fname)
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        return (vocab_path,)


@dataclass
class GPTConfig:
    vocab_size: int = 2048
    embed_dim: int = 512
    hidden_dim: int = 2048
    num_heads: int = 8
    num_layers: int = 12
    ctx_len: int = 1024
    dropout: float = 0.1


def calculate_positional_encodings(positions: torch.Tensor, embed_dim: int):
    i = torch.arange(embed_dim // 2, dtype=torch.float32)
    div_term = 10000 ** (2 * i / embed_dim)  # (D/2,)
    pos = positions.float().unsqueeze(1)  # (T, 1)
    args = pos / div_term  # (T, D/2)
    enc = torch.zeros(len(positions), embed_dim)
    enc[:, 0::2] = torch.sin(args)
    enc[:, 1::2] = torch.cos(args)
    return enc


class SwiGLU(nn.Module):
    """Implement the SwiGLU activation function"""

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, embed_dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MultiHeadAttention(nn.Module):
    """Implement multi head attention"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask=None):
        # attn_mask: (B, 1, T, T) boolean — True means KEEP, False means MASK OUT
        B, T, _ = query.shape
        _, S, _ = key.shape

        queries = self.q_proj(query).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.k_proj(key).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.v_proj(value).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        dropout_p = self.dropout if self.training else 0.0
        ctx_embeds = F.scaled_dot_product_attention(
            queries, keys, values,
            dropout_p=dropout_p,
            attn_mask=attn_mask,
            is_causal=False
        )

        ctx_embeds = ctx_embeds.transpose(1, 2).reshape(B, T, self.embed_dim)
        return self.out_proj(ctx_embeds)


class GPTTransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attention_layer = MultiHeadAttention(config.embed_dim, config.num_heads, config.dropout)
        self.mlp = nn.Sequential(
            SwiGLU(config.embed_dim, config.hidden_dim),
            nn.Dropout(config.dropout)
        )
        self.layer_norm1 = nn.LayerNorm(config.embed_dim)
        self.layer_norm2 = nn.LayerNorm(config.embed_dim)

    def forward(self, x: torch.Tensor, attn_mask=None):
        # x -> B, T, D
        normed = self.layer_norm1(x)
        x = x + self.attention_layer(normed, normed, normed, attn_mask=attn_mask)
        x = x + self.mlp(self.layer_norm2(x))
        return x


class GPTModel(nn.Module):
    _keys_to_ignore_on_save = None

    def __init__(self, config: GPTConfig, pad_index: int):
        super().__init__()
        self.pad_index = pad_index
        self.embedding_layer = nn.Embedding(config.vocab_size, config.embed_dim)
        self.register_buffer("positional_encodings",
                             calculate_positional_encodings(torch.arange(config.ctx_len), config.embed_dim))
        self.transformer_blocks = nn.ModuleList([
            GPTTransformerBlock(config) for _ in range(config.num_layers)
        ])
        self.layer_norm = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

    def _build_attn_mask(self, input_ids, attention_mask):
        """
        Builds a combined boolean causal + padding mask.
        SDPA expects: True = attend, False = ignore.
        Shape: (B, 1, T, T)
        """
        B, T = input_ids.shape
        device = input_ids.device
        # Causal mask: upper triangle is False (masked), lower triangle True
        causal = torch.ones(T, T, dtype=torch.bool, device=device).tril()  # (T, T)
        if attention_mask is not None:
            # attention_mask: (B, T), 1=real token, 0=pad
            # Expand to (B, 1, 1, T) so it broadcasts over query positions
            pad_mask = attention_mask.bool().unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            combined = causal.unsqueeze(0).unsqueeze(0) & pad_mask  # (B, 1, T, T)
        else:
            combined = causal.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
        return combined

    def forward(self, input_ids, attention_mask=None, labels=None):
        # input_ids -> (B, T)
        embeds = self.embedding_layer(input_ids)
        T = input_ids.shape[1]
        embeds = embeds + self.positional_encodings[:T].unsqueeze(0)
        attn_mask = self._build_attn_mask(input_ids, attention_mask)
        for block in self.transformer_blocks:
            embeds = block(embeds, attn_mask=attn_mask)
        embeds = self.layer_norm(embeds)
        logits = self.lm_head(embeds)
        loss = None

        # Always calculate the loss
        # shift for causal LM
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.pad_index
        )
        return CausalLMOutput(
            loss=loss,
            logits=logits
        )


class DecoderTransformerBlock(nn.Module):
    """Transformer block; cross-attention (with a zero-init tanh gate) is optional."""

    def __init__(self, config: GPTConfig, use_cross_attention: bool = True):
        super().__init__()
        self.use_cross_attention = use_cross_attention
        self.attention_layer = MultiHeadAttention(config.embed_dim, config.num_heads, dropout=config.dropout)

        if self.use_cross_attention:
            self.cross_attention_layer = MultiHeadAttention(config.embed_dim, config.num_heads, dropout=config.dropout)
            self.layer_norm1_5 = nn.LayerNorm(config.embed_dim)
            # Zero-init tanh gate (Flamingo-style): tanh(0) == 0, so the cross-attention
            # branch contributes nothing at init and is phased in as the gate trains.
            self.cross_attn_gate = nn.Parameter(torch.zeros(1))

        self.mlp = nn.Sequential(
            SwiGLU(config.embed_dim, config.hidden_dim),
            nn.Dropout(config.dropout)
        )

        self.layer_norm1 = nn.LayerNorm(config.embed_dim)
        self.layer_norm2 = nn.LayerNorm(config.embed_dim)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, attn_mask=None, padding_mask=None):
        normed = self.layer_norm1(x)
        attn_out = self.attention_layer(normed, normed, normed, attn_mask)
        x = x + attn_out

        if self.use_cross_attention:
            cross_in = self.layer_norm1_5(x)
            cross_out = self.cross_attention_layer(cross_in, encoder_output, encoder_output, padding_mask)
            x = x + torch.tanh(self.cross_attn_gate) * cross_out

        mlp_in = self.layer_norm2(x)
        mlp_out = self.mlp(mlp_in)
        x = x + mlp_out

        return x


class TextDecoder(nn.Module):
    """A text decoder model of the transformer model"""
    _keys_to_ignore_on_save = None

    def __init__(self, config: GPTConfig, pad_index: int):
        super().__init__()
        self.pad_index = pad_index
        self.embedding_layer = nn.Embedding(config.vocab_size, config.embed_dim)
        self.register_buffer("positional_encodings",
                             calculate_positional_encodings(torch.arange(config.ctx_len), config.embed_dim))
        # Cross-attention lives on even-indexed blocks only (every 2nd block).
        self.transformer_blocks = nn.ModuleList([
            DecoderTransformerBlock(config, use_cross_attention=(i % 2 == 0))
            for i in range(config.num_layers)
        ])
        self.layer_norm = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

    def _build_causal_attn_mask(self, input_ids, padding_mask):
        """
        Builds a combined boolean causal + padding mask.
        SDPA expects: True = attend, False = ignore.
        Shape: (B, 1, T, T)
        """
        B, T = input_ids.shape
        device = input_ids.device
        # Causal mask: upper triangle is False (masked), lower triangle True
        causal = torch.ones(T, T, dtype=torch.bool, device=device).tril()  # (T, T)
        if padding_mask is not None:
            # padding_mask: (B, T), 1=real token, 0=pad
            # Expand to (B, 1, 1, T) so it broadcasts over query positions
            pad_mask = padding_mask.bool().unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            combined = causal.unsqueeze(0).unsqueeze(0) & pad_mask  # (B, 1, T, T)
        else:
            combined = causal.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
        return combined

    def forward(self, input_ids, encoder_output, text_padding_mask=None, img_text_padding_mask=None, labels=None):
        # input_ids -> (B, T)
        embeds = self.embedding_layer(input_ids)
        T = input_ids.shape[1]
        embeds = embeds + self.positional_encodings[:T].unsqueeze(0)
        causal_attn_mask = self._build_causal_attn_mask(input_ids, text_padding_mask)
        for block in self.transformer_blocks:
            embeds = block(embeds, encoder_output, attn_mask=causal_attn_mask, padding_mask=img_text_padding_mask)
        embeds = self.layer_norm(embeds)
        logits = self.lm_head(embeds)
        loss = None

        # Always calculate the loss
        # shift for causal LM
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.pad_index
        )
        return CausalLMOutput(
            loss=loss,
            logits=logits
        )


class EncoderDecoder(nn.Module):
    """An Image encoder and text decoder based transformer model"""

    def __init__(self, encoder_config: CTCEncoderConfig, decoder_config: GPTConfig, pad_index: int):
        super().__init__()
        self.encoder_model = ImageEncoderCTC(encoder_config)
        self.decoder_model = TextDecoder(decoder_config, pad_index)
        # Bridge the encoder width (384) to the decoder width (512) so the frame
        # features can feed the decoder's cross-attention keys/values. Trainable
        # (the encoder is frozen); a no-op nn.Identity when the widths already match.
        if encoder_config.embed_dim != decoder_config.embed_dim:
            self.enc_to_dec = nn.Linear(encoder_config.embed_dim, decoder_config.embed_dim)
        else:
            self.enc_to_dec = nn.Identity()

        # STAGE-1: the encoder is FROZEN and its features feed the trainable bridge
        # (enc_to_dec) as a plain input, so NO gradient needs to flow back through it.
        # Running encode() under torch.no_grad() makes that explicit. This is SAFE for
        # the adapters — the bridge/cross-attention gradients are computed from the
        # (constant) encoder features and are unchanged.
        # ⚠️ SET TO False FOR FULL / STAGE-2 TRAINING: when the encoder is unfrozen it
        #    must receive gradients, so the no_grad wrapper has to be turned off.
        self.encoder_no_grad = True

    @torch.no_grad()
    def generate(self, pixel_values, bos_id, eos_id, max_new_tokens=256, no_repeat_cycle=True):
        """ADDED (stage-1): single-sample (B=1) greedy decode for CER.
        NOTE: in stage-1 the decoder lm_head / embeddings are FROZEN, so the model
        cannot learn to emit EOS here; the repetition + length guards stop the
        otherwise-runaway decode so CER reflects the characters it DOES produce."""
        self.eval()
        # All frames are valid for a single un-padded image, so input_lengths=None
        # (encode() then treats every frame as real and skips the cross-attn mask).
        enc_out, key_padding_mask = self.encoder_model.encode(pixel_values, None)
        enc_out = self.enc_to_dec(enc_out)
        cross_key_mask = None if key_padding_mask is None else (~key_padding_mask).unsqueeze(1).unsqueeze(2)
        ids = torch.full((pixel_values.shape[0], 1), bos_id, dtype=torch.long, device=pixel_values.device)
        ctx = self.decoder_model.positional_encodings.shape[0]
        for _ in range(min(max_new_tokens, ctx - 1)):
            out = self.decoder_model(ids, enc_out, text_padding_mask=None,
                                     img_text_padding_mask=cross_key_mask, labels=None)
            nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
            if nxt.item() == eos_id:
                break
            if no_repeat_cycle and ids.shape[1] > 24:  # stop short repeating loops
                tail = ids[0, -12:].tolist()
                if any(tail == tail[-k:] * (12 // k) for k in (1, 2, 3, 4)):
                    break
        return ids[0].tolist()

    def forward(self, pixel_values, input_ids, input_lengths=None, text_padding_mask=None, return_loss=True):
        # pixel_values -> (B, 1, H, W); input_lengths -> (B,) valid frames = W_real // downsample
        # Frozen encoder -> run under no_grad (see self.encoder_no_grad in __init__).
        enc_ctx = torch.no_grad() if self.encoder_no_grad else contextlib.nullcontext()
        with enc_ctx:
            encoder_output, key_padding_mask = self.encoder_model.encode(pixel_values, input_lengths)
        encoder_output = self.enc_to_dec(encoder_output)         # (B, T, dec_embed_dim); bridge is trainable

        if key_padding_mask is not None:
            # Custom cross-attention expects an SDPA-style mask (True = KEEP), shape (B, 1, 1, T).
            cross_key_mask = (~key_padding_mask).unsqueeze(1).unsqueeze(2)
        else:
            cross_key_mask = None
        decoder_output = self.decoder_model(input_ids, encoder_output, text_padding_mask, cross_key_mask, None)
        return decoder_output


# ============================================================================
# ADDED (stage-1): CER evaluation utilities + callback.
# ============================================================================
def _edit_distance(a, b):
    # Levenshtein, dependency-free
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[n]


def compute_cer(preds, refs):
    tot_e = tot_c = 0
    for p, r in zip(preds, refs):
        tot_e += _edit_distance(p, r)
        tot_c += max(len(r), 1)
    return tot_e / max(tot_c, 1)


# B=1 greedy generation with no KV cache is SLOW (~seconds/sequence). This runs on
# rank 0 only while the other ranks wait at the next barrier, so keep the TOTAL small:
# at 48/slice x 2 slices (natural / random) that is ~96 sequences, a few minutes —
# comfortably inside the raised ddp_timeout. Raise for a sharper CER estimate only if
# you also raise ddp_timeout.
CER_EVAL_SAMPLES = 48   # eval images decoded per slice per CER eval (B=1 greedy)
CER_PRINT_K = 5  # print this many ref/hyp pairs each eval


class CEREvalCallback(TrainerCallback):
    """Generate-based CER during evaluate(), reported PER WIDTH SLICE.

    ``eval_slices`` is a dict ``{slice_name: [raw_row, ...]}`` where each raw row
    has the image / text columns. For each slice we B=1 greedy-decode up to
    CER_EVAL_SAMPLES images and inject ``eval_<slice>_cer`` into the metrics dict
    (also logged to wandb). A macro-average ``eval_cer`` over slices is added too.

    Early stopping was removed, so this metric is purely for monitoring; CER is
    still computed on every rank (identical subset -> identical value) while
    printing / wandb logging happen on rank 0 only.
    """

    def __init__(self, model, eval_slices, preprocessor, tokenizer,
                 image_col="image", text_col="text"):
        self.model = model
        self.slices = eval_slices  # {name: list of raw rows with image/text cols}
        self.prep = preprocessor   # clean preprocessor (no augmentation)
        self.tok = tokenizer
        self.image_col = image_col
        self.text_col = text_col

    def _slice_cer(self, rows, device, ctx, image_col, text_col):
        n = min(CER_EVAL_SAMPLES, len(rows))
        preds, refs = [], []
        for i in range(n):
            ex = rows[i]
            pix = self.prep(ex[image_col]).unsqueeze(0).to(device)  # (1, 1, H, W)
            ref_ids = self.tok.encode(ex[text_col])
            cap = min(ctx - 1, int(1.5 * len(ref_ids)) + 10)        # length-aware cap
            ids = self.model.generate(pix, self.tok.bos_token_id, self.tok.eos_token_id,
                                      max_new_tokens=cap)
            preds.append(self.tok.decode(ids, skip_special_tokens=True))
            refs.append(ex[text_col])
        return compute_cer(preds, refs), preds, refs

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # RANK-0 ONLY. This CER is monitoring-only (no early stopping / best-model
        # selection reads it), and the generation is a long, collective-free stretch of
        # work. Running it independently on every rank lets the ranks desync — one
        # finishes and hits the next NCCL barrier while the other is still decoding — and
        # blows past ddp_timeout (the 30-min ALLGATHER watchdog that killed the run).
        # Doing it on rank 0 alone keeps control flow deterministic; the other ranks skip
        # to the following save/step barrier and wait (ddp_timeout is raised to cover it).
        if not state.is_world_process_zero:
            return
        try:
            self.model.eval()
            device = next(self.model.parameters()).device
            ctx = self.model.decoder_model.positional_encodings.shape[0]
            image_col, text_col = self.image_col, self.text_col
            per_slice = {}
            for name, rows in self.slices.items():
                if not rows:
                    continue
                cer, preds, refs = self._slice_cer(rows, device, ctx, image_col, text_col)
                per_slice[name] = cer
                if metrics is not None:
                    metrics[f"eval_{name}_cer"] = cer
                print(f"[eval] step {state.global_step}  {name} CER={cer:.4f} (n={min(CER_EVAL_SAMPLES, len(rows))})", flush=True)
                for p, r in list(zip(preds, refs))[:CER_PRINT_K]:
                    print(f"    [{name}] ref: {r!r}")
                    print(f"    [{name}] hyp: {p!r}")
            if per_slice:
                macro = sum(per_slice.values()) / len(per_slice)  # macro-average across slices
                if metrics is not None:
                    metrics["eval_cer"] = macro
                print(f"[eval] step {state.global_step}  macro CER={macro:.4f}", flush=True)
                try:
                    import wandb
                    wandb.log({**{f"eval_{k}_cer": v for k, v in per_slice.items()},
                               "eval_cer": macro}, step=state.global_step)
                except Exception:
                    pass
        except Exception as exc:
            print(f"[CER] failed at step {state.global_step}: {exc}", flush=True)
        finally:
            self.model.train()  # restore train mode for the rest of training


class OCRCollator:
    """
    Each dataset example is expected to be a dict with:
      - 'pixel_values': float tensor (1, H, W_i)   # H fixed (image_height=64), W_i variable
      - 'input_ids'   : 1D long tensor / list      # variable length, NO padding yet

    Produces a batch dict whose keys match EncoderDecoder.forward exactly:
      pixel_values      (B, 1, H, W_max)   float, right-padded on width with zeros
      input_ids         (B, T_max)         long, right-padded with pad_token_id
      input_lengths     (B,)               long, valid CTC frames = W_i // downsample
      text_padding_mask (B, T_max)         float, 1 = REAL token  (decoder convention)

    The CTC encoder collapses image height to 1, so each image contributes
    W_i // downsample frame tokens. ``input_lengths`` tells encode() how many of
    the padded-batch frames are real; it builds the cross-attention key mask from it.

    Assumes every image width W_i is already a multiple of ``downsample`` (your
    ImagePreprocessor pads to that). If not, the batch is padded up to a multiple.
    """

    def __init__(self, pad_token_id: int, downsample: int = 8):
        self.pad_token_id = pad_token_id
        self.downsample = downsample

    def __call__(self, batch):
        images = [ex["pixel_values"] for ex in batch]
        token_seqs = [torch.as_tensor(ex["input_ids"], dtype=torch.long) for ex in batch]
        B = len(batch)

        # ---- Images ----
        widths = [img.shape[-1] for img in images]
        max_w = max(widths)
        if max_w % self.downsample != 0:  # safety; should already be a multiple
            max_w += self.downsample - (max_w % self.downsample)

        padded_imgs = [F.pad(img, (0, max_w - w)) for img, w in zip(images, widths)]
        pixel_values = torch.stack(padded_imgs, dim=0)  # (B, 1, H, max_w)

        # Valid frame count per sample = real_width // downsample (ceil so a partial
        # edge frame is counted as real, not padded).
        input_lengths = torch.tensor(
            [min((w + self.downsample - 1) // self.downsample, max_w // self.downsample) for w in widths],
            dtype=torch.long,
        )

        # ---- Text (right-pad) ----
        max_t = max(seq.shape[0] for seq in token_seqs)
        input_ids = torch.full((B, max_t), self.pad_token_id, dtype=torch.long)
        text_padding_mask = torch.zeros((B, max_t), dtype=torch.float)  # 1 = real
        for i, seq in enumerate(token_seqs):
            n = seq.shape[0]
            input_ids[i, :n] = seq
            text_padding_mask[i, :n] = 1.0

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "text_padding_mask": text_padding_mask
        }


# ============================================================================
# Optimizer / scheduler — PORTED from train_ctc_encoder.py.
# ============================================================================
def build_optimizer(model, lr, weight_decay, betas):
    """Single LR tier. Biases / norm weights / positional embeddings get no weight
    decay (matches the prior Trainer default); other weights get `weight_decay`.

    In stage-1 only the newly-added adapter params are trainable (cross-attention
    q/k/v/out + the bridge Linear -> decay; layer_norm1_5 + the cross_attn_gate ->
    no decay); the frozen encoder/decoder params are skipped."""
    decay, nodecay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name == "pos_embed" or p.ndim < 2 or "norm" in name.lower():
            nodecay.append(p)                             # embeddings/bias/norm -> no WD
        else:
            decay.append(p)
    groups = [
        {"params": decay, "lr": lr, "weight_decay": weight_decay},
        {"params": nodecay, "lr": lr, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, betas=betas)


# Scheduler: warmup -> cosine decay to an ABSOLUTE floor `min_lr`. Single LR tier.
def build_scheduler(optimizer, warmup_steps, total_steps, min_lr):
    """Per-group linear warmup -> cosine decay to an ABSOLUTE floor `min_lr`.
    Each group decays from its own peak (group 'lr') down to `min_lr`."""
    from torch.optim.lr_scheduler import LambdaLR

    def make(peak):
        def f(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            prog = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
            cos = 0.5 * (1.0 + math.cos(math.pi * prog))
            return (min_lr + (peak - min_lr) * cos) / peak    # absolute floor min_lr
        return f

    return LambdaLR(optimizer, [make(g["lr"]) for g in optimizer.param_groups])


class EncoderDecoderTrainer(Trainer):
    """Trainer that builds the warmup->cosine-to-min_lr scheduler with the step
    count Trainer computes (so we don't reason about DDP/epochs ourselves).

    STAGE-1 ONLY — trains with the model held in ``.eval()`` mode
    =========================================================================
    In stage-1 the image encoder and the pretrained decoder layers are FROZEN and
    only the newly-added adapters (cross-attention + zero-init gates + the 384->512
    bridge) train. We do NOT want the frozen backbone's dropout / stochastic-depth
    (DropPath) injecting noise into the features the adapters learn from, so every
    TRAINING forward is run in eval mode (see ``compute_loss``).

    HF ``Trainer.training_step`` calls ``model.train()` at the start of every step,
    so a one-off ``model.eval()`` before ``trainer.train()`` would not stick; forcing
    it inside ``compute_loss`` (right before the forward) is what makes it hold. Note
    ``eval()`` only flips the dropout/DropPath flags — it does NOT stop gradients, so
    the adapters still train normally.

    This ALSO disables dropout inside the trainable cross-attention adapters, which is
    acceptable for stage-1 adapter warm-up.

    ⚠️  CHANGE FOR FULL / STAGE-2 TRAINING: once the backbone is unfrozen and trained
    end-to-end, set ``FORCE_EVAL_DURING_TRAIN = False`` (or drop the ``compute_loss``
    override) so dropout / stochastic depth are active again for regularization.
    """

    # Stage-1 default: run the training forward in eval mode. Flip to False for
    # full / end-to-end (stage-2) training where dropout should be ON.
    FORCE_EVAL_DURING_TRAIN = True

    def __init__(self, *args, warmup_steps=1500, min_lr=1e-5, **kwargs):
        self._warmup_steps = warmup_steps
        self._min_lr = min_lr
        super().__init__(*args, **kwargs)

    def create_scheduler(self, num_training_steps, optimizer=None):
        if self.lr_scheduler is None:
            self.lr_scheduler = build_scheduler(
                optimizer or self.optimizer, self._warmup_steps, num_training_steps, self._min_lr)
        return self.lr_scheduler

    def compute_loss(self, model, inputs, *args, **kwargs):
        # STAGE-1: keep the frozen backbone (and, acceptably, the adapters) deterministic
        # for the training forward — dropout / DropPath off. Trainer.training_step has
        # already called model.train(); we override it here, per step, right before the
        # forward. Gradients still flow, so the adapter layers train.
        # ⚠️ Remove / gate off (FORCE_EVAL_DURING_TRAIN=False) for full/stage-2 training.
        if self.FORCE_EVAL_DURING_TRAIN:
            model.eval()
        return super().compute_loss(model, inputs, *args, **kwargs)


class GradNormAlert(TrainerCallback):
    """PORTED from train_ctc_encoder.py: alert on grad-norm spikes > threshold."""

    def __init__(self, threshold=10.0):
        self.threshold = threshold

    def on_log(self, args, state, control, logs=None, **kwargs):
        gn = (logs or {}).get("grad_norm")
        if gn is not None and gn > self.threshold:
            print(f"[ALERT] grad_norm={gn:.2f} > {self.threshold} at step {state.global_step}")


# ============================================================================
# Eval slices — PORTED from train_ctc_encoder.py, adapted to the encoder-decoder
# data format (each item is {pixel_values, input_ids} for OCRCollator).
# ============================================================================
class ListEvalDataset(torch.utils.data.Dataset):
    """In-memory map-style eval dataset over raw row dicts. Applies the (no-aug)
    eval preprocessing per item and produces the encoder-decoder input format."""

    def __init__(self, rows, preprocessor, tokenizer, image_col, text_col):
        self.rows = rows
        self.prep = preprocessor       # clean ImagePreprocessor (augment_fn=None)
        self.tok = tokenizer
        self.image_col = image_col
        self.text_col = text_col

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return {
            "pixel_values": self.prep(r[self.image_col]),          # (1, H, W_i)
            "input_ids": self.tok.encode(r[self.text_col]),        # BOS ... EOS
        }


def build_eval_slices(rows, image_col, text_col, source_col, slice_cap, seed):
    """Partition materialized eval rows by TEXT SOURCE (natural / random) when the
    source column is present; otherwise a single 'all' slice. Returns a dict
    ``{slice_name: [raw_row, ...]}`` (raw rows, for the generate-based CER).

    (Width-based short/long slicing was dropped: two source slices keep the total
    generation count small enough for the rank-0-only CER eval.)"""
    has_src = bool(rows) and source_col in rows[0]

    def subset(predicate):
        idx = [i for i in range(len(rows)) if predicate(i)]
        if not idx:
            return None
        if slice_cap and len(idx) > slice_cap:
            g = torch.Generator().manual_seed(seed)
            idx = [idx[j] for j in torch.randperm(len(idx), generator=g)[:slice_cap].tolist()]
        return [rows[i] for i in idx]

    slices = {}
    if has_src:
        for s in ("natural", "random"):
            sub = subset(lambda i, ss=s: rows[i].get(source_col) == ss)
            if sub is not None:
                slices[s] = sub
    else:
        sub = subset(lambda i: True)
        if sub is not None:
            slices["all"] = sub
    print(f"[eval] slices: { {k: len(v) for k, v in slices.items()} }")
    return slices


def find_last_checkpoint(output_dir, prev_run_dir):
    for d in (output_dir, prev_run_dir):
        if d and os.path.isdir(d):
            ckpt = get_last_checkpoint(d)
            if ckpt is not None:
                return ckpt
    return None


if __name__ == '__main__':

    # DDP topology sanity: this line prints ONCE PER PROCESS running the script as
    # __main__ (i.e. per rank). If you see it more times than --nproc_per_node, extra
    # processes are running. Read the fields to tell WHICH:
    #   * The 2 real ranks share one ppid (the torchrun agent) and one MASTER_PORT,
    #     with LOCAL_RANK 0 and 1.
    #   * A straggler from an EARLIER torchrun has a DIFFERENT MASTER_PORT (and usually
    #     ppid=1, reparented to init after its launcher died) -> kill it / restart kernel.
    print(f"[proc] pid={os.getpid()} ppid={os.getppid()} "
          f"RANK={os.environ.get('RANK')} LOCAL_RANK={os.environ.get('LOCAL_RANK')} "
          f"WORLD_SIZE={os.environ.get('WORLD_SIZE')} MASTER_PORT={os.environ.get('MASTER_PORT')}",
          flush=True)

    tokenizer = TeluguGraphemeTokenizer(
        vocab_file="/kaggle/input/datasets/harshadesaraju1999/telugu-tokenizer-vocab/telugu-vocab.json")
    print(tokenizer.pad_token_id)

    # -------------- Step-1: Load the pretrained models --------------
    # Pretrained text decoder model
    pretrained_config = GPTConfig(
        vocab_size=len(tokenizer),
        embed_dim=512,
        hidden_dim=1368,  # 2.67 * 512 = 2/3 * 4 * hidden_dim
        num_heads=8,
        num_layers=16,
        ctx_len=256,
        dropout=0.1
    )

    pretrained_text_model = GPTModel(pretrained_config, pad_index=tokenizer.pad_token_id)
    pretrained_text_model.load_state_dict(
        torch.load("/kaggle/input/models/harshadesaraju1999/telugugpt/pytorch/default/1/final_model.pt")
    )

    # Load the pretrained CTC image encoder (config must match the checkpoint).
    img_encoder_cfg = CTCEncoderConfig(
        max_image_width=2048,
        max_frames=256,
    )

    pretrained_encoder_model = ImageEncoderCTC(img_encoder_cfg)
    pretrained_encoder_model.load_state_dict(
        torch.load(
            '/kaggle/input/models/harshadesaraju1999/telugu-ctc-image-encoder/pytorch/default/1/final_model.pt',
        )
    )

    # -------------- Step-2: Initialize the new encoder-decoder model --------------
    model = EncoderDecoder(
        encoder_config=img_encoder_cfg,
        decoder_config=pretrained_config,
        pad_index=tokenizer.pad_token_id
    )

    # -------------- Step-3: Replace the random weights with pretrained weights --------------
    # Replace the text decoder weights
    match_result = model.decoder_model.load_state_dict(
        pretrained_text_model.state_dict(),
        strict=False
    )

    # Replace the image encoder weights (full CTC encoder, incl. the CTC head we won't use)
    model.encoder_model.load_state_dict(pretrained_encoder_model.state_dict())

    # -------------- Step-4: Verify the weight transfer for decoder model --------------
    # Verify the layers not matched are the newly added layers (cross-attention +
    # its layer norm + the zero-init tanh gate; present on even blocks only).
    newly_added_layers = ['cross_attention', 'layer_norm1_5', 'cross_attn_gate']
    all_matched = True
    for layer_name in match_result.missing_keys:
        is_new = any([new_layer in layer_name for new_layer in newly_added_layers])
        if not is_new:
            all_matched = False

    assert all_matched, "Some of the old layers' weights did not match"

    # -------------- Step-4.5: image contribution starts at zero --------------
    # The zero-init tanh gate inside each cross-attention block (cross_attn_gate,
    # tanh(0) == 0) already makes the image contribute NOTHING at init, so the model
    # starts EXACTLY as the pretrained LM and phases the image in gradually
    # (Flamingo/adapter style). No manual out_proj zeroing is needed anymore, and the
    # old per-block loop would AttributeError on the odd blocks that have no cross-attn.

    # -------------- Step-5: Freeze the weights --------------
    # Decoder
    # Now, freeze the old pretrained layers and allow only new layers to train
    for name, params in model.decoder_model.named_parameters():
        if name not in match_result.missing_keys:
            params.requires_grad = False

    # Do some sanity checks
    # 1 - Check if the newly added layers are the unfrozen layers
    unfrozen_layers = []
    for name, params in model.decoder_model.named_parameters():
        if params.requires_grad:
            unfrozen_layers.append(name)

    # 2 - Check the missing_keys layers and unfrozen layers match
    all_matched = True
    for name1, name2 in zip(match_result.missing_keys, unfrozen_layers):
        if name1 != name2:
            print(f"{name1} --- {name2}", flush=True)
            all_matched = False

    assert all_matched, "Some pretrained layers are not frozen!"

    # Encoder
    # Load and freeze the image encoder also
    for name, parameters in model.encoder_model.named_parameters():
        parameters.requires_grad = False

    # No need for any checks for encoder as the model configuration is the same

    # -------------- Step-6: Sanity check on the unfrozen layers --------------
    trainable_layers = []
    for name, params in model.named_parameters():
        if params.requires_grad:
            trainable_layers.append(name)

    # print("List of trainable layers in the model:")
    # for name in trainable_layers:
    #     print(name)

    # -------------- Print model sizes --------------
    encoder_params = 0
    for params in model.encoder_model.parameters():
        encoder_params += params.numel()
    print(f"No. of parameters in encoder: {encoder_params}")

    decoder_params = 0
    for params in model.decoder_model.parameters():
        decoder_params += params.numel()
    print(f"No. of parameters in decoder: {decoder_params}")

    trainable_params = 0
    for params in model.parameters():
        if params.requires_grad:
            trainable_params += params.numel()
    print(f"No. of trainable parameters: {trainable_params}")
    print(f"Percentage of trainable parameters: {((trainable_params / (encoder_params + decoder_params)) * 100):.2f}%")

    # ---------------------------- Data config (PORTED: same datasets as train_ctc_encoder.py) ----
    DATASET_NAME1 = "harsha-desaraju/sample-dataset-new"          # synthetic (has text_source)
    DATASET_NAME2 = "harsha-desaraju/telugu-pdf-line-image-text"  # real PDF-line crops
    DATASET_SPLIT = "train"          # NORMAL (downloaded, map-style) dataset
    EVAL_SPLIT = "validation"        # small held-out split for the eval slices
    IMAGE_COLUMN = "image"
    TEXT_COLUMN = "text"
    SOURCE_COLUMN = "text_source"    # natural / random -> eval slices (optional column)
    TRAIN_CONFIGS = ['train_0000', 'train_0001', 'train_0002', 'train_0003',
                     #'train_0004', 'train_0005', 'train_0006', 'train_0007',
                     ]
    RND_SAM_FRAC = 0.05

    # frames / widths
    DOWNSAMPLE = 8                   # conv-stem width reduction (T = W // DOWNSAMPLE)
    IMAGE_HEIGHT = 64
    MAX_IMAGE_WIDTH = 2048           # matches the CTC encoder (max_frames=256)

    # batching — T4 (16 GB): stage-1 trains only cross-attn + gates + bridge, but the
    # full 16-layer decoder forward is retained for backprop, so keep the per-device
    # batch modest and recover global batch via grad-accum. Raise if you have headroom.
    PER_DEVICE_BATCH = 32
    GRAD_ACCUM = 2                   # global batch = PER_DEVICE_BATCH * GRAD_ACCUM * #GPUs
    EVAL_BATCH = 32                  # smaller (long slices can be up to 2048px)

    # optimizer / schedule (PORTED: single-tier build_optimizer + warmup -> cosine to min_lr)
    LR = 1e-4                        # single LR tier (matches train_ctc_encoder.py)
    MIN_LR = 1e-5
    WARMUP_STEPS = 2000
    WEIGHT_DECAY = 0.05
    BETAS = (0.9, 0.98)
    EPOCHS = 4

    # eval / io
    EVAL_SLICE_CAP = 1500            # cap each eval slice for speed
    EVAL_LOSS_CAP = 512              # rows used for the (teacher-forced) eval_loss set
    SAVE_EVAL_STEPS = 1000
    OUTPUT_DIR = "/kaggle/working/telugu-ocr-stage1"
    PREV_RUN_DIR = "/kaggle/input/models/harshadesaraju1999/telugu-ocr-checkpoint/transformers/default/1"              # prior run dir re-mounted read-only, for 12h resume

    SEED = 42
    set_seed(SEED)                   # reproducible augmentation (random + np.random)

    # ---- Preprocessors: train augmented (composed degrade pipeline), eval clean ----
    train_preprocessor = ImagePreprocessor(
        IMAGE_HEIGHT, MAX_IMAGE_WIDTH, DOWNSAMPLE, augment_fn=make_augmenter())
    eval_preprocessor = ImagePreprocessor(IMAGE_HEIGHT, MAX_IMAGE_WIDTH, DOWNSAMPLE)


    def make_sample_transformer(preprocessor):
        def sample_transformer(batch):
            images = [preprocessor(img) for img in batch[IMAGE_COLUMN]]     # list of (1, H, W_i)
            input_ids = [tokenizer.encode(t) for t in batch[TEXT_COLUMN]]   # BOS ... EOS
            return {"pixel_values": images, "input_ids": input_ids}
        return sample_transformer


    # ---- Train: synthetic configs (DATASET_NAME1) + real PDF-line crops (DATASET_NAME2) ----
    train_parts = [
        load_dataset(DATASET_NAME1, c, columns=[IMAGE_COLUMN, TEXT_COLUMN, SOURCE_COLUMN])[DATASET_SPLIT]
        for c in TRAIN_CONFIGS
    ]
    # 2nd dataset — the real images; tag them all as 'natural' for the source column.
    real_ds = load_dataset(DATASET_NAME2, split=DATASET_SPLIT, columns=[IMAGE_COLUMN, TEXT_COLUMN])
    real_ds = real_ds.add_column(SOURCE_COLUMN, ['natural'] * len(real_ds))
    train_parts.append(real_ds)

    train_hf = concatenate_datasets(train_parts, axis=0)

    # Make the random samples only a given percentage
    indices = defaultdict(list)

    for i, value in enumerate(train_hf["text_source"]):
        indices[value].append(i)

    nat_ds = train_hf.select(indices["natural"])
    rnd_ds = train_hf.select(indices["random"])

    num_rnd = int((RND_SAM_FRAC/(1+RND_SAM_FRAC))*len(nat_ds))
    rnd_ds = rnd_ds.shuffle(seed=SEED).select(range(num_rnd))

    train_hf = concatenate_datasets([nat_ds, rnd_ds], axis=0)
    train_hf = train_hf.shuffle(seed=SEED)

    del nat_ds, rnd_ds
    gc.collect()

    train_dataset = train_hf.with_transform(make_sample_transformer(train_preprocessor))
    print(f"[data] train -> {len(train_hf)} rows; cols {train_hf.column_names}")

    # ---- Validation: materialize rows, build width/source eval slices ----
    val_ds = load_dataset(DATASET_NAME1, EVAL_SPLIT)[EVAL_SPLIT]
    val_rows = list(val_ds)
    print(f"[data] validation -> {len(val_rows)} rows")
    eval_slices = build_eval_slices(val_rows, IMAGE_COLUMN, TEXT_COLUMN, SOURCE_COLUMN,
                                    EVAL_SLICE_CAP, SEED)

    # A single small transformed eval set drives Trainer's eval_loss and fires
    # on_evaluate exactly ONCE; per-slice generate-based CER is added by CEREvalCallback.
    eval_ds = ListEvalDataset(val_rows[:EVAL_LOSS_CAP], eval_preprocessor, tokenizer,
                              IMAGE_COLUMN, TEXT_COLUMN)

    data_collator = OCRCollator(pad_token_id=tokenizer.pad_token_id, downsample=DOWNSAMPLE)

    # ---- Optimizer (fresh, single LR tier); scheduler built by EncoderDecoderTrainer ----
    optimizer = build_optimizer(model, LR, WEIGHT_DECAY, BETAS)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=EVAL_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_grad_norm=1.0,                       # PORTED: grad clip
        # NCCL collective timeout (default 1800s = 30min). The rank-0-only CER eval makes
        # the other ranks idle-wait at the next barrier; raise this so a slow eval / save
        # can never trip the watchdog. This is the timeout that killed the 2-GPU run.
        ddp_timeout=7200,                        # 2h

        # Optimizer + LR are supplied explicitly via `optimizers=` below (build_optimizer),
        # and the warmup->cosine-to-min_lr scheduler is built by create_scheduler, so no
        # optim / learning_rate / weight_decay / betas / lr_scheduler_type are set here.

        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        ignore_data_skip=True,                   # fast resume (don't replay skipped data)

        fp16=torch.cuda.is_available(),          # T4 -> fp16 + GradScaler
        gradient_checkpointing=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=4,
        dataloader_persistent_workers=True,

        eval_strategy="steps",
        eval_steps=SAVE_EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_EVAL_STEPS,
        save_total_limit=2,
        load_best_model_at_end=False,            # PORTED: no best-model tracking / no early stopping

        logging_strategy="steps",
        logging_steps=50,
        logging_first_step=True,
        report_to="wandb",
        run_name="stage-1-finetuning",
        seed=SEED,
    )

    trainer = EncoderDecoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_ds,                    # drives eval_loss (fires on_evaluate once)
        data_collator=data_collator,
        optimizers=(optimizer, None),            # scheduler built by create_scheduler
        callbacks=[
            # Per-slice generate-based CER -> eval_<slice>_cer + macro eval_cer.
            CEREvalCallback(model, eval_slices, eval_preprocessor, tokenizer,
                            image_col=IMAGE_COLUMN, text_col=TEXT_COLUMN),
            GradNormAlert(threshold=10.0),
        ],
        warmup_steps=WARMUP_STEPS,
        min_lr=MIN_LR,
    )

    # ---- Resume: this run's checkpoints, else a prior-run dir (PORTED) ----
    last_ckpt = find_last_checkpoint(OUTPUT_DIR, PREV_RUN_DIR)
    print(f"Resuming from: {last_ckpt}" if last_ckpt else "Starting fresh (no checkpoint found)")

    trainer.train(resume_from_checkpoint=last_ckpt)

    # Save final artifacts (HF checkpoint + stage-1-style .pt)
    trainer.save_model(OUTPUT_DIR)
    if trainer.is_world_process_zero():
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "final_model.pt"))
        print("Finished stage-1 training!")