"""Degradation pipeline that makes synthetic Telugu line crops look like real PDF scans.

WHY THIS EXISTS
    The model trained on clean synthetic renders reads clean text well and falls apart on
    real book/PDF crops. The parameters here were calibrated by measuring the real crops
    in `data/actual_samples` against a clean synthetic render at the height the model
    actually sees (64px), then tuning until the augmented distribution *contains* the real
    one rather than merely averaging to it.

    Where it stands (600 draws vs the 10 real crops, per-metric samples covered by the
    augmented 5-95 band): sharpness 10/10, background grain 10/10, stroke width 9/10,
    component density 9/10, ink darkness 8/10, background variance 7/10, contrast 7/10,
    background mean 2/10. Overall 78%.

    Note these eight are all GLOBAL statistics, so they are close to blind to localized
    effects -- ink_dropout barely moves them while changing the image a lot. Do not treat
    the percentage as the whole picture; look at the preview sheet too.

    The background mean is the one that is genuinely short: the real crops sit at 246.8-
    251.7 (a five-level spread) while this pipeline centres near 242. That is partly a
    property of the reference set -- these ten are PDF-viewer screenshots, whose paper has
    already been auto-brightened -- and slightly greyer paper is harmless headroom, so it
    is left rather than over-fitted to ten samples. If you later collect crops rendered
    straight from the PDFs at full resolution, re-measure before trusting these numbers.

WHAT THE REAL CROPS ACTUALLY DIFFER BY (measured, at h=64)
    1. Resolution loss. Real crops are downsampled page scans, so glyph edges arrive
       soft; a synthetic render is pixel-exact. This is the single biggest gap.
    2. Stroke weight. Scanned book ink runs heavier (stroke width 2.39-2.98 vs 2.38
       clean). Nothing in the old pipeline thickened ink.
    3. Neighbouring-line intrusion. 6 of the 10 real samples carry a fragment of the line
       above/below, or a ruled line, inside the crop. The model has never seen this and
       tries to read the fragments as glyphs.
    4. Localized ink dropout. Individual strokes are partly missing while their neighbours
       are solid -- the loop of a `ర` broken open next to a solid `ప`, one end of a line
       heavy and the other fragmented. Global thinning cannot produce this because it
       thins every character equally; see `ink_dropout`.
    Everything else (grime, speckle, bleed-through, tone drift) is secondary.

ORDER IS LOAD-BEARING
    bilevel -> ink -> paper/grime -> tone -> crop artefacts -> geometry -> noise
        -> resample -> ink-growth backstop -> compression -> auto-levels
    * Binarisation goes FIRST, not in the tone stage. A bilevel source PDF was thresholded
      at print/scan time, before the ink spread and the resample. Thresholding ink that has
      already been thickened fuses the strokes into a solid blob, and no amount of eroding
      recovers the counters because the threshold threw that information away.
    * Noise goes BEFORE the resample. The real crops have very little per-pixel grain
      (high-freq residual 0.86-1.51); noise added after a resample measures 4-15 and
      looks like video static rather than paper. Resampled grain measures ~0.97.
    * Auto-levels goes LAST. Stacked gamma/contrast/texture leaves the paper mid-grey,
      which no real crop shows (all ten have a background mean of 247-252).

DELIBERATELY NOT USED
    LowInkRandomLines / LowInkPeriodicLines (page-scale streaks shatter 64px glyphs into
    dotted outlines), Hollow (hollows glyphs out entirely), DirtyRollers (one full-width
    brightness gradient, not a scan artefact), InkBleed with kernel_size >= 7 (glyphs
    merge into a blob). These cannot be fixed by changing their numbers at this scale.

LEGIBILITY IS A HARD CONSTRAINT
    A Telugu akshara whose counters close up is a WRONG LABEL, not a hard training
    example, so the per-effect ceilings below (dilate +1.5, InkBleed severity 0.5, blur
    sigma 1.5) are limits rather than suggestions.

    Per-effect ceilings are not sufficient on their own: stacked, three individually safe
    effects still produced solid blobs in the worst ~1% of draws. `_limit_ink_change` is
    the two-sided backstop that catches those AND the opposite failure, where dropout plus
    fading erases so much of a glyph that it is no longer identifiable. It is what lets the
    probabilities here stay aggressive. If you raise any ceiling, re-check BOTH tails (sort
    a few hundred draws by ink fraction and look at each end) rather than trusting averages.

USAGE
    augment = make_augmenter(p_clean=0.2)      # build once, reuse per sample
    out = augment(img)                         # HxW or HxWx3 uint8 -> same shape

    Call it on the RESIZED 64px-tall crop, not the full-resolution render: the
    scale-dependent parameters (resample scale, blur sigma, band widths, speckle size)
    are all tuned for that height. This matches the ImagePreprocessor in
    src/encoder_decoder/train_stage_2.py, which augments after the resize.

Verified against augraphy 8.2.6 and OpenCV 5.0. Constructor signatures drift between
augraphy releases; re-check the parameter names if you upgrade.
"""

import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

from augraphy import (
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


# ======================================================================================
# Reproducibility
# ======================================================================================
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

# Exposed so the ladders can be swept one augmentation at a time (see __main__).
LADDERS = {
    "InkBleed": _INK_BLEED,
    "InkMottling": _INK_MOTTLING,
    "Letterpress": _LETTERPRESS,
    "BleedThrough": _BLEED_THROUGH,
    "NoiseTexturize": _NOISE_TEXTURIZE,
    "BrightnessTexturize": _BRIGHTNESS_TEXTURIZE,
    "DirtyDrum": _DIRTY_DRUM,
    "BadPhotoCopy": _BAD_PHOTOCOPY,
    "SubtleNoise": _SUBTLE_NOISE,
    "Jpeg": _JPEG,
}


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
    if _scaled_p(0.06, s):
        g = _safe(_pick(_LETTERPRESS, s), g)          # rare: ~79ms/call
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
    if _scaled_p(0.05, s):
        g = _safe(_pick(_BAD_PHOTOCOPY, s), g)        # rare: ~33ms/call
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


def make_augmenter(p_clean=0.20, severity=None):
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


# ======================================================================================
# Preview: augmented synthetic crops next to the real ones they are imitating.
# ======================================================================================
if __name__ == "__main__":
    from pathlib import Path

    SYN_IMAGE = "/Users/xai/Personal/Projects/TeluguOCR/data/syn_img.jpg"
    REAL_DIR = "/Users/xai/Personal/Projects/TeluguOCR/data/actual_samples"
    OUT_PATH = "/Users/xai/Personal/Projects/TeluguOCR/misc/augmentation_preview.png"
    N_SAMPLES = 8          # augmented variants to draw
    PANEL_WIDTH = 1000     # px; crops are cropped/padded to this for a tidy stack
    SEED = None

    set_seed(SEED)

    def to_h64(g):
        h, w = g.shape[:2]
        if h == 64:
            return g
        return cv2.resize(g, (max(8, int(round(w * 64 / h))), 64),
                          interpolation=cv2.INTER_AREA)

    def panel(g, tag):
        g = to_h64(g)
        if g.shape[1] >= PANEL_WIDTH:
            g = g[:, :PANEL_WIDTH]
        else:
            g = np.hstack([g, np.full((64, PANEL_WIDTH - g.shape[1]), 255, np.uint8)])
        label = np.full((18, PANEL_WIDTH), 240, np.uint8)
        cv2.putText(label, tag, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, 0, 1)
        return np.vstack([label, g, np.full((3, PANEL_WIDTH), 150, np.uint8)])

    syn = cv2.imread(SYN_IMAGE, cv2.IMREAD_GRAYSCALE)
    if syn is None:
        raise FileNotFoundError(f"could not read {SYN_IMAGE}")
    syn = to_h64(syn)

    rows = [panel(syn, "[CLEAN SYNTHETIC INPUT]")]

    reals = sorted(p for p in Path(REAL_DIR).iterdir()
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    for p in reals[:4]:
        real = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if real is not None:
            rows.append(panel(real, f"[REAL]  {p.name}"))

    for i in range(N_SAMPLES):
        sev = i / max(1, N_SAMPLES - 1)      # walk the range so the sheet shows both ends
        rows.append(panel(degrade(syn, sev), f"[AUGMENTED]  severity {sev:.2f}"))

    sheet = np.vstack(rows)
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    plt.imshow(sheet, cmap='gray')
    plt.show()

    # cv2.imwrite(OUT_PATH, sheet)
    # print(f"wrote {OUT_PATH}  ({sheet.shape[1]}x{sheet.shape[0]})")
