"""Consensus pseudo-labelling for real PDF line crops.

Runs three independent OCR engines over a HuggingFace dataset of Telugu text-line images
and tiers every sample by how much the engines agree, so real (non-synthetic) crops can be
turned into training data.

    Tier 1  all three agree                                  -> trust outright
    Tier 2  Tesseract + Paddle agree, our model disagrees     -> two-engine consensus
    Tier 3  Paddle + our model agree, Tesseract disagrees     -> two-engine consensus
    Tier 4  our model + Tesseract agree and Paddle disagrees, or all three differ
    Tier 0  not tierable (an engine errored, or every prediction was empty)

Tier 0 is not in the original spec but is unavoidable: with fewer than three usable
predictions there is nothing to take a majority over, and silently folding those into a
disagreement tier would pad the review pile with rows that carry no signal at all.

STOPPING AND RESUMING
    Results are streamed to PROGRESS_JSONL a chunk at a time and fsynced, so you can stop
    the run at any point -- Ctrl-C, kill, power cut -- and lose at most the chunk in
    flight. Ctrl-C is caught, so the dataset still gets written from everything recorded so
    far. Re-running the same command skips the rows already in that file and carries on;
    the resume is keyed on row index, so changing CHUNK_SIZE between runs is fine. Delete
    the file (or set RESUME=False) to start from scratch.

    A partial run writes a partial dataset: OUTPUT_DIR then holds only the rows completed
    so far, in source order, rather than the whole input.

OUTPUTS
    One dataset, at OUTPUT_DIR: the input rows unchanged, plus all three engines' raw OCR
    output (pred_model / pred_tesseract / pred_paddle), paddle_score, tier, tier_reason,
    consensus_text, has_english, english_frac, and the three pairwise disagreement scores.
    Plus STATS_JSON, the tier/engine/disagreement summary that print_stats renders.

WHAT THIS DOES NOT DO
    It assigns tiers and records evidence. It does not decide what to train on, filter
    anything out, or write per-tier review files -- every row of the input comes back with
    its tier, all three raw predictions, the consensus text where one exists, and the
    pairwise CERs. Downstream use is your call.

TWO CORRECTNESS TRAPS, BOTH HANDLED
    * Empty predictions must not count as agreement. Tesseract and Paddle both returning
      "" on an unreadable crop is a shared failure, not a consensus, and would otherwise
      manufacture a large pile of Tier-1 rows labelled with the empty string -- the single
      worst thing you could add to an OCR training set.
    * Agreement is compared on NORMALIZED text (NFKC, zero-width stripped, punctuation
      folded to one spelling, whitespace collapsed and trimmed -- see `normalize`). Telugu
      is full of sequences that are visually identical but differ in codepoint order,
      composition or invisible joiners, and the engines disagree on dash and quote
      characters for identical glyphs, so raw string equality understates agreement badly.
      The raw predictions are kept alongside so nothing is lost.

THERE IS NO GROUND TRUTH HERE
    Nothing in this script measures accuracy, because at labelling time no correct answer
    exists -- that is the whole point of the exercise. The `disagree_*` columns are
    PAIRWISE DISAGREEMENT between two engines' outputs (symmetric grapheme edit distance
    over the longer of the two), not error rates. 0 = the two engines produced identical
    text, 1 = they share nothing. Two engines can agree perfectly on a reading that is
    wrong, so low disagreement means "corroborated", not "correct".

    A true `cer(ref, hyp)` is also provided but is NOT used for tiering; it is there for
    later, once you have ground truth (e.g. evaluating a model against the Tier-1 labels
    this script produces).

    Both work over GRAPHEME CLUSTERS rather than codepoints, matching the grapheme
    tokenizer in src/telugu_ocr/tokenizer: one visual Telugu akshara is
    routinely three or four codepoints, so a codepoint-level distance overstates the
    difference severalfold.

EFFICIENCY
    Parallelism is per-engine rather than per-sample, because the three engines want
    completely different treatment:
      * our CTC model  -- one batched GPU pass, width-bucketed so padding is minimal
      * Tesseract      -- a thread pool; pytesseract shells out to a binary, so the GIL is
                          released and threads give real concurrency without pickling images
      * PaddleOCR      -- its own internal batching, one instance (its predictors are not
                          thread-safe)
    Engine passes then run CONCURRENTLY with each other. CPU engines always run at once.
    Two GPU engines are serialized against each other ONLY when they share a device: the
    lock is keyed on the resolved device (see `_gpu_lock_key`), so on one GPU the model and
    Paddle take turns, while on a 2-GPU box with the model on cuda:1 and Paddle on gpu:0
    they run fully in parallel. Either way Tesseract works through a chunk on the CPU while
    the GPU engines work through it on the GPU. Work is chunked so peak memory stays
    bounded regardless of dataset size.

USAGE
    Configure the inline block at the bottom and run:
        python3 -m pipelines.label.consensus_labelling

    Every engine is optional and imported lazily; a missing one is skipped with a warning
    (and then no row can be tiered, so they all land in Tier 0). Dependencies:
        uv add paddleocr paddlepaddle rapidfuzz
    rapidfuzz only speeds up the edit-distance computation; there is a pure-Python fallback.
    tqdm drives the progress bar / ETA and also degrades to periodic prints if absent.

    Verified working against paddleocr 3.7.0 / paddle 3.3.1, tesseract 5.x with `tel`,
    and the CTC checkpoint at models/image_encoder/ctc_encoder_stage-2/.

======================================================================================
RUNNING ON KAGGLE (2x T4)
======================================================================================
Set REPO_ROOT below to wherever you attached this repo, and in the config block set
PADDLE_DEVICE / MODEL_DEVICE to put the two GPU engines on DIFFERENT T4s -- then they
run concurrently instead of taking turns (see `_gpu_lock_key`):

    PADDLE_DEVICE = "gpu:0"      # paddleocr on the first T4
    MODEL_DEVICE  = "cuda:1"     # our torch model on the second T4
    TESSERACT_THREADS = 4        # Kaggle gives ~4 CPU cores

Attach as Kaggle Datasets (Internet may stay OFF for these two): this repo -> REPO_ROOT,
and the checkpoint -> CHECKPOINT. Internet ON is still needed to load the INPUT dataset
from the Hub and to push. To go fully offline, attach the input dataset too and point
INPUT_REPO at its path (load_from_disk is used automatically for an existing path).

Install cells, in this order. The order and the pins BOTH matter -- this sequence is the
fix for three failures that each look like something else:

    # 1. Paddle GPU build. `paddle` on plain PyPI is an UNRELATED package -- never
    #    install it. And `pip install paddlepaddle-gpu` off plain PyPI grabs a STALE GPU
    #    build missing symbols current paddleocr/paddlex import, which surfaces as
    #        cannot import name 'forward_complete_op_role' from paddle.distributed.passes
    #    GPU wheels must come from Paddle's OWN index, and exactly ONE paddle build may
    #    be installed. cu126 runs on any CUDA 12.x driver; for a CUDA 11.x image use
    #    cu118 + paddlepaddle-gpu==3.2.0.
    !pip uninstall -y -q paddle paddlepaddle paddlepaddle-gpu paddleocr paddlex
    !pip install -q "paddlepaddle-gpu==3.3.0" -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

    # 2. PaddleOCR + deps.
    !pip install -q "paddleocr>=3.0,<4" datasets tqdm rapidfuzz

    # 3. Repair torch's NCCL, which step 1 downgraded. paddlepaddle-gpu pins an older
    #    nvidia-nccl-cu12 (e.g. 2.25.1) than Kaggle's prebuilt torch needs (2.27.3+,
    #    which added `ncclCommShrink`), so `import torch` then dies with
    #        libtorch_cuda.so: undefined symbol: ncclCommShrink
    #    Restore the version THIS torch was built against; newer NCCL is backward
    #    compatible so paddle keeps working. Reading torch's metadata does not import
    #    torch, so this works while torch is broken.
    import re, sys, subprocess, importlib.metadata as md
    want = "2.27.3"
    for r in (md.requires("torch") or []):
        if r.startswith("nvidia-nccl-cu12"):
            m = re.search(r"==\\s*([\\d.]+)", r)
            if m: want = m.group(1)
            break
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    f"nvidia-nccl-cu12=={want}"], check=True)

    # 4. Tesseract needs its BINARY plus Telugu traineddata, not just the pip wheel.
    !apt-get -qq install -y tesseract-ocr tesseract-ocr-tel

Then verify both GPU stacks import before starting a long run (`import paddle` then
`import torch`, checking device_count on each), and add your token as a Kaggle secret
named HF_TOKEN (Add-ons -> Secrets) so the push can authenticate.
"""

from __future__ import annotations

import sys
import json
import os
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import regex
from PIL import Image
from src.telugu_ocr.data.preprocess import resize_line_image

# `src...` must be importable: the engines load the model and tokenizer from the repo.
# Locally that is this file's own repo root; on Kaggle the repo is attached as a Dataset,
# so point REPO_ROOT at the directory that CONTAINS `src/`, e.g.
#   REPO_ROOT = "/kaggle/input/telugu-ocr-repo/TeluguOCR"
REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Pin CUDA ordinal ordering so paddle's "gpu:N" and torch's "cuda:N" name the SAME
# physical device. Must be set before any CUDA context exists (i.e. before the engines
# import torch / paddle). Harmless on identical GPUs; cheap insurance regardless -- and
# load-bearing on a 2-GPU box where the two engines are deliberately split across devices.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

# ======================================================================================
# Text normalization, English detection, grapheme-level CER
# ======================================================================================

_GRAPHEME_RE = regex.compile(r"\X")
_WS_RE = re.compile(r"\s+")
_LATIN_RE = re.compile(r"[A-Za-z]")
# Punctuation that OCR engines disagree on constantly without the reading changing:
# quote style, dash width, stray full stops at a line end.
_LOOSE_PUNCT_RE = re.compile(r"[‘’“”'\"`‐-―\-·.,;:!?()\[\]{}]")
# Zero-width: ZWSP, ZWNJ, ZWJ, word-joiner, BOM. Tesseract emits ZWNJ inside Telugu
# conjuncts; the other two engines do not.
_ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")

# Punctuation the engines pick differently for the SAME glyph. NFKC does not help here --
# dashes and curly quotes are distinct Unicode characters, not compatibility variants, so
# NFKC leaves U+2013 EN DASH and U+201C alone (it only folds the fullwidth/small forms).
# Measured over 300 real rows: Paddle emits U+2013 where our model and Tesseract emit
# ASCII "-" ("కాళహస్తి–517" vs "కాళహస్తి-517", "08578–222543" vs "08578-222543"), and
# Tesseract emits curly quotes (26 occurrences) where the others do not. Both are pure
# character-choice noise on identical glyphs.
# These FOLD rather than delete, so "300-00" keeps its hyphen -- deleting it (which is what
# strip_punct does) would merge the digits and change the reading.
_DASH_CHARS = "\u2010\u2011\u2012\u2013\u2014\u2015\u2043\u2212\ufe58\ufe63\uff0d"
_SINGLE_QUOTE_CHARS = "\u2018\u2019\u201a\u201b\u2032\u00b4\u02bc"
_DOUBLE_QUOTE_CHARS = "\u201c\u201d\u201e\u201f\u2033\u00ab\u00bb"
_PUNCT_FOLD = str.maketrans(
    {**{c: "-" for c in _DASH_CHARS},
     **{c: "'" for c in _SINGLE_QUOTE_CHARS},
     **{c: '"' for c in _DOUBLE_QUOTE_CHARS},
     "\u00ad": "",        # soft hyphen: invisible, so drop rather than fold
     "\u2044": "/",       # fraction slash
     "\u2026": "..."}     # ellipsis -> three dots (NFKC leaves it alone)
)


def normalize(text: str, strip_punct: bool = False) -> str:
    """Canonical form used for agreement comparison, and for the stored consensus label.

    Four steps, in this order:

    1. NFKC. Compatibility composition, not just canonical (NFC). Verified safe for this
       script: NOT ONE codepoint in the Telugu block U+0C00-U+0C7F is altered by NFKC, so
       there is nothing to lose, and it fixes real OCR output that NFC leaves alone --
       no-break space and narrow no-break space collapse to a plain space, fullwidth Latin
       folds to ASCII, and ligatures/superscripts (fi, squared) fold to their plain forms.
       Those show up around the numerals and embedded English in scanned tables.
    2. Strip zero-width characters. NFKC does NOT touch these, and they matter twice over:
       the grapheme tokenizer's vocab has ZERO entries containing ZWNJ/ZWJ, so a label
       carrying one is unrepresentable; and ZWNJ changes grapheme clustering outright --
       "ల్‌లు" is two clusters with it and one without -- which silently inflates the edit
       distance between two engines that read the same text.
    3. Fold equivalent punctuation to one spelling: every dash variant to ASCII "-",
       curly/prime quotes to ASCII quotes, soft hyphen dropped, ellipsis to "...". This is
       the step NFKC cannot do -- dashes and curly quotes are distinct characters rather
       than compatibility variants, so NFKC passes U+2013 EN DASH and U+201C straight
       through. It is where the real cross-engine noise lives in this corpus: Paddle writes
       U+2013 where the model and Tesseract write "-", and Tesseract writes curly quotes
       where the others do not. Folding (not deleting) keeps "300-00" readable.
    4. Collapse runs of whitespace to a single space. Must come after step 1, since that
       is what turns a no-break space into a collapsible one.
    5. Strip leading/trailing whitespace.

    Measured on 300 real rows: NFC vs NFKC and zero-width stripping changed nothing
    (identical tiering, mean disagreement 0.2915 -> 0.2913), so those two are correctness
    insurance for text that has not turned up yet. The punctuation fold in step 3 is the
    one that acts on this corpus. Beyond these, the remaining disagreement is genuine
    character-level disagreement between the engines, not formatting: comparing with spaces
    removed entirely was also tested and moved zero rows.
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = _ZERO_WIDTH_RE.sub("", t)
    t = t.translate(_PUNCT_FOLD)
    if strip_punct:
        t = _LOOSE_PUNCT_RE.sub("", t)
    return _WS_RE.sub(" ", t).strip()


# Shared splitter (phase 3). This module's `normalize` below stays local and NFKC-based
# on purpose -- it decides ENGINE AGREEMENT, not benchmark CER.
from src.telugu_ocr.metrics.normalize import graphemes  # noqa: E402


def has_english(text: str) -> bool:
    """True if the text contains Latin letters. Driven by OUR model's output, per spec."""
    return bool(_LATIN_RE.search(text or ""))


def english_frac(text: str) -> float:
    """Fraction of the letters that are Latin -- lets you tell an embedded English word
    apart from a stray misrecognized character."""
    letters = [c for c in (text or "") if c.isalpha()]
    if not letters:
        return 0.0
    return sum(bool(_LATIN_RE.match(c)) for c in letters) / len(letters)


try:                                     # progress bar with ETA; degrades to prints
    from tqdm.auto import tqdm

    _HAVE_TQDM = True
except ImportError:
    _HAVE_TQDM = False


try:                                     # optional: ~50x faster, same numbers
    from rapidfuzz.distance import Levenshtein as _Lev

    def _edit_distance(a: list[str], b: list[str]) -> int:
        return _Lev.distance(a, b)

    _HAVE_RAPIDFUZZ = True
except ImportError:                      # pure-Python single-row DP fallback
    def _edit_distance(a: list[str], b: list[str]) -> int:
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

    _HAVE_RAPIDFUZZ = False


def cer(ref: str, hyp: str) -> float:
    """True grapheme-level CER: edit distance from `hyp` to `ref`, divided by len(ref).

    Requires a GROUND-TRUTH reference. It is deliberately asymmetric -- the reference
    length is the denominator -- and can exceed 1.0 when the hypothesis is much longer,
    which is normal for CER and is not clipped here.

    This is NOT what the tiering uses; there is no ground truth at labelling time. Use
    `disagreement` for engine-vs-engine comparison. This function is here for later, when
    you evaluate a model against the Tier-1 labels this script produces.
    """
    r, h = graphemes(ref), graphemes(hyp)
    if not r:
        return 0.0 if not h else 1.0
    return _edit_distance(r, h) / len(r)


def disagreement(a: str, b: str) -> float:
    """Symmetric normalized edit distance between two OCR hypotheses, in [0, 1].

    THERE IS NO GROUND TRUTH ANYWHERE IN THIS SCRIPT, so nothing here is an error rate.
    This measures how far two engines are from EACH OTHER: 0.0 = identical, 1.0 = nothing
    in common. Read it as a disagreement rate, never as accuracy -- two engines can agree
    perfectly on a reading that is wrong.

    Normalized by the LONGER side, which matters:
      * symmetric, so it does not depend on which engine you pass first. Dividing by one
        arbitrary side made the same pair score 1.00 or 0.63 depending on argument order.
      * naturally bounded by 1.0, so no clipping is needed. Clipping to 1.0 (as dividing
        by the shorter side forces you to) collapses "quite different" and "wildly
        different" into the same value and throws away exactly the signal you would want
        when re-tiering at a threshold.
    """
    ga, gb = graphemes(a), graphemes(b)
    if not ga and not gb:
        return 0.0
    if not ga or not gb:
        return 1.0
    return _edit_distance(ga, gb) / max(len(ga), len(gb))


# ======================================================================================
# Engines
#
# Contract: .run(images) takes a list of PIL images and returns a list of the same length.
# A str (possibly "") is a prediction; None means the engine FAILED on that sample and it
# must not be mistaken for an empty reading -- the tiering treats the two differently.
# ======================================================================================

@dataclass
class EngineResult:
    name: str
    texts: list[str | None]
    seconds: float = 0.0
    n_failed: int = 0
    scores: list[float | None] | None = None


from benchmark.engines_ext.base import OCREngine
from benchmark.engines_ext.paddle import PaddleOCREngine
from benchmark.engines_ext.tesseract import TesseractEngine


# --------------------------------------------------------------------------------------
# Tesseract
# --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
# PaddleOCR
# --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
# Our CTC model
# --------------------------------------------------------------------------------------
def preprocess_for_ctc(img: Image.Image, image_height: int = 64,
                       max_image_width: int = 2048, downsample: int = 8) -> np.ndarray:
    """Grayscale -> height `image_height` (aspect preserved) -> width padded to a multiple
    of `downsample`, white fill. Returns float32 in [-1, 1], shape (1, H, W).

    The geometry is `resize_line_image`, shared with every other caller since phase 3 --
    it used to be a hand-copy that had to be "kept in sync", which is what let seven
    copies drift. Only the return convention is local: a normalized ndarray with a
    channel dim, because the batching below sorts on `.shape[2]` and stacks with numpy.
    """
    arr = resize_line_image(img, image_height, max_image_width, downsample, out="np")
    x = arr.astype(np.float32) / 255.0
    return ((x - 0.5) / 0.5)[None, ...]


def _load_state(checkpoint: str) -> dict:
    """Read a .pt or .safetensors checkpoint into a flat state dict."""
    import torch

    if checkpoint.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(checkpoint)
    else:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        state = state.get("state_dict", state)
    # Strip wrapper prefixes a checkpoint may carry: "module." from DDP, "model." when it
    # was saved through a wrapper (see extend_from_checkpoint in the CTC trainer).
    for prefix in ("module.", "model."):
        if any(k.startswith(prefix) for k in state):
            state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            break
    return state


def _strict_load(model, state: dict, name: str, checkpoint: str) -> None:
    """Load weights and REFUSE to continue on any mismatch.

    This is deliberately fatal. A tolerant load here is the single most dangerous thing in
    this whole script: load_state_dict(strict=False) against the wrong architecture leaves
    every weight at its random init, the model happily emits fluent-looking garbage, and
    that garbage gets written out as pseudo-labels. It already happened once in testing --
    150 missing / 426 unexpected keys, printed as a warning, and the run completed with a
    99.9% CER that looked like a hard dataset rather than a broken load.
    """
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"[{name}] checkpoint does not match the model architecture: "
            f"{len(missing)} missing and {len(unexpected)} unexpected keys.\n"
            f"  checkpoint: {checkpoint}\n"
            f"  first missing   : {list(missing)[:4]}\n"
            f"  first unexpected: {list(unexpected)[:4]}\n"
            f"Refusing to run on partially-initialized weights."
        )


def build_model_engine(checkpoint: str, vocab_file: str, **kwargs) -> OCREngine:
    """Pick the right engine by sniffing the checkpoint's keys.

    The repo has two trained architectures with incompatible checkpoints (the CTC encoder
    and the older ViT+GPT encoder-decoder), and nothing in the filename says which is
    which. Detecting it beats making the caller keep a flag in sync with a path.
    """
    keys = set(_load_state(checkpoint).keys())
    if any(k.startswith("decoder_model.") for k in keys):
        print(f"[model] detected encoder-decoder checkpoint ({len(keys)} tensors)")
        return EncoderDecoderEngine(checkpoint, vocab_file, **kwargs)
    if "ctc_head.weight" in keys:
        print(f"[model] detected CTC encoder checkpoint ({len(keys)} tensors)")
        return TeluguCTCEngine(checkpoint, vocab_file, **kwargs)
    raise RuntimeError(
        f"cannot tell which architecture {checkpoint} belongs to; it has neither "
        f"'decoder_model.*' (encoder-decoder) nor 'ctc_head.weight' (CTC) keys. "
        f"Sample keys: {sorted(keys)[:5]}"
    )


class EncoderDecoderEngine(OCREngine):
    """CTC image encoder + cross-attending GPT decoder, batched greedy decoding.

    Configs are pinned to the values stage-1 trained with; they have to match exactly or
    the strict load below rejects the checkpoint. configs/models/encoder_decoder_stage1.yaml
    records the same values, and tests/test_checkpoint_compat.py asserts they still load.

    NOT A ViT. This engine used to build a ViT encoder (embed_dim=512, patch_size=8) and
    feed the decoder a per-patch padding mask, because that is what the encoder-decoder was
    before the CTC refactor. Every checkpoint that exists has a conv-stem encoder instead:
    encoder_model.stem.blocks.N.conv.*, pos_embed (1, 256, 384), embed_dim 384. The old
    code could not have run against one -- it imported a ViTConfig that no longer exists
    anywhere in the repo, and it skipped enc_to_dec, so the decoder would have received
    384-dim features where it expects 512. Both are fixed here.

    The consequence for batching is that the encoder wants per-sample VALID FRAME COUNTS
    (`input_lengths`, frames = ceil(W_real / downsample)) and derives its own frame mask.
    There is no patch grid and no (B, hp*wp) mask any more.
    """

    name = "model"
    device_kind = "gpu"

    def __init__(self, checkpoint: str, vocab_file: str, device: str | None = None,
                 max_tokens: int = 160, max_pixels_per_batch: int = 64 * 2048 * 8,
                 width_bucket: int = 128, amp: bool = True):
        import torch
        from src.telugu_ocr.models.encoder_decoder import EncoderDecoder
        from src.telugu_ocr.models.image_encoder import CTCEncoderConfig
        from src.telugu_ocr.models.text_decoder import GPTConfig
        from src.telugu_ocr.tokenizer.grapheme import TeluguGraphemeTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "mps" if torch.backends.mps.is_available()
                                 else "cpu")
        self.tokenizer = TeluguGraphemeTokenizer(vocab_file=vocab_file)
        self.image_height = 64
        self.downsample = 8              # conv-stem width reduction: T = W // downsample
        self.max_image_width = 2048      # 256 frames, matching the trained checkpoint
        self.max_tokens = max_tokens

        # These values are the ones the stage-1 checkpoint was trained with, verified by
        # tests/test_checkpoint_compat.py against configs/models/encoder_decoder_stage1.yaml
        # (strict load, 388 tensors). They are inlined rather than read from that YAML so
        # this file stays runnable standalone on Kaggle -- if you change one, change both.
        decoder_config = GPTConfig(vocab_size=len(self.tokenizer), embed_dim=512,
                                   hidden_dim=1368, num_heads=8, num_layers=16,
                                   ctx_len=256, dropout=0.1)
        encoder_config = CTCEncoderConfig(max_image_width=self.max_image_width,
                                          max_frames=self.max_image_width // self.downsample)
        self.model = EncoderDecoder(encoder_config=encoder_config,
                                    decoder_config=decoder_config,
                                    pad_index=self.tokenizer.pad_token_id)
        _strict_load(self.model, _load_state(checkpoint), self.name, checkpoint)
        self.model.eval().to(self.device)
        self.max_pixels = max_pixels_per_batch
        self.width_bucket = width_bucket
        self.amp = amp and self.device == "cuda"

    def _decode_batch(self, arrays: list[np.ndarray], idxs: list[int],
                      widest: int, out: list[str | None]) -> None:
        torch = self.torch
        B = len(idxs)
        rows = B
        if self.width_bucket:
            # See TeluguCTCEngine: bound the distinct (rows, width) shapes, because the
            # MPS backend caches a compiled graph per shape and never evicts it. Dummy
            # rows are zero images at FULL valid length (a zero length would make the
            # encoder mask every frame, so cross-attention would see nothing but padding)
            # and start the greedy loop already `done`.
            widest = min(-(widest // -self.width_bucket) * self.width_bucket,
                         self.max_image_width)
            rows = 1 << (B - 1).bit_length()
        # The encoder is the conv-stem CTC encoder: it takes per-sample VALID FRAME COUNTS
        # and builds its own frame mask, not a ViT-style per-patch padding mask. Frames are
        # ceil(W_real / downsample), capped at the padded width's frame count.
        imgs = torch.zeros((rows, 1, self.image_height, widest), dtype=torch.float32)
        frames_total = widest // self.downsample
        lengths = torch.full((rows,), frames_total, dtype=torch.long)
        for slot, i in enumerate(idxs):
            a = arrays[i]
            w = a.shape[2]
            imgs[slot, :, :, :w] = torch.from_numpy(a)
            lengths[slot] = min(-(w // -self.downsample), frames_total)      # ceil

        eos, bos = self.tokenizer.eos_token_id, self.tokenizer.bos_token_id
        try:
            imgs = imgs.to(self.device)
            lengths = lengths.to(self.device)
            with torch.no_grad():
                ctx = (torch.autocast("cuda", dtype=torch.float16) if self.amp
                       else _nullcontext())
                with ctx:
                    # Mirrors EncoderDecoder.forward, with encode() hoisted out of the
                    # greedy loop -- calling the full forward per token would re-encode the
                    # image once per emitted grapheme. enc_to_dec is NOT optional: it
                    # projects the encoder's 384 dims to the decoder's 512.
                    enc, key_padding_mask = self.model.encoder_model.encode(imgs, lengths)
                    enc = self.model.enc_to_dec(enc)
                    cross_key_mask = (None if key_padding_mask is None else
                                      (~key_padding_mask).unsqueeze(1).unsqueeze(2))
                    ids = torch.full((rows, 1), bos, dtype=torch.long, device=self.device)
                    done = torch.zeros(rows, dtype=torch.bool, device=self.device)
                    done[B:] = True
                    for _ in range(self.max_tokens):
                        logits = self.model.decoder_model(ids, enc, None,
                                                          cross_key_mask, None).logits
                        nxt = logits[:, -1, :].argmax(dim=-1)
                        # once a row has emitted EOS, keep feeding EOS so its slice of the
                        # batch stops changing while the others finish
                        nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
                        ids = torch.cat([ids, nxt.unsqueeze(1)], dim=1)
                        done |= nxt == eos
                        if bool(done.all()):
                            break
            seqs = ids.cpu().tolist()
            for slot, i in enumerate(idxs):
                seq = seqs[slot][1:]                       # drop BOS
                if eos in seq:
                    seq = seq[:seq.index(eos)]
                out[i] = self.tokenizer.decode(seq, skip_special_tokens=True)
        except Exception as exc:
            print(f"[{self.name}] batch of {B} failed: {exc}")
            for i in idxs:
                out[i] = None
        self._tick(B)

    def run(self, images: list[Image.Image]) -> list[str | None]:
        arrays = [preprocess_for_ctc(im, self.image_height, self.max_image_width,
                                     self.downsample) for im in images]
        order = sorted(range(len(arrays)), key=lambda i: arrays[i].shape[2])
        out: list[str | None] = [None] * len(arrays)
        batch: list[int] = []
        widest = 0
        for i in order:
            w = max(widest, arrays[i].shape[2])
            if batch and w * (len(batch) + 1) * self.image_height > self.max_pixels:
                self._decode_batch(arrays, batch, widest, out)
                batch, widest = [], 0
                w = arrays[i].shape[2]
            batch.append(i)
            widest = w
        if batch:
            self._decode_batch(arrays, batch, widest, out)
        return out


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


class TeluguCTCEngine(OCREngine):
    """Our ImageEncoderCTC checkpoint, batched and width-bucketed.

    Samples are sorted by width and batched under a PIXEL budget rather than a fixed count,
    so a batch of narrow crops is large and a batch of 2048px crops is small. With a fixed
    batch size the widest sample sets the padded width for the whole batch, and most of the
    GPU work goes into padding.

    `width_bucket` bounds how many DISTINCT (rows, width) shapes the backend ever sees:
    the padded width is rounded up to a multiple of it and the row count up to a power of
    two, with the extra rows blank and their output discarded. This is a memory fix, not a
    speed knob. torch's MPS backend compiles and caches a graph per distinct input shape,
    the cache is never evicted and torch.mps.empty_cache() does not touch it; with widths
    on any multiple of 8 and free batch sizes, almost every batch is a fresh shape at
    ~10-25 MB of native heap each -- measured at ~13 MB/page, 17 GB over 1300 pages, on
    the wikisource build. Bucketing caps the shape set at a few dozen. 0 disables it.
    """

    name = "model"
    device_kind = "gpu"

    def __init__(self, checkpoint: str, vocab_file: str, device: str | None = None,
                 max_pixels_per_batch: int = 64 * 2048 * 24, width_bucket: int = 128,
                 amp: bool = True):
        import torch
        from src.telugu_ocr.tokenizer.grapheme import TeluguGraphemeTokenizer

        # Which width a checkpoint supports is decided by HOW MANY POSITION ROWS it
        # carries, not by whether a `pos_embed_ext` tensor is present. Those two used to
        # coincide -- the widening from 128 to 256 frames was first shipped as a second
        # parameter -- but the table was later merged back into a single `pos_embed`, so
        # a merged 2048px checkpoint has no `pos_embed_ext` at all. Sniffing for that
        # name reported the stage-3 checkpoint as a 1024px model and printed a max width
        # it does not have. Harmless today only because both trainer modules now declare
        # max_image_width = 2048, which makes the misidentified branch build the same
        # thing; count the rows instead so it stays right if they ever diverge.
        state = _load_state(checkpoint)
        rows = (state["pos_embed"].shape[1]
                + (state["pos_embed_ext"].shape[1] if "pos_embed_ext" in state else 0))
        self.variant = "2048" if rows >= 256 else "1024"
        # FIXME(phase-3): this engine is broken, and was broken before the refactor.
        #   1. The "2048" branch used to import src.image_encoder.train_ctc_encoder_2048,
        #      a module that has never existed in git history or on disk -- so the branch
        #      that fires for EVERY real checkpoint (all carry 256 position rows) raised
        #      ModuleNotFoundError. Both branches now point at the canonical module, which
        #      makes the import resolve but does not make the engine work, because:
        #   2. `CTCEncoderConfig()` below builds from the DATACLASS DEFAULTS (1024px /
        #      128 frames). No trained artifact uses those, so _strict_load then fails on
        #      the shape of pos_embed. The fix is to build from
        #      configs/models/ctc_encoder_2048.yaml instead of from defaults; that is a
        #      behaviour change, so it belongs in phase 3, not in this mechanical pass.
        from src.telugu_ocr.models.image_encoder import (CTCEncoderConfig,
                                                         ImageEncoderCTC)
        _split = "split" if "pos_embed_ext" in state else "merged"
        print(f"[{self.name}] CTC variant '{self.variant}' "
              f"({rows} position rows, {_split}; max width "
              f"{CTCEncoderConfig().max_image_width})")

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "mps" if torch.backends.mps.is_available()
                                 else "cpu")
        self.tokenizer = TeluguGraphemeTokenizer(vocab_file=vocab_file)
        self.cfg = CTCEncoderConfig()
        self.model = ImageEncoderCTC(self.cfg)

        _strict_load(self.model, state, self.name, checkpoint)

        self.model.eval().to(self.device)
        self.blank_id = self.model.blank_id
        self.max_pixels = max_pixels_per_batch
        self.width_bucket = width_bucket
        # autocast only helps on CUDA; on MPS/CPU it is a slowdown or unsupported
        self.amp = amp and self.device == "cuda"

    def _collapse(self, row) -> list[int]:
        """Standard greedy CTC collapse: drop repeats, then drop blanks."""
        out, prev = [], None
        for t in row:
            t = int(t)
            if t != prev and t != self.blank_id and t >= 0:
                out.append(t)
            prev = t
        return out

    def run(self, images: list[Image.Image]) -> list[str | None]:
        torch = self.torch
        arrays = [preprocess_for_ctc(im, self.cfg.image_height,
                                     self.cfg.max_image_width, self.cfg.downsample)
                  for im in images]
        order = sorted(range(len(arrays)), key=lambda i: arrays[i].shape[2])

        texts: list[str | None] = [None] * len(arrays)
        batch: list[int] = []
        widest = 0

        def flush(idxs, widest_w):
            if not idxs:
                return
            B = len(idxs)
            rows = B
            if self.width_bucket:
                widest_w = min(-(widest_w // -self.width_bucket) * self.width_bucket,
                               self.cfg.max_image_width)
                rows = 1 << (B - 1).bit_length()
            # dummy rows stay all-white with length 1; slots B..rows are never read back
            imgs = torch.full((rows, 1, self.cfg.image_height, widest_w), 1.0,
                              dtype=torch.float32)
            lengths = torch.ones(rows, dtype=torch.long)
            for slot, i in enumerate(idxs):
                a = arrays[i]
                imgs[slot, :, :, :a.shape[2]] = torch.from_numpy(a)
                lengths[slot] = a.shape[2] // self.cfg.downsample
            try:
                imgs = imgs.to(self.device)
                lengths_d = lengths.to(self.device)
                with torch.no_grad():
                    if self.amp:
                        with torch.autocast("cuda", dtype=torch.float16):
                            out = self.model(imgs, input_lengths=lengths_d)
                    else:
                        out = self.model(imgs, input_lengths=lengths_d)
                # ImageEncoderCTC returns the greedy PER-FRAME IDS under the key
                # "logits" (HF Trainer convention -- it is not a logit tensor), with
                # padded frames already forced to blank.
                pred = out["logits"] if isinstance(out, dict) else out
                pred = pred.cpu().numpy()
                for slot, i in enumerate(idxs):
                    ids = self._collapse(pred[slot][:int(lengths[slot])])
                    texts[i] = self.tokenizer.decode(ids, skip_special_tokens=True)
            except Exception as exc:
                print(f"[{self.name}] batch of {B} failed: {exc}")
                for i in idxs:
                    texts[i] = None
            self._tick(B)

        for i in order:
            w = max(widest, arrays[i].shape[2])
            if batch and w * (len(batch) + 1) * self.cfg.image_height > self.max_pixels:
                flush(batch, widest)
                batch, widest = [], 0
                w = arrays[i].shape[2]
            batch.append(i)
            widest = w
        flush(batch, widest)
        return texts


# ======================================================================================
# Tiering
# ======================================================================================
TIER_REASONS = (
    "all_three_agree",                        # tier 1
    "tesseract_paddle_agree_model_differs",   # tier 2
    "model_paddle_agree_tesseract_differs",   # tier 3
    "model_tesseract_agree_paddle_differs",   # tier 4
    "all_three_differ",                       # tier 4
    "engine_error",                           # tier 0
    "all_predictions_empty",                  # tier 0
)


def assign_tier(model: str | None, tesseract: str | None, paddle: str | None,
                strip_punct: bool = False) -> tuple[int, str, str]:
    """Return (tier, reason, consensus_text).

    The four tiers, in the order they are tested:
        1  all three agree
        2  tesseract + paddle agree, model differs
        3  model + paddle agree, tesseract differs
        4  model + tesseract agree and paddle differs, OR all three differ
        0  not tierable (an engine errored, or every prediction was empty)

    Order matters: the all-three case has to be tested before any pair, and each pair is
    then tested in tier order so a row lands in the strongest tier it qualifies for.

    `consensus_text` is the agreed NORMALIZED text whenever at least two engines agree --
    including the model+tesseract pairing in tier 4, since that is still evidence and the
    decision about what to trust is yours. It is "" only when all three differ.

    Empty predictions never count as a vote: two engines both failing to read a crop is a
    shared failure, not agreement, and treating it as one would mint tier-1 rows whose
    label is the empty string.
    """
    if model is None or tesseract is None or paddle is None:
        return 0, "engine_error", ""

    m = normalize(model, strip_punct)
    t = normalize(tesseract, strip_punct)
    p = normalize(paddle, strip_punct)

    if not (m or t or p):
        return 0, "all_predictions_empty", ""

    if m and m == t == p:
        return 1, "all_three_agree", m
    if t and t == p:
        return 2, "tesseract_paddle_agree_model_differs", t
    if m and m == p:
        return 3, "model_paddle_agree_tesseract_differs", m
    if m and m == t:
        return 4, "model_tesseract_agree_paddle_differs", m
    return 4, "all_three_differ", ""


# ======================================================================================
# Orchestration
# ======================================================================================
_DEVICE_LOCKS: dict[str, threading.Lock] = {}
_DEVICE_LOCKS_GUARD = threading.Lock()


def _gpu_lock_key(engine: OCREngine) -> str:
    """Which lock a GPU engine contends for: its resolved device, normalized.

    paddle spells a device "gpu:0" and torch spells the same one "cuda:0", so the two have
    to be folded onto one name or engines sharing a device would take different locks and
    fight over it anyway. An engine that never chose a device gets the default ordinal 0,
    which is what both frameworks use when told nothing.
    """
    dev = str(getattr(engine, "device", None) or "cuda:0").strip().lower()
    if dev.startswith("gpu"):
        dev = "cuda" + dev[3:]
    if dev in ("cuda", "cuda:"):
        dev = "cuda:0"
    return dev


def _device_lock(key: str) -> threading.Lock:
    with _DEVICE_LOCKS_GUARD:
        return _DEVICE_LOCKS.setdefault(key, threading.Lock())


def run_engines(engines: list[OCREngine], images: list[Image.Image],
                on_items=None) -> dict[str, EngineResult]:
    """Run every engine over the same chunk, engines concurrently.

    CPU engines always run at once. GPU engines take a lock keyed on their DEVICE, so they
    are serialized only against engines sharing that device and always overlap with the CPU
    ones. On one GPU that reproduces the old single-lock behaviour -- Tesseract's thread
    pool saturates the CPU while one GPU engine at a time saturates the GPU. On a 2-GPU box
    with the model on cuda:1 and Paddle on gpu:0 the keys differ, nothing is serialized,
    and both GPUs run flat out.

    `on_items(k)` is called as engines finish images, from whichever worker thread got
    there first, so it must be cheap and thread-safe.
    """
    for e in engines:
        e._on_items = on_items

    def _one(engine: OCREngine) -> EngineResult:
        t0 = time.perf_counter()
        if engine.device_kind == "gpu":
            with _device_lock(_gpu_lock_key(engine)):
                out = engine.run(images)
        else:
            out = engine.run(images)
        # engines may return texts, or (texts, scores) when they expose a confidence
        scores = None
        if isinstance(out, tuple):
            texts, scores = out
        else:
            texts = out
        if len(texts) != len(images):
            raise RuntimeError(f"engine {engine.name} returned {len(texts)} results for "
                               f"{len(images)} images; refusing to misalign rows")
        return EngineResult(name=engine.name, texts=texts,
                            seconds=time.perf_counter() - t0,
                            n_failed=sum(t is None for t in texts),
                            scores=scores)

    try:
        with ThreadPoolExecutor(max_workers=max(1, len(engines))) as pool:
            results = list(pool.map(_one, engines))
    finally:
        for e in engines:
            e._on_items = None      # don't leave a stale bar attached to an engine
    return {r.name: r for r in results}


@dataclass
class LabellingStats:
    n_total: int = 0
    tier_counts: dict = field(default_factory=dict)
    reason_counts: dict = field(default_factory=dict)
    english_by_tier: dict = field(default_factory=dict)
    engine_seconds: dict = field(default_factory=dict)
    engine_failures: dict = field(default_factory=dict)
    engine_empty: dict = field(default_factory=dict)
    disagree_sums: dict = field(default_factory=dict)
    disagree_n: dict = field(default_factory=dict)
    paddle_score_sum: float = 0.0
    paddle_score_n: int = 0
    wall_seconds: float = 0.0
    interrupted: bool = False
    n_resumed: int = 0

    def as_dict(self) -> dict:
        mean_dis = {k: (self.disagree_sums[k] / self.disagree_n[k])
                    if self.disagree_n.get(k) else None for k in self.disagree_sums}
        return {
            "n_total": self.n_total,
            "tier_counts": self.tier_counts,
            "tier_pct": {k: round(100.0 * v / max(1, self.n_total), 2)
                         for k, v in sorted(self.tier_counts.items())},
            "reason_counts": self.reason_counts,
            "english_by_tier": self.english_by_tier,
            "engine_seconds": {k: round(v, 1) for k, v in self.engine_seconds.items()},
            "engine_failures": self.engine_failures,
            "engine_empty": self.engine_empty,
            "mean_pairwise_disagreement": {k: (round(v, 4) if v is not None else None)
                                           for k, v in mean_dis.items()},
            "mean_paddle_score": (round(self.paddle_score_sum / self.paddle_score_n, 4)
                                  if self.paddle_score_n else None),
            "wall_seconds": round(self.wall_seconds, 1),
            "interrupted": self.interrupted,
            "n_resumed": self.n_resumed,
        }


NEW_COLUMNS = ("pred_model", "pred_tesseract", "pred_paddle", "paddle_score",
               "tier", "tier_reason", "consensus_text",
               "has_english", "english_frac",
               "disagree_model_tesseract", "disagree_model_paddle",
               "disagree_tesseract_paddle")

def _read_progress(path: str) -> dict:
    """Load an existing progress file into {row_index: record}.

    Tolerates a truncated final line: if the process was killed mid-write, the last record
    may be half-flushed, and one lost row is preferable to refusing to resume.
    """
    done: dict[int, dict] = {}
    if not path or not Path(path).exists():
        return done
    bad = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done[int(rec["row_index"])] = rec
            except Exception:
                bad += 1
    if bad:
        print(f"[warn] skipped {bad} unparseable line(s) in {path} "
              f"(probably a kill mid-write)")
    return done


def _append_progress(path: str, records: list[dict]) -> None:
    """Append a chunk's records and force them to disk.

    fsync is the point of this function: without it the records sit in the OS page cache
    and a `kill -9` loses everything since the last flush. One fsync per chunk (512 rows)
    is far too cheap to matter next to ~90s of OCR.
    """
    if not path or not records:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _stats_from_records(records: list[dict], stats: LabellingStats) -> None:
    """Fill the row-derived counters from records, so a RESUMED run reports totals over
    everything on disk rather than only the rows this process happened to do.

    Engine timings are left alone: those can only describe the current process.
    A prediction stored as null means that engine errored on that row, which is why the
    records keep null distinct from "" -- an empty string is a failed READ and still counts
    as a vote-less prediction, while null means no prediction at all and must stay out of
    the disagreement means.
    """
    stats.tier_counts, stats.reason_counts, stats.english_by_tier = {}, {}, {}
    for pair in ("model_tesseract", "model_paddle", "tesseract_paddle"):
        stats.disagree_sums[pair] = 0.0
        stats.disagree_n[pair] = 0
    stats.paddle_score_sum, stats.paddle_score_n = 0.0, 0
    for r in records:
        tier, reason = r["tier"], r["tier_reason"]
        stats.tier_counts[tier] = stats.tier_counts.get(tier, 0) + 1
        stats.reason_counts[reason] = stats.reason_counts.get(reason, 0) + 1
        if r["has_english"]:
            stats.english_by_tier[tier] = stats.english_by_tier.get(tier, 0) + 1
        m_ok = r["pred_model"] is not None
        t_ok = r["pred_tesseract"] is not None
        p_ok = r["pred_paddle"] is not None
        for key, col, ok in (("model_tesseract", "disagree_model_tesseract", m_ok and t_ok),
                             ("model_paddle", "disagree_model_paddle", m_ok and p_ok),
                             ("tesseract_paddle", "disagree_tesseract_paddle", t_ok and p_ok)):
            if ok:
                stats.disagree_sums[key] += r[col]
                stats.disagree_n[key] += 1
        if r["paddle_score"] is not None and r["paddle_score"] >= 0:
            stats.paddle_score_sum += r["paddle_score"]
            stats.paddle_score_n += 1
    stats.n_total = len(records)


def label_dataset(ds, engines: list[OCREngine], image_col: str = "image",
                  chunk_size: int = 512, strip_punct: bool = False,
                  show_progress: bool = True, progress_path: str | None = None,
                  resume: bool = True):
    """Run the engines over `ds` and return (dataset_with_new_columns, LabellingStats).

    STREAMING / CRASH SAFETY
        When `progress_path` is set, every chunk's results are appended to that JSONL and
        fsynced before the next chunk starts. Nothing is held only in memory, so a Ctrl-C,
        a `kill -9` or a power cut costs at most the chunk in flight -- on a multi-hour run
        that is the difference between losing 512 rows and losing everything.

        On restart the file is read back and those rows are SKIPPED, so re-running the same
        command resumes where it stopped. Resume is keyed on the row index rather than on
        chunk boundaries, so it still works if you change chunk_size between runs.
        Pass resume=False to start over (the file is then overwritten, not appended to).

        Ctrl-C is caught: the loop stops, the dataset is still assembled from everything
        recorded so far, and stats.interrupted is set so the caller can say so.

    Columns are attached with add_column at the end, so the image column is never
    re-encoded -- mapping over the dataset instead would decode and re-encode every image.

    Progress is a tqdm bar over ROWS (not chunks), so the rate and ETA are meaningful from
    the first chunk onward; the running tier split is shown in the postfix. Expect the ETA
    to settle downward after the first chunk -- that one absorbs model warm-up.
    """
    if progress_path and not resume and Path(progress_path).exists():
        Path(progress_path).unlink()
    done = _read_progress(progress_path) if progress_path else {}
    done = {i: r for i, r in done.items() if 0 <= i < len(ds)}
    pending = [i for i in range(len(ds)) if i not in done]
    if done:
        print(f"resuming: {len(done)} of {len(ds)} rows already in {progress_path}; "
              f"{len(pending)} to go", flush=True)

    stats = LabellingStats(n_total=len(ds))
    stats.interrupted = False
    stats.n_resumed = len(done)

    have = {e.name for e in engines}
    for missing in sorted({"model", "tesseract", "paddle"} - have):
        print(f"[warn] engine '{missing}' unavailable -- every row will be Tier 0 "
              f"(reason 'engine_error')")

    t_start = time.perf_counter()
    bar = None
    tick = None
    if show_progress and _HAVE_TQDM:
        bar = tqdm(total=len(ds), initial=len(done), unit="row", smoothing=0.05,
                   desc=f"consensus ({len(engines)} engines)", dynamic_ncols=True,
                   mininterval=0.5)
        # Each engine reports its own images, so N rows produce N*len(engines) ticks. Count
        # each tick as a FRACTION of a row (1/n_engines) so the bar still totals len(ds)
        # while advancing smoothly during a chunk. Updating once per chunk instead means one
        # redraw every CHUNK_SIZE rows -- at 512 rows and ~6 rows/s that is a single jump
        # every ~85 seconds, which reads as a hung bar.
        _share = 1.0 / max(1, len(engines))
        _lock = threading.Lock()
        _acc = [float(len(done))]

        def tick(k: int) -> None:
            # tqdm's update is not thread-safe and these arrive from several engine threads.
            with _lock:
                _acc[0] += k * _share
                whole = int(_acc[0]) - bar.n
                if whole > 0:
                    bar.update(whole)

    try:
        for start in range(0, len(pending), chunk_size):
            idx = pending[start:start + chunk_size]
            images = ds.select(idx)[image_col]
            images = [im if isinstance(im, Image.Image) else Image.fromarray(np.asarray(im))
                      for im in images]

            results = run_engines(engines, images, on_items=tick)
            for name, res in results.items():
                stats.engine_seconds[name] = stats.engine_seconds.get(name, 0.0) + res.seconds
                stats.engine_failures[name] = stats.engine_failures.get(name, 0) + res.n_failed
                stats.engine_empty[name] = stats.engine_empty.get(name, 0) + sum(
                    1 for t in res.texts if t is not None and not normalize(t))

            n = len(idx)
            blank: list[str | None] = [None] * n
            m_txt = results["model"].texts if "model" in results else blank
            t_txt = results["tesseract"].texts if "tesseract" in results else blank
            p_txt = results["paddle"].texts if "paddle" in results else blank
            p_scores = (results["paddle"].scores if "paddle" in results else None) or blank

            batch = []
            for k, row_index in enumerate(idx):
                m, t, p = m_txt[k], t_txt[k], p_txt[k]
                tier, reason, consensus = assign_tier(m, t, p, strip_punct)
                batch.append({
                    "row_index": int(row_index),
                    # null (not "") when an engine errored -- see _stats_from_records
                    "pred_model": m, "pred_tesseract": t, "pred_paddle": p,
                    "paddle_score": (round(float(p_scores[k]), 4)
                                     if p_scores[k] is not None else -1.0),
                    "tier": tier, "tier_reason": reason, "consensus_text": consensus,
                    "has_english": has_english(m or ""),
                    "english_frac": round(english_frac(m or ""), 4),
                    "disagree_model_tesseract": round(
                        disagreement(normalize(m or ""), normalize(t or "")), 4),
                    "disagree_model_paddle": round(
                        disagreement(normalize(m or ""), normalize(p or "")), 4),
                    "disagree_tesseract_paddle": round(
                        disagreement(normalize(t or ""), normalize(p or "")), 4),
                })

            _append_progress(progress_path, batch)     # fsynced before we move on
            for rec in batch:
                done[rec["row_index"]] = rec

            if bar is not None:
                tc: dict = {}
                for r in done.values():
                    tc[r["tier"]] = tc.get(r["tier"], 0) + 1
                bar.set_postfix_str(
                    f"T1={tc.get(1, 0)} T2={tc.get(2, 0)} T3={tc.get(3, 0)} "
                    f"T4={tc.get(4, 0)} T0={tc.get(0, 0)}", refresh=True)
            elif show_progress:
                rate = len(done) / max(1e-6, time.perf_counter() - t_start)
                print(f"  {len(done)}/{len(ds)} rows  {rate:.1f} rows/s", flush=True)
    except KeyboardInterrupt:
        stats.interrupted = True
        print(f"\n[interrupted] stopping after {len(done)} of {len(ds)} rows. "
              f"Everything recorded so far is saved; re-run the same command to resume.",
              flush=True)

    if bar is not None:
        bar.close()
    stats.wall_seconds = time.perf_counter() - t_start

    order = sorted(done)
    records = [done[i] for i in order]
    _stats_from_records(records, stats)

    out = ds.select(order)
    for name in NEW_COLUMNS:
        if name in out.column_names:
            out = out.remove_columns(name)
        # null in a record means "engine errored"; the dataset column carries "" for it
        out = out.add_column(name, [(r[name] if r[name] is not None else
                                     ("" if name.startswith("pred_") else r[name]))
                                    for r in records])
    return out, stats


# Measured throughput, for reference (11-core M-series Mac, 150 rows, engine alone):
#   model on MPS 117 rows/s | model on CPU 52 | tesseract 8 threads 44 | PaddleOCR 8.8.
# Paddle is the wall and is pinned to ~1 of 11 cores; its cpu_threads/enable_mkldnn
# arguments are accepted but change nothing, and batch size and width-sorting make no
# difference either. Sharding this script over disjoint row ranges was measured too and did
# NOT help on that machine (~4.4 rows/s over 6 shards vs 5.5 single) -- the box is already
# saturated. A CUDA machine is the real fix.

# ======================================================================================
# Stats reporting
# ======================================================================================
_TIER_LABEL = {
    1: "Tier 1  all three agree",
    2: "Tier 2  tesseract+paddle agree, model differs",
    3: "Tier 3  model+paddle agree, tesseract differs",
    4: "Tier 4  model+tesseract agree, or all differ",
    0: "Tier 0  not tierable (engine error / all empty)",
}


def print_stats(stats: LabellingStats) -> None:
    d = stats.as_dict()
    n = max(1, d["n_total"])
    print()
    print("=" * 78)
    fresh = d["n_total"] - d.get("n_resumed", 0)
    rate = fresh / max(1e-6, d["wall_seconds"])
    print(f"CONSENSUS LABELLING -- {d['n_total']} samples total"
          + (f" ({d['n_resumed']} resumed from disk, {fresh} this run)"
             if d.get("n_resumed") else "")
          + f" | {d['wall_seconds']:.1f}s this run ({rate:.1f} rows/s)")
    if d.get("interrupted"):
        print("*** INTERRUPTED -- partial output. Re-run the same command to resume. ***")
    print("=" * 78)

    print("\nTIERS")
    for tier in (1, 2, 3, 4, 0):
        c = d["tier_counts"].get(tier, 0)
        bar = "#" * int(40 * c / n)
        eng = d["english_by_tier"].get(tier, 0)
        print(f"  {_TIER_LABEL[tier]:<46} {c:>7}  {100 * c / n:5.1f}%  {bar}")
        if c:
            print(f"  {'':<46} {'':>7}  english: {eng} ({100 * eng / c:.1f}% of tier)")

    print("\nBREAKDOWN BY REASON")
    for reason in TIER_REASONS:
        c = d["reason_counts"].get(reason, 0)
        if c:
            print(f"  {reason:<40} {c:>7}  {100 * c / n:5.1f}%")

    print("\nPER-ENGINE")
    print(f"  {'engine':<12}{'seconds':>10}{'rows/s':>10}{'failed':>9}{'empty':>9}")
    for name in ("model", "tesseract", "paddle"):
        if name not in d["engine_seconds"]:
            continue
        s = d["engine_seconds"][name]
        print(f"  {name:<12}{s:>10.1f}{d['n_total'] / max(1e-6, s):>10.1f}"
              f"{d['engine_failures'].get(name, 0):>9}{d['engine_empty'].get(name, 0):>9}")

    print("\nMEAN PAIRWISE DISAGREEMENT (grapheme edit distance / longer side)")
    print("  NOT accuracy -- there is no ground truth. 0 = identical, 1 = nothing shared.")
    for pair, v in d["mean_pairwise_disagreement"].items():
        print(f"  {pair:<24}{'n/a' if v is None else f'{v:.4f}'}")
    if d.get("mean_paddle_score") is not None:
        print(f"\nMEAN PADDLE CONFIDENCE: {d['mean_paddle_score']:.4f}"
              f"   (stored per row as `paddle_score`; -1.0 = unavailable)")

    total_english = sum(d["english_by_tier"].values())
    print(f"\nENGLISH-CONTAINING (from model output): {total_english} "
          f"({100 * total_english / n:.1f}%)")
    if not _HAVE_RAPIDFUZZ:
        print("\n[note] rapidfuzz not installed; CER used the slow pure-Python fallback. "
              "`uv add rapidfuzz` speeds this up ~50x.")
    print("=" * 78)


def hf_login() -> None:
    """Log in to the Hub from HF_TOKEN: env var, .env file, or Kaggle secret."""
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
        raise SystemExit(
            "No HuggingFace token found. Set the HF_TOKEN environment variable, put it in "
            ".env, or add a Kaggle secret named HF_TOKEN (Add-ons -> Secrets).")
    from huggingface_hub import login

    login(token=token)


# ======================================================================================
# Inline configuration
# ======================================================================================
if __name__ == "__main__":
    from datasets import load_dataset

    # ---- input ----
    INPUT_REPO = "harsha-desaraju/telugu-book-line-images"   # or a local save_to_disk path
    INPUT_SPLIT = "train"
    INPUT_CONFIG = "set_1"                # HF config/subset name, or None
    IMAGE_COL = "line_image"
    LIMIT = 50000                     # set to an int for a smoke test, None for all rows

    # ---- devices ----
    # None lets each engine pick (single GPU, or MPS/CPU on a laptop) -- and then the two
    # GPU engines share a lock and take turns. On a multi-GPU host give them DIFFERENT
    # devices and they run concurrently: e.g. PADDLE_DEVICE="gpu:0", MODEL_DEVICE="cuda:1"
    # on Kaggle's 2x T4. paddle spells it "gpu:N", torch "cuda:N"; both name the same card.
    PADDLE_DEVICE = None
    MODEL_DEVICE = None

    # ---- output ----
    OUT = Path(REPO_ROOT) / "data"           # /kaggle/working on Kaggle
    OUTPUT_DIR = str(OUT / "consensus_labelled")   # images + all 3 predictions + tiers
    STATS_JSON = str(OUT / "consensus_labelled_stats.json")
    # Streaming/crash-safety: every chunk is appended here and fsynced before the next one
    # starts, so stopping the run never costs more than the chunk in flight. Re-running the
    # same command reads this back and resumes. Set RESUME=False to start from scratch.
    PROGRESS_JSONL = str(OUT / "consensus_labelled.progress.jsonl")
    RESUME = True
    PUSH_TO_HUB = None                 # e.g. "harsha-desaraju/telugu-book-line-consensus"
    PUSH_PRIVATE = True

    # ---- our model ----
    # CHECKPOINT = ("models/image_encoder/ctc_encoder_stage-2/"
    #               "ctc-encoder-2048/final_model.pt")
    CHECKPOINT = (f"{REPO_ROOT}/models/image_encoder/ctc_encoder/ctc-encoder/"
                  f"checkpoint-152000/model.safetensors")
    VOCAB_FILE = f"{REPO_ROOT}/src/telugu_ocr/tokenizer/assets/telugu-vocab.json"

    # Chunk size is now also the SAVE granularity: nothing reaches disk until a chunk
    # finishes, so at ~6 rows/s a 512-row chunk means the first save is ~85s in and a
    # Ctrl-C can cost that much work. 128 saves roughly every 20s instead. The engines
    # still batch fine at 128 (Paddle's own batch is 128) and the per-chunk barrier cost is
    # proportional, not fixed, so throughput is unaffected.
    CHUNK_SIZE = 128
    TESSERACT_LANG = "tel"
    TESSERACT_THREADS = 8
    PADDLE_LANG = "te"          # -> te_PP-OCRv5_mobile_rec (the only Telugu rec model)
    PADDLE_BATCH = 128
    # Compare with punctuation stripped as well? Loosens agreement on quote/dash style,
    # which engines differ on constantly. False keeps Tier 1 strict.
    STRIP_PUNCT = False

    print(f"Loading {INPUT_REPO} (split={INPUT_SPLIT}, config={INPUT_CONFIG}) ...",
          flush=True)
    if Path(INPUT_REPO).exists():
        from datasets import load_from_disk
        ds = load_from_disk(INPUT_REPO)
        if not hasattr(ds, "column_names") or isinstance(ds.column_names, dict):
            ds = ds[INPUT_SPLIT]
    else:
        ds = load_dataset(INPUT_REPO, INPUT_CONFIG, split=INPUT_SPLIT)
    if LIMIT:
        ds = ds.select(range(min(LIMIT, len(ds))))
    print(f"  {len(ds)} rows | columns: {ds.column_names}", flush=True)

    # Each engine is optional: a missing dependency downgrades the run rather than
    # aborting it, so you can get Tesseract+model numbers before installing Paddle.
    engines: list[OCREngine] = []
    for label, build in (
        ("model", lambda: build_model_engine(CHECKPOINT, VOCAB_FILE,
                                             device=MODEL_DEVICE)),
        ("tesseract", lambda: TesseractEngine(lang=TESSERACT_LANG,
                                              num_threads=TESSERACT_THREADS)),
        ("paddle", lambda: PaddleOCREngine(lang=PADDLE_LANG, batch_size=PADDLE_BATCH,
                                           device=PADDLE_DEVICE)),
    ):
        try:
            engines.append(build())
            print(f"  engine ready: {label}", flush=True)
        except Exception as exc:
            print(f"  engine UNAVAILABLE: {label} -- {type(exc).__name__}: {exc}",
                  flush=True)

    if not engines:
        raise SystemExit("no OCR engine could be constructed; nothing to do")

    labelled, stats = label_dataset(ds, engines, image_col=IMAGE_COL,
                                    chunk_size=CHUNK_SIZE, strip_punct=STRIP_PUNCT,
                                    progress_path=PROGRESS_JSONL, resume=RESUME)
    print_stats(stats)
    Path(OUTPUT_DIR).parent.mkdir(parents=True, exist_ok=True)
    labelled.save_to_disk(OUTPUT_DIR)
    print(f"\nWrote dataset -> {OUTPUT_DIR}  ({len(labelled)} rows"
          + (" -- PARTIAL" if stats.interrupted else "") + ")")
    print(f"Progress file -> {PROGRESS_JSONL}  (delete it, or set RESUME=False, "
          f"to start over)")

    Path(STATS_JSON).parent.mkdir(parents=True, exist_ok=True)
    Path(STATS_JSON).write_text(json.dumps(stats.as_dict(), indent=2, ensure_ascii=False))
    print(f"Wrote stats   -> {STATS_JSON}")

    if PUSH_TO_HUB:
        hf_login()
        labelled.push_to_hub(PUSH_TO_HUB, private=PUSH_PRIVATE)
        print(f"Pushed        -> {PUSH_TO_HUB}")

    for e in engines:
        e.close()
