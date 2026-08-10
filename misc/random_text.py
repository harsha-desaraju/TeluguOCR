"""
Plausible-but-random Telugu text generation for OCR training.

Purpose
-------
Random text serves two goals for the OCR model:
  1) make the model rely on the VISUAL signal instead of the language prior it
     learned during decoder pretraining, and
  2) expose it to RARE aksharas, including ones that are NOT in the tokenizer vocab
     and therefore must be emitted as a fallback sequence of codepoints.

An earlier version of this file achieved (1)/(2) by sampling uniformly and by COMPOSING
aksharas from Unicode components. That produced combinations that occur nowhere in real
Telugu ("ఖ్ఝ", "ఫ్హ్ఞో"): the fonts have no shaping rules for them, and the tokenizer's
codepoint fallback then trained the decoder on token sequences that can never be the
right answer for real text.

This version keeps goals (1) and (2) but draws every akshara from the set that is
ATTESTED in the corpora, and wraps them in plausible word / sentence structure:

  * inventory  = grapheme-distribution entries (Telugu + Sanskrit dists, merged) that
    are well-formed standalone aksharas, obey Telugu encoding rules (see
    ``follows_telugu_orthography`` — a threshold of 100 occurrences in a 3.3B grapheme
    corpus also admits misencoded clusters, which are broken rather than rare), and clear
    a frequency threshold, and are renderable (see ``UNRENDERABLE_CHARS``). Nothing is
    invented. Both in-vocab and out-of-vocab (OOV) aksharas are kept — the OOV ones are
    exactly what trains the codepoint-fallback path.
  * weighting  = ``count ** alpha`` (tempered). alpha=1 reproduces corpus frequencies,
    alpha=0 is uniform. ~0.35 keeps frequent aksharas frequent (so the line still looks
    like Telugu) while lifting the rare tail by orders of magnitude.
  * OOV share  = an explicit target (``oov_share``); the boost factor needed to hit it is
    derived analytically from the tempered weights, so the knob is interpretable.
  * coverage   = a shuffled queue over the whole OOV inventory guarantees every rare
    akshara is emitted, instead of leaving it to the tail of a random draw. Queued
    aksharas are placed only at word positions where their SHAPE actually occurs.
  * structure  = word length, and the shape of the akshara at each position in the word,
    come from a real Telugu word list; punctuation / digit / Latin rates come from the
    corpus grapheme distribution. Punctuation attaches at word boundaries, brackets and
    quotes are balanced, numbers are whole single-script runs, and English appears as
    whole words (never as stray letters inside a Telugu word). Lines end with terminal
    punctuation only sometimes, and may contain a mid-line sentence break, because a
    cropped text line is usually a fragment.

The result is text whose SHAPE is plausible (so the decoder's structural prior is not
corrupted) but whose CONTENT is unguessable (so the model must read the pixels).

Token length and the CTC frame budget
-------------------------------------
The tokenizer maps an in-vocab akshara to 1 token and an OOV akshara to one token PER
CODEPOINT (see ``TeluguGraphemeTokenizer._tokenize``), so an OOV akshara costs ~4.6
tokens. The CTC encoder has ``T = image_width // 8`` frames (256 at width 2048) and uses
``CTCLoss(..., zero_infinity=True)``: a sample whose label sequence needs more frames
than it has contributes EXACTLY ZERO loss, silently. OOV-dense lines are the ones most
at risk, i.e. the very samples that carry the signal we want.

So this generator is token-aware:
  * ``max_tokens`` is a hard cap. When the remaining grapheme budget could push a line
    over it, sampling falls back to in-vocab aksharas (1 token each) instead of
    truncating the line, so the grapheme target is still met exactly.
  * ``generate_line`` returns the text together with its token count and the number of
    CTC frames it needs (= tokens + adjacent-duplicate tokens, the CTC repeat-blank
    requirement). Token cost is computed from the vocab with the exact same rule as the
    tokenizer, so no tokenizer/transformers import is needed at generation time; the
    ``__main__`` block cross-checks it against the real tokenizer.

Inputs are FILE PATHS (vocab, grapheme dists, word list) so the module can be uploaded
and run as-is on Kaggle. Every knob is set inline in ``__main__``.

Not derived from data: word-level co-occurrence (how often a line contains an English
word, whether it is parenthesised, ...) cannot be recovered from a unigram grapheme
distribution. Those rates are hand-set in ``DEFAULT_PROBS``, except the overall digit and
Latin CHARACTER shares, which ARE calibrated against the corpus.
"""

import collections
import json
import random
import unicodedata

import regex


# --------------------------------------------------------------------------------------
# Telugu Unicode structure (used only to classify an akshara's SHAPE, never to invent one)
# --------------------------------------------------------------------------------------
TELUGU_LO, TELUGU_HI = "ఀ", "౿"

VIRAMA = "్"
SIGN_CHARS = set("ఀఁంఃఄ")          # candrabindu/anusvara/visarga
MATRA_CHARS = set(
    "".join(chr(c) for c in range(0x0C3E, 0x0C45))           # ా .. ౄ
    + "".join(chr(c) for c in range(0x0C46, 0x0C49))         # ె ే ై
    + "".join(chr(c) for c in range(0x0C4A, 0x0C4D))         # ొ ో ౌ
    + "ౕౖ"                                        # length marks
    + "ౢౣ"                                        # vocalic l / ll signs
)
INDEP_VOWEL_CHARS = set(
    "".join(chr(c) for c in range(0x0C05, 0x0C15)) + "ౠౡ")
CONSONANT_CHARS = set(
    "".join(chr(c) for c in range(0x0C15, 0x0C3A)) + "ఴౘౙౚ")
# Telugu-block characters that are NOT aksharas and must never be sampled as one: the
# Telugu digits, and the numeric / fraction / sign block at U+0C77 and above. They are
# well-formed single grapheme clusters, so only an explicit exclusion keeps them out of
# the middle of words. (Telugu digits are still used, as digits, by the number builder.)
NON_AKSHARA_TELUGU = set(
    "".join(chr(c) for c in range(0x0C66, 0x0C70))
    + "".join(chr(c) for c in range(0x0C77, 0x0C80)))
# Characters excluded for a RENDERING reason rather than a linguistic one: no akshara
# containing one of these enters the inventory.
#   ఀ  U+0C00 TELUGU SIGN COMBINING CANDRABINDU ABOVE — only 2 of the 30 rendering fonts
#      have a glyph for it, so every line carrying it would be drawn by (effectively) a
#      single font and the model would key the glyph to that font instead of its shape.
#      It is also Sanskrit-only here: all 88 attested aksharas carrying it come from the
#      Sanskrit distribution, none from the Telugu one, and the bare sign occurs once in
#      3.3B graphemes. The image pipeline strips it from natural text for the same reason.
UNRENDERABLE_CHARS = set("ఀ")

_GRAPHEME_RE = regex.compile(r"\X")
_PROBE = "క"          # plain base consonant, used to test cluster boundaries

# --------------------------------------------------------------------------------------
# Non-akshara inventories. Everything here is checked against the tokenizer vocab at
# construction time and dropped if absent — an out-of-vocab SYMBOL would tokenize to
# [UNK] (unlike an out-of-vocab akshara, which decomposes into in-vocab codepoints).
# --------------------------------------------------------------------------------------
TERMINAL_PUNCT = ".?!"
DANDA_PUNCT = "।॥"                                 # । ॥ (Sanskrit lines)
CLAUSE_PUNCT = ",;:"
PAIRED_PUNCT = [("(", ")"), ("[", "]"), ("{", "}"), ('"', '"'), ("'", "'")]
INLINE_PUNCT = "/&*+=@#|~<>\\"
PERCENT, CURRENCY, HYPHEN = "%", "₹", "-"

ARABIC_DIGITS = "0123456789"
TELUGU_DIGITS = "౦౧౨౩౪౫౬౭౮౯"

LATIN_LOWER = "abcdefghijklmnopqrstuvwxyz"
LATIN_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# English pseudo-word building blocks: real English phonotactics, no real words. A
# unigram letter distribution cannot give these, so they are hand-written; letters used
# for ACRONYMS are weighted by the English grapheme distribution when it is supplied.
EN_ONSETS = ["b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "r", "s", "t",
             "v", "w", "y", "z", "bl", "br", "ch", "cl", "cr", "dr", "fl", "fr", "gl",
             "gr", "pl", "pr", "sc", "sh", "sk", "sl", "sm", "sn", "sp", "st", "sw",
             "th", "tr", "tw", "wh", "str", "spr", ""]
EN_ONSET_W = [42, 30, 40, 25, 22, 32, 8, 12, 40, 38, 40, 34, 44, 60, 55, 18, 22, 12, 6,
              8, 10, 14, 12, 10, 12, 8, 8, 6, 12, 12, 14, 6, 14, 6, 8, 6, 6, 10, 24,
              20, 6, 26, 4, 8, 6, 3, 30]
EN_NUCLEI = ["a", "e", "i", "o", "u", "ai", "ea", "ee", "ie", "oa", "oo", "ou", "ay",
             "ey", "ow", "au", "oi"]
EN_NUCLEUS_W = [80, 95, 70, 62, 32, 14, 20, 16, 10, 10, 14, 16, 10, 8, 10, 8, 5]
EN_CODAS = ["", "b", "ck", "d", "ft", "g", "l", "ld", "lt", "m", "mp", "n", "nd", "ng",
            "nk", "nt", "p", "r", "rd", "rk", "rn", "rt", "s", "sh", "sk", "sp", "ss",
            "st", "t", "th", "v", "x", "z"]
EN_CODA_W = [70, 8, 14, 40, 6, 8, 36, 12, 8, 26, 6, 46, 20, 16, 8, 22, 16, 44, 12, 6, 8,
             14, 40, 10, 5, 4, 10, 26, 55, 14, 6, 4, 4]

# --------------------------------------------------------------------------------------
# Sentence-level knobs. Akshara shape/frequency, word length and the digit/Latin
# character shares are DERIVED from the input files; these are the ones that cannot be.
# --------------------------------------------------------------------------------------
DEFAULT_PROBS = {
    # --- rare-akshara emphasis ---
    "alpha": 0.35,             # weight = count ** alpha (1 = corpus, 0 = uniform)
    "oov_share": 0.35,         # target share of aksharas that are out-of-vocab
    "queue_per_line": {0: 0.35, 1: 0.35, 2: 0.20, 3: 0.10},   # coverage-queue injections

    # --- sentence shape ---
    "terminal": 0.55,          # line ends with . ? ! (else it is a mid-sentence fragment)
    "terminal_weights": {".": 0.80, "?": 0.10, "!": 0.10},
    "sentence_break": 0.04,    # per word-boundary: a sentence ends and another starts
                               # mid-line, which is what a cropped text line looks like
    "danda": 0.05,              # use । / ॥ as the terminal instead (needs a danda font)
    "clause_punct": 0.10,      # per word-boundary prob of , ; :
    "clause_weights": {",": 0.85, ";": 0.05, ":": 0.10},
    "paired": 0.10,            # per line prob of one bracket/quote pair
    "paired_span": {1: 0.45, 2: 0.30, 3: 0.15, 4: 0.10},
    "hyphen_compound": 0.03,   # per word prob of joining the next word with '-'
    "inline_symbol": 0.01,     # per word prob of a stray inline symbol as its own token

    # --- numbers (digit CHARACTER share is calibrated to the corpus; see number_word) ---
    "number_word": None,       # per-word prob of a number; None -> auto-calibrate
    "telugu_digit": 0.04,      # given a number, prob it uses Telugu digits
    "number_len": {1: 0.20, 2: 0.30, 3: 0.20, 4: 0.20, 5: 0.06, 6: 0.04},
    "number_decimal": 0.10,    # 12.5
    "number_grouped": 0.08,    # 1,20,000  (Indian grouping)
    "number_percent": 0.06,    # 25%
    "number_currency": 0.05,   # ₹1500
    "number_slashed": 0.04,    # 12/5/2024

    # --- English (Latin CHARACTER share is calibrated to the corpus) ---
    "english_line": 0.25,      # prob a line may contain English at all
    "english_word": None,      # per-word prob inside such a line; None -> auto-calibrate
    "english_syllables": {1: 0.35, 2: 0.45, 3: 0.20},
    "english_acronym": 0.20,   # given English, emit an ALL-CAPS acronym instead
    "acronym_len": {2: 0.40, 3: 0.40, 4: 0.15, 5: 0.05},
    "english_capitalized": 0.30,   # given a pseudo-word, Capitalise it
    "english_parenthesised": 0.15,  # wrap it in ( )
}

# CTC frames at the current max image width: 2048 // 8 = 256. The default leaves a small
# margin, because CTC also needs a blank frame between two IDENTICAL adjacent tokens.
DEFAULT_MAX_TOKENS = 240

# Smallest OOV boost used instead of exactly zero, so no sampling table is ever empty.
_MIN_BOOST = 1e-9


def _glen(s):
    return len(_GRAPHEME_RE.findall(s))


def _is_telugu(s):
    return any(TELUGU_LO <= ch <= TELUGU_HI for ch in s)


def _ends_in_akshara(word):
    """True if the word's last non-punctuation character is Telugu."""
    k = len(word)
    while k and not (word[k - 1].isalnum() or _is_telugu(word[k - 1])):
        k -= 1
    return k > 0 and _is_telugu(word[k - 1])


def is_wellformed_akshara(g):
    """True if `g` is a self-contained akshara, safe to place between other aksharas.

    Same empirical test as ``random_text_aksharas.py``: `g` must be exactly one grapheme
    cluster, must not be absorbed by a preceding akshara, and must not swallow a
    following one. That rejects bare matras, anusvara/visarga fragments, vattus like
    "్క", trailing viramas, and internal dangling viramas.
    """
    if not g or any(ch.isspace() for ch in g):
        return False
    if unicodedata.category(g[0]) in ("Mn", "Mc", "Cf"):
        return False
    if _glen(g) != 1:
        return False
    if _glen(_PROBE + g) != 2:            # would merge into a preceding akshara
        return False
    if _glen(g + _PROBE) != 2:            # would swallow the following akshara
        return False
    return True


def follows_telugu_orthography(g):
    """True if `g` obeys Telugu encoding rules for a single akshara.

    ``is_wellformed_akshara`` only checks cluster boundaries, so misencoded clusters from
    scraped / OCR'd text survive it — and at a threshold of 100 occurrences in a 3.3B
    grapheme corpus, plenty of that noise clears the bar (~4% of the inventory). These
    are not rare aksharas, they are broken ones, and they would teach the model a
    codepoint sequence no correct text produces:

        వెు   ె + ు      two matras (a misencoding of వొ)
        మీ్స  matra then virama, i.e. the vattu encoded after the vowel sign
        యంం   doubled anusvara
        0ు    a digit carrying a vowel sign

    Rules: the base is an independent vowel or a consonant; an independent vowel takes
    neither a matra nor a vattu (it is already the nucleus — "ఋా" is not a thing, though
    "అం" is); at most one matra; no virama after a matra (vattulu are encoded first); at
    most one sign, and it comes last.
    """
    vowel_base = g[0] in INDEP_VOWEL_CHARS
    if not vowel_base and g[0] not in CONSONANT_CHARS:
        return False
    n_matra = n_sign = 0
    for ch in g:
        if ch in MATRA_CHARS:
            n_matra += 1
        elif ch == VIRAMA:
            if n_matra or vowel_base:
                return False
        elif ch in SIGN_CHARS:
            n_sign += 1
    if n_matra > 1 or n_sign > 1 or (vowel_base and n_matra):
        return False
    return not n_sign or g[-1] in SIGN_CHARS


def akshara_shape(g):
    """Return a hashable SHAPE key for an akshara: (base_kind, n_vattu, matra, sign).

    The shape is the akshara minus the identity of its consonants, e.g. "క్కా" and "స్తీ"
    differ in identity but share the shape ('C', 1, 'ా'-vs-'ీ' ...). Shapes are what the
    word-position model is estimated over, so a rare akshara can be placed wherever its
    shape legitimately occurs.
    """
    base = "V" if g[0] in INDEP_VOWEL_CHARS else ("C" if _is_telugu(g[0]) else "X")
    n_vattu = min(g.count(VIRAMA), 2)
    matra = "".join(ch for ch in g if ch in MATRA_CHARS)
    sign = "".join(ch for ch in g if ch in SIGN_CHARS)
    return (base, n_vattu, matra, sign)


def _norm(counter):
    """Counter -> (keys, cumulative weights) for random.choices(cum_weights=...)."""
    keys = list(counter)
    cum, run = [], 0.0
    for k in keys:
        run += counter[k]
        cum.append(run)
    return keys, cum


def _pick(rng, keys, cum):
    return rng.choices(keys, cum_weights=cum, k=1)[0]


def _tilt_to_mean(len_weights, target_mean, lo=0.05, hi=20.0, iters=60):
    """Exponentially tilt a length distribution so its mean becomes `target_mean`.

    A word LIST gives the distribution over word *types*, which skews long: running text
    repeats short words far more often. The corpus space frequency gives the true mean
    word length, so we reweight by ``lam ** length`` and bisect on ``lam`` to match it.
    """
    lens = sorted(len_weights)
    w = [len_weights[l] for l in lens]

    def mean_at(lam):
        num = sum(wi * lam ** l * l for wi, l in zip(w, lens))
        den = sum(wi * lam ** l for wi, l in zip(w, lens))
        return num / den if den else 0.0

    if mean_at(1.0) > target_mean:
        lo, hi = lo, 1.0
    else:
        lo, hi = 1.0, hi
    for _ in range(iters):
        mid = (lo + hi) / 2
        if mean_at(mid) < target_mean:
            lo = mid
        else:
            hi = mid
    lam = (lo + hi) / 2
    return {l: len_weights[l] * lam ** l for l in lens}


class RandomTeluguTextGenerator:
    """Generate random Telugu lines that follow real text distributions.

    Args:
        vocab_file: tokenizer vocab JSON (``telugu-vocab.json``). Decides which aksharas
            are in-vocab (1 token) vs OOV (one token per codepoint), and which symbols are
            usable at all.
        dist_files: list of grapheme-distribution JSONs (Telugu + Sanskrit), merged by
            summing counts. This is the only source of aksharas — nothing is invented.
        word_file: optional word list (one word per line) used for the word-length
            distribution and the shape-per-position model. Without it, both fall back to
            corpus-wide shape frequencies and a fixed length distribution.
        english_dist_file: optional English grapheme distribution, used to weight acronym
            letters.
        min_freq: keep only aksharas whose merged corpus count is >= this. Lower values
            widen the rare tail: with the Telugu + Sanskrit dists, freq>=100 gives ~8.3k
            OOV aksharas, freq>=20 about twice that. Note that a lower threshold also
            admits more corpus noise, which ``follows_telugu_orthography`` only partly
            filters, so check a sample of the output when lowering it.
        min_char / max_char: line length in GRAPHEMES (spaces and punctuation counted),
            drawn uniformly from the range when `generate` gets no explicit target. At
            height 64 and the usual font sizes, ~80 graphemes is around the 2048px width
            cap, so keep `max_char` consistent with the image pipeline.
        max_tokens: hard cap on tokenizer tokens per line (default 240, for the 256 CTC
            frames of a 2048px image). Lines never exceed it: when the budget gets tight
            the generator switches to in-vocab aksharas rather than shortening the line.
        seed: int for a reproducible RNG (seeded once, at construction).
        probs: dict overriding DEFAULT_PROBS.
    """

    def __init__(self, vocab_file, dist_files, word_file=None, english_dist_file=None,
                 min_freq=100, min_char=20, max_char=80,
                 max_tokens=DEFAULT_MAX_TOKENS, seed=None, probs=None):
        # Every grapheme costs at least one token (an in-vocab akshara, a space, a digit,
        # a punctuation mark), so a line of `max_char` graphemes can never fit in fewer
        # than `max_char` tokens. Without this check the cap would be silently violated.
        if max_tokens < max_char:
            raise ValueError(
                f"max_tokens={max_tokens} < max_char={max_char}: a line of {max_char} "
                f"graphemes needs at least {max_char} tokens. Either lower max_char or "
                f"raise max_tokens (frames = image_width // 8).")

        self.rng = random.Random(seed)
        self.min_char = min_char
        self.max_char = max_char
        self.max_tokens = max_tokens
        self.p = dict(DEFAULT_PROBS)
        if probs:
            self.p.update(probs)

        with open(vocab_file, encoding="utf-8") as f:
            self.vocab = json.load(f)

        counts = collections.Counter()
        for path in dist_files:
            with open(path, encoding="utf-8") as f:
                counts.update(json.load(f))
        self._corpus_counts = counts
        self._corpus_total = sum(counts.values())

        # Order matters: the word model filters itself against the akshara shapes that
        # exist, and the OOV boost is calibrated against the word model's shape mixture,
        # so the sampling tables can only be built once both are known.
        self._build_inventory(min_freq)
        self._build_symbols(english_dist_file)
        self._build_word_model(word_file)
        self._calibrate_oov_boost()
        self._build_sampling_tables()

        # Coverage queue over the OOV inventory: shuffled once, cycled, reshuffled on
        # wrap, so every rare akshara is emitted before any is emitted a second time.
        self._queue = list(self.oov_aksharas)
        self.rng.shuffle(self._queue)
        self._queue_pos = 0
        self.queue_cycles = 0
        self._qk, self._qc = _norm(collections.Counter(self.p["queue_per_line"]))

        # Needs a working generator, so it runs last.
        self._calibrate_mixing()

    # ---------------------------------------------------------------------------------
    # Construction
    # ---------------------------------------------------------------------------------
    def _build_inventory(self, min_freq):
        """Filter the merged distribution down to attested, well-formed aksharas."""
        alpha = self.p["alpha"]
        self.aksharas = []          # every usable akshara
        self.oov_aksharas = []      # the out-of-vocab subset (fallback-path targets)
        self.token_cost = {}        # akshara -> tokens, same rule as the tokenizer
        self.shape = {}
        self._w_in = {}             # tempered weight, in-vocab aksharas
        self._w_oov = {}            # tempered weight, out-of-vocab aksharas

        for g, count in self._corpus_counts.items():
            if count < min_freq or not _is_telugu(g) or not is_wellformed_akshara(g):
                continue
            if any(ch in NON_AKSHARA_TELUGU or ch in UNRENDERABLE_CHARS for ch in g) or \
                    not follows_telugu_orthography(g):
                continue
            in_vocab = g in self.vocab
            # An OOV akshara must decompose into in-vocab codepoints, otherwise the
            # tokenizer emits [UNK] and the sample teaches nothing.
            if not in_vocab and any(cp not in self.vocab for cp in g):
                continue
            self.aksharas.append(g)
            self.token_cost[g] = 1 if in_vocab else len(g)
            self.shape[g] = akshara_shape(g)
            (self._w_in if in_vocab else self._w_oov)[g] = count ** alpha
            if not in_vocab:
                self.oov_aksharas.append(g)

        if not self.aksharas:
            raise ValueError(f"no attested well-formed aksharas with count >= {min_freq}")
        self._shapes_available = set(self.shape.values())
        # Per-shape weight totals, needed to calibrate the boost before sampling tables
        # exist. (Word shapes seen in the word list but absent here are dropped there.)
        self._shape_w_in = collections.Counter()
        self._shape_w_oov = collections.Counter()
        for g, w in self._w_in.items():
            self._shape_w_in[self.shape[g]] += w
        for g, w in self._w_oov.items():
            self._shape_w_oov[self.shape[g]] += w

    def _calibrate_oov_boost(self):
        """Find the boost factor that makes the realised OOV share hit `oov_share`.

        Aksharas are not drawn from one global distribution: a SHAPE is drawn first (from
        the word-position model) and the akshara is drawn within that shape. So the
        realised OOV share is the shape-weighted average

            E[oov | b] = sum_s P(s) * b*W_oov(s) / (W_in(s) + b*W_oov(s))

        which the closed-form global ratio gets wrong by ~2x. Bisect on `b` instead. The
        coverage queue injects OOV aksharas on top of this, so its expected contribution
        is subtracted from the target first.
        """
        share = self.p["oov_share"]
        # Expected queue injections per line, as a share of the line's aksharas.
        exp_queue = sum(k * w for k, w in self.p["queue_per_line"].items()) / sum(
            self.p["queue_per_line"].values())
        mean_target = (self.min_char + self.max_char) / 2
        exp_aksharas = max(1.0, mean_target * self.mean_word_len / (self.mean_word_len + 1))
        residual = max(0.0, share - exp_queue / exp_aksharas)

        shapes = list(self._shape_marginal)
        p_s = [self._shape_marginal[s] for s in shapes]
        w_in = [self._shape_w_in.get(s, 0.0) for s in shapes]
        w_oov = [self._shape_w_oov.get(s, 0.0) for s in shapes]

        def realised(b):
            out = 0.0
            for ps, wi, wo in zip(p_s, w_in, w_oov):
                denom = wi + b * wo
                if denom:
                    out += ps * (b * wo) / denom
            return out

        # The floor keeps every akshara's sampling weight positive: a shape whose only
        # members are OOV would otherwise end up with a zero-weight table (which happens
        # for short lines, where the coverage queue alone already meets the target).
        if residual <= 0 or not any(w_oov):
            self.oov_boost = _MIN_BOOST
        elif realised(1e9) < residual:      # target unreachable; use the maximum
            self.oov_boost = 1e9
        else:
            lo, hi = _MIN_BOOST, 1e9
            for _ in range(200):
                mid = (lo * hi) ** 0.5      # geometric bisection: b spans many decades
                if realised(mid) < residual:
                    lo = mid
                else:
                    hi = mid
            self.oov_boost = max((lo * hi) ** 0.5, _MIN_BOOST)
        self.oov_share_random = realised(self.oov_boost)

    def _build_sampling_tables(self):
        """Per-shape cumulative-weight tables, with the calibrated OOV boost applied.

        Two variants per shape: the normal one, and an in-vocab-only one used when the
        token budget is tight (in-vocab aksharas cost 1 token instead of ~4.6).
        """
        self.weights = {g: self._w_in.get(g, 0.0) + self.oov_boost * self._w_oov.get(g, 0.0)
                        for g in self.aksharas}
        by_shape = collections.defaultdict(collections.Counter)
        by_shape_in = collections.defaultdict(collections.Counter)
        for g in self.aksharas:
            by_shape[self.shape[g]][g] = self.weights[g]
            if self.token_cost[g] == 1:
                by_shape_in[self.shape[g]][g] = self.weights[g]
        self._shape_tab = {s: _norm(c) for s, c in by_shape.items()}
        self._shape_tab_in = {s: _norm(c) for s, c in by_shape_in.items() if c}
        self._all_tab_in = _norm(collections.Counter(
            {g: self.weights[g] for g in self.aksharas if self.token_cost[g] == 1}))

    def _build_symbols(self, english_dist_file):
        """Build punctuation / digit / Latin tables, weighted by corpus frequency.

        Only symbols present in the vocab are kept. In-vocab symbols the corpus has never
        seen (₹ was added to the vocab deliberately; Telugu digits and dandas are almost
        absent) get a small frequency floor so they are still exercised.
        """
        cnt, tot = self._corpus_counts, self._corpus_total
        floor = 1e-6 * tot          # 1 ppm, so vocab-only symbols still appear

        def usable(ch):
            return ch in self.vocab

        def weights(chars):
            return collections.Counter(
                {ch: max(cnt.get(ch, 0), floor) for ch in chars if usable(ch)})

        p = self.p
        term = collections.Counter(
            {ch: p["terminal_weights"].get(ch, 0.0) for ch in TERMINAL_PUNCT
             if usable(ch)})
        self._term_tab = _norm(term)
        self._danda_tab = _norm(weights(DANDA_PUNCT)) if all(
            usable(ch) for ch in DANDA_PUNCT) else None
        self._clause_tab = _norm(collections.Counter(
            {ch: p["clause_weights"].get(ch, 0.0) for ch in CLAUSE_PUNCT if usable(ch)}))
        self._paired = [(a, b) for a, b in PAIRED_PUNCT if usable(a) and usable(b)]
        self._paired_tab = _norm(collections.Counter(
            {i: max(cnt.get(a, 0), floor) for i, (a, _) in enumerate(self._paired)}))
        self._inline_tab = _norm(weights(INLINE_PUNCT))
        self._arabic_tab = _norm(weights(ARABIC_DIGITS))
        self._telugu_digit_tab = _norm(weights(TELUGU_DIGITS))
        self.has_currency = usable(CURRENCY)
        self.has_percent = usable(PERCENT)
        self.has_hyphen = usable(HYPHEN)

        self._latin_lower = [ch for ch in LATIN_LOWER if usable(ch)]
        self._latin_upper = [ch for ch in LATIN_UPPER if usable(ch)]
        en_counts = None
        if english_dist_file:
            with open(english_dist_file, encoding="utf-8") as f:
                en_counts = json.load(f)
        self._acronym_tab = _norm(collections.Counter(
            {ch: (en_counts.get(ch.lower(), 1) if en_counts else 1)
             for ch in self._latin_upper})) if self._latin_upper else None
        self._en_pieces = [
            (EN_ONSETS, EN_ONSET_W), (EN_NUCLEI, EN_NUCLEUS_W), (EN_CODAS, EN_CODA_W)]
        # Latin pieces are useless if the letters are not in the vocab.
        self._english_ok = bool(self._latin_lower) and bool(self._latin_upper)

        # Corpus character shares, used to calibrate how often numbers / English appear.
        self.digit_share = sum(
            cnt.get(ch, 0) for ch in ARABIC_DIGITS + TELUGU_DIGITS) / tot
        self.latin_share = sum(
            cnt.get(ch, 0) for ch in LATIN_LOWER + LATIN_UPPER) / tot

    def _build_word_model(self, word_file):
        """Word-length distribution and P(shape | position-in-word), from a word list.

        Position buckets: 'S' single-akshara word, 'I' initial, 'M' medial, 'F' final.
        The type-level length distribution is tilted so its mean matches the corpus mean
        word length implied by the space frequency.
        """
        len_counts = collections.Counter()
        shape_pos = collections.defaultdict(collections.Counter)
        if word_file:
            with open(word_file, encoding="utf-8") as f:
                for raw in f:
                    w = raw.strip()
                    if not w or len(w) > 40 or not all(
                            TELUGU_LO <= ch <= TELUGU_HI for ch in w):
                        continue
                    gs = _GRAPHEME_RE.findall(w)
                    if not all(g in self.token_cost or is_wellformed_akshara(g)
                               for g in gs):
                        continue
                    n = len(gs)
                    len_counts[n] += 1
                    for i, g in enumerate(gs):
                        bucket = "S" if n == 1 else ("I" if i == 0
                                                     else "F" if i == n - 1 else "M")
                        shape_pos[bucket][akshara_shape(g)] += 1

        if not len_counts:
            len_counts = collections.Counter(
                {1: 3, 2: 10, 3: 18, 4: 20, 5: 17, 6: 13, 7: 9, 8: 5, 9: 3, 10: 2})

        # True mean word length from the corpus: with a space share s, a "word + its
        # following space" costs 1/s graphemes, so the word itself is 1/s - 1.
        space_share = self._corpus_counts.get(" ", 0) / self._corpus_total
        self.corpus_word_len = (1.0 / space_share - 1.0) if space_share else None
        if self.corpus_word_len:
            len_counts = collections.Counter(
                _tilt_to_mean(dict(len_counts), self.corpus_word_len))
        self._len_tab = _norm(len_counts)
        self.mean_word_len = (sum(l * c for l, c in len_counts.items())
                              / sum(len_counts.values()))

        # Keep only shapes we can actually realise, then invert for queue placement.
        self._pos_tab, bucket_by_shape = {}, collections.defaultdict(collections.Counter)
        bucket_totals = collections.Counter()
        for bucket, shapes in shape_pos.items():
            kept = collections.Counter(
                {s: c for s, c in shapes.items() if s in self._shapes_available})
            if kept:
                self._pos_tab[bucket] = _norm(kept)
                bucket_totals[bucket] = sum(kept.values())
                for s, c in kept.items():
                    bucket_by_shape[s][bucket] = c
        self._bucket_tab = {s: _norm(c) for s, c in bucket_by_shape.items()}

        # P(shape), marginalised over positions — the mixture the OOV boost is calibrated
        # against. Falls back to corpus shape frequencies when there is no word list.
        marginal = collections.Counter()
        total_buckets = sum(bucket_totals.values())
        if total_buckets:
            for bucket, (shapes, cum) in self._pos_tab.items():
                p_bucket = bucket_totals[bucket] / total_buckets
                prev = 0.0
                for s, c in zip(shapes, cum):
                    marginal[s] += p_bucket * (c - prev) / cum[-1]
                    prev = c
        else:
            corpus_w = collections.Counter()
            for g in self.aksharas:
                corpus_w[self.shape[g]] += self._corpus_counts[g]
            tot = sum(corpus_w.values())
            marginal = collections.Counter({s: c / tot for s, c in corpus_w.items()})
        self._shape_marginal = marginal
        self._shape_prior = _norm(marginal)

    def _calibrate_mixing(self, n_lines=800, iters=3):
        """Set the per-word number / English rates so the CHARACTER shares match the corpus.

        The first guess is analytic (share ≈ P(word is a number) * mean number length /
        mean word length), then refined empirically: a word that does not fit the line's
        remaining budget falls back to Telugu, which biases the realised share downwards
        by an amount that is not worth modelling. A few hundred throwaway lines per
        iteration are enough to correct it.
        """
        p = self.p
        auto_number = p["number_word"] is None
        auto_english = p["english_word"] is None
        mean_num_len = sum(l * w for l, w in p["number_len"].items()) / sum(
            p["number_len"].values())
        if auto_number:
            p["number_word"] = min(
                0.5, self.digit_share * self.mean_word_len / mean_num_len)
        if auto_english:
            mean_en_len = 4.5   # pseudo-word mean over the syllable tables
            share_within = (self.latin_share / p["english_line"]) if p["english_line"] \
                else 0.0
            p["english_word"] = min(
                0.5, share_within * self.mean_word_len / mean_en_len)
        if not (auto_number or auto_english):
            return

        latin_all = set(LATIN_LOWER + LATIN_UPPER)
        digits_all = set(ARABIC_DIGITS + TELUGU_DIGITS)
        r = random.Random(0x5EED)       # own stream: does not disturb self.rng
        queue_state = (list(self._queue), self._queue_pos, self.queue_cycles)
        for _ in range(iters):
            n_g = n_digit = n_latin = 0
            for _ in range(n_lines):
                text, st = self.generate_line(rng=r)
                n_g += st["graphemes"]
                n_digit += sum(1 for ch in text if ch in digits_all)
                n_latin += sum(1 for ch in text if ch in latin_all)
            if auto_number and n_digit:
                ratio = self.digit_share / (n_digit / n_g)
                p["number_word"] = min(0.5, p["number_word"] * min(max(ratio, 0.5), 2.0))
            if auto_english and n_latin:
                ratio = self.latin_share / (n_latin / n_g)
                p["english_word"] = min(0.5, p["english_word"] * min(max(ratio, 0.5), 2.0))
        self._queue, self._queue_pos, self.queue_cycles = queue_state

    # ---------------------------------------------------------------------------------
    # Unit sampling
    # ---------------------------------------------------------------------------------
    def _next_queued(self, r):
        """Pop the next akshara from the rare-akshara coverage queue."""
        if not self._queue:
            return None
        if self._queue_pos >= len(self._queue):
            r.shuffle(self._queue)
            self._queue_pos = 0
            self.queue_cycles += 1
        g = self._queue[self._queue_pos]
        self._queue_pos += 1
        return g

    def _sample_akshara(self, r, bucket, cheap):
        """Sample one akshara for position bucket `bucket`.

        `cheap` restricts the draw to in-vocab aksharas (1 token each), which is how the
        token budget is respected without shortening the line.
        """
        tab = self._pos_tab.get(bucket)
        shape = _pick(r, *tab) if tab else _pick(r, *self._shape_prior)
        if cheap:
            st = self._shape_tab_in.get(shape)
            return _pick(r, *st) if st else _pick(r, *self._all_tab_in)
        return _pick(r, *self._shape_tab[shape])

    def _telugu_word(self, r, length, cheap, queued=None):
        """Build one Telugu word of `length` aksharas.

        `queued` is a rare akshara from the coverage queue; it is placed at a position
        whose bucket its shape actually occurs in, so the word stays plausible.
        """
        buckets = ["S"] if length == 1 else \
            ["I"] + ["M"] * (length - 2) + ["F"]
        slot = None
        if queued is not None:
            tab = self._bucket_tab.get(self.shape[queued])
            want = _pick(r, *tab) if tab else "M"
            options = [i for i, b in enumerate(buckets) if b == want]
            slot = r.choice(options) if options else r.randrange(length)
        out = []
        for i, bucket in enumerate(buckets):
            if i == slot:
                out.append(queued)
            else:
                out.append(self._sample_akshara(r, bucket, cheap))
        return out

    def _grow_tail(self, word, n, r, cheap):
        """Append `n` graphemes to `word`, in whatever script it ends in.

        Used only to land a line on its exact grapheme target. Trailing punctuation stays
        trailing (so a comma never ends up mid-word), a number grows by digits of its own
        script, and an English word grows by letters — never a Telugu akshara glued onto
        a Latin word. Returns ``(new_word, units_added)``.
        """
        k = len(word)
        while k and not (word[k - 1].isalnum() or _is_telugu(word[k - 1])):
            k -= 1
        core, tail = word[:k], word[k:]
        last = core[-1:]
        if last and last in ARABIC_DIGITS:
            units = [_pick(r, *self._arabic_tab) for _ in range(n)]
        elif last and last in TELUGU_DIGITS:
            units = [_pick(r, *self._telugu_digit_tab) for _ in range(n)]
        elif last and last in LATIN_LOWER and self._latin_lower:
            units = [r.choice(self._latin_lower) for _ in range(n)]
        elif last and last in LATIN_UPPER and self._latin_upper:
            units = [r.choice(self._latin_upper) for _ in range(n)]
        else:
            units = [self._sample_akshara(r, "F" if i == n - 1 else "M", cheap)
                     for i in range(n)]
        return core + "".join(units) + tail, units

    def _number(self, r):
        """A whole number token: one script throughout, optional decoration."""
        p = self.p
        telugu = r.random() < p["telugu_digit"]
        digits = self._telugu_digit_tab if telugu else self._arabic_tab
        zero = TELUGU_DIGITS[0] if telugu else ARABIC_DIGITS[0]
        n = _pick(r, *_norm(collections.Counter(p["number_len"])))
        body = "".join(_pick(r, *digits) for _ in range(n))
        # Real numbers do not have leading zeros (except a lone 0).
        while n > 1 and body[0] == zero:
            body = _pick(r, *digits) + body[1:]
        x = r.random()
        acc = p["number_decimal"]
        if x < acc:
            frac = "".join(_pick(r, *digits) for _ in range(r.randint(1, 2)))
            return f"{body}.{frac}"
        acc += p["number_grouped"]
        if x < acc and n >= 4:
            head, tail = body[:-3], body[-3:]
            groups = []
            while len(head) > 2:
                groups.insert(0, head[-2:])
                head = head[:-2]
            if head:
                groups.insert(0, head)
            return ",".join(groups + [tail])
        acc += p["number_percent"]
        if x < acc and self.has_percent:
            return body + PERCENT
        acc += p["number_currency"]
        if x < acc and self.has_currency:
            return CURRENCY + body
        acc += p["number_slashed"]
        if x < acc:
            other = "".join(_pick(r, *digits) for _ in range(r.randint(1, 2)))
            return f"{body}/{other}"
        return body

    def _english_word(self, r):
        """A pronounceable pseudo-English word, or an acronym. Never a real word."""
        p = self.p
        if self._acronym_tab and r.random() < p["english_acronym"]:
            n = _pick(r, *_norm(collections.Counter(p["acronym_len"])))
            return "".join(_pick(r, *self._acronym_tab) for _ in range(n))
        n_syl = _pick(r, *_norm(collections.Counter(p["english_syllables"])))
        out = []
        for i in range(n_syl):
            onset = r.choices(EN_ONSETS, weights=EN_ONSET_W, k=1)[0]
            nucleus = r.choices(EN_NUCLEI, weights=EN_NUCLEUS_W, k=1)[0]
            coda = r.choices(EN_CODAS, weights=EN_CODA_W, k=1)[0] \
                if i == n_syl - 1 or r.random() < 0.3 else ""
            if i == 0 and not onset and n_syl > 1:
                onset = "s"
            out.append(onset + nucleus + coda)
        word = "".join(out)
        if r.random() < p["english_capitalized"]:
            word = word[0].upper() + word[1:]
        return word

    # ---------------------------------------------------------------------------------
    # Line assembly
    # ---------------------------------------------------------------------------------
    def _cost(self, s):
        """Tokenizer token cost of a string, by the tokenizer's own rule."""
        total = 0
        for g in _GRAPHEME_RE.findall(s):
            total += 1 if g in self.vocab else len(g)
        return total

    def generate(self, target=None, rng=None):
        """Build and return one random line (text only)."""
        return self.generate_line(target=target, rng=rng)[0]

    def generate_line(self, target=None, rng=None):
        """Build one random line and report its cost.

        Returns ``(text, stats)`` where stats is a dict with:
            graphemes  line length in grapheme clusters (== `target`)
            tokens     tokenizer tokens (in-vocab akshara 1, OOV akshara 1 per codepoint)
            frames     CTC frames required = tokens + adjacent duplicate tokens
            oov        number of out-of-vocab aksharas in the line

        target: exact line length in graphemes; if None, drawn uniformly from
            [min_char, max_char].
        rng: optional random.Random (defaults to the instance RNG); pass a per-worker RNG
            to avoid duplicated sequences across processes.
        """
        r = rng if rng is not None else self.rng
        p = self.p
        target = r.randint(self.min_char, self.max_char) if target is None else target

        # --- per-line decisions -------------------------------------------------------
        use_danda = self._danda_tab is not None and r.random() < p["danda"]
        terminal = ""
        if r.random() < p["terminal"]:
            terminal = _pick(r, *(self._danda_tab if use_danda else self._term_tab))
        allow_english = self._english_ok and r.random() < p["english_line"]
        queue_left = _pick(r, *(self._qk, self._qc))   # rare aksharas to inject

        open_ch = close_ch = None
        pair_words_left = 0
        if self._paired and r.random() < p["paired"]:
            open_ch, close_ch = self._paired[_pick(r, *self._paired_tab)]
            pair_start = r.random() < 0.5   # else the pair opens later in the line
            pair_span = _pick(r, *_norm(collections.Counter(p["paired_span"])))
        else:
            pair_start, pair_span = False, 0

        words = []          # finished words, punctuation attached
        n_g = len(terminal)  # graphemes committed (the terminal is reserved up front)
        tokens = self._cost(terminal)
        n_oov = 0
        prev_clause = True   # no clause punctuation before the first word
        prev_kind = None     # so two numbers never end up side by side
        glue = False         # this word attaches to the previous one (hyphen compound)

        while n_g < target:
            space = 0 if glue else (1 if words else 0)
            reserved = 1 if pair_words_left > 0 else 0      # room for the closing char
            remaining = target - n_g - space - reserved
            if remaining <= 0:
                break

            # Open a bracket / quote pair? Both characters have to be paid for here: the
            # opening one now, and the closing one whichever word it lands on (`reserved`
            # only covers pairs that are already open).
            prefix = ""
            if open_ch and pair_words_left == 0 and not glue and remaining >= 3 and \
                    (pair_start if not words else r.random() < 0.15):
                prefix, pair_words_left = open_ch, pair_span
                open_ch = None
                remaining -= 2

            # --- pick what this word is ---------------------------------------------
            kind = "telugu"
            x = r.random()
            acc = p["number_word"]
            if x < acc and prev_kind != "number":
                kind = "number"
            elif x < acc + p["inline_symbol"] and words and prev_kind != "symbol":
                kind = "symbol"          # a lone / & * | between words, as in documents
            elif allow_english and x < acc + p["inline_symbol"] + p["english_word"]:
                kind = "english"

            if kind == "number":
                core = self._number(r)
                if _glen(core) > remaining:
                    kind = "telugu"
            elif kind == "symbol":
                core = _pick(r, *self._inline_tab)
            elif kind == "english":
                core = self._english_word(r)
                # Only parenthesise when no other pair is open, so brackets never nest.
                if pair_words_left == 0 and not prefix and \
                        r.random() < p["english_parenthesised"] and \
                        remaining >= _glen(core) + 2:
                    core = "(" + core + ")"
                if _glen(core) > remaining:
                    kind = "telugu"

            if kind == "telugu":
                # Token budget. Every grapheme still to be placed — including the spaces
                # between words — costs at least one token, so `slack` is how many EXTRA
                # tokens (above that floor) the rest of the line can still afford, i.e.
                # the room left for OOV aksharas, which cost one token per codepoint.
                slack = self.max_tokens - tokens - (target - n_g)
                cheap = slack <= 0
                length = min(_pick(r, *self._len_tab), remaining)
                queued = None
                if queue_left and not cheap and \
                        r.random() < queue_left / max(1, length):
                    queued = self._next_queued(r)
                units = self._telugu_word(r, length, cheap, queued)
                if queued is not None:
                    queue_left -= 1
                cost = sum(self.token_cost[u] for u in units)
                if cost - length > slack:       # would break the cap: redo in-vocab only
                    units = self._telugu_word(r, length, True)
                    cost = length
                core = "".join(units)
                oov_here = sum(1 for u in units if self.token_cost[u] > 1)
            else:
                cost = self._cost(core)         # numbers / English are all in-vocab
                oov_here = 0

            word = prefix + core
            n_oov += oov_here
            prev_kind = kind

            # --- trailing punctuation ------------------------------------------------
            if pair_words_left > 0:
                pair_words_left -= 1
                if pair_words_left == 0:
                    word += close_ch

            # A pair still open after this word keeps one grapheme reserved for its
            # closing character, so punctuation here must not spend it.
            pending = 1 if pair_words_left > 0 else 0
            room = target - n_g - space - _glen(word) - pending
            if room >= 2 and not prev_clause and r.random() < p["sentence_break"]:
                word += _pick(r, *(self._danda_tab if use_danda else self._term_tab))
                prev_clause = True          # a new sentence starts after this word
            elif room >= 1 and not prev_clause and r.random() < p["clause_punct"]:
                word += _pick(r, *self._clause_tab)
                prev_clause = True
            else:
                prev_clause = False

            if glue:
                words[-1] += word          # hyphen compound: no space in between
            else:
                words.append(word)
            n_g += space + _glen(word)
            tokens += space + self._cost(word)      # the joining space is a token too

            # Hyphenated compound: the NEXT word attaches with no space. Skipped after
            # punctuation, so lines never contain ",-".
            tail_ok = bool(word) and _is_telugu(word[-1])   # not after a number/bracket
            glue = (self.has_hyphen and n_g + 2 + pending < target and tail_ok
                    and r.random() < p["hyphen_compound"])
            if glue:
                words[-1] += HYPHEN
                n_g += 1
                tokens += 1

        # A pair whose span outlived the line's budget still has to be closed, using the
        # grapheme `reserved` kept free for it while it was open.
        if pair_words_left > 0 and words:
            words[-1] += close_ch
            n_g += 1
            tokens += 1

        # --- land exactly on the target ----------------------------------------------
        # Whatever the word loop could not place (it stops when the next space + word no
        # longer fits) is absorbed by extending the last word with position-appropriate
        # aksharas, so the line lands on `target` graphemes exactly.
        pad = target - n_g
        if pad > 0:
            # `pad` counts every grapheme still missing, so the token floor for the rest
            # of the line is `pad` whether or not one of them turns into a space.
            slack = self.max_tokens - tokens - pad
            if not words:
                words.append("")
            # A new word is preferable when the last one does not end in an akshara, but
            # it costs a space; with a single grapheme left, grow the last token instead.
            if pad >= 2 and not _ends_in_akshara(words[-1]):
                words.append("")
                pad -= 1
                tokens += 1                     # the joining space
            last, units = self._grow_tail(words[-1], pad, r, slack <= 0)
            extra = sum(self.token_cost.get(u, 1) for u in units) - len(units)
            if extra > slack:                   # same cap check as in the word loop
                last, units = self._grow_tail(words[-1], pad, r, True)
            words[-1] = last
            tokens += sum(self.token_cost.get(u, 1) for u in units)
            n_oov += sum(1 for u in units if self.token_cost.get(u, 1) > 1)

        # Two artefacts of building the line under a fixed grapheme budget: a trailing
        # hyphen with nothing glued after it, and a clause comma directly before the
        # terminal (",."). Swap the offending character for an akshara, which keeps the
        # line at exactly `target` graphemes.
        stray = set(HYPHEN + (CLAUSE_PUNCT + TERMINAL_PUNCT if terminal else ""))
        if words and words[-1][-1:] in stray:
            # _grow_tail keeps the replacement in the word's own script, so a comma at the
            # end of an English word is not replaced by an akshara. The replacement is one
            # grapheme for one, so its extra token cost must fit what the cap has left.
            core = words[-1][:-1]
            last, units = self._grow_tail(core, 1, r, tokens >= self.max_tokens)
            extra = sum(self.token_cost.get(u, 1) for u in units) - 1
            if tokens + extra > self.max_tokens:
                last, units = self._grow_tail(core, 1, r, True)
                extra = 0
            words[-1] = last
            tokens += extra
            n_oov += sum(1 for u in units if self.token_cost.get(u, 1) > 1)

        text = " ".join(words) + terminal
        stats = {"graphemes": _glen(text), "tokens": self._cost(text),
                 "frames": self.frames_needed(text), "oov": n_oov}
        return text, stats

    def frames_needed(self, text):
        """CTC frames this label sequence needs: tokens + blanks between equal neighbours."""
        toks = []
        for g in _GRAPHEME_RE.findall(text):
            toks.extend([g] if g in self.vocab else list(g))
        return len(toks) + sum(1 for a, b in zip(toks, toks[1:]) if a == b)

    def generate_many(self, num_sentences, detailed=False):
        """Return `num_sentences` lines: strings, or (text, stats) pairs if `detailed`."""
        if detailed:
            return [self.generate_line() for _ in range(num_sentences)]
        return [self.generate() for _ in range(num_sentences)]


if __name__ == "__main__":
    import math
    import os
    import sys

    # ---- inline arguments ----------------------------------------------------------
    ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    TOK_DIR = os.path.join(ROOT, "src", "telugu_ocr", "tokenizer", "assets")
    DIST_DIR = os.path.join(TOK_DIR, "token_dist")

    vocab_file = os.path.join(TOK_DIR, "telugu-vocab.json")
    dist_files = [os.path.join(DIST_DIR, "telugu_grapheme_dist.json"),
                  os.path.join(DIST_DIR, "sanskrit_grapheme_dist.json")]
    english_dist_file = os.path.join(DIST_DIR, "english_grapheme_dist.json")
    # NOTE(phase-2): this pointed at data_curation/synthetic/, which was gitignored local
    # data and is now gone. Repointed under the /data/ root to match every other local
    # artifact; confirm the location if this script is ever run again.
    word_file = os.path.join(ROOT, "data", "synthetic",
                             "word_image_generator", "vocab.txt")

    min_freq = 100          # rare-tail threshold: 100 -> ~8.7k OOV aksharas, 20 -> ~17k
    min_char, max_char = 10, 70
    max_tokens = 240        # 2048 // 8 frames, minus a margin for CTC repeat-blanks
    num_report_lines = 20   # lines to print
    num_stat_lines = 5000   # lines to measure the distributions over
    seed = None

    gen = RandomTeluguTextGenerator(
        vocab_file=vocab_file, dist_files=dist_files, word_file=word_file,
        english_dist_file=english_dist_file, min_freq=min_freq,
        min_char=min_char, max_char=max_char, max_tokens=max_tokens, seed=seed)

    n_oov_inv = len(gen.oov_aksharas)
    print(f"inventory: {len(gen.aksharas)} attested aksharas "
          f"({n_oov_inv} out-of-vocab, {len(gen.aksharas) - n_oov_inv} in-vocab) "
          f"| min_freq={min_freq} alpha={gen.p['alpha']} "
          f"oov_boost={gen.oov_boost:.2f}x "
          f"(random path {gen.oov_share_random*100:.1f}% + coverage queue)")
    print(f"word length: corpus mean {gen.corpus_word_len:.2f} -> "
          f"generated mean {gen.mean_word_len:.2f} aksharas")
    print(f"calibrated: number_word={gen.p['number_word']:.4f} "
          f"english_word={gen.p['english_word']:.4f} "
          f"(corpus digit share {gen.digit_share*100:.2f}%, "
          f"latin share {gen.latin_share*100:.2f}%)")
    print()

    for text, st in gen.generate_many(num_report_lines, detailed=True):
        print(f"[g={st['graphemes']:>3} tok={st['tokens']:>3} frames={st['frames']:>3} "
              f"oov={st['oov']:>2}] {text}")

    # ---- verification --------------------------------------------------------------
    print(f"\nmeasuring {num_stat_lines} lines ...")
    # Targets are drawn here rather than inside generate_line, so the exact-length
    # guarantee can actually be checked against what was asked for.
    _r = random.Random(1234)
    targets = [_r.randint(min_char, max_char) for _ in range(num_stat_lines)]
    lines = [gen.generate_line(target=t) for t in targets]

    # Invariants: exact grapheme length, the token cap, no mixed-script words, and
    # balanced brackets / quotes.
    wrong_len = sum(1 for (text, st), t in zip(lines, targets) if st["graphemes"] != t)
    mixed = sum(1 for text, _ in lines if regex.search(
        r"[a-zA-Z]\p{Telugu}|\p{Telugu}[a-zA-Z]|[0-9]\p{Telugu}|\p{Telugu}[0-9]", text))
    unbalanced = 0
    for text, _ in lines:
        for a, b in PAIRED_PUNCT:
            if (text.count(a) % 2 if a == b else text.count(a) - text.count(b)):
                unbalanced += 1
                break
    print(f"  invariants: wrong length {wrong_len} | mixed-script words {mixed} "
          f"| unbalanced pairs {unbalanced}")

    n_g = n_tok = n_oov = 0
    max_frames = 0
    over_budget = 0
    unigram = collections.Counter()
    exposure = collections.Counter()
    word_lens, digit_chars, latin_chars, punct_chars = [], 0, 0, 0
    for text, st in lines:
        n_g += st["graphemes"]
        n_tok += st["tokens"]
        n_oov += st["oov"]
        max_frames = max(max_frames, st["frames"])
        over_budget += st["tokens"] > max_tokens
        for g in _GRAPHEME_RE.findall(text):
            if g in gen.token_cost:
                unigram[g] += 1
                if gen.token_cost[g] > 1:
                    exposure[g] += 1
            elif g.isdigit():
                digit_chars += 1
            elif g in LATIN_LOWER + LATIN_UPPER:
                latin_chars += 1
            elif g != " ":
                punct_chars += 1
        for w in text.split(" "):
            gs = [g for g in _GRAPHEME_RE.findall(w) if g in gen.token_cost]
            if gs:
                word_lens.append(len(gs))

    print(f"  tokens/grapheme      {n_tok / n_g:.3f}   (all-in-vocab text = 1.000)")
    print(f"  OOV akshara share    {n_oov / max(1, sum(unigram.values()))*100:.1f}%"
          f"   (target {gen.p['oov_share']*100:.0f}%)")
    print(f"  max frames needed    {max_frames} / {max_tokens} budget"
          f"   | lines over budget: {over_budget}")
    print(f"  mean word length     {sum(word_lens)/len(word_lens):.2f} aksharas"
          f"   (corpus {gen.corpus_word_len:.2f})")
    print(f"  digit char share     {digit_chars/n_g*100:.2f}%"
          f"   (corpus {gen.digit_share*100:.2f}%)")
    print(f"  latin char share     {latin_chars/n_g*100:.2f}%"
          f"   (corpus {gen.latin_share*100:.2f}%)")
    corpus_punct = sum(gen._corpus_counts.get(ch, 0) for ch in
                       TERMINAL_PUNCT + CLAUSE_PUNCT + INLINE_PUNCT + "()[]{}\"'-")
    print(f"  punct char share     {punct_chars/n_g*100:.2f}%"
          f"   (corpus {corpus_punct/gen._corpus_total*100:.2f}%)")

    # Rare-akshara coverage: how well the queue spreads exposure over the OOV inventory.
    covered = len(exposure)
    counts = sorted(exposure.values())
    print(f"  OOV coverage         {covered}/{n_oov_inv} distinct aksharas seen"
          f"   | min {counts[0] if counts else 0}"
          f" median {counts[len(counts)//2] if counts else 0}"
          f" max {counts[-1] if counts else 0}")
    if n_oov:
        per_line = n_oov / num_stat_lines
        print(f"  {per_line:.2f} OOV aksharas/line -> "
              f"{math.ceil(n_oov_inv * 100 / max(per_line, 1e-9)):,} lines for "
              f"~100 exposures each")

    # KL(generated || corpus) over the akshara unigram distribution: 0 = identical to
    # real Telugu, large = uniform-ish. It should be small but non-zero — the rare tail
    # is deliberately lifted.
    gen_tot = sum(unigram.values())
    corpus_sub = {g: gen._corpus_counts[g] for g in gen.aksharas}
    corpus_tot = sum(corpus_sub.values())
    kl = sum((c / gen_tot) * math.log((c / gen_tot) / (corpus_sub[g] / corpus_tot))
             for g, c in unigram.items() if corpus_sub.get(g))
    uni_kl = sum((1 / len(gen.aksharas)) * math.log(
        (1 / len(gen.aksharas)) / (corpus_sub[g] / corpus_tot)) for g in gen.aksharas)
    print(f"  KL(gen || corpus)    {kl:.3f} nats   (uniform would be {uni_kl:.3f})")

    # ---- cross-check the token cost against the real tokenizer ---------------------
    sys.path.insert(0, ROOT)
    try:
        from src.telugu_ocr.tokenizer.grapheme import TeluguGraphemeTokenizer
    except ImportError as exc:                       # transformers not installed
        print(f"\n[skip] tokenizer cross-check: {exc}")
    else:
        tk = TeluguGraphemeTokenizer(vocab_file=vocab_file)
        unk_id = tk.unk_token_id
        mismatch = unk_total = 0
        for text, st in lines[:1000]:
            ids = tk.encode(text, add_special_tokens=False)
            mismatch += len(ids) != st["tokens"]
            unk_total += sum(1 for i in ids if i == unk_id)
        print(f"\ntokenizer cross-check (1000 lines): "
              f"token-count mismatches {mismatch} | [UNK] tokens {unk_total}")
