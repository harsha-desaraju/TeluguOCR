"""Cut a page's proofread text into per-line ground truth, by forced alignment.

THE PROBLEM
    Wikisource gives one block of human-verified text per scanned page. Proofreaders
    reflow prose, so the newlines in that block are paragraph breaks, not the physical
    line breaks of the scan: a page with 27 printed lines typically arrives as 2 or 3
    paragraphs. The OCR model trains on single lines, so the text has to be cut at the
    27 places the printer cut it, and nothing in the text says where those are.

THE METHOD
    Segment the page into N line images, run a recogniser over each to get N noisy
    hypotheses, concatenate those into one string H, and align H against the ground
    truth G with a single global edit-distance alignment. The alignment induces a map
    from positions in H to positions in G; pushing each line's boundary in H through
    that map gives the cut points in G. The hypotheses only ever locate the boundaries
    -- the text that gets stored is always G, never H.

    Alignment is over GRAPHEME CLUSTERS, not codepoints. Telugu aksharas are 1-4
    codepoints, so a codepoint-level alignment can cut a cluster in half and produce a
    label starting with a bare vowel sign, and its edit distances would weight a
    three-codepoint akshara three times as heavily as a one-codepoint one.

WHAT COMES OUT, AND WHY IT IS A PARTITION
    The cut points partition G exactly: every grapheme of the ground truth lands in
    exactly one line, and no line's text overlaps another's. That is deliberate. When
    the page contains something the transcript omits -- a running header, a folio
    number, a plate caption -- or something the segmenter missed, the orphaned text
    has to go somewhere, and gluing it onto a neighbour makes that neighbour's CER
    blow up, which the filter then catches. Dropping unmatched ground truth silently
    would instead leave a clean-looking line whose label is missing a chunk, which is
    exactly the kind of label that poisons a training set without ever looking wrong.

    So a line is accepted only when its own hypothesis agrees with the span it was
    given. Rejection is the point, not a failure mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rapidfuzz.distance import Levenshtein

from pipelines.label.consensus_labelling import (
    cer,
    english_frac,
    graphemes,
    normalize,
)

__all__ = ["LineAlignment", "PageAlignment", "AcceptPolicy", "align_page"]

ARABIC_DIGITS = "0123456789"
TELUGU_DIGITS = "౦౧౨౩౪౫౬౭౮౯"
_ARABIC_TO_TELUGU = str.maketrans(ARABIC_DIGITS, TELUGU_DIGITS)


# ======================================================================================
# Ground truth preparation
# ======================================================================================
def _prepare_ground_truth(text: str) -> tuple[list[str], set[int]]:
    """Normalize the page text to graphemes, remembering where its own line breaks were.

    The transcript's newlines are usually paragraph breaks, but not always: on title
    pages, tables of contents and real verse the source carries `<br>` and the breaks
    line up with the printed lines exactly. Collapsing them to spaces (as plain
    `normalize` does) throws away a strong hint for free, so their positions are kept
    and used to break ties when choosing a cut.
    """
    gt: list[str] = []
    newline_at: set[int] = set()
    for i, raw in enumerate(ln for ln in (normalize(x) for x in text.split("\n")) if ln):
        if i:
            gt.append(" ")
            newline_at.add(len(gt))   # a cut here matches a transcript line break
        gt.extend(graphemes(raw))
    return gt, newline_at


# ======================================================================================
# Index mapping
# ======================================================================================
def _index_map(src: list[str], dest: list[str]) -> np.ndarray:
    """Map every position in `src` to the corresponding position in `dest`.

    Built from the edit-distance opcodes. Inside an `equal`/`replace` block the map is
    linear across the block, which handles replace blocks whose two sides differ in
    length; a `delete` block (src text with no counterpart) collapses onto the single
    dest position it sits at; an `insert` block contributes no src positions at all.

    The result is non-decreasing, so cut points derived from it can never cross.
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


def _word_boundaries(gt: list[str]) -> np.ndarray:
    """Positions in `gt` where a cut would not land inside a word."""
    edges = {0, len(gt)}
    edges.update(i for i in range(1, len(gt)) if gt[i - 1] == " ")
    return np.array(sorted(edges), dtype=np.int64)


def _neighbouring_boundaries(cut: int, boundaries: np.ndarray) -> list[int]:
    """The word boundaries immediately left and right of `cut`."""
    idx = int(np.searchsorted(boundaries, cut))
    out = []
    if idx - 1 >= 0:
        out.append(int(boundaries[idx - 1]))
    if idx < len(boundaries):
        out.append(int(boundaries[idx]))
    return out


def _refine_cuts(raw_cuts: list[int], gt: list[str], hypotheses: list[str],
                 boundaries: np.ndarray, newline_at: set[int],
                 newline_bonus: float) -> list[int]:
    """Place each line boundary at whichever nearby position best explains the images.

    The obvious rule -- always snap the cut to the nearest space, so labels are whole
    words -- is wrong, and measurably so. Printers break words across lines, and the
    transcript has already reflowed them back together, so for a genuinely broken word
    the correct cut falls INSIDE a word of the ground truth. Snapping then hands the
    whole word to the upper line and steals the lower line's opening, corrupting both
    labels. Measured on this corpus, 17% of raw cuts land inside a word, and every one
    of them is within 5 graphemes of a boundary -- so a fixed snap window always fires
    and can never tell the two cases apart.

    Distance cannot separate them, but evidence can. For each cut, try the raw mapped
    position and the word boundaries either side of it, and keep whichever gives the
    lowest combined CER for the two line images that share it. A word that really is
    split scores best at the raw position; alignment jitter scores best at a boundary.

    Transcript line breaks get a small bonus, because on the pages where the source
    preserved real line structure (verse, tables of contents, title pages) they are
    the right answer and the recogniser's opinion is the weaker signal.
    """
    cuts = list(raw_cuts)
    for i in range(1, len(cuts) - 1):
        lo, hi = cuts[i - 1], cuts[i + 1]
        if lo >= hi:
            cuts[i] = lo
            continue

        candidates = {min(max(cuts[i], lo), hi)}
        candidates.update(c for c in _neighbouring_boundaries(cuts[i], boundaries)
                          if lo <= c <= hi)

        best, best_score = None, None
        for cand in candidates:
            above = "".join(gt[lo:cand]).strip()
            below = "".join(gt[cand:hi]).strip()
            score = cer(above, hypotheses[i - 1]) + cer(below, hypotheses[i])
            if cand in newline_at:
                score -= newline_bonus
            if best_score is None or score < best_score:
                best, best_score = cand, score
        cuts[i] = best
    return cuts


# ======================================================================================
# Results
# ======================================================================================
@dataclass
class LineAlignment:
    index: int
    text: str                  # ground-truth span assigned to this line
    hypothesis: str            # what the recogniser read (normalized)
    cer: float                 # cer(text, hypothesis) -- ground truth is the reference
    n_graphemes: int
    accepted: bool = False
    reject_reason: str | None = None
    digits_converted: bool = False   # Arabic numerals rewritten as Telugu
    starts_mid_word: bool = False    # a word broken across the previous line break
    ends_mid_word: bool = False


@dataclass
class PageAlignment:
    lines: list[LineAlignment] = field(default_factory=list)
    page_cer: float = 1.0      # all hypotheses joined, vs the whole page text
    n_gt_graphemes: int = 0
    page_accepted: bool = False
    page_reject_reason: str | None = None
    # Fraction of detected lines explainable at a FIXED loose cut. This is the page
    # guard's input and the number worth storing -- unlike the strict yield it does not
    # move when max_line_cer moves, so it stays comparable across runs and can be used
    # to re-derive the guard downstream.
    broad_yield: float = 0.0

    @property
    def n_lines(self) -> int:
        return len(self.lines)

    @property
    def n_accepted(self) -> int:
        return sum(line.accepted for line in self.lines)

    @property
    def strict_yield(self) -> float:
        """Fraction accepted under the full policy. Reporting only -- do NOT feed this
        back into the page guard, and do not store it as the page's yield: it is read
        after a rejected page has had all its lines flipped, so it would always be 0."""
        return self.n_accepted / self.n_lines if self.lines else 0.0


# ======================================================================================
# Acceptance
# ======================================================================================
@dataclass
class AcceptPolicy:
    """What counts as a usable line.

    max_line_cer
        The main filter, and the only threshold worth tuning. A line is kept only when
        the recogniser's independent reading of the crop is within this of the span the
        alignment handed it.

        Read it as a corroboration test, not an error rate: the stored label is the
        human-proofread transcript, so this measures whether the span was cut in the
        right place, not whether the text is right. 0.10 is the default because the
        chosen recogniser reads this corpus at roughly 0.19 CER, so demanding agreement
        within 0.10 is demanding notably closer agreement than the model manages on
        average. Measured yields on 163 pages: 0.05 keeps 28% of detected lines, 0.10
        keeps 40%, 0.15 keeps 50%.

        Re-measure after retraining the recogniser -- the right cut moves with how well
        the model reads.

    edge_max_line_cer
        A tighter cut for the first and last text lines of a page. Wikisource
        proofreaders adjust page boundaries for continuity: a word or clause split
        across two scans is often completed on one page and dropped from the other, so
        the transcript's first and last lines are the ones least likely to correspond
        to what is actually printed there. They are not rejected outright -- most are
        fine -- but they have to clear a higher bar.

    min_graphemes
        Very short spans pass the CER test far too easily -- a 2-akshara span needs one
        correct akshara to score 0.5 -- and contribute almost nothing to training.

    max_grapheme_ratio
        Catches local alignment failure: a span several times the page's median line
        length means the aligner dumped an unmatched region onto this line.

    convert_digits
        Rewrite Arabic numerals in the label as Telugu numerals when the recogniser saw
        no Arabic numerals in the crop. Transcribers routinely type 1893 where the page
        prints ౧౮౯౩, which would otherwise train the model to read a glyph as something
        it does not depict. Both digit sets are in the tokenizer vocabulary, so either
        form is learnable and the conversion is safe in that sense -- but it does edit
        human-verified text on the strength of an OCR reading, so every converted line
        is flagged (`digits_converted`) and can be filtered back out.

    require_digit_evidence
        Only convert when the reading actually contains a Telugu numeral, rather than
        merely lacking an Arabic one. See _convert_digits: without this, the rewrite
        fires overwhelmingly on folio and verse numbers that leaked into a span from
        elsewhere on the page, inventing numerals the crop does not show.

    reject_script_mismatch
        The mojibake filter. Some transcripts were pasted from documents typed in a
        legacy 8-bit Telugu font, so a passage that is Latin in the scan arrives as
        Telugu-shaped nonsense -- one page has "Eighteen Fity Seven" stored as
        "జూరివీనీశిలిలిదీ ఓరిశిగి ఐలిఖీలిదీ". The signature is one-directional: the crop
        reads as Latin while the ground truth is Telugu with no Latin at all. The
        reverse is just a recogniser that cannot spell, so it is not treated as a
        mismatch, and a genuinely English line is unaffected because its ground truth
        is Latin too.

    min_page_yield / page_yield_cer / max_page_cer
        Page-level guards. A page where most lines fail is usually not a page of bad
        lines -- it is a transcript that does not belong to this scan, a plate, or a
        wholesale encoding problem -- and its few passing lines are not to be trusted
        either, since on such a page a good score is as likely luck as correctness.

        The yield the guard reads is measured at `page_yield_cer`, a FIXED loose cut,
        not at `max_line_cer`. Tying it to the strict cut made the two knobs fight:
        tightening max_line_cer mechanically lowers every page's yield, so more pages
        trip the guard and lose their good lines as well as their bad ones. Measured at
        max_line_cer=0.10, that discarded 37 pages of 163 that the recogniser had read
        perfectly well (page CER <= 0.25) -- one of them had 15 of 32 lines passing at
        0.10 and lost all 15 because 15/32 fell a hair under a 0.5 threshold. The guard
        is supposed to detect a page that does not correspond to its transcript, and
        that question has nothing to do with where the training threshold sits.
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
    """True for the legacy-font mojibake signature: Latin crop, Telugu-only transcript."""
    if not gt or not hyp:
        return False
    if english_frac(hyp) < 0.6 or english_frac(gt) > 0.1:
        return False
    return any("ఀ" <= ch <= "౿" for ch in gt)


def _convert_digits(span: str, hyp: str, require_evidence: bool = True) -> tuple[str, bool]:
    """Rewrite Arabic numerals in the label as Telugu ones, when the crop shows Telugu.

    The absence of Arabic numerals in the reading is not enough on its own to justify
    the rewrite, because it is satisfied two ways: the crop shows Telugu numerals (what
    this is for), or the crop shows no numerals at all. The second case is the common
    one and it is not a transcription convention, it is a span that picked up a folio
    or verse number printed somewhere else on the page. Measured over a 40-page build,
    every single conversion under the looser rule sat at the leading or trailing edge
    of its label and none in the interior -- the signature of exactly that leakage --
    and converting there invents a numeral the image does not contain.

    So `require_evidence` additionally demands at least one Telugu numeral in the
    reading. Set it False for the looser rule; the CER filter catches most of the
    damage either way (15 of those 17 lines were rejected regardless), but there is no
    reason to manufacture bad labels and lean on a downstream filter to remove them.
    """
    if not any(ch in ARABIC_DIGITS for ch in span):
        return span, False
    if any(ch in ARABIC_DIGITS for ch in hyp):
        return span, False
    if require_evidence and not any(ch in TELUGU_DIGITS for ch in hyp):
        return span, False
    return span.translate(_ARABIC_TO_TELUGU), True


# ======================================================================================
# The aligner
# ======================================================================================
def align_page(hypotheses: list[str | None], page_text: str,
               policy: AcceptPolicy | None = None) -> PageAlignment:
    """Assign each line hypothesis a span of `page_text`, and judge the result.

    `hypotheses` must be in reading order and the same length as the page's line
    boxes. A None (engine failure) or "" (nothing read) is kept in place rather than
    dropped, so line i of the output always corresponds to line box i; it collapses to
    a zero-width cut, its own span comes out empty, and it is rejected.
    """
    policy = policy or AcceptPolicy()
    result = PageAlignment()

    gt, newline_at = _prepare_ground_truth(page_text)
    result.n_gt_graphemes = len(gt)
    if not gt:
        result.page_reject_reason = "empty_page_text"
        return result

    clean = [normalize(h or "") for h in hypotheses]

    # Concatenate the hypotheses into one sequence, remembering where each line starts.
    # The single space between lines mirrors how the ground truth joins two printed
    # lines of one paragraph, so the alignment is not charged an insertion per line.
    joined: list[str] = []
    starts: list[int] = []
    for i, text in enumerate(clean):
        if i:
            joined.append(" ")
        starts.append(len(joined))
        joined.extend(graphemes(text))
    starts.append(len(joined))

    result.page_cer = cer("".join(gt), "".join(joined))

    mapping = _index_map(joined, gt)
    boundaries = _word_boundaries(gt)

    raw_cuts = [int(mapping[s]) for s in starts]
    raw_cuts[0], raw_cuts[-1] = 0, len(gt)
    for i in range(1, len(raw_cuts)):          # the map is monotone; be certain
        raw_cuts[i] = max(raw_cuts[i], raw_cuts[i - 1])
    cuts = _refine_cuts(raw_cuts, gt, clean, boundaries, newline_at, policy.newline_bonus)

    spans = ["".join(gt[a:b]).strip() for a, b in zip(cuts, cuts[1:])]
    if policy.convert_digits:
        converted = [_convert_digits(s, h, policy.require_digit_evidence)
                     for s, h in zip(spans, clean)]
        spans = [c[0] for c in converted]
        digit_flags = [c[1] for c in converted]
    else:
        digit_flags = [False] * len(spans)

    lengths = [len(graphemes(s)) for s in spans]
    median_len = float(np.median([n for n in lengths if n])) if any(lengths) else 0.0

    # The first and last lines carrying text, which is not the same as line 0 and line
    # N-1: a running header or folio number usually takes one of those boxes and comes
    # back with an empty span, since the transcript does not include it.
    with_text = [i for i, n in enumerate(lengths) if n]
    edges = {with_text[0], with_text[-1]} if with_text else set()

    boundary_set = set(boundaries.tolist())
    for i, (span, hyp, n) in enumerate(zip(spans, clean, lengths)):
        line = LineAlignment(index=i, text=span, hypothesis=hyp,
                             cer=cer(span, hyp), n_graphemes=n,
                             digits_converted=digit_flags[i],
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
