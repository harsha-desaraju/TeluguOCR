"""Error metrics for OCR output: CER, AER, WER, edit-op breakdown, confusions.

THREE UNITS, THREE ERROR RATES
    All three are the same computation over a different tokenisation of the line, and
    all are (substitutions + insertions + deletions) / len(reference), aggregated
    corpus-wide as sum(errors) / sum(reference lengths).

      CER  Unicode code points. The conventional number, and what
           scripts/eval/encoder_decoder.py reports, so results stay comparable.
      AER  Akshara (grapheme cluster) error rate -- split with `regex.\\X`, the SAME
           definition src/telugu_ocr/tokenizer/grapheme.py uses, so one
           akshara here is exactly one token to the model.
      WER  Whitespace-delimited words.

WHY AER MATTERS FOR TELUGU, AND WHY CER ALONE MISLEADS
    A Telugu akshara is several code points (consonant + vowel sign + virama + ...), so
    code-point CER does not weight errors the way a reader perceives them. Miss one
    vowel sign and CER charges 1 edit; mangle a whole cluster into a different cluster
    and CER may charge 2-3 -- yet to a reader both are exactly one wrong syllable. Two
    engines with equal CER can therefore be meaningfully different, and an engine that
    smears diacritics is flattered relative to one that garbles whole aksharas. AER
    counts one wrong syllable as one error, which is the unit Indic OCR work reports.

EDIT-OP BREAKDOWN
    The same alignment that produces the error rate is backtracked to say HOW an engine
    fails: substitutions (read the wrong thing), insertions (invented output), deletions
    (dropped input). An engine at CER 0.10 that is nearly all insertions is a very
    different problem from one that is nearly all deletions.

CONFUSIONS
    The substitution pairs from that alignment, counted. `ref -> hyp` with the empty
    string shown as EMPTY for insertions/deletions. This is the directly actionable
    output: the aksharas an engine reliably gets wrong are the ones to target with
    training data.

Reference length is taken from the REFERENCE, so rates are comparable across engines
scoring the same lines; an engine cannot lower its own denominator by emitting less.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass, field

import regex

EMPTY = "∅"                      # stands in for "nothing" in a confusion pair
_GRAPHEME = regex.compile(r"\X")


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------
def normalize(text) -> str:
    """NFC + collapsed whitespace: compare characters, not spacing conventions."""
    return " ".join(unicodedata.normalize("NFC", str(text)).split())


def to_chars(text) -> list[str]:
    return list(str(text))


def to_aksharas(text) -> list[str]:
    """Unicode grapheme clusters — the same split the model's tokenizer uses."""
    return _GRAPHEME.findall(str(text))


def to_words(text) -> list[str]:
    return str(text).split()


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------
def align(ref: list, hyp: list) -> list[tuple[str, str, str]]:
    """Levenshtein alignment of hyp against ref.

    Returns the edit script as (op, ref_token, hyp_token) with op in
    equal/sub/del/ins. `del` = present in ref, missing from hyp; `ins` = present in
    hyp, absent from ref. Directionality is what makes the breakdown meaningful, so it
    is fixed here rather than left to the caller.

    The distance implied (sub + ins + del) is the ordinary Levenshtein distance, so
    error rates computed from this agree exactly with test_model.edit_distance.
    """
    m, n = len(ref), len(hyp)
    # dp[i][j] = distance between ref[:i] and hyp[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        ri = ref[i - 1]
        for j in range(1, n + 1):
            if ri == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1],   # substitute
                                   dp[i - 1][j],       # delete from ref
                                   dp[i][j - 1])       # insert into hyp
    ops: list[tuple[str, str, str]] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops.append(("equal", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("sub", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], EMPTY))
            i -= 1
        else:
            ops.append(("ins", EMPTY, hyp[j - 1]))
            j -= 1
    ops.reverse()
    return ops


# ---------------------------------------------------------------------------
# Accumulators
# ---------------------------------------------------------------------------
@dataclass
class LevelStats:
    """Corpus-wide totals for one tokenisation level."""

    unit: str
    sub: int = 0
    ins: int = 0
    dele: int = 0
    ref_len: int = 0
    exact: int = 0
    n: int = 0
    confusions: Counter = field(default_factory=Counter)

    @property
    def errors(self) -> int:
        return self.sub + self.ins + self.dele

    @property
    def error_rate(self) -> float:
        return self.errors / max(self.ref_len, 1)

    @property
    def exact_rate(self) -> float:
        return self.exact / max(self.n, 1)

    def update(self, ref_tokens: list, hyp_tokens: list, collect_confusions=True):
        ops = align(ref_tokens, hyp_tokens)
        for op, r, h in ops:
            if op == "equal":
                continue
            if op == "sub":
                self.sub += 1
            elif op == "ins":
                self.ins += 1
            else:
                self.dele += 1
            if collect_confusions:
                self.confusions[(r, h)] += 1
        self.ref_len += max(len(ref_tokens), 1)
        self.exact += int(ref_tokens == hyp_tokens)
        self.n += 1
        return ops

    def as_dict(self) -> dict:
        return {
            "unit": self.unit,
            "error_rate": self.error_rate,
            "exact": self.exact_rate,
            "sub": self.sub,
            "ins": self.ins,
            "del": self.dele,
            "errors": self.errors,
            "ref_len": self.ref_len,
        }

    def top_confusions(self, k=15) -> list[dict]:
        return [{"ref": r, "hyp": h, "count": c,
                 "kind": "del" if h == EMPTY else "ins" if r == EMPTY else "sub"}
                for (r, h), c in self.confusions.most_common(k)]


def score_pairs(predictions, references, collect_confusions=True) -> dict:
    """Score one engine's output. -> {"char": LevelStats, "akshara": ..., "word": ...}"""
    levels = {
        "char": (LevelStats("char"), to_chars),
        "akshara": (LevelStats("akshara"), to_aksharas),
        "word": (LevelStats("word"), to_words),
    }
    for hyp, ref in zip(predictions, references):
        hyp_n, ref_n = normalize(hyp), normalize(ref)
        for name, (stats, tokenize) in levels.items():
            # Confusions over words are unbounded and rarely actionable; the useful
            # ones are aksharas (what to train on) and characters (what to normalise).
            stats.update(tokenize(ref_n), tokenize(hyp_n),
                         collect_confusions=collect_confusions and name != "word")
    return {name: stats for name, (stats, _) in levels.items()}
