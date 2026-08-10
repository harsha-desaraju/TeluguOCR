"""Text normalisation and grapheme splitting for scoring.

ONE grapheme splitter, TWO normalisers -- on purpose.

The splitter is shared: `to_aksharas` in benchmark/metrics.py and `graphemes` in the
pseudo-labeller were the same `regex.\\X` call, and it must stay the same one the
tokenizer uses, or a two-codepoint akshara gets scored as two errors instead of one.

The normalisers are NOT merged, because they answer different questions:

  normalize_nfc   NFC + collapsed whitespace. Deliberately minimal, so the benchmark's
                  CER stays comparable with what scripts/eval/ reports.
  normalize_nfkc  NFKC, plus optional punctuation stripping, used to decide whether two
                  ENGINES AGREE. Agreement wants aggressive folding -- no-break space,
                  fullwidth Latin and ligatures should not count as disagreement -- but
                  applying that to a benchmark would quietly flatter every engine.

Collapsing those two into one would move published numbers in whichever direction the
loser was. They are one concept only if you do not look at what they are for.
"""

from __future__ import annotations

import unicodedata

import regex

_GRAPHEME_RE = regex.compile(r"\X")


def graphemes(text: str) -> list[str]:
    """Split into user-perceived characters (Unicode extended grapheme clusters).

    The same split the model's tokenizer performs, which is what makes a CER computed
    over this list an akshara-level rate rather than a codepoint-level one.
    """
    return _GRAPHEME_RE.findall(text) if text else []


def normalize_nfc(text) -> str:
    """NFC + collapsed whitespace: compare characters, not spacing conventions."""
    return " ".join(unicodedata.normalize("NFC", str(text)).split())
