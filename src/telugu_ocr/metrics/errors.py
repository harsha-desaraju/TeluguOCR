"""Edit distance and character error rate.

One implementation, previously five: `_edit_distance` in three training loops (2
versions), `edit_distance` in two eval scripts (2 more), and a separate alignment-based
scorer in benchmark/metrics.py. The training loops' version is kept -- it is the one
whose numbers appear in every training log and eval slice.

benchmark/metrics.py keeps its own richer scorer: it aligns rather than just counting,
so it can report substitutions/insertions/deletions and a confusion table. It calls the
same distance underneath.
"""

from __future__ import annotations


def edit_distance(a, b) -> int:
    """Levenshtein distance between two sequences. Dependency-free.

    Operates on any sequence, which is the point: callers pass a list of grapheme
    clusters, not a string, so a two-codepoint akshara counts as ONE error rather than
    two. Passing a raw str silently scores at codepoint level instead.
    """
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[n]


# The training loops used a leading underscore; keep it as an alias so the inlined
# call sites read the same after the collapse.
_edit_distance = edit_distance


def compute_cer(preds, refs) -> float:
    """Corpus CER: total edit distance over total reference length.

    Corpus-level, NOT the mean of per-sample CERs -- long lines therefore carry more
    weight, and a one-character reference cannot contribute a 100% error on its own.
    """
    tot_e = tot_c = 0
    for p, r in zip(preds, refs):
        tot_e += edit_distance(p, r)
        tot_c += max(len(r), 1)
    return tot_e / max(tot_c, 1)
