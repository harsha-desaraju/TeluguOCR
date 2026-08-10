"""Eval-slice construction, shared by the encoder-decoder training loops.

`build_eval_slices` existed three times with three signatures; the differences were two
parameters stage 1 accepted and never used. `ListEvalDataset` existed three times in two
real versions -- the CTC loop's emits the CTC input format and stays in that loop; the
two encoder-decoder copies were identical and live here.
"""

from __future__ import annotations

import torch

def build_eval_slices(rows, source_col, slice_cap, seed):
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
