"""Decoding strategies over a trained EncoderDecoder: CTC greedy, beam, CTC rescoring.

These used to live in scripts/eval/encoder_decoder.py, which CLAUDE.md describes as an
ad-hoc diagnostic runner. Three callers imported them from there -- that script itself,
benchmark/engines.py and inference/recognize.py -- so the inference path depended on a
scratch script. The functions are unchanged; only their address is.

Everything here takes the BRIDGED encoder output (already through `enc_to_dec`) rather
than an image, so one encoder forward can feed several decoders.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def ctc_collapse(frame_ids: Sequence[int], blank_id: int) -> list[int]:
    """Collapse consecutive repeats, then drop blanks -- the CTC decoding rule."""
    out, prev = [], None
    for t in frame_ids:
        if t != prev and t != blank_id:
            out.append(t)
        prev = t
    return out


@torch.no_grad()
def ctc_greedy_ids(ctc_log_probs: torch.Tensor, blank_id: int) -> list[int]:
    """(T, C) frame log-probs -> argmax -> collapse consecutive repeats -> drop blanks."""
    return ctc_collapse(ctc_log_probs.argmax(dim=-1).tolist(), blank_id)


def _reorder_cache(past, idx: torch.Tensor):
    """Select/duplicate the batch dimension of every cached K/V after a beam reshuffle."""
    out = []
    for self_kv, cross_kv in past:
        s = (self_kv[0].index_select(0, idx), self_kv[1].index_select(0, idx))
        c = None if cross_kv is None else (
            cross_kv[0].index_select(0, idx), cross_kv[1].index_select(0, idx))
        out.append((s, c))
    return out


@torch.no_grad()
def beam_search(model, bridged, bos_id, eos_id, beam_width, max_new_tokens):
    """KV-cached beam search over the text decoder for a single (un-padded) image.

    bridged: (1, T, dec_dim) encoder output already through enc_to_dec.
    Returns the n-best list as dicts {"ids": [token ids, no BOS/EOS], "logp": float}
    where logp is the summed attention log-prob INCLUDING the EOS step (hypotheses
    that never emitted EOS within the budget are closed as-is).
    """
    device = bridged.device
    ctx = model.decoder_model.positional_encodings.shape[0]
    max_steps = min(max_new_tokens, ctx - 1)

    # Step 1: one BOS forward seeds the beams (all beams would be identical anyway).
    bos = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
    out, past = model.decoder_model(bos, bridged, None, None, None,
                                    past_kvs=None, use_cache=True)
    logprobs = F.log_softmax(out.logits[0, -1].float(), dim=-1)
    top_scores, top_tokens = logprobs.topk(beam_width)

    finished = []
    beam_ids, beam_scores = [], []
    for s, t in zip(top_scores.tolist(), top_tokens.tolist()):
        if t == eos_id:
            finished.append({"ids": [], "logp": s})
        else:
            beam_ids.append([t])
            beam_scores.append(s)
    if not beam_ids:
        return finished
    past = _reorder_cache(past, torch.zeros(len(beam_ids), dtype=torch.long, device=device))
    beam_scores = torch.tensor(beam_scores, device=device)

    for _ in range(max_steps - 1):
        B = len(beam_ids)
        step_in = torch.tensor([[ids[-1]] for ids in beam_ids], dtype=torch.long, device=device)
        out, past = model.decoder_model(step_in, bridged.expand(B, -1, -1), None, None, None,
                                        past_kvs=past, use_cache=True)
        logprobs = F.log_softmax(out.logits[:, -1].float(), dim=-1)      # (B, V)
        cand = beam_scores.unsqueeze(1) + logprobs
        V = logprobs.shape[-1]
        # Top 2*beam_width candidates: EOS extensions retire to `finished`, the rest
        # refill the active beam until it is full again.
        flat_scores, flat_idx = cand.flatten().topk(min(2 * beam_width, cand.numel()))
        new_ids, new_scores, parents = [], [], []
        for sc, fi in zip(flat_scores.tolist(), flat_idx.tolist()):
            parent, tok = divmod(fi, V)
            if tok == eos_id:
                finished.append({"ids": beam_ids[parent][:], "logp": sc})
            else:
                new_ids.append(beam_ids[parent] + [tok])
                new_scores.append(sc)
                parents.append(parent)
            if len(new_ids) == beam_width:
                break
        if len(finished) >= beam_width or not new_ids:
            beam_ids = new_ids
            break
        beam_ids = new_ids
        beam_scores = torch.tensor(new_scores, device=device)
        past = _reorder_cache(past, torch.tensor(parents, dtype=torch.long, device=device))

    # Budget exhausted (or beam retired early): close the remaining actives without EOS.
    for ids, sc in zip(beam_ids, beam_scores.tolist()[:len(beam_ids)]):
        finished.append({"ids": ids, "logp": sc})
    return finished


@torch.no_grad()
def ctc_hyp_logprobs(ctc_log_probs: torch.Tensor, hyps: list, blank_id: int) -> torch.Tensor:
    """logP_ctc(y|x) for each hypothesis via the CTC forward algorithm (= -ctc_loss).

    ctc_log_probs: (T, C) fp32 log-softmax frame posteriors, on CPU (MPS has no
    ctc_loss kernel, and this is cheap). Hypotheses that CTC cannot emit (longer
    than the frame count, or containing ids outside the CTC class space) score -inf,
    which simply removes them from the combined ranking.
    """
    T, C = ctc_log_probs.shape
    n = len(hyps)
    scores = torch.full((n,), float("-inf"))
    valid = [i for i, h in enumerate(hyps)
             if len(h) <= T and all(0 <= t < blank_id for t in h)]
    if not valid:
        return scores
    max_len = max(1, max(len(hyps[i]) for i in valid))
    targets = torch.zeros(len(valid), max_len, dtype=torch.long)
    tlens = torch.zeros(len(valid), dtype=torch.long)
    for row, i in enumerate(valid):
        h = hyps[i]
        targets[row, :len(h)] = torch.tensor(h, dtype=torch.long)
        tlens[row] = len(h)
    lp = ctc_log_probs.unsqueeze(1).expand(T, len(valid), C).contiguous()
    loss = F.ctc_loss(lp, targets,
                      input_lengths=torch.full((len(valid),), T, dtype=torch.long),
                      target_lengths=tlens,
                      blank=blank_id, reduction="none", zero_infinity=False)
    scores[torch.tensor(valid)] = -loss
    return scores


@torch.no_grad()
def greedy_decode(model, bridged, bos_id, eos_id, max_new_tokens,
                  cross_key_mask=None, no_repeat_cycle: bool = True) -> list[list[int]]:
    """KV-cached greedy decode. Returns the token ids per row, BOS/EOS excluded.

    Batched: rows that emit EOS stop contributing and are padded no further, so a batch
    costs as many steps as its longest row rather than the sum. `no_repeat_cycle` breaks
    out when every live row has fallen into a short repeating loop.
    """
    device = bridged.device
    B = bridged.shape[0]
    ctx = model.decoder_model.positional_encodings.shape[0]

    ids = [[] for _ in range(B)]
    done = torch.zeros(B, dtype=torch.bool, device=device)
    step_input = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    past_kvs = None

    for _ in range(min(max_new_tokens, ctx - 1)):
        out, past_kvs = model.decoder_model(step_input, bridged, text_padding_mask=None,
                                            img_text_padding_mask=cross_key_mask,
                                            labels=None, past_kvs=past_kvs, use_cache=True)
        nxt = out.logits[:, -1, :].argmax(-1)                # (B,)
        done = done | (nxt == eos_id)
        for b in range(B):
            if not done[b]:
                ids[b].append(int(nxt[b]))
        if bool(done.all()):
            break
        if no_repeat_cycle and all(_is_cycling(ids[b]) for b in range(B) if not done[b]):
            break
        step_input = nxt.unsqueeze(1)

    return ids


def _is_cycling(ids: list[int]) -> bool:
    """True when the tail has collapsed into a short repeating loop."""
    if len(ids) < 24:        # matches the old generate(): BOS + 23 generated tokens
        return False
    tail = ids[-12:]
    return any(tail == tail[-k:] * (12 // k) for k in (1, 2, 3, 4))
