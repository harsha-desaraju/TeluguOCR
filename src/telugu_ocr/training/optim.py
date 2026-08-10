"""Optimizer construction and LR schedule, shared by every training loop.

There were three copies of build_optimizer and three of build_scheduler, and unlike the
model code they genuinely differed -- because the stages genuinely differ. Stage 1 trains
one tier (the new adapter params) at a single LR; stage 2 trains three (encoder /
cross-attention / LM head) each on its own cosine curve. The CTC loop is single-tier
again.

So this is not "pick the right copy": it is the tiered form, with the single-tier case
expressed as one tier. Numerically identical, because AdamW applies lr and weight_decay
per param group and a single tier produces exactly the two groups the old code built.
"""

from __future__ import annotations

import math

import torch

# A param is in a tier if its name matches. Order matters: first match wins.
DEFAULT_TIER = "all"


def single_tier(_name: str) -> str:
    """Every trainable parameter in one tier -- stage 1 and the CTC loop."""
    return DEFAULT_TIER


def no_decay(name: str, param) -> bool:
    """Biases, norm weights and positional embeddings get no weight decay.

    Matches the convention the loops already used. `pos_embed` is matched as a substring
    so it catches the encoder's table wherever it is nested.
    """
    return "pos_embed" in name or param.ndim < 2 or "norm" in name.lower()


def build_optimizer(model, lrs, weight_decay, betas, min_lrs=None, tier_fn=single_tier,
                    tiers=(DEFAULT_TIER,)):
    """AdamW over per-tier, decay/no-decay param groups.

    lrs / min_lrs may be a scalar (applied to every tier) or a dict keyed by tier. Each
    group carries its own 'min_lr' so build_scheduler can decay it to its own floor.
    Frozen params (requires_grad False) are skipped entirely rather than added with
    lr=0 -- that keeps them out of the optimizer state, which is the memory win stage 1
    is after.
    """
    def pick(v, tier):
        return v[tier] if isinstance(v, dict) else v

    buckets = {t: {"decay": [], "nodecay": []} for t in tiers}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        t = tier_fn(name)
        buckets[t]["nodecay" if no_decay(name, p) else "decay"].append(p)

    groups = []
    for t in tiers:
        for kind, wd in (("decay", weight_decay), ("nodecay", 0.0)):
            if buckets[t][kind]:
                g = {"params": buckets[t][kind], "lr": pick(lrs, t), "weight_decay": wd}
                if min_lrs is not None:
                    g["min_lr"] = pick(min_lrs, t)
                groups.append(g)
    return torch.optim.AdamW(groups, betas=betas)


def build_scheduler(optimizer, warmup_steps, total_steps, min_lr=None):
    """Per-group linear warmup -> cosine decay, from the group's peak to its floor.

    The floor is the group's own 'min_lr' when build_optimizer attached one (stage 2's
    per-tier floors), otherwise the absolute `min_lr` argument (stage 1's single floor).
    """
    from torch.optim.lr_scheduler import LambdaLR

    def make(peak, floor):
        def f(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            prog = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
            cos = 0.5 * (1.0 + math.cos(math.pi * prog))
            return (floor + (peak - floor) * cos) / peak
        return f

    lambdas = []
    for g in optimizer.param_groups:
        floor = g.get("min_lr", min_lr)
        if floor is None:
            raise ValueError("build_scheduler needs a floor: pass min_lr, or build the "
                             "optimizer with min_lrs so each group carries its own")
        lambdas.append(make(g["lr"], floor))
    return LambdaLR(optimizer, lambdas)
