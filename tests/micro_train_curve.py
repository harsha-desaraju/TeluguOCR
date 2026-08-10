"""Phase 3 step 5 oracle: a CPU micro-training run, as a stand-in for the real one.

The handoff's oracle for the training collapse is "a 50-step run against a loss curve
captured before you start". That needs a GPU, the Hub datasets and hours. This is the
honest substitute: a tiny model on synthetic line images, seeded end to end, driven
through the SAME components the real loop uses -- the stage's collator, its optimizer
grouping, its scheduler, and the model's forward including stage 2's CTC auxiliary term.

It cannot tell you the merged loop trains well. It CAN tell you the merged loop computes
the same numbers as the code it replaced, which is the only question a refactor raises.

    python oracle_train.py capture   # writes /tmp/train_curves.json from current code
    python oracle_train.py check     # re-runs and diffs against that file
"""
import importlib
import json
import os
import sys

import pathlib

import numpy as np
import torch
from PIL import Image

ROOT = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)
os.chdir(ROOT)

SNAP = str(pathlib.Path(__file__).resolve().parent / "train_curves.json")
STEPS = 12
TOL = 1e-5


def seed_all(n=0):
    import random
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    torch.use_deterministic_algorithms(True, warn_only=True)


def synth_batch(tok, n=4):
    """n (PIL line image, text) pairs -- height 64, as the datasets hold them."""
    rng = np.random.RandomState(7)
    rows = []
    texts = ["అమ్మ నాన్న", "శ్రీరామ", "తెలుగు భాష", "పుస్తకం చదువు"]
    for i in range(n):
        w = 128 + 32 * i
        a = np.full((64, w), 240, dtype=np.uint8)
        for k in range(0, w - 10, 15):
            a[20:44, k + 2: k + 9] = 35
        a = np.clip(a.astype(np.int16) + rng.randint(-10, 10, (64, w)), 0, 255).astype(np.uint8)
        rows.append({"image": Image.fromarray(a),
                     "input_ids": tok(texts[i % len(texts)])["input_ids"],
                     "text": texts[i % len(texts)]})
    return rows


def small_cfgs(mod, vocab):
    ECfg, DCfg = mod.CTCEncoderConfig, mod.GPTConfig
    enc = ECfg(image_height=64, max_image_width=512, downsample=8, max_frames=64,
               stem_channels=[8, 16, 32, 32, 48, 64], num_groups=8, embed_dim=64,
               num_layers=2, num_heads=4, mlp_dim=128, dropout=0.0, drop_path_rate=0.0,
               vocab_size=vocab)
    dec = DCfg(vocab_size=vocab, embed_dim=64, hidden_dim=128, num_heads=4,
               num_layers=2, ctx_len=64, dropout=0.0)
    return enc, dec


def _resolve(mod, name):
    """The loop module may import a helper or not re-export it; fall back to the package."""
    if hasattr(mod, name):
        return getattr(mod, name)
    import src.telugu_ocr.training.optim as _o
    return getattr(_o, name)


def run_stage(loop_module: str, stage: int) -> list:
    mod = importlib.import_module(loop_module)
    from tests.model_registry import build_tokenizer
    tok = build_tokenizer()
    seed_all(0)
    enc, dec = small_cfgs(mod, len(tok))

    kw = dict(encoder_config=enc, decoder_config=dec, pad_index=tok.pad_token_id)
    if stage == 1:
        kw.update(encoder_no_grad=True, ctc_loss_weight=0.0)
    else:
        kw.update(encoder_no_grad=False,
                  ctc_loss_weight=(mod.STAGE_CONFIGS[2]["ctc_loss_weight"]
                                   if hasattr(mod, "STAGE_CONFIGS")
                                   else getattr(mod, "CTC_LOSS_WEIGHT", 0.3)))
    model = mod.EncoderDecoder(**kw)
    model.train()

    # the stage's own collator, with the stage's own signature
    if stage == 1:
        coll = mod.OCRCollator(pad_token_id=tok.pad_token_id, downsample=8)
    else:
        # mirror the stage-2 call site exactly: emit_ctc on, BOS/EOS stripped
        coll = mod.OCRCollator(emit_ctc=True, pad_token_id=tok.pad_token_id, downsample=8,
                               ctc_strip_ids=(tok.bos_token_id, tok.eos_token_id))

    from src.telugu_ocr.data.collators import LineTensorizer
    tens = LineTensorizer()
    rows = [{"pixel_values": tens(r["image"]), "input_ids": r["input_ids"]}
            for r in synth_batch(tok)]
    batch = coll(rows)

    # the stage's own optimizer grouping + scheduler
    if stage == 1:
        opt = _resolve(mod, 'build_optimizer')(model, lrs=3e-4, weight_decay=0.01, betas=(0.9, 0.95))
        sched = _resolve(mod, 'build_scheduler')(opt, warmup_steps=2, total_steps=STEPS, min_lr=1e-5)
    else:
        tiers = list(_resolve(mod, 'LR_TIERS'))
        opt = _resolve(mod, 'build_optimizer')(model,
                                  lrs={t: lr for t, lr in zip(tiers, (3e-4, 1e-4, 5e-5))},
                                  min_lrs={t: 1e-5 for t in tiers},
                                  weight_decay=0.01, betas=(0.9, 0.95),
                                  tier_fn=_resolve(mod, '_param_tier'), tiers=_resolve(mod, 'LR_TIERS'))
        sched = _resolve(mod, 'build_scheduler')(opt, warmup_steps=2, total_steps=STEPS)

    losses = []
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        out = model(**batch)
        loss = out["loss"] if isinstance(out, dict) else out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(round(float(loss.item()), 6))
    return losses


def run_callback(loop_module: str, stage: int) -> dict:
    """CEREvalCallback's per-slice CER, on a tiny model. Covers the eval metrics the
    loss curve does not: generation CER, and (stage 2) teacher-forced and CTC CER."""
    mod = importlib.import_module(loop_module)
    from tests.model_registry import build_tokenizer
    from src.telugu_ocr.data.collators import LineTensorizer
    tok = build_tokenizer()
    seed_all(0)
    enc, dec = small_cfgs(mod, len(tok))
    kw = dict(encoder_config=enc, decoder_config=dec, pad_index=tok.pad_token_id)
    kw.update(encoder_no_grad=(stage == 1),
              ctc_loss_weight=(0.0 if stage == 1 else
                               (mod.STAGE_CONFIGS[2]["ctc_loss_weight"]
                                if hasattr(mod, "STAGE_CONFIGS")
                                else getattr(mod, "CTC_LOSS_WEIGHT", 0.3))))
    model = mod.EncoderDecoder(**kw).eval()

    rows = [{"image": r["image"], "text": r["text"]} for r in synth_batch(tok)]
    # mirror each stage's call site: stage 2 also reports teacher-forced and CTC CER
    extra = {"report_tf": True, "report_ctc": True} if stage == 2 else {}
    cbk = mod.CEREvalCallback(model, {"all": rows}, LineTensorizer(), tok,
                              image_col="image", text_col="text", **extra)
    import contextlib
    out = cbk._slice_cer(rows, "cpu", dec.ctx_len, "image", "text")
    # stage 1 returns (cer, preds, refs); stage 2 a dict of three CERs plus the
    # hypotheses. Keep both the numbers AND the decoded strings -- a CER can stay put
    # while the text behind it changes.
    if isinstance(out, tuple):
        cer, preds, refs = out
        return {"gen_cer": round(float(cer), 6), "gen": list(preds), "refs": list(refs)}
    return {k: (round(float(v), 6) if isinstance(v, (int, float)) else list(v))
            for k, v in sorted(out.items())}


# Both stages come from one module, selected by STAGE_CONFIGS.
TARGETS = {
    "stage1": ("src.telugu_ocr.training.loops.encdec", 1),
    "stage2": ("src.telugu_ocr.training.loops.encdec", 2),
}

mode = sys.argv[1] if len(sys.argv) > 1 else "capture"
curves = {}
for name, (mod, stage) in TARGETS.items():
    try:
        curves[name] = run_stage(mod, stage)
        print(f"{name}: {curves[name][:4]} ... {curves[name][-2:]}")
        try:
            curves[name + "_cer"] = run_callback(mod, stage)
            print(f"{name}_cer: {curves[name + '_cer']}")
        except Exception as e:
            print(f"{name}_cer: FAILED {type(e).__name__}: {str(e)[:70]}")
            curves[name + "_cer"] = None
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"{name}: FAILED {type(e).__name__}: {e}")
        curves[name] = None

if mode == "capture":
    json.dump(curves, open(SNAP, "w"), indent=1)
    print(f"\ncaptured -> {SNAP}")
else:
    prev = json.load(open(SNAP))
    bad = 0
    print()
    for name in list(TARGETS) + [n + "_cer" for n in TARGETS]:
        a, b = prev.get(name), curves.get(name)
        if isinstance(a, dict) or isinstance(b, dict):
            if a != b:
                print(f"  {name}: MOVED\n      before {a}\n      after  {b}"); bad += 1
            else:
                print(f"  {name}: identical {a}")
            continue
        if a is None or b is None:
            print(f"  {name}: MISSING (before={a is not None}, after={b is not None})"); bad += 1
        elif len(a) != len(b) or any(abs(x - y) > TOL for x, y in zip(a, b)):
            print(f"  {name}: CURVE MOVED")
            print(f"      before {a}")
            print(f"      after  {b}")
            bad += 1
        else:
            print(f"  {name}: identical over {len(a)} steps (max delta "
                  f"{max(abs(x - y) for x, y in zip(a, b)):.2e})")
    sys.exit(1 if bad else 0)
