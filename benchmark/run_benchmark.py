"""Benchmark OCR engines on human-reviewed Telugu line crops.

WHAT IS MEASURED
    Every engine transcribes the SAME lines and is scored against the same references,
    through the one contract in benchmark/engines.py (image(s) -> string(s)). Adding an
    engine is an entry in ENGINES; nothing else in this file changes.

THE REFERENCES
    A HuggingFace dataset -- by default harsha-desaraju/telugu-line-ocr-bench, split
    "test". Every row is a line crop whose transcription a human checked against the
    image, published by data_curation/wikisource/push_to_hub.py (WHICH="eval").

    Using the Hub copy rather than the local working directory means the benchmark is
    pinned to a published, versioned artifact: a result can be reproduced by anyone,
    and it does not drift as more lines are annotated locally. Deduplication, empty
    removal and the Latin-script filter already happened at build time, so this script
    re-derives none of it.

    Why human labels and not lines.jsonl: that file's `text` comes from an alignment
    anchored on our own CTC model, and its `accepted` flag means that model agreed with
    the alignment. Scoring against it would hand our model an advantage no other engine
    gets.

    SAMPLING CAVEAT. Those lines were annotated from a `line_cer` band -- lines our
    model already found hard -- so they are NOT a uniform sample of the corpus.
    Absolute error rates here are pessimistic for our model; the comparison BETWEEN
    engines on identical inputs is the meaningful part.

FAIRNESS RULES
    * Identical sample set for every engine. A line the model cannot encode (wider than
      2048px at height 64) is dropped for ALL engines rather than scored as empty for
      one of them.
    * Everything is scored on NFC-normalized, whitespace-collapsed text, so an engine is
      not punished for spacing conventions instead of character errors. `cer_raw`
      (whitespace-stripped only) is reported alongside because that is what
      src/encoder_decoder/test_model.py prints, keeping past runs comparable.
    * Rates are corpus-aggregated: sum(errors) / sum(reference lengths), NOT the mean of
      per-line rates -- a 4-character line otherwise weighs as much as an 80-character
      one.

METRICS (see benchmark/metrics.py)
    CER / AER / WER   the same error rate over code points, aksharas, and words. AER is
                      the one to rank on for Telugu: an akshara is several code points,
                      so CER charges a missing vowel sign 1 edit and a garbled cluster
                      2-3, while a reader sees one wrong syllable either way.
    sub / ins / del   how each engine fails -- reading the wrong thing, inventing
                      output, or dropping input. Very different problems at equal CER.
    confusions        the akshara pairs an engine reliably gets wrong; the directly
                      actionable output for choosing what to train on next.

OUTPUT
    A table, per-engine confusions, and sample lines on stdout, plus a JSON with the
    full summary and every prediction, so a disagreement can be inspected without
    re-running the models.

    python3 -m benchmark.run_benchmark
"""

from __future__ import annotations

import gc
import json
import os
import random
import time
from collections import Counter

from datasets import load_dataset

from benchmark.engines import build_engine
from benchmark.metrics import EMPTY, score_pairs
from src.encoder_decoder.test_model import edit_distance


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_bench(repo, split, token=None):
    """The published benchmark split from the Hub.

    Deduplication, empty-text removal and the human-verified filter all happened when
    the dataset was built (data_curation/wikisource/push_to_hub.py, WHICH="eval"), so
    nothing is re-derived here -- the published rows ARE the reference set. The guards in
    select_samples remain only for things a consumer must still decide, like whether a
    given engine can encode a given image.
    """
    ds = load_dataset(repo, split=split, token=token)
    print(f"[data] {repo} (split={split})")
    print(f"[data] {len(ds)} rows, columns: {ds.column_names}")
    return ds


def select_samples(ds, n_samples, seed, max_source_width,
                   text_field="text", id_field="line_image"):
    """-> (images, references, keys), at most n_samples of them.

    Rows are walked in a seeded shuffle and decoded lazily, stopping once n_samples are
    in hand, so a small run does not decode the whole split. n_samples=None walks the
    split in order and takes everything.

    A row no engine could fairly be scored on is skipped for ALL engines -- images too
    wide for the model's encoder would otherwise be scored as empty for it alone.
    """
    order = list(range(len(ds)))
    if n_samples is not None:
        random.Random(seed).shuffle(order)

    images, references, keys = [], [], []
    dropped = Counter()
    for i in order:
        if n_samples is not None and len(images) >= n_samples:
            break
        row = ds[i]
        text = str(row.get(text_field, "")).strip()
        if not text:
            dropped["empty reference"] += 1
            continue
        img = row["image"]
        if max_source_width and img.width / max(img.height, 1) * 64 > max_source_width:
            dropped["too wide for the encoder"] += 1
            continue
        images.append(img)
        references.append(text)
        keys.append(row.get(id_field) or f"row-{i}")

    for reason, count in dropped.items():
        print(f"[data] dropped {count} ({reason})")
    scope = f"{len(images)} sampled (seed {seed})" if n_samples is not None \
        else f"all {len(images)}"
    print(f"[data] scoring {scope} of {len(ds)} rows")
    return images, references, keys


# ---------------------------------------------------------------------------
# Engine lifecycle
#
# Engines are built one at a time and released before the next one is constructed.
# They bring incompatible native runtimes -- torch/MPS, paddle, tesseract, surya --
# and holding them all at once segfaulted the full-split run: two copies of our
# 325MB model sat on MPS alongside paddle's and surya's, and surya died mid-batch.
# Nothing here is needed for a single-engine run; it is the price of comparing them
# in one process.
# ---------------------------------------------------------------------------
def preflight(engine_specs):
    """Missing files named in the specs, checked before anything heavy is built.

    Building lazily costs the old fail-fast behaviour: a typo'd checkpoint used to
    surface at startup and would now surface after the earlier engines had run. A
    path check is cheap and catches that same mistake.
    """
    missing = []
    for spec in engine_specs:
        for key in ("checkpoint", "vocab_file"):
            path = spec.get(key)
            if path and not os.path.exists(path):
                missing.append(f"{spec.get('engine', '?')}.{key}: {path}")
    return missing


def max_source_width(engine_specs):
    """Widest image the engines can take, without constructing them.

    Only our model has a limit, and it is a declared config value rather than
    something learned from the checkpoint, so the spec is enough.
    """
    widths = [spec.get("max_image_width", 2048)
              for spec in engine_specs if spec.get("engine") == "model"]
    return max(widths, default=0)


def release(engine):
    """Drop an engine and give its device memory back."""
    del engine
    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score(predictions, references):
    """Error rates at three units, the edit-op breakdown, and confusion pairs.

    `cer_raw` is kept alongside the normalized rates because it is the number
    src/encoder_decoder/test_model.py prints (whitespace-stripped, no NFC), so past
    eval_results.json runs stay comparable. Everything else is on normalized text.
    """
    levels = score_pairs(predictions, references)

    raw_edits = raw_ref = empty = 0
    for hyp, ref in zip(predictions, references):
        hyp_s, ref_s = str(hyp).strip(), str(ref).strip()
        raw_edits += edit_distance(hyp_s, ref_s)
        raw_ref += max(len(ref_s), 1)
        empty += int(not hyp_s)

    return {
        "n": len(references),
        "cer_raw": raw_edits / max(raw_ref, 1),
        "cer": levels["char"].error_rate,
        "aer": levels["akshara"].error_rate,
        "wer": levels["word"].error_rate,
        "exact": levels["char"].exact_rate,
        "empty_outputs": empty,
        "levels": {name: st.as_dict() for name, st in levels.items()},
        "confusions": {
            "akshara": levels["akshara"].top_confusions(25),
            "char": levels["char"].top_confusions(25),
        },
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_table(results):
    """Error rates, then WHERE the errors come from (sub/ins/del at akshara level)."""
    order = sorted(results, key=lambda k: results[k]["summary"]["aer"])
    w = max(len(k) for k in results) + 2
    width = w + 74

    print()
    print("=" * width)
    print(f"{'engine':<{w}}{'CER':>8}{'AER':>8}{'WER':>8}{'exact':>8}"
          f"{'sub':>7}{'ins':>7}{'del':>7}{'empty':>7}{'sec/line':>10}")
    print("=" * width)
    for name in order:
        s = results[name]["summary"]
        ak = s["levels"]["akshara"]
        print(f"{name:<{w}}{s['cer']:>8.4f}{s['aer']:>8.4f}{s['wer']:>8.4f}"
              f"{s['exact']:>8.3f}{ak['sub']:>7}{ak['ins']:>7}{ak['del']:>7}"
              f"{s['empty_outputs']:>7}{results[name]['seconds_per_line']:>10.3f}")
    print("=" * width)
    any_sum = next(iter(results.values()))["summary"]["levels"]
    print(f"lower is better. CER=code points, AER=aksharas, WER=words; sub/ins/del "
          f"are akshara counts over {any_sum['akshara']['ref_len']} reference aksharas "
          f"({any_sum['char']['ref_len']} chars, {any_sum['word']['ref_len']} words).")
    print("ranked by AER — one wrong syllable counts once, which is how Telugu reads.")


def _show(token):
    """Render a token so whitespace is visible.

    Space confusions are among the most common and the most actionable -- dropped
    spaces are the single biggest source of our model's deletions -- but printed raw
    they look like a blank and get read as noise.
    """
    return "␣" if token == " " else token.replace(" ", "␣")


def print_confusions(results, unit="akshara", k=10):
    """The aksharas each engine reliably gets wrong — what to target with more data."""
    print(f"\n--- top {k} {unit} confusions per engine  (ref -> hyp; "
          f"{EMPTY} = nothing, ␣ = space) ---")
    for name in sorted(results, key=lambda x: results[x]["summary"]["aer"]):
        rows = results[name]["summary"]["confusions"][unit][:k]
        if not rows:
            continue
        cells = [f"{_show(r['ref'])}->{_show(r['hyp'])} ({r['count']})" for r in rows]
        print(f"\n  {name}")
        for i in range(0, len(cells), 5):
            print("    " + "   ".join(f"{c:<16}" for c in cells[i:i + 5]).rstrip())


def print_examples(results, references, keys, k=3):
    """Same lines across every engine — the quickest way to see how failures differ."""
    if k <= 0 or not references:
        return
    names = list(results)
    print(f"\n--- {min(k, len(references))} sample lines ---")
    for i in range(min(k, len(references))):
        print(f"\n[{keys[i]}]")
        print(f"  {'reference':<16} {references[i]!r}")
        for name in names:
            print(f"  {name:<16} {results[name]['predictions'][i]!r}")


def main(dataset_repo, split, engine_specs, n_samples, seed, out_json,
         token=None, image_field="line_image", text_field="text",
         print_k=3, confusion_k=10):
    ds = load_bench(dataset_repo, split, token=token)
    if len(ds) == 0:
        raise SystemExit(f"{dataset_repo} split={split} is empty")

    missing = preflight(engine_specs)
    if missing:
        raise SystemExit("missing files referenced by ENGINES:\n  " +
                         "\n  ".join(missing))

    images, references, keys = select_samples(
        ds, n_samples, seed, max_source_width(engine_specs),
        text_field=text_field, id_field=image_field)
    if not images:
        raise SystemExit("no usable samples")

    results, unavailable = {}, {}
    for spec in engine_specs:
        label = spec.get("engine", "?")
        # Construct, run, then release before the next engine is built. Holding all
        # of them at once segfaulted: two copies of our 325MB model on MPS, plus
        # paddle's and surya's runtimes, and surya died mid-batch on the full split.
        try:
            engine = build_engine(spec)
        except Exception as exc:
            unavailable[label] = f"{type(exc).__name__}: {exc}"
            print(f"[setup] SKIPPING {label}: {type(exc).__name__}: {exc}")
            continue

        print(f"\n[run] {engine.name} over {len(images)} lines ...")
        t0 = time.time()
        predictions = engine(images)
        elapsed = time.time() - t0

        summary = score(predictions, references)
        results[engine.name] = {
            "summary": summary,
            "seconds": elapsed,
            "seconds_per_line": elapsed / max(len(images), 1),
            "predictions": predictions,
            "n_failed": getattr(engine, "n_failed", 0),
        }
        print(f"[run] {engine.name}: CER {summary['cer']:.4f}  "
              f"AER {summary['aer']:.4f}  WER {summary['wer']:.4f}  "
              f"exact {summary['exact']:.3f}  ({elapsed:.1f}s)")
        engine.close()
        release(engine)

    if not results:
        raise SystemExit("no engine produced any results")

    print_table(results)
    print_confusions(results, unit="akshara", k=confusion_k)
    print_examples(results, references, keys, k=print_k)
    if unavailable:
        print("\nnot benchmarked:")
        for name, why in unavailable.items():
            print(f"  {name}: {why}")

    if out_json:
        payload = {
            "config": {
                "dataset": dataset_repo, "split": split,
                "n_samples": n_samples, "seed": seed,
                "engines": engine_specs, "unavailable": unavailable,
            },
            "summary": {k: v["summary"] | {"seconds_per_line": v["seconds_per_line"]}
                        for k, v in results.items()},
            "samples": [
                {"line_image": keys[i], "reference": references[i],
                 **{name: results[name]["predictions"][i] for name in results}}
                for i in range(len(references))
            ],
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nper-sample results -> {out_json}")

    return results


if __name__ == "__main__":
    ROOT = "/Users/xai/Personal/Projects/TeluguOCR"

    # ------------------------------- CONFIG -------------------------------
    # The published benchmark split (built by
    # data_curation/wikisource/push_to_hub.py (WHICH="eval") from the hand-corrected lines).
    DATASET_REPO = "harsha-desaraju/telugu-line-ocr-bench"
    SPLIT = "test"
    HF_TOKEN = None       # None uses the cached login / HF_TOKEN env var; public repo

    VOCAB_FILE = f"{ROOT}/src/text_decoder/grapheme_tokenizer/telugu-vocab.json"
    CKPT = (f"{ROOT}/models/encoder_decoder/results_stage_2_mid/telugu-ocr-stage2/"
            f"checkpoint-34000/model.safetensors")

    N_SAMPLES = None        # None = the whole split; an int subsamples it (seeded)
    SEED = 42
    OUT_JSON = f"{ROOT}/benchmark/benchmark_results.json"
    PRINT_K = 3           # sample lines printed side by side
    CONFUSION_K = 10      # top akshara confusions printed per engine

    # Add an engine here; the rest of the file does not change.
    ENGINES = [
        # psm 13 + `tel` alone, not the repo's usual psm 7 + `tel+eng` — see
        # TesseractEngine's docstring for the measurements behind that.
        {"engine": "tesseract", "lang": "tel", "psm": 13, "upscale": 2.0},
        {"engine": "paddle", "lang": "te"},
        # math_mode off, longest <br>-segment — see SuryaEngine for the measurements.
        {"engine": "surya", "math_mode": False, "take": "longest"},
        {"engine": "model", "checkpoint": CKPT, "vocab_file": VOCAB_FILE,
         "decode": "ctc"},
        {"engine": "model", "checkpoint": CKPT, "vocab_file": VOCAB_FILE,
         "decode": "joint", "lam": 0.3, "beam_width": 5},
    ]
    # -----------------------------------------------------------------------

    main(DATASET_REPO, SPLIT, ENGINES, N_SAMPLES, SEED, OUT_JSON,
         token=HF_TOKEN, print_k=PRINT_K, confusion_k=CONFUSION_K)
