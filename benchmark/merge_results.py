"""Combine benchmark result files from separate runs into one.

WHY THIS EXISTS
    Engines do not always survive being run together. Surya in particular exhausts
    something internally partway through a long pass and takes the process down with
    it, so it is run on its own and merged in afterwards. This keeps that workflow
    honest: separate runs, one comparison table, with the conditions for combining
    them actually checked rather than assumed.

WHAT IS CHECKED BEFORE MERGING (all fatal)
    * same dataset and split
    * identical sample sets -- the same line ids, no more, no fewer
    * identical references for every line
    * no engine appearing in two files
    Corpus error rates are sums over a sample set, so merging results scored on
    DIFFERENT lines would produce a table whose rows are not comparable, silently.
    Rather than recompute or interpolate, this refuses.

    Per-sample rows are joined by line id, not by position, so file order is
    irrelevant even though today's inputs happen to agree on it.

The merged file has the same shape as a single run's output, so anything that reads
one reads the other.

    python3 -m benchmark.merge_results
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmark.run_benchmark import print_confusions, print_table

ID = "line_image"
REF = "reference"


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_compatible(runs):
    """Fail loudly unless these runs scored the same lines against the same truth."""
    first_path, first = runs[0]
    base_ids = {r[ID] for r in first["samples"]}
    base_ref = {r[ID]: r[REF] for r in first["samples"]}
    seen_engines = {}

    for path, run in runs:
        cfg, other = first["config"], run["config"]
        for key in ("dataset", "split"):
            if cfg.get(key) != other.get(key):
                raise SystemExit(
                    f"{path}: {key} is {other.get(key)!r}, but {first_path} used "
                    f"{cfg.get(key)!r}. Refusing to merge runs over different data.")

        ids = {r[ID] for r in run["samples"]}
        if ids != base_ids:
            missing, extra = base_ids - ids, ids - base_ids
            raise SystemExit(
                f"{path}: sample set differs from {first_path} "
                f"({len(missing)} missing, {len(extra)} extra). Corpus error rates "
                f"are sums over the scored lines, so these numbers are not "
                f"comparable. Re-run the engines over the same rows.")

        mismatched = [r[ID] for r in run["samples"] if base_ref[r[ID]] != r[REF]]
        if mismatched:
            raise SystemExit(
                f"{path}: {len(mismatched)} references differ from {first_path}, "
                f"e.g. {mismatched[0]}. The dataset changed between runs.")

        for engine in run["summary"]:
            if engine in seen_engines:
                raise SystemExit(
                    f"engine {engine!r} appears in both {seen_engines[engine]} and "
                    f"{path}. Rename one, or drop the older run.")
            seen_engines[engine] = path

    return base_ids


def merge(runs):
    _, first = runs[0]

    summary, engine_specs, sources = {}, [], []
    for path, run in runs:
        summary.update(run["summary"])
        engine_specs.extend(run["config"].get("engines", []))
        sources.append({
            "file": str(path),
            "engines": list(run["summary"]),
            "n_samples": run["config"].get("n_samples"),
            "seed": run["config"].get("seed"),
        })

    # Join per-line rows by id, keeping the first run's order.
    rows = {r[ID]: dict(r) for r in first["samples"]}
    for _, run in runs[1:]:
        for r in run["samples"]:
            row = rows[r[ID]]
            for key, value in r.items():
                if key not in (ID, REF):
                    row[key] = value

    return {
        "config": {
            "dataset": first["config"].get("dataset"),
            "split": first["config"].get("split"),
            "n_samples": first["config"].get("n_samples"),
            "seed": first["config"].get("seed"),
            "engines": engine_specs,
            "merged_from": sources,
        },
        "summary": summary,
        "samples": [rows[r[ID]] for r in first["samples"]],
    }


def as_results(summary):
    """Shape the merged summary like a live run so print_table can render it."""
    return {name: {"summary": s,
                   "seconds_per_line": s.get("seconds_per_line", float("nan"))}
            for name, s in summary.items()}


def main(inputs, out_path, confusion_k=10):
    runs = [(Path(p), load(p)) for p in inputs]
    for path, run in runs:
        print(f"[in]  {path}: {len(run['samples'])} samples, "
              f"engines {list(run['summary'])}")

    ids = check_compatible(runs)
    merged = merge(runs)
    print(f"[ok]  {len(runs)} runs agree on {len(ids)} lines and their references")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print_table(as_results(merged["summary"]))
    if confusion_k:
        print_confusions(as_results(merged["summary"]), unit="akshara", k=confusion_k)
    print(f"\nmerged {len(merged['summary'])} engines over "
          f"{len(merged['samples'])} lines -> {out_path}")
    return merged


if __name__ == "__main__":
    ROOT = Path("/Users/xai/Personal/Projects/TeluguOCR/benchmark")

    # ------------------------------- CONFIG -------------------------------
    INPUTS = [
        ROOT / "benchmark_results.json",
        ROOT / "benchmark_results_surya.json",
    ]
    # Written fresh; the inputs are left untouched so a bad merge costs nothing.
    OUT_JSON = ROOT / "benchmark_results_all.json"
    CONFUSION_K = 10
    # -----------------------------------------------------------------------

    main(INPUTS, OUT_JSON, confusion_k=CONFUSION_K)
