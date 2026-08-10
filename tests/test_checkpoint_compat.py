"""Assert the model code still matches the trained checkpoints.

WHAT THIS IS FOR
    It is the regression net for the package refactor. Moving files is safe; collapsing the
    duplicated model definitions is not, because 28 of the repo's duplicated top-level
    definitions have drifted apart. When four copies of `ImageEncoderCTC` become one, the
    question "did I pick the copy the checkpoints were trained with?" has to be answerable
    mechanically. Run this before the refactor to get a green baseline, and after every step
    of it.

    A failure here does not mean the code is worse. It means the architecture changed, and
    you now have to decide whether that was intentional. If it was, regenerate the baseline
    with `python3 -m tests.make_fingerprints` -- never to silence a failure you have not
    explained.

TWO MODES, PICKED AUTOMATICALLY
    full          models/ is present (~5.8 GB, gitignored). Every checkpoint is loaded with
                  strict=True and a deterministic forward pass is compared against the
                  recorded baseline. This catches structural AND behavioural drift.
    fingerprint   models/ is absent, e.g. a fresh clone or CI. Models are still built from
                  configs/ and their state_dict key set and shapes are compared against the
                  baseline recorded in tests/checkpoint_fingerprints.json. This catches
                  structural drift with no weights on disk, which is the drift that breaks
                  checkpoint loading.

    So the useful check runs anywhere; only the numerical half needs the weights.

USAGE
    python3 -m tests.test_checkpoint_compat        # exits non-zero on failure
    pytest tests/test_checkpoint_compat.py         # also works, if you add pytest
"""

from __future__ import annotations

import sys

from tests.model_registry import (build_model, build_tokenizer, checkpoint_entries,
                                  checkpoint_path, forward_probe, load_state, load_yaml,
                                  read_fingerprints, structure_signature, tensor_stats)

# Numerical tolerance for the forward probe. Deliberately not an exact comparison: the same
# computation on a different BLAS, torch build or CPU gives slightly different last bits,
# and a test that fails on that teaches you to ignore it. These bounds are tight enough
# that computing something genuinely different cannot slip through.
RTOL = 1e-4
ATOL = 1e-5
ABS_SUM_RTOL = 1e-3        # a sum over ~100k elements accumulates more error


def _close(a: float, b: float, rtol: float) -> bool:
    return abs(a - b) <= (ATOL + rtol * abs(b))


def check_tokenizer() -> list[str]:
    """The vocab defines every model's output space, so a swap must be loud."""
    cfg = load_yaml("configs/tokenizer.yaml")
    expected = cfg["expected"]
    tok = build_tokenizer(cfg)
    got = {
        "length": len(tok),
        "pad_token_id": tok.pad_token_id,
        "bos_token_id": tok.bos_token_id,
        "eos_token_id": tok.eos_token_id,
    }
    return [f"tokenizer.{k}: expected {v}, got {got[k]}"
            for k, v in expected.items() if got[k] != v]


def check_structure(name: str, model_cfg: dict, baseline: dict, tokenizer) -> list[str]:
    """Build from config alone and compare the state_dict signature. No weights needed."""
    model = build_model(model_cfg, tokenizer)
    got = structure_signature(model.state_dict())
    want = baseline["structure"]
    failures = []
    for field in ("n_tensors", "n_parameters", "keys_sha256", "shapes_sha256"):
        if got[field] != want[field]:
            failures.append(f"{name}.structure.{field}: expected {want[field]}, "
                            f"got {got[field]}")
    return failures


def check_forward(name: str, entry: dict, model_cfg: dict, baseline: dict,
                  tokenizer) -> list[str]:
    """Load the real weights strictly and compare a deterministic forward pass."""
    model = build_model(model_cfg, tokenizer)
    state = load_state(checkpoint_path(entry))
    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        return [f"{name}: strict load FAILED: {type(exc).__name__}: "
                f"{str(exc).replace(chr(10), ' ')[:300]}"]

    got = tensor_stats(forward_probe(model, model_cfg["kind"], model_cfg))
    want = baseline["forward"]
    failures = []
    if got["shape"] != want["shape"]:
        return [f"{name}.forward.shape: expected {want['shape']}, got {got['shape']}"]
    for field in ("mean", "std", "min", "max"):
        if not _close(got[field], want[field], RTOL):
            failures.append(f"{name}.forward.{field}: expected {want[field]}, "
                            f"got {got[field]}")
    if not _close(got["abs_sum"], want["abs_sum"], ABS_SUM_RTOL):
        failures.append(f"{name}.forward.abs_sum: expected {want['abs_sum']}, "
                        f"got {got['abs_sum']}")
    return failures


def run() -> tuple[list[str], list[str]]:
    """Returns (failures, notes)."""
    failures, notes = [], []

    fingerprints = read_fingerprints()
    if not fingerprints:
        return (["tests/checkpoint_fingerprints.json is missing -- run "
                 "`python3 -m tests.make_fingerprints` on a machine with models/"], notes)

    failures += check_tokenizer()
    tokenizer = build_tokenizer()
    baselines = fingerprints.get("checkpoints", {})

    for name, entry, model_cfg in checkpoint_entries():
        baseline = baselines.get(name)
        if baseline is None:
            notes.append(f"{name}: no baseline recorded; not checked")
            continue

        failures += check_structure(name, model_cfg, baseline, tokenizer)

        if checkpoint_path(entry).exists():
            failures += check_forward(name, entry, model_cfg, baseline, tokenizer)
            notes.append(f"{name}: full check (structure + strict load + forward)")
        else:
            notes.append(f"{name}: structure only -- {entry['path']} not on this machine")

    for name in baselines:
        if name not in {n for n, _, _ in checkpoint_entries()}:
            failures.append(f"{name}: in the fingerprint baseline but no longer in "
                            f"configs/checkpoints.yaml")
    return failures, notes


def main() -> int:
    failures, notes = run()
    for n in notes:
        print(f"  [info] {n}")
    print()
    if failures:
        print(f"FAILED ({len(failures)} problem(s)):")
        for f in failures:
            print(f"   - {f}")
        print("\nIf the change was intentional, re-record the baseline:")
        print("   python3 -m tests.make_fingerprints")
        return 1
    print("PASSED: model code matches every recorded checkpoint signature.")
    return 0


# --- pytest entry points (optional; the module runs standalone too) ---
def test_tokenizer():
    assert check_tokenizer() == []


def test_checkpoint_compat():
    failures, _ = run()
    assert failures == [], "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
