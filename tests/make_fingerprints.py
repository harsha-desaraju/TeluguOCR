"""Regenerate tests/checkpoint_fingerprints.json from the checkpoints on this machine.

Run this ONLY when a change to the models or the configs is intentional -- it overwrites
the baseline that tests/test_checkpoint_compat.py checks against, so running it to make a
failing test pass discards the very signal the test exists to give you.

Needs the checkpoints in models/ (gitignored, ~5.8 GB), so it only runs on a machine that
has them. The JSON it writes is a few KB and IS committed, which is what lets the test
detect architecture drift on a fresh clone that has no weights at all.

    python3 -m tests.make_fingerprints
"""

from __future__ import annotations

import json
import sys

from tests.model_registry import (FINGERPRINT_FILE, build_model, build_tokenizer,
                                  checkpoint_entries, checkpoint_path, forward_probe,
                                  load_state, load_yaml, structure_signature, tensor_stats)


def main() -> int:
    tok_cfg = load_yaml("configs/tokenizer.yaml")
    tokenizer = build_tokenizer(tok_cfg)

    out = {
        "_README": (
            "Baseline signatures for every checkpoint in configs/checkpoints.yaml. "
            "Regenerate with `python3 -m tests.make_fingerprints` only when a model or "
            "config change is intentional. Checked by tests/test_checkpoint_compat.py."
        ),
        "tokenizer": {
            "vocab_file": tok_cfg["vocab_file"],
            "length": len(tokenizer),
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        },
        "checkpoints": {},
    }

    missing = []
    for name, entry, model_cfg in checkpoint_entries():
        path = checkpoint_path(entry)
        if not path.exists():
            missing.append(f"{name} -> {entry['path']}")
            continue

        print(f"[{name}] building {model_cfg['kind']} from {entry['model']}")
        model = build_model(model_cfg, tokenizer)
        state = load_state(path)

        # strict=True is the point: it fails on a missing key, an extra key, OR a shape
        # mismatch, which together are every way a refactor can silently change the model.
        model.load_state_dict(state, strict=True)
        print(f"[{name}] strict load OK ({len(state)} tensors)")

        probe = forward_probe(model, model_cfg["kind"], model_cfg)
        record = {
            "path": entry["path"],
            "model_config": entry["model"],
            "kind": model_cfg["kind"],
            "structure": structure_signature(state),
            "forward": tensor_stats(probe),
        }
        out["checkpoints"][name] = record
        print(f"[{name}] probe {record['forward']['shape']} "
              f"mean={record['forward']['mean']}\n")

    if missing:
        print("NOT fingerprinted (checkpoint absent on this machine):")
        for m in missing:
            print(f"   {m}")
        print()

    if not out["checkpoints"]:
        print("No checkpoints found -- refusing to write an empty baseline.")
        return 1

    with open(FINGERPRINT_FILE, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"wrote {FINGERPRINT_FILE.relative_to(FINGERPRINT_FILE.parents[1])} "
          f"({len(out['checkpoints'])} checkpoints)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
