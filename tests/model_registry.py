"""Build the repo's models from configs/, and probe them deterministically.

Shared by tests/make_fingerprints.py (which records what the models look like) and
tests/test_checkpoint_compat.py (which asserts they still look like that). Both go through
this one module on purpose: if the generator and the checker each had their own idea of how
to construct a model, a refactor could change both together and the test would pass while
the architecture drifted.

Everything here is CPU + float32 + eval-mode, with inputs built from arange rather than an
RNG, so a probe is reproducible across machines and torch versions to within floating-point
tolerance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_REGISTRY = REPO_ROOT / "configs" / "checkpoints.yaml"
FINGERPRINT_FILE = Path(__file__).resolve().parent / "checkpoint_fingerprints.json"

# Probe geometry. Fixed constants, not defaults read from a config: the probe must stay
# byte-identical across refactors even if the configs change, or fingerprints from
# different runs are not comparable.
PROBE_BATCH = 2
PROBE_WIDTH = 256          # multiple of downsample(8), well under max_image_width
PROBE_TOKENS = 16


def load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with open(p, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------------
# Tokenizer
# ----------------------------------------------------------------------------------
def build_tokenizer(cfg: dict | None = None):
    """The grapheme tokenizer described by configs/tokenizer.yaml."""
    cfg = cfg or load_yaml("configs/tokenizer.yaml")
    from src.text_decoder.grapheme_tokenizer.tokenizer import TeluguGraphemeTokenizer

    return TeluguGraphemeTokenizer(vocab_file=str(REPO_ROOT / cfg["vocab_file"]))


# ----------------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------------
def build_model(model_cfg: dict, tokenizer):
    """Construct an un-loaded model from a configs/models/*.yaml body."""
    kind = model_cfg["kind"]

    if kind == "ctc_encoder":
        from src.image_encoder.model import CTCEncoderConfig, ImageEncoderCTC

        return ImageEncoderCTC(CTCEncoderConfig(**model_cfg["config"]))

    if kind == "gpt_decoder":
        from src.text_decoder.model import GPTConfig, GPTModel

        return GPTModel(GPTConfig(**model_cfg["config"]), pad_index=tokenizer.pad_token_id)

    if kind == "encoder_decoder":
        from src.encoder_decoder.model import EncoderDecoder
        from src.image_encoder.model import CTCEncoderConfig
        from src.text_decoder.model import GPTConfig

        enc = load_yaml(model_cfg["encoder"])
        dec = load_yaml(model_cfg["decoder"])
        return EncoderDecoder(
            encoder_config=CTCEncoderConfig(**enc["config"]),
            decoder_config=GPTConfig(**dec["config"]),
            pad_index=tokenizer.pad_token_id,
        )

    raise ValueError(f"unknown model kind {kind!r} in configs/models")


def load_state(path: str | Path) -> dict:
    """A checkpoint's state dict, from .safetensors or a torch .pt."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(p))
    return torch.load(str(p), map_location="cpu")


# ----------------------------------------------------------------------------------
# Structural signature: what the architecture IS, independent of weight values
# ----------------------------------------------------------------------------------
def structure_signature(state: dict) -> dict:
    """Key set and per-key shape/dtype, hashed.

    This is the part that catches a refactor silently changing the model: renaming a
    submodule, reordering blocks, changing a width. It needs only a state dict, so it
    works against a checkpoint OR against a freshly constructed model -- which is what
    lets the test run on a clone that has no weights.
    """
    keys = sorted(state.keys())
    shapes = [f"{k}:{tuple(state[k].shape)}" for k in keys]
    return {
        "n_tensors": len(keys),
        "n_parameters": int(sum(state[k].numel() for k in keys)),
        "keys_sha256": _sha("\n".join(keys)),
        "shapes_sha256": _sha("\n".join(shapes)),
    }


# ----------------------------------------------------------------------------------
# Behavioural signature: does a loaded model still compute the same thing
# ----------------------------------------------------------------------------------
def _probe_images(width: int = PROBE_WIDTH) -> torch.Tensor:
    """Deterministic (B, 1, 64, W) input in [-1, 1]. arange, not randn."""
    n = PROBE_BATCH * 64 * width
    x = (torch.arange(n, dtype=torch.float32) % 251) / 251.0     # 251 prime -> no row alias
    return (x.reshape(PROBE_BATCH, 1, 64, width) - 0.5) / 0.5


def _probe_ids(vocab_size: int) -> torch.Tensor:
    n = PROBE_BATCH * PROBE_TOKENS
    return (torch.arange(n, dtype=torch.long) % max(1, vocab_size - 8)).reshape(
        PROBE_BATCH, PROBE_TOKENS)


def forward_probe(model, kind: str, model_cfg: dict) -> torch.Tensor:
    """Run one deterministic forward pass and return a float output tensor."""
    model = model.to("cpu").eval()
    torch.manual_seed(0)                      # only matters if a layer ignores eval()

    with torch.no_grad():
        if kind == "ctc_encoder":
            images = _probe_images()
            lengths = torch.full((PROBE_BATCH,), PROBE_WIDTH // model_cfg["config"]["downsample"],
                                dtype=torch.long)
            out = model(images, lengths)
            return out["log_probs"].float()

        if kind == "gpt_decoder":
            ids = _probe_ids(model_cfg["config"]["vocab_size"])
            out = model(ids)
            return (out.logits if hasattr(out, "logits") else out).float()

        if kind == "encoder_decoder":
            dec = load_yaml(model_cfg["decoder"])
            enc = load_yaml(model_cfg["encoder"])
            images = _probe_images()
            lengths = torch.full((PROBE_BATCH,), PROBE_WIDTH // enc["config"]["downsample"],
                                 dtype=torch.long)
            ids = _probe_ids(dec["config"]["vocab_size"])
            out = model(images, ids, lengths)
            return (out.logits if hasattr(out, "logits") else out).float()

    raise ValueError(f"no probe defined for kind {kind!r}")


def tensor_stats(t: torch.Tensor) -> dict:
    """Summary statistics, compared with a tolerance rather than hashed.

    A hash of the output would be exact but useless in practice: it flips on the last bit
    of a float, so a torch upgrade or a different BLAS would fail the test for no real
    reason. Stats plus a tolerance distinguish "numerically equivalent" from "computes
    something else", which is the actual question.
    """
    t = t.detach().float()
    return {
        "shape": list(t.shape),
        "mean": round(float(t.mean()), 6),
        "std": round(float(t.std()), 6),
        "min": round(float(t.min()), 6),
        "max": round(float(t.max()), 6),
        "abs_sum": round(float(t.abs().sum()), 3),
    }


# ----------------------------------------------------------------------------------
# Registry access
# ----------------------------------------------------------------------------------
def registry() -> dict:
    return load_yaml(CHECKPOINT_REGISTRY)


def checkpoint_entries() -> list[tuple[str, dict, dict]]:
    """[(name, registry_entry, model_cfg)] for every checkpoint in the registry."""
    reg = registry()
    out = []
    for name, entry in (reg.get("checkpoints") or {}).items():
        out.append((name, entry, load_yaml(entry["model"])))
    return out


def checkpoint_path(entry: dict) -> Path:
    return REPO_ROOT / entry["path"]


def read_fingerprints() -> dict:
    if not FINGERPRINT_FILE.exists():
        return {}
    with open(FINGERPRINT_FILE, encoding="utf-8") as fh:
        return json.load(fh)
