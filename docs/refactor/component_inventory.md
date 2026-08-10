# Component inventory — what is duplicated, what has drifted, what is canonical

> **Paths in this document are PRE-REFACTOR and deliberately left that way.** It is a dated
> record of the state phase 0 measured, and several of its claims are about files at their
> old locations (e.g. "`src/image_encoder/train.py` — removed"); rewriting the paths would
> make those claims false. Phase 1 moved everything and phase 2 rewrote the references
> everywhere else. Translate with: `src/image_encoder/model.py` →
> `src/telugu_ocr/models/image_encoder.py`, `src/image_encoder/utils.py` →
> `src/telugu_ocr/data/preprocess.py`, `src/text_decoder/grapheme_tokenizer/tokenizer.py` →
> `src/telugu_ocr/tokenizer/grapheme.py`, `src/*/train*.py` →
> `src/telugu_ocr/training/loops/`, `data_curation/` → `pipelines/`.

Phase 0 of the package refactor. This file answers the one question the refactor cannot
proceed without: **when N copies of a component become one, which copy is correct?**

Regenerate the numbers with the AST scan described at the bottom; verify the conclusions
with `python3 -m tests.test_checkpoint_compat`.

## The headline

Across the 14 files that carry model, tokenizer, preprocessing or augmentation code:

| | count |
|---|---|
| top-level definitions appearing in more than one file | 59 |
| byte-identical across all their copies | 28 |
| **drifted** — same name, different code | **31** |

Drift by fan-out: one definition has 5 copies, 14 have 4, 9 have 3, 7 have 2.

Worst cases:

| definition | copies | distinct versions |
|---|---|---|
| `TeluguGraphemeTokenizer` | 5 | 3 |
| `MultiHeadAttention` | 4 | 4 |
| `ImagePreprocessor` | 4 | 3 |
| `CTCEncoderConfig` | 4 | 3 |
| `ImageEncoderCTC` | 4 | 3 |
| `GPTModel` | 4 | 3 |
| `ConvStem` / `ConvBlock` / `DropPath` | 4 | 3 |
| `degrade` / `make_augmenter` (augmentation) | 4 | 2 |

This is the direct consequence of the repo's stated convention — training scripts inline
copies of the model so they can be uploaded to Kaggle standalone. The convention is sound;
hand-copying is what let the copies diverge. `scripts/bundle.py` (Phase 4) is meant to keep
the benefit and remove the mechanism.

## Which copy is canonical

**The `src/*/model.py` modules are structurally correct.** All three trained checkpoints
load against them with `strict=True` — no missing keys, no unexpected keys, no shape
mismatches (149 / 181 / 388 tensors). So the refactor should collapse *onto* them.

An earlier reading of this said the opposite. That was wrong, and the reason is worth
recording because it is the same trap the refactor has to avoid: the modules were tested
with their **dataclass defaults**, and the defaults describe no trained artifact.

| config | dataclass default | what the checkpoints actually used |
|---|---|---|
| `CTCEncoderConfig.max_image_width` | 1024 | **2048** |
| `CTCEncoderConfig.max_frames` | 128 | **256** |
| `GPTConfig.num_layers` | (shallower) | **16** |

Build with the defaults and the encoder fails on one tensor's shape, while the decoder
silently ignores 44 of the checkpoint's tensors. So the problem was never the architecture
code — it was that **nothing recorded which config went with which checkpoint.** That is
now `configs/checkpoints.yaml`, and the compat test holds it in place.

Corollary: the 31 drifted definitions differ in ways that are mostly *not* structural —
default values, forward-path details, comments. Structural equivalence is already proven by
the strict loads. What is not yet proven is **behavioural** equivalence between the inlined
training-script copies and the `src/` copies; the compat test's forward probe covers the
`src/` copies only. Establishing it for the training-script copies needs them importable
first (see the blocker below).

## Confirmed dead code, removed or to remove

| item | status |
|---|---|
| `src/image_encoder/train.py` | **removed** — imported `ViTConfig`/`MaskedAutoEncoder`, deleted during the CTC refactor. Unrunnable. |
| `EncoderDecoderEngine` in `pseudo_labelling/consensus_labelling.py` | **fixed** — imported a `ViTConfig` that exists nowhere, called the encoder with the ViT-MAE signature `(imgs, padding_mask=, mask_ratio=)`, and skipped `enc_to_dec`, so the decoder would have got 384-dim features where it wants 512. Rewritten against `encode(images, input_lengths)`; verified transcribing real crops. |
| `CLAUDE.md` architecture section | **stale** — documents a ViT Masked Auto-Encoder image encoder. The trained encoder is a conv-stem CTC model; no MAE class exists in the repo. |

The `EncoderDecoderEngine` case is the pattern to watch for: a lazy import inside a
constructor means nothing exercises the code until someone selects that engine, so it can
rot for months without a single error.

## Known blocker for Phase 3

`src/image_encoder/train_ctc_encoder.py` **cannot be imported** — `image_augmentation`-style
code in it reaches `cv2.FONT_HERSHEY_SIMPLEX` at import time, and the pinned opencv broke
that. Until it imports, the inlined copies cannot be diffed behaviourally against `src/`,
which is Phase 3's oracle for the augmentation and layer collapse. `pyproject.toml` now caps
`opencv-python<5`; run `uv sync` before starting Phase 3.

## Reproducing the scan

```python
# For each file: hash the source segment of every top-level class/def, group by name,
# and report names whose copies do not all share one hash.
import ast, pathlib, hashlib, collections
```

Files in scope: the three training scripts (`train_stage_1`, `train_stage_2`,
`train_ctc_encoder`), the four library modules (`image_encoder/model`, `image_encoder/utils`,
`text_decoder/model`, `encoder_decoder/model`), the tokenizer, `text_decoder/train`,
`text_line_images/image_augmentation`, `pseudo_labelling/consensus_labelling`,
`benchmark/engines`, and the two `test_model` scripts.
