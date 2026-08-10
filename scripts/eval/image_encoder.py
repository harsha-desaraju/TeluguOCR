"""
Diagnostic eval for the conv-stem CTC image encoder (ctc_encoder.py).
=====================================================================

Loads a trained ``ImageEncoderCTC`` checkpoint (.safetensors) and runs it over a
HuggingFace dataset of Telugu line images to surface *where* the model is failing,
following the investigation playbook:

  1. Print N predictions next to ground truth (+ per-sample CER). Signatures:
       - systematic missing/wrong vowel signs or conjunct pieces everywhere
         -> renderer/tokenizer label mismatch (also flagged: vocab-size check);
       - right-but-shifted/truncated on long lines -> input_lengths / bucketing bug;
       - garbage on some lines, perfect on others -> a broken data subset;
       - plausible look-alike confusions on degraded images -> difficulty, not a bug.
  2. Per-sample CER histogram. Uniform mediocrity -> global cause; bimodal
     (a 0% cluster + an 80-100% cluster) -> broken subset (usually a font).
  3. CER split by degradation: this dataset is clean-rendered, so the reported CER
     IS the clean baseline. Set RUN_AUGMENTED=True to also apply the train-time
     augmenter on the fly and print the clean-vs-degraded gap (needs augraphy).
  4. The CER computation is self-checked on hand-worked examples at startup
     (grapheme-level edit distance + CTC collapse), so a metric bug is ruled out.

Breakdowns reported (deliverables): overall CER, then CER per `font`,
`text_source`, `language`, `is_fragment`, `cut_type`, `font_size`, and
`text_len` buckets — each with group size so small-sample groups are obvious.

The functions above (``run_inference``, ``results_dataframe``, ``render_samples``,
``cer_by_column`` / ``cer_table``, ``print_histogram`` …) power the labelled-dataset
diagnostic and are what ``test_model.ipynb`` imports.

``__main__`` instead runs the simplest thing: point ``FOLDER_PATH`` at a folder of
line images, run the model over them, and print each ``filename: prediction``. Set
``PLOT=True`` to also stack the images with their predictions into one matplotlib
figure (optionally via ``FONT_PATH`` for the Telugu text). No ground truth needed.

Runs on a Mac (MPS if available, else CPU). Edit the CONFIG block in __main__.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from collections import defaultdict

import torch
import regex
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from safetensors.torch import load_file

# --- imports resolve whether run as `python3 test_model.py` (from this dir) or
#     `python3 -m src.image_encoder.test_model` (from the repo root) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import CTCEncoderConfig, ImageEncoderCTC  # noqa: E402
from src.text_decoder.grapheme_tokenizer.tokenizer import (  # noqa: E402
    TeluguGraphemeTokenizer,
)


# ============================================================================
# Grapheme-level CER
# ============================================================================
_GRAPHEME = regex.compile(r"\X")


def grapheme_split(text: str) -> list[str]:
    """Split a string into Unicode grapheme clusters (aksharas)."""
    return _GRAPHEME.findall(text or "")


def edit_distance(a: list, b: list) -> int:
    """Levenshtein edit distance between two sequences."""
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def sample_cer(pred: str, ref: str) -> tuple[int, int, float]:
    """Grapheme-level (edits, ref_length, CER) for one prediction/reference pair."""
    p, r = grapheme_split(pred), grapheme_split(ref)
    edits = edit_distance(p, r)
    ref_len = len(r)
    return edits, ref_len, edits / max(ref_len, 1)


def ctc_collapse(ids: list[int], blank_id: int, pad_id: int = -100) -> list[int]:
    """Collapse a CTC frame-id sequence: drop consecutive dups, blanks, pads."""
    out, prev = [], None
    for t in ids:
        t = int(t)
        if t != prev and t != blank_id and t != pad_id:
            out.append(t)
        prev = t
    return out


# ============================================================================
# Preprocessing (grayscale -> aspect-resize to h=64 -> pad width to a multiple
# of `downsample` with white -> ToTensor + Normalize). Matches training's
# height-64 / white-pad convention; optional augment_fn mirrors train-time aug.
# ============================================================================
_TO_TENSOR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def preprocess(
    img: Image.Image,
    image_height: int = 64,
    max_image_width: int = 1024,
    downsample: int = 8,
    augment_fn=None,
) -> torch.Tensor:
    w, h = img.size
    scale = image_height / h
    if scale * w > max_image_width:
        # Too wide: scale to fit max width, center-pad height.
        tw = max_image_width
        th = max(1, round((max_image_width / w) * h))
        pad_t = (image_height - th) // 2
        pad_b = image_height - th - pad_t
        pad_l, pad_r = 0, 0
    else:
        th, tw = image_height, max(1, round(scale * w))
        pad_t, pad_b = 0, 0
        pad_l, pad_r = 0, (-tw) % downsample  # right-pad to a multiple of 8

    base = img.convert("RGB") if augment_fn is not None else img.convert("L")
    base = base.resize((tw, th))

    if augment_fn is not None:
        arr = augment_fn(np.array(base))              # (H, W, 3) uint8
        base = Image.fromarray(np.asarray(arr, dtype=np.uint8)).convert("L")

    base = transforms.Pad((pad_l, pad_t, pad_r, pad_b), fill=255)(base)  # white pad
    return _TO_TENSOR(base)                            # (1, 64, W), W % 8 == 0


# ============================================================================
# Model loading (config inferred from the checkpoint so it can't silently drift)
# ============================================================================
def load_model(path: str, num_heads: int, device: str):
    state = load_file(path)

    # Infer architecture from tensor shapes.
    num_classes = state["ctc_head.weight"].shape[0]
    embed_dim = state["ctc_head.weight"].shape[1]
    max_frames = state["pos_embed"].shape[1]
    mlp_dim = state["blocks.0.mlp.0.weight"].shape[0]
    num_layers = 1 + max(
        int(k.split(".")[1]) for k in state if k.startswith("blocks.")
    )

    cfg = CTCEncoderConfig(
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        max_frames=max_frames,
        max_image_width=max_frames * 8,
        vocab_size=num_classes - 1,   # blank appended at index vocab_size
        dropout=0.0,
        drop_path_rate=0.0,
    )
    model = ImageEncoderCTC(cfg)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load] missing={missing} unexpected={unexpected}")
    model.eval().to(device)
    print(
        f"[load] layers={num_layers} d={embed_dim} heads={num_heads} mlp={mlp_dim} "
        f"max_frames={max_frames} classes={num_classes} (blank={model.blank_id})"
    )
    return model, cfg


# ============================================================================
# Inference over a list of dataset rows
# ============================================================================
@torch.no_grad()
def run_inference(model, cfg, tokenizer, rows, image_col, device, batch_size,
                  augment_fn=None):
    """Returns a list of per-sample result dicts (predictions + CER + metadata)."""
    blank_id = model.blank_id

    tensors = [
        preprocess(r[image_col], cfg.image_height, cfg.max_image_width,
                   cfg.downsample, augment_fn)
        for r in rows
    ]
    # Width-sorted batching keeps padding waste (and MPS memory) low.
    order = sorted(range(len(tensors)), key=lambda i: tensors[i].shape[-1])

    results = [None] * len(rows)
    for start in range(0, len(order), batch_size):
        idxs = order[start:start + batch_size]
        max_w = max(tensors[i].shape[-1] for i in idxs)
        images = tensors[0].new_full((len(idxs), 1, cfg.image_height, max_w), 1.0)
        input_lengths = torch.empty(len(idxs), dtype=torch.long)
        for j, i in enumerate(idxs):
            w = tensors[i].shape[-1]
            images[j, :, :, :w] = tensors[i]
            input_lengths[j] = w // cfg.downsample

        out = model(images.to(device), input_lengths=input_lengths.to(device))
        pred_ids = out["logits"].cpu().tolist()

        for j, i in enumerate(idxs):
            ids = ctc_collapse(pred_ids[j], blank_id)
            pred = tokenizer.decode(ids, skip_special_tokens=True)
            ref = rows[i]["text"]
            edits, ref_len, cer = sample_cer(pred, ref)
            width = tensors[i].shape[-1]
            results[i] = {
                "idx": i, "ref": ref, "pred": pred,
                "edits": edits, "ref_len": ref_len, "cer": cer,
                "width": width, "frames": width // cfg.downsample,
                "row": rows[i],
            }
    return results


# ============================================================================
# Reporting
# ============================================================================
def corpus_cer(results) -> float:
    edits = sum(r["edits"] for r in results)
    ref_len = sum(r["ref_len"] for r in results)
    return edits / max(ref_len, 1)


def print_samples(results, n, meta_cols):
    print("\n" + "=" * 88)
    print(f"PER-SAMPLE PREDICTIONS (first {min(n, len(results))} of {len(results)})")
    print("=" * 88)
    for r in results[:n]:
        row = r["row"]
        meta = "  ".join(f"{c}={row.get(c)}" for c in meta_cols if c in row)
        print(f"\n[{r['idx']}] CER={r['cer']*100:6.2f}%  {meta}")
        print(f"  ref : {r['ref']}")
        print(f"  pred: {r['pred']}")


def print_histogram(results, bins=20):
    cers = [min(r["cer"], 1.0) for r in results]
    mean = sum(cers) / len(cers)
    median = sorted(cers)[len(cers) // 2]
    print("\n" + "=" * 88)
    print("PER-SAMPLE CER HISTOGRAM")
    print("=" * 88)
    print(f"corpus CER: {corpus_cer(results)*100:.2f}%   "
          f"mean: {mean*100:.2f}%   median: {median*100:.2f}%   n={len(results)}")

    counts = [0] * bins
    for c in cers:
        counts[min(int(c * bins), bins - 1)] += 1
    peak = max(counts) or 1
    for b in range(bins):
        lo, hi = b * 100 // bins, (b + 1) * 100 // bins
        bar = "#" * round(40 * counts[b] / peak)
        print(f"  {lo:3d}-{hi:3d}% | {counts[b]:5d} {bar}")

    near0 = sum(c < 0.05 for c in cers) / len(cers)
    near1 = sum(c > 0.80 for c in cers) / len(cers)
    if near0 > 0.15 and near1 > 0.15:
        print(f"  -> BIMODAL: {near0*100:.0f}% near-perfect, {near1*100:.0f}% near-broken"
              f" — check the per-font table for a broken subset.")


def cer_stats(results, col, bucket_fn=None, min_count=1):
    """Group `results` by a column (optionally bucketed) and compute per-group
    corpus CER, mean per-sample CER and count. Returns a worst-first list of
    dicts: {group, n, corpus_cer, mean_cer}."""
    groups = defaultdict(list)
    for r in results:
        val = r["row"].get(col)
        key = bucket_fn(val) if bucket_fn is not None else val
        groups[key].append(r)

    rows = []
    for key, rs in groups.items():
        if len(rs) < min_count:
            continue
        rows.append({
            "group": key,
            "n": len(rs),
            "corpus_cer": corpus_cer(rs),
            "mean_cer": sum(r["cer"] for r in rs) / len(rs),
        })
    rows.sort(key=lambda x: x["corpus_cer"], reverse=True)  # worst first
    return rows


def cer_by_column(results, col, bucket_fn=None, min_count=1):
    rows = cer_stats(results, col, bucket_fn, min_count)
    print("\n" + "=" * 88)
    print(f"CER BY {col.upper()}  ({len(rows)} groups, worst first)")
    print("=" * 88)
    print(f"  {'group':<28} {'n':>6} {'corpusCER':>10} {'meanCER':>9}")
    for r in rows:
        print(f"  {str(r['group']):<28} {r['n']:>6} "
              f"{r['corpus_cer']*100:>9.2f}% {r['mean_cer']*100:>8.2f}%")


def _img_tag(img, height=48) -> str:
    """PIL image -> an inline base64 <img> tag (grayscale, scaled to `height`px)."""
    import io
    import base64
    im = img.convert("L")
    w, h = im.size
    if h != height:
        im = im.resize((max(1, round(w * height / h)), height))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" height="{height}"/>'


def results_dataframe(results, meta_cols=("font", "text_source", "language",
                                          "cut_type", "is_fragment", "font_size"),
                      with_image=False, image_col="image", thumb_height=48):
    """Per-sample table (idx, grapheme length, frames, CER%, metadata, ref, pred)
    as a pandas DataFrame — for inline display in the notebook.

    with_image=True adds an `image` column of inline <img> thumbnails; render it
    with ``render_samples`` or ``HTML(df.to_html(escape=False))`` (plain display
    escapes the tags and shows raw HTML)."""
    import pandas as pd
    data = []
    for r in results:
        row = r["row"]
        d = {"idx": r["idx"], "graphemes": r["ref_len"], "frames": r.get("frames"),
             "CER%": round(r["cer"] * 100, 2)}
        if with_image and image_col in row:
            d["image"] = _img_tag(row[image_col], thumb_height)
        for c in meta_cols:
            if c in row:
                d[c] = row[c]
        d["ref"] = r["ref"]
        d["pred"] = r["pred"]
        data.append(d)
    return pd.DataFrame(data)


def render_samples(results, sort_by="CER%", ascending=False, limit=None,
                   thumb_height=48, image_col="image",
                   meta_cols=("font", "text_source", "text_len", "font_size"),
                   content_cols=("CER%", "ref", "pred")):
    """Return an IPython.display.HTML table of per-sample rows WITH the line image
    rendered inline. Defaults to worst-CER-first so degraded/hard images surface at
    the top. Just evaluate the call in a notebook cell to render it.

    Only the metadata columns you pass in ``meta_cols`` are shown (the auto
    idx/graphemes/frames columns are dropped). ``content_cols`` are the always-kept
    prediction columns — pass ``content_cols=()`` for an image + meta_cols-only
    table. The image column is included whenever an image is available."""
    from IPython.display import HTML
    df = results_dataframe(results, meta_cols=meta_cols, with_image=True,
                           image_col=image_col, thumb_height=thumb_height)
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)
    if limit is not None:
        df = df.head(limit)
    # Display only: image + the requested meta_cols + the content columns.
    cols = (["image"] if "image" in df.columns else []) \
        + [c for c in meta_cols if c in df.columns] \
        + [c for c in content_cols if c in df.columns]
    return HTML(df[cols].to_html(escape=False, index=False))


def cer_table(results, col, bucket_fn=None, min_count=1):
    """cer_stats as a pandas DataFrame (group, n, corpus CER%, mean CER%)."""
    import pandas as pd
    rows = cer_stats(results, col, bucket_fn, min_count)
    return pd.DataFrame([{
        col: r["group"], "n": r["n"],
        "corpus_CER%": round(r["corpus_cer"] * 100, 2),
        "mean_CER%": round(r["mean_cer"] * 100, 2),
    } for r in rows])


def _bucket_int(edges):
    def f(v):
        if v is None:
            return "None"
        v = float(v)
        lo = 0
        for e in edges:
            if v <= e:
                return f"{lo}-{e}"
            lo = e + 1
        return f">{edges[-1]}"
    return f


# ============================================================================
# CER self-test (rules out a metric bug — investigation step 4)
# ============================================================================
def selftest_cer():
    # identical -> 0
    assert sample_cer("కారు", "కారు")[2] == 0.0
    # one unit substituted out of 4 (Latin = 1 grapheme each) -> exactly 0.25
    e, n, c = sample_cer("abxd", "abcd")
    assert (e, n) == (1, 4) and abs(c - 0.25) < 1e-9, (e, n, c)
    # akshara clustering: consonant+matra merge into ONE grapheme, so "కారు"
    # is 2 clusters (కా, రు) — CER is measured over these, not codepoints.
    assert grapheme_split("కారు") == ["కా", "రు"], grapheme_split("కారు")
    # one akshara substituted out of 2 -> 0.5
    assert abs(sample_cer("కీరు", "కారు")[2] - 0.5) < 1e-9
    # CTC collapse: [5,5,blank,7,7,7,5] -> [5,7,5] (dups collapse, blank drops)
    assert ctc_collapse([5, 5, 99, 7, 7, 7, 5], blank_id=99) == [5, 7, 5]
    # blank/pad only -> empty
    assert ctc_collapse([99, 99, -100], blank_id=99) == []
    print("[selftest] grapheme CER + CTC collapse: OK")


# ============================================================================
# ============================================================================
# Folder inference (no ground truth — just decode predictions)
# ============================================================================
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(folder: str) -> list[Path]:
    """All image files directly inside `folder` (non-recursive), name-sorted."""
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXTS)


@torch.no_grad()
def predict_images(model, cfg, tokenizer, images, device, batch_size=16,
                   augment_fn=None) -> list[str]:
    """Greedy-CTC-decode a list of PIL images; returns predicted strings in the
    same order as `images`. No ground truth needed."""
    tensors = [
        preprocess(im, cfg.image_height, cfg.max_image_width, cfg.downsample, augment_fn)
        for im in images
    ]
    order = sorted(range(len(tensors)), key=lambda i: tensors[i].shape[-1])
    preds = [None] * len(images)
    for start in range(0, len(order), batch_size):
        idxs = order[start:start + batch_size]
        max_w = max(tensors[i].shape[-1] for i in idxs)
        batch = tensors[0].new_full((len(idxs), 1, cfg.image_height, max_w), 1.0)
        input_lengths = torch.empty(len(idxs), dtype=torch.long)
        for j, i in enumerate(idxs):
            w = tensors[i].shape[-1]
            batch[j, :, :, :w] = tensors[i]
            input_lengths[j] = w // cfg.downsample
        out = model(batch.to(device), input_lengths=input_lengths.to(device))
        pred_ids = out["logits"].cpu().tolist()
        for j, i in enumerate(idxs):
            ids = ctc_collapse(pred_ids[j], model.blank_id)
            preds[i] = tokenizer.decode(ids, skip_special_tokens=True)
    return preds


def predict_folder(folder, model, cfg, tokenizer, device, batch_size=16):
    """Run the model over every image in `folder`. Returns a list of dicts
    {name, path, image, pred}."""
    paths = list_images(folder)
    images = [Image.open(p) for p in paths]
    preds = predict_images(model, cfg, tokenizer, images, device, batch_size)
    return [{"name": p.name, "path": str(p), "image": img, "pred": pred}
            for p, img, pred in zip(paths, images, preds)]


def _resolve_telugu_font(font_path=None):
    """Register and return a Telugu-capable matplotlib font family name, or None.
    Prefers an explicit path, then the repo's bundled Telugu .ttf files, then any
    Telugu font already known to matplotlib."""
    from matplotlib import font_manager as fm
    candidates = [font_path] if font_path else []
    repo_fonts = os.path.join(_ROOT, "data_curation/text_line_images/fonts")
    candidates += [os.path.join(repo_fonts, n)
                   for n in ("Gautami.ttf", "Vani.ttf", "mallanna.ttf", "NTR-Regular.ttf")]
    for c in candidates:
        if c and os.path.exists(c):
            try:
                fm.fontManager.addfont(c)
                return fm.FontProperties(fname=c).get_name()
            except Exception:
                pass
    for f in fm.fontManager.ttflist:
        if any(k in f.name.lower() for k in ("telugu", "gautami", "pothana")):
            return f.name
    return None


def plot_predictions(items, font_path=None, max_images=None, save_path=None,
                     row_height=1.1):
    """Stack each line image with its predicted text (filename → pred) into one
    matplotlib figure and display it. Returns the figure.

    NOTE: matplotlib lacks full complex-script shaping, so Telugu matras may be
    slightly mispositioned in the titles even with a Telugu font — the printed
    console output renders the text correctly."""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm

    if max_images is not None:
        items = items[:max_images]
    n = len(items)
    if n == 0:
        print("[plot] nothing to plot")
        return None

    fam = _resolve_telugu_font(font_path)
    if fam is None:
        print("[plot][WARN] no Telugu font found; predictions may render as boxes. "
              "Set FONT_PATH to a Telugu .ttf.")
    fp = fm.FontProperties(family=fam, size=11) if fam else None

    fig, axes = plt.subplots(n, 1, figsize=(12, row_height * n), squeeze=False)
    for ax, it in zip(axes[:, 0], items):
        ax.imshow(it["image"].convert("L"), cmap="gray", aspect="auto")
        ax.set_title(f"{it['name']}  →  {it['pred']}", fontproperties=fp,
                     fontsize=11, loc="left")
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"[plot] saved -> {save_path}")
    plt.show()
    return fig


if __name__ == "__main__":
    # -------------------------- CONFIG (edit me) --------------------------
    # MODEL_PATH = "/Users/xai/Personal/Projects/TeluguOCR/models/image_encoder/ctc_encoder/ctc-encoder/checkpoint-152000/model.safetensors"
    MODEL_PATH = "/Users/xai/Personal/Projects/TeluguOCR/models/image_encoder/ctc_encoder_stage-2/ctc-encoder-2048/model.safetensors"
    FOLDER_PATH = "/Users/xai/Personal/Projects/TeluguOCR/data/temp_test"   # folder of line images to read
    VOCAB_FILE = os.path.join(_ROOT, "src/text_decoder/grapheme_tokenizer/telugu-vocab.json")

    NUM_HEADS = 8                  # not inferable from weights; must match training
    BATCH_SIZE = 16
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

    PLOT = False                    # optionally plot images + predictions with matplotlib
    PLOT_MAX = 30                  # cap the number of images drawn in the plot
    FONT_PATH = None               # optional Telugu .ttf for the plot text; None = auto
    # ---------------------------------------------------------------------

    tokenizer = TeluguGraphemeTokenizer(vocab_file=VOCAB_FILE)
    print(f"[tokenizer] vocab size: {len(tokenizer)}")

    model, cfg = load_model(MODEL_PATH, NUM_HEADS, DEVICE)
    print(f"[device] {DEVICE}")
    if len(tokenizer) != cfg.vocab_size:
        print(f"[WARN] tokenizer vocab ({len(tokenizer)}) != model classes-1 "
              f"({cfg.vocab_size}); decoding will be wrong.")

    items = predict_folder(FOLDER_PATH, model, cfg, tokenizer, DEVICE, BATCH_SIZE)

    print(f"\n{len(items)} images in {FOLDER_PATH}")
    print("=" * 88)
    for it in items:
        print(f"{it['name']}: {it['pred']}")

    if PLOT and items:
        plot_predictions(items, font_path=FONT_PATH, max_images=PLOT_MAX,
                         save_path=os.path.join(_HERE, "folder_predictions.png"))
