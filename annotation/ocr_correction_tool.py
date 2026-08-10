"""
OCR Correction Tool — a lightweight desktop-style annotation app (Gradio).

Purpose
-------
Efficiently create ground-truth labels for a Telugu OCR model by *correcting*
existing OCR predictions instead of typing transcriptions from scratch.

Input
-----
Set `DATASET` to pick one of the profiles in `PROFILES`; everything else — paths,
field names, viewer sizing — follows from it. Two ship with the tool:

  "lines_eval"  data/wikisource_lines_eval/lines.jsonl — one record per *line*
                crop (64px tall), with `line_cer` from the model:
                {"line_image": "lines/<slug>_p0004_l007.jpg", "text": "ఆంధ్ర ...",
                 "line_cer": 0.0714, "accepted": true, ...}
                You are correcting a machine transcription.

  "benchmark"   data/wikisource_benchmark/pages.jsonl — one record per *page*
                scan (~1280x1900), with the human-proofread wikisource text:
                {"slug": ..., "image": "images/<slug>.jpg", "text": "<page>",
                 "page_no": 40, "n_chars": 742, "page_url": ...}
                You are correcting that transcription against the scan — the
                proofread text can carry running heads, folio numbers and
                reflowed lines that the scan does not support as ground truth.

A profile names the text/image/key fields, so any similarly-shaped JSONL works:
add an entry rather than editing the code.

The editable box is pre-filled with the source text, so there is no separate
read-only copy of it on screen; the untouched original is still recorded as
`original_text` on every saved row.

Which samples are shown is controlled by the profile's `filter_field` plus
`filter_min`/`filter_max` (either bound may be None, and `filter_field: None`
disables filtering): `line_cer` for lines_eval, so `filter_min = 0.05` reviews
only the lines the model most likely got wrong; `n_chars` for benchmark. Set
`SHUFFLE = True` to present them in random order (seeded by `SHUFFLE_SEED`)
rather than in file order.

Output
------
A *new* JSONL in the **profile's own directory**, named after its input:
`<input stem>_corrected.jsonl` (lines.jsonl -> lines_corrected.jsonl,
pages.jsonl -> pages_corrected.jsonl). The name is derived, not configured, so
a new profile can never write into another dataset's file.

Each record is the **full original record** with the text field replaced by
your correction, plus annotation bookkeeping:

  {...all original fields..., "text": "<corrected>",
   "original_text": "<what the model produced>", "corrected": true,
   "status": "completed", "annotated_at": "...", "source_index": 123}
  {...all original fields..., "status": "skipped", "corrected": false, ...}

The input JSONL is never modified. Skipped samples are still written out (with
`status == "skipped"`) so downstream code can filter them explicitly.

Crash safety
------------
Every Save / Skip appends immediately to disk (flush + fsync). The JSONL is an
append-only log keyed by the profile's `key_field`: if the same sample is
written more than once (e.g. you go back and re-correct it), the *last* record
for it wins on reload. Work is therefore never lost on a crash, and the key is
stable even if you change the filter or the shuffle between sessions — only the
*display order* changes, never which samples count as already done. Each
dataset has its own output file, so the two never mix.

Navigation model (design decision)
-----------------------------------
  - Previous / Next  : move by ±1 through *all* samples (for reviewing anything,
                       including already-annotated ones), without saving.
  - Save & Next      : save the current correction, then jump to the next
                       *remaining* (not-yet-done) sample.
  - Skip             : mark the sample skipped, then jump to the next remaining.
  - On startup       : resume at the first remaining sample.
Already-annotated samples are thus never surfaced by the default workflow, but
remain reachable via Previous/Next or Jump for review.

Dependencies: gradio, pillow, standard library only.
"""

# ----------------------------------------------------------------------------
# Configuration — the only values a user should need to edit.
# ----------------------------------------------------------------------------
ROOT = "/Users/xai/Personal/Projects/TeluguOCR/data"

# Which dataset to annotate. Everything else follows from the profile below.
DATASET = "lines_eval"        # "lines_eval" | "benchmark"

PROFILES = {
    # Single text lines, 64px tall, with a model CER per line.
    "lines_eval": {
        "data_dir": f"{ROOT}/wikisource_lines_eval",
        "input_jsonl": "lines.jsonl",
        "text_field": "text",
        "image_field": "line_image",   # path relative to data_dir
        "key_field": "line_image",     # stable identity of a sample
        # Only samples with filter_min <= <filter_field> <= filter_max are shown.
        # filter_field=None shows everything; either bound may be None.
        "filter_field": "line_cer",
        "filter_min": 0.0,
        "filter_max": 1.0,
        # Fields echoed into the metadata line above the image.
        "meta_fields": ["line_cer", "accepted", "n_graphemes"],
        "instruction": "Correct the OCR prediction in the textbox",
        # Viewer: lines are short and wide, so upscale hard and scroll sideways.
        "view_mode": "Zoom",
        "zoom": 4.0,
        "max_view_height": None,       # no cap: a zoomed line is only ~256px tall
        "edit_lines": 6,
        "unit": "line",
    },
    # Full page scans (~1280x1900) with the proofread text for the whole page.
    "benchmark": {
        "data_dir": f"{ROOT}/wikisource_benchmark",
        "input_jsonl": "pages.jsonl",
        "text_field": "text",
        "image_field": "image",
        "key_field": "slug",           # `image` would work too; slug is shorter
        "filter_field": "n_chars",
        "filter_min": None,
        "filter_max": None,
        "meta_fields": ["page_no", "n_chars", "source_file"],
        "instruction": "Correct the proofread text against the scan",
        # Viewer: a whole page, so fit it and cap the height — otherwise the box
        # is taller than the screen and the edit box is pushed out of sight.
        "view_mode": "Fit to width",
        "zoom": 1.0,
        "max_view_height": "78vh",
        "edit_lines": 24,              # page text is ~1000 chars, not one line
        "unit": "page",
    },
}

# Present samples in random order instead of file order. Useful for getting an
# unbiased sample across books when you won't annotate the whole set (file
# order walks one book at a time).
SHUFFLE = True
# Seed for that shuffle. A fixed int gives the *same* order every launch, so
# "Sample index" stays meaningful across sessions. Set to None for a different
# order each launch — resuming still works (annotations are keyed by
# `key_field`, not position), but indices no longer refer to the same sample.
SHUFFLE_SEED = 42

# ----------------------------------------------------------------------------

import base64
import io
import json
import os
import random
from datetime import datetime, timezone

import gradio as gr
from PIL import Image

if DATASET not in PROFILES:
    raise SystemExit(f"DATASET must be one of {list(PROFILES)}, got {DATASET!r}")
P = PROFILES[DATASET]

DATA_DIR = P["data_dir"]
TEXT_FIELD = P["text_field"]
IMAGE_FIELD = P["image_field"]
KEY_FIELD = P["key_field"]
FILTER_FIELD = P["filter_field"]
FILTER_MIN = P["filter_min"]
FILTER_MAX = P["filter_max"]
UNIT = P["unit"]

# Display defaults for the image viewer.
DEFAULT_VIEW_MODE = P["view_mode"]
DEFAULT_ZOOM = P["zoom"]
MAX_VIEW_HEIGHT = P["max_view_height"]
FIT_MAX_WIDTH = 1600        # cap on the rendered width in "Fit to width" mode
STATUS_COMPLETED = "completed"
STATUS_SKIPPED = "skipped"

LINES_PATH = os.path.join(DATA_DIR, P["input_jsonl"])
# Output always sits next to its input, named "<input stem>_corrected.jsonl".
# Derived rather than configured so a new profile cannot accidentally point at
# another dataset's file: "pages.jsonl" -> "pages_corrected.jsonl".
OUTPUT_JSONL = os.path.splitext(P["input_jsonl"])[0] + "_corrected.jsonl"
OUTPUT_PATH = os.path.join(DATA_DIR, OUTPUT_JSONL)


# ============================================================================
# Annotation store: append-only JSONL log with last-write-wins semantics,
# keyed by the profile's `key_field` (stable across changes to the filter).
# ============================================================================
class AnnotationStore:
    def __init__(self, path):
        self.path = path
        self.records = {}  # key_field value -> latest record dict
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a partially-written trailing line
                key = rec.get(KEY_FIELD)
                if key is not None:
                    self.records[key] = rec

    def _append(self, rec):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.records[rec[KEY_FIELD]] = rec

    def _base(self, source_rec, status):
        """Full copy of the source record + annotation bookkeeping."""
        rec = dict(source_rec)
        rec.pop("_source_index", None)
        rec["source_index"] = source_rec["_source_index"]
        rec["status"] = status
        rec["annotated_at"] = datetime.now(timezone.utc).isoformat()
        return rec

    def save_completed(self, source_rec, ground_truth):
        original = str(source_rec.get(TEXT_FIELD, ""))
        rec = self._base(source_rec, STATUS_COMPLETED)
        rec[TEXT_FIELD] = ground_truth
        rec["original_text"] = original
        rec["corrected"] = (ground_truth != original)
        self._append(rec)

    def save_skipped(self, source_rec):
        rec = self._base(source_rec, STATUS_SKIPPED)
        rec["original_text"] = str(source_rec.get(TEXT_FIELD, ""))
        rec["corrected"] = False
        self._append(rec)

    # --- lookups -----------------------------------------------------------
    def status_of(self, key):
        rec = self.records.get(key)
        return rec.get("status") if rec else None

    def ground_truth_of(self, key):
        rec = self.records.get(key)
        if rec and rec.get("status") == STATUS_COMPLETED:
            return rec.get(TEXT_FIELD)
        return None

    def is_done(self, key):
        return self.status_of(key) in (STATUS_COMPLETED, STATUS_SKIPPED)

    def counts(self, keys):
        """Counts restricted to the currently-visible working set."""
        completed = sum(1 for k in keys
                        if self.status_of(k) == STATUS_COMPLETED)
        skipped = sum(1 for k in keys if self.status_of(k) == STATUS_SKIPPED)
        return completed, skipped


# ============================================================================
# Dataset access — records held in memory (JSON only), images decoded lazily.
# ============================================================================
def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec["_source_index"] = i  # position in lines.jsonl, filter-independent
            records.append(rec)
    return records


def in_filter_range(rec):
    """Either bound may be None (unbounded); FILTER_FIELD=None shows everything."""
    if FILTER_FIELD is None:
        return True
    value = rec.get(FILTER_FIELD)
    if value is None:
        return True  # never hide a sample just because the metric is missing
    value = float(value)
    if FILTER_MIN is not None and value < FILTER_MIN:
        return False
    if FILTER_MAX is not None and value > FILTER_MAX:
        return False
    return True


def filter_label():
    if FILTER_FIELD is None or (FILTER_MIN is None and FILTER_MAX is None):
        return "no filter"
    lo = "-∞" if FILTER_MIN is None else FILTER_MIN
    hi = "∞" if FILTER_MAX is None else FILTER_MAX
    return f"{FILTER_FIELD} ∈ [{lo}, {hi}]"


def order_label():
    if not SHUFFLE:
        return "file order"
    return (f"shuffled (seed {SHUFFLE_SEED})" if SHUFFLE_SEED is not None
            else "shuffled (unseeded — order differs every launch)")


print(f"Loading '{LINES_PATH}' ...")
ALL_RECORDS = load_records(LINES_PATH)
RECORDS = [r for r in ALL_RECORDS if in_filter_range(r)]
if SHUFFLE:
    # Shuffle a local Random so the app never touches global random state, and
    # so a fixed seed reproduces the same order on the next launch.
    random.Random(SHUFFLE_SEED).shuffle(RECORDS)
KEYS = [r[KEY_FIELD] for r in RECORDS]
TOTAL = len(RECORDS)
print(f"[{DATASET}] loaded {len(ALL_RECORDS)} {UNIT}s; {TOTAL} within "
      f"{filter_label()}; {order_label()}.")

STORE = AnnotationStore(OUTPUT_PATH)
print(f"Annotations -> {OUTPUT_PATH} "
      f"({len(STORE.records)} previously written).")


def get_record(index):
    return RECORDS[int(index)]


def get_key(index):
    return KEYS[int(index)]


def get_prediction(index):
    return str(get_record(index).get(TEXT_FIELD, ""))


def get_image_path(index):
    return os.path.join(DATA_DIR, get_record(index)[IMAGE_FIELD])


def first_remaining():
    for i in range(TOTAL):
        if not STORE.is_done(KEYS[i]):
            return i
    return 0  # everything done -> land on the first sample


def clamp(index):
    if TOTAL == 0:
        return 0
    return max(0, min(int(index), TOTAL - 1))


def next_remaining_from(index):
    """First not-done sample strictly after `index`; falls back to next index."""
    for i in range(int(index) + 1, TOTAL):
        if not STORE.is_done(KEYS[i]):
            return i
    # nothing remaining ahead -> just advance one (clamped) so we don't get stuck
    return clamp(int(index) + 1)


# ============================================================================
# Rendering helpers.
# ============================================================================
def render_image_html(index, mode, zoom):
    """The scan, in a scrollable box sized for the profile.

    A `gr.Image` crops a zoomed image instead of letting you pan it, which is
    useless for reading diacritics on a 64px-tall, ~1000px-wide line. So the
    file is embedded as a data URI and placed in a scrolling box: zoom stays
    readable and the whole image stays reachable.

    The two profiles need different boxes. A zoomed line is only ~256px tall, so
    it scrolls sideways and must NOT scroll vertically (a stray wheel event
    would hide the text). A page scan is ~1900px tall, so it needs both axes and
    a max-height — uncapped, the box is taller than the viewport and pushes the
    edit box off screen, which makes correcting while reading impossible.
    """
    if not TOTAL:
        return ""
    img = Image.open(get_image_path(index)).convert("RGB")
    w, h = img.size

    if mode == "Fit to width":
        # Shrink to the column via CSS — the container width isn't knowable
        # here, so a pixel size would either overflow or waste space.
        scale = min(1.0, FIT_MAX_WIDTH / w) if w else 1.0
        style = "max-width:100%;height:auto;"
    else:  # "Zoom" — explicit pixel size; the box scrolls horizontally.
        scale = max(0.1, float(zoom))
        style = "max-width:none;"
    # Resample here rather than letting the browser scale: LANCZOS keeps the
    # thin Telugu diacritics legible where bilinear upscaling smears them.
    if scale != 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                         Image.LANCZOS)

    buf = io.BytesIO()
    # JPEG for big page scans: a 1280x1900 PNG data URI is ~4MB of base64 per
    # navigation, which makes every click feel slow. Lines stay PNG (lossless,
    # and tiny anyway) so no compression artefact is ever mistaken for the scan.
    if MAX_VIEW_HEIGHT is None:
        img.save(buf, format="PNG")
        mime = "image/png"
    else:
        img.save(buf, format="JPEG", quality=92, subsampling=0)
        mime = "image/jpeg"
    uri = f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"

    box = ("overflow-x:auto;overflow-y:hidden;" if MAX_VIEW_HEIGHT is None
           else f"overflow:auto;max-height:{MAX_VIEW_HEIGHT};")
    return (
        f'<div style="{box}background:#fff;'
        'border:1px solid rgba(128,128,128,0.35);border-radius:6px;padding:8px;">'
        f'<img src="{uri}" alt="scan" style="display:block;{style}" />'
        '</div>'
    )


def progress_html():
    completed, skipped = STORE.counts(KEYS)
    remaining = TOTAL - completed - skipped
    pct = (completed / TOTAL * 100.0) if TOTAL else 0.0
    bar = f"""
    <div style="font-family:system-ui,sans-serif;">
      <div style="display:flex;gap:16px;flex-wrap:wrap;font-size:14px;margin-bottom:6px;">
        <span><b>Total:</b> {TOTAL}</span>
        <span style="color:#16a34a;"><b>Completed:</b> {completed}</span>
        <span style="color:#6b7280;"><b>Skipped:</b> {skipped}</span>
        <span style="color:#2563eb;"><b>Remaining:</b> {remaining}</span>
        <span><b>{pct:.2f}%</b> complete</span>
      </div>
      <div style="background:rgba(128,128,128,0.25);border-radius:6px;height:12px;width:100%;overflow:hidden;">
        <div style="background:#16a34a;height:100%;width:{pct:.3f}%;transition:width .2s;"></div>
      </div>
    </div>
    """
    return bar


def format_meta_field(rec, field):
    """One metadata chip, formatted per field. None when the field is absent."""
    value = rec.get(field)
    if value is None:
        return None
    if field == "accepted":
        return "accepted" if value else f"rejected ({rec.get('reject_reason')})"
    if field == "line_cer":
        return f"line_cer **{float(value):.4f}**"
    if field == "n_graphemes":
        return f"{value} graphemes"
    if field == "n_chars":
        return f"{value} chars"
    if field == "page_no":
        return f"page {value}"
    return f"{field} {value}"


def sample_meta_line(index):
    """Provenance + quality metadata for the sample, shown above the image."""
    rec = get_record(index)
    bits = [f"`{rec[IMAGE_FIELD]}`"]
    bits += [chip for chip in
             (format_meta_field(rec, f) for f in P["meta_fields"])
             if chip is not None]
    bits.append(f"source line {rec['_source_index']}")
    return " · ".join(bits)


def sample_status_line(index):
    st = STORE.status_of(get_key(index))
    if st == STATUS_COMPLETED:
        return f"● Sample {index} — already **completed** (editing will overwrite)."
    if st == STATUS_SKIPPED:
        return f"○ Sample {index} — previously **skipped**."
    return f"◇ Sample {index} — not annotated yet."


def render(index, mode, zoom, status_msg=None):
    """Return the full set of UI values for `index`."""
    index = clamp(index)
    prediction = get_prediction(index) if TOTAL else ""
    existing_gt = STORE.ground_truth_of(get_key(index)) if TOTAL else None
    edit_value = existing_gt if existing_gt is not None else prediction

    if status_msg is None:
        status_msg = (sample_status_line(index) if TOTAL
                      else "No samples match the filter.")

    return (
        render_image_html(index, mode, zoom),                # image_view
        sample_meta_line(index) if TOTAL else "",            # meta_bar
        edit_value,                                          # edit_box
        index,                                               # index_box
        progress_html(),                                     # progress
        status_msg,                                          # status_bar
        index,                                               # cur_state
    )


# ============================================================================
# Event callbacks.
# ============================================================================
def on_prev(cur, mode, zoom):
    new = clamp(int(cur) - 1)
    return render(new, mode, zoom)


def on_next(cur, mode, zoom):
    new = clamp(int(cur) + 1)
    return render(new, mode, zoom)


def on_save(cur, text, mode, zoom):
    """Save current correction without navigating (Ctrl+S)."""
    idx = clamp(int(cur))
    STORE.save_completed(get_record(idx), text or "")
    return render(idx, mode, zoom, status_msg=f"✔ Saved sample {idx}.")


def on_save_next(cur, text, mode, zoom):
    idx = clamp(int(cur))
    STORE.save_completed(get_record(idx), text or "")
    nxt = next_remaining_from(idx)
    msg = f"✔ Saved sample {idx} → now at {nxt}."
    return render(nxt, mode, zoom, status_msg=msg)


def on_skip(cur, mode, zoom):
    idx = clamp(int(cur))
    STORE.save_skipped(get_record(idx))
    nxt = next_remaining_from(idx)
    return render(nxt, mode, zoom, status_msg=f"⊘ Skipped {idx} → now at {nxt}.")


def on_jump(target, mode, zoom):
    if target is None:
        return render(0, mode, zoom, status_msg="Enter an index to jump to.")
    idx = clamp(int(target))
    return render(idx, mode, zoom, status_msg=f"Jumped to sample {idx}.")


def on_display_change(cur, mode, zoom):
    """Re-render only the image when the viewer settings change."""
    return render_image_html(clamp(int(cur)), mode, zoom)


# ============================================================================
# Keyboard shortcuts (injected on page load).
#   Ctrl+Enter -> Save & Next   Ctrl+Right -> Next   Ctrl+Left -> Previous
#   Ctrl+S     -> Save          Ctrl+K     -> focus Jump box
# ============================================================================
# `launch(js=...)` *executes* this string on page load (it does not call it),
# so it has to be self-invoking.
SHORTCUTS_JS = """
(() => {
  if (window.__ocr_shortcuts_installed) return;
  window.__ocr_shortcuts_installed = true;
  const click = (id) => {
    const el = document.getElementById(id);
    if (!el) return;
    const btn = el.querySelector('button') || el;
    btn.click();
  };
  document.addEventListener('keydown', (e) => {
    if (!e.ctrlKey) return;
    const k = e.key;
    if (k === 'Enter') { e.preventDefault(); click('save_next_btn'); }
    else if (k === 'ArrowRight') { e.preventDefault(); click('next_btn'); }
    else if (k === 'ArrowLeft') { e.preventDefault(); click('prev_btn'); }
    else if (k === 's' || k === 'S') { e.preventDefault(); click('save_btn'); }
    else if (k === 'k' || k === 'K') {
      e.preventDefault();
      const el = document.getElementById('index_box');
      if (el) { const inp = el.querySelector('input'); if (inp) { inp.focus(); inp.select(); } }
    }
  }, true);
})()
"""


# ============================================================================
# UI.
# ============================================================================
def build_ui():
    start_idx = first_remaining()

    # Gradio >=6: `js` belongs on launch(), not on the Blocks constructor.
    with gr.Blocks(title="Telugu OCR Correction Tool") as demo:
        gr.Markdown(
            "# Telugu OCR Correction Tool\n"
            f"`{P['input_jsonl']}` → `{OUTPUT_JSONL}` in `{DATA_DIR}` — "
            f"showing {TOTAL} {UNIT}s ({filter_label()}), in "
            f"{order_label()}.\n\n"
            f"{P['instruction']}, then **Save & Next** "
            "(`Ctrl+Enter`). Shortcuts: `Ctrl+←/→` prev/next, `Ctrl+S` save, "
            "`Ctrl+K` jump."
        )

        cur_state = gr.State(start_idx)

        progress = gr.HTML(progress_html())

        with gr.Row():
            with gr.Column(scale=3):
                meta_bar = gr.Markdown(
                    sample_meta_line(start_idx) if TOTAL else "")
                image_view = gr.HTML(
                    render_image_html(start_idx, DEFAULT_VIEW_MODE, DEFAULT_ZOOM),
                    label="Line image",
                    container=False,
                )
                with gr.Row():
                    display_mode = gr.Radio(
                        choices=["Zoom", "Fit to width"],
                        value=DEFAULT_VIEW_MODE,
                        label="View mode",
                    )
                    zoom = gr.Slider(
                        minimum=0.5, maximum=10.0, step=0.5, value=DEFAULT_ZOOM,
                        label="Zoom (×)",
                    )
            with gr.Column(scale=2):
                # No read-only copy of the source text: this box is pre-filled
                # with it. The untouched original is still recorded as
                # `original_text` on every saved row.
                edit_box = gr.Textbox(
                    label="Ground truth — edit me",
                    interactive=True,
                    lines=P["edit_lines"],
                    autofocus=True,
                    elem_id="edit_box",
                )

        with gr.Row():
            prev_btn = gr.Button("◀ Previous", elem_id="prev_btn")
            next_btn = gr.Button("Next ▶", elem_id="next_btn")
            save_btn = gr.Button("Save", elem_id="save_btn")
            save_next_btn = gr.Button(
                "💾 Save & Next", variant="primary", elem_id="save_next_btn")
            skip_btn = gr.Button("Skip", variant="stop", elem_id="skip_btn")

        with gr.Row():
            index_box = gr.Number(
                label=f"{UNIT.capitalize()} index (0 – {max(0, TOTAL - 1)})",
                value=start_idx, precision=0, elem_id="index_box",
            )
            jump_btn = gr.Button("Jump to index", elem_id="jump_btn")

        status_bar = gr.Markdown(
            sample_status_line(start_idx) if TOTAL else
            "No samples match the filter.")

        # Outputs shared by all navigation callbacks (order matters).
        outs = [image_view, meta_bar, edit_box, index_box,
                progress, status_bar, cur_state]

        prev_btn.click(on_prev, [cur_state, display_mode, zoom], outs)
        next_btn.click(on_next, [cur_state, display_mode, zoom], outs)
        save_btn.click(on_save, [cur_state, edit_box, display_mode, zoom], outs)
        save_next_btn.click(
            on_save_next, [cur_state, edit_box, display_mode, zoom], outs)
        skip_btn.click(on_skip, [cur_state, display_mode, zoom], outs)
        jump_btn.click(on_jump, [index_box, display_mode, zoom], outs)
        index_box.submit(on_jump, [index_box, display_mode, zoom], outs)

        # Image-only refresh when the viewer controls change.
        display_mode.change(
            on_display_change, [cur_state, display_mode, zoom], image_view)
        zoom.change(
            on_display_change, [cur_state, display_mode, zoom], image_view)

        # Populate the first sample on load.
        demo.load(
            lambda: render(start_idx, DEFAULT_VIEW_MODE, DEFAULT_ZOOM),
            None, outs,
        )

    return demo


if __name__ == "__main__":
    build_ui().queue().launch(js=SHORTCUTS_JS)
