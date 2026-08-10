"""Measure what the aligner would yield, and at which CER cut, on a sample of pages.

This started as a segmenter x engine grid and has been cut down to its one lasting
job. The comparison it was built for is settled -- tesseract layout analysis beat
PP-OCR detection and an ink-projection segmenter, and the repo's own CTC encoder beat
Tesseract and PaddleOCR as the recogniser (see segmentation.py for the table) -- so
sweeping those axes again is only worth doing after the recogniser is retrained, and
git history has the full grid for that.

What still needs measuring, every time the recogniser changes, is WHERE TO PUT
`max_line_cer`. That is the single knob deciding how much of the corpus survives, and
the right value moves with how well the model reads. So this reports:

    corpus CER  total grapheme edit distance over total ground-truth graphemes, for
                all of a page's line hypotheses joined and scored against the whole
                page text. Raw reading accuracy, blind to where the line boundaries
                fall. Aggregated over the corpus rather than averaged per page, so a
                12-grapheme title page cannot count as much as a 900-grapheme page of
                prose.

    yield@X     the fraction of detected lines whose own hypothesis agrees with the
                ground-truth span the aligner gave it, to within X. This is the size
                of the dataset you would get at that cut, and unlike corpus CER it
                punishes bad segmentation directly -- a merged band is handed a span
                twice the right size and fails.

    struct%     lines rejected before any CER test at all (empty or too-short span,
                runaway span, mojibake script mismatch). A property of the
                segmentation and the transcript, not of the threshold.

Expect corpus CER to look worse than the yields suggest. A minority of pages are
simply unusable -- a transcript that does not correspond to the scan, a ruined print,
a plate -- and they drag the aggregate while most lines on good pages align tightly.
That gap is the argument for keeping the page-level guards on, not for loosening the
line-level cut.

The alignment runs once per page with the CER filter DISABLED and the cuts applied
afterwards to the recorded per-line CERs, so one pass produces the whole curve. The
threshold-independent rejections stay on, since leaving them off would inflate every
number by the same irrelevant amount.

Results stream to RESULTS_JSONL per page and are read back on a re-run, so this is
interruptible and resumable.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from rapidfuzz.distance import Levenshtein
from tqdm.auto import tqdm

from data_curation.pseudo_labelling.consensus_labelling import graphemes, normalize
from data_curation.wikisource.alignment import AcceptPolicy, align_page
from data_curation.wikisource.segmentation import (
    BoxFilter,
    TesseractLayoutSegmenter,
    prepare_page,
)

# Anything looser than ~0.4 is not usable as unreviewed ground truth; the loose cuts
# are reported only to show where a run's errors actually sit.
CER_CUTS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)


@dataclass
class Page:
    slug: str
    image_path: str
    text: str


def load_pages(data_dir, limit=None) -> list[Page]:
    """Read <data_dir>/images/<slug>.jpg + <data_dir>/text/<slug>.txt pairs."""
    data_dir = Path(data_dir)
    img_dir, txt_dir = data_dir / "images", data_dir / "text"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"{img_dir} does not exist; run scrape.py first")

    pages = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        txt_path = txt_dir / f"{img_path.stem}.txt"
        if not txt_path.exists():
            continue
        text = txt_path.read_text(encoding="utf-8").strip()
        if text:
            pages.append(Page(img_path.stem, str(img_path), text))
    return pages[:limit] if limit else pages


def build_engine(checkpoint: str, vocab_file: str, **kwargs):
    """Load the CTC recogniser used to anchor the alignment.

    consensus_labelling picks the model module by COUNTING position rows in the
    checkpoint, so a stage-3 checkpoint that stores its table merged into a single
    `pos_embed` of (1, 256, 384) is correctly reported as the 2048px variant. An earlier
    version sniffed for a separate `pos_embed_ext` tensor and announced "CTC variant
    '1024'" for exactly that checkpoint -- harmless then only because both trainer
    modules declare `max_image_width = 2048`. If you see that label on a 2048px
    checkpoint, you are running the old code.
    """
    from data_curation.pseudo_labelling.consensus_labelling import build_model_engine

    return build_model_engine(checkpoint=checkpoint, vocab_file=vocab_file, **kwargs)


def _run_engine(engine, crops) -> list[str | None]:
    """Call the engine and normalize its return shape to a plain list of texts."""
    if not crops:
        return []
    out = engine.run(crops)
    if isinstance(out, tuple):  # engines with per-line confidence return (texts, scores)
        out = out[0]
    return list(out)


def _load_done(path) -> dict:
    done = {}
    if not Path(path).exists():
        return done
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:  # a torn last line from a hard kill
                continue
            done[rec["slug"]] = rec
    return done


def _append(path, rec) -> None:
    """Append one page and fsync, so a kill costs at most the page in flight."""
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def run(pages, segmenter, engine, results_jsonl, policy=None, crop_pad=3):
    """Segment, read and align every page, streaming one record each to disk."""
    policy = policy or AcceptPolicy()
    scan_policy = AcceptPolicy(
        max_line_cer=float("inf"),          # the report sweeps the cuts itself
        min_graphemes=policy.min_graphemes,
        max_grapheme_ratio=policy.max_grapheme_ratio,
        reject_script_mismatch=policy.reject_script_mismatch,
        min_page_yield=0.0,
        max_page_cer=float("inf"),
    )

    done = _load_done(results_jsonl)
    if done:
        print(f"[resume] {len(done)} pages already measured in {results_jsonl}")
    todo = [p for p in pages if p.slug not in done]
    print(f"[run] {len(pages)} pages, {len(todo)} to measure")

    for page in tqdm(todo, desc="pages", unit="pg", smoothing=0.05):
        with Image.open(page.image_path) as img:
            prepared = prepare_page(img)

        t0 = time.time()
        boxes = segmenter.segment(prepared)
        seg_seconds = time.time() - t0

        crops = [b.crop(prepared, pad=crop_pad) for b in boxes]
        t0 = time.time()
        try:
            hypotheses = _run_engine(engine, crops)
        except Exception as exc:
            tqdm.write(f"[error] {page.slug}: {type(exc).__name__}: {exc}")
            hypotheses = [None] * len(crops)
        ocr_seconds = time.time() - t0

        result = align_page(hypotheses, page.text, scan_policy)
        gt = graphemes(normalize(page.text))
        hyp_all = graphemes(" ".join(normalize(h or "") for h in hypotheses))

        rec = {
            "slug": page.slug,
            "n_lines": len(boxes),
            "n_gt_graphemes": len(gt),
            "edit_distance": Levenshtein.distance(gt, hyp_all),
            "page_cer": result.page_cer,
            "seg_seconds": seg_seconds,
            "ocr_seconds": ocr_seconds,
            "lines": [
                {"cer": ln.cer, "n": ln.n_graphemes, "reject": ln.reject_reason}
                for ln in result.lines
            ],
        }
        _append(results_jsonl, rec)
        done[page.slug] = rec

    return list(done.values())


def summarize(records) -> dict:
    from collections import defaultdict

    total = {"pages": 0, "n_lines": 0, "gt_graphemes": 0, "edit_distance": 0,
             "seg_seconds": 0.0, "ocr_seconds": 0.0}
    structural = defaultdict(int)
    line_cers, line_lengths = [], []

    for rec in records:
        total["pages"] += 1
        total["n_lines"] += rec["n_lines"]
        total["gt_graphemes"] += rec["n_gt_graphemes"]
        total["edit_distance"] += rec["edit_distance"]
        total["seg_seconds"] += rec["seg_seconds"]
        total["ocr_seconds"] += rec["ocr_seconds"]
        for line in rec["lines"]:
            if line["reject"]:
                structural[line["reject"]] += 1
            else:
                line_cers.append(line["cer"])
                line_lengths.append(line["n"])

    n_lines = total["n_lines"] or 1
    yields, kept_graphemes = {}, {}
    for cut in CER_CUTS:
        kept = [n for c, n in zip(line_cers, line_lengths) if c <= cut]
        yields[cut] = len(kept) / n_lines
        kept_graphemes[cut] = sum(kept)

    return {
        "pages": total["pages"],
        "lines": total["n_lines"],
        "lines_per_page": total["n_lines"] / max(1, total["pages"]),
        "corpus_cer": total["edit_distance"] / max(1, total["gt_graphemes"]),
        "structural_reject": dict(structural),
        "structural_reject_frac": sum(structural.values()) / n_lines,
        "yield": yields,
        "kept_graphemes": kept_graphemes,
        "seconds_per_page": (total["seg_seconds"] + total["ocr_seconds"])
                            / max(1, total["pages"]),
    }


def print_report(summary, corpus_pages=None) -> None:
    print("\n" + "=" * 78)
    print(f"{summary['pages']} pages | {summary['lines']} detected lines "
          f"({summary['lines_per_page']:.1f}/page) | {summary['seconds_per_page']:.2f}s/page")
    print(f"corpus CER {summary['corpus_cer']:.3f}   "
          f"structural rejections {100 * summary['structural_reject_frac']:.1f}%")
    print("=" * 78)

    if summary["structural_reject"]:
        print("\nSTRUCTURAL REJECTIONS")
        for reason, count in sorted(summary["structural_reject"].items()):
            print(f"  {reason:<20}{count:>7}")

    print("\nYIELD BY CER CUT")
    header = f"  {'cut':>6}{'lines kept':>13}{'of detected':>13}{'graphemes':>12}"
    if corpus_pages:
        header += f"{'lines @ ' + str(corpus_pages) + ' pages':>24}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for cut in CER_CUTS:
        kept = int(round(summary["yield"][cut] * summary["lines"]))
        row = (f"  {cut:>6.2f}{kept:>13}{100 * summary['yield'][cut]:>12.1f}%"
               f"{summary['kept_graphemes'][cut]:>12}")
        if corpus_pages:
            row += f"{summary['lines_per_page'] * summary['yield'][cut] * corpus_pages:>24,.0f}"
        print(row)
    print("=" * 78)


if __name__ == "__main__":
    # ---- input ----
    DATA_DIR = "data/wikisource_sample"   # written by scrape.py with MODE = "sample"
    LIMIT_PAGES = None                    # int for a smoke test, None for the whole sample

    # ---- output ----
    RESULTS_JSONL = "data/wikisource_experiment/pages.jsonl"
    SUMMARY_JSON = "data/wikisource_experiment/summary.json"

    # ---- the chosen configuration ----
    VOCAB = "src/text_decoder/grapheme_tokenizer/telugu-vocab.json"
    CHECKPOINT = "models/image_encoder/ctc_encoder_stage-3/ctc-encoder-2048/final_model.pt"

    BOX_FILTER = BoxFilter(min_height=12, min_width=40, max_height_frac=0.25,
                           min_ink_frac=0.005, max_ink_frac=0.55)
    # max_line_cer is ignored here -- the report sweeps every cut in CER_CUTS.
    POLICY = AcceptPolicy(min_graphemes=5, max_grapheme_ratio=3.0,
                          reject_script_mismatch=True)

    # Only used to project the sample's yield onto the whole corpus, in the last column
    # of the report. NOT the category size: the validated-pages category had 53,714
    # members when last counted, but 208 of those have no scan behind them -- 207 whose
    # imageforpage record is empty because the source file is gone (every page of
    # "26 1981 krishna gazzette.pdf", for instance), plus one stray ".../14/బ్యాకప్"
    # subpage with no page number. Those are skipped at listing time, so the number of
    # pages this pipeline can actually process is the smaller one.
    CORPUS_PAGES = 53_506

    Path(RESULTS_JSONL).parent.mkdir(parents=True, exist_ok=True)

    pages = load_pages(DATA_DIR, limit=LIMIT_PAGES)
    print(f"[data] {len(pages)} pages from {DATA_DIR}")

    segmenter = TesseractLayoutSegmenter(box_filter=BOX_FILTER)
    engine = build_engine(CHECKPOINT, VOCAB)

    summary = summarize(run(pages, segmenter, engine, RESULTS_JSONL, policy=POLICY))
    Path(SUMMARY_JSON).write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
    print_report(summary, corpus_pages=CORPUS_PAGES)
    print(f"\nsummary written to {SUMMARY_JSON}")
