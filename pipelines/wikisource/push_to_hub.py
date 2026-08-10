"""Publish the Wikisource line datasets to the HuggingFace Hub.

Two datasets are built from the same pipeline and pushed by this one script; pick which
with `WHICH` in the __main__ config block.

    WHICH = "train"   the FULL line set built by build_local.py, machine-labelled
    WHICH = "eval"    the small HUMAN-VERIFIED gold set, for evaluation

WHY THEY ARE SEPARATE DATASETS
    The train set's `text` comes from an alignment anchored on our own CTC model, and its
    `accepted` flag means that model agreed with the alignment. Any benchmark scored
    against those labels hands our model an advantage no other engine gets. The eval set
    has no such circularity: a person compared the text to the image in
    annotation/ocr_correction_tool.py.

Both push crops that are already in encoder form (grayscale, height 64, width a multiple
of 8), so training and eval can consume them without re-preprocessing. Both need HF_TOKEN
(env var, .env, or Kaggle secret) -- see hf_login in scrape.py.

    python3 -m data_curation.wikisource.push_to_hub

======================================================================================
TRAIN  --  data/wikisource_lines/lines.jsonl  ->  telugu-wikisource-text-images
======================================================================================
Pushes ONE train split carrying every stored line together with its quality columns
(line_cer, accepted, reject_reason, ...). Nothing is filtered here on purpose: the
pipeline stores everything above a loose CER floor precisely so that strictness stays a
query at training time instead of being frozen into the dataset. Consumers cut it down
with ordinary dataset ops, e.g.

    ds = load_dataset(REPO, split="train")
    strict = ds.filter(lambda r: r["accepted"])                 # the strict policy
    loose  = ds.filter(lambda r: r["line_cer"] <= 0.15)         # a custom cut

Re-running after more pages have been built re-uploads the whole split and replaces the
previous version on the Hub -- push it when a build milestone is reached, not per run.
The upload embeds the JPEGs into parquet shards (max_shard_size each), so expect roughly
the size of <data_dir>/lines on the wire.

======================================================================================
EVAL  --  data/wikisource_lines_eval/lines_corrected.jsonl  ->  telugu-line-ocr-bench
======================================================================================
Only the lines whose transcription a human read against the scan and signed off.

WHAT IS FILTERED (report printed at the end)
    1. duplicates      lines_corrected.jsonl is an append-only log, so a line corrected
                       twice appears twice. Deduped on `line_image`, last write wins.
    2. not completed   `status == "skipped"` rows are dropped: skipped means the
                       annotator did NOT vouch for the text.
    3. empty text      dropped after stripping whitespace.
    4. missing/bad     the crop must exist and open, and be non-degenerate (both
       images          dimensions > 0).

TWO COLUMNS THAT WOULD MISLEAD, SO THEY ARE NOT PUBLISHED AS-IS
    * `line_cer` in the source rows is the model's error against the OLD machine label,
      measured before the human touched it -- not against `text`. It is read here (it
      records HOW each line came to be annotated: it is the field the annotation tool
      filtered on) but kept out of the published schema so nobody reads it as a
      label-quality score for this dataset.
    * `n_graphemes` in the source rows likewise counts the pre-correction text. It IS
      published, but RECOMPUTED from the final `text` with the same `regex.\\X` split the
      model's tokenizer uses.

SCRIPT FILTER
    `has_english` is derived from `text`: true when it contains any Latin-script letter.
    Digits and punctuation do NOT set it -- they are Unicode Common, and Telugu lines
    carry Arabic numerals often enough ("1509 సం॥", "6-1-1949 లో") that counting them as
    English would throw away good data.

    With DROP_LATIN (the default) only Latin-free lines are published, so the set is
    Telugu plus numerals and punctuation. Turn it off to publish everything -- but note
    `has_english` is not in the published schema, so consumers cannot re-make the cut
    themselves; add it to EVAL_FEATURES if you turn the filter off.

    This does drop lines that are mostly Telugu but carry a Latin fragment -- inline
    glosses like "... (Sexual Science) ..." -- and one where a Latin 'o' was typed for a
    Telugu sunna. Six such lines exist today.

SAMPLING BIAS -- READ BEFORE REPORTING NUMBERS ON THIS SET
    These lines were served to the annotator through a `line_cer` band, so they are
    enriched for lines our model got WRONG and are not a uniform sample of the corpus
    (the corpus is 56% `line_cer == 0`; this set is far harder). Absolute error rates
    measured here are pessimistic; comparisons BETWEEN engines on identical lines are the
    sound use. The generated dataset card says so too.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import regex
from datasets import Dataset, Features, Value
from datasets import Image as HFImage
from PIL import Image

from data_curation.wikisource.scrape import hf_login

REPO_ROOT = Path(__file__).resolve().parents[2]

_GRAPHEME = regex.compile(r"\X")
# Script membership, not character ranges: \p{Latin} and \p{Telugu} match letters of
# those scripts wherever they live in Unicode, and both ignore digits and punctuation,
# which are script-neutral (Unicode "Common").
_LATIN = regex.compile(r"\p{Latin}")


def n_graphemes(text: str) -> int:
    return len(_GRAPHEME.findall(str(text)))


def build_dataset(rows: list[dict], features: Features) -> Dataset:
    """Project `rows` onto `features` -- columns absent from a row become None."""
    columns = {name: [rec.get(name) for rec in rows] for name in features}
    return Dataset.from_dict(columns, features=features)


# ======================================================================================
# TRAIN: every stored line with its machine labels, unfiltered
# ======================================================================================
TRAIN_FEATURES = Features(
    {
        "image": HFImage(),
        "text": Value("string"),
        "line_cer": Value("float32"),
        "page_cer": Value("float32"),
        "page_yield": Value("float32"),
        "accepted": Value("bool"),
        "reject_reason": Value("string"),
        "n_graphemes": Value("int32"),
        "digits_converted": Value("bool"),
        "starts_mid_word": Value("bool"),
        "ends_mid_word": Value("bool"),
        "slug": Value("string"),
        "page_no": Value("int32"),
        "source_file": Value("string"),
        "line_no": Value("int32"),
    }
)


def load_train_rows(data_dir: Path) -> tuple[list[dict], int]:
    """All rows of lines.jsonl whose crop exists on disk, image paths made absolute.

    A missing crop should not happen -- build_local writes the JPEG before the JSONL
    row -- so any skips reported here mean files were deleted after the build.
    """
    rows, missing = [], 0
    with open(data_dir / "lines.jsonl", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:  # a torn last line from a hard kill
                continue
            img_path = data_dir / rec["line_image"]
            if not img_path.exists():
                missing += 1
                continue
            rec["image"] = str(img_path)
            rec.pop("checkpoint", None)
            rows.append(rec)
    return rows, missing


def report_train(rows: list[dict], missing: int, data_dir: Path) -> None:
    if missing:
        print(f"[warn] {missing} rows skipped: crop missing on disk")
    accepted = sum(r["accepted"] for r in rows)
    print(f"[data] {len(rows):,} lines ({accepted:,} strict-accepted, "
          f"{100 * accepted / len(rows):.1f}%) from {data_dir}")


# ======================================================================================
# EVAL: human-verified gold lines only
# ======================================================================================
# Deliberately NOT published, though load_eval_rows reads them: original_text, corrected,
# has_english, page_no, line_no, starts_mid_word, ends_mid_word, digits_converted,
# model_cer_before_correction (the source `line_cer`), accepted, reject_reason,
# checkpoint. The first three drive the filtering and the report below; the rest are the
# machine pipeline's PRE-correction verdicts and say nothing about the gold text.
EVAL_FEATURES = Features(
    {
        "image": HFImage(),
        "text": Value("string"),            # gold: human-verified transcription
        "n_graphemes": Value("int32"),      # recomputed from `text`
        "line_image": Value("string"),      # stable id, relative path in the source dir
        "slug": Value("string"),
        "source_file": Value("string"),
    }
)


def has_english(text: str) -> bool:
    """True when the line contains any Latin-script letter.

    Digits do NOT count. Telugu lines routinely carry Arabic numerals for years and
    verse numbers ("1509 సం॥", "6-1-1949 లో") -- 75 of the 1101 gold lines do -- and
    calling those English would throw away good Telugu data. Neither does punctuation:
    ".,()-" are Unicode Common, shared by both scripts.

    Named `has_english` because that is what it is for, but what it detects is the
    Latin SCRIPT. A stray Latin letter inside otherwise-Telugu text trips it, which is
    usually a transcription slip worth seeing: 'దేవిసతెo చూడండీ' has a Latin 'o' where
    a Telugu sunna (ం) belongs.
    """
    return bool(_LATIN.search(str(text)))


def load_eval_rows(data_dir: Path, labels_name: str = "lines_corrected.jsonl"):
    """Deduped, completed, non-empty rows whose crop is present and readable."""
    labels_file = data_dir / labels_name
    latest: dict[str, dict] = {}
    total = 0
    with open(labels_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue                      # tolerate a partial trailing line
            total += 1
            key = rec.get("line_image")
            if key is not None:
                latest[key] = rec             # last write wins

    dropped = Counter()
    dropped["duplicate (same line_image)"] = total - len(latest)

    rows = []
    for rec in latest.values():
        if rec.get("status") != "completed":
            dropped["not completed (skipped)"] += 1
            continue
        text = str(rec.get("text", "")).strip()
        if not text:
            dropped["empty text"] += 1
            continue

        path = data_dir / rec["line_image"]
        if not path.exists():
            dropped["image missing on disk"] += 1
            continue
        try:
            with Image.open(path) as im:
                width, height = im.size
                im.verify()
        except Exception:
            dropped["image unreadable"] += 1
            continue
        if width <= 0 or height <= 0:
            dropped["degenerate image"] += 1
            continue

        rows.append({
            "image": str(path),
            "text": text,
            "original_text": str(rec.get("original_text", "")),
            "corrected": bool(rec.get("corrected", False)),
            "n_graphemes": n_graphemes(text),
            "has_english": has_english(text),
            "line_image": rec["line_image"],
            "slug": rec.get("slug"),
            "source_file": rec.get("source_file"),
            "page_no": rec.get("page_no"),
            "line_no": rec.get("line_no"),
            "starts_mid_word": rec.get("starts_mid_word"),
            "ends_mid_word": rec.get("ends_mid_word"),
            "digits_converted": rec.get("digits_converted"),
            "model_cer_before_correction": rec.get("line_cer"),
            "accepted": rec.get("accepted"),
            "reject_reason": rec.get("reject_reason"),
            "checkpoint": rec.get("checkpoint"),
        })

    rows.sort(key=lambda r: r["line_image"])   # stable order across runs
    return rows, total, dropped


def drop_latin(rows: list[dict]) -> list[dict]:
    """Keep every line free of Latin letters.

    Digits and punctuation are fine and stay -- only the Latin alphabet disqualifies a
    line, so numerals ('1903') and punctuation-only lines are kept deliberately.

    Note this also drops lines that are mostly Telugu but contain a Latin fragment: an
    inline gloss like '... (Sexual Science) ...', and one transcription slip where a
    Latin 'o' stands in for a Telugu sunna.
    """
    kept = [r for r in rows if not r["has_english"]]
    print(f"[lang] kept {len(kept)} of {len(rows)} Latin-free lines; "
          f"dropped {len(rows) - len(kept)} containing Latin letters")
    return kept


def report_eval(rows, total, dropped, data_dir) -> None:
    print(f"[data] {data_dir}")
    print(f"[data] {total} log rows -> {len(rows)} published lines")
    for reason, count in dropped.most_common():
        if count:
            print(f"[drop] {count:>5}  {reason}")
    if not rows:
        return
    changed = sum(r["corrected"] for r in rows)
    graphemes = sorted(r["n_graphemes"] for r in rows)
    books = len({r["source_file"] for r in rows})
    print(f"[data] {changed} of {len(rows)} were edited by the annotator "
          f"({100 * changed / len(rows):.1f}%); the rest were verified unchanged")
    print(f"[data] {books} distinct books; graphemes/line "
          f"min {graphemes[0]}, median {graphemes[len(graphemes) // 2]}, "
          f"max {graphemes[-1]}, total {sum(graphemes):,}")
    eng = sum(r["has_english"] for r in rows)
    print(f"[lang] has_english {eng} of {len(rows)} "
          f"({100 * eng / len(rows):.1f}%); Latin-free {len(rows) - eng}")


def eval_dataset_card(rows) -> str:
    """A card that states the provenance and the sampling bias up front.

    Written because both are easy to get wrong from the columns alone: `text` looks
    like an ordinary label, and nothing in the data hints that the lines were selected
    by model error rate.
    """
    return f"""---
license: cc-by-sa-4.0
task_categories:
- image-to-text
language:
- te
tags:
- ocr
- telugu
- wikisource
size_categories:
- n<1K
---

# Telugu Wikisource OCR — human-verified line crops

{len(rows)} single-line crops from Telugu Wikisource page scans, each with a
transcription **checked against the image by a human**. Grayscale, height 64px,
width a multiple of 8 — the form the encoder consumes.

## Columns

| column | meaning |
|---|---|
| `image` | the line crop |
| `text` | **gold** transcription, human-verified |
| `n_graphemes` | akshara count of `text` (`regex.\\X`) |
| `line_image` | stable id: the crop's relative path in the source build |
| `slug`, `source_file` | provenance |

## Script content

Every line is free of Latin letters — the set is Telugu plus numerals and punctuation.
Digits and punctuation are not treated as English: they are script-neutral in Unicode,
and Telugu lines carry Arabic numerals often.

## Intended use

Evaluating Telugu line-level OCR. The labels carry no dependence on any particular
recogniser, so engines can be compared fairly against them.

## Sampling bias

These lines were selected through a model-error band, so they are enriched for lines our
own model got wrong and are **not** a uniform sample of the corpus. Absolute error rates
measured here are pessimistic; the sound use is comparing engines against each other on
these identical lines.

## Source

Built from Telugu Wikisource proofread pages, segmented into lines and aligned, then
hand-corrected. Text inherits Wikisource's CC BY-SA 4.0.
"""


if __name__ == "__main__":
    # ------------------------------- CONFIG -------------------------------
    WHICH = "eval"                    # "train" (full machine-labelled) | "eval" (gold)

    PUSH = True                       # False = build and inspect locally, upload nothing
    PRIVATE = False
    MAX_SHARD_SIZE = "500MB"
    SAVE_LOCAL = None                 # e.g. REPO_ROOT / "data/hf_eval_dataset"; None = skip

    # ---- train ----
    TRAIN_DATA_DIR = REPO_ROOT / "data" / "wikisource_lines"
    TRAIN_REPO = "harsha-desaraju/telugu-wikisource-text-images"
    TRAIN_SPLIT = "train"

    # ---- eval ----
    EVAL_DATA_DIR = REPO_ROOT / "data" / "wikisource_lines_eval"
    EVAL_LABELS_NAME = "lines_corrected.jsonl"
    EVAL_REPO = "harsha-desaraju/telugu-line-ocr-bench"
    EVAL_SPLIT = "test"               # it is an evaluation set, not training data
    # Publish only Latin-free lines. Numbers and punctuation are fine; only the Latin
    # alphabet disqualifies a line. See the module docstring before turning this off.
    DROP_LATIN = True
    WRITE_CARD = True                 # upload README.md alongside the data
    # -----------------------------------------------------------------------

    if WHICH not in ("train", "eval"):
        raise SystemExit(f"WHICH must be 'train' or 'eval', not {WHICH!r}")

    card = None

    if WHICH == "train":
        data_dir, repo, split, features = (
            TRAIN_DATA_DIR, TRAIN_REPO, TRAIN_SPLIT, TRAIN_FEATURES)
        rows, missing = load_train_rows(data_dir)
        if not rows:
            raise SystemExit(f"no lines found in {data_dir / 'lines.jsonl'}")
        report_train(rows, missing, data_dir)
    else:
        data_dir, repo, split, features = (
            EVAL_DATA_DIR, EVAL_REPO, EVAL_SPLIT, EVAL_FEATURES)
        rows, total, dropped = load_eval_rows(data_dir, EVAL_LABELS_NAME)
        report_eval(rows, total, dropped, data_dir)
        if DROP_LATIN:
            rows = drop_latin(rows)
        if not rows:
            raise SystemExit(f"no usable rows in {data_dir / EVAL_LABELS_NAME}")
        if WRITE_CARD:
            card = eval_dataset_card(rows)

    ds = build_dataset(rows, features)
    print()
    print(ds)

    if SAVE_LOCAL:
        ds.save_to_disk(str(SAVE_LOCAL))
        print(f"saved locally -> {SAVE_LOCAL}")

    if not PUSH:
        print("\nPUSH is False — nothing uploaded.")
        raise SystemExit(0)

    if "<" in repo or ">" in repo:
        raise SystemExit(
            f"REPO is still a placeholder ({repo!r}). Set it to a real "
            f"'<user-or-org>/<dataset-name>' before pushing.")

    hf_login()
    ds.push_to_hub(repo, split=split, max_shard_size=MAX_SHARD_SIZE, private=PRIVATE)
    print(f"pushed {len(rows):,} lines to "
          f"https://huggingface.co/datasets/{repo} (split={split})")

    if card is not None:
        from huggingface_hub import HfApi

        HfApi().upload_file(
            path_or_fileobj=card.encode("utf-8"),
            path_in_repo="README.md", repo_id=repo, repo_type="dataset")
        print("dataset card -> README.md")
