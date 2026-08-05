"""Build the line-level dataset locally, from pages already on disk.

The local counterpart to kaggle_build_wikisource_lines.py: same pipeline, same output
format, no Hub upload. It reads (page scan, page text) pairs written by scrape.py and
writes line crops plus a JSONL of labels and scores. Unlike the Kaggle file this one
imports from the package rather than inlining everything, so there is one copy of the
logic to maintain and this file stays short.

    python3 -m data_curation.wikisource.scrape         # MODE="full", REPO=None
    python3 -m data_curation.wikisource.build_local

PARALLELISM
    The bottleneck is tesseract layout analysis: CPU-bound, and roughly ten times the
    per-page cost of the recogniser. So the two run concurrently -- a thread pool
    loads, deskews, segments and preprocesses pages while the main thread pulls
    finished pages off the queue and batches them through the model on the GPU.
    pytesseract shells out to the tesseract binary, so it releases the GIL and threads
    genuinely parallelise; no process pool, and no pickling of page arrays.

    The pool is kept full by holding PREFETCH pages in flight and submitting a
    replacement as each is consumed, rather than mapping over a chunk at a time.
    Chunked mapping alternates between a fully-busy pool and a fully-idle one; topping
    the queue up means segmentation never stops while the GPU works.

    cv2's own thread pool is switched off. Parallelism is already at the page level,
    and letting each of N workers spin up its own OpenCV threads oversubscribes the
    cores and makes the whole thing slower.

WHAT IS STORED
    Everything clearing a loose BASE_MAX_CER floor, with the numbers used to judge it,
    so strictness stays a query at training time instead of being frozen into a run
    that takes hours to redo. `accepted` carries the verdict of the full strict policy.
    Two things are dropped: lines whose aligned span is empty (no label exists for
    them -- running headers, folio numbers, plate captions) and lines above the floor.

OUTPUT
    <out_dir>/lines/<slug>_l007.jpg   crops at height 64, width a multiple of 8
    <out_dir>/lines.jsonl             one row per line: label, scores, flags
    <out_dir>/pages.jsonl             one row per page; also the resume marker
    <out_dir>/stats.json              summary

Resumable at page granularity: pages already in pages.jsonl are skipped, and both
JSONLs are appended and fsynced per page.
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from PIL import Image
from tqdm.auto import tqdm

from data_curation.wikisource.alignment import AcceptPolicy, align_page
from data_curation.wikisource.experiment import Page, build_engine, load_pages
from data_curation.wikisource.scrape import (
    fetch_image_bytes,
    fetch_text,
    iter_category_pages,
    slugify,
    title_from_url,
)
from data_curation.wikisource.segmentation import (
    BoxFilter,
    TesseractLayoutSegmenter,
    encode_jpeg,
    prepare_page,
    preprocess_line,
)


def _release_cache(device: str) -> None:
    """Hand the accelerator's cached blocks back.

    Every batch this pipeline runs has a different padded width -- crops are sorted by
    width and packed under a pixel budget -- so the caching allocator ends up holding a
    separate block for every distinct shape it has ever seen and never reuses most of
    them. Measured over 815 pages on MPS, live tensors stayed flat at 80 MB while the
    driver allocation climbed to 2.5 GB; releasing it drops that back to ~360 MB.

    Cheap, and only called every few hundred pages, so the throughput cost is noise.
    """
    try:
        if device == "mps":
            import torch

            torch.mps.empty_cache()
        elif device == "cuda":
            import torch

            torch.cuda.empty_cache()
    except Exception:
        pass


def _append(path: Path, records) -> None:
    """Append rows and fsync, so an interrupted run loses at most the page in flight."""
    if not records:
        return
    with open(path, "a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _done_slugs(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["slug"])
            except (json.JSONDecodeError, KeyError):
                continue  # a torn last line from a hard kill
    return done


def iter_jobs(cfg):
    """Yield the pages to process, from disk or straight off Wikisource.

    In "scrape" mode this is a generator over the live category listing, so the
    pipeline starts working on page 1 while the listing is still being walked --
    there is no upfront wait for all 53k titles.
    """
    if cfg["source"] == "dir":
        for page in load_pages(cfg["data_dir"], limit=cfg["limit_pages"]):
            yield {"slug": page.slug, "image_path": page.image_path, "text": page.text}
        return

    cache = Path(cfg["page_cache_dir"]) if cfg["cache_pages"] else None
    if cache:
        (cache / "images").mkdir(parents=True, exist_ok=True)
        (cache / "text").mkdir(parents=True, exist_ok=True)
    for page in iter_category_pages(title_from_url(cfg["start_url"]), cfg["limit_pages"]):
        slug = slugify(page)
        job = {"slug": slug, "title": page["title"], "image_urls": page["image_urls"],
               "image_path": None, "text": None}
        if cache:
            img_path, txt_path = cache / "images" / f"{slug}.jpg", cache / "text" / f"{slug}.txt"
            if img_path.exists() and txt_path.exists():
                job["image_path"] = str(img_path)     # already fetched: no request
                job["text"] = txt_path.read_text(encoding="utf-8")
            else:
                job["cache_to"] = (img_path, txt_path)
        yield job


def load_source(job, cfg, fetch_gate):
    """Get one page's scan and text, from disk or from Wikisource.

    Downloads are gated by a semaphore rather than by the worker count. The two need
    to be separate: the pool is sized for CPU work (segmentation is the slow part) and
    a worker blocked on a socket is not using a core, so letting all of them fetch at
    once would fire far more concurrent requests at Wikimedia than intended for no
    throughput gain. The gate caps requests in flight while the rest of the pool keeps
    segmenting.
    """
    if job.get("image_path"):
        with Image.open(job["image_path"]) as img:
            img.load()
            return img, job["text"]

    with fetch_gate:
        raw = fetch_image_bytes(job["image_urls"])
        if raw is None:
            return None, None
        text = fetch_text(job["title"])
    if not text.strip():
        return None, None

    if job.get("cache_to"):
        img_path, txt_path = job["cache_to"]
        img_path.write_bytes(raw)
        txt_path.write_text(text, encoding="utf-8")
    return Image.open(io.BytesIO(raw)), text


def segment_page(job, segmenter, cfg, fetch_gate):
    """Fetch (or load), deskew, segment and preprocess one page. Runs in a worker.

    Returns the crops already in encoder form, so the same array serves both the
    recogniser and the stored JPEG -- the bytes on disk are then provably the pixels
    the model saw when it judged the label.
    """
    try:
        img, text = load_source(job, cfg, fetch_gate)
        if img is None:
            return job, None, [], "no scan or no text"
        with img:
            prepared = prepare_page(img, deskew=cfg["deskew"])
        boxes = segmenter.segment(prepared)
        arrays = [preprocess_line(b.crop(prepared, pad=cfg["crop_pad"]),
                                  cfg["store_height"], cfg["max_width"],
                                  cfg["downsample"])
                  for b in boxes]
        return job, text, arrays, None
    except Exception as exc:
        return job, None, [], f"{type(exc).__name__}: {exc}"


def _jobs_and_total(cfg, done):
    """The jobs to run, and how many the progress bar should expect.

    tqdm can only show a time-remaining estimate when it knows the total, and in
    "scrape" mode the job source is a lazy generator over the category listing -- the
    count is not known until the walk finishes, which is the whole point of streaming
    it. So the total is taken from `expected_pages` there and is an ESTIMATE: the
    category had 53,714 members at last count, of which 53,506 have a scan behind them.
    The bar will drift if that has changed; the ETA is still worth far more than no ETA.

    In "dir" mode the jobs are just dicts over a directory listing, so materialising
    them is cheap and the total is exact.
    """
    if cfg["source"] == "dir":
        jobs = list(iter_jobs(cfg))
        return jobs, sum(1 for j in jobs if j["slug"] not in done)

    expected = cfg.get("expected_pages") or 53_506
    if cfg.get("limit_pages"):
        expected = min(expected, cfg["limit_pages"])
    return iter_jobs(cfg), max(0, expected - len(done))


def build(segmenter, engine, cfg, policy) -> Counter:
    out_dir = Path(cfg["out_dir"])
    (out_dir / "lines").mkdir(parents=True, exist_ok=True)
    lines_jsonl, pages_jsonl = out_dir / "lines.jsonl", out_dir / "pages.jsonl"

    done = _done_slugs(pages_jsonl)
    if done:
        print(f"[resume] {len(done)} pages already built")
    where = (f"reading pages from {cfg['data_dir']}" if cfg["source"] == "dir"
             else f"scraping" + (f", caching pages to {cfg['page_cache_dir']}"
                                 if cfg["cache_pages"] else " (pages not cached)"))
    print(f"[build] {where}")
    print(f"[build] {cfg['num_workers']} workers, "
          f"{cfg['fetch_concurrency']} concurrent fetches -> {cfg['out_dir']}")

    stats: Counter = Counter()
    started = time.time()
    jobs, total = _jobs_and_total(cfg, done)
    pending = (j for j in jobs if j["slug"] not in done)
    fetch_gate = threading.Semaphore(cfg["fetch_concurrency"])

    with ThreadPoolExecutor(max_workers=cfg["num_workers"]) as pool:
        inflight = deque()

        def submit_next():
            job = next(pending, None)
            if job is not None:
                inflight.append(pool.submit(segment_page, job, segmenter, cfg, fetch_gate))

        for _ in range(cfg["prefetch"]):
            submit_next()

        # smoothing=0.02 rather than tqdm's default 0.3: per-page time is very spiky
        # (a cold fetch that has to be rendered server-side can take seconds, a cached
        # one milliseconds), and a responsive average turns the ETA on a 16-hour run
        # into noise. Heavy smoothing gives a stable estimate of the hours remaining.
        bar = tqdm(total=total, desc="pages", unit="pg", smoothing=0.02)
        while inflight:
            job, text, arrays, error = inflight.popleft().result()
            submit_next()          # keep the pool saturated while the GPU works
            bar.update(1)

            if error:
                tqdm.write(f"  [page] {job['slug']}: {error}")
                stats["pages_failed"] += 1
                continue
            page = Page(job["slug"], job.get("image_path") or "", text)

            hypotheses = engine.run([Image.fromarray(a) for a in arrays]) if arrays else []
            if isinstance(hypotheses, tuple):     # engines with per-line confidence
                hypotheses = hypotheses[0]
            result = align_page(list(hypotheses), page.text, policy)

            source_file = page.slug.rsplit("_p", 1)[0]
            page_no = page.slug.rsplit("_p", 1)[-1]
            rows = []
            for line in result.lines:
                stats["lines_detected"] += 1
                if not line.text:
                    stats["dropped_empty_span"] += 1
                    continue
                if line.cer > cfg["base_max_cer"]:
                    stats["dropped_above_base_cer"] += 1
                    continue

                rel = f"lines/{page.slug}_l{line.index:03d}.jpg"
                (out_dir / rel).write_bytes(
                    encode_jpeg(arrays[line.index], cfg["jpeg_quality"]))
                rows.append({
                    "line_image": rel,
                    "text": line.text,
                    "line_cer": round(line.cer, 4),
                    "page_cer": round(result.page_cer, 4),
                    "page_yield": round(result.broad_yield, 4),
                    "accepted": line.accepted,
                    "reject_reason": line.reject_reason,
                    "n_graphemes": line.n_graphemes,
                    "digits_converted": line.digits_converted,
                    "starts_mid_word": line.starts_mid_word,
                    "ends_mid_word": line.ends_mid_word,
                    "slug": page.slug,
                    "page_no": int(page_no) if page_no.isdigit() else -1,
                    "source_file": source_file,
                    "line_no": line.index,
                    "checkpoint": cfg["checkpoint_name"],
                })
                stats["lines_stored"] += 1
                stats["lines_accepted"] += int(line.accepted)
                if line.reject_reason:
                    stats[f"reason_{line.reject_reason}"] += 1
                if line.digits_converted:
                    stats["digits_converted"] += 1
                if line.starts_mid_word or line.ends_mid_word:
                    stats["mid_word_break"] += 1

            _append(lines_jsonl, rows)
            _append(pages_jsonl, [{
                "slug": page.slug,
                "n_lines": result.n_lines,
                "n_stored": len(rows),
                "n_accepted": result.n_accepted,
                "page_cer": round(result.page_cer, 4),
                "page_yield": round(result.broad_yield, 4),
                "page_accepted": result.page_accepted,
                "page_reject_reason": result.page_reject_reason,
            }])
            stats["pages"] += 1
            stats["pages_accepted"] += int(result.page_accepted)

            every = cfg.get("release_cache_every")
            if every and stats["pages"] % every == 0:
                _release_cache(getattr(engine, "device", "cpu"))
        bar.close()

    stats["seconds"] = int(time.time() - started)
    return stats


def print_stats(stats: Counter, out_dir) -> None:
    pages = stats.get("pages", 0) or 1
    detected = stats.get("lines_detected", 0) or 1
    seconds = stats.get("seconds", 0) or 1
    print("\n" + "=" * 72)
    print(f"pages processed   {stats.get('pages', 0):>9,}   "
          f"accepted {stats.get('pages_accepted', 0):,} "
          f"({100 * stats.get('pages_accepted', 0) / pages:.1f}%)   "
          f"failed {stats.get('pages_failed', 0)}")
    print(f"lines detected    {stats.get('lines_detected', 0):>9,}   "
          f"({stats.get('lines_detected', 0) / pages:.1f}/page)")
    print(f"lines stored      {stats.get('lines_stored', 0):>9,}   "
          f"({100 * stats.get('lines_stored', 0) / detected:.1f}% of detected)")
    print(f"  of which strict {stats.get('lines_accepted', 0):>9,}   "
          f"({100 * stats.get('lines_accepted', 0) / detected:.1f}% of detected)")
    print(f"dropped: empty span {stats.get('dropped_empty_span', 0):,}, "
          f"above base CER {stats.get('dropped_above_base_cer', 0):,}")
    print(f"flags: digits converted {stats.get('digits_converted', 0):,}, "
          f"mid-word breaks {stats.get('mid_word_break', 0):,}")
    reasons = {k[len("reason_"):]: v for k, v in stats.items() if k.startswith("reason_")}
    if reasons:
        print("\nstored-but-not-accepted, by reason")
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {reason:<20}{count:>9,}")
    print(f"\nthroughput  {stats.get('pages', 0) / seconds:.2f} pages/s "
          f"({seconds / 60:.1f} min for {stats.get('pages', 0):,} pages)")
    print("=" * 72)
    print(f"written to {out_dir}")


if __name__ == "__main__":
    # ---- input / output ----
    START_URL = "https://te.wikisource.org/wiki/%E0%B0%B5%E0%B0%B0%E0%B1%8D%E0%B0%97%E0%B0%82:%E0%B0%86%E0%B0%AE%E0%B1%8B%E0%B0%A6%E0%B0%BF%E0%B0%82%E0%B0%9A%E0%B0%AC%E0%B0%A1%E0%B1%8D%E0%B0%A1%E0%B0%B5%E0%B0%BF"
    # Where page scans live. ONE setting for both directions on purpose: "scrape"
    # caches fetched pages here, "dir" reads them back from here. They used to default
    # to different directories, which meant that after a full scrape had filled
    # data/wikisource, flipping to source="dir" for a fast rebuild would silently read
    # data/wikisource_sample instead -- rebuilding from the 163-page dev set rather
    # than the 53k-page corpus, with nothing in the output saying so.
    # Point it at data/wikisource_sample to rebuild against the dev set instead.
    PAGE_DIR = "data/wikisource"
    OUT_DIR = "data/wikisource_lines"
    LIMIT_PAGES = 20000                      # int for a smoke test

    # ---- model ----
    VOCAB = "src/text_decoder/grapheme_tokenizer/telugu-vocab.json"
    CHECKPOINT = "models/image_encoder/ctc_encoder_stage-3/ctc-encoder-2048/final_model.pt"

    CONFIG = {
        "out_dir": OUT_DIR,

        # ---- where the pages come from ----
        # "scrape" fetches from Wikisource as it goes, so downloading overlaps
        # segmentation instead of running as a separate 15-hour pass beforehand -- the
        # processing rate becomes the pacer and no artificial sleep is needed.
        # "dir" reads pages already on disk (from scrape.py, or from a previous
        # "scrape" run's cache).
        # "scrape" fetches from Wikisource (caching into PAGE_DIR); "dir" reads pages
        # already in PAGE_DIR and makes no network calls at all.
        "source": "scrape",             # "scrape" | "dir"
        "data_dir": PAGE_DIR,           # read from, when source="dir"
        "start_url": START_URL,         # used by source="scrape"
        "limit_pages": LIMIT_PAGES,
        # Only feeds the progress bar's time-remaining estimate, since the listing is
        # walked lazily and its length is not known up front. 53,506 = the category's
        # 53,714 members minus the 208 with no scan behind them.
        "expected_pages": 53_506,

        # Keep the fetched scans. Costs ~16 GB for the full category and is worth it:
        # every rebuild after a threshold change or a bug fix then runs off disk in
        # ~5 hours instead of re-downloading 53k pages, which is both slow and rude to
        # a volunteer-funded host. Set False if disk is tight.
        "cache_pages": True,
        "page_cache_dir": PAGE_DIR,     # written to, when source="scrape"

        "checkpoint_name": Path(CHECKPOINT).parent.parent.name,

        # ---- parallelism ----
        # Workers do the CPU work (tesseract, deskew, resize); the main thread owns the
        # GPU. Sized to the DOWNLOAD ceiling, not to the machine: with source="scrape"
        # the run is bound by Wikimedia at ~0.90 pages/s, so any processing rate above
        # that is wasted and just heats the laptop. Measured on 11 cores:
        #     2 workers -> 1.25 pages/s   (1.4x the ceiling)
        #     3 workers -> 1.70 pages/s   (1.9x)   <-- default
        #     4 workers -> 2.17 pages/s   (2.4x)
        #     6 workers -> 2.77 pages/s   (3.1x)
        #     9 workers -> 3.43 pages/s   (3.8x)
        # 3 leaves 8 cores free for other work and still keeps ~2x headroom, so a burst
        # of CDN-warm pages cannot make processing the bottleneck. Wall-clock for the
        # full category is ~16.5h at any of these, because downloads bind throughout.
        #
        # RAISE THIS FOR A REBUILD. With source="dir", or a re-run against a populated
        # page cache, there is no download ceiling and the run IS processing-bound:
        # 9 workers finish in ~4.3h where 3 would take ~8.7h.
        "num_workers": 3,
        "prefetch": 6,              # pages in flight; 2x workers keeps the pool full
        # Release the accelerator's cached blocks this often (in pages). See
        # _release_cache: without it the MPS allocator grows to ~2.5 GB over a few
        # hundred pages and holds it for the rest of the run.
        "release_cache_every": 50,
        # Concurrent HTTP requests, capped separately from the worker count: workers
        # are sized for CPU work, and a worker blocked on a socket is not using a core.
        #
        # 2 is not conservatism, it is the measured optimum. On pages nobody had
        # rendered before (cold, so MediaWiki has to rasterise the PDF page on demand):
        #     concurrency 2  ->  0.90 img/s, no throttling
        #     concurrency 4  ->  0.65 img/s, 8 x HTTP 429
        #     concurrency 8  ->  0.58 img/s, 19 x HTTP 429
        # Asking for more makes it SLOWER -- Wikimedia throttles and the backoff waits
        # swamp the extra parallelism. ~0.9 pages/s is a server-side ceiling, not
        # something to engineer around, and this is a volunteer-funded host telling us
        # to slow down.
        "fetch_concurrency": 2,

        # ---- crops ----
        "store_height": 64,
        "max_width": 2048,          # the recogniser's cap; wider lines get squashed
        "downsample": 8,            # width padded to a multiple of this
        "jpeg_quality": 90,         # ~8 KB/line
        "crop_pad": 3,
        "deskew": False,

        # ---- what to keep ----
        # The storage floor, NOT the training threshold. See the module docstring.
        "base_max_cer": 0.25,
    }

    POLICY = AcceptPolicy(
        max_line_cer=0.10,
        edge_max_line_cer=0.05,
        min_graphemes=5,
        max_grapheme_ratio=3.0,
        convert_digits=True,
        require_digit_evidence=True,
        reject_script_mismatch=True,
        min_page_yield=0.35,
        page_yield_cer=0.40,
        max_page_cer=0.5,
    )

    # Page-level parallelism already saturates the cores; per-call OpenCV threads on
    # top of it only oversubscribe them.
    cv2.setNumThreads(1)

    segmenter = TesseractLayoutSegmenter(box_filter=BoxFilter())
    engine = build_engine(CHECKPOINT, VOCAB)

    stats = build(segmenter, engine, CONFIG, POLICY)
    Path(OUT_DIR, "stats.json").write_text(
        json.dumps(dict(stats), ensure_ascii=False, indent=2), encoding="utf-8")
    print_stats(stats, OUT_DIR)
