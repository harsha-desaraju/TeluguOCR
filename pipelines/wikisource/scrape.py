"""Extract (page image, proofread text) pairs from Telugu Wikisource.

The start URL is a category page (e.g. వర్గం:ఆమోదించబడ్డవి = "validated pages").
Its members live in the పుట: (Page:) namespace, where every page is one scanned
page of a book: a PDF/DjVu page image on Commons plus its human-proofread text.

Traversal uses the MediaWiki API:
  1. list the category members in batches, following the `continue` token
     (this is the API equivalent of clicking "next page" on the category);
  2. per page, download the scan as JPEG and the rendered text, with NUM_WORKERS
     threads doing the downloads.

RESOLUTION IS NOT A DETAIL -- see image_url_candidates. The API hands back the
wiki's default thumbnail size, which for a large share of this category is 500px:
line heights of ~20px, which upscale to the encoder's 64px as mush. Every scan is
requested at 1280px instead. Getting this wrong is silent and ruins the dataset.

TWO MODES, one listing
    MODE = "full"    every page in the category, sharded and optionally pushed
                     to the Hub. This is the production scrape.
    MODE = "sample"  a stratified dev set: N pages spread as evenly as possible
                     over the distinct BOOKS in the category, downloaded locally
                     and nothing else. This is what the segmenter/engine
                     experiment runs on, and stratifying matters -- the category
                     is dominated by a handful of large books, so a naive head or
                     random sample lands on two or three typefaces and the
                     experiment then picks an engine that is good at those.

    Both modes share `list_category_pages`, which caches the full member listing
    to <out_dir>/category_pages.jsonl. The listing is ~54k rows and takes a few
    minutes of API calls, so the cache means the sample run pays for it once and
    the production run reuses it.

WHY THE TEXT IS NOT LINE-ALIGNED
    Proofreaders reflow prose: the wikitext for a page of prose is one unbroken
    paragraph regardless of how many physical lines the scan has, and the <poem>
    tag (present on roughly a third of pages) is frequently used as a paragraph
    wrapper rather than to preserve verse lines. Where the source DOES carry line
    breaks -- title pages, tables of contents, real verse -- they survive as <br>
    and `fetch_text` keeps them. So the text here is *sometimes* line-aligned and
    usually not, which is why alignment.py exists rather than a split on "\\n".

Pages are processed in shards of SHARD_SIZE. Each finished shard is uploaded to
the Hub as its own config (train_0000, train_0001, ...), so a crash costs at most
one shard. Already-uploaded configs are skipped, and already-downloaded pages are
read off disk instead of re-fetched, making a re-run resumable. Shard k always
holds the same pages because the category listing order is stable.

Local layout:
  <out_dir>/category_pages.jsonl      cached category listing (all modes)
  <out_dir>/images/<slug>.jpg
  <out_dir>/text/<slug>.txt
  <out_dir>/shards/train_0000.jsonl   one row per pair in that shard (full mode)

To consume everything later, concatenate the "train_*" configs of REPO.

Running on Kaggle: add your token as a Kaggle secret named HF_TOKEN
(Add-ons -> Secrets), and keep CLEANUP_AFTER_PUSH = True -- the full set of
images is ~17 GB, well over the writable disk of a Kaggle session, so each
shard's local files are deleted once it is safely on the Hub.
"""

import json
import os
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

API = "https://te.wikisource.org/w/api.php"
HEADERS = {"User-Agent": "TeluguOCR-dataset/0.1 (your@email.com)"}

# one requests.Session per thread (connection pooling; Sessions aren't thread-safe)
_local = threading.local()


def session():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers.update(HEADERS)
    return _local.s


def get(params, retries=6, delay=1.0):
    """GET the MediaWiki API as JSON, retrying on failure.

    Throttling is handled separately from other errors, for the same reason
    fetch_image_bytes does it: the API answers 429 under concurrent load, and
    retrying on a short linear backoff just hammers straight back into the
    throttle. A 429/503 is waited out with exponential backoff, honouring
    Retry-After when the server sends one -- otherwise a parallel run loses
    every page in a batch to a limit that would have cleared in seconds.
    """
    params = {**params, "format": "json", "formatversion": 2}
    backoff = 2.0
    for attempt in range(retries):
        try:
            r = session().get(API, params=params, timeout=60)
            if r.status_code in (429, 503):
                wait = float(r.headers.get("Retry-After") or backoff)
                tqdm.write(f"  api throttled ({r.status_code}), waiting {wait:.0f}s")
                time.sleep(wait)
                backoff = min(backoff * 2, 60.0)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            tqdm.write(f"  api error ({e}), retrying...")
            time.sleep(delay * (attempt + 1))
    return None


def title_from_url(url):
    """https://te.wikisource.org/wiki/%E0%B0%B5...  ->  'వర్గం:ఆమోదించబడ్డవి'"""
    path = urlparse(url).path
    return unquote(path.split("/wiki/", 1)[1]).replace("_", " ")


IMAGE_PROPS = "filename|size|fullsize|url|responsiveimages"


def iter_category_pages(category, limit=None):
    """Yield {title, filename, page_no, image_urls} for every Page: in the category.

    One API call returns 50 members together with their scan-file info; the
    `continue` token moves to the next batch until the category is exhausted.
    """
    params = {
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": category,
        "gcmnamespace": 104,  # పుట: (Page:) namespace
        "gcmlimit": 50,
        "prop": "imageforpage",
        "prppifpprop": IMAGE_PROPS,
    }
    seen = 0
    while True:
        data = get(params)
        if data is None:
            return
        for page in data.get("query", {}).get("pages", []):
            info = page.get("imagesforpage") or {}
            if not info.get("filename"):
                continue  # page with no scan behind it
            # "పుట:somebook.pdf/12" -> page number 12
            match = re.search(r"/(\d+)$", page["title"])
            if not match:
                continue
            yield {
                "title": page["title"],
                "filename": info["filename"],
                "page_no": int(match.group(1)),
                "image_urls": image_url_candidates(info),
            }
            seen += 1
            if limit and seen >= limit:
                return
        if "continue" not in data:
            return
        params.update(data["continue"])


_THUMB_WIDTH_RE = re.compile(r"-(\d+)px-")


def image_url_candidates(info, target_width=1280):
    """URLs for one page's scan, best first, for `fetch_image` to try in order.

    The base URL must come from the API rather than be assembled from the file name.
    Two things break the assembled form, both silently and both common here: these
    books are hosted per-wiki (upload.wikimedia.org/wikisource/te/...) rather than on
    Commons, and the width baked into a thumbnail path has to be one the server will
    serve. Roughly two thirds of a sample of pages 404'd on hand-built Commons URLs.

    THE WIDTH IS THE PART THAT MATTERS FOR OCR. What the API offers is the wiki's
    default thumbnail size, not the best rendering available, and for a large share
    of this category that default is 500px -- a page image that small has ~20px line
    heights, and upscaling those to the encoder's 64px produces crops nothing can
    read. In a 163-page sample, 47% came back at 500px, and their page CER was far
    worse than the 1280px pages'.

    MediaWiki renders PDF and DjVu pages on demand at whatever width the thumbnail
    path asks for, so the width token is rewritten to `target_width` and that is
    tried first. It is a request, not a guarantee: asking above the source's native
    resolution answers HTTP 400 (verified -- a file whose native width is 1280 serves
    1280 and refuses 1600), which is why the API's own URL is kept as the fallback.

    1280px is the target because it is where the corpus tops out and it puts line
    heights in the 38-70px range, straddling the 64px the encoder wants. Asking for
    more would only cost bandwidth and disk: the crops get resized to 64px high
    regardless.
    """
    seen, scored = set(), []
    candidates = [info.get("thumbnail"), info.get("fullsize")]
    candidates.extend((info.get("responsiveimages") or {}).values())

    for url in candidates:
        if not url:
            continue
        if url.startswith("//"):
            url = "https:" + url
        url = url.split("?", 1)[0]  # drop the API's utm_* tracking params
        if url in seen:
            continue
        seen.add(url)
        match = _THUMB_WIDTH_RE.search(url)
        scored.append((int(match.group(1)) if match else 0, url))

    if not scored:
        return []

    # Prefer the smallest rendering that already meets the target; otherwise ask the
    # thumbnailer to render the widest one we know about at the target width.
    big_enough = sorted(s for s in scored if s[0] >= target_width)
    if big_enough:
        return [big_enough[0][1]]

    width, best = max(scored)
    upscaled = _THUMB_WIDTH_RE.sub(f"-{target_width}px-", best, count=1)
    return [upscaled, best] if upscaled != best else [best]


def title_batches(titles, max_count=50, max_chars=3500):
    """Group titles into API calls bounded by BOTH count and encoded URL length.

    Count alone is not enough. A Telugu title percent-encodes to 300-700 bytes
    ("పుట:షహీద్-యే-ఆజం అష్ఫాఖుల్లా ఖాన్.pdf/16" is ~380), so 50 of them make a
    query string several times over the server's URI limit and every call comes
    back 414. ASCII-titled books never hit it, which is why the production
    scrape never saw this.
    """
    batch, size = [], 0
    for title in titles:
        cost = len(quote(str(title))) + 3          # +3 for the "|" separator
        if batch and (len(batch) >= max_count or size + cost > max_chars):
            yield batch
            batch, size = [], 0
        batch.append(title)
        size += cost
    if batch:
        yield batch


def resolve_image_urls(pages, batch_size=50):
    """Fill in `image_url` for pages that lack it, batched per API call.

    Listings cached before image URLs were recorded still work: the pages they hold
    are topped up here instead of forcing a re-pull of the whole 53k-row category.
    """
    missing = [p for p in pages if not p.get("image_urls")]
    if not missing:
        return pages

    by_title = {p["title"]: p for p in missing}
    batches = list(title_batches(list(by_title), max_count=batch_size))
    for chunk in tqdm(batches, desc="image urls", unit="batch"):
        data = get({"action": "query", "prop": "imageforpage",
                    "prppifpprop": IMAGE_PROPS, "titles": "|".join(chunk)})
        if data is None:
            continue
        for page in data.get("query", {}).get("pages", []):
            target = by_title.get(page.get("title"))
            info = page.get("imagesforpage") or {}
            if target is not None and info:
                target["image_urls"] = image_url_candidates(info)
    resolved = sum(bool(p.get("image_urls")) for p in missing)
    print(f"[image urls] resolved {resolved}/{len(missing)}")
    return pages


def list_category_pages(category, cache_path, limit=None, refresh=False):
    """Full category listing, cached to disk as JSONL.

    The listing costs ~1100 API calls for the validated-pages category, so both
    the sample run and the production run read it from here after the first time.
    Pass refresh=True to re-pull (the category grows as pages are proofread).
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        pages = [json.loads(ln) for ln in cache_path.read_text(encoding="utf-8").splitlines() if ln]
        print(f"[listing] {len(pages)} pages from cache {cache_path}")
        return pages

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pages = []
    for page in tqdm(iter_category_pages(category, limit), desc="listing", unit="pg"):
        pages.append(page)
    cache_path.write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pages), encoding="utf-8"
    )
    print(f"[listing] {len(pages)} pages -> {cache_path}")
    return pages


def stratified_sample(pages, n, seed=0):
    """Pick `n` pages spread as evenly as possible over the distinct books.

    Round-robin across books, taking pages in a shuffled order within each book,
    so a category dominated by two 5000-page books does not hand back a sample of
    two typefaces. Books are visited in a shuffled but seed-stable order, so a
    given (pages, n, seed) always yields the same sample and the local download
    cache stays valid across runs.
    """
    import random

    rng = random.Random(seed)
    by_book = defaultdict(list)
    for page in pages:
        by_book[page["filename"]].append(page)

    books = sorted(by_book)  # sort first so the shuffle is reproducible
    rng.shuffle(books)
    for book in books:
        by_book[book].sort(key=lambda p: p["page_no"])
        rng.shuffle(by_book[book])

    picked, cursor = [], 0
    while len(picked) < n:
        took_any = False
        for book in books:
            if cursor < len(by_book[book]):
                picked.append(by_book[book][cursor])
                took_any = True
                if len(picked) >= n:
                    break
        if not took_any:  # every book exhausted
            break
        cursor += 1

    print(f"[sample] {len(picked)} pages from {len({p['filename'] for p in picked})} books "
          f"(category has {len(books)} books)")
    return picked


def fetch_text(title):
    """Rendered (template-expanded) page text, without header/footer/notes.

    Rendering rather than raw wikitext is deliberate: templates like {{Center|...}}
    expand to real <br> breaks, so the pages that ARE line-per-line in the source
    come out line-per-line here. Raw wikitext would hand back template calls.
    """
    data = get(
        {
            "action": "parse",
            "page": title,
            "prop": "text",
            "disablelimitreport": 1,
            "disableeditsection": 1,
        }
    )
    if data is None or "parse" not in data:
        return ""
    soup = BeautifulSoup(data["parse"]["text"], "html.parser")
    body = soup.select_one("div.pagetext")
    if body is None:
        return ""
    for tag in body.select("sup.reference, div.reflist, ol.references"):
        tag.decompose()
    lines = [ln.strip() for ln in body.get_text("\n").split("\n")]
    return "\n".join(ln for ln in lines if ln)


def fetch_image_bytes(urls, retries=4, timeout=90):
    """Download one page scan into memory, trying each candidate URL in preference order.

    Two failure modes are handled separately because they need opposite responses:

    Throttling. upload.wikimedia.org answers 429 under concurrent load, and a plain
    retry loop treats that as a transient error and hammers straight back into the
    throttle -- which is how a 200-page download stalls at 13 files with every worker
    spinning. A 429 or 503 is waited out with exponential backoff, honouring the
    server's Retry-After when it sends one.

    A rendering that does not exist. 400 (asked for a width above the source's native
    resolution) and 404 (no such rendering) will not become correct on a retry, so
    they move straight on to the next candidate -- which is the whole point of taking
    a list: the first entry asks the thumbnailer for a readable 1280px render, and
    the fallback is whatever the API said already exists.
    """
    if isinstance(urls, str):
        urls = [urls]
    if not urls:
        return None

    for url in urls:
        delay = 2.0
        for _ in range(retries):
            try:
                r = session().get(url, timeout=timeout)
                if r.status_code in (429, 503):
                    wait = float(r.headers.get("Retry-After") or delay)
                    tqdm.write(f"  throttled ({r.status_code}), waiting {wait:.0f}s")
                    time.sleep(wait)
                    delay = min(delay * 2, 60.0)
                    continue
                if r.status_code in (400, 404):
                    break  # this rendering does not exist; try the next candidate
                r.raise_for_status()
                return r.content
            except Exception as e:
                tqdm.write(f"  image error ({e}), retrying...")
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
    tqdm.write(f"  no rendering available: {urls[-1]}")
    return None


def fetch_image(urls, dest, retries=4, timeout=90):
    """`fetch_image_bytes` straight to a file. Kept so the page-level scrape writes
    scans to disk while build_local can stream them without touching it."""
    raw = fetch_image_bytes(urls, retries, timeout)
    if raw is None:
        return False
    dest.write_bytes(raw)
    return True


def slugify(page):
    stem = re.sub(r"\.(pdf|djvu)$", "", page["filename"], flags=re.I)
    stem = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")
    return f"{stem}_p{page['page_no']:04d}"


def fetch_pair(page, img_dir, txt_dir, delay=0.0, slug_fn=slugify):
    """Download one (image, text) pair. Runs in a worker thread.

    Returns a record dict, or None if the page has no text / the image failed.
    Pages already on disk are read back instead of re-downloaded (resume).

    `slug_fn` overrides how the on-disk name is derived. It exists because
    `slugify` drops non-ASCII, which is fine for this scrape (ASCII book names)
    but collapses Telugu-titled books to a bare "_pNNNN" — see
    build_benchmark_set.py, which passes a Unicode-preserving one.
    """
    slug = slug_fn(page)
    img_path = img_dir / f"{slug}.jpg"
    txt_path = txt_dir / f"{slug}.txt"

    if img_path.exists() and txt_path.exists():
        text = txt_path.read_text(encoding="utf-8")
    else:
        text = fetch_text(page["title"])
        if not text:
            return None
        if not fetch_image(page.get("image_urls"), img_path):
            return None
        txt_path.write_text(text, encoding="utf-8")
        if delay:
            time.sleep(delay)

    return {
        "slug": slug,
        "image": str(img_path),
        "text": text,
        "title": page["title"],
        "source_file": page["filename"],
        "page_no": page["page_no"],
    }


def download_pages(pages, out_dir, num_workers=8, delay=0.0, desc="pages",
                   slug_fn=slugify):
    """Download a list of pages into <out_dir>/{images,text}. Returns the records."""
    out_dir = Path(out_dir)
    img_dir, txt_dir = out_dir / "images", out_dir / "text"
    for d in (img_dir, txt_dir):
        d.mkdir(parents=True, exist_ok=True)

    records = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(fetch_pair, p, img_dir, txt_dir, delay, slug_fn)
                   for p in pages]
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="pg"):
            record = future.result()
            if record:
                records.append(record)
    records.sort(key=lambda r: r["slug"])  # as_completed order is arbitrary
    return records


def chunked(iterable, size):
    """Group an iterable into lists of at most `size` items."""
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def hf_login():
    """Log in to the Hub from HF_TOKEN: env var, .env file, or Kaggle secret."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from dotenv import load_dotenv

            load_dotenv()
            token = os.environ.get("HF_TOKEN")
        except Exception:
            pass
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient

            token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            token = None
    if token:
        from huggingface_hub import login

        login(token=token)
    else:
        print("warning: no HF_TOKEN found, uploads will fail")


def push_shard(records, repo, config_name, max_shard_size="500MB"):
    """Upload one shard of records to the Hub as its own config."""
    from datasets import Dataset, Features, Value
    from datasets import Image as HF_Image

    ds = Dataset.from_dict(
        {k: [r[k] for r in records] for k in records[0]},
        features=Features(
            {
                "slug": Value("string"),
                "image": HF_Image(),
                "text": Value("string"),
                "title": Value("string"),
                "source_file": Value("string"),
                "page_no": Value("int64"),
            }
        ),
    )
    ds.push_to_hub(
        repo,
        config_name=config_name,
        split="train",
        max_shard_size=max_shard_size,
    )


def scrape_sample(start_url, out_dir, n_pages=200, num_workers=8, delay=0.0,
                  seed=0, refresh_listing=False):
    """Download a stratified dev set. Local only, no Hub push."""
    category = title_from_url(start_url)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pages = list_category_pages(category, out_dir / "category_pages.jsonl",
                                refresh=refresh_listing)
    picked = resolve_image_urls(stratified_sample(pages, n_pages, seed=seed))
    records = download_pages(picked, out_dir, num_workers, delay, desc="sample")

    manifest = out_dir / "sample_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )
    print(f"\ndone. {len(records)}/{len(picked)} pairs in {out_dir}  (manifest: {manifest})")
    return records


def scrape(
    start_url,
    out_dir,
    repo=None,
    shard_size=5000,
    num_workers=8,
    limit=None,
    total=53_714,
    delay=0.0,
    cleanup_after_push=False,
):
    category = title_from_url(start_url)
    out_dir = Path(out_dir)
    img_dir, txt_dir, shard_dir = out_dir / "images", out_dir / "text", out_dir / "shards"
    for d in (img_dir, txt_dir, shard_dir):
        d.mkdir(parents=True, exist_ok=True)

    # configs already on the Hub -> skip those shards entirely
    existing = set()
    if repo:
        from datasets import get_dataset_config_names

        hf_login()
        try:
            existing = set(get_dataset_config_names(repo))
        except Exception:
            existing = set()  # repo doesn't exist yet
    print(f"category: {category}")
    print(f"repo: {repo or '(local only)'}  |  existing configs: {sorted(existing) or 'none'}")

    bar = tqdm(total=limit or total, desc="pages", unit="pg", smoothing=0.05)
    kept = 0
    for shard_idx, chunk in enumerate(chunked(iter_category_pages(category, limit), shard_size)):
        config_name = f"train_{shard_idx:04d}"
        if config_name in existing:
            bar.update(len(chunk))
            tqdm.write(f"[skip] {config_name} already on the Hub")
            continue

        records = []
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = [pool.submit(fetch_pair, p, img_dir, txt_dir, delay) for p in chunk]
            for future in as_completed(futures):
                record = future.result()
                if record:
                    records.append(record)
                bar.update(1)

        records.sort(key=lambda r: r["slug"])  # as_completed order is arbitrary
        kept += len(records)
        (shard_dir / f"{config_name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
            encoding="utf-8",
        )
        tqdm.write(f"[shard] {config_name}: {len(records)}/{len(chunk)} pairs")

        if repo and records:
            tqdm.write(f"[upload] {config_name} ({len(records)} rows) ...")
            push_shard(records, repo, config_name)
            if cleanup_after_push:
                # the shard is safely on the Hub; free the local copies (Kaggle disk)
                for record in records:
                    Path(record["image"]).unlink(missing_ok=True)
                    (txt_dir / f"{record['slug']}.txt").unlink(missing_ok=True)
                tqdm.write(f"[clean] removed local files for {config_name}")

    bar.close()
    print(f"\ndone. {kept} pairs in {out_dir}")


if __name__ == "__main__":
    START_URL = "https://te.wikisource.org/wiki/%E0%B0%B5%E0%B0%B0%E0%B1%8D%E0%B0%97%E0%B0%82:%E0%B0%86%E0%B0%AE%E0%B1%8B%E0%B0%A6%E0%B0%BF%E0%B0%82%E0%B0%9A%E0%B0%AC%E0%B0%A1%E0%B1%8D%E0%B0%A1%E0%B0%B5%E0%B0%BF"

    MODE = "sample"  # "sample" = stratified dev set, local only; "full" = whole category

    # ---- sample mode ----
    SAMPLE_OUT_DIR = "data/wikisource_sample"
    SAMPLE_PAGES = 200  # pages, spread round-robin over the distinct books
    SAMPLE_SEED = 0  # same seed -> same pages -> the local cache stays valid
    REFRESH_LISTING = False  # True to re-pull the category listing

    # ---- full mode ----
    OUT_DIR = "data/wikisource"
    REPO = "harsha-desaraju/telugu-wikisource-pages"  # None = download locally, no upload
    SHARD_SIZE = 5000  # pages per Hub config; uploaded as soon as it is complete
    # Category size, for the progress bar only. The bar will finish ~208 short of this:
    # that many members have no scan behind them and are skipped at listing time.
    TOTAL = 53_714
    LIMIT = None  # None = all pages
    CLEANUP_AFTER_PUSH = True  # delete a shard's local files once it is on the Hub

    # ---- both ----
    # Do NOT raise these hoping for speed -- measured on cold pages, more concurrency
    # is strictly slower: 2 workers give 0.90 pages/s with no throttling, 4 give 0.65
    # with eight 429s, 8 give 0.58 with nineteen. ~0.9 pages/s is a server-side
    # ceiling. build_local.py with source="scrape" overlaps this with processing,
    # which is the only real way to save wall-clock time.
    NUM_WORKERS = 2  # threads downloading pages
    DELAY = 0.3  # extra per-page sleep inside each worker

    if MODE == "sample":
        scrape_sample(
            START_URL,
            SAMPLE_OUT_DIR,
            n_pages=SAMPLE_PAGES,
            num_workers=NUM_WORKERS,
            delay=DELAY,
            seed=SAMPLE_SEED,
            refresh_listing=REFRESH_LISTING,
        )
    elif MODE == "full":
        scrape(
            START_URL,
            OUT_DIR,
            repo=REPO,
            shard_size=SHARD_SIZE,
            num_workers=NUM_WORKERS,
            limit=LIMIT,
            total=TOTAL,
            delay=DELAY,
            cleanup_after_push=CLEANUP_AFTER_PUSH,
        )
    else:
        raise SystemExit(f"unknown MODE {MODE!r}; use 'sample' or 'full'")
