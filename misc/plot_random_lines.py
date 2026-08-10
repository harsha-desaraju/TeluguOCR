"""Eyeball a random sample of the wikisource line dataset.

Picks N random rows from <data_dir>/lines.jsonl and shows their crops with their
labels, three ways because Telugu rendering is renderer-dependent:

    HTML    the one to trust. Browsers run a real OpenType shaping engine, so
            conjuncts and matras come out right. Crops are embedded as base64, so
            the file is self-contained and survives the data dir moving.
    console byte-accurate and grep/copy-paste-able, but terminals do not shape
            Indic scripts -- expect detached matras and dotted circles.
    figure  optional (SHOW_PLOT). matplotlib does not shape either; the titles are
            only good enough to match a crop to its console line by number.

Run from the repo root:  python3 -m misc.plot_random_lines
"""

import base64
import html
import json
import random
from pathlib import Path

TELUGU_FONT = "pipelines/synth/fonts/NotoSansTelugu_Condensed-Regular.ttf"

PAGE = """<!DOCTYPE html>
<html lang="te">
<head>
<meta charset="utf-8">
<title>wikisource line sample</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 2rem auto;
         padding: 0 1rem; background: #fafafa; color: #222; }}
  .line {{ background: #fff; border: 1px solid #ddd; border-radius: 6px;
           padding: .8rem 1rem; margin-bottom: 1rem; }}
  .meta {{ font-size: .8rem; color: #777; margin-bottom: .4rem; }}
  .meta .ok {{ color: #1a7f37; font-weight: 600; }}
  .meta .rej {{ color: #b35900; font-weight: 600; }}
  img {{ max-width: 100%; image-rendering: auto; border: 1px solid #eee; }}
  .text {{ font-size: 1.5rem; margin-top: .5rem; line-height: 2; }}
</style>
</head>
<body>
<h2>{title}</h2>
{blocks}
</body>
</html>
"""

BLOCK = """<div class="line">
  <div class="meta">[{i}] {name} &middot; cer {cer:.3f} &middot; <span class="{cls}">{status}</span></div>
  <img src="data:image/jpeg;base64,{b64}" alt="line crop {i}">
  <div class="text">{text}</div>
</div>
"""


def load_rows(data_dir: Path, only_accepted: bool) -> list[dict]:
    rows = []
    with open(data_dir / "lines.jsonl", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if only_accepted and not rec.get("accepted"):
                continue
            rows.append(rec)
    return rows


def write_html(picked: list[dict], data_dir: Path, out_path: Path, title: str) -> None:
    blocks = []
    for i, rec in enumerate(picked):
        b64 = base64.b64encode((data_dir / rec["line_image"]).read_bytes()).decode()
        accepted = rec["accepted"]
        blocks.append(BLOCK.format(
            i=i, name=html.escape(rec["line_image"]), cer=rec["line_cer"],
            cls="ok" if accepted else "rej",
            status="accepted" if accepted else (rec["reject_reason"] or "stored"),
            b64=b64, text=html.escape(rec["text"]),
        ))
    out_path.write_text(PAGE.format(title=html.escape(title), blocks="".join(blocks)),
                        encoding="utf-8")


def show_figure(picked: list[dict], data_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from PIL import Image

    try:
        font_manager.fontManager.addfont(TELUGU_FONT)
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=TELUGU_FONT).get_name()
    except Exception:
        pass  # titles fall back to tofu boxes; the HTML is the reference anyway

    fig, axes = plt.subplots(len(picked), 1, figsize=(14, 1.1 * len(picked)),
                             squeeze=False)
    for i, (ax, rec) in enumerate(zip(axes[:, 0], picked)):
        with Image.open(data_dir / rec["line_image"]) as img:
            ax.imshow(img, cmap="gray", aspect="equal")
        ax.set_axis_off()
        ax.set_title(f"[{i}] cer={rec['line_cer']:.3f} "
                     f"{'accepted' if rec['accepted'] else rec['reject_reason'] or 'stored'}"
                     f"  {rec['text']}", fontsize=9, loc="left")
    fig.tight_layout()
    plt.show()


def plot_sample(data_dir: Path, n_images: int, only_accepted: bool = False,
                seed: int | None = None, html_out: Path | None = None,
                show_plot: bool = False) -> None:
    rows = load_rows(data_dir, only_accepted)
    if not rows:
        raise SystemExit(f"no lines found in {data_dir / 'lines.jsonl'}"
                         + (" with accepted=true" if only_accepted else ""))
    n = min(n_images, len(rows))
    picked = random.Random(seed).sample(rows, n)

    title = (f"{n} of {len(rows):,} lines from {data_dir}"
             + (" (accepted only)" if only_accepted else ""))
    print(title)
    for i, rec in enumerate(picked):
        print(f"[{i}] {rec['line_image']}  cer={rec['line_cer']:.3f}  "
              f"accepted={rec['accepted']}")
        print(f"    {rec['text']}")

    if html_out is not None:
        write_html(picked, data_dir, html_out, title)
        print(f"\nhtml -> {html_out}   (open in a browser for correctly shaped text)")
    if show_plot:
        show_figure(picked, data_dir)


if __name__ == "__main__":
    DATA_DIR = Path("data/wikisource_lines")
    N_IMAGES = 8
    ONLY_ACCEPTED = True    # True -> sample only lines the strict policy accepted
    SEED = None             # int for a repeatable sample
    HTML_OUT = Path("misc/line_sample.html")   # None -> skip the html report
    SHOW_PLOT = False       # matplotlib window; titles will not shape Telugu properly

    plot_sample(DATA_DIR, N_IMAGES, ONLY_ACCEPTED, SEED, HTML_OUT, SHOW_PLOT)
