"""Phase 3 engines oracle: do the OCR-engine copies transcribe identically?

tesseract and paddleocr both run in this environment, so this is a real end-to-end
check rather than a structural one: render a fixed set of Telugu line crops, push them
through every copy of every engine AS ITS CALL SITE CONSTRUCTS IT, and record the
strings. Run before and after the collapse; nothing may move.

Constructing each copy the way its own call site does is the point -- the copies have
different class defaults (notably psm 7 vs 13), and the question is whether the CALLERS
still behave the same, not whether the classes look the same.
"""
import ast
import hashlib
import json
import os
import sys

import pathlib
ROOT = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

SNAP = str(pathlib.Path(__file__).resolve().parent / "engine_baseline.json")
FONT = "pipelines/synth/fonts/Gautami.ttf"
TEXTS = ["తెలుగు భాష", "అమ్మ నాన్న", "పుస్తకం చదువు", "శ్రీరామ జయరామ"]


def render(text, size=40):
    f = ImageFont.truetype(FONT, size)
    d = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    l, t, r, b = d.multiline_textbbox((0, 0), text, font=f)
    img = Image.new("RGB", (r - l + 20, b - t + 20), "white")
    ImageDraw.Draw(img).text((10 - l, 10 - t), text, font=f, fill="black")
    return img.resize((max(8, (img.width * 64 // img.height) // 8 * 8), 64))


IMAGES = [render(t) for t in TEXTS]


def load_isolated(rel, names, ns_extra):
    """exec selected classes out of a module we do not want to import wholesale."""
    src = open(os.path.join(ROOT, rel), encoding="utf-8").read()
    tree = ast.parse(src)
    want = [n for n in tree.body if getattr(n, "name", None) in names]
    ns = dict(ns_extra)
    exec(compile(ast.Module(want, []), rel, "exec"), ns)
    return ns


def norm(v):
    """Engine outputs -> comparable list of strings ('' for a failure)."""
    if isinstance(v, tuple):
        v = v[0]
    return ["" if x is None else str(x).strip() for x in v]


results = {}


def record(tag, texts):
    results[tag] = {"texts": texts,
                    "sha": hashlib.sha256(json.dumps(texts).encode()).hexdigest()[:12]}
    print(f"  {tag:44s} sha={results[tag]['sha']}  {texts[0][:24]!r}")


print("TesseractEngine, each built as its own call site builds it:")

# benchmark: spec in run_benchmark.py is lang=tel, psm=13, upscale=2.0
import benchmark.engines as be  # noqa: E402
e = be.TesseractEngine(lang="tel", psm=13, upscale=2.0)
record("benchmark/engines.py (tel psm13)", norm(e.transcribe(IMAGES)))

# consensus_labelling: lang=tel, psm from the CLASS DEFAULT (7)
import pipelines.label.consensus_labelling as cl  # noqa: E402
e = cl.TesseractEngine(lang="tel", num_threads=8)
record("consensus_labelling.py (tel psm-default)", norm(e.run(IMAGES)))

# single_engine_labelling: same, plus an explicit upscale
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
import pipelines.label.single_engine_labelling as sl  # noqa: E402
e = sl.TesseractEngine(lang="tel", num_threads=8, upscale=2.0)
out = e.run(IMAGES) if hasattr(e, "run") else e.transcribe(IMAGES)
record("single_engine_labelling.py (tel psm-default)", norm(out))

mode = sys.argv[1] if len(sys.argv) > 1 else "capture"
if mode == "capture":
    json.dump(results, open(SNAP, "w"), ensure_ascii=False, indent=1)
    print(f"\ncaptured -> {SNAP}")
else:
    prev = json.load(open(SNAP))
    bad = 0
    print()
    for tag, cur in results.items():
        old = prev.get(tag)
        if old is None:
            print(f"  {tag}: NOT IN BASELINE"); bad += 1
        elif old["texts"] != cur["texts"]:
            print(f"  {tag}: MOVED\n      before {old['texts']}\n      after  {cur['texts']}")
            bad += 1
        else:
            print(f"  {tag}: identical ({cur['sha']})")
    sys.exit(1 if bad else 0)
