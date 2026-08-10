"""Render synthetic Telugu line images: text -> font -> degraded height-64 crop.

DOES NOT RUN AS-IS, and has not for a long time. Two separate reasons:

  1. There is no ``if __name__ == "__main__"`` guard -- the whole file executes on
     import, including the load_dataset below. That also makes the module unimportable,
     so nothing can reuse anything in it.
  2. It reads ``ocr_text_lines.parquet`` from the CURRENT WORKING DIRECTORY, and that
     file is not in the repo. Supply it, or repoint the path.

Fixing (1) means indenting the script body into a __main__ block, which is the
convention the rest of this repo follows. Left alone so far because the file cannot be
executed here to check the result.
"""

import random
from pathlib import Path
from torchvision import transforms
from datasets import load_dataset, DatasetDict, Features, Value
from datasets import Image as HF_Image



seed=42
num_proc=6

result = load_dataset("parquet", data_files="ocr_text_lines.parquet")['train']
print(result)


# ======================================================================================
# Image generation + preprocessing setup (fonts, sizes, encoder preprocessing params)
# ======================================================================================


# Fonts uploaded as a Kaggle dataset (same layout as image-generation.ipynb).
FONTS_DIR = "/Users/xai/Personal/Projects/TeluguOCR/pipelines/synth/fonts"

rejected_fonts = ['Pothana2000', 'ponnala', 'Lohit_Telugu', 'Vemana']
danda_non_support_fonts = ["Manu_Bold", "hind-guntur", "Deva_Normal",
                           "AkayaTelivigala-Regular", "AnekTelugu_Condensed-Regular",
                           "Menaka-Italic", "Amruta-Bold", "Sitara-Bold-Italic", "Akshar"]

FONT_SIZES = [30, 32, 34, 36, 38, 40]

font_names = [f.stem for f in Path(FONTS_DIR).rglob("*.ttf")]
font_names = list(set(font_names).difference(set(rejected_fonts)))
# Sanskrit lines use dandas; drop fonts that cannot render them.
sanskrit_fonts = list(set(font_names).difference(set(danda_non_support_fonts)))

# Reserve NUM_VAL_FONTS fonts EXCLUSIVELY for the validation set (unseen at training
# time). They are sampled from the danda-capable set so validation works for every
# language, including Sanskrit. Set `val_fonts` explicitly to pin specific font stems;
# leave it empty to auto-select a fixed (seeded) sample.
NUM_VAL_FONTS = 5
val_fonts = []   # e.g. ["Gautami", "NTR-Regular", ...]; empty -> auto-select
if not val_fonts:
    assert len(sanskrit_fonts) > NUM_VAL_FONTS, "not enough fonts to reserve for validation"
    val_fonts = random.Random(seed).sample(sorted(sanskrit_fonts), NUM_VAL_FONTS)

# Training fonts = everything except the reserved validation fonts.
_val_set = set(val_fonts)
train_fonts = [f for f in font_names if f not in _val_set]
train_sanskrit_fonts = [f for f in sanskrit_fonts if f not in _val_set]

# Encoder image-preprocessing params (see CLAUDE.md): grayscale line images of
# height 64, patch size 8, variable width up to 1024.
image_height = 64
patch_size = 8
max_image_width = 1024

print(f"{len(font_names)} fonts total | train: {len(train_fonts)} "
      f"({len(train_sanskrit_fonts)} sanskrit-capable) | "
      f"val (reserved, unseen): {val_fonts}")



# --------------------------------------------------------------------------------------
# Render one line of text to an auto-sized PIL image (from image-generation.ipynb).
# --------------------------------------------------------------------------------------
from PIL import Image, ImageDraw, ImageFont
from src.telugu_ocr.data.preprocess import resize_line_image


def generate_image(text, font_path, font_size: int = 40, margin: int = 5,
                   background="white", text_color="black"):
    primary_font = ImageFont.truetype(font_path, font_size)

    # Temporary image just for measuring text
    dummy_img = Image.new("RGB", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)

    # Measure text
    left, top, right, bottom = dummy_draw.multiline_textbbox(
        (0, 0), text, font=primary_font, spacing=4)

    text_width = right - left
    text_height = bottom - top

    # Image size = text size + margins
    width = text_width + 2 * margin
    height = text_height + 2 * margin

    img = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(img)
    draw.text((margin - left, margin - top), text, font=primary_font, fill=text_color)
    return img



# --------------------------------------------------------------------------------------
# Image preprocessing for the encoder (from src/telugu_ocr/data/preprocess.py), WITHOUT the
# final tensor conversion: grayscale -> resize to `image_height` (aspect-preserving,
# capped at `max_image_width`) -> pad width to the nearest multiple of `patch_size`
# (fill=255). Returns a PIL.Image; the ToTensor + Normalize step is intentionally omitted.
# --------------------------------------------------------------------------------------



class ImagePreprocessor:
    """Line image -> preprocessed PIL image (no tensor conversion).

    Same geometry as the encoder's own preprocessing -- it is literally the same
    function since phase 3 -- but returns PIL, because this pipeline writes the crops
    out as image files rather than feeding them to a model.
    """

    def __init__(self, image_height: int, max_image_width: int, patch_size: int):
        assert image_height % patch_size == 0, "Image height should be a multiple of patch size"
        self.image_height = image_height
        self.max_image_width = max_image_width
        self.patch_size = patch_size

    def _transform(self, img: Image.Image) -> Image.Image:
        return resize_line_image(img, self.image_height, self.max_image_width,
                                 self.patch_size, out="pil")

    def __call__(self, img: Image.Image) -> Image.Image:
        return self._transform(img)


# --------------------------------------------------------------------------------------
# For every text line: render it with a random (language-appropriate) font + size, then
# preprocess to the encoder's grayscale height-64 format. The preprocessed PIL image
# overwrites the `image` column; `font` / `font_size` are kept as metadata.
# --------------------------------------------------------------------------------------
preprocessor = ImagePreprocessor(image_height, max_image_width, patch_size)


def create_image_for_sample(sample, font_pool, sanskrit_pool):
    if sample['language'] == 'sanskrit':
        chosen_font = random.choice(sanskrit_pool)
    else:
        chosen_font = random.choice(font_pool)
    font_path = f"{FONTS_DIR}/{chosen_font}.ttf"
    chosen_font_size = random.choice(FONT_SIZES)

    img = generate_image(sample['text'], font_path, chosen_font_size)
    sample['image'] = preprocessor(img)      # grayscale, height 64, padded; no tensor
    sample['font'] = chosen_font
    sample['font_size'] = chosen_font_size
    return sample


# Hold out VAL_SIZE lines for validation; the rest are training. Validation images are
# rendered ONLY with the reserved fonts (unseen during training); training images use
# the remaining fonts. Font pools are passed per-split via fn_kwargs.
VAL_SIZE = 5_000
_split = result.train_test_split(test_size=VAL_SIZE, seed=seed, shuffle=True)


# Pin the output schema so the generated image is written as compact encoded bytes
# (the `datasets` Image feature) instead of being buffered as raw Python PIL objects.
def image_output_features(ds):
    feats = ds.features.copy()
    feats["image"] = HF_Image()
    feats["font"] = Value("string")
    feats["font_size"] = Value("int64")
    return feats


# --------------------------------------------------------------------------------------
# Chunked generation + upload. A single push_to_hub of the whole image dataset stalls, so
# instead we process the train split in fixed-size chunks and upload each chunk as its own
# config (subset) on the Hub; the small validation split is uploaded once. Each chunk is
# generated, pushed, then freed so peak RAM/disk stays bounded. Already-uploaded configs
# are skipped, making the run resumable after an interruption.
#
# Layout on the Hub `REPO`:
#   config "validation"   -> split "validation"   (VAL_SIZE rows, reserved fonts)
#   config "train_0000"   -> split "train"        (CHUNK_SIZE rows)
#   config "train_0001"   -> split "train"        ...
# To consume all training data later, concatenate the "train_*" configs.
# --------------------------------------------------------------------------------------
import gc
from datasets import get_dataset_config_names

REPO = "harsha-desaraju/synthetic-line-text-images"
CHUNK_SIZE = 100_000
MAX_SHARD_SIZE = "500MB"

# writer_batch_size is kept small: each buffered row holds an encoded image, so the
# default (1000) x num_proc workers is what spikes RAM at the end of the map.
TRAIN_WRITER_BATCH = 500
VAL_WRITER_BATCH = 100

# Existing configs on the Hub -> skip them so a re-run resumes where it left off.
try:
    existing_configs = set(get_dataset_config_names(REPO))
except Exception:
    existing_configs = set()   # repo doesn't exist yet / not created
print(f"Existing configs on {REPO}: {sorted(existing_configs) or 'none'}")

# ---- validation (small, uploaded once as its own config) ----
if "validation" not in existing_configs:
    val_ds = _split["test"].map(
        create_image_for_sample, num_proc=num_proc, writer_batch_size=VAL_WRITER_BATCH,
        fn_kwargs={"font_pool": val_fonts, "sanskrit_pool": val_fonts},
        features=image_output_features(_split["test"]),
    )
    print(f"[upload] config 'validation' ({len(val_ds)} rows) ...", flush=True)
    val_ds.push_to_hub(REPO, config_name="validation", split="validation",
                       max_shard_size=MAX_SHARD_SIZE)
    del val_ds
    gc.collect()
else:
    print("[skip] config 'validation' already uploaded")

# ---- train (chunked: generate -> upload -> free) ----
train_split = _split["train"]
n_train = len(train_split)
n_chunks = (n_train + CHUNK_SIZE - 1) // CHUNK_SIZE
print(f"Train: {n_train} rows -> {n_chunks} chunks of {CHUNK_SIZE}")

for c in range(n_chunks):
    config_name = f"train_{c:04d}"
    if config_name in existing_configs:
        print(f"[skip] config '{config_name}' already uploaded")
        continue

    start = c * CHUNK_SIZE
    end = min(start + CHUNK_SIZE, n_train)
    chunk = train_split.select(range(start, end))

    chunk = chunk.map(
        create_image_for_sample, num_proc=num_proc, writer_batch_size=TRAIN_WRITER_BATCH,
        fn_kwargs={"font_pool": train_fonts, "sanskrit_pool": train_sanskrit_fonts},
        features=image_output_features(train_split),
    )

    print(f"[upload] config '{config_name}' (rows {start}:{end}) ...", flush=True)
    chunk.push_to_hub(REPO, config_name=config_name, split="train",
                      max_shard_size=MAX_SHARD_SIZE)

    del chunk
    gc.collect()

print("[done] all chunks uploaded.", flush=True)