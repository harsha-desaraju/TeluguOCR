"""
Visualize every font in a folder.

Renders a fixed sample of Telugu aksharas + English letters + digits +
special characters with each font found in a folder, and displays each font as
its own matplotlib figure so the fonts can be inspected one at a time.

Text is rendered with PIL (proper Telugu glyph shaping) and shown via
matplotlib's imshow. Fonts that fail to load (corrupt / unsupported) are skipped
with a warning.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# Sample text rendered with every font: English (upper/lower), digits, a broad
# spread of Telugu vowels / consonants / vattulu / conjuncts, and ASCII specials.
SAMPLE_TEXT = r"""ABCDEFGHIJKLMNOPQRSTUVWXYZ
abcdefghijklmnopqrstuvwxyz

0123456789

౦  ౧  ౨  ౩  ౪  ౫  ౬  ౭  ౮  ౯

అ ఆ ఇ ఈ ఉ ఊ ఋ ౠ ఌ ౡ ఎ ఏ ఐ ఒ ఓ ఔ అం అః

క ఖ గ ఘ ఙ    చ ఛ జ ఝ ఞ    ట ఠ డ ఢ ణ
త థ ద ధ న    ప ఫ బ భ మ    య ర ఱ ల ళ వ
శ ష స హ    క్ష ఱ

కా కి కీ కు కూ కృ కౄ కె కే కై కొ కో కౌ కం కః

క్క క్ట క్త క్న క్ర క్ల క్వ    గ్గ గ్న గ్ర    చ్చ జ్ఞ
ట్ట డ్డ ణ్ణ    త్త ద్ద న్న    ప్ప బ్బ మ్మ
య్య ర్ర ల్ల వ్వ    శ్శ ష్ష స్స హ్హ
క్ష జ్ఞ శ్ర త్ర ద్ర స్త స్థ స్క స్మ స్వ


! " # $ % & ' ( ) * + , - . / : ; < = > ? @ [ \ ] ^ _ ` { | } ~"""

FONT_EXTENSIONS = {".ttf", ".otf", ".ttc"}


def render_font_image(text, font_path, font_size, width, margin, line_spacing):
    """Render one font's sample onto a white PIL image (returns None on failure)."""
    lines = text.split("\n")

    try:
        font = ImageFont.truetype(str(font_path), font_size)
    except Exception as e:  # corrupt / unsupported font file
        print(f"  ! skipping {font_path.name}: {e}")
        return None

    ascent, descent = font.getmetrics()
    line_height = ascent + descent + line_spacing
    height = margin + line_height * len(lines) + margin

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    y = margin
    for line in lines:
        draw.text((margin, y), line, font=font, fill="black")
        y += line_height

    return img


def visualize_fonts(
    folder_path,
    save_dir=None,
    font_size=34,
    width=1600,
    margin=30,
    line_spacing=12,
):
    """Render every font in `folder_path`, each in its own matplotlib figure.

    If `save_dir` is given, each figure is also saved there as <font_stem>.png.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder}")

    font_paths = sorted(
        p for p in folder.rglob("*") if p.suffix.lower() in FONT_EXTENSIONS
    )
    if not font_paths:
        raise FileNotFoundError(f"No font files ({sorted(FONT_EXTENSIONS)}) in {folder}")

    print(f"Found {len(font_paths)} font file(s) in {folder}")

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for path in font_paths:
        print(f"  rendering {path.name}")
        img = render_font_image(
            SAMPLE_TEXT, path, font_size, width, margin, line_spacing
        )
        if img is None:
            continue

        fig, ax = plt.subplots(figsize=(width / 150, img.height / 150))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(path.stem, fontsize=12)
        fig.tight_layout()

        if save_dir is not None:
            fig.savefig(save_dir / f"{path.stem}.png", dpi=150, bbox_inches="tight")

        plt.show()
        plt.close(fig)




if __name__ == "__main__":
    folder_path = "/Users/xai/Personal/Projects/TeluguOCR/data_curation/text_line_images/fonts"
    save_dir = None  # set to a folder path to also save each figure as a PNG

    visualize_fonts(folder_path, save_dir=save_dir)
