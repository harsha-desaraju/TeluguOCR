import pandas as pd
import html
from playwright.sync_api import sync_playwright
import base64
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
from pathlib import Path
from typing import List, Dict
import os
import json
import re


class ImageGenerator:
    '''
    Generates images from the given texts using a random font and random text length.
    Stores the images and corresponding image generation information in the given path.

    Args:
        df: Dataframe containing a column `text` (rest are ignored)
        fonts_path: path to directory containing .ttf files of fonts to be used
        output_folder_path: path to save the images and image generation information to
        dpi: the dpi of the image ot be generated
        n_jobs: no. of processes to run parallely
        font_size: size of the font to use
        random_state: random state for randomly choosing chunk size and font
    '''

    def __init__(self, texts: List[str], index: int, fonts_path: str, output_folder_path: str, dpi: int = 300,
                 n_jobs: int = 4, font_size: int = 16, random_state: int = None):
        np.random.seed(random_state)

        self.texts = texts

        self.ids = [i for i in range(index+1, index + len(texts))]

        # Create folders to store images and source text
        self.images_path = f"{output_folder_path}/images"
        self.source_text_path = f"{output_folder_path}/source_texts"

        os.makedirs(output_folder_path, exist_ok=True)
        os.makedirs(self.images_path, exist_ok=True)
        os.makedirs(self.source_text_path, exist_ok=True)

        self.fonts = self._load_fonts(fonts_path)

        self.font_size = font_size
        self.n_jobs = n_jobs
        self.dpi = dpi

    @staticmethod
    def _load_fonts(fonts_path: str):
        ''' Loads fonts. Only accepts fonts of format .ttf, .otf, .woff, .woff2 '''
        font_paths = []
        valid_exts = ['.ttf', '.otf', '.woff', '.woff2']
        for path in Path(fonts_path).rglob('*'):
            if path.suffix.lower() in valid_exts:
                font_paths.append(path)

        fonts = []
        for font_path in font_paths:
            font_name = font_path.name
            with open(font_path, 'rb') as f:
                font_data = base64.b64encode(f.read()).decode('utf-8')

            # Determine font format
            font_format = 'truetype'
            if font_path.suffix.lower() == '.otf':
                font_format = 'opentype'
            elif font_path.suffix.lower() == '.woff':
                font_format = 'woff'
            elif font_path.suffix.lower() == '.woff2':
                font_format = 'woff2'

            font_css = f"""
                @font-face {{
                    font-family: 'CustomFont';
                    src: url(data:font/{font_format};base64,{font_data}) format('{font_format}');
                }}
            """
            fonts.append((font_css, font_name))
        return fonts

    @staticmethod
    def text_to_html(text: str):
        """
        Convert plain text into HTML-safe text that preserves: newlines and tabs
        """
        # 1. Escape HTML special characters
        safe_text = html.escape(text)
        return safe_text

    @staticmethod
    def text_to_image(text: str, output_path: str, font_css: str, font_size: int, dpi: int = 300,
                      bg_color: str = "white", text_color: int = "black", padding: int = 20,
                      src_text_dct: Dict = None, src_text_path: str = None):
        """
        Create an image from text using Playwright.

        Args:
            text: The text to render
            output_path: Path where the image will be saved
            font_css: css string for the font
            font_size: Font size in pixels
            bg_color: Background color (CSS color value)
            text_color: Text color (CSS color value)
            padding: Padding around text in pixels
        """

        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                {font_css}

                body {{
                    margin: 0;
                    padding: 0;
                    background: {bg_color};
                }}
                .text-container {{
                    display: inline-block;
                    font-family: 'CustomFont', sans-serif;
                    font-size: {font_size}px;
                    color: {text_color};
                    padding: {padding}px;
                    white-space: pre-wrap;
                }}
            </style>
        </head>
        <body>
            <div class="text-container">{text}</div>
        </body>
        </html>
        """

        # Create and save the image
        scale = dpi / 96
        with sync_playwright() as p:
            browser = p.chromium.launch()
            context = browser.new_context(device_scale_factor=scale)
            page = context.new_page()
            page.set_content(html_content)

            # Take screenshot of just the text element
            element = page.locator(".text-container")
            element.screenshot(path=output_path)
            browser.close()

        # Store the src_text
        if src_text_dct and src_text_path:
            with open(src_text_path, 'w') as f:
                json.dump(src_text_dct, f, ensure_ascii=False)

    def generate_images(self):
        args = []
        for i, img_id in enumerate(self.ids):
            raw_text = self.texts[i]
            text = self.text_to_html(raw_text)
            font_css, font_name = self.fonts[np.random.randint(0, len(self.fonts))]
            img_path = f"{self.images_path}/image_{img_id}.png"
            text_path = f"{self.source_text_path}/text_{img_id}.json"
            ground_truth = {
                "src_text": raw_text, "font": font_name,
                "image_id": f"image_{img_id}.png",
            }
            args.append({
                'text': text, 'output_path': img_path, 'font_css': font_css,
                'font_size': self.font_size, 'dpi': self.dpi,
                'src_text_dct': ground_truth, 'src_text_path': text_path})

        with Parallel(n_jobs=self.n_jobs) as parallel:
            parallel([delayed(self.text_to_image)(**dct) for dct in args])


def preprocess_text(txt):
    non_telugu = re.compile(r"[^\u0C00-\u0C7F0-9\s.,!?;:'\"()\[\]{}\-\–—_+=/@#₹%&*<>|\\~`]")
    txt = non_telugu.sub("", txt)
    return txt


def remove_padding(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    h_sum = np.mean(bin_img, axis=1)
    v_sum = np.mean(bin_img, axis=0)

    xl, xr = 0, len(h_sum) - 1
    while h_sum[xl] == 255.0:
        xl += 1
    while h_sum[xr] == 255.0:
        xr -= 1

    yt, yb = 0, len(v_sum) - 1
    while v_sum[yt] == 255.0:
        yt += 1
    while v_sum[yb] == 255.0:
        yb -= 1

    return bin_img[xl:xr, yt:yb]


if __name__ == '__main__':

    # files = [
    #     '/Users/xai/Personal/Projects/Datasets/text_dump/annamayya_keertanalu.txt',
    #     '/Users/xai/Personal/Projects/Datasets/text_dump/krithis.txt',
    #     '/Users/xai/Personal/Projects/Datasets/text_dump/sataka_padyalu.txt',
    #     '/Users/xai/Personal/Projects/Datasets/text_dump/telugu_old_newspapers.txt',
    #     '/Users/xai/Personal/Projects/Datasets/text_dump/telugu_movie_songs.txt',
    #     '/Users/xai/Personal/Projects/Datasets/text_dump/wiki1.txt',
    # ]

    # all_words = set()
    #
    # for file in files:
    #     with open(file, 'r') as f:
    #         full_txt = f.read()
    #
    #     full_txt = preprocess_text(full_txt)
    #     words = set(full_txt.split())
    #     all_words = all_words.union(words)
    #
    #     print(len(all_words))
    #
    # all_words = sorted(list(all_words))
    # all_words = [word for word in all_words if len(word)<30]
    #
    # with open(f"vocab.txt", 'w') as f:
    #     for word in all_words:
    #         f.write(word+'\n')
    #
    # filtered_vocab = [all_words[i] for i in range(len(all_words)) if i%5==0]
    #
    # with open(f"small_vocab.txt", 'w') as f:
    #     for word in filtered_vocab:
    #         f.write(word+'\n')

    with open("small_vocab.txt", 'r') as f:
        words = f.readlines()
        words = [word[:-1] for word in words]


    words = words[100000:115000]

    generator = ImageGenerator(
        texts=words,
        index=100000,
        fonts_path="/Users/xai/Personal/Projects/TeluguOCR/correction_model/assets/fonts",
        output_folder_path="/Users/xai/Personal/Projects/TeluguOCR/image_encoder/words",
        font_size=22,
        dpi=300,
        random_state=42,
        n_jobs=-1,
    )
    generator.generate_images()


