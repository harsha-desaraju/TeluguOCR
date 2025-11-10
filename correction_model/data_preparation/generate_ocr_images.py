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




def preprocess_text(text):
    # For wikipedia articles
    text = re.sub(r'\[[0-9]+]', '', text)
    # For news articles
    text = re.sub(r'<DOCNO>(.*?)</DOCNO>', '', text)
    text = re.sub(r'<DOC>|</DOC>', '', text)
    text = re.sub(r'<TEXT>|</TEXT>', '', text)
    text = re.sub(r'([0-9]{2}-){2}[0-9]{4} ([0-9]{2}:){2}[0-9]{2}', '', text)
    text = re.sub(r'\n\n+', '\n\n', text).strip()
    return text





class ImageGenerator:
    '''
    Generates images from the given texts using a random font and random text length.
    Stores the images and corresponding image generation information in the given path.

    Args:
        df: Dataframe containing a column `text` (rest are ignored)
        fonts_path: path to directory containing .ttf files of fonts to be used
        output_folder_path: path to save the images and image generation information to
        dpi: the dpi of the image ot be generated
        max_num_images: maximum number fo images to genrate from the given texts
        preprocessor: The preprocessing function to apply to the text before generating image
        n_jobs: no. of processes to run parallely
        font_size: size of the font to use
        max_chunk_size: maximum number of words to be written to the image
        random_state: random state for randomly choosing chunk size and font
    '''
    def __init__(self, texts: List[str], fonts_path: str, output_folder_path: str, dpi: int = 300,
                 max_num_images: int = None, preprocessor: callable = None, n_jobs: int = 4, 
                 font_size: int = 16, max_chunk_size: int = 150, random_state: int = None):
        np.random.seed(random_state)
    
        # Preprocess the text
        if preprocessor:
            texts = [preprocessor(text) for text in texts]

        # Create folders to store images and source text
        self.images_path = f"{output_folder_path}/images"
        self.source_text_path = f"{output_folder_path}/source_texts"

        os.makedirs(output_folder_path, exist_ok=True)
        os.makedirs(self.images_path, exist_ok=True)
        os.makedirs(self.source_text_path, exist_ok=True)
        existing_files = [file for file in os.listdir(self.images_path) if file != '.DS_Store']
        if existing_files:
            last_doc_id = sorted([int(file.split('_')[1]) for file in existing_files])[-1]
        else:
            last_doc_id = 0

        # Create chunks of the text
        self.texts, self.doc_ids, self.chunk_ids = [], [], []
        for i, text in enumerate(texts, last_doc_id+1):
            chunked_text, doc_id, chunk_id = self._create_chunks(text, i, max_chunk_size)
            self.texts += chunked_text
            self.doc_ids += doc_id
            self.chunk_ids += chunk_id

        if max_num_images:
            self.texts = self.texts[:max_num_images]
            self.doc_ids = self.doc_ids[:max_num_images]
            self.chunk_ids = self.chunk_ids[:max_num_images]

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
    def _create_chunks(text: str, doc_id: int, max_chunk_size: int):
        tokens = text.split(' ')

        if len(tokens) <= max_chunk_size:
            return [text], [doc_id], [1]
        else:
            chunks, chunk_id = [], []
            i = 0
            while i*max_chunk_size < len(tokens):
                chunk = " ".join(tokens[i*max_chunk_size:(i+1)*max_chunk_size])
                chunks.append(chunk)
                i += 1
                chunk_id.append(i)
            return chunks, [doc_id]*len(chunks), chunk_id


    @staticmethod
    def text_to_html(text: str):
        """
        Convert plain text into HTML-safe text that preserves: newlines and tabs
        """
        # 1. Escape HTML special characters
        safe_text = html.escape(text)
        
        # 2. Replace tabs with 4 non-breaking spaces
        safe_text = safe_text.replace("\t", "&nbsp;&nbsp;&nbsp;&nbsp;")
    
        # 4. Replace newlines with <br> tags
        safe_text = safe_text.replace("\n", "<br>")

        return safe_text
    

    @staticmethod
    def text_to_image(text: str, output_path: str, font_css: str, font_size: int, dpi: int=300,
                        bg_color: str="white", text_color: int="black", padding: int=20,
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
        scale = dpi/96
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
        for i in range(len(self.texts)):
            raw_text = self.texts[i]
            doc_id, chunk_id = self.doc_ids[i], self.chunk_ids[i]
            text = self.text_to_html(raw_text)
            font_css, font_name = self.fonts[np.random.randint(0, len(self.fonts))]
            img_path = f"{self.images_path}/image_{doc_id}_{chunk_id}.png"
            text_path = f"{self.source_text_path}/text_{doc_id}_{chunk_id}.json"
            ground_truth  = {
                "src_text": raw_text, "font": font_name, 
                "image_id": f"image_{doc_id}_{chunk_id}.png",
            }
            args.append({
                'text': text, 'output_path': img_path, 'font_css': font_css, 
                'font_size': self.font_size, 'dpi': self.dpi, 
                'src_text_dct': ground_truth, 'src_text_path': text_path})

        with Parallel(n_jobs=self.n_jobs) as parallel:
            parallel([delayed(self.text_to_image)(**dct) for dct in args])








if __name__ == '__main__':

    df = pd.read_parquet("/Users/xai/Personal/Projects/Datasets/text_dump/collated_data-0.parquet")
    texts = df['text'].tolist()

    texts = texts[:50000]

    generator = ImageGenerator(
        texts=texts,
        fonts_path="correction_model/assets/fonts",
        output_folder_path="correction_model/data/generated/",
        preprocessor=preprocess_text,
        font_size=18,
        # max_num_images = 50,
        dpi=96,
        random_state=42,
        n_jobs=-1,
        max_chunk_size=150
    )
    generator.generate_images()
