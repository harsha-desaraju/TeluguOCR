import pandas as pd
import html
from playwright.sync_api import sync_playwright
import base64
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
from pathlib import Path







class ImageGenerator:
    '''
    Generates images from the given texts using a random font and random text length.
    Stores the images and corresponding image generation information in the given path.

    Args:
        df: Dataframe containing a column `text` (rest are ignored)
        fonts_path: path to directory containing .ttf files of fonts to be used
        output_folder_path: path to save the images and image generation information to
        max_num_images: maximum number fo images to genrate from the given texts
        preprocessor: The preprocessing function to apply to the text before generating image
        n_jobs: no. of processes to run parallely
        font_size: size of the font to use
        min_chunk_size: minimum number of words to be written to the image
        max_chunk_size: maximum number of words to be written to the image
        random_state: random state for randomly choosing chunk size and font
    '''
    def __init__(self, df: pd.DataFrame, fonts_path: str, output_folder_path: str, 
                 max_num_images: int = 10000, preprocessor: callable = None, n_jobs: int = 4, 
                 font_size: int = 24, min_chunk_size: int = 5, max_chunk_size: int = 150, random_state: int = None):
        np.random.seed(random_state)
    
        # Preprocess the text
        texts = df['text'].tolist()
        if preprocessor:
            texts = [preprocessor(text) for text in texts]

        # Create chunks of the text
        self.texts = []
        for text in texts:
            self.texts += self._create_chunks(text, min_chunk_size, max_chunk_size)

        self.fonts = self._load_fonts(fonts_path)

        self.output_folder_path = output_folder_path
        self.max_num_images = max_num_images
        self.font_size = font_size
        self.n_jobs = n_jobs
    

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
            fonts.append(font_css)
        return fonts
        
    
    @staticmethod
    def _create_chunks(text: str, min_chunk_size: int, max_chunk_size: int):
        tokens = text.split()
        chunks = []
        i = 0
        while i <= len(tokens):
            num_toks = np.random.randint(min_chunk_size, max_chunk_size+1)
            chunk = " ".join(tokens[i:i+num_toks])
            chunks.append(chunk)
            i += num_toks
        return chunks


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
    def text_to_image(text, output_path, font_css, font_size, 
                         bg_color="white", text_color="black", padding=20):
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
        
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.set_content(html_content)
            
            # Take screenshot of just the text element
            element = page.locator(".text-container")
            element.screenshot(path=output_path)
            browser.close()


    def generate_images(self):
        # limit text for creating images
        inds = np.arange(len(self.texts))
        np.random.shuffle(inds)
        inds = inds[:self.max_num_images]

        args = []
        for i, ind in enumerate(inds):
            text = self.texts[ind]
            font = self.fonts[np.random.randint(0, len(self.fonts)+1)]
            img_path = f"{self.output_folder_path}/image_{i}.png"
            args.append((text, img_path, font, self.font_size))

        with Parallel(n_jobs=self.n_jobs) as parallel:
            parallel([delayed(self.text_to_html)(*tup) for tup in args])







if __name__ == '__main__':

    pass