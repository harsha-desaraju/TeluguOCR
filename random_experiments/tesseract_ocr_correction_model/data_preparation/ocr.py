
import os
import json
from typing import List
import pytesseract
from PIL import Image
from pathlib import Path
from joblib import Parallel, delayed



class TesseractOCR:
    def __init__(self, img_paths: List[str], lang: str, n_jobs: int = 1):
        self.n_jobs = n_jobs
        self.lang = lang
        self.img_paths = img_paths

    @staticmethod
    def _perform_tesseract_ocr(img_path: str, lang: str, src_text_dct: str):
        try:
            image = Image.open(img_path)
            ocr_text = pytesseract.image_to_string(image, lang=lang)

            with open(src_text_dct, 'r') as f:
                dct = json.load(f)
            
            dct['ocr_text'] = ocr_text

            with open(src_text_dct, 'w') as f:
                json.dump(dct, f, ensure_ascii=False)
        except Exception as e:
            print(img_path)
            # print(e)
    

    def extract_text(self, src_text_path: str):
        text_paths = []
        for img_path in self.img_paths:
            img_name = Path(img_path).name
            _, doc_id, chunk_id = img_name.split('.')[0].split('_')
            src_text_dct = f"{src_text_path}/text_{doc_id}_{chunk_id}.json"
            text_paths.append(src_text_dct)

        with Parallel(n_jobs=self.n_jobs) as parallel:
            parallel([delayed(self._perform_tesseract_ocr)(img_pth, self.lang, txt_pth) for img_pth, txt_pth in zip(self.img_paths, text_paths)])

            




    



    


if __name__ == '__main__':

    fol_path = "correction_model/test_data/generated/source_texts"
    files = [file for file in os.listdir(fol_path) if file != '.DS_Store']

    done_files, not_done_files = [], []

    for file in files:
        with open(f"{fol_path}/{file}", 'r') as f:
            dct = json.load(f)

            if not 'ocr_text' in dct:
                not_done_files.append(file)
            else:
                done_files.append(file)

    images_path = "correction_model/test_data/generated/images"

    not_done_image_paths = []
    for file in not_done_files:
        _, doc_id, chunk_id = file.split('.')[0].split('_')
        image_name = f"image_{doc_id}_{chunk_id}.png"
        img_pth = f"{images_path}/{image_name}"
        not_done_image_paths.append(img_pth)

    print("No. of images to OCR: ", len(not_done_image_paths))
    
    text_detector = TesseractOCR(
        img_paths=not_done_image_paths,
        lang='tel+eng',
        n_jobs=1
    )

    text_detector.extract_text(src_text_path='correction_model/test_data/generated/source_texts')