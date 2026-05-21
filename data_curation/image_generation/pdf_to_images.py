""" Converts a PDF file into images of lines """

import os
import gc
import cv2
import fitz
import pdf2image
import numpy as np
import pytesseract
from pathlib import Path
from PIL import Image
from joblib import Parallel, delayed
from datasets import Dataset, Features, Value, Image as DImage
from huggingface_hub import login
from deskew import determine_skew
from skimage.transform import rotate
from random import choice
from dotenv import load_dotenv

import datasets
datasets.disable_caching()
datasets.disable_progress_bars()



def preprocess_image(img: np.ndarray):
    """Preprocess the image for OCR"""

    # 1 - Convert to gray scale
    image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2 - Determine and correct skew
    angle = determine_skew(image)
    image = rotate(image, angle, resize=True)
    image = (image * 255).astype("uint8")

    # 3 - Denoise
    gray = cv2.GaussianBlur(image, (3, 3), 0)

    # 4 - Adaptive threshold - Binarize the image
    image = cv2.adaptiveThreshold(
        gray,255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,15
    )

    return image




def page_to_line_images(img: np.ndarray, min_width: int, min_height: int, max_width_percent: float, max_height_percent: float):
    """
    splits PDF image to line images based on layout detection using tesseract

    Args:
        img: The image in numpy format
        min_width: The minimum width of line image to be considered in final set
        min_height: The minimum height of the line image to be considered in final set
        max_width_percent: The percentage of page width a line image can have at max (float between 0-1)
        max_height_percent: The percentage of page height a line image can have at max (float between 0-1)

    Returns:
        list of images of text lines meeting the criteria
    """
    img = preprocess_image(img)
    page_height, page_width = img.shape

    max_width = int(max_width_percent * page_width)
    max_height = int(max_height_percent * page_height)

    custom_config = r'--oem 3 --psm 3'
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DATAFRAME, config=custom_config, lang='tel')
    data = data[
        (data['level']==4) &
        (data['width'] >= min_width) &
        (data['width'] <= max_width) &
        (data['height'] > min_height) &
        (data['height'] <= max_height)
    ]

    line_images = []
    for i in data.index:
        line = data.loc[i]
        l_img = img[line['top']: line['top']+line['height'], line['left']: line['left']+line['width']]
        line_images.append(l_img)

    return line_images



def pdf_to_line_images(file_path: Path, out_dir: Path, first_page: int | None = None):

    images = pdf2image.convert_from_path(str(file_path), dpi=IMAGE_DPI[-1], first_page=first_page, thread_count=1)
    images = [np.array(image) for image in images]

    # Save
    file_dir = out_dir / f"{file_path.stem}"
    os.makedirs(file_dir, exist_ok=True)

    first_page = first_page if first_page else 1

    for page_num, image in enumerate(images, first_page):
        line_imgs = page_to_line_images(image, min_width=MIN_WIDTH, min_height=MIN_HEIGHT,
                                        max_width_percent=MAX_WIDTH_PERCENT, max_height_percent=MAX_HEIGHT_PERCENT)
        for line_num,  line_img in enumerate(line_imgs):
            img_path = file_dir / f"{page_num}_{line_num}.{IMAGE_FORMAT}"
            line_img = Image.fromarray(line_img)
            line_img.save(img_path, IMAGE_FORMAT)

    print(f"{file_path.stem} done!", flush=True)


def load_page_lazy(file_path: Path, first_page: int | None = None):
    """Load pages of the PDF lazily, one at a time"""
    doc = fitz.open(file_path)
    start_index = (first_page - 1) if first_page else 0
    try:
        for page in doc[start_index:]:
            pix = page.get_pixmap(dpi=choice(IMAGE_DPI), colorspace=fitz.csRGB)
            page_img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n).copy()
            del pix
            yield page_img
    finally:
        doc.close()


def pdf_to_line_images_hf(file_path: Path, split: int, first_page: int | None = None):
    try:
        file_name = file_path.stem
        first_page = first_page if first_page else 1

        b_num = 1
        ds_buffer = []
        ds_features = Features({
            "line_image": DImage(),
            "file_name": Value("string"),
            "page_number": Value("int64")
        })

        for page_num, page_image in enumerate(load_page_lazy(file_path, first_page), first_page):
            line_images = page_to_line_images(page_image, MIN_WIDTH, MIN_HEIGHT, MAX_WIDTH_PERCENT, MAX_HEIGHT_PERCENT)
            ds_buffer += [
                {"line_image": Image.fromarray(line_img), "file_name": file_name, "page_number": page_num}
                for line_img in line_images
            ]
            print(f"Processing page: {page_num}, Line Images: {len(ds_buffer)}")

            if len(ds_buffer) >= MIN_BATCH_SIZE:
                dataset = Dataset.from_list(ds_buffer, ds_features)
                dataset.push_to_hub(
                    repo_id=HF_REPO_ID,
                    split=f"book-{split}_{b_num}",
                    commit_message=f"Uploaded {file_name}"
                )
                b_num += 1
                del ds_buffer, dataset
                gc.collect()
                ds_buffer = []

        # Upload remaining images
        if ds_buffer:
            dataset = Dataset.from_list(ds_buffer, ds_features)
            dataset.push_to_hub(
                repo_id=HF_REPO_ID,
                split=f"book-{split}_{b_num}",
                commit_message=f"Uploaded {file_name}"
            )
            del ds_buffer, dataset
            gc.collect()

        print(f"Uploaded book-{split}:{file_name} to the HF hub!", flush=True)

    except Exception as e:
        print(f"The following exception occurred while processing {file_path.stem}:\n\n{str(e)}\n\n", flush=True)


IMAGE_DPI = [200, 300]
MIN_WIDTH = 50
MIN_HEIGHT = 10
MAX_WIDTH_PERCENT = 0.95
MAX_HEIGHT_PERCENT = 0.05
IMAGE_FORMAT = "jpeg"
MAX_FILE_SIZE_IN_MB = 20
NUM_JOBS = 8
HF_REPO_ID = "harsha-desaraju/telugu-text-line-images"
MIN_BATCH_SIZE = 5000


if __name__ == '__main__':
    load_dotenv()
    login(os.getenv("HF_TOKEN"))

    pdfs_folder = Path(__file__).parents[2] / "data/pdf_files/free_gurukul"
    pdf_file_paths = list(Path(pdfs_folder).rglob("*.pdf"))

    # Limit the PDFs to files of MAX_FILE_SIZE_IN_MB size
    pdf_file_paths = [pdf_path for pdf_path in pdf_file_paths if pdf_path.stat().st_size/1e6 < MAX_FILE_SIZE_IN_MB][61:]
    print(len(pdf_file_paths))

    # Parallelize the process across the PDFs
    with Parallel(n_jobs=NUM_JOBS) as parallel:
        parallel([delayed(pdf_to_line_images_hf)(pdf_path, i+1, 6) for i, pdf_path in enumerate(pdf_file_paths, 61)])
