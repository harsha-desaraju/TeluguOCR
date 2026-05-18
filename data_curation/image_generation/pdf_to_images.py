""" Converts a PDF file into images of lines """

import os
import pdf2image
import numpy as np
import pytesseract
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from joblib import Parallel, delayed


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

    page_height, page_width, _ = img.shape

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
        l_img = img[line['top']: line['top']+line['height'], line['left']: line['left']+line['width'], :]
        line_images.append(l_img)

    return line_images



def pdf_to_line_images(file_path: Path, out_dir: Path, first_page: int | None = None):

    images = pdf2image.convert_from_path(str(file_path), dpi=IMAGE_DPI, first_page=first_page, thread_count=1)
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



if __name__ == '__main__':

    IMAGE_DPI = 300
    MIN_WIDTH = 50
    MIN_HEIGHT = 10
    MAX_WIDTH_PERCENT = 0.95
    MAX_HEIGHT_PERCENT = 0.05
    IMAGE_FORMAT = "jpeg"
    MAX_FILE_SIZE_IN_MB = 40
    NUM_JOBS = os.cpu_count() - 2

    # pdf_file_path = Path("/Users/xai/Personal/Projects/TeluguOCR/data/pdf_files/free_gurukul/AnjaneyaDandakam_342.pdf")

    output_dir = Path(__file__).parents[2] / "data/images/sanatanadharm"

    pdfs_folder = Path(__file__).parents[2] / "data/pdf_files/sanatanadharm"
    pdf_file_paths = list(Path(pdfs_folder).rglob("*.pdf"))

    # Remove files which are already done
    done_files = os.listdir(output_dir)
    pdf_file_paths = [pdf_path for pdf_path in pdf_file_paths if pdf_path.stem not in done_files]

    # Limit the PDFs to files of MAX_FILE_SIZE_IN_MB size
    pdf_file_paths = [pdf_path for pdf_path in pdf_file_paths if pdf_path.stat().st_size/1e6 < MAX_FILE_SIZE_IN_MB]

    with Parallel(n_jobs=NUM_JOBS) as parallel:
        parallel([delayed(pdf_to_line_images)(pdf_path, output_dir) for pdf_path in pdf_file_paths])

