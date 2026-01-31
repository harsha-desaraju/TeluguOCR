
import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from pdf2image import convert_from_path
from typing import Union, List
from datasets import Dataset, Features, Value, Image as DImage





class PreprocessImage:
    @staticmethod
    def crop_white_borders(image: np.ndarray, threshold: int = 250, residual_border: float = 0.05):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(image, (61, 61), 61, sigmaY=61)

        mask = blurred_image < threshold
        if not np.any(mask):
            return image

        # Find bounding box of non-white area
        coordinates = np.column_stack(np.where(mask))
        y_min, x_min = coordinates.min(axis=0)
        y_max, x_max = coordinates.max(axis=0)

        h, w = image.shape[:2]
        y_min = max(0, int((1 - residual_border) * y_min))
        x_min = max(0, int((1 - residual_border) * x_min))
        y_max = min(h - 1, int((1 + residual_border) * y_max))
        x_max = min(w - 1, int((1 + residual_border) * x_max))

        image = image[y_min:y_max + 1, x_min:x_max + 1]
        return image


    @staticmethod
    def detect_text_boxes(image: np.ndarray, min_area: int = 100, plot_boxes: bool = False):

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Binarize (invert: text -> white)
        _, thresh = cv2.threshold(
            image, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Merge text into blocks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        max_box_area = 0
        num_words = 0
        text_area = 0

        page_area = image.shape[0] * image.shape[1]

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h

            if area > min_area:
                num_words += 1
                max_box_area = max(max_box_area, (area / page_area)*100)
                text_area += area
                if plot_boxes:
                    cv2.rectangle(image, (x, y), (x + w, y + h), 255, 2)

        text_ratio = (text_area / page_area) * 100

        return image, text_ratio, num_words, max_box_area


class ImageGenerator:
    def __init__(self, file_paths: Union[str, Path, List[str | Path]], skip_first_pages: int = 0,
                 output_folder: Path = None, dpi: int = 300, num_jobs: int = 1,gray_scale: bool = True,
                 residual_border: float = 0.05, min_box_area: int = 100):
        if isinstance(file_paths, str) or isinstance(file_paths, Path):
            file_paths = [file_paths]

        self.file_paths = [Path(file_path) if isinstance(file_path, str) else file_path for file_path in file_paths]

        self.skip_first_pages = skip_first_pages
        self.output_folder = output_folder
        self.dpi = dpi
        self.num_jobs = num_jobs
        self.gray_scale = gray_scale

        self.preprocessor = PreprocessImage()
        self.residual_border = residual_border
        self.min_box_area = min_box_area



    @staticmethod
    def pdf_as_images(file_path: Path, skip_first_pages: int = 0, output_folder: Path = None, dpi: int = 300, num_jobs: int = 1,
                      gray_scale: bool = True):
        images = convert_from_path(file_path, dpi=dpi, output_file=output_folder, thread_count=num_jobs,
                                   grayscale=gray_scale)

        # remove the first few pages (like title or index)
        images = images[skip_first_pages:]

        # Save each image
        file_name = file_path.name
        if output_folder:
            for i, image in enumerate(images):
                output_path = os.path.join(output_folder, f'{file_name}_page_{i}.png')
                image.save(output_path, 'PNG')

        # Convert numpy arrays
        images = [np.array(image) for image in images]
        return images


    def generate(self, min_num_words: int, max_text_ratio: int, max_box_area_ratio: int):
        for file_path in self.file_paths:
            try:
                pdf_sample = []
                try:
                    images = self.pdf_as_images(file_path, skip_first_pages=self.skip_first_pages, output_folder=self.output_folder,
                                                dpi=self.dpi, num_jobs=self.num_jobs, gray_scale=self.gray_scale)
                except Exception as e:
                    print(f"Error while loading {file_path}: {e}")
                    images = []

                # Preprocess the images
                images = [self.preprocessor.crop_white_borders(image, residual_border=self.residual_border) for image in images]

                file_name = file_path.name

                for page_num, image in enumerate(images, 1):
                    image, text_ratio, num_words, max_box_area = self.preprocessor.detect_text_boxes(image, self.min_box_area)

                    if num_words < min_num_words or text_ratio > max_text_ratio or max_box_area > max_box_area_ratio:
                        pass
                    else:
                        pdf_sample.append({
                            "image": Image.fromarray(image),
                            "file_name": file_name,
                            "page_number": page_num
                        })

                yield pdf_sample
            except Exception as e:
                print(f"Failed generating images for file: {file_path.name}\n\n"+str(e))
                yield []


    @staticmethod
    def get_features():
        return Features({
            "image": DImage(),
            "file_name": Value("string"),
            "page_number": Value("int64")
        })




if __name__ == '__main__':

    # Thresholds for selecting or rejecting an image
    MIN_NUM_WORDS = 30
    MAX_TEXT_RATIO = 80
    MAX_BOX_AREA_RATIO = 10

    repo_id = "harsha-desaraju/Telugu-book-text-images"
    split_batch_size = 2500

    pdf_file_paths = list(Path('./downloaded_books').rglob('*.pdf'))
    pdf_file_paths = sorted(pdf_file_paths)

    image_generator = ImageGenerator(file_paths=pdf_file_paths, dpi=200, num_jobs=4)


    buffer = []
    batch_id = 0


    for file_num, pdf_images in enumerate(image_generator.generate(MIN_NUM_WORDS, MAX_TEXT_RATIO, MAX_BOX_AREA_RATIO), 1):
        buffer += pdf_images

        print(f"Finished processing [{file_num}/{len(pdf_file_paths)}] files")

        if len(buffer) >=split_batch_size:
            ds = Dataset.from_list(buffer[:split_batch_size], features=image_generator.get_features())

            ds.push_to_hub(
                repo_id=repo_id,
                split=f"split_{batch_id}",
                commit_message=f"Upload split {batch_id}"
            )

            print(f"✅ Uploaded batch {batch_id}")

            buffer = buffer[split_batch_size:]
            batch_id += 1

    # Upload the remaining images if they are more than 100
    if len(buffer) > 100:
        ds = Dataset.from_list(buffer[:split_batch_size], features=image_generator.get_features())

        ds.push_to_hub(
            repo_id=repo_id,
            split=f"split_{batch_id}",
            commit_message=f"Upload split {batch_id}"
        )

        print(f"✅ Uploaded last batch {batch_id}")
