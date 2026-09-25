"""Text-line detection: Tesseract finds the line boxes, this returns them with the page."""

import os
import cv2
import tesserocr
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from .models import LineInfo, DetectorOutput
from .deskew_utils import determine_skew, rotate_image
from .utils import VALID_IMAGE_TYPES, read_image


TESSDATA_PATH = os.getenv("TESSDATA_PREFIX")

class TextDetector:
    def __init__(self, tessdata_path: str = TESSDATA_PATH,
                 lang: str = 'tel', tesseract_mode=tesserocr.PSM.AUTO) -> None:
        self.tessdata_path = tessdata_path
        self.lang = lang
        self.tesseract_mode = tesseract_mode

    @staticmethod
    def preprocess_for_tesseract(image: Image.Image) -> Image.Image:
        """Binarise for line detection: CLAHE, light blur, adaptive threshold.

        For SetImage ONLY. The recogniser was trained on natural grayscale, so feeding
        it these two-valued pixels costs about a quarter of the characters.
        """
        gray = np.array(image if image.mode == 'L' else image.convert('L'))

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11
        )
        return Image.fromarray(binary)

    @staticmethod
    def _deskew(image: Image.Image) -> Image.Image:
        """Straighten the page, measuring the angle on the natural grayscale.

        Measuring on a binarised page instead yields a 45 degree Hough peak and rotates
        the scan into nothing. `determine_skew` also rejects implausible angles.
        """
        angle = determine_skew(np.array(image))
        if not angle:
            return image
        rotated = Image.fromarray(rotate_image(np.array(image), angle))
        # tesserocr's SetImage branches on image.format, and Image.fromarray leaves it
        # None: on identical pixels that is 18 detected lines instead of 20. Carry the
        # source's format so that deskew=True with nothing to rotate matches deskew=False.
        rotated.format = image.format
        return rotated

    def detect(self, image: VALID_IMAGE_TYPES, deskew: bool = False,
               preprocess_image: bool = False, plot_image: bool = False) -> DetectorOutput:
        """Detect the text lines in the image"""
        page = read_image(image)
        if deskew:
            page = self._deskew(page)

        # Two roles, deliberately two variables: `detect_on` may be binarised, `page` is
        # what the boxes are reported against and what the recogniser will crop.
        detect_on = self.preprocess_for_tesseract(page) if preprocess_image else page

        with tesserocr.PyTessBaseAPI(path=self.tessdata_path, lang=self.lang) as api:
            api.SetPageSegMode(self.tesseract_mode)
            api.SetImage(detect_on)
            api.Recognize()

            iterator = api.GetIterator()
            detected_boxes = []

            idx = 0
            while iterator:
                bbox = iterator.BoundingBox(tesserocr.RIL.TEXTLINE)
                if bbox:
                    detected_boxes.append(LineInfo(id=idx, bbox=bbox))
                    idx += 1

                if not iterator.Next(tesserocr.RIL.TEXTLINE):
                    break

        if plot_image:
            plot_img = page.copy()
            draw = ImageDraw.Draw(plot_img)
            for line in detected_boxes:
                draw.rectangle(line.bbox, outline='green', width=2)
            plt.imshow(plot_img)
            plt.show()

        return DetectorOutput(detected_lines=detected_boxes, image=page)


if __name__ == '__main__':

    img = Image.open('/Users/xai/Desktop/page1.png')

    detector = TextDetector()
    output = detector.detect(image=img, plot_image=True)

    for line in output.detected_lines:
        print(line.id, line.bbox)
