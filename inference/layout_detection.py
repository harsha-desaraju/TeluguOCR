import cv2
import tesserocr
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from .models import LineInfo, DetectorOutput
from .deskew_utils import determine_skew, rotate_image
from .utils import VALID_IMAGE_TYPES, read_image


class TextDetector:
    def __init__(self, tessdata_path: str = "/opt/homebrew/opt/tesseract/share/tessdata",
                 lang: str = 'tel', tesseract_mode = tesserocr.PSM.AUTO):
        self.tessdata_path = tessdata_path
        self.lang = lang
        self.tesseract_mode = tesseract_mode

    @staticmethod
    def preprocess_for_tesseract(image: Image.Image):
        """ Preprocess an image for Tesseract OCR. """
        # Convert to grayscale
        if image.mode != 'L':
            gray = np.array(image.convert('L'))
        else:
            gray = np.array(image)

        # Improve local contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Light denoising while preserving character boundaries
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Adaptive thresholding for uneven backgrounds
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11
        )

        return binary


    def detect(self, image: VALID_IMAGE_TYPES, deskew: bool = True, preprocess_image: bool = False, plot_image: bool = False):
        """Detect the text lines in the image"""

        image = read_image(image)

        if preprocess_image:
            image = self.preprocess_for_tesseract(image)

        if deskew:
            if isinstance(image, Image.Image):
                image = np.array(image)
            skew_angle = determine_skew(image)
            image = rotate_image(image, skew_angle)

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        with tesserocr.PyTessBaseAPI(
                path=self.tessdata_path,
                lang=self.lang) as api:

            api.SetPageSegMode(self.tesseract_mode)
            api.SetImage(image)
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
                plot_img = image.copy()
                draw = ImageDraw.Draw(plot_img)

                for line in detected_boxes:
                    draw.rectangle(line.bbox, outline='green', width=2)

                plt.imshow(plot_img)
                plt.show()

            detector_output = DetectorOutput(detected_lines=detected_boxes, image=image)
            return detector_output




if __name__ == '__main__':

    img = Image.open('/Users/xai/Desktop/page1.png')

    detector = TextDetector()
    output = detector.detect(image=img, plot_image=True)

    for line in output.detected_lines:
        print(line.id, line.bbox)



