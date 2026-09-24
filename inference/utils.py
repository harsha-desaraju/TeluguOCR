
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import TypeAlias



VALID_IMAGE_TYPES: TypeAlias = Path | str | Image.Image | np.ndarray


def crop_image(image: Image.Image, bboxes: list[tuple[float | int, float | int, float | int, float | int]]):
    """Crop image based on the bounding boxes provided"""

    cropped_images = []

    for bbox in bboxes:
        cimg = image.crop(bbox)
        cropped_images.append(cimg)

    return cropped_images


def read_image(image: VALID_IMAGE_TYPES):
    """Read image and return PIL image"""
    if isinstance(image, Path) or isinstance(image, str):
        image = Image.open(str(image))
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        image = image
    else:
        raise TypeError(f"Got image of unexpected type. Expected one of {VALID_IMAGE_TYPES}. Got {type(image)}")
    return image


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    else:
        return "cpu"