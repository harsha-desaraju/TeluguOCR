"""Shared helpers for the inference package: image coercion, cropping, device pick."""

import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import TypeAlias


VALID_IMAGE_TYPES: TypeAlias = Path | str | Image.Image | np.ndarray


def crop_image(image: Image.Image,
               bboxes: list[tuple[float | int, float | int, float | int, float | int]]
               ) -> list[Image.Image]:
    """Crop image based on the bounding boxes provided"""
    return [image.crop(bbox) for bbox in bboxes]


def read_image(image: VALID_IMAGE_TYPES) -> Image.Image:
    """Read image and return PIL image"""
    if isinstance(image, (Path, str)):
        return Image.open(str(image))
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    if isinstance(image, Image.Image):
        return image
    raise TypeError(
        f"Got image of unexpected type. Expected one of {VALID_IMAGE_TYPES}. Got {type(image)}")


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.mps.is_available():
        return "mps"
    return "cpu"