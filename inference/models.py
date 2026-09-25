"""Pydantic records passed between the detector and the recognizer."""

from PIL import Image
from pydantic import BaseModel, ConfigDict


class LineInfo(BaseModel):
    id: int
    bbox: tuple[int | float, int | float, int | float, int | float]


class Line(LineInfo):
    text: str


class DetectorOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image: Image.Image
    detected_lines: list[LineInfo]


class OCROutput(BaseModel):
    lines: list[Line]