
from PIL import Image
from pydantic import BaseModel, ConfigDict



class Line(BaseModel):
    id: int
    bbox: tuple[int|float, int|float, int|float, int|float]
    text: str


class LineInfo(BaseModel):
    id: int
    bbox: tuple[int|float, int|float, int|float, int|float]


class DetectorOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image: Image.Image
    detected_lines: list[LineInfo]


class OCROutput(BaseModel):
    lines: list[Line]