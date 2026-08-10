"""Per-sample dataset transforms used by the training loops.

WHY THESE ARE NOT `ImagePreprocessor`
    Until phase 3 both classes here were also called `ImagePreprocessor`, which made
    five same-named classes across the repo look like one drifted class. They are not.
    `ImagePreprocessor` in preprocess.py does the ENCODER GEOMETRY -- resize to height
    64, aspect-preserving, over-wide cap, pad to a multiple of the downsample. These
    two do NO geometry at all, deliberately: training reads datasets whose crops were
    already written at height 64 by the pipeline that built them, so resizing again
    would be a second resample of an image that is already the right size.

    The trap was that the encdec copy took `(image_height, max_image_width,
    downsample)` in its constructor and stored all three -- then never used them. It
    read as the geometry class and behaved as the tensorizer. Collapsing the five onto
    the geometry version would have silently added a resize to every training step,
    with no error and no shape change, just a slow bleed in accuracy.

    Hence the names: whatever these do, it is not "preprocessing an image".
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# grayscale [0,255] -> float tensor in [-1,1], the range every encoder here expects
_TO_TENSOR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def _augment_then_gray(img: Image.Image, augment_fn) -> torch.Tensor:
    """Optional augmentation on the RGB crop, then grayscale, then tensor.

    Augmentation runs on RGB because the augraphy pipeline expects three channels
    (paper texture, bleed-through and ink colour all assume it); the grayscale
    conversion comes after, so it sees the degraded image rather than degrading a
    single channel.
    """
    if augment_fn is not None:
        arr = augment_fn(np.array(img.convert("RGB")))
        img = Image.fromarray(np.asarray(arr, dtype=np.uint8))
    return _TO_TENSOR(img.convert("L"))


class LineTensorizer:
    """Line image -> float tensor (1, H, W). NO resizing: H is whatever came in.

    Used by the encoder-decoder loops, where the dataset already holds height-64
    crops. `augment_fn` is set on the train split only.
    """

    def __init__(self, augment_fn=None):
        self.augment_fn = augment_fn
        self.to_tensor = _TO_TENSOR

    def _transform(self, img: Image.Image) -> torch.Tensor:
        return _augment_then_gray(img, self.augment_fn)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self._transform(img)


class CTCBatchMapper:
    """HuggingFace `datasets` batch mapper for CTC training. NO resizing.

    Maps a batch dict to {"line_image": [tensor], "target_ids": [[int]]}. Labels are
    tokenized WITHOUT special tokens -- CTC aligns raw grapheme classes against frames
    and has its own blank; a BOS/EOS in the target would be scored as text the image
    does not contain.
    """

    def __init__(self, tokenizer, image_col: str, text_col: str, augment_fn=None):
        self.tokenizer = tokenizer
        self.image_col = image_col
        self.text_col = text_col
        self.augment_fn = augment_fn
        self.to_tensor = _TO_TENSOR

    def _img(self, img: Image.Image) -> torch.Tensor:
        return _augment_then_gray(img, self.augment_fn)

    def __call__(self, batch: dict) -> dict:
        return {
            "line_image": [self._img(im) for im in batch[self.image_col]],
            "target_ids": [self.tokenizer(t, add_special_tokens=False)["input_ids"]
                           for t in batch[self.text_col]],
        }
