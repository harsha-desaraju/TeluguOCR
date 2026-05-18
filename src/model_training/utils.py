import math
import torch
import pytesseract
import numpy as np
from datasets import load_dataset
from pytesseract import Output
from PIL import Image
import torchvision.transforms.functional as F
from torch.utils.data import IterableDataset
from typing import Literal
import itertools
from src.model_training.image_augmentation import scanned_word_augmentation


import random

class TrOCRStyleImageProcessor:
    def __init__(
        self,
        target_width = 256,
        target_height = 64,
        interpolation=Image.BICUBIC,
        rescale_factor=1.0 / 255.0,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        pad_value=0,
        device="cpu"
    ):
        self.target_width = target_width
        self.target_height = target_height
        self.interpolation = interpolation
        self.rescale_factor = rescale_factor
        self.image_mean = image_mean
        self.image_std = image_std
        self.pad_value = pad_value
        self.device = device

    def resize_preserve_aspect(self, image):
        w, h = image.size
        scale = min(self.target_width/w, self.target_height/h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return image.resize((new_w, new_h), self.interpolation)

    def pad_to_target_size(self, image):
        w, h = image.size
        pad_w = self.target_width - w
        pad_h = self.target_height - h

        padding = [0, 0, pad_w, pad_h]  # left, top, right, bottom

        return F.pad(image, padding, fill=self.pad_value)

    def __call__(self, image):
        """
        image: PIL.Image (RGB or grayscale)
        returns: torch.FloatTensor (3, target_size, target_size)
        """
        try:
            # Convert grayscale → RGB
            if image.mode != "RGB":
                image = image.convert("RGB")
            # image = image.convert("L")

            # 1. Resize (aspect ratio preserved)
            image = self.resize_preserve_aspect(image)

            # 2. Pad to square
            image = self.pad_to_target_size(image)

            # 3. To tensor (C, H, W) in [0, 1]
            image = F.to_tensor(image)

            # 4. Rescale (redundant if using to_tensor, kept for clarity)
            image = image * (self.rescale_factor * 255.0)

            # 5. Normalize → [-1, 1] by default
            image = F.normalize(image, self.image_mean, self.image_std)

            return image.to(self.device)

        except Exception as e:
            print(f"Ran into error... but handling...")
            return torch.zeros(size=(3, self.target_height, self.target_width))







def detect_text_boxes(image: Image.Image, min_dimension: int = 5, split: Literal['train', 'test'] = "test"):
    img = np.array(image)

    data = pytesseract.image_to_data(
        img,
        lang="tel",
        output_type=Output.DATAFRAME
    )

    # Take only word bboxes
    data = data[data['level']==5].iloc[:, 6:10]

    word_images = []
    for row in data.itertuples():
        if min(row.width, row.height) > min_dimension:
            x1, x2 = row.left, row.left+row.width
            y1, y2 = row.top, row.top +row.height
            word_img = img[y1:y2, x1:x2]
            # Apply the image augmentation for training
            if split == 'train':
                word_img = scanned_word_augmentation(word_img)
            word_images.append(Image.fromarray(word_img))
    return word_images




def create_batch(image_list):
    """Creates a batched tensor from a list of images"""

    processor = TrOCRStyleImageProcessor(target_width=128, target_height=32)

    batch = []
    for image in image_list:
        batch.append(processor(image))

    batch = torch.stack(batch)
    return batch


class ImageDataset(IterableDataset):
    def __init__(self, split: Literal["train", "test"], num_images_per_epoch: int = 10):
        self.split = split
        self.num_images_per_epoch = num_images_per_epoch
        self.data_files = {
            'train': [
                'data/split_[0-5]-00000-of-00001.parquet'
            ],
            'test': [
                'data/split_88-00000-of-00001.parquet'
            ]
        }

        # Don't load dataset in __init__ to avoid pickle issues with multiprocessing
        self.dataset = None

    def _load_dataset(self):
        """Lazy load dataset in each worker process"""
        if self.dataset is None:
            self.dataset = load_dataset(
                "harsha-desaraju/Telugu-book-text-images",
                split=self.split,
                columns=["image"],
                data_files=self.data_files,
            )

    def __iter__(self):
        # Load dataset in worker process
        self._load_dataset()

        # Get worker info for data sharding
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single-process data loading
            start_idx = 0
            end_idx = self.num_images_per_epoch
            worker_id = 0
            num_workers = 1
        else:
            # Multi-process data loading
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

            # Split workload across workers
            per_worker = int(math.ceil(self.num_images_per_epoch / num_workers))
            start_idx = worker_id * per_worker
            end_idx = min(start_idx + per_worker, self.num_images_per_epoch)

        # Skip to this worker's portion and iterate
        iterator = iter(self.dataset)

        # Skip samples for other workers
        for _ in range(start_idx):
            try:
                next(iterator)
            except StopIteration:
                return

        # Process this worker's samples
        num_samples = end_idx - start_idx
        for sample in itertools.islice(iterator, num_samples):
            image = sample['image']

            word_images = detect_text_boxes(image, split=self.split)

            for word_img in word_images:
                yield word_img

        # # Only print from worker 0 to avoid spam
        # if worker_id == 0:
        #     print(f"Finished running...")






# class ImageDataset(IterableDataset):
#     def __init__(self, split: Literal["train", "test"], num_images_per_epoch: int = 10):
#
#         self.dataset = load_dataset(
#             "harsha-desaraju/Telugu-book-text-images",
#             split=split,
#             columns=["image"],
#             data_files={
#                 'train': [
#                     'data/split_[0-2]-00000-of-00001.parquet'
#                 ],
#                 'test': [
#                     'data/split_88-00000-of-00001.parquet'
#                 ]
#             },
#         )
#         self.num_images_per_epoch = num_images_per_epoch
#
#
#     def __iter__(self):
#         iterator = iter(self.dataset)
#
#         for sample in itertools.islice(iterator, self.num_images_per_epoch):
#             image = sample['image']
#
#             word_images = detect_text_boxes(image)
#
#             for word_img in word_images:
#                 yield word_img
#
#         print(f"Finished running...")
#         return






# class ImageDataset(IterableDataset):
#     def __init__(self, split: Literal["train", "test"], num_images_per_epoch: int = 10):
#         # import pyarrow.dataset
#         #
#         # fragment_scan_options = pyarrow.dataset.ParquetFragmentScanOptions(
#         #     cache_options=pyarrow.CacheOptions(
#         #         prefetch_limit=1,
#         #         range_size_limit=128 << 20
#         #     ),
#         # )
#
#         self.dataset = load_dataset(
#             "harsha-desaraju/Telugu-book-text-images",
#             split=split,
#             # streaming=True,
#             columns=["image"],
#             # data_files="/Users/xai/.cache/huggingface/datasets/harsha-desaraju___telugu-book-text-images/train-2f965dd3f31fd1e4/0.0.0/1af8208e0c9021e9b706a382f380d3682a83f859/telugu-book-text-images-test.arrow"
#             # fragment_scan_options=fragment_scan_options
#         )
#         self.num_images_per_epoch = num_images_per_epoch
#
#     def __iter__(self):
#         iterator = iter(self.dataset)
#
#         for sample in itertools.islice(iterator, self.num_images_per_epoch):
#             image = sample['image']
#             word_images = detect_text_boxes(image)
#
#             for word_img in word_images:
#                 yield word_img
#                 print(f"yielded...")
#
#         print(f"Finished Running")
#
#         return
#
#     def cleanup(self):
#         """Cleanup dataset resources"""
#         if hasattr(self.dataset, '_ex'):
#             # Close the underlying pyarrow dataset
#             self.dataset._ex = None
#         del self.dataset

# class ImageDataset(Dataset):
#     def __init__(self, images_path: str):
#         from pathlib import Path
#
#         self.image_paths = list(Path(images_path).rglob("*.png"))
#
#
#     def __len__(self):
#         return len(self.image_paths)
#
#
#     def __getitem__(self, item):
#         img = Image.open(self.image_paths[item])
#         return img



