
import gc
import os
import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from model import ViTConfig, MaskedAutoEncoder
from utils import ImagePreprocessor
from transformers.trainer_utils import get_last_checkpoint

# --------------------------------------------------
# CONFIG
# --------------------------------------------------


# MODEL_NAME = "facebook/vit-mae-base"

BATCH_SIZE = 32
EPOCHS = 100
TEST_SIZE = 0.2

IMAGE_HEIGHT = 64
MAX_IMAGE_WIDTH = 1024
PATCH_SIZE = 8
MASK_RATIO = 0.75





# Datasets
# --------------------------------------------------
# Dataset
# --------------------------------------------------


###### ??????????? Decide the test size ?????????????????????


ds = load_dataset(
    "harsha-desaraju/telugu-book-line-images-sample",
    columns=['line_image']
)["train"]

split_dataset = ds.train_test_split(test_size=TEST_SIZE, seed=42)

train_ds = split_dataset["train"]
test_ds = split_dataset["test"]

del split_dataset
gc.collect()

preprocessor = ImagePreprocessor(IMAGE_HEIGHT, MAX_IMAGE_WIDTH, PATCH_SIZE)

def process_sample(sample):
    sample['line_image'] = preprocessor(sample['line_image'])
    sample['image_width'] = sample['line_image'].shape[-1]
    return sample


train_ds = train_ds.map(process_sample)
test_ds = test_ds.map(process_sample)

train_ds = train_ds.sort('image_width')
test_ds = test_ds.sort('image_width')

train_ds = train_ds.remove_columns(['image_width'])
test_ds = test_ds.remove_columns(['image_width'])


# --------------------------------------------------
# COLLATOR
# --------------------------------------------------


def collator_function(batch):
    # Also think about the device ??

    img_height = 0
    max_batch_width = 0
    for image in batch:
        if image.shape[-1] > max_batch_width:
            max_batch_width = image.shape[2]
            img_height = image.shape[1]

    num_channels = batch[0].shape[0]

    zeros_img = batch[0].new_zeros((num_channels, img_height, max_batch_width))

    padded_images, padding_masks = [], []
    for image in batch:
        img_h, img_w = image.shape[1], image.shape[2]
        h_p, w_p = img_h // PATCH_SIZE, img_w // PATCH_SIZE

        pad_len = max_batch_width - img_w
        padded_img = torch.cat([image, zeros_img[:, :, :pad_len]], dim=2)

        pad_mask = torch.ones((h_p, w_p + pad_len // PATCH_SIZE), dtype=torch.bool)
        pad_mask[:, :w_p] = False

        padded_images.append(padded_img.unsqueeze(0))
        padding_masks.append(pad_mask.flatten().unsqueeze(0))

    padded_images = torch.cat(padded_images, dim=0)
    padding_masks = torch.cat(padding_masks, dim=0)

    return {
        "images": padded_images,
        "padding_masks": padding_masks
    }

# --------------------------------------------------
# MODEL
# --------------------------------------------------

encoder_config = ViTConfig(
    embed_dim=512, num_heads=8, dropout=0.0, hidden_layer_size=2048,
    num_blocks=12, patch_size=PATCH_SIZE, image_height=IMAGE_HEIGHT,
    max_image_width=MAX_IMAGE_WIDTH)

decoder_config = ViTConfig(
    embed_dim=256, num_heads=4, dropout=0.0, hidden_layer_size=1024,
    num_blocks=8, patch_size=PATCH_SIZE, image_height=IMAGE_HEIGHT,
    max_image_width=MAX_IMAGE_WIDTH)

mae_model = MaskedAutoEncoder(
    encoder_config=encoder_config,
    decoder_config=decoder_config,
    mask_ratio=MASK_RATIO
)

num_params = 0
for layer in mae_model.parameters():
    num_params += layer.numel()

print(f"No. of parameters in the full MAE model: {num_params}")

num_params = 0
for layer in mae_model.encoder_model.parameters():
    num_params += layer.numel()

print(f"No. of parameters in the encoder model: {num_params}")

num_params = 0
for layer in mae_model.decoder_model.parameters():
    num_params += layer.numel()

print(f"No. of parameters in the decoder model: {num_params}")




# New checkpoints are written here (writable on Kaggle).
OUTPUT_DIR = "/kaggle/working/telugu-vitmae"

# A previous session's output, added as a Kaggle Dataset / notebook-output input.
# Read-only. Leave as None for the very first run.
PREV_RUN_DIR = None  # e.g. "/kaggle/input/telugu-vitmae-prev/telugu-vitmae"

def find_last_checkpoint():
    # Prefer the working dir (same/committed session), then a prior run's input.
    for d in (OUTPUT_DIR, PREV_RUN_DIR):
        if d and os.path.isdir(d):
            ckpt = get_last_checkpoint(d)
            if ckpt is not None:
                return ckpt
    return None

last_checkpoint = find_last_checkpoint()
print(f"Resuming from: {last_checkpoint}" if last_checkpoint else "No checkpoint — starting fresh.")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=False,        # don't wipe existing checkpoints
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,     # raise to grow effective batch on T4
    learning_rate=1e-4,
    num_train_epochs=EPOCHS,           # keep IDENTICAL across resumes
    weight_decay=0.05,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    adam_beta2=0.95,
    eval_strategy="steps",             # renamed from evaluation_strategy
    eval_steps=2000,
    save_strategy="steps",
    save_steps=2000,                   # checkpoint often — sessions can be cut
    save_total_limit=2,                # keep storage under the ~20 GB cap
    logging_steps=50,
    remove_unused_columns=False,
    fp16=torch.cuda.is_available(),    # T4 = fp16 (no bf16 on Turing)
    dataloader_num_workers=2,
    report_to="none",
)

trainer = Trainer(
    model=mae_model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    data_collator=collator_function,
)

# A path resumes; None starts fresh — neither raises (unlike passing True).
trainer.train(resume_from_checkpoint=last_checkpoint)
trainer.save_model(OUTPUT_DIR)