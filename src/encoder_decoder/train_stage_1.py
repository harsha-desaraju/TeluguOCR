

import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from src.text_decoder.grapheme_tokenizer.tokenizer import TeluguGraphemeTokenizer
from .model import EncoderDecoder



class OCRCollator:
    """
    Each dataset example is expected to be a dict with:
      - 'pixel_values': float tensor (1, H, W_i)   # H fixed (image_height), W_i variable
      - 'input_ids'   : 1D long tensor / list      # variable length, NO padding yet

    Produces a batch dict whose keys match EncoderDecoder.forward exactly:
      pixel_values      (B, 1, H, W_max)   float
      input_ids         (B, T_max)         long, right-padded with pad_token_id
      img_padding_mask  (B, hp*wp)         float, 1 = PAD patch   (ViT/MHA convention)
      text_padding_mask (B, T_max)         float, 1 = REAL token  (decoder convention)

    NOTE the two opposite conventions are intentional and load-bearing:
      * the image mask feeds nn.MultiheadAttention's key_padding_mask (True = ignore)
      * the text mask feeds the decoder's causal-mask builder (1 = keep)
    Getting either polarity wrong is silent, so they are set explicitly here.

    Assumes every image width W_i is already a multiple of patch_size (your
    ImagePreprocessor pads to that). If not, partial edge patches are counted as real.
    """

    def __init__(self, pad_token_id: int, patch_size: int, image_height: int):
        self.pad_token_id = pad_token_id
        self.patch_size = patch_size
        self.hp = image_height // patch_size  # number of patch ROWS (fixed)

    def __call__(self, batch):
        images = [ex["pixel_values"] for ex in batch]
        token_seqs = [torch.as_tensor(ex["input_ids"], dtype=torch.long) for ex in batch]
        B = len(batch)

        # ---- Images ----
        widths = [img.shape[-1] for img in images]
        max_w = max(widths)
        if max_w % self.patch_size != 0:  # safety; should already be a multiple
            max_w += self.patch_size - (max_w % self.patch_size)
        wp = max_w // self.patch_size  # number of patch COLUMNS in the batch

        padded_imgs, img_pad_masks = [], []
        for img, w in zip(images, widths):
            padded_imgs.append(F.pad(img, (0, max_w - w)))  # right-pad width with zeros

            # ceil so a partially-real edge patch is treated as real, not pad
            real_wp = min((w + self.patch_size - 1) // self.patch_size, wp)
            col_pad = torch.zeros(wp, dtype=torch.bool)
            col_pad[real_wp:] = True  # True = padded patch column

            # Conv flattens (hp, wp) row-major -> patch index = h*wp + w.
            # Validity is identical across all hp rows, so tile the column mask.
            img_pad_masks.append(col_pad.unsqueeze(0).expand(self.hp, wp).reshape(-1))

        pixel_values = torch.stack(padded_imgs, dim=0)                  # (B, 1, H, max_w)
        img_padding_mask = torch.stack(img_pad_masks, dim=0).float()    # (B, hp*wp), 1=pad

        # ---- Text (right-pad) ----
        max_t = max(seq.shape[0] for seq in token_seqs)
        input_ids = torch.full((B, max_t), self.pad_token_id, dtype=torch.long)
        text_padding_mask = torch.zeros((B, max_t), dtype=torch.float)  # 1 = real
        for i, seq in enumerate(token_seqs):
            n = seq.shape[0]
            input_ids[i, :n] = seq
            text_padding_mask[i, :n] = 1.0

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "img_padding_mask": img_padding_mask,
            "text_padding_mask": text_padding_mask,
        }


if __name__ == '__main__':

    PATCH_SIZE = 8
    IMG_HEIGHT = 64

    OUTPUT_DIR = ""
    EPOCHS = 3
    BATCH_SIZE = 64


    model = EncoderDecoder()

    # Load the tokenizer
    tokenizer = TeluguGraphemeTokenizer(
        vocab_file="/Users/xai/Personal/Projects/TeluguOCR/src/text_decoder/grapheme_tokenizer/telugu-vocab.json")

    collator = OCRCollator(pad_token_id=tokenizer.pad_token_id, patch_size=PATCH_SIZE, image_height=IMG_HEIGHT)


    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        # --- Training Duration ---
        num_train_epochs=EPOCHS,

        # --- Batch Size & Accumulation ---
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,

        # --- Optimizer & Scheduler ---
        optim="adamw_torch_fused",
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        weight_decay=0.1,
        warmup_ratio=0.05,
        adam_beta2=0.95,

        ddp_find_unused_parameters=False,
        remove_unused_columns=False,

        # --- Precision & Performance ---
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=4,
        dataloader_persistent_workers=True,

        # --- Evaluation & Saving ---
        eval_strategy="steps",
        eval_steps=2000,
        save_strategy="steps",
        save_steps=2000,
        save_total_limit=2,

        # --- Logging ---
        logging_steps=100,
        logging_first_step=True,
        report_to="none",
        # run_name="llm-training-1"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model("out/stage1/final")

    # Save the model
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/final_model.pt")
    print(f"Finished training!")
