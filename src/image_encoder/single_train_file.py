import os
import gc
import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from dataclasses import dataclass
from transformers.trainer_utils import get_last_checkpoint


def random_masking(x, mask_ratio):
    B, N, D = x.shape

    len_keep = int(N * (1 - mask_ratio))

    noise = torch.rand(B, N, device=x.device)

    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]

    x_masked = torch.gather(
        x,
        dim=1,
        index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
    )

    mask = torch.ones(B, N, device=x.device)
    mask[:, :len_keep] = 0

    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore, ids_keep

def patchify(images, patch_size):
    """
    images: B, C, H, W
    returns: B, N, patch_dim
    """
    B, C, H, W = images.shape

    assert H % patch_size == 0
    assert W % patch_size == 0

    h = H // patch_size
    w = W // patch_size

    patches = images.reshape(
        B, C, h,
        patch_size,
        w,
        patch_size
    )

    patches = torch.einsum('nchpwq->nhwpqc', patches)

    patches = patches.reshape(
        B, h * w,
        patch_size * patch_size * C
    )

    return patches

def get_2d_sinusoidal_encoding(h_patches, w_patches, embed_dim):
    assert embed_dim % 4 == 0  # split evenly between h and w dims

    def sinusoid(length, dim):
        positions = torch.arange(length).unsqueeze(1)          # L, 1
        dims = torch.arange(0, dim, 2).unsqueeze(0)            # 1, D/2
        freqs = 1.0 / (10000 ** (dims / dim))
        args = positions * freqs                                # L, D/2
        emb = torch.cat([args.sin(), args.cos()], dim=-1)      # L, D
        return emb

    h_enc = sinusoid(h_patches, embed_dim // 2)  # H, D/2
    w_enc = sinusoid(w_patches, embed_dim // 2)  # W, D/2

    # Broadcast and combine
    h_enc = h_enc.unsqueeze(1).repeat(1, w_patches, 1)  # H, W, D/2
    w_enc = w_enc.unsqueeze(0).repeat(h_patches, 1, 1)  # H, W, D/2

    encoding = torch.cat([h_enc, w_enc], dim=-1)         # H, W, D
    return encoding.view(h_patches, w_patches, embed_dim)  # N, D

class ImagePreprocessor:
    """
    Preprocesses the image before encoding the image
    1) Change the image to gray scale
    2) Resize the image
    3) Pad the image to the nearest multiple of patch size
    4) Normalize the image
    """
    def __init__(self, image_height: int, max_image_width: int, patch_size: int):
        assert image_height % patch_size == 0, "Image height should be a multiple of patch size"
        self.image_height = image_height
        self.max_image_width = max_image_width
        self.patch_size = patch_size
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def _transform(self, img: Image.Image) -> torch.Tensor:
        # Convert to GrayScale
        img = img.convert('L')

        # Calculate the resize target for the image while preserving the aspect ratio
        img_w, img_h = img.size
        scale_factor = self.image_height/img_h
        if scale_factor * img_w > self.max_image_width:
            scale_factor = self.max_image_width/img_w
            target_size = (int(scale_factor * img_h), self.max_image_width)
            diff = self.image_height - target_size[0]
            pad_t, pad_b = diff//2, diff - diff//2
            pad_l, pad_r = 0, 0
        else:
            target_size = (self.image_height, int(scale_factor*img_w))
            # Find the nearest multiple of patch size for padding
            diff = (-target_size[1]) % self.patch_size
            pad_l, pad_r = 0, diff
            pad_t, pad_b = 0, 0

        img = transforms.Resize(target_size)(img)
        img = transforms.Pad((pad_l, pad_t, pad_r, pad_b), fill=255)(img)

        return self.to_tensor(img)

    def __call__(self, sample: dict) -> dict:
        return {
            "line_image": self._transform(sample['line_image']),
            "image_width": sample["image_width"]
        }


@dataclass
class ViTConfig:
    embed_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.2
    hidden_layer_size: int = 1024
    num_blocks: int = 8
    patch_size: int = 16
    image_height: int = 64
    max_image_width:int = 1024




class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, hidden_layer_size: int, dropout: float):
        super().__init__()
        self.multi_head_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_layer_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_layer_size, embed_dim),
            nn.Dropout(dropout)
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, key_padding_mask = None):
        # query, key, value - B, T, D
        norm_x = self.layer_norm1(x)
        ctx_embed, _ = self.multi_head_attention(
            norm_x, norm_x, norm_x,
            key_padding_mask=key_padding_mask,
            need_weights=False)
        ctx_embed = ctx_embed + x

        norm_ctx_embed = self.layer_norm2(ctx_embed)
        norm_ctx_embed = self.mlp(norm_ctx_embed)
        norm_ctx_embed = norm_ctx_embed + ctx_embed
        return norm_ctx_embed



class ViTEncoder(nn.Module):
    def __init__(self, vit_config: ViTConfig):
        super().__init__()
        self.image_embedding = nn.Conv2d(in_channels=1, out_channels=vit_config.embed_dim, kernel_size=vit_config.patch_size, stride=vit_config.patch_size)

        enc = get_2d_sinusoidal_encoding(
            vit_config.image_height // vit_config.patch_size,
            vit_config.max_image_width // vit_config.patch_size,
            vit_config.embed_dim
        )
        self.register_buffer("positional_encoding", enc)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(vit_config.embed_dim, vit_config.num_heads, vit_config.hidden_layer_size, vit_config.dropout)
              for _ in range(vit_config.num_blocks)]
        )
        self.layer_norm = nn.LayerNorm(vit_config.embed_dim)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor, mask_ratio: float | None = None):
        # x -> B, C, H, W
        embeds = self.image_embedding(x)
        B, D, hp, wp = embeds.shape
        embeds = embeds.flatten(2).transpose(1, 2)
        pos_encodings = self.positional_encoding[:hp, :wp, :].reshape(hp*wp, D)
        embeds = embeds + pos_encodings

        if mask_ratio is not None:
            embeds, mask, restore_ids, ids_keep = random_masking(embeds, mask_ratio)
        else:
            mask, restore_ids, ids_keep = None, None, None

        if ids_keep is not None:
            visible_padding_mask = torch.gather(
                padding_mask, dim=1,
                index=ids_keep
            )
        else:
            visible_padding_mask = padding_mask

        visible_padding_mask = visible_padding_mask.bool()

        ctx_embeds = embeds
        for block in self.transformer_blocks:
            ctx_embeds = block(ctx_embeds, visible_padding_mask)

        ctx_embeds = self.layer_norm(ctx_embeds)
        return ctx_embeds, mask, restore_ids

class ViTDecoder(nn.Module):
    def __init__(self, config: ViTConfig, encoder_dim: int):
        super().__init__()
        self.embedding_layer = nn.Linear(encoder_dim, config.embed_dim)

        self.h_full = config.image_height // config.patch_size
        enc = get_2d_sinusoidal_encoding(
            config.image_height // config.patch_size,
            config.max_image_width // config.patch_size,
            config.embed_dim
        )
        self.register_buffer("positional_encoding", enc)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.num_heads, config.hidden_layer_size, config.dropout)
            for _ in range(config.num_blocks)
        ])
        self.decoder_norm = nn.LayerNorm(config.embed_dim)

        self.output_projection = nn.Linear(config.embed_dim, config.patch_size ** 2)


    def forward(self, latent: torch.Tensor, restore_ids: torch.Tensor, padding_mask: torch.Tensor):
        x = self.embedding_layer(latent)

        B, L, D = x.shape
        N = restore_ids.shape[1]

        mask_tokens = self.mask_token.repeat(B, N-L, 1)

        _x = torch.cat([x, mask_tokens], dim=1)
        _x = torch.gather(
            _x, dim=1,
            index=restore_ids.unsqueeze(-1).repeat(1, 1, D)
        )
        wp = N // self.h_full
        _x = _x + self.positional_encoding[:, :wp ,:].reshape(N, D)
        for block in self.transformer_blocks:
            _x = block(_x, padding_mask.bool())
        _x = self.decoder_norm(_x)
        proj = self.output_projection(_x)
        return proj

class MaskedAutoEncoder(nn.Module):
    def __init__(self, encoder_config: ViTConfig, decoder_config: ViTConfig, mask_ratio: float):
        super().__init__()

        self.mask_ratio = mask_ratio
        self.patch_size = encoder_config.patch_size
        self.encoder_model = ViTEncoder(encoder_config)
        self.decoder_model = ViTDecoder(decoder_config, encoder_config.embed_dim)


    def forward(self, images: torch.Tensor, padding_masks: torch.Tensor):
        latent, mask, restore_ids = self.encoder_model(images, padding_masks,  self.mask_ratio)

        pred = self.decoder_model(latent, restore_ids, padding_masks)

        target = patchify(images, self.patch_size)

        loss = ((target - pred)**2).mean(dim=-1)

        valid_patch_mask = (~padding_masks.bool()).float()
        effective_mask = mask * valid_patch_mask
        loss = (loss * effective_mask).sum() / effective_mask.sum()

        return {"loss": loss, "logits": pred}



def collator_function(batch):
    # Also think about the device ??

    img_height = 0
    max_batch_width = 0
    for sample in batch:
        image = sample['line_image']
        if image.shape[-1] > max_batch_width:
            max_batch_width = image.shape[2]
            img_height = image.shape[1]

    num_channels = batch[0]['line_image'].shape[0]

    zeros_img = batch[0]['line_image'].new_zeros((num_channels, img_height, max_batch_width))

    padded_images, padding_masks = [], []
    for sample in batch:
        image = sample['line_image']
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


def process_sample(sample):
    sample['line_image'] = preprocessor(sample['line_image'])
    sample['image_width'] = sample['line_image'].shape[-1]
    return sample


def find_last_checkpoint():
    # Prefer the working dir (same/committed session), then a prior run's input.
    for d in (OUTPUT_DIR, PREV_RUN_DIR):
        if d and os.path.isdir(d):
            ckpt = get_last_checkpoint(d)
            if ckpt is not None:
                return ckpt
    return None


if __name__ == '__main__':
    BATCH_SIZE = 256
    EPOCHS = 50
    TEST_SIZE = 0.05

    IMAGE_HEIGHT = 64
    MAX_IMAGE_WIDTH = 1024
    PATCH_SIZE = 8
    MASK_RATIO = 0.75

    # New checkpoints are written here (writable on Kaggle).
    OUTPUT_DIR = "/kaggle/working/telugu-vitmae"
    # OUTPUT_DIR = "/Users/xai/Personal/Projects/TeluguOCR/src/image_encoder/telugu-vitmae"

    # A previous session's output, added as a Kaggle Dataset / notebook-output input.
    # Read-only. Leave as None for the very first run.
    PREV_RUN_DIR = None  # e.g. "/kaggle/input/telugu-vitmae-prev/telugu-vitmae"


    # Load and Prepare the datasets

    ds = load_dataset(
        "harsha-desaraju/telugu-book-line-images-sample",
        columns=['line_image', 'image_width'],
        # download_mode="force_redownload"
    )["train"]

    split_dataset = ds.train_test_split(test_size=TEST_SIZE, seed=42)

    train_ds = split_dataset["train"]
    test_ds = split_dataset["test"]

    del split_dataset
    gc.collect()

    preprocessor = ImagePreprocessor(IMAGE_HEIGHT, MAX_IMAGE_WIDTH, PATCH_SIZE)

    train_ds = train_ds.map(preprocessor)
    test_ds = test_ds.map(preprocessor)

    train_ds = train_ds.with_format("torch", columns=['line_image'], output_all_columns=True)
    test_ds = test_ds.with_format("torch", columns=['line_image'], output_all_columns=True)

    # train_ds = train_ds.with_transform(preprocessor)
    # test_ds = test_ds.with_transform(preprocessor)

    # Initialize the models

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


    # Define the trainer and train the model
    last_checkpoint = find_last_checkpoint()
    print(f"Resuming from: {last_checkpoint}" if last_checkpoint else "No checkpoint — starting fresh.")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,  # raise to grow effective batch on T4
        optim="adamw_torch_fused",
        learning_rate=(1e-4*BATCH_SIZE/256),
        num_train_epochs=EPOCHS,  # keep IDENTICAL across resumes
        weight_decay=0.05,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        adam_beta2=0.95,
        # max_steps=20000, #   ---------------- ????????????
        group_by_length=True,
        length_column_name="image_width",
        eval_strategy="steps",  # renamed from evaluation_strategy
        eval_steps=2000,
        save_strategy="steps",
        save_steps=2000,  # checkpoint often — sessions can be cut
        save_total_limit=2,  # keep storage under the ~20 GB cap
        logging_steps=50,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        fp16=torch.cuda.is_available(),  # T4 = fp16 (no bf16 on Turing)
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=4,
        dataloader_persistent_workers=True,
        report_to="none",
    )


    # # Add the image_width column to the dataset in hugging face.
    # # Do the preprocessing using `with_transform` instead of map to save a lot of upfront time

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