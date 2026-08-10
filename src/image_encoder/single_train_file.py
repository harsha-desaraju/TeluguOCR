import os
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
        self.to_tensor = transforms.PILToTensor()

    def _transform(self, img: Image.Image) -> torch.Tensor:
        # Convert to GrayScale
        img = img.convert('L')
        return self.to_tensor(img)

    def __call__(self, sample: dict) -> dict:
        return {
            "line_image": [self._transform(sample['line_image'][0])],
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
    def __init__(self, encoder_config: ViTConfig, decoder_config: ViTConfig,mask_ratio: float, norm_pix_loss: bool = True):
        super().__init__()

        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.patch_size = encoder_config.patch_size
        self.encoder_model = ViTEncoder(encoder_config)
        self.decoder_model = ViTDecoder(decoder_config, encoder_config.embed_dim)

    def forward(self, images: torch.Tensor, padding_masks: torch.Tensor):
        images = (images.float() / 255.0 - 0.5) / 0.5

        latent, mask, restore_ids = self.encoder_model(images, padding_masks,  self.mask_ratio)

        pred = self.decoder_model(latent, restore_ids, padding_masks)

        target = patchify(images, self.patch_size)          # (B, N, patch_size**2)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        loss = ((target - pred) ** 2).mean(dim=-1)

        valid_patch_mask = (~padding_masks.bool()).float()
        effective_mask = mask * valid_patch_mask
        loss = (loss * effective_mask).sum() / effective_mask.sum()

        return {"loss": loss, "logits": pred, "mask": mask}



def collator_function(batch):
    imgs = [s['line_image'] for s in batch]          # each (C, H, W); C and H fixed
    B = len(imgs)
    Cc, H = imgs[0].shape[0], imgs[0].shape[1]
    widths = torch.tensor([im.shape[2] for im in imgs])
    W_max = int(widths.max())

    # One preallocation + one memcpy per image.
    # No per-image torch.cat and no final list-of-tensors cat (that was the ~2x data movement).
    images = imgs[0].new_zeros((B, Cc, H, W_max))
    for i, im in enumerate(imgs):
        images[i, :, :, :im.shape[2]] = im

    # Fully-vectorized patch padding mask. Row-major over (h_p, w_p) so it matches
    # the encoder's embeds.flatten(2) order (index = h*w_p + w).
    h_p  = H // PATCH_SIZE
    w_mp = W_max // PATCH_SIZE
    wp   = widths // PATCH_SIZE                        # real patch-width per image
    col  = torch.arange(w_mp)
    col_pad = col.unsqueeze(0) >= wp.unsqueeze(1)      # (B, w_mp)  True = padding
    padding_masks = col_pad.unsqueeze(1).expand(B, h_p, w_mp).reshape(B, h_p * w_mp)

    return {"images": images, "padding_masks": padding_masks}


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


# =============================================================================
# Drop-in replacement for your `if __name__ == '__main__':` block.
# Keep all your model/dataset/collator definitions above unchanged.
#
# Three things done here:
#   1) Balanced settings, 150 epochs (from scratch).
#   2) Robust checkpointing + resume (survives Kaggle's 12h session wall).
#   3) eval_loss fix: label_names=[] + trainer.can_return_loss=True
#      (+ prediction_loss_only=True so eval doesn't OOM accumulating logits).
# =============================================================================

if __name__ == '__main__':
    # ---- Balanced recipe (from scratch) ----------------------------------
    PER_DEVICE_BATCH = 256  # start safe on 16GB T4; bump to 64 if headroom
    GRAD_ACCUM = 4  # eff batch = 48 * 2 GPUs * 8 = 768
    EPOCHS = 5
    # LEARNING_RATE = 1.2e-3  # 1.5e-4 * eff_batch/256  (MAE linear scaling)
    LEARNING_RATE = 1.5e-4 * GRAD_ACCUM * 2 * PER_DEVICE_BATCH / 256
    print(f"Using a batch size of  : {PER_DEVICE_BATCH}")
    print(f"Using learning rate of : {LEARNING_RATE:e}")
    TEST_SIZE = 0.005

    # Full eval set is ~133k images (5% of 2.67M) — far more than you need just
    # to watch a loss curve, and it makes every eval slow. Monitor on a fixed
    # random subset. Set to `None` to evaluate on the full held-out split.
    EVAL_SUBSET_SIZE = 6000

    IMAGE_HEIGHT = 64
    MAX_IMAGE_WIDTH = 1024
    PATCH_SIZE = 8
    MASK_RATIO = 0.75

    # New checkpoints are written here (writable on Kaggle).
    OUTPUT_DIR = "/Users/xai/Personal/Projects/TeluguOCR/models/telugu-vitmae"

    # A previous session's committed output, re-added as a notebook-output /
    # dataset input. Read-only. Leave None for the very first run, then set it
    # to the prior session's output path for every continuation run.
    PREV_RUN_DIR = None  # e.g. "/kaggle/input/telugu-vitmae-prev/telugu-vitmae"

    ds = load_dataset(
        "harsha-desaraju/sample-line-images", columns=['line_image', "image_width"], split='train',
    )

    split_dataset = ds.train_test_split(test_size=TEST_SIZE, seed=42)
    train_ds = split_dataset['train']
    test_ds = split_dataset['test']

    # Subsample eval for fast, frequent monitoring (before transform).
    if EVAL_SUBSET_SIZE is not None and EVAL_SUBSET_SIZE < len(test_ds):
        test_ds = test_ds.shuffle(seed=42).select(range(EVAL_SUBSET_SIZE))

    preprocessor = ImagePreprocessor(IMAGE_HEIGHT, MAX_IMAGE_WIDTH, PATCH_SIZE)

    train_ds = train_ds.with_transform(preprocessor)
    test_ds = test_ds.with_transform(preprocessor)

    # ---- Model (FROM SCRATCH — no load_state_dict) -----------------------
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
        mask_ratio=MASK_RATIO,
    )

    def _count(m):
        return sum(p.numel() for p in m.parameters())


    print(f"Full MAE params:  {_count(mae_model):,}")
    print(f"Encoder params:   {_count(mae_model.encoder_model):,}")
    print(f"Decoder params:   {_count(mae_model.decoder_model):,}")


    # ---- Resume logic ----------------------------------------------------
    def find_last_checkpoint():
        # Prefer the current working dir, then a prior run's read-only input.
        for d in (OUTPUT_DIR, PREV_RUN_DIR):
            if d and os.path.isdir(d):
                ckpt = get_last_checkpoint(d)
                if ckpt is not None:
                    return ckpt
        return None


    last_checkpoint = find_last_checkpoint()
    print(f"Resuming from: {last_checkpoint}" if last_checkpoint
          else "No checkpoint — starting fresh.")

    # If the checkpoint lives in the read-only PREV_RUN_DIR, copy it into the
    # writable OUTPUT_DIR first, so the Trainer can keep writing new checkpoints
    # and correctly continues its global_step / optimizer / scheduler state.
    if last_checkpoint is not None and last_checkpoint.startswith(str(PREV_RUN_DIR or "")):
        import shutil

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        dst = os.path.join(OUTPUT_DIR, os.path.basename(last_checkpoint))
        if not os.path.isdir(dst):
            print(f"Copying checkpoint into writable dir: {dst}")
            shutil.copytree(last_checkpoint, dst)
        last_checkpoint = dst

    # ---- TrainingArguments ----------------------------------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,

        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,

        optim="adamw_torch_fused",
        learning_rate=LEARNING_RATE,
        weight_decay=0.05,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        adam_beta2=0.95,
        max_grad_norm=1.0,  # keep — fp16 stability at high LR

        # --- Checkpointing: frequent, to survive the 12h Kaggle wall ---
        save_strategy="steps",
        save_steps=250,
        save_total_limit=3,
        load_best_model_at_end=False,  # pure continuation; don't reload "best"

        # --- Eval / the eval_loss fix ---
        eval_strategy="steps",
        eval_steps=500,
        prediction_loss_only=True,  # don't accumulate (B,N,64) logits -> no OOM
        label_names=[],  # <-- tells Trainer the model returns its own loss

        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        # group_by_length=True,
        # length_column_name="image_width",

        # Precision / perf (T4 = Turing -> fp16, no bf16)
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,  # Kaggle has few CPU cores; 2 per DDP proc
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=4,
        dataloader_persistent_workers=True,

        logging_strategy="steps",
        logging_steps=50,
        logging_first_step=True,
        # report_to="wandb",
        # run_name="vit-mae-scratch-150ep",
    )

    trainer = Trainer(
        model=mae_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=collator_function,
    )
    # Second half of the eval_loss fix: the model computes loss without a
    # `labels` input, so tell the Trainer it's allowed to return that loss.
    trainer.can_return_loss = True

    trainer.train(resume_from_checkpoint=last_checkpoint)

    # Final weights (only reached if the full run completes in one session).
    trainer.save_model(OUTPUT_DIR)  # HF-style, resumable
    torch.save(mae_model.state_dict(), f"{OUTPUT_DIR}/final_model.pt")  # your raw state_dict