
"""
See if the model has learned anything at all
"""

import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from safetensors.torch import load_file
from single_train_file import MaskedAutoEncoder, ViTConfig, ImagePreprocessor, patchify



def unpatchify(patches, image_height, image_width, patch_size):
    """
    patches: (B, N, patch_size * patch_size)

    returns:
        (B, 1, H, W)
    """
    B, N, patch_dim = patches.shape

    h = image_height // patch_size
    w = image_width // patch_size

    assert N == h * w

    x = patches.reshape(
        B,
        h,
        w,
        patch_size,
        patch_size
    )

    x = torch.einsum("nhwpq->nhpwq", x)

    x = x.reshape(
        B,
        1,
        image_height,
        image_width
    )

    return x


class FolderImageDataset(Dataset):
    def __init__(self, image_dir, preprocessor):
        self.preprocessor = preprocessor

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

        self.files = [
            p for p in Path(image_dir).iterdir()
            if p.suffix.lower() in exts
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        img = Image.open(self.files[idx])

        sample = {
            "line_image": [img],
            "image_width": img.width
        }

        processed = self.preprocessor(sample)

        return {
            "line_image": processed["line_image"][0]
        }



def mae_collate_fn(batch, patch_size):

    # images = torch.stack(
    #     [x["line_image"] for x in batch],
    #     dim=0
    # )

    images = collator_function(batch)["images"]

    B, C, H, W = images.shape

    h_patches = H // patch_size
    w_patches = W // patch_size

    patchified = patchify(images, patch_size)

    # padded region becomes +1 after normalization
    # because:
    # white=255
    # ToTensor -> 1
    # Normalize((0.5),(0.5)) -> +1

    padded_patch = (patchified == 1.0).all(dim=-1)

    return {
        "line_image": images,
        "padding_mask": padded_patch
    }





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



if __name__ == '__main__':
    IMAGE_HEIGHT = 64
    MAX_IMAGE_WIDTH = 1024
    PATCH_SIZE = 8
    MASK_RATIO = 0.75

    device = "cpu"


    preprocessor = ImagePreprocessor(
        image_height=IMAGE_HEIGHT,
        max_image_width=MAX_IMAGE_WIDTH,
        patch_size=PATCH_SIZE
    )

    dataset = FolderImageDataset(
        "/Users/xai/Personal/Projects/TeluguOCR/data/temp_test",
        preprocessor
    )

    val_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=lambda x: mae_collate_fn(x, patch_size=PATCH_SIZE)
    )

    encoder_config = ViTConfig(
        embed_dim=512, num_heads=8, dropout=0.0, hidden_layer_size=2048,
        num_blocks=12, patch_size=PATCH_SIZE, image_height=IMAGE_HEIGHT,
        max_image_width=MAX_IMAGE_WIDTH)

    decoder_config = ViTConfig(
        embed_dim=256, num_heads=4, dropout=0.0, hidden_layer_size=1024,
        num_blocks=8, patch_size=PATCH_SIZE, image_height=IMAGE_HEIGHT,
        max_image_width=MAX_IMAGE_WIDTH)

    model = MaskedAutoEncoder(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        mask_ratio=MASK_RATIO,
        norm_pix_loss=False
    )


    # model.load_state_dict(load_file("/Users/xai/Personal/Projects/TeluguOCR/models/image_encoder/telugu-vitmae/model.safetensors"))
    model.load_state_dict(torch.load("/Users/xai/Personal/Projects/TeluguOCR/models/image_encoder/results/telugu-vitmae/final_model.pt", map_location="cpu"))
    print(model)



    model.eval()

    batch = next(iter(val_loader))

    images = batch["line_image"].to(device)
    padding_masks = batch["padding_mask"].to(device)

    with torch.no_grad():
        out = model(images, padding_masks)

    pred = out["logits"]
    mask = out['mask'].unsqueeze(-1)

    # target_patches = patchify(images, model.patch_size)
    #
    # combined = target_patches * (1 - mask) + pred * mask

    target_patches = patchify(images, model.patch_size)
    mean = target_patches.mean(dim=-1, keepdim=True)
    var = target_patches.var(dim=-1, keepdim=True)
    pred_pixels = pred * (var + 1e-6).sqrt() + mean  # normalized -> pixel space

    combined = target_patches * (1 - mask) + pred_pixels * mask

    B, N, _ = target_patches.shape

    H = images.shape[2]
    W = images.shape[3]

    reconstructed = unpatchify(
        combined,
        image_height=H,
        image_width=W,
        patch_size=model.patch_size
    )

    orig = images * 0.5 + 0.5
    recon = reconstructed * 0.5 + 0.5

    orig = orig.clamp(0, 1)
    recon = recon.clamp(0, 1)

    import matplotlib.pyplot as plt


    n = min(8, orig.shape[0])

    fig, axes = plt.subplots(n, 2, figsize=(8, 3 * n))

    for i in range(n):
        axes[i, 0].imshow(
            orig[i, 0].cpu(),
            cmap="gray"
        )
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(
            recon[i, 0].cpu(),
            cmap="gray"
        )
        axes[i, 1].set_title("Reconstruction")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()