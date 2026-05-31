
import torch
from torchvision import transforms
from PIL import Image


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

    return x_masked, mask, ids_restore




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
    return encoding.view(h_patches * w_patches, embed_dim)  # N, D




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

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self._transform(img)
