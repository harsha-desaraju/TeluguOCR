
import numpy as np
import torch
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




# ============================================================================
# Line-image geometry — THE one implementation
# ============================================================================
# Before phase 3 this existed seven times: as `ImagePreprocessor` here, in the synth
# pipeline and in two training loops, and as the functions `preprocess_for_ctc`,
# `preprocess_line` and `preprocess_image`. Four of those were pixel-identical and the
# copies differed only in what they RETURNED -- PIL, uint8 ndarray, or a normalized
# tensor -- which is why they drifted apart without anyone noticing: nothing compares
# a PIL image to a tensor.
#
# Getting this wrong is silent. There is no exception and no shape error; the encoder
# simply reads badly. So it lives once, and the `out` parameter covers the three return
# conventions the callers actually need.

_RESAMPLE = Image.BILINEAR


def resize_line_image(img: Image.Image,
                      image_height: int = 64,
                      max_image_width: int = 2048,
                      downsample: int = 8,
                      resample=_RESAMPLE,
                      out: str = "np"):
    """A line crop in the form the encoder consumes.

    grayscale -> scale to `image_height` preserving aspect -> pad width to a multiple
    of `downsample` with white.

    THE OVER-WIDE BRANCH MATTERS. Once the aspect-preserving scale would exceed
    `max_image_width`, the scale is driven by WIDTH instead and the height shortfall is
    padded top/bottom. The alternative -- squashing horizontally to fit -- destroys the
    glyph aspect ratio the model was trained on, and reads as a mysterious accuracy
    cliff on long lines rather than as a bug.

    out:
      "np"  -> uint8 ndarray (H, W)            -- storage, JPEG encoding
      "pil" -> PIL.Image mode "L"              -- the synth pipeline, which saves crops
      "pt"  -> float32 tensor (1, H, W) in [-1, 1], i.e. Normalize(0.5, 0.5) applied
    """
    if out not in ("np", "pil", "pt"):
        raise ValueError(f"out must be 'np', 'pil' or 'pt'; got {out!r}")

    im = img.convert("L")
    w, h = im.size
    scale = image_height / h

    if scale * w > max_image_width:
        target_h = max(1, int((max_image_width / w) * h))
        im = im.resize((max_image_width, target_h), resample)
        arr = np.asarray(im, dtype=np.uint8)
        pad_top = (image_height - target_h) // 2
        arr = np.pad(arr, ((pad_top, image_height - target_h - pad_top), (0, 0)),
                     constant_values=255)
    else:
        im = im.resize((max(1, int(scale * w)), image_height), resample)
        arr = np.asarray(im, dtype=np.uint8)

    pad_w = (-arr.shape[1]) % downsample
    if pad_w:
        arr = np.pad(arr, ((0, 0), (0, pad_w)), constant_values=255)

    if out == "np":
        return arr
    if out == "pil":
        return Image.fromarray(arr)
    x = arr.astype(np.float32) / 255.0
    return torch.from_numpy(((x - 0.5) / 0.5)[None, ...])


class ImagePreprocessor:
    """Line image -> normalized float tensor (1, H, W), ready for the encoder.

    Thin wrapper over `resize_line_image(out="pt")`, kept because callers construct it
    once with the geometry and then call it per image.

    `patch_size` is the encoder's width reduction (conv-stem downsample); the name is
    historical, from when the encoder was patch-based.
    """

    def __init__(self, image_height: int, max_image_width: int, patch_size: int):
        assert image_height % patch_size == 0, "Image height should be a multiple of patch size"
        self.image_height = image_height
        self.max_image_width = max_image_width
        self.patch_size = patch_size

    def _transform(self, img: Image.Image) -> torch.Tensor:
        return resize_line_image(img, self.image_height, self.max_image_width,
                                 self.patch_size, out="pt")

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self._transform(img)
