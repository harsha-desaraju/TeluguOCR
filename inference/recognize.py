
import torch
from PIL import Image
from pathlib import Path
from typing import Literal, Any
from safetensors.torch import load_file
from pydantic import BaseModel, PrivateAttr
from inference.layout_detection import get_textline_boxes, crop_image
from src.telugu_ocr.models.image_encoder import CTCEncoderConfig
from src.telugu_ocr.models.encoder_decoder import GPTConfig, EncoderDecoder
from src.telugu_ocr.tokenizer.grapheme import TeluguGraphemeTokenizer
from src.telugu_ocr.data.preprocess import resize_line_image
from src.telugu_ocr.decoding import (
    beam_search, ctc_collapse, ctc_hyp_logprobs, greedy_decode,
)





def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    else:
        return "cpu"



DecodeMode = Literal["ctc", "llm-greedy", "llm-beam", "joint"]


class InferenceConfig(BaseModel):
    """Config for running the OCR"""
    ctc_encoder_config: CTCEncoderConfig
    decoder_config: GPTConfig
    encoder_decoder_path: Path | str
    vocab_path: Path | str
    _tokenizer: TeluguGraphemeTokenizer = PrivateAttr()
    pad_token_id: int | None = None
    blank_id: int | None = None
    device: str = get_device()
    joint_weight: float = 0.3
    beam_width: int = 5
    max_new_tokens: int = 256
    beam_len_alpha: float = 0.0

    def model_post_init(self, context: Any, /) -> None:
        self._tokenizer = TeluguGraphemeTokenizer(vocab_file=self.vocab_path)
        size = len(self._tokenizer)
        # Check if the vocab sizes of both encoder and decoder are the same.
        if {self.ctc_encoder_config.vocab_size, self.decoder_config.vocab_size} != {size}:
            raise ValueError(
                f"vocab_size mismatch: tokenizer={size}, "
                f"encoder={self.ctc_encoder_config.vocab_size}, "
                f"decoder={self.decoder_config.vocab_size}")
        self.pad_token_id = self._tokenizer.pad_token_id
        self.blank_id = size


class OCRInference:
    def __init__(self, config: InferenceConfig):

        self.model = EncoderDecoder(
            encoder_config=config.ctc_encoder_config,
            decoder_config=config.decoder_config,
            pad_index=config.pad_token_id,
            ctc_loss_weight=0.0,
            encoder_no_grad=True
        )
        self.model.load_state_dict(self.load_model(str(config.encoder_decoder_path), device=config.device))

        self.device = config.device
        self.config = config
        self.tokenizer = config._tokenizer

        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def load_model(file_path: str, device: str = "cpu"):
        if file_path.endswith("safetensors"):
            weights = load_file(file_path)
        elif file_path.endswith("pt") or file_path.endswith("pts"):
            weights = torch.load(file_path, map_location=device)
        else:
            raise ValueError("Got unexpected model type. Model extension should be one of [`safetensors`, `pt`, `pts`]")
        return weights



    def convert_ids_to_tokens(self, token_ids: list[int] | list[list[int]]):
        # An all-blank line decodes to [], which has no token_ids[0] to inspect.
        if token_ids and isinstance(token_ids[0], list):
            return [self.tokenizer.decode(ids) for ids in token_ids]
        return self.tokenizer.decode(token_ids)

    def preprocess_image(self, img: Image.Image):
        """Grayscale + resize to the encoder's geometry -> (1, 1, H, W) in [-1, 1]."""
        cfg = self.config.ctc_encoder_config
        return resize_line_image(img.convert('L'),
                                 image_height=cfg.image_height,
                                 max_image_width=cfg.max_image_width,
                                 downsample=cfg.downsample,
                                 out='pt').unsqueeze(0)


    def create_batch(self, images: list[torch.Tensor]):
        """Right-pad to the widest image, returning the batch and its valid frame counts.

        Zero padding matches OCRCollator, which is what the model was trained on --
        NOT the white that `resize_line_image` uses for a single line.

        `input_lengths` is not optional: without it the encoder attends over the padding
        and transcribes it. It still does not make a batch equal to one-at-a-time --
        the stem's GroupNorm takes its statistics over the whole padded (H, W), so the
        amount of padding perturbs EVERY frame. Batch by similar widths to keep it small.
        """
        downsample = self.config.ctc_encoder_config.downsample
        widths = [img.shape[-1] for img in images]
        width = max(widths)

        padded = []
        for img in images:
            pad = width - img.shape[-1]
            if pad:
                fill = torch.zeros((1, 1, img.shape[-2], pad), dtype=img.dtype)
                img = torch.concatenate([img, fill], dim=-1)
            padded.append(img)

        input_lengths = torch.tensor([w // downsample for w in widths], dtype=torch.long)
        return torch.concatenate(padded, dim=0), input_lengths


    def _encode(self, tensors: list[torch.Tensor]):
        """Padded batch -> (raw encoder frames, bridged frames, cross-attention key mask)."""
        batch, input_lengths = self.create_batch(tensors)
        enc_raw, key_padding_mask = self.model.encoder_model.encode(
            batch.to(self.device), input_lengths.to(self.device))
        cross_key_mask = (None if key_padding_mask is None
                          else (~key_padding_mask).unsqueeze(1).unsqueeze(2))
        return enc_raw, self.model.enc_to_dec(enc_raw), cross_key_mask

    def run_ctc(self, images: list[Image.Image], batch_size: int = 32) -> list[str]:
        """Greedy CTC over the encoder's per-frame argmax."""
        tensors = [self.preprocess_image(img) for img in images]

        texts = []
        for i in range(0, len(tensors), batch_size):
            batch, input_lengths = self.create_batch(tensors[i: i + batch_size])
            with torch.inference_mode():
                out = self.model.encoder_model(batch.to(self.device),
                                               input_lengths=input_lengths.to(self.device))
            texts.extend(self.tokenizer.decode(ctc_collapse(row, self.config.blank_id))
                         for row in out["logits"].tolist())
        return texts

    def run_greedy(self, images: list[Image.Image], batch_size: int = 32) -> list[str]:
        """Greedy decode through the text decoder, batched.

        Beam search cannot batch this way -- it spends the batch dimension on its own
        beams -- so `llm-beam` and `joint` stay one image at a time in `run_llm`.
        """
        tensors = [self.preprocess_image(img) for img in images]

        texts = []
        for i in range(0, len(tensors), batch_size):
            with torch.inference_mode():
                _, bridged, cross_key_mask = self._encode(tensors[i: i + batch_size])
                rows = greedy_decode(self.model, bridged,
                                     self.tokenizer.bos_token_id, self.tokenizer.eos_token_id,
                                     self.config.max_new_tokens, cross_key_mask=cross_key_mask)
            texts.extend(self.tokenizer.decode(r, skip_special_tokens=True).strip()
                         for r in rows)
        return texts

    def run_llm(self, image: Image.Image, decode_mode: Literal["llm-beam", "joint"]) -> str:
        """Beam search over one image, optionally rescored against the CTC head."""
        with torch.inference_mode():
            # One encoder forward feeds both the beam and the CTC rescorer.
            enc_raw, bridged, _ = self._encode([self.preprocess_image(image)])

            nbest = beam_search(self.model, bridged, self.tokenizer.bos_token_id,
                                self.tokenizer.eos_token_id, self.config.beam_width,
                                self.config.max_new_tokens)
            texts = [self.tokenizer.decode(h["ids"], skip_special_tokens=True).strip()
                     for h in nbest]
            attn_lp = torch.tensor([h["logp"] for h in nbest])

            if decode_mode == "llm-beam":
                norm = attn_lp / torch.tensor(
                    [(len(h["ids"]) + 1.0) ** self.config.beam_len_alpha for h in nbest])
                return texts[int(norm.argmax())]

            # joint: rescore the same n-best with the CTC forward algorithm
            ctc_lp = self.model.encoder_model.ctc_head(enc_raw)[0].float().log_softmax(-1).cpu()
            ctc_scores = ctc_hyp_logprobs(ctc_lp, [h["ids"] for h in nbest],
                                          self.config.blank_id)
            combined = self.config.joint_weight * ctc_scores + (1.0 - self.config.joint_weight) * attn_lp
            if torch.isinf(combined).all():      # CTC rejected every hypothesis
                combined = attn_lp
            return texts[int(combined.argmax())]

    def run(self, images: Image.Image | list[Image.Image], decode_mode: DecodeMode,
            batch_size: int = 32) -> str | list[str]:
        """Transcribe one image or a list of them, returning output shaped like the input."""
        single = isinstance(images, Image.Image)
        batch = [images] if single else images
        if not (isinstance(batch, list) and all(isinstance(im, Image.Image) for im in batch)):
            raise TypeError(f"Expected a PIL image or a list of them, got {type(images)}")

        if decode_mode == "ctc":
            texts = self.run_ctc(batch, batch_size)
        elif decode_mode == "llm-greedy":
            texts = self.run_greedy(batch, batch_size)
        elif decode_mode in ("llm-beam", "joint"):
            texts = [self.run_llm(img, decode_mode) for img in batch]
        else:
            raise ValueError(f"Unknown decode_mode {decode_mode!r}")

        return texts[0] if single else texts


if __name__ == '__main__':
    from pathlib import Path

    path_to_image = "/Users/xai/Desktop/page.png"

    image = Image.open(path_to_image)
    image_layout = get_textline_boxes(image, plot_image=False)
    line_images = crop_image(image, image_layout)


    print(f"Running on : {get_device()}")



    gpt_config = GPTConfig(
        vocab_size=2048,
        embed_dim=512,
        hidden_dim=1368,  # 2.67 * 512 = 2/3 * 4 * hidden_dim
        num_heads=8,
        num_layers=16,
        ctx_len=256,
        dropout=0.1
    )


    ROOT = Path(__file__).resolve().parent.parent

    inference_config = InferenceConfig(
        # max_image_width is 2048 for every trained checkpoint; the dataclass default
        # (1024) describes no artifact -- see configs/models/ctc_encoder_2048.yaml.
        ctc_encoder_config=CTCEncoderConfig(max_image_width=2048, max_frames=256),
        decoder_config=gpt_config,
        encoder_decoder_path=ROOT / "models/encoder_decoder/results_stage_2_mid/telugu-ocr-stage2/checkpoint-34000/model.safetensors",
        vocab_path=str(ROOT / "src/telugu_ocr/tokenizer/assets/telugu-vocab.json"),
    )



    ocr_engine = OCRInference(inference_config)
    num = 4
    # CTC single
    out = ocr_engine.run(line_images[num], decode_mode="ctc", batch_size=16)
    print(f"Decode Strategy - CTC        : {out}\n")
    out = ocr_engine.run(line_images[num], decode_mode="llm-greedy", batch_size=16)
    print(f"Decode Strategy - llm-greedy : {out}\n")
    out = ocr_engine.run(line_images[num], decode_mode="llm-beam", batch_size=16)
    print(f"Decode Strategy - llm-beam   : {out}\n")
    out = ocr_engine.run(line_images[num], decode_mode="joint", batch_size=16)
    print(f"Decode Strategy - joint      : {out}\n")
 