
import torch
from PIL import Image
from pathlib import Path
from typing import Literal, Any
from inference.layout_detection import get_textline_boxes, crop_image
from src.telugu_ocr.models.image_encoder import CTCEncoderConfig
from src.telugu_ocr.models.encoder_decoder import GPTConfig, EncoderDecoder
from src.telugu_ocr.tokenizer.grapheme import TeluguGraphemeTokenizer
from src.telugu_ocr.data.preprocess import resize_line_image
from scripts.eval.encoder_decoder import (
            beam_search, ctc_hyp_logprobs
        )
from safetensors.torch import load_file
from pydantic import BaseModel, Field, model_validator, PrivateAttr

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    else:
        return "cpu"



class InferenceConfig(BaseModel):
    """Config for running the OCR"""
    inference_mode: Literal["ctc", "llm-greedy", "llm-beam", "joint"]
    ctc_encoder_config: CTCEncoderConfig
    decoder_config: GPTConfig
    encoder_decoder_path: Path | str
    vocab_path: Path | str
    _tokenizer: TeluguGraphemeTokenizer = PrivateAttr()
    pad_token_id: int | None = None
    vocab_size: int | None = None
    blank_id: int | None = None
    ctc_loss_weight: float = 0.3
    device: str = get_device()
    joint_weight: float = 0.3
    beam_width: int = 5
    max_new_tokens: int = 256
    beam_len_alpha: float = 0.0

    def model_post_init(self, context: Any, /) -> None:
        self._tokenizer = TeluguGraphemeTokenizer(vocab_file=self.vocab_path)

    @model_validator(mode="after")
    def set_pad_token_id(self):
        self.pad_token_id = self._tokenizer.pad_token_id
        return self

    @model_validator(mode="after")
    def set_vocab_size(self):
        self.vocab_size = len(self._tokenizer)
        return self

    @model_validator(mode='after')
    def set_blank_id(self):
        self.blank_id = len(self._tokenizer)
        return self



@torch.no_grad()
def ctc_greedy_decoding(pred_ids: torch.Tensor, blank_id: int):
    """Convert the predicted ids to token ids. Remove consecutive repeats and blanks"""

    if len(pred_ids.shape) > 2:
        raise AssertionError("pred_ids should be either 1D or 2D only")

    ids = pred_ids.tolist()

    if len(pred_ids.shape) == 1:
        token_ids, prev = [], None
        for id in ids:
            if id!= prev and id != blank_id:
                token_ids.append(id)
            prev = id
    else:
        token_ids = []
        for i in range(len(ids)):
            token_lst, prev = [], None
            for id in ids[i]:
                if id != prev and id != blank_id:
                    token_lst.append(id)
                prev = id
            token_ids.append(token_lst)

    return token_ids






class OCRInference:
    def __init__(self, config: InferenceConfig):

        self.model = EncoderDecoder(
            encoder_config=config.ctc_encoder_config,
            decoder_config=config.decoder_config,
            pad_index=config.pad_token_id,
            ctc_loss_weight=config.ctc_loss_weight,
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


    def run_ctc(self, images: Image.Image | list[Image.Image], batch_size: int = 32):
        if isinstance(images, Image.Image):
            img = self.preprocess_image(images).to(self.device)

            with torch.inference_mode():
                out = self.model.encoder_model(img)
            token_ids = ctc_greedy_decoding(out['logits'][0], blank_id=self.config.blank_id)
            text = self.convert_ids_to_tokens(token_ids)
            return text
        elif isinstance(images, list) and all([isinstance(img, Image.Image) for img in images]):
            tensors = [self.preprocess_image(img) for img in images]

            texts = []
            for i in range(0, len(tensors), batch_size):
                img_batch, input_lengths = self.create_batch(tensors[i: i + batch_size])
                with torch.inference_mode():
                    out = self.model.encoder_model(img_batch.to(self.device),
                                                   input_lengths=input_lengths.to(self.device))
                token_ids = ctc_greedy_decoding(out['logits'], blank_id=self.config.blank_id)
                texts.extend(self.convert_ids_to_tokens(token_ids))
            return texts
        else:
            raise TypeError(f"Expected a PIL image or a list of them, got {type(images)}")


    def run_llm(self, image: Image.Image):
        """Runs the encoder_decoder for a single sample with the configured strategy."""
        pixels = self.preprocess_image(image).to(self.device)

        with torch.inference_mode():
            # One encoder forward feeds whichever decoder runs below.
            enc_raw, _ = self.model.encoder_model.encode(pixels, None)
            bridged = self.model.enc_to_dec(enc_raw)

            if self.config.inference_mode == "llm-greedy":
                ids = self.model.generate(pixels,
                                          self.tokenizer.bos_token_id,
                                          self.tokenizer.eos_token_id,
                                          max_new_tokens=self.config.max_new_tokens,
                                          enc_out=bridged)
                return self.tokenizer.decode(ids, skip_special_tokens=True).strip()

            nbest = beam_search(self.model, bridged, self.tokenizer.bos_token_id,
                                self.tokenizer.eos_token_id, self.config.beam_width,
                                self.config.max_new_tokens)
            texts = [self.tokenizer.decode(h["ids"], skip_special_tokens=True).strip()
                     for h in nbest]
            attn_lp = torch.tensor([h["logp"] for h in nbest])

            if self.config.inference_mode == "llm-beam":
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


    def run(self, images: Image.Image | list[Image.Image], batch_size: int = 16):
        """Run the images through the OCR pipeline"""
        if self.config.inference_mode == "ctc":
            output = self.run_ctc(images, batch_size)
            return output

        else:
            if isinstance(images, Image.Image):
                return self.run_llm(images)
            elif isinstance(images, list) and all([isinstance(img, Image.Image) for img in images]):
                texts = []
                for i in range(len(images)):
                    texts.append(self.run_llm(images[i]))
                return texts
            else:
                raise ValueError("Got unexpected type in the input")












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
        inference_mode="llm-beam"
    )



    ocr_engine = OCRInference(inference_config)
    num = 24
    out = ocr_engine.run(line_images[num], batch_size=16)
    if isinstance(out, list):
        for txt in out:
            print(txt)
    else:
        print(out)
 