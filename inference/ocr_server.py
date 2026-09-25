"""A Fast API server for OCR detector, inference and pipeline"""

from io import BytesIO
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pathlib import Path
from src.telugu_ocr.models.image_encoder import CTCEncoderConfig
from src.telugu_ocr.models.text_decoder import GPTConfig
from .layout_detection import TextDetector
from .recognize import InferenceConfig, OCRInference, DecodeMode

app = FastAPI()




# Initialize the objects
gpt_config = GPTConfig(
    vocab_size=2048,
    embed_dim=512,
    hidden_dim=1368,
    num_heads=8,
    num_layers=16,
    ctx_len=256,
    dropout=0.1
)

inference_config = InferenceConfig(
        ctc_encoder_config=CTCEncoderConfig(max_image_width=2048, max_frames=256),
        decoder_config=gpt_config,
        encoder_decoder_path="assets/model.safetensors",
        vocab_path="assets/vocab.json",
    )

detector = TextDetector()
ocr_engine = OCRInference(inference_config)





# Define the APIs

@app.post("/detect")
async def detect_text(
        image: UploadFile = File(...),
        deskew: bool = Form(False),
        preprocess_image: bool = Form(False)):
    image_bytes = await image.read()
    try:
        image = Image.open(BytesIO(image_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad request. Invalid input in the image")


    detector_output = detector.detect(image, deskew, preprocess_image, False)

    lines = []
    for line in detector_output.detected_lines:
        lines.append({'id': line.id, 'bbox': line.bbox})

    return lines



@app.post("/get_text")
async def get_text(
            image: UploadFile = File(...),
            decode_mode: DecodeMode = Form(...),
            deskew: bool = Form(False),
            preprocess_image: bool = Form(False),
            batch_size: int = Form(16)
    ):
    image_bytes = await image.read()
    image = Image.open(BytesIO(image_bytes))

    detector_output = detector.detect(image, deskew, preprocess_image, False)

    ocr_output = ocr_engine.get_text(detector_output, decode_mode, batch_size)

    return ocr_output


