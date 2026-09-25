FROM python:3.12-slim
WORKDIR /app


RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


# Copy the src files also
COPY src/telugu_ocr/models/image_encoder.py ./src/telugu_ocr/models/
COPY src/telugu_ocr/models/encoder_decoder.py ./src/telugu_ocr/models/
COPY src/telugu_ocr/tokenizer/grapheme.py ./src/telugu_ocr/tokenizer
COPY src/telugu_ocr/decoding.py ./src/telugu_ocr/
COPY src/telugu_ocr/data/preprocess.py ./src/telugu_ocr/data/


COPY inference/* ./

RUN pip install --no-cache-dir -r requirements.txt

# Download the model and vocab file
RUN python set_up.py

CMD ["uvicorn", "ocr_server:app", "--host", "0.0.0.0", "--port", "8080"]