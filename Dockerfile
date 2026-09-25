FROM python:3.12-slim
WORKDIR /app


RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

COPY inference/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY inference/ ./inference/
COPY src/telugu_ocr ./src/telugu_ocr/

# Download the model and vocab file
RUN python -m  inference.set_up

EXPOSE 8080
CMD ["uvicorn", "inference.ocr_server:app", "--host", "0.0.0.0", "--port", "8080"]