

from huggingface_hub import hf_hub_download



# Download the vocab file and the model file
repo = "harsha-desaraju/telugu-ocr-model"

print("Downloading the vocab file...")
vocab_path = hf_hub_download(
    repo_id=repo,
    filename="vocab.json",
    local_dir="./assets"
)
print("Successfully downloaded the vocab file")


print("Downloading the model file...")
model_path = hf_hub_download(
    repo_id=repo,
    filename="model.safetensors",
    local_dir="./assets"
)
print("Successfully downloaded the model file")