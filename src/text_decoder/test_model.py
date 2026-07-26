

import torch
from safetensors.torch import  load_file
from src.text_decoder.grapheme_tokenizer.tokenizer import TeluguGraphemeTokenizer
from src.text_decoder.model import GPTConfig, GPTModel
from typing import Optional
import torch.nn.functional as F




@torch.no_grad()
def sample_generate(
    model,
    tokenizer,
    prefill: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.95,
    add_bos: bool = True,
    seed: Optional[int] = None,
) -> str:
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)

    ctx_len = model.positional_encodings.shape[0]   # 256 in your config

    ids = tokenizer(prefill, add_special_tokens=False)["input_ids"]
    if add_bos and tokenizer.bos_token_id is not None:
        ids = [tokenizer.bos_token_id] + ids
    input_ids = torch.tensor([ids], dtype=torch.long)

    eos_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        cond = input_ids[:, -ctx_len:]                       # crop to context window
        logits = model(cond, None).logits[:, -1, :]          # (1, vocab)

        logits = logits / max(temperature, 1e-5)

        # --- top-k ---
        if top_k and top_k > 0:
            k = min(top_k, logits.size(-1))
            kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))

        # --- top-p (nucleus) ---
        if top_p and 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            cum_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            remove = cum_probs > top_p
            remove[..., 1:] = remove[..., :-1].clone()        # keep the first token over the threshold
            remove[..., 0] = False
            remove = remove.scatter(-1, sorted_idx, remove)
            logits = logits.masked_fill(remove, float("-inf"))

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)     # (1, 1)
        input_ids = torch.cat([input_ids, next_id], dim=1)

        if eos_id is not None and next_id.item() == eos_id:
            break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)







if __name__ == '__main__':
    tokenizer = TeluguGraphemeTokenizer(
        vocab_file="/Users/xai/Personal/Projects/TeluguOCR/src/text_decoder/grapheme_tokenizer/telugu-vocab.json")
    print(len(tokenizer))

    model_config = GPTConfig(
        vocab_size=2048,
        embed_dim=512,
        hidden_dim=1368,
        num_heads=8,
        num_layers=16,
        ctx_len=256,
        dropout=0.1
    )
    model = GPTModel(model_config, pad_index=tokenizer.pad_token_type_id)

    state_dict = torch.load("models/text_decoder/telugu-grapheme-gpt/final_model.pt", map_location=torch.device('cpu'))
    # state_dict = load_file("/Users/xai/Personal/Projects/TeluguOCR/models/text_decoder/results/telugu-grapheme-gpt/checkpoint-49479/model.safetensors")
    model.load_state_dict(state_dict)
    print(model)

    params = 0
    for layer in model.parameters():
        params += layer.numel()
    print(f"The number of parameters in the model: {params}")

    prefill = "ప్రియుడిని "          # Telugu
    # prefill = "World"          # English
    # prefill = "పూర్వోక్త ఏవం"      # Sanskrit

    # A few samples at a moderate temperature
    print("=== top-k 40 + top-p 0.95, temp 0.8 ===")
    for s in range(5):
        out = sample_generate(model, tokenizer, prefill, max_new_tokens=50, seed=s)
        print(f"[{s}] {out}")

    # Temperature sweep — see the distribution sharpen/loosen
    print("\n=== temperature sweep ===")
    for t in (0.5, 0.7, 1.0, 1.2):
        out = sample_generate(model, tokenizer, prefill, max_new_tokens=50,
                              temperature=t, top_k=0, top_p=1.0, seed=0)
        print(f"[T={t}] {out}")
