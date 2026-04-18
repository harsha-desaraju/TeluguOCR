
# A simple benchmarking script to measure the speed improvements upon changes to the model code.

import time
import torch
import numpy as np


DEVICE = "mps"


def synchronize():
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == 'mps':
        torch.mps.synchronize()


def benchmark_model(model, batch_size: int = 16, seq_len: int = 128, embed_dim: int = 256, vocab_size: int = 16384,
                    num_iters: int = 100, warmup_iters: int = 10):
    """Function for benchmarking a model"""

    model_dtype = next(model.parameters())[0].dtype

    # Create the input
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=DEVICE)
    enc_out = torch.randn((batch_size, seq_len, embed_dim), device=DEVICE)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=DEVICE)

    model = model.to(DEVICE)

    if hasattr(model, "encoder_transformer_blocks"):
        inp = x
    elif hasattr(model, "decoder_transformer_blocks"):
        inp = (x, enc_out)
    else:
        raise TypeError("Unknown model type!")

    criterion = torch.nn.CrossEntropyLoss()

    # Warm up the GPU
    for _ in range(warmup_iters):
        logits = model(*inp) if isinstance(inp, tuple) else model(inp)

        loss = criterion(logits.view(-1, logits.shape[-1]), y.view(-1))
        loss.backward()
        model.zero_grad(set_to_none=True)

    synchronize()

    forward_times, backward_times = [], []

    for i in range(num_iters):
        synchronize()
        start = time.perf_counter()

        logits = model(*inp) if isinstance(inp, tuple) else model(inp)

        synchronize()
        mid = time.perf_counter()

        loss = criterion(logits.view(-1, logits.shape[-1]), y.view(-1))
        loss.backward()

        synchronize()
        end = time.perf_counter()

        forward_times.append(mid-start)
        backward_times.append(end-mid)

        model.zero_grad(set_to_none=True)

    avg_fwd_time = np.mean(forward_times)
    avg_bwd_time = np.mean(backward_times)

    tokens_per_sec = (batch_size * seq_len) / avg_fwd_time

    print(f"Avg forward time : {avg_fwd_time:.5f} sec")
    print(f"Avg backward time: {avg_bwd_time:.5f} sec")
    print(f"Tokens/sec: {tokens_per_sec:.2f}")


if __name__ == '__main__':

    from src.model_training.transformer_models.scratch_implementation import Encoder, Decoder

    vocab_size = 16384
    embed_dim = 512
    ctx_len = 256
    num_encoder_blocks = 12
    num_decoder_blocks = 12
    num_heads = 8
    ffn_hidden_size = 4 * embed_dim

    batch_size, num_tokens = 16, ctx_len

    encoder = Encoder(vocab_size, embed_dim, ctx_len, num_encoder_blocks, num_heads, ffn_hidden_size)
    decoder = Decoder(vocab_size, embed_dim, ctx_len, num_decoder_blocks, num_heads, ffn_hidden_size)

    print(f"Device: {DEVICE}")

    num_params = 0
    for layer in encoder.parameters():
        num_params += layer.numel()
    print(f"No. of parameters in Encoder model: {num_params}")

    num_params = 0
    for layer in decoder.parameters():
        num_params += layer.numel()
    print(f"No. of parameters in Decoder model: {num_params}")

    print("\nEncoder Benchmark:")
    benchmark_model(encoder, batch_size, seq_len=num_tokens, embed_dim=embed_dim, vocab_size=vocab_size)

    print("\nDecoder Benchmark:")
    benchmark_model(decoder, batch_size, seq_len=num_tokens, embed_dim=embed_dim, vocab_size=vocab_size)