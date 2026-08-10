from transformers import PreTrainedTokenizer
import os
from typing import Dict, List, Optional, Tuple
import json
import regex
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from transformers.modeling_outputs import CausalLMOutput
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass



# ------------ Tokenizer ------------
from src.telugu_ocr.tokenizer.grapheme import TeluguGraphemeTokenizer




@dataclass
class GPTConfig:
    vocab_size: int = 2048
    embed_dim: int = 512
    hidden_dim: int = 2048
    num_heads: int = 8
    num_layers: int = 12
    ctx_len: int = 1024
    dropout: float = 0.1


def calculate_positional_encodings(positions: torch.Tensor, embed_dim: int):
    i = torch.arange(embed_dim // 2, dtype=torch.float32)
    div_term = 10000 ** (2 * i / embed_dim)          # (D/2,)
    pos = positions.float().unsqueeze(1)              # (T, 1)
    args = pos / div_term                             # (T, D/2)
    enc = torch.zeros(len(positions), embed_dim)
    enc[:, 0::2] = torch.sin(args)
    enc[:, 1::2] = torch.cos(args)
    return enc


class SwiGLU(nn.Module):
    """Implement the SwiGLU activation function"""
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, embed_dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class MultiHeadAttention(nn.Module):
    """Implement multi head attention"""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.dropout   = dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask=None):
        # attn_mask: (B, 1, T, T) boolean — True means KEEP, False means MASK OUT
        B, T, _ = query.shape
        _, S, _ = key.shape

        queries = self.q_proj(query).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        keys    = self.k_proj(key).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        values  = self.v_proj(value).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        dropout_p = self.dropout if self.training else 0.0
        ctx_embeds = F.scaled_dot_product_attention(
            queries, keys, values,
            dropout_p=dropout_p,
            attn_mask=attn_mask,
            is_causal=False
        )

        ctx_embeds = ctx_embeds.transpose(1, 2).reshape(B, T, self.embed_dim)
        return self.out_proj(ctx_embeds)


class GPTTransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attention_layer = MultiHeadAttention(config.embed_dim, config.num_heads, config.dropout)
        self.mlp = nn.Sequential(
            SwiGLU(config.embed_dim, config.hidden_dim),
            nn.Dropout(config.dropout)
        )
        self.layer_norm1 = nn.LayerNorm(config.embed_dim)
        self.layer_norm2 = nn.LayerNorm(config.embed_dim)

    def forward(self, x: torch.Tensor, attn_mask = None):
        # x -> B, T, D
        normed = self.layer_norm1(x)
        x = x + self.attention_layer(normed, normed, normed, attn_mask=attn_mask)
        x = x + self.mlp(self.layer_norm2(x))
        return x



class GPTModel(nn.Module):
    _keys_to_ignore_on_save = None
    def __init__(self, config: GPTConfig, pad_index: int):
        super().__init__()
        self.pad_index = pad_index
        self.embedding_layer = nn.Embedding(config.vocab_size, config.embed_dim)
        self.register_buffer("positional_encodings", calculate_positional_encodings(torch.arange(config.ctx_len), config.embed_dim))
        self.transformer_blocks = nn.ModuleList([
            GPTTransformerBlock(config) for _ in range(config.num_layers)
        ])
        self.layer_norm = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear( config.embed_dim, config.vocab_size, bias=False)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _build_attn_mask(self, input_ids, attention_mask):
        """
        Builds a combined boolean causal + padding mask.
        SDPA expects: True = attend, False = ignore.
        Shape: (B, 1, T, T)
        """
        B, T = input_ids.shape
        device = input_ids.device
        # Causal mask: upper triangle is False (masked), lower triangle True
        causal = torch.ones(T, T, dtype=torch.bool, device=device).tril()  # (T, T)
        if attention_mask is not None:
            # attention_mask: (B, T), 1=real token, 0=pad
            # Expand to (B, 1, 1, T) so it broadcasts over query positions
            pad_mask = attention_mask.bool().unsqueeze(1).unsqueeze(2)      # (B, 1, 1, T)
            combined = causal.unsqueeze(0).unsqueeze(0) & pad_mask          # (B, 1, T, T)
        else:
            combined = causal.unsqueeze(0).unsqueeze(0)                     # (1, 1, T, T)
        return combined

    def forward(self, input_ids, attention_mask=None, labels = None):
        # input_ids -> (B, T)
        embeds = self.embedding_layer(input_ids)
        T = input_ids.shape[1]
        embeds = embeds + self.positional_encodings[:T].unsqueeze(0)
        attn_mask = self._build_attn_mask(input_ids, attention_mask)
        for block in self.transformer_blocks:
            embeds = block(embeds, attn_mask=attn_mask)
        embeds = self.layer_norm(embeds)
        logits = self.lm_head(embeds)
        loss = None

        # Always calculate the loss
        # shift for causal LM
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.pad_index
        )
        return CausalLMOutput(
            loss=loss,
            logits=logits
        )




def tokenize(example):
    out = tokenizer(
        example["text"],
        truncation=True,
        padding=False,
        max_length=CTX_LEN
    )
    return out


if __name__ == '__main__':

    CTX_LEN = 256

    tokenizer = TeluguGraphemeTokenizer(
        vocab_file="/kaggle/input/datasets/harshadesaraju1999/telugu-tokenizer-vocab/telugu-vocab.json")

    model_config = GPTConfig(
        vocab_size=len(tokenizer),
        embed_dim=512,
        hidden_dim=1368,  # 2.67 * 512 = 2/3 * 4 * hidden_dim
        num_heads=8,
        num_layers=16,
        ctx_len=CTX_LEN,
        dropout=0.1
    )

    model = GPTModel(model_config, pad_index=tokenizer.pad_token_id)

    total_params = 0
    for layer in model.parameters():
        total_params += layer.numel()
    print(f"Total parameters of the model are: {total_params}")

    ds = load_dataset(
        "harsha-desaraju/telugu-sanskrit-english-text-1024",
        columns=['text']
    )['train']

    data_dct = ds.train_test_split(test_size=0.001)

    train_dataset = data_dct['train']
    test_dataset = data_dct['test']

    train_dataset = train_dataset.with_transform(tokenize)
    test_dataset = test_dataset.with_transform(tokenize)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    OUTPUT_DIR = "./telugu-grapheme-gpt"
    BATCH_SIZE = 64
    EPOCHS = 1

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        # overwrite_output_dir=True,

        # --- Training Duration ---
        num_train_epochs=EPOCHS,

        # --- Batch Size & Accumulation ---
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,

        # --- Optimizer & Scheduler ---
        optim="adamw_torch_fused",
        learning_rate=3e-4,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        adam_beta2=0.95,

        ddp_find_unused_parameters=False,
        remove_unused_columns=False,

        # --- Precision & Performance ---
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=4,
        dataloader_persistent_workers=True,

        # --- Evaluation & Saving ---
        eval_strategy="steps",
        eval_steps=2000,
        save_strategy="steps",
        save_steps=2000,
        save_total_limit=2,

        # --- Logging ---
        logging_steps=100,
        logging_first_step=True,
        report_to="wandb",
        run_name="llm-training-1"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        # compute_metrics=compute_metrics
    )
    trainer.train()

    # --- Save final model ---
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/final_model.pt")
    print("Training complete for this session.")