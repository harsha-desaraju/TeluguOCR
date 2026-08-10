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




from src.telugu_ocr.models.text_decoder import (GPTConfig, GPTModel, GPTTransformerBlock,
                                              MultiHeadAttention, SwiGLU,
                                              calculate_positional_encodings)














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