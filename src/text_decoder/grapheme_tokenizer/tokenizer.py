"""
Telugu Grapheme Cluster Tokenizer — HuggingFace PreTrainedTokenizer
====================================================================
A deterministic, lossless tokenizer that splits Telugu text into Unicode
grapheme clusters (aksharas). Each grapheme cluster maps to exactly one
token ID.  No BPE / SentencePiece training required.

Install:
    pip install transformers regex

Usage:
    # --- Build vocabulary from a corpus ---
    tokenizer = TeluguGraphemeTokenizer.build_from_corpus(["path/to/file.txt"])
    tokenizer.save_pretrained("./telugu_tokenizer")

    # --- Load later ---
    tokenizer = TeluguGraphemeTokenizer.from_pretrained("./telugu_tokenizer")

    # --- Encode / decode ---
    ids   = tokenizer.encode("కారు నడిపాడు")
    text  = tokenizer.decode(ids)

    # --- Batch encode (returns tensors, padding, attention masks) ---
    batch = tokenizer(["కారు నడిపాడు", "తెలుగు"], padding=True, return_tensors="pt")
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import regex
from transformers import PreTrainedTokenizer



class TeluguGraphemeTokenizer(PreTrainedTokenizer):
    """HuggingFace-compatible grapheme-cluster tokenizer for Telugu.

    Parameters
    ----------
    vocab : dict[str, int]
        Mapping from grapheme string → token ID.  Must include all
        ``SPECIAL_TOKENS_LIST`` entries.
    add_bos_token : bool
        Automatically prepend ``[BOS]`` on ``encode()``.
    add_eos_token : bool
        Automatically append ``[EOS]`` on ``encode()``.
    """

    vocab_files_names = {"vocab_file": "vocab.json"}  # ← fix 1
    model_input_names = ["input_ids", "attention_mask"]  # ← fix 2

    def __init__(
            self,
            vocab_file: Optional[str] = None,
            vocab_list: Optional[list[str]] = None,
            add_bos_token: bool = True,
            add_eos_token: bool = True,
            pad_token: str = "[PAD]",
            unk_token: str = "[UNK]",
            bos_token: str = "[BOS]",
            eos_token: str = "[EOS]",
            mask_token: str = "[MASK]",
            **kwargs,
    ):
        vocab = {}

        if vocab_file is not None:
            with open(vocab_file, encoding="utf-8") as f:
                vocab = json.load(f)

        if not vocab and vocab_list is not None:
            for grapheme in vocab_list:
                vocab[grapheme] = len(vocab)

        if not vocab:
            raise AssertionError("Either `vocab_file` or `vocab_list` has to be given.")

        self.SPECIAL_TOKENS_LIST = [pad_token, unk_token, bos_token, eos_token, mask_token]

        for tok in self.SPECIAL_TOKENS_LIST:
            if tok not in vocab:
                vocab[tok] = len(vocab)

        self.vocab = vocab  # ← must be BEFORE super().__init__()
        self._inv_vocab = {v: k for k, v in vocab.items()}
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.UNK = unk_token

        super().__init__(
            pad_token=pad_token,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            mask_token=mask_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            padding_side="right",
            model_max_length=4096,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.vocab)

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        tokens = []
        for grapheme in regex.findall(r"\X", text):
            if grapheme in self.vocab:
                tokens.append(grapheme)
            else:
                for codepoint in grapheme:
                    tokens.append(codepoint)
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab.get(self.UNK, 1))

    def _convert_id_to_token(self, index: int) -> str:
        return self._inv_vocab.get(index, self.UNK)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        cleaned = [t for t in tokens if t not in self.SPECIAL_TOKENS_LIST]
        return "".join(cleaned)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos = [self.bos_token_id] if self.add_bos_token else []
        eos = [self.eos_token_id] if self.add_eos_token else []
        out = bos + token_ids_0 + eos
        if token_ids_1 is not None:
            out += bos + token_ids_1 + eos
        return out

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        if already_has_special_tokens:
            return super().get_special_tokens_mask(token_ids_0, token_ids_1, already_has_special_tokens=True)
        bos = [1] if self.add_bos_token else []
        eos = [1] if self.add_eos_token else []
        res = bos + [0] * len(token_ids_0) + eos
        if token_ids_1 is not None:
            res += bos + [0] * len(token_ids_1) + eos
        return res

    def save_vocabulary(
            self,
            save_directory: str,
            filename_prefix: Optional[str] = None,
    ) -> Tuple[str]:
        os.makedirs(save_directory, exist_ok=True)
        fname = (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        vocab_path = os.path.join(save_directory, fname)
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        return (vocab_path,)




if __name__ == "__main__":
    samples = [
        "కారు నడిపాడు",
        "తెలుగు భాష చాలా అందంగా ఉంది",
        "క్ష్ణ సంయుక్తాక్షరం",
        "రామాయణం",
    ]

    import ast

    vocab_list = []
    with open("grapheme_list.txt", 'r') as f:
        graphemes_list = f.readlines()
        for grapheme in graphemes_list:
            vocab_list.append(ast.literal_eval(grapheme.strip()))


    print(vocab_list)

    tokenizer = TeluguGraphemeTokenizer(vocab_list=vocab_list)

    for sample in samples:
        tokens = tokenizer(sample, add_special_tokens=True)
        decoded_text = tokenizer.decode(tokens['input_ids'], skip_special_tokens=False)
        print(tokens)
        print(decoded_text)
        print('='*50)


    # tokenizer.save_vocabulary("./", filename_prefix="telugu")
