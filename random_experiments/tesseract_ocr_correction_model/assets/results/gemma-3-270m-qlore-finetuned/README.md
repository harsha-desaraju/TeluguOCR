---
base_model: google/gemma-3-270m
library_name: peft
model_name: gemma-3-270m-qlore-finetuned
tags:
- base_model:adapter:google/gemma-3-270m
- lora
- sft
- transformers
- trl
licence: license
pipeline_tag: text-generation
---

# Model Card for gemma-3-270m-qlore-finetuned

This model is a fine-tuned version of [google/gemma-3-270m](https://huggingface.co/google/gemma-3-270m).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 


This model was trained with SFT.

### Framework versions

- PEFT 0.18.0
- TRL: 0.25.1
- Transformers: 4.57.1
- Pytorch: 2.6.0+cu124
- Datasets: 4.4.1
- Tokenizers: 0.22.1

## Citations



Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```