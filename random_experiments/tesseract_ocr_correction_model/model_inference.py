
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json



model_name = "google/gemma-3-270m"

tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name)


peft_model = PeftModel.from_pretrained(base_model, "correction_model/assets/results/gemma-3-270m-qlore-finetuned")
model = peft_model.merge_and_unload()
model.eval()


with open("correction_model/data/generated/source_texts/text_900_2.json", 'r') as f:
    dct = json.load(f)


messages =  {
    'messages': [
            {"role": "user", "content": f"{dct['ocr_text']}"},
            {"role": "assistant", "content": f"{dct['src_text']}"}
        ]
    }


inputs = tokenizer.encode(messages, return_tensors='pt')

output = model.generate(**inputs)

response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)


