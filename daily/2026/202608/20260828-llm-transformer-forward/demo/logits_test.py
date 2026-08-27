from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
print(model)

import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
inputs = tokenizer("我爱吃苹果", return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

print(outputs.logits.shape)
