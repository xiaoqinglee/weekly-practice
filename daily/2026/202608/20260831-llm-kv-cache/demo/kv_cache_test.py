import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")

prompt = "The quick brown fox"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 第一次前向：处理完整 prompt
with torch.no_grad():
    out1 = model(**inputs, use_cache=True)
past1 = out1.past_key_values
print("缓存层数:", len(past1))
print("第 0 层 K 形状:", past1.layers[0].keys.shape)
print("第 0 层 V 形状:", past1.layers[0].values.shape)

# 第二次前向：只输入新 token，带上已有缓存
next_token = torch.argmax(out1.logits[0, -1, :]).reshape(1, 1)
with torch.no_grad():
    out2 = model(next_token, past_key_values=past1, use_cache=True)
past2 = out2.past_key_values
print("第二次前向后第 0 层 K 形状:", past2.layers[0].keys.shape)
print("第二次前向后第 0 层 V 形状:", past2.layers[0].values.shape)
