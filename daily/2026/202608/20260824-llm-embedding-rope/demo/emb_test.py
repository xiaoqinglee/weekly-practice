import torch
from transformers import AutoModelForCausalLM

model_name = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)

# 输入嵌入层，本质就是一个 nn.Embedding
emb = model.get_input_embeddings()
print(emb.weight.shape)
