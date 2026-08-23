import torch
from transformers import AutoModelForCausalLM

model_name = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)

# 输入嵌入层，本质就是一个 nn.Embedding
emb = model.get_input_embeddings()
print(emb.weight.shape)


import torch.nn.functional as F
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name)

def vec(word):
    # 中文词经 BPE 可能拆成多个子词，这里取第一个子词的向量做演示
    token_ids = tokenizer.encode(word, add_special_tokens=False)
    print(word, token_ids)
    token_id = token_ids[0]
    return emb.weight[token_id]

pairs = [("猫", "狗"), ("猫", "汽车")]
for a, b in pairs:
    sim = F.cosine_similarity(vec(a), vec(b), dim=0)
    print(f"{a} vs {b}: {sim.item():.4f}")
