import time
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

model_id = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "用三句话解释什么是 KV Cache"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 流式输出器：每生成一段文本就吐出来一段，而不是等全部生成完
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

start = time.perf_counter()
thread = Thread(
    target=model.generate,
    kwargs={**inputs, "streamer": streamer, "max_new_tokens": 128},
)
thread.start()

ttft = None
arrival_times = []
for chunk in streamer:
    now = time.perf_counter()
    if ttft is None:
        ttft = now - start  # 第一段文本到达，就是 TTFT
    arrival_times.append(now)
thread.join()

tpot = (arrival_times[-1] - arrival_times[0]) / max(len(arrival_times) - 1, 1)
print(f"TTFT: {ttft * 1000:.1f} ms")
print(f"TPOT: {tpot * 1000:.1f} ms")
print(f"总耗时: {(arrival_times[-1] - start) * 1000:.1f} ms")
