import numpy as np

rng = np.random.default_rng(0)
d = 8
Wq, Wk, Wv = (rng.normal(size=(d, d)) for _ in range(3))  # 三个 8×8 投影矩阵

def attention(x):
    # 最简自注意力：softmax(QKᵀ/√d)V
    # @ 是矩阵乘法运算符：(4,8) @ (8,8) -> (4,8)，一次算出所有 token 的投影
    # Wq, Wk, Wv 是可学习的参数（真实模型里从训练中学出来，这里用随机矩阵代替）
    q, k, v = x @ Wq, x @ Wk, x @ Wv
    # k.T 是转置，(4,8) @ (8,4) -> (4,4)，得到 4 个 query 对 4 个 key 的两两打分表
    scores = q @ k.T / np.sqrt(d)
    # softmax：对每行分数先取 exp 再归一化，转成和为 1 的权重
    # 分数越高的 key 权重越大；减去最大值是为了防止 exp 数值溢出，不改变结果
    scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = scores / scores.sum(axis=-1, keepdims=True)
    return weights @ v

x = rng.normal(size=(4, d))  # 4 个 token，每个 8 维
out1 = attention(x)

perm = [2, 0, 3, 1]          # 打乱后的 token 顺序
# 花式索引：用数组当下标，按给定顺序取行
# x[perm] 等价于 [x[2], x[0], x[3], x[1]]，即打乱语序后的输入
out2 = attention(x[perm])

# argsort 求逆排列：inv[i] 表示原位置 i 的 token 被打乱到了哪里
# out2[inv] 把打乱的输出按原顺序排回去，这样才能和 out1 逐位置对比
inv = np.argsort(perm)
print(f"打乱前后的最大差异: {np.abs(out1 - out2[inv]).max():.2e}")
