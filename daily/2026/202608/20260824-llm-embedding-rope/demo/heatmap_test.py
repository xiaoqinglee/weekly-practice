import numpy as np
import matplotlib.pyplot as plt

d_model, max_pos = 64, 100
positions = np.arange(max_pos)[:, None]
omega = 1 / 10000 ** (2 * np.arange(d_model // 2) / d_model)
angles = positions * omega                    # (100, 32) 个角度

pe = np.zeros((max_pos, d_model))
pe[:, 0::2] = np.sin(angles)                  # 偶数维度填 sin
pe[:, 1::2] = np.cos(angles)                  # 奇数维度填 cos

plt.figure(figsize=(10, 4))
plt.imshow(pe.T, aspect="auto", cmap="RdBu")
plt.xlabel("position")
plt.ylabel("dimension")
plt.colorbar()
plt.show()
