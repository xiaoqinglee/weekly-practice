import math

def sinusoidal_pe(pos, d_model=8):
    # 偶数维度填 sin，奇数维度填 cos，频率随维度指数下降
    return [
        math.sin(pos / 10000 ** (i / d_model)) if i % 2 == 0
        else math.cos(pos / 10000 ** ((i - 1) / d_model))
        for i in range(d_model)
    ]

for pos in range(4):
    print(f"位置 {pos}:", [round(x, 2) for x in sinusoidal_pe(pos)])

# 对比一下相邻位置和相隔很远的位置，编码差多少
def dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

print("位置 0 和 1 的距离:", round(dist(sinusoidal_pe(0), sinusoidal_pe(1)), 2))
# print("位置 1 和 2 的距离:", round(dist(sinusoidal_pe(1), sinusoidal_pe(2)), 2))
# print("位置 2 和 3 的距离:", round(dist(sinusoidal_pe(2), sinusoidal_pe(3)), 2))
# print("位置 3 和 4 的距离:", round(dist(sinusoidal_pe(3), sinusoidal_pe(4)), 2))

print("位置 0 和 2 的距离:", round(dist(sinusoidal_pe(0), sinusoidal_pe(2)), 2))
print("位置 0 和 3 的距离:", round(dist(sinusoidal_pe(0), sinusoidal_pe(3)), 2))
print("位置 0 和 4 的距离:", round(dist(sinusoidal_pe(0), sinusoidal_pe(4)), 2))
print("位置 0 和 5 的距离:", round(dist(sinusoidal_pe(0), sinusoidal_pe(5)), 2))
print("位置 0 和 6 的距离:", round(dist(sinusoidal_pe(0), sinusoidal_pe(6)), 2))
print("位置 0 和 7 的距离:", round(dist(sinusoidal_pe(0), sinusoidal_pe(7)), 2))
print("位置 0 和 8 的距离:", round(dist(sinusoidal_pe(0), sinusoidal_pe(8)), 2))
print("位置 0 和 9 的距离:", round(dist(sinusoidal_pe(0), sinusoidal_pe(9)), 2))
print("位置 0 和 10 的距离:", round(dist(sinusoidal_pe(0), sinusoidal_pe(10)), 2))

print("位置 0 和 100 的距离:", round(dist(sinusoidal_pe(0), sinusoidal_pe(100)), 2))
