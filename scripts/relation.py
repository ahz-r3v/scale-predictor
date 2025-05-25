import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
# df = pd.read_csv("edo_400_shift.csv")
df = pd.read_csv("lr_400_shift.csv")

# 过滤掉 mean_y 或 rmse 为 0 的数据（对数坐标不能有0或负值）
df = df[(df["mean_y"] > 0) & (df["rmse"] > 0)]

# 绘制散点图
plt.figure(figsize=(8, 5))
plt.scatter(df["mean_y"], df["rmse"], alpha=0.7)

# 设置对数坐标轴
plt.xscale("log")
plt.yscale("log")

# 添加标签与标题
plt.xlabel("Average Concurrency (per second, log scale)")
plt.ylabel("Prediction Error (RMSE, log scale)")
plt.title("Prediction Error vs. Average Concurrency (log-log)")
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.tight_layout()

# 保存图片
plt.savefig("rmse_vs_concurrency_loglog.png", dpi=300)
plt.show()
