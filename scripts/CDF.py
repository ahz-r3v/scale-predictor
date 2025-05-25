import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取 CSV 文件
df = pd.read_csv("edo_400_shift.csv")

# 提取 RMSE 列并排序
rmse_values = np.sort(df["rmse"].values)

# 计算 CDF
cdf = np.arange(1, len(rmse_values) + 1) / len(rmse_values)

# 绘图
plt.figure(figsize=(8, 5))
plt.plot(rmse_values, cdf, marker='.', linestyle='-', label="CDF of RMSE")

# 设置对数横坐标
plt.xscale("log")

# 图像细节
plt.xlabel("RMSE (log scale)")
plt.ylabel("Cumulative Probability")
plt.title("CDF of Prediction Error (RMSE, log scale)")
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()

# 保存图像
plt.savefig("rmse_cdf_logx.png", dpi=300)
plt.show()