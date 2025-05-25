import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 模型名与对应文件路径
model_files = {
    # "N-HiTS": "edo_400_top10.csv",
    # "Linear Regression": "lr_400_top10.csv",
    # "Knative": "knative_400_top10.csv"
    "N-HiTS": "edo_400_shift.csv",
    "Linear Regression": "lr_400_shift.csv",
    "Knative": "knative_400_shift.csv"
    # 可以继续添加更多模型
}

plt.figure(figsize=(8, 5))

for label, file_path in model_files.items():
    try:
        df = pd.read_csv(file_path)
        rmse_values = np.sort(df["rmse"].values)
        rmse_values = rmse_values[rmse_values > 0]  # 移除 0 值，避免 log 坐标报错
        cdf = np.arange(1, len(rmse_values) + 1) / len(rmse_values)
        plt.plot(rmse_values, cdf, linestyle='-', marker='.', label=label)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

# 对数横轴
plt.xscale("log")

# 图像细节
plt.xlabel("RMSE (log scale)")
plt.ylabel("Cumulative Probability")
plt.title("CDF of Prediction Error (RMSE, log scale)")
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()

# 保存图像
plt.savefig("rmse_cdf_comparison_logx.png", dpi=300)
plt.show()
