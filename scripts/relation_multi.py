import pandas as pd
import matplotlib.pyplot as plt

# 模型名与对应文件路径
model_files = {
    # "N-HiTS": "edo_400_top10.csv",
    # "Linear Regression": "lr_400_top10.csv",
    # "Knative": "knative_400_top10.csv"
    "N-HiTS": "edo_400_shift.csv",
    "Linear Regression": "lr_400_shift.csv",
    "Knative": "knative_400_shift.csv"
}

plt.figure(figsize=(8, 5))

for label, file_path in model_files.items():
    try:
        df = pd.read_csv(file_path)
        df = df[(df["mean_y"] > 0) & (df["rmse"] > 0)]
        plt.scatter(df["mean_y"], df["rmse"], alpha=0.6, label=label)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

# 对数坐标轴
plt.xscale("log")
plt.yscale("log")

# 图像细节
plt.xlabel("Average Concurrency (per second, log scale)")
plt.ylabel("Prediction Error (RMSE, log scale)")
plt.title("Prediction Error vs. Average Concurrency (log-log)")
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()

# 保存图像
plt.savefig("rmse_vs_concurrency_comparison_loglog.png", dpi=300)
plt.show()
