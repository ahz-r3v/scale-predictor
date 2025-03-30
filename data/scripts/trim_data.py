import pandas as pd

df = pd.read_csv("../top10_train.csv")

filtered_df = df[~(
    ((df["timestamp"] >= 0) & (df["timestamp"] <= 59)) |
    ((df["timestamp"] >= 3561) & (df["timestamp"] <= 3620))
)]

filtered_df.to_csv("../top10_train_trimed.csv", index=False)

print(f"lines before: {len(df)}, lines after: {len(filtered_df)}")