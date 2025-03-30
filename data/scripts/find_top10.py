import pandas as pd

df = pd.read_csv("../out/400_test.csv")

y_sum = df.groupby("unique_id")["y"].sum().reset_index()

top10_ids = y_sum.sort_values("y", ascending=False).head(10)["unique_id"].tolist()

top10_df = df[df["unique_id"].isin(top10_ids)]

top10_df.to_csv("../out/top10_test.csv", index=False)

print("Top 10 unique_id by total y saved")
