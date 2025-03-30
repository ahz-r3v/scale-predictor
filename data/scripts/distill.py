import pandas as pd

df = pd.read_csv('../out/400_test.csv')
df['timestamp'] = df['timestamp'].astype(int)
df = df.sort_values(['unique_id', 'timestamp'])

TEN_MINUTES = 300

removed_ids = []

def is_valid_group(group):
    min_time = group['timestamp'].min()
    max_time = min_time + TEN_MINUTES
    first_10_min = group[group['timestamp'] < max_time]
    
    if (first_10_min['y'] == 0).all():
        removed_ids.append(group['unique_id'].iloc[0])
        return False
    return True


filtered_df = df.groupby('unique_id').filter(is_valid_group)
filtered_df.to_csv('../distilled/400_test_d.csv', index=False)

print("deleted unique_id:")
print(removed_ids)
