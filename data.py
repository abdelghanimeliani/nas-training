import pandas as pd

df1 = pd.read_csv("./data/timetrack/compute_dataset.csv")
df2 = pd.read_csv("./data/materna/15.csv")

# Take the first 8000 values from the chosen columns
ts1 = df1['averageCpuUsagePerse'].iloc[:8000].astype(float)
ts2 = df2['CPU usage [%]'].iloc[:8000].str.replace(',', '.').astype(float)

# Min-max normalization function
def min_max_norm(series):
    return (series - series.min()) / (series.max() - series.min())

ts1_norm = min_max_norm(ts1)
ts2_norm = min_max_norm(ts2)

# Save normalized data
ts1_norm.to_csv("./dataset1_8000.csv", index=False)
ts2_norm.to_csv("./dataset2_8000.csv", index=False)
