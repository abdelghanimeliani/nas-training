'''
this script is mainly to generate the datasets for the experiments, it maunly extract the 
needed columns from materna and time track datasets and put them in other csv files
for the scrips generate one single column csv files with the first 8000 values of the
chosen columns from each dataset.
'''

import pandas as pd

df1 = pd.read_csv("./data/timetrack/compute_dataset.csv")
df2 = pd.read_csv("./data/materna/15.csv")
df3 = pd.read_csv("./data/alibaba/data.csv")

# Take the first 8000 values from the chosen columns
ts1 = df1['averageCpuUsagePerse'].iloc[:140].astype(float)
ts2 = df2['CPU usage [%]'].iloc[:140].str.replace(',', '.').astype(float)
ts3 = df3['util:CPU'].iloc[:140].astype(float)

# Min-max normalization function
def min_max_norm(series):
    return (series - series.min()) / (series.max() - series.min())

ts1_norm = min_max_norm(ts1)
ts2_norm = min_max_norm(ts2)
ts3_norm = min_max_norm(ts3)

# Save normalized data
ts1_norm.to_csv("./dataset1_140.csv", index=False)
ts2_norm.to_csv("./dataset2_140.csv", index=False)
ts3_norm.to_csv("./dataset3_140.csv", index=False)
