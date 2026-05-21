import pandas as pd
import numpy as np

# Load datasets
# Note: Paths are kept as per your environment structure
df1 = pd.read_csv("./data/timetrack/compute_dataset.csv")
df2 = pd.read_csv("./data/materna/15.csv")
df3 = pd.read_csv("./data/alibaba/data.csv")

def clean_series(series):
    """
    Forces valid numerical types and safely fills gaps for time-series.
    """
    # 1. Force convert to numeric. Any spaces, letters, or garbage become NaN.
    numeric_series = pd.to_numeric(series, errors='coerce')
    
    # 2. Interpolate linearly to fill the gaps without breaking time continuity.
    clean_series = numeric_series.interpolate(method='linear')
    
    # 3. Catch-all for edge cases (e.g., if the very first row was NaN)
    clean_series = clean_series.bfill().ffill()
    
    return clean_series

# Extract and clean raw series
# 1. TimeTrack (45s)
ts1_raw = clean_series(df1['averageCpuUsagePerse'])

# 2. Materna (5m) - handling comma as decimal first, then cleaning
materna_str = df2['CPU usage [%]'].astype(str).str.replace(',', '.')
ts2_raw = clean_series(materna_str).iloc[:1000]

# 3. Alibaba (5m)
ts3_raw = clean_series(df3['util:CPU']).iloc[:1000]

# Define the "Foundational" base slices
# tt_base: first 10k samples
tt_base = ts1_raw.iloc[:1000]
# mat_base: first 1.5k samples 
mat_base = ts2_raw # already sliced above
# ali_base: first 1.5k samples
ali_base = ts3_raw # already sliced above

# Define "Local" data: last 500 samples of TimeTrack
local_raw = ts1_raw.iloc[1000:1501]

# Normalization function
def min_max_norm(series):
    # Ensure no division by zero if a completely flat series exists
    range_val = series.max() - series.min()
    if range_val == 0:
        return series - series.min() 
    return (series - series.min()) / range_val

# Normalize components before mixing 
# This ensures each data source contributes features on the same [0, 1] scale
tt_norm = min_max_norm(tt_base)
mat_norm = min_max_norm(mat_base)
ali_norm = min_max_norm(ali_base)
local_norm = min_max_norm(local_raw)

# Create the 7 combinations
datasets = {
    "local_only": local_norm,
    "tt_only": tt_norm,
    "mat_only": mat_norm,
    "ali_only": ali_norm,
    "local_plus_tt": pd.concat([tt_norm, local_norm], ignore_index=True),
    "local_plus_mat": pd.concat([mat_norm, local_norm], ignore_index=True),
    "local_plus_ali": pd.concat([ali_norm, local_norm], ignore_index=True)
}

# Save to CSV files
for name, data in datasets.items():
    file_path = f"./{name}.csv"
    data.to_csv(file_path, index=False, header=['cpu_usage'])
    print(f"Saved: {file_path} | Samples: {len(data)}")