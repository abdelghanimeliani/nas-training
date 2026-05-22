import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
# Load your CSV data
# df = pd.read_csv('average_computational_complexity_results.csv')  # Replace with your actual file path
df = pd.read_csv('results.csv') 

print("DataFrame loaded successfully!")
print(f"Shape: {df.shape}")

# Exclude MLP and TCN models
df = df[~df['Model'].isin(['MLP', 'TCN'])]

# Create the x-axis labels as tuples (Dataset2_3_Size, TimeTrack_Size)
df['Data_Pair'] = df.apply(lambda row: f"({row['Dataset2_3_Size']};{row['TimeTrack_Size']})", axis=1)

# Get unique data pairs and sort them
data_pairs = sorted(df['Data_Pair'].unique(), 
                   key=lambda x: (int(x.split(';')[0].strip('(')), int(x.split(';')[1].strip(')'))))

print("Data pairs:", data_pairs)
print("Models included:", df['Model'].unique())

# Define metrics to plot - using Avg CPU instead of Max CPU
metrics = ['Train Time (s)', 'Memory Used (MB)', 'Avg CPU (%)']
metric_names = ['Training Time (s)', 'Memory Used (MB)', 'Average CPU Usage (%)']

# Define colors for each dataset
dataset_colors = {
    'TimeTrack': '#1f77b4',  # blue
    'Dataset2': '#ff7f0e',   # orange
    'Dataset3': '#2ca02c'    # green
}

models = df['Model'].unique()

# Create ONE comprehensive figure with 3 rows (metrics) x 5 columns (models)
print("\n" + "="*60)
print("COMPREHENSIVE COMPARISON - 3 METRICS x 5 MODELS")
print("="*60)

fig, axes = plt.subplots(3, 5, figsize=(20, 12))
fig.suptitle('Computational Resources Comparison: TimeTrack vs GWA-Materna-13 vs Alibaba-CD-2018', 
             fontsize=16, fontweight='bold', y=0.98)

# Plot: rows = metrics, columns = models
for row_idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
    for col_idx, model in enumerate(models):
        ax = axes[row_idx, col_idx]
        model_data = df[df['Model'] == model]
        
        # Plot each dataset
        for dataset in ['TimeTrack', 'Dataset2', 'Dataset3']:
            dataset_data = model_data[model_data['Dataset'] == dataset]
            
            if not dataset_data.empty:
                # Sort by data pair
                dataset_data = dataset_data.sort_values('Data_Pair', key=lambda x: x.map({pair: i for i, pair in enumerate(data_pairs)}))
                
                x_positions = np.arange(len(dataset_data))
                y_values = dataset_data[metric].values
                dataset_label=''
                if dataset== "TimeTrack":
                    dataset_label="TimeTrack"
                elif dataset=="Dataset2":
                    dataset_label="GWA-Materna-13"
                elif dataset=="Dataset3":
                    dataset_label="Alibaba-CDV-2018"
                
                ax.plot(x_positions, y_values, 
                       marker='o', 
                       linewidth=2, 
                       markersize=5,
                       color=dataset_colors[dataset],
                       label=dataset_label)
        
        # Y-axis label only for first column
        if col_idx == 0:
            ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
        
        # X-axis label only for last row
        if row_idx == 2:  # Last row (index 2)
            ax.set_xlabel('Data Injection', fontsize=10, fontweight='bold')
        
        ax.set_xticks(range(len(data_pairs)))
        ax.set_xticklabels(data_pairs, rotation=45, fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Title for first row only (metric names)
        if row_idx == 0:
            ax.set_title(model, fontsize=13, fontweight='bold')
        
        # Set logarithmic scale for time
        if metric == 'Train Time (s)':
            ax.set_yscale('log')
        
        # Add legend only to first subplot
        if row_idx == 0 and col_idx == 0:
            ax.legend(fontsize=10, loc='best')

plt.tight_layout()
plt.subplots_adjust(top=0.93)
    # Save the combined figure
os.makedirs("./plots/", exist_ok=True)
png_path = "./plots/computational_resources_comparison.png"
pdf_path = "./plots/computational_resources_comparison.pdf"

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.close()