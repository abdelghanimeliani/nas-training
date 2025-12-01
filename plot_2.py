import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your CSV data
# df = pd.read_csv('average_computational_complexity_results.csv')  # Replace with your actual file path
df = pd.read_csv('file1_updated.csv') 

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
            ax.set_xlabel('Data Injection', fontsize=10)
        
        ax.set_xticks(range(len(data_pairs)))
        ax.set_xticklabels(data_pairs, rotation=45, fontsize=9)
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
plt.show()

# Create a more compact version with better spacing
print("\n" + "="*60)
print("COMPACT VERSION - BETTER SPACING")
print("="*60)

fig, axes = plt.subplots(3, 5, figsize=(22, 10))
fig.suptitle('Computational Resources Comparison: TimeTrack vs GWA-Materna-13 vs Alibaba-CD-2018', 
             fontsize=14, fontweight='bold', y=0.95)

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
                
                line = ax.plot(x_positions, y_values, 
                       marker='o', 
                       linewidth=1.5, 
                       markersize=4,
                       color=dataset_colors[dataset],
                       label=dataset_label)[0]
                
                # Add value labels for extreme points
                if len(y_values) > 0:
                    max_idx = np.argmax(y_values)
                    min_idx = np.argmin(y_values)
                    
                    # Label max value
                    if y_values[max_idx] > np.mean(y_values) * 1.2:  # Only label if significantly higher than mean
                        ax.annotate(f'{y_values[max_idx]:.2f}', 
                                   (x_positions[max_idx], y_values[max_idx]),
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=7, bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))
        
        # Y-axis label
        if col_idx == 0:
            ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
        else:
            ax.tick_params(labelleft=False)  # Hide y-axis labels for non-first columns
        
        # X-axis label
        if row_idx == 2:
            ax.set_xlabel('Data Injection', fontsize=9)
            ax.set_xticklabels(data_pairs, rotation=45, fontsize=8)
        else:
            ax.set_xticklabels([])  # Hide x-axis labels for non-last rows
        
        ax.set_xticks(range(len(data_pairs)))
        ax.grid(True, alpha=0.3)
        
        # Model names as titles
        if row_idx == 0:
            ax.set_title(model, fontsize=12, fontweight='bold', pad=10)
        
        # Set logarithmic scale for time
        if metric == 'Train Time (s)':
            ax.set_yscale('log')
        
        # Add legend to first subplot only
        if row_idx == 0 and col_idx == 0:
            ax.legend(fontsize=9, loc='upper left', framealpha=0.9)

# Add a common legend at the bottom
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), 
           ncol=3, fontsize=11, framealpha=0.9)

plt.tight_layout()
plt.subplots_adjust(bottom=0.08, top=0.90)
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

summary_stats = df.groupby(['Dataset', 'Model'])[metrics].mean()
print(summary_stats)

# Calculate and print overall ratios
print("\n" + "="*60)
print("OVERALL RESOURCE RATIOS (TimeTrack vs Dataset2)")
print("="*60)

time_track_avg = df[df['Dataset'] == 'TimeTrack'][metrics].mean()
dataset2_avg = df[df['Dataset'] == 'Dataset2'][metrics].mean()

print(f"Training Time: {time_track_avg['Train Time (s)']/dataset2_avg['Train Time (s)']:.2f}x")
print(f"Memory Used: {time_track_avg['Memory Used (MB)']/dataset2_avg['Memory Used (MB)']:.2f}x") 
print(f"Average CPU: {time_track_avg['Avg CPU (%)']/dataset2_avg['Avg CPU (%)']:.2f}x")

# Show data size ratio
sample_comparison = df.groupby('Dataset')['Samples'].mean()
print(f"\nData Size Ratio: {sample_comparison['TimeTrack']/sample_comparison['Dataset2']:.2f}x")
print(f"(TimeTrack uses {sample_comparison['TimeTrack']/sample_comparison['Dataset2']:.2f}x more samples)")

# Model-specific insights
print("\n" + "="*60)
print("MODEL-SPECIFIC INSIGHTS")
print("="*60)

for model in models:
    model_data = df[df['Model'] == model]
    time_ratio = (model_data[model_data['Dataset'] == 'TimeTrack']['Train Time (s)'].mean() / 
                  model_data[model_data['Dataset'] == 'Dataset2']['Train Time (s)'].mean())
    print(f"{model}: Time ratio = {time_ratio:.2f}x")