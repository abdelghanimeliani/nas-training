import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# HEATMAP COMPARISON PLOTS
# ============================================================

# Load the results
df_all_results = pd.read_csv("computational_complexity_scaling_results.csv")

# Set style for clean plots
plt.style.use('default')

# Create a figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Model Training Time and Resource Usage Comparison Across Datasets and Sizes', fontsize=16, fontweight='bold')

# ============================================================
# TRAINING TIME HEATMAPS
# ============================================================

# Heatmap 1: Training Time for TimeTrack dataset
time_track_pivot = df_all_results[df_all_results['Dataset'] == 'TimeTrack'].pivot_table(
    index='Model', 
    columns='TimeTrack_Size', 
    values='Train Time (s)',
    aggfunc='mean'
)
sns.heatmap(time_track_pivot, ax=axes[0, 0], cmap='YlOrRd', annot=True, fmt='.2f', cbar_kws={'label': 'Training Time (s)'})
axes[0, 0].set_title('Training Time - TimeTrack Dataset\n(Lower is Better)')
axes[0, 0].set_xlabel('Data Size')
axes[0, 0].set_ylabel('Model')

# Heatmap 2: Training Time for Dataset2
dataset2_pivot = df_all_results[df_all_results['Dataset'] == 'Dataset2'].pivot_table(
    index='Model', 
    columns='Dataset2_3_Size', 
    values='Train Time (s)',
    aggfunc='mean'
)
sns.heatmap(dataset2_pivot, ax=axes[0, 1], cmap='YlOrRd', annot=True, fmt='.2f', cbar_kws={'label': 'Training Time (s)'})
axes[0, 1].set_title('Training Time - Dataset2\n(Lower is Better)')
axes[0, 1].set_xlabel('Data Size')
axes[0, 1].set_ylabel('Model')

# Heatmap 3: Training Time for Dataset3
dataset3_pivot = df_all_results[df_all_results['Dataset'] == 'Dataset3'].pivot_table(
    index='Model', 
    columns='Dataset2_3_Size', 
    values='Train Time (s)',
    aggfunc='mean'
)
sns.heatmap(dataset3_pivot, ax=axes[0, 2], cmap='YlOrRd', annot=True, fmt='.2f', cbar_kws={'label': 'Training Time (s)'})
axes[0, 2].set_title('Training Time - Dataset3\n(Lower is Better)')
axes[0, 2].set_xlabel('Data Size')
axes[0, 2].set_ylabel('Model')

# ============================================================
# RESOURCE USAGE HEATMAPS
# ============================================================

# Heatmap 4: Memory Usage for TimeTrack dataset
memory_time_track_pivot = df_all_results[df_all_results['Dataset'] == 'TimeTrack'].pivot_table(
    index='Model', 
    columns='TimeTrack_Size', 
    values='Memory Used (MB)',
    aggfunc='mean'
)
sns.heatmap(memory_time_track_pivot, ax=axes[1, 0], cmap='Blues', annot=True, fmt='.1f', cbar_kws={'label': 'Memory (MB)'})
axes[1, 0].set_title('Memory Usage - TimeTrack Dataset\n(Lower is Better)')
axes[1, 0].set_xlabel('Data Size')
axes[1, 0].set_ylabel('Model')

# Heatmap 5: CPU Usage for TimeTrack dataset
cpu_time_track_pivot = df_all_results[df_all_results['Dataset'] == 'TimeTrack'].pivot_table(
    index='Model', 
    columns='TimeTrack_Size', 
    values='Avg CPU (%)',
    aggfunc='mean'
)
sns.heatmap(cpu_time_track_pivot, ax=axes[1, 1], cmap='Reds', annot=True, fmt='.1f', cbar_kws={'label': 'CPU (%)'})
axes[1, 1].set_title('CPU Usage - TimeTrack Dataset\n(Lower is Better)')
axes[1, 1].set_xlabel('Data Size')
axes[1, 1].set_ylabel('Model')

# Heatmap 6: Combined Resource Score for TimeTrack dataset
# Create a normalized combined resource score (Memory + CPU)
resource_time_track = df_all_results[df_all_results['Dataset'] == 'TimeTrack'].copy()
# Normalize memory and CPU to 0-1 scale
resource_time_track['Memory_Norm'] = (resource_time_track['Memory Used (MB)'] - resource_time_track['Memory Used (MB)'].min()) / (resource_time_track['Memory Used (MB)'].max() - resource_time_track['Memory Used (MB)'].min())
resource_time_track['CPU_Norm'] = (resource_time_track['Avg CPU (%)'] - resource_time_track['Avg CPU (%)'].min()) / (resource_time_track['Avg CPU (%)'].max() - resource_time_track['Avg CPU (%)'].min())
resource_time_track['Resource_Score'] = (resource_time_track['Memory_Norm'] + resource_time_track['CPU_Norm']) / 2

resource_pivot = resource_time_track.pivot_table(
    index='Model', 
    columns='TimeTrack_Size', 
    values='Resource_Score',
    aggfunc='mean'
)
sns.heatmap(resource_pivot, ax=axes[1, 2], cmap='Purples', annot=True, fmt='.3f', cbar_kws={'label': 'Resource Score (0-1)'})
axes[1, 2].set_title('Combined Resource Score - TimeTrack Dataset\n(Lower is Better)')
axes[1, 2].set_xlabel('Data Size')
axes[1, 2].set_ylabel('Model')

plt.tight_layout()
plt.savefig('training_time_resource_heatmaps.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================
# DATASET COMPARISON HEATMAPS
# ============================================================

# Compare datasets for the largest common size
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
fig2.suptitle('Model Performance Comparison Across Datasets (Largest Common Size)', fontsize=16, fontweight='bold')

# Find the largest common size that has data for all datasets
common_sizes_time = df_all_results[df_all_results['Dataset'] == 'TimeTrack']['TimeTrack_Size'].unique()
common_sizes_other = df_all_results[df_all_results['Dataset'] == 'Dataset2']['Dataset2_3_Size'].unique()

if len(common_sizes_time) > 0 and len(common_sizes_other) > 0:
    largest_time_size = common_sizes_time.max()
    largest_other_size = common_sizes_other.max()
    
    # Training time comparison across datasets
    time_comparison_data = []
    for dataset in ['TimeTrack', 'Dataset2', 'Dataset3']:
        if dataset == 'TimeTrack':
            dataset_data = df_all_results[(df_all_results['Dataset'] == dataset) & (df_all_results['TimeTrack_Size'] == largest_time_size)]
        else:
            dataset_data = df_all_results[(df_all_results['Dataset'] == dataset) & (df_all_results['Dataset2_3_Size'] == largest_other_size)]
        
        for model in dataset_data['Model'].unique():
            model_data = dataset_data[dataset_data['Model'] == model]
            if not model_data.empty:
                time_comparison_data.append({
                    'Model': model,
                    'Dataset': dataset,
                    'Train_Time': model_data['Train Time (s)'].mean()
                })
    
    time_comparison_df = pd.DataFrame(time_comparison_data)
    time_comparison_pivot = time_comparison_df.pivot_table(
        index='Model', 
        columns='Dataset', 
        values='Train_Time'
    )
    sns.heatmap(time_comparison_pivot, ax=axes2[0], cmap='YlOrRd', annot=True, fmt='.2f', cbar_kws={'label': 'Training Time (s)'})
    axes2[0].set_title(f'Training Time Comparison\n(Size: {largest_time_size}/{largest_other_size})')
    
    # Memory usage comparison across datasets
    memory_comparison_data = []
    for dataset in ['TimeTrack', 'Dataset2', 'Dataset3']:
        if dataset == 'TimeTrack':
            dataset_data = df_all_results[(df_all_results['Dataset'] == dataset) & (df_all_results['TimeTrack_Size'] == largest_time_size)]
        else:
            dataset_data = df_all_results[(df_all_results['Dataset'] == dataset) & (df_all_results['Dataset2_3_Size'] == largest_other_size)]
        
        for model in dataset_data['Model'].unique():
            model_data = dataset_data[dataset_data['Model'] == model]
            if not model_data.empty:
                memory_comparison_data.append({
                    'Model': model,
                    'Dataset': dataset,
                    'Memory': model_data['Memory Used (MB)'].mean()
                })
    
    memory_comparison_df = pd.DataFrame(memory_comparison_data)
    memory_comparison_pivot = memory_comparison_df.pivot_table(
        index='Model', 
        columns='Dataset', 
        values='Memory'
    )
    sns.heatmap(memory_comparison_pivot, ax=axes2[1], cmap='Blues', annot=True, fmt='.1f', cbar_kws={'label': 'Memory (MB)'})
    axes2[1].set_title(f'Memory Usage Comparison\n(Size: {largest_time_size}/{largest_other_size})')
    
    # CPU usage comparison across datasets
    cpu_comparison_data = []
    for dataset in ['TimeTrack', 'Dataset2', 'Dataset3']:
        if dataset == 'TimeTrack':
            dataset_data = df_all_results[(df_all_results['Dataset'] == dataset) & (df_all_results['TimeTrack_Size'] == largest_time_size)]
        else:
            dataset_data = df_all_results[(df_all_results['Dataset'] == dataset) & (df_all_results['Dataset2_3_Size'] == largest_other_size)]
        
        for model in dataset_data['Model'].unique():
            model_data = dataset_data[dataset_data['Model'] == model]
            if not model_data.empty:
                cpu_comparison_data.append({
                    'Model': model,
                    'Dataset': dataset,
                    'CPU': model_data['Avg CPU (%)'].mean()
                })
    
    cpu_comparison_df = pd.DataFrame(cpu_comparison_data)
    cpu_comparison_pivot = cpu_comparison_df.pivot_table(
        index='Model', 
        columns='Dataset', 
        values='CPU'
    )
    sns.heatmap(cpu_comparison_pivot, ax=axes2[2], cmap='Reds', annot=True, fmt='.1f', cbar_kws={'label': 'CPU (%)'})
    axes2[2].set_title(f'CPU Usage Comparison\n(Size: {largest_time_size}/{largest_other_size})')

plt.tight_layout()
plt.savefig('dataset_comparison_heatmaps.png', dpi=300, bbox_inches='tight')
plt.show()

print("Heatmap plots saved:")
print("- training_time_resource_heatmaps.png")
print("- dataset_comparison_heatmaps.png")