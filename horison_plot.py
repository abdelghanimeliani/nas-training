import pandas as pd
import matplotlib.pyplot as plt

def plot_mape_horizontal(csv_file="./values/metrics_horizon.csv"):
    # 1. Load and clean data
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=['Min_MAPE'])
    df['Dataset'] = df['Dataset'].str.replace('./', '', regex=False).str.replace('.csv', '', regex=False)
    df['Horizon'] = df['Horizon'].astype(int)
    df['Min_MAPE'] = df['Min_MAPE'].astype(float)

    # 2. Extract unique categories
    datasets = df['Dataset'].unique()
    methods = df['Method'].unique()  # This contains your 5 methods
    
    # Define distinct styles for the Dataset lines
    markers = ['o', 's', '^', 'D', 'X', 'v', 'p']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    dataset_styles = {dataset: {'marker': markers[i % len(markers)], 'color': colors[i % len(colors)]} 
                      for i, dataset in enumerate(datasets)}

    # 3. Setup Grid: Exactly 1 row and 5 columns
    cols = 5
    rows = 1
    
    # Tweak figsize to roughly match 5 squares + margins (e.g., 20x5 or 25x6)
    fig, axes = plt.subplots(rows, cols, figsize=(25, 5.5), sharex=True)

    # 4. Loop over each method and plot
    for i, method in enumerate(methods):
        ax = axes[i]
        
        # --- THE FIX: Force the physical plot box to be exactly square ---
        ax.set_box_aspect(1) 
        
        method_df = df[df['Method'] == method]
        
        for dataset in datasets:
            dataset_df = method_df[method_df['Dataset'] == dataset].sort_values('Horizon')
            
            if not dataset_df.empty:
                # Force NumPy arrays to prevent the environment indexing crash
                x_data = dataset_df['Horizon'].to_numpy()
                y_data = dataset_df['Min_MAPE'].to_numpy()
                
                ax.plot(
                    x_data, 
                    y_data, 
                    label=dataset,
                    linewidth=2,
                    marker=dataset_styles[dataset]['marker'],
                    color=dataset_styles[dataset]['color'],
                    markersize=6
                )
        
        # Subplot appearance
        ax.set_title(method, fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel("Horizon Steps", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xticks(sorted(df['Horizon'].unique()))
        
        # Only put the Y-label on the first plot to keep things clean
        if i == 0:
            ax.set_ylabel("Min MAPE", fontsize=11, fontweight='bold')

    # 5. Global legend at the very top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
               ncol=len(datasets), frameon=True, fontsize=11)

    # Give the legend breathing room at the top
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    
    # Save the wide banner layout
    plt.savefig("mape_horizontal_5cols.pdf", dpi=300)
    print("Plot successfully saved as 'mape_horizontal_5cols.pdf'!")
    plt.show()

if __name__ == "__main__":
    plot_mape_horizontal()