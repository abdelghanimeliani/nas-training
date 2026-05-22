import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_execution_time_heatmap(csv_path="./values/time_steps.csv", colormap="viridis"):
    if not os.path.exists(csv_path):
        print(f"Error: The file '{csv_path}' was not found. Please verify the path.")
        return

    print(f"Loading execution data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. Standardize column names and strip whitespace
    df.columns = [col.lower().strip() for col in df.columns]
    df['dataset'] = df['dataset'].astype(str).str.strip()
    df['method'] = df['method'].astype(str).str.strip()

    # 2. Map dataset IDs to clean display names
    dataset_mapping = {
        "local_only": "Local",
        "tt_only": "TT",
        "local_plus_tt": "TT+Local",
        "mat_only": "Mat",
        "ali_only": "Ali",
        "local_plus_mat": "Mat+Local",
        "local_plus_ali": "Ali+Local"
    }
    df['dataset_display'] = df['dataset'].map(dataset_mapping)

    # 3. Handle axes sorting
    durations = sorted(df['duration'].unique())
    search_methods = ['GridSearch', 'random', 'tpe', 'anneal', 'evolution']
    search_methods = [m for m in search_methods if m in df['method'].unique()]
    dataset_labels = ['Local', 'TT', 'TT+Local', 'Mat', 'Ali', 'Mat+Local', 'Ali+Local']

    # 4. Set styling configuration
    sns.set_style("white")
    n_rows = 1
    n_cols = len(search_methods)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.5 * n_rows), squeeze=False)

    for c, method in enumerate(search_methods):  
        ax = axes[0, c]

        # Filter out observations for this optimization strategy
        method_df = df[df['method'] == method].copy()
        method_df['exec_time'] = method_df['exec_time'].fillna(0).astype(float)

        # Pivot to Matrix: Rows = Datasets, Columns = Number of evaluations
        matrix_df = method_df.pivot_table(index='dataset_display', columns='duration', values='exec_time', aggfunc='max')
        matrix_df = matrix_df.reindex(index=dataset_labels, columns=durations)

        # Draw Heatmap panel
        sns.heatmap(
            matrix_df,
            annot=True,
            fmt=".1f",  # Keep 1 decimal place to show runtime variants cleanly
            annot_kws={"size": 10, "weight": "bold"}, 
            xticklabels=durations,
            yticklabels=dataset_labels if c == 0 else [], 
            cmap=colormap,  # A sequential map like 'rocket_r' or 'magma_r' works wonderfully for execution cost
            cbar=True,
            ax=ax,
            cbar_kws={'label': 'Seconds'} if c == n_cols - 1 else {}
        )

        # Panel titles
        ax.set_title(method, fontsize=14, fontweight="bold", pad=12)

        # Labeling and target configuration highlights (Y-Axis)
        if c == 0:
            ax.set_ylabel("Dataset Configuration", fontsize=13, fontweight="bold")
            for label in ax.get_yticklabels():
                if label.get_text() == "TT+Local":
                    label.set_color("red")
                    label.set_fontweight("bold")
                    label.set_fontsize(12)
        else:
            ax.set_ylabel("")

        # Labeling (X-Axis)
        ax.set_xlabel("Evaluations Count", fontsize=12, fontweight="bold", labelpad=8)
        ax.tick_params(axis='x', rotation=0)

    plt.tight_layout()
    
    output_dir = "./plots/heatmaps/"
    os.makedirs(output_dir, exist_ok=True)
    
    out_pdf = os.path.join(output_dir, "exec_time_heatmap.pdf")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nHeatmap successfully exported to:\n -> {out_pdf}")

if __name__ == "__main__":
    plot_execution_time_heatmap()