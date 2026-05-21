import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_trials_heatmap(csv_path="./values/trials_duration.csv", colormap="viridis"):
    if not os.path.exists(csv_path):
        print(f"Error: The file '{csv_path}' was not found.")
        return

    print(f"Loading trial data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. Normalize column headers and strip spaces
    df.columns = [col.lower().strip() for col in df.columns]
    df['dataset'] = df['dataset'].astype(str).str.strip()
    df['method'] = df['method'].astype(str).str.strip()

    # 2. Map raw CSV entries to clean display labels
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

    # 3. Sort durations numerically 
    def extract_numeric_time(val):
        digits = ''.join(c for c in str(val) if c.isdigit())
        return int(digits) if digits else 0

    durations = sorted(df['duration'].unique(), key=extract_numeric_time)
    search_methods = ['GridSearch', 'random', 'tpe', 'anneal', 'evolution']
    search_methods = [m for m in search_methods if m in df['method'].unique()]

    # 4. Grid Setup (1 row for Trials, columns = search methods)
    n_rows = 1
    n_cols = len(search_methods)
    dataset_labels = ['Local', 'TT', 'TT+Local', 'Mat', 'Ali', 'Mat+Local', 'Ali+Local']

    sns.set_style("white")
    plt.figure(figsize=(4.8 * n_cols, 4.2 * n_rows))

    for c, method in enumerate(search_methods):  
        ax = plt.subplot(n_rows, n_cols, c + 1)

        # Isolate rows for this search method
        method_df = df[df['method'] == method].copy()
        
        # Fill any missing values with 0 and convert trials to integer
        method_df['trials'] = method_df['trials'].fillna(0).astype(int)

        # Pivot values into 2D matrix structure
        matrix_df = method_df.pivot(index='dataset_display', columns='duration', values='trials')
        matrix_df = matrix_df.reindex(index=dataset_labels, columns=durations)

        # Draw the heatmap
        sns.heatmap(
            matrix_df,
            annot=True,
            fmt="d",  
            annot_kws={"size": 11, "weight": "bold"}, 
            xticklabels=durations,
            yticklabels=dataset_labels if c == 0 else [], 
            cmap=colormap,  # Dynamically switchable (Defaults to viridis)
            cbar=True
        )

        # Title for each search method column
        plt.title(method, fontsize=14, fontweight="bold", pad=12)

        # Highlight "TT+Local" in Bold and Red on the Y-Axis
        if c == 0:
            plt.ylabel("Dataset Configuration", fontsize=13, fontweight="bold")
            for label in ax.get_yticklabels():
                if label.get_text() == "TT+Local":
                    label.set_color("red")
                    label.set_fontweight("bold")
                    label.set_fontsize(12)
        else:
            plt.ylabel("")

        # X-Axis styling
        plt.xlabel("Experiment Duration", fontsize=12, fontweight="bold", labelpad=8)
        plt.xticks(rotation=45)

    plt.tight_layout()
    
    output_dir = "./plots/heatmaps/"
    os.makedirs(output_dir, exist_ok=True)
    
    out_pdf = os.path.join(output_dir, "trials_duration_heatmap.pdf")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSuccessfully generated trials heatmap using '{colormap}' at: {out_pdf}")

if __name__ == "__main__":
    # If you prefer the pink/purple theme instead, change "viridis" to "plasma" below
    plot_trials_heatmap(colormap="viridis")