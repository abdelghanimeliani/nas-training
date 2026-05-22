import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_heatmaps_from_csv(csv_path="./values/metrics_duration.csv"):
    if not os.path.exists(csv_path):
        print(f"Error: The file '{csv_path}' was not found.")
        return

    print(f"Loading experiment results from {csv_path}...")
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

    # 3. Sort durations numerically (300s to 1800s)
    def extract_numeric_time(val):
        digits = ''.join(c for c in str(val) if c.isdigit())
        return int(digits) if digits else 0

    durations = sorted(df['duration'].unique(), key=extract_numeric_time)
    search_methods = ['GridSearch', 'random', 'tpe', 'anneal', 'evolution']
    search_methods = [m for m in search_methods if m in df['method'].unique()]

    # 4. Define metrics and scaled titles for the axis
    metrics = ["min_mape", "min_mae", "min_mse"]
    metric_titles = {
        "min_mape": "MAPE (%)", 
        "min_mae": "MAE", 
        "min_mse": "MSE (×10⁻⁴)"  
    }

    n_rows = len(metrics)
    n_cols = len(search_methods)
    dataset_labels = ['Local', 'TT', 'TT+Local', 'Mat', 'Ali', 'Mat+Local', 'Ali+Local']

    plt.figure(figsize=(4.8 * n_cols, 3.8 * n_rows))

    for r, metric in enumerate(metrics):            
        for c, method in enumerate(search_methods):  

            ax = plt.subplot(n_rows, n_cols, r * n_cols + c + 1)

            # Isolate rows for this method
            method_df = df[df['method'] == method].copy()
            
            # --- VISUAL FIX FOR TINY NUMBERS ---
            if metric == "min_mse":
                method_df[metric] = method_df[metric] * 10000
                fmt_str = ".2f"      
                cmap_choice = "rocket_r"
            elif metric == "min_mae":
                fmt_str = ".4f"      
                cmap_choice = "plasma"
            else:
                fmt_str = ".2f"      
                cmap_choice = "viridis"
            # ------------------------------------

            # Pivot values into 2D matrix structure
            matrix_df = method_df.pivot(index='dataset_display', columns='duration', values=metric)
            matrix_df = matrix_df.reindex(index=dataset_labels, columns=durations)

            # Draw the heatmap
            sns.heatmap(
                matrix_df,
                annot=True,
                fmt=fmt_str,
                annot_kws={"size": 10, "weight": "bold"}, 
                xticklabels=durations,
                yticklabels=dataset_labels if c == 0 else [], 
                cmap=cmap_choice,
                cbar=True 
            )

            # --- HIGHLIGHT "TT+Local" IN BOLD RED ---
            if c == 0:
                for label in ax.get_yticklabels():
                    if label.get_text() == "TT+Local":
                        label.set_color("red")
                        label.set_fontweight("bold")
                        # Optional: Slightly increase size to make it pop even more
                        label.set_fontsize(12) 

            if r == 0:
                plt.title(method, fontsize=14, fontweight="bold", pad=12)

            if c == 0:
                plt.ylabel(metric_titles[metric], fontsize=13, fontweight="bold")
            else:
                plt.ylabel("")

            plt.xlabel("")
            plt.xticks(rotation=45)

    plt.tight_layout()
    
    output_dir = "./plots/heatmaps/"
    os.makedirs(output_dir, exist_ok=True)
    
    out_png = os.path.join(output_dir, "all_metrics_clean_heatmaps.pdf")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSuccessfully generated clean plots at: {out_png}")

if __name__ == "__main__":
    generate_heatmaps_from_csv()