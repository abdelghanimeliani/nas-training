import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_step_heatmaps_from_csv(csv_path="./values/metrics_steps.csv"):
    if not os.path.exists(csv_path):
        print(f"Error: The file '{csv_path}' was not found. Check your path setup.")
        return

    print(f"Loading experiment trial results from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. Normalize column headers and strip extra white spaces
    df.columns = [col.lower().strip() for col in df.columns]
    df['dataset'] = df['dataset'].astype(str).str.strip()
    df['method'] = df['method'].astype(str).str.strip()

    # 2. Structural Mapping: Align row IDs to clean display names
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

    # 3. Handle discrete numbers of trials cleanly (Column header is 'duration')
    df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
    steps = sorted(df['duration'].dropna().unique().astype(int))

    # Match exact method columns
    search_methods = ['GridSearch', 'random', 'tpe', 'anneal', 'evolution']
    search_methods = [m for m in search_methods if m in df['method'].unique()]

    # 4. Metric names and scaled layout labels
    metrics = ["min_mape", "min_mae", "min_mse"]
    metric_titles = {
        "min_mape": "MAPE (%)", 
        "min_mae": "MAE", 
        "min_mse": "MSE (×10⁻⁴)"  # Clarified scaling notice
    }

    n_rows = len(metrics)
    n_cols = len(search_methods)
    dataset_labels = ['Local', 'TT', 'TT+Local', 'Mat', 'Ali', 'Mat+Local', 'Ali+Local']

    plt.figure(figsize=(4.8 * n_cols, 3.8 * n_rows))

    for r, metric in enumerate(metrics):            
        for c, method in enumerate(search_methods):  

            ax = plt.subplot(n_rows, n_cols, r * n_cols + c + 1)

            # Isolate rows for this specific technique column
            method_df = df[df['method'] == method].copy()
            
            # Coerce empty cells to NaN numbers cleanly
            method_df[metric] = pd.to_numeric(method_df[metric], errors='coerce')
            
            # --- SCALE THE MSE DECIMALS ---
            if metric == "min_mse":
                method_df[metric] = method_df[metric] * 10000
                fmt_str = ".2f"      
                cmap_choice = "inferno"
            elif metric == "min_mae":
                fmt_str = ".4f"      
                cmap_choice = "plasma"
            else:
                fmt_str = ".2f"      
                cmap_choice = "viridis"
            # -------------------------------

            # Pivot values into 2D structural matrix
            matrix_df = method_df.pivot(index='dataset_display', columns='duration', values=metric)
            matrix_df = matrix_df.reindex(index=dataset_labels, columns=steps)

            # Safeguard against entirely unexecuted configurations (like step '100' in parts of the file)
            is_all_nan = matrix_df.isnull().all().all()

            # Draw Heatmap
            sns.heatmap(
                matrix_df,
                annot=not is_all_nan, # Only attempt drawing overlay text if data points exist
                fmt=fmt_str,
                annot_kws={"size": 10, "weight": "bold"}, 
                xticklabels=steps,
                yticklabels=dataset_labels if c == 0 else [], 
                cmap=cmap_choice,
                cbar=True,
                vmin=0 if is_all_nan else None, # Force mock limit bounds on empty frames
                vmax=1 if is_all_nan else None
            )

            # --- CRITICAL HIGHLIGHT LOGIC: TT+Local turned Red and Bold ---
            if c == 0:
                for label in ax.get_yticklabels():
                    if label.get_text() == "TT+Local":
                        label.set_color("red")
                        label.set_fontweight("bold")
                        label.set_fontsize(12) # Blown up slightly so it jumps out immediately

            if r == 0:
                plt.title(method, fontsize=14, fontweight="bold", pad=12)

            if c == 0:
                plt.ylabel(metric_titles[metric], fontsize=13, fontweight="bold")
            else:
                plt.ylabel("")

            # Set descriptive bottom row X-labels
            if r == n_rows - 1:
                plt.xlabel("Number of Trials (Steps)", fontsize=11, fontweight="bold", labelpad=8)
            else:
                plt.xlabel("")
                
            plt.xticks(rotation=0) # Steps numbers (5, 10, 20) stay horizontal for effortless reading

    plt.tight_layout()
    
    output_dir = "./plots/heatmaps/"
    os.makedirs(output_dir, exist_ok=True)
    
    out_pdf = os.path.join(output_dir, "all_metrics_steps_heatmaps.pdf")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSuccessfully generated step-based heatmap grid asset at: {out_pdf}")

if __name__ == "__main__":
    generate_step_heatmaps_from_csv()