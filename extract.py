import os
import csv

def extract_existing_results():
    base_dir = "./horison_results/new_metric_data"
    output_csv = "metrics_horizon.csv"
    
    if not os.path.exists(base_dir):
        print(f"ERROR: The directory '{base_dir}' does not exist. Check your path spelling!")
        return

    # 1. Grab every file actually sitting in the folder
    all_files = os.listdir(base_dir)
    print(f"Found {len(all_files)} total files inside {base_dir}. Starting extraction...\n")

    # Define our target structural keys
    SEARCH_METHODS = ["random", "GridSearch", "tpe", "anneal", "evolution"]
    HORIZONS = [1, 2, 3, 4, 5]
    
    # Sort datasets by length descending so "local_plus_ali" matches before "ali_only"
    DATASETS_ORDERED = [
        "local_plus_mat", "local_plus_ali", "local_plus_tt", 
        "local_only", "mat_only", "ali_only", "tt_only"
    ]

    # Initialize container for matching metrics
    # Key: (dataset_name, method, horizon) -> [min_mape, min_mae, min_mse]
    database = {}

    processed_files_count = 0

    # 2. Loop through every file and dynamically figure out what experiment it belongs to
    for f_name in all_files:
        if not f_name.endswith(".csv"):
            continue

        # Detect the optimization method
        matched_method = None
        for m in SEARCH_METHODS:
            if f"exp_{m}_" in f_name:
                matched_method = m
                break
        if not matched_method:
            continue

        # Detect the horizon step number (extracting the number following the prefix)
        prefix = f"exp_{matched_method}_h"
        remaining_str = f_name[len(prefix):] # e.g., "1_local_only..."
        horizon_str = remaining_str.split("_")[0]
        try:
            matched_horizon = int(horizon_str)
        except ValueError:
            continue

        # Detect the dataset scenario
        matched_dataset = None
        for d in DATASETS_ORDERED:
            if d in f_name:
                matched_dataset = d
                break
        if not matched_dataset:
            continue

        # 3. Read the actual file data directly by index
        file_path = os.path.join(base_dir, f_name)
        
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                
            if len(lines) < 2:
                continue # Skip empty or header-only runs
                
            # Parse header columns cleanly to establish direct mappings
            header = [col.strip().lower() for col in lines[0].split(",")]
            if "mape" not in header or "mae" not in header or "mse" not in header:
                continue

            mape_idx = header.index("mape")
            mae_idx  = header.index("mae")
            mse_idx  = header.index("mse")

            file_min_mape = float('inf')
            file_min_mae  = float('inf')
            file_min_mse  = float('inf')

            # Parse data rows
            for line in lines[1:]:
                parts = line.split(",")
                if len(parts) > max(mape_idx, mae_idx, mse_idx):
                    try:
                        val_mape = float(parts[mape_idx].strip())
                        val_mae  = float(parts[mae_idx].strip())
                        val_mse  = float(parts[mse_idx].strip())

                        # Ignore error catch boundaries
                        if val_mape < 9999.0: file_min_mape = min(file_min_mape, val_mape)
                        if val_mae < 9999.0:  file_min_mae  = min(file_min_mae, val_mae)
                        if val_mse < 9999.0:  file_min_mse  = min(file_min_mse, val_mse)
                    except ValueError:
                        continue

            # If valid tracking data was extracted, save it
            if file_min_mape != float('inf'):
                combo_key = (matched_dataset, matched_method, matched_horizon)
                database[combo_key] = [file_min_mape, file_min_mae, file_min_mse]
                processed_files_count += 1
                print(f"Processed: {f_name} -> Min MAPE: {file_min_mape:.4f}")

        except Exception as e:
            print(f"Error reading file {f_name}: {e}")

    print(f"\nExtraction phase complete. Extracted data from {processed_files_count} files.")
    print(f"Writing structured grid out to {output_csv}...")

    # 4. Generate the structured matrix output matching your exact CSV template format
    # Restoring original dataset names array for consistent row structure mapping
    DATASETS_ORIGINAL_FORMAT = [
        "local_only", "tt_only", "local_plus_tt", "mat_only", "ali_only", "local_plus_mat", "local_plus_ali"
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["Dataset", "Method", "Horizon", "Min_MAPE", "Min_MAE", "Min_MSE"])

        for dataset in DATASETS_ORIGINAL_FORMAT:
            for method in SEARCH_METHODS:
                for horizon in HORIZONS:
                    combo_key = (dataset, method, horizon)
                    
                    if combo_key in database:
                        metrics = database[combo_key]
                        writer.writerow([
                            f"./{dataset}.csv",
                            method,
                            horizon,
                            metrics[0],
                            metrics[1],
                            metrics[2]
                        ])
                    else:
                        # Log empty rows if an experiment combination file hasn't run or completed yet
                        writer.writerow([f"./{dataset}.csv", method, horizon, "", "", ""])

    print(f"File successfully created! Run 'cat {output_csv}' to confirm your values are there.")

if __name__ == "__main__":
    extract_existing_results()