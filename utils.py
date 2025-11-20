import csv
import json
from pathlib import Path
import os
import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import re
import numpy as np
import seaborn as sns

def get_file_name_based_on_exp_duration(base_dir, dataset, duration, method):
    """
    Build the experiment file name string matching the pattern:
    exp_{method}_{duration}s_._{dataset}_140.csv_{dataset}_140.csv
    """

    import re

    # Ensure dataset ends with _140
    ds = dataset if str(dataset).endswith("_140") else f"{dataset}_140"

    # Sanitize method and duration
    method_safe = re.sub(r'[^A-Za-z0-9_.-]', '_', str(method))
    duration_safe = re.sub(r'[^A-Za-z0-9_.-]', '_', str(duration))

    return f"{base_dir}/exp_{method_safe}_{duration_safe}_._{ds}.csv_{ds}.csv"

def plot_metrics_based_on_exp_duration(search_methods,durations,datasets):
    
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for method in search_methods:
        for dataset in datasets:
            for duration in durations:

                min_mape = float('inf')
                min_mae  = float('inf')
                min_mse  = float('inf')

                file_name = get_file_name_based_on_exp_duration(
                    base_dir="./csv/new_metric_data",
                    dataset=dataset,
                    duration=duration,
                    method=method
                )

                if os.path.exists(file_name):
                    with open(file_name, "r") as f:
                        reader = csv.reader(f)
                        next(reader)
                        for row in reader:
                            min_mape = min(min_mape, float(row[7]))
                            min_mae  = min(min_mae,  float(row[6]))
                            min_mse  = min(min_mse,  float(row[5]))
                else:
                    print(f"Missing: {file_name}")

                results[dataset][method][duration]["min_mape"] = min_mape
                results[dataset][method][duration]["min_mae"]  = min_mae
                results[dataset][method][duration]["min_mse"]  = min_mse

    # --------------------
    # Build 3 × 5 grid
    # --------------------
    metrics = ["min_mape", "min_mae", "min_mse"]
    metric_titles = {"min_mape": "MAPE (%)", "min_mae": "MAE", "min_mse": "MSE"}

    n_rows = len(metrics)
    n_cols = len(search_methods)

    plt.figure(figsize=(4*n_cols, 3.5*n_rows))
    print("Loading experiment results...")
    print(str(results))
    for r, metric in enumerate(metrics):            # rows
        for c, method in enumerate(search_methods):  # columns

            ax = plt.subplot(n_rows, n_cols, r*n_cols + c + 1)

            # Build heatmap data matrix
            matrix = []
            for dataset in datasets:
                row_vals = []
                for duration in durations:
                    row_vals.append(results[dataset][method][duration][metric])
                matrix.append(row_vals)

            matrix = np.array(matrix)

            sns.heatmap(
                matrix,
                annot=True,
                fmt=".3f",
                xticklabels=durations,
                yticklabels=datasets if c == 0 else [],  # only left-most column shows dataset labels
                cmap="viridis",
                cbar=(c == n_cols - 1)  # only last column shows colorbar
            )

            # Titles
            if r == 0:
                plt.title(method, fontsize=12, fontweight="bold")

            if c == 0:
                plt.ylabel(metric_titles[metric], fontsize=12, fontweight="bold")

            plt.xlabel("")

    plt.tight_layout()
    os.makedirs("./plots/heatmaps/", exist_ok=True)
    out_png = "./plots/heatmaps/all_metrics_heatmaps.png"
    out_pdf = "./plots/heatmaps/all_metrics_heatmaps.pdf"

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    print(f"\nSaved combined heatmap grid to:\n{out_png}\n{out_pdf}")


def get_file_name_based_on_trials_number(base_dir, dataset, steps, method):
    """
    Build the experiment file name string matching the pattern:
    exp_{method}_{steps}_._{dataset}_140.csv_{dataset}_140.csv

    Args:
        base_dir (str): Path to the directory containing the files.
        dataset (str): Dataset name, e.g., "dataset1" or "dataset1_140".
        steps (int|str): Number of steps, e.g., 5 or "5".
        method (str): Search method, e.g., "anneal" or "gridsearch".

    Returns:
        str: Full path of the experiment file following the exact messy format.
    """
    import re

    # Ensure dataset ends with _140
    ds = dataset if str(dataset).endswith("_140") else f"{dataset}_140"
    # Sanitize method and steps for safe filenames
    method_safe = re.sub(r'[^A-Za-z0-9_.-]', '_', str(method))
    steps_safe = re.sub(r'[^A-Za-z0-9_.-]', '_', str(steps))

    return f"{base_dir}/exp_{method_safe}_{steps_safe}_._{ds}.csv_{ds}.csv"

def plot_metrics_based_on_the_number_of_trials(search_methods,number_of_trials,datasets):
    
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for method in search_methods:
        for dataset in datasets:
            for trial in number_of_trials:

                min_mape = float('inf')
                min_mae  = float('inf')
                min_mse  = float('inf')

                file_name = get_file_name_based_on_trials_number(
                    base_dir="./csv/new_metric_data",
                    dataset=dataset,
                    steps=trial,
                    method=method
                )

                if os.path.exists(file_name):
                    with open(file_name, "r") as f:
                        reader = csv.reader(f)
                        next(reader)
                        for row in reader:
                            min_mape = min(min_mape, float(row[7]))
                            min_mae  = min(min_mae,  float(row[6]))
                            min_mse  = min(min_mse,  float(row[5]))
                else:
                    print(f"File missing: {file_name}")

                results[dataset][method][trial]["min_mape"] = min_mape
                results[dataset][method][trial]["min_mae"]  = min_mae
                results[dataset][method][trial]["min_mse"]  = min_mse

    # --------------------
    # Heatmap grid layout
    # --------------------
    metrics = ["min_mape", "min_mae", "min_mse"]
    metric_titles = {"min_mape": "MAPE (%)", "min_mae": "MAE", "min_mse": "MSE"}

    n_rows = len(metrics)     # 3 metrics
    n_cols = len(search_methods)  # e.g., 5 search methods


    plt.figure(figsize=(4*n_cols, 3.5*n_rows))

    # --------------------
    # Build the 3×5 grid
    # --------------------
    for r, metric in enumerate(metrics):
        for c, method in enumerate(search_methods):

            ax = plt.subplot(n_rows, n_cols, r*n_cols + c + 1)

            # Build matrix (datasets × trials)
            matrix = []
            for dataset in datasets:
                row_vals = []
                for trial in number_of_trials:
                    row_vals.append(results[dataset][method][trial][metric])
                matrix.append(row_vals)

            matrix = np.array(matrix)

            sns.heatmap(
                matrix,
                annot=True,
                fmt=".3f",
                xticklabels=number_of_trials,
                yticklabels=datasets if c == 0 else [],
                cmap="viridis",
                cbar=(c == n_cols - 1)  # only last column has colorbar
            )

            # Titles
            if r == 0:
                plt.title(method, fontsize=12, fontweight="bold")

            if c == 0:
                plt.ylabel(metric_titles[metric], fontsize=12, fontweight="bold")

            plt.xlabel("")

    plt.tight_layout()

    # Save the combined figure
    os.makedirs("./plots/heatmaps/", exist_ok=True)
    png_path = "./plots/heatmaps/all_metrics_trials_heatmap.png"
    pdf_path = "./plots/heatmaps/all_metrics_trials_heatmap.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"\nSaved trials heatmap grid to:\n{png_path}\n{pdf_path}")

def plot_time_based_on_the_number_of_trials(search_methods,number_of_trials,datasets):

    # results[dataset][method][trial] = execution_time_in_seconds
    results = defaultdict(lambda: defaultdict(dict))
    print("Results dictionary:")
    print(results)
    print("========================================")


    for method in search_methods:
        for dataset in datasets:
            for trial in number_of_trials:

                file_name = get_file_name_based_on_trials_number(
                    base_dir="./csv/new_exp_profiles",
                    dataset=dataset,
                    steps=trial,
                    method=method
                )

                if not os.path.exists(file_name):
                    print(f"File does not exist: {file_name}")
                    results[dataset][method][trial] = 0
                    continue

                with open(file_name, "r") as f:
                    reader = csv.reader(f)
                    last_row = None
                    for row in reader:
                        last_row = row

                if last_row is None:
                    results[dataset][method][trial] = 0
                    continue

                try:
                    start_ts = int(last_row[3]) / 1000
                    end_ts   = int(last_row[4]) / 1000
                    time_delta = end_ts - start_ts
                    exec_time = max(time_delta, 0)
                except:
                    exec_time = 0

                results[dataset][method][trial] = exec_time
    n_rows = 1
    n_cols = len(search_methods)


    plt.figure(figsize=(4*n_cols, 4))

    for c, method in enumerate(search_methods):

        ax = plt.subplot(n_rows, n_cols, c + 1)

        # Build matrix = datasets × trials
        matrix = []
        for dataset in datasets:
            row_vals = []
            for trial in number_of_trials:
                val = results[dataset][method].get(trial, 0)
                row_vals.append(val)
            matrix.append(row_vals)

        matrix = np.array(matrix)

        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            xticklabels=number_of_trials,
            yticklabels=datasets if c == 0 else [],
            cmap="viridis",
            cbar=(c == n_cols - 1)
        )

        plt.title(f"{method}", fontsize=12, fontweight="bold")
        if c == 0:
            plt.ylabel("Dataset", fontsize=12, fontweight="bold")
        plt.xlabel("Number of Trials")

    plt.tight_layout()

    # -------------------------------------------------------------
    # Save output
    # -------------------------------------------------------------
    out_dir = "./plots/trial_based_plots/"
    os.makedirs(out_dir, exist_ok=True)

    png_path = out_dir + "trial_time_heatmaps.png"
    pdf_path = out_dir + "trial_time_heatmaps.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"\nSaved trial time heatmap grid to:\n{png_path}\n{pdf_path}")

def plot_trials_based_on_exp_duration(search_methods,durations,datasets):
    
    # results[dataset][method][duration] = number_of_trials
    results = defaultdict(lambda: defaultdict(dict))
    for dataset in datasets:
        for method in search_methods:
            for duration in durations:

                file_name = get_file_name_based_on_exp_duration(
                    base_dir="./csv/new_trial_job_event",
                    dataset=dataset,
                    duration=duration,
                    method=method
                )

                max_trials = 0

                if os.path.exists(file_name):
                    with open(file_name, "r") as f:
                        reader = csv.reader(f)
                        next(reader)  # skip header
                        for row in reader:
                            if row[2] == "SUCCEEDED":
                                max_trials += 1
                else:
                    print(f"File does not exist: {file_name}")

                results[dataset][method][duration] = max_trials


    n_rows = 1
    n_cols = len(search_methods)
    print(results)
    plt.figure(figsize=(4 * n_cols, 4))

    for c, method in enumerate(search_methods):

        ax = plt.subplot(n_rows, n_cols, c + 1)

        # Build matrix: rows = datasets, columns = durations
        matrix = []
        for dataset in datasets:
            row_vals = []
            for duration in durations:
                val = results[dataset][method].get(duration, 0)
                row_vals.append(val)
            matrix.append(row_vals)

        matrix = np.array(matrix)

        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            xticklabels=durations,
            yticklabels=datasets if c == 0 else [],
            cmap="viridis",
            cbar=(c == n_cols - 1)
        )

        plt.title(f"{method}", fontsize=12, fontweight="bold")
        if c == 0:
            plt.ylabel("Dataset", fontsize=12, fontweight="bold")
        plt.xlabel("Experiment Duration")

    plt.tight_layout()

    # -------------------------------------------------------------
    # Save output
    # -------------------------------------------------------------
    out_dir = "./plots/duration_based_plots/"
    os.makedirs(out_dir, exist_ok=True)

    png_path = out_dir + "duration_trials_heatmaps.png"
    pdf_path = out_dir + "duration_trials_heatmaps.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"\nSaved duration heatmap grid to:\n{png_path}\n{pdf_path}")
# =============== call the ploting functions ===============
def plot_trials_based_exp_resutls():
    '''
    this methods is used to plot the resutlts of the trial-based expiriments
    on the 4 search methods:
    TPE, Random, gridSearch, Evolution
    the plots are,  the time token for each exp for each database based algorithm, and the defined number of trials
    
    '''
    search_methods = ['tpe', 'random', 'GridSearch', 'evolution', 'anneal']
    number_of_trials=  [5,20,50,80,100]
    datasets= ['dataset1', 'dataset2', 'dataset3']
    plot_metrics_based_on_the_number_of_trials(search_methods,number_of_trials,datasets)
    plot_time_based_on_the_number_of_trials(search_methods,number_of_trials,datasets)

def plot_time_based_exp_resutls():
    search_methods = ['tpe', 'random', 'GridSearch', 'evolution', 'anneal']
    duratrions=  ["300s","600s","1200s","2400s","3600s"]
    datasets= ['dataset1', 'dataset2', 'dataset3']
    plot_metrics_based_on_exp_duration(search_methods,duratrions,datasets)
    plot_trials_based_on_exp_duration(search_methods,duratrions,datasets)


def get_expiriments_ids_list(base_path):
    ids= [p.name for p in Path(base_path).iterdir() if p.is_dir() ]
    try:
        ids.remove('_latest')
    except ValueError:
        pass
    return ids  
def change_files_names(base_path, ids):
    new_names = []

    # Read experiment names
    for exp_id in ids:
        with open(f"{base_path}/{exp_id}.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)
            row1 = next(reader)

            exp_name = row1[7]
            # Sanitize experiment name for filesystem
            safe_name = re.sub(r'[^A-Za-z0-9_.-]', '_', exp_name)

            new_names.append({
                "id": exp_id,
                "experiment_name": safe_name,
            })

    # Create directories if missing
    os.makedirs("./csv/new_exp_profiles", exist_ok=True)
    os.makedirs("./csv/new_metric_data", exist_ok=True)
    os.makedirs("./csv/new_trial_job_event", exist_ok=True)

    # Copy content into new files
    for name in new_names:
        safe_name = name["experiment_name"]

        with open(f"./csv/exp_profiles/{name['id']}.csv", "r") as f:
            with open(f"./csv/new_exp_profiles/{safe_name}.csv", "w") as f2:
                f2.write(f.read())

        with open(f"./csv/metric_data/{name['id']}.csv", "r") as f:
            with open(f"./csv/new_metric_data/{safe_name}.csv", "w") as f2:
                f2.write(f.read())

        with open(f"./csv/trial_job_event/{name['id']}.csv", "r") as f:
            with open(f"./csv/new_trial_job_event/{safe_name}.csv", "w") as f2:
                f2.write(f.read())

    print("Files names changed successfully.")
def convert_MetricData_table_to_csv(data, output_file):
    """
    Convert only FINAL MetricData values to CSV with JSON values in separate columns
    
    Args:
        data: List of tuples from the MetricData table
        output_file: Path for the output CSV file
    """
    
    headers = [
        'timestamp', 'trial_job_id', 'parameter_id', 'sequence',
        'default_metric', 'mse', 'mae', 'mape'
    ]

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for row in data:
            # Parse tuple elements
            timestamp = row[0]
            trial_job_id = row[1]
            parameter_id = row[2]
            metric_type = row[3]
            sequence = row[4]
            value = row[5]
            
            # Only process FINAL metrics
            if metric_type == 'FINAL':
                # Parse the JSON value
                    # Parse JSON for final metrics
                    metrics_json = json.loads(value)
                    # Check if the JSON is valid
                    if not isinstance(metrics_json, dict):
                        # print(f"Invalid JSON for trial_job_id {trial_job_id} at timestamp {timestamp}")
                        metrics_json = json.loads(metrics_json)
                    
                    # Extract all metrics from JSON
                    default_metric = metrics_json["default"]
                    mse = metrics_json["mse"] if 'mse' in metrics_json else ''
                    mae = metrics_json["mae"] if 'mae' in metrics_json else ''
                    mape = metrics_json["mape"] if 'mape' in metrics_json else ''
                    
                    # Create CSV row
                    csv_row = [
                        timestamp,
                        trial_job_id,
                        parameter_id,
                        sequence,
                        default_metric,
                        mse,
                        mae,
                        mape
                    ]
                    
                    writer.writerow(csv_row)
                    
    
    print(f"CSV file created at: {output_file}")
    print(f"Only FINAL metrics have been exported")
def convert_ExpirimentProfile_tables_to_csv(data,output_file):
    
    headers = [
    'experiment_id', 'trial_id', 'status', 'start_time', 'end_time',
    'working_directory', 'exit_code', 'experiment_name', 'experiment_type',
    'search_space_file', 'trial_command', 'trial_code_directory',
    'trial_concurrency', 'max_duration', 'max_trials', 'use_annotation',
    'debug_mode', 'log_level', 'exp_working_dir', 'model_types',
    'unit_values', 'layer_counts', 'dropout_rates', 'activations',
    'learning_rate_range', 'window_sizes', 'batch_sizes', 'epoch_options',
    'kernel_sizes', 'attention_heads', 'tuner_name', 'optimize_mode',
    'training_platform'
    ]

    with open(output_file, 'w', newline='') as f:
         writer = csv.writer(f)
         writer.writerow(headers)
         for row in data:
        # Parse tuple elements
             config_json = row[0]
             experiment_id = row[1]
             status = row[2]
             start_time = row[3]
             end_time = row[4] if row[4] is not None else ''
             working_dir = row[5]
             exit_code = row[6]
             trial_id = row[7]
             config = json.loads(config_json)
             search_space = config['searchSpace']
             main_config = [
            config['experimentName'],
            config['experimentType'],
            config['searchSpaceFile'],
            config['trialCommand'],
            config['trialCodeDirectory'],
            config['trialConcurrency'],
            config['maxExperimentDuration'],
            config['maxTrialNumber'],
            config['useAnnotation'],
            config['debug'],
            config['logLevel'],
            config['experimentWorkingDirectory']
        ]
        
        # Extract search space parameters
             search_space_data = [
            str(search_space['model_type']['_value']),
            str(search_space['units']['_value']),
            str(search_space['num_layers']['_value']),
            str(search_space['dropout']['_value']),
            str(search_space['activation']['_value']),
            str(search_space['lr']['_value']),
            str(search_space['window_size']['_value']),
            str(search_space['batch_size']['_value']),
            str(search_space['epochs']['_value']),
            str(search_space['kernel_size']['_value']),
            str(search_space['attention_heads']['_value'])
        ]
        
        # Extract tuner and training service info
             tuner_data = [
            config['tuner']['name'],
            config['tuner']['classArgs']['optimize_mode']
             ]
        
             training_service = [
            config['trainingService']['platform']
             ]
        
        # Combine all data for CSV row
             csv_row = (
            [experiment_id, trial_id, status, start_time, end_time, working_dir, exit_code] +
            main_config +
            search_space_data +
            tuner_data +
            training_service
        )
             writer.writerow(csv_row)
         print(f"CSV file created at: {output_file}")     
def convert_TrialJobEvent_to_csv(data, output_file):
    """
    Convert TrialJobEvent table to CSV format with extracted hyperparameters
    
    Args:
        data: List of tuples from the TrialJobEvent table
        output_file: Path for the output CSV file
    """
    
    headers = [
        'timestamp', 'trial_job_id', 'event_type', 'parameter_id', 
        'parameter_source', 'parameter_index', 'log_dir', 'job_id', 
        'message', 'environment', 'model_type', 'units', 'num_layers', 
        'dropout', 'activation', 'lr', 'window_size', 'batch_size', 
        'epochs', 'kernel_size', 'attention_heads'
    ]

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for row in data:
            # Parse tuple elements
            timestamp = row[0]
            trial_job_id = row[1]
            event_type = row[2]
            event_data = row[3] if row[3] else '{}'  # Handle empty event_data
            log_dir = row[4]
            job_id = row[5]
            message = row[6] if len(row) > 6 else ''
            environment = row[7] if len(row) > 7 else ''
            
            # Initialize parameter fields
            parameter_id = ''
            parameter_source = ''
            parameter_index = ''
            model_type = ''
            units = ''
            num_layers = ''
            dropout = ''
            activation = ''
            lr = ''
            window_size = ''
            batch_size = ''
            epochs = ''
            kernel_size = ''
            attention_heads = ''
            
            # Parse event_data if it exists and is not empty
            if event_data and event_data != '{}':
                try:
                    event_json = json.loads(event_data)
                    
                    # Extract parameter metadata
                    parameter_id = event_json.get('parameter_id', '')
                    parameter_source = event_json.get('parameter_source', '')
                    parameter_index = event_json.get('parameter_index', '')
                    
                    # Extract parameters if they exist
                    parameters = event_json.get('parameters', {})
                    if parameters:
                        model_type = parameters.get('model_type', '')
                        units = parameters.get('units', '')
                        num_layers = parameters.get('num_layers', '')
                        dropout = parameters.get('dropout', '')
                        activation = parameters.get('activation', '')
                        lr = parameters.get('lr', '')
                        window_size = parameters.get('window_size', '')
                        batch_size = parameters.get('batch_size', '')
                        epochs = parameters.get('epochs', '')
                        kernel_size = parameters.get('kernel_size', '')
                        attention_heads = parameters.get('attention_heads', '')
                        
                except json.JSONDecodeError:
                    print(f"Warning: Failed to parse JSON for trial {trial_job_id}: {event_data}")
            
            # Create CSV row
            csv_row = [
                timestamp,
                trial_job_id,
                event_type,
                parameter_id,
                parameter_source,
                parameter_index,
                log_dir,
                job_id,
                message,
                environment,
                model_type,
                units,
                num_layers,
                dropout,
                activation,
                lr,
                window_size,
                batch_size,
                epochs,
                kernel_size,
                attention_heads
            ]
            
            writer.writerow(csv_row)
    
    print(f"CSV file created at: {output_file}")