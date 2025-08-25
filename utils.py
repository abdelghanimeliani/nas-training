import csv
import json
from pathlib import Path
import os
import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def get_file_name_based_on_exp_duration(base_dir, dataset, duration, method):
    """
    Build the experiment file name string.

    Args:
        base_dir (str): Path to the directory containing the files.
        dataset (str): Dataset name, e.g., "dataset2".
        steps (int): Number of steps, e.g., 50.
        method (str): Search method, e.g., "evolution" or "gridsearch".

    Returns:
        str: Full path of the experiment file.
    """
    return f"{base_dir}/exp_{{{method}}}_optimizer_in_{{{duration}}}_duration_with__{dataset}_8000.csv"


def plot_metrics_based_on_exp_duration(search_methods,durations,datasets):
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for method in search_methods:
        for dataset in datasets:
            for method in search_methods:
                for duration in durations:
                    min_mape_value = float('inf')
                    min_mae_value= float('inf')
                    min_mse_value=float('inf')
                    file_name= get_file_name_based_on_exp_duration(base_dir="./csv/new_metric_data",dataset=dataset,duration=duration,method=method)
                    if os.path.exists(file_name):
                        print(f"Processing file: {file_name}")
                        with open(file_name, 'r') as f:
                            reader = csv.reader(f)
                            next(reader)
                            for row in reader:
                                print(row[7])
                                print(min_mape_value<float(row[7]))
                                if min_mape_value > float(row[7]):
                                    min_mape_value = float(row[7])
                                if min_mae_value > float(row[6]):
                                    min_mae_value = float(row[6])
                                if min_mse_value > float(row[5]):
                                    min_mse_value = float(row[5])
                    else:
                        print(f"File does not exist: {file_name}")
                    results[dataset][method][duration]['min_mape']= min_mape_value
                    results[dataset][method][duration]['min_mae']= min_mae_value
                    results[dataset][method][duration]['min_mse']= min_mse_value
    metrics = ['min_mape', 'min_mae', 'min_mse']
    metric_names = {'min_mape': 'MAPE (%)', 'min_mae': 'MAE', 'min_mse': 'MSE'}
    metric_titles = {'min_mape': 'Minimum MAPE', 'min_mae': 'Minimum MAE', 'min_mse': 'Minimum MSE'}

    colors = {'tpe': '#1f77b4', 'random': '#ff7f0e', 'GridSearch': '#2ca02c', 'evolution': '#d62728'}
    patterns = {'dataset1': '.', 'dataset2': ''}

    os.makedirs('./plots/time_based_plots/', exist_ok=True)

# Create a separate plot for each metric
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(14, 8))
        bar_width = 0.2
        group_width = bar_width * len(datasets)
        method_positions = np.arange(len(durations)) * (len(search_methods) * group_width + 0.5)
        
        # Plot bars for each method, trial, and dataset
        for i, method in enumerate(search_methods):
            for j, dataset in enumerate(datasets):
                values = []
                for duration in durations:
                    value = results[dataset][method][duration][metric]
                    values.append(value)
                
                offset = method_positions + i * group_width + j * bar_width
                bars = ax.bar(offset, values, bar_width, 
                            color=colors[method], hatch=patterns[dataset],
                            edgecolor='black', label=f'{method} ({dataset})' if i == 0 and j == 0 else "")
        
        # Customize the plot
        ax.set_xlabel('Number of Trials')
        ax.set_ylabel(metric_names[metric])
        ax.set_xticks(method_positions + (len(search_methods) * group_width - bar_width) / 2)
        ax.set_xticklabels(durations)
        ax.legend(handles=[
            plt.Rectangle((0,0),1,1, color=colors[m], label=m) for m in search_methods
        ] + [
            plt.Rectangle((0,0),1,1, hatch=patterns[d], fill=False, label=d, edgecolor='black') for d in datasets
        ], loc='upper right')
        
        plt.title(f'{metric_titles[metric]} vs. exp duration by Search Method and Dataset')
        
        # Save the plot
        plt.savefig(f'./plots/time_based_plots/{metric}_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'./plots/time_based_plots/{metric}_comparison.pdf', bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
        
        print(f"{metric} plot saved to ./plots/metrics_based_plots/")
     

def get_file_name_based_on_trials_number(base_dir, dataset, steps, method):
    """
    Build the experiment file name string.

    Args:
        base_dir (str): Path to the directory containing the files.
        dataset (str): Dataset name, e.g., "dataset2".
        steps (int): Number of steps, e.g., 50.
        method (str): Search method, e.g., "evolution" or "gridsearch".

    Returns:
        str: Full path of the experiment file.
    """
    return f"{base_dir}/exp_{{{method}}}_optimizer_with_{{{steps}}}_steps_with__{dataset}_8000.csv"

def plot_metrics_based_on_the_number_of_trials(search_methods,number_of_trials,datasets):
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for method in search_methods:
        for dataset in datasets:
            for method in search_methods:
                for trial in number_of_trials:
                    min_mape_value = float('inf')
                    min_mae_value= float('inf')
                    min_mse_value=float('inf')
                    file_name= get_file_name_based_on_trials_number(base_dir="./csv/new_metric_data",dataset=dataset,steps=trial,method=method)
                    if os.path.exists(file_name):
                        print(f"Processing file: {file_name}")
                        with open(file_name, 'r') as f:
                            reader = csv.reader(f)
                            next(reader)
                            for row in reader:
                                print(row[7])
                                print(min_mape_value<float(row[7]))
                                if min_mape_value > float(row[7]):
                                    min_mape_value = float(row[7])
                                if min_mae_value > float(row[6]):
                                    min_mae_value = float(row[6])
                                if min_mse_value > float(row[5]):
                                    min_mse_value = float(row[5])
                    else:
                        print(f"File does not exist: {file_name}")
                    results[dataset][method][trial]['min_mape']= min_mape_value
                    results[dataset][method][trial]['min_mae']= min_mae_value
                    results[dataset][method][trial]['min_mse']= min_mse_value
    metrics = ['min_mape', 'min_mae', 'min_mse']
    metric_names = {'min_mape': 'MAPE (%)', 'min_mae': 'MAE', 'min_mse': 'MSE'}
    metric_titles = {'min_mape': 'Minimum MAPE', 'min_mae': 'Minimum MAE', 'min_mse': 'Minimum MSE'}

# Define colors and patterns
    colors = {'tpe': '#1f77b4', 'random': '#ff7f0e', 'GridSearch': '#2ca02c', 'evolution': '#d62728'}
    patterns = {'dataset1': '.', 'dataset2': ''}

# Create directory if it doesn't exist
    os.makedirs('./plots/metrics_based_plots/', exist_ok=True)

# Create a separate plot for each metric
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(14, 8))
        bar_width = 0.2
        group_width = bar_width * len(datasets)
        method_positions = np.arange(len(number_of_trials)) * (len(search_methods) * group_width + 0.5)
        
        # Plot bars for each method, trial, and dataset
        for i, method in enumerate(search_methods):
            for j, dataset in enumerate(datasets):
                values = []
                for trial in number_of_trials:
                    value = results[dataset][method][trial][metric]
                    values.append(value)
                
                offset = method_positions + i * group_width + j * bar_width
                bars = ax.bar(offset, values, bar_width, 
                            color=colors[method], hatch=patterns[dataset],
                            edgecolor='black', label=f'{method} ({dataset})' if i == 0 and j == 0 else "")
        
        # Customize the plot
        ax.set_xlabel('Number of Trials')
        ax.set_ylabel(metric_names[metric])
        ax.set_xticks(method_positions + (len(search_methods) * group_width - bar_width) / 2)
        ax.set_xticklabels(number_of_trials)
        ax.legend(handles=[
            plt.Rectangle((0,0),1,1, color=colors[m], label=m) for m in search_methods
        ] + [
            plt.Rectangle((0,0),1,1, hatch=patterns[d], fill=False, label=d, edgecolor='black') for d in datasets
        ], loc='upper right')
        
        plt.title(f'{metric_titles[metric]} vs. Number of Trials by Search Method and Dataset')
        
        # Save the plot
        plt.savefig(f'./plots/metrics_based_plots/{metric}_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'./plots/metrics_based_plots/{metric}_comparison.pdf', bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
        
        print(f"{metric} plot saved to ./plots/metrics_based_plots/")

def plot_time_based_on_the_number_of_trials(search_methods,number_of_trials,datasets):
    results = defaultdict(lambda: defaultdict(dict)) 
    for method in search_methods:
        for dataset in datasets:
            for trial in number_of_trials:
                file_name= get_file_name_based_on_trials_number(base_dir="./csv/new_exp_profiles",dataset=dataset,steps=trial,method=method)
                if os.path.exists(file_name):
                    print(f"Processing file: {file_name}")
                    with open(file_name, 'r') as f:
                        
                        reader = csv.reader(f)
                        for row in reader:
                            last_row = row
                        try:
                            start = datetime.datetime.fromtimestamp(int(last_row[3]) / 1000)
                            end = datetime.datetime.fromtimestamp(int(last_row[4]) / 1000)
                            exp_time= end - start
                            print(print(f"Method: {method}, Dataset: {dataset}, Trials: {trial}, Time: {exp_time} seconds"))
                        except ValueError as e:
                            exp_time=0
                        results[dataset][method][trial]= exp_time
                else:
                    print(f"File does not exist: {file_name}")
    # Define colors and patterns
    colors = {'tpe': '#1f77b4', 'random': '#ff7f0e', 'GridSearch': '#2ca02c', 'evolution': '#d62728'}
    patterns = {'dataset1': '.', 'dataset2': ''}  # No pattern for dataset1, 'x' for dataset2# Plot configuration
    fig, ax = plt.subplots(figsize=(8, 8))
    bar_width = 0.2
    group_width = bar_width * len(datasets)
    method_positions = np.arange(len(number_of_trials)) * (len(search_methods) * group_width + 0.2 )

# Plot bars for each method, trial, and dataset
    for i, method in enumerate(search_methods):
        for j, dataset in enumerate(datasets):
            times = []
            for trial in number_of_trials:
                td = results[dataset][method].get(trial, datetime.timedelta(0))
                times.append(td.total_seconds() if isinstance(td, datetime.timedelta) else 0)
            
            offset = method_positions + i * group_width + j * bar_width
            bars = ax.bar(offset, times, bar_width, 
                        color=colors[method], hatch=patterns[dataset],
                        edgecolor='black', label=f'{method} ({dataset})' if i == 0 and j == 0 else "")

    # Customize the plot
    ax.set_xlabel('Number of Trials')
    ax.set_ylabel('Time (seconds)')
    ax.set_yscale('log')  # Logarithmic scale due to large value range
    ax.set_xticks(method_positions + (len(search_methods) * group_width - bar_width) / 2)
    ax.set_xticklabels(number_of_trials)
    ax.legend(handles=[
        plt.Rectangle((0,0),1,1, color=colors[m], label=m) for m in search_methods
    ] + [
        plt.Rectangle((0,0),1,1, hatch=patterns[d], fill=False, label=d, edgecolor='black') for d in datasets
    ], loc='upper left')

    plt.title('Time vs. Number of Trials by Search Method and Dataset')
    plt.tight_layout()
    # Create directory if it doesn't exist
    os.makedirs('./plots/trial_based_plots/', exist_ok=True)

    # Save the plot
    plt.savefig('./plots/trial_based_plots/trial_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('./plots/trial_based_plots/trial_time_comparison.pdf', bbox_inches='tight')
    plt.show()               
         
def plot_trials_based_on_exp_duration(search_methods,duratrions,datasets):
    results=defaultdict(lambda: defaultdict(dict))
    for dataset in datasets:
        for method in search_methods:
            for duration in duratrions:
                max_trials=0
                file_name= get_file_name_based_on_exp_duration(base_dir="./csv/new_trial_job_event",dataset=dataset,duration=duration,method=method)
                if os.path.exists(file_name):
                    print(f"Processing file: {file_name}")
                    with open(file_name, 'r') as f:
                        reader = csv.reader(f)
                        next(reader)
                        for row in reader:
                            if row[2] == 'SUCCEEDED':
                                max_trials += 1
                else:
                    print(f"File does not exist: {file_name}")
                results[dataset][method][duration]= max_trials

    # Define colors and patterns
    colors = {'tpe': '#1f77b4', 'random': '#ff7f0e', 'GridSearch': '#2ca02c', 'evolution': '#d62728'}
    patterns = {'dataset1': '.', 'dataset2': ''}  # No pattern for dataset1, 'x' for dataset2# Plot configuration
    fig, ax = plt.subplots(figsize=(8, 8))
    bar_width = 0.2
    group_width = bar_width * len(datasets)
    method_positions = np.arange(len(duratrions)) * (len(search_methods) * group_width + 0.2 )

# Plot bars for each method, trial, and dataset
    for i, method in enumerate(search_methods):
        for j, dataset in enumerate(datasets):
            times = []
            for duration in duratrions:
                td = results[dataset][method].get(duration)
                times.append(td)
            
            offset = method_positions + i * group_width + j * bar_width
            bars = ax.bar(offset, times, bar_width, 
                        color=colors[method], hatch=patterns[dataset],
                        edgecolor='black', label=f'{method} ({dataset})' if i == 0 and j == 0 else "")

    # Customize the plot
    ax.set_xlabel('Exmeriment Duration')
    ax.set_ylabel('Number of Trials')
    # ax.set_yscale('log')  # Logarithmic scale due to large value range
    ax.set_xticks(method_positions + (len(search_methods) * group_width - bar_width) / 2)
    ax.set_xticklabels(duratrions)
    ax.legend(handles=[
        plt.Rectangle((0,0),1,1, color=colors[m], label=m) for m in search_methods
    ] + [
        plt.Rectangle((0,0),1,1, hatch=patterns[d], fill=False, label=d, edgecolor='black') for d in datasets
    ], loc='upper left')

    plt.title('Number of trials vs. Expiriment duration by Search Method and Dataset')
    plt.tight_layout()
    # Create directory if it doesn't exist
    os.makedirs('./plots/trial_based_plots/', exist_ok=True)

    # Save the plot
    plt.savefig('./plots/trial_based_plots/trial_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('./plots/time_based_plots/time_trial_comparison.pdf', bbox_inches='tight')
    plt.show()               
# =============== call the ploting functions ===============
def plot_trials_based_exp_resutls():
    '''
    this methods is used to plot the resutlts of the trial-based expiriments
    on the 4 search methods:
    TPE, Random, gridSearch, Evolution
    the plots are,  the time token for each exp for each database based algorithm, and the defined number of trials
    
    '''
    search_methods = ['tpe', 'random', 'GridSearch', 'evolution']
    number_of_trials=  [5,20,50,80,100]
    datasets= ['dataset2', 'dataset1']
    plot_metrics_based_on_the_number_of_trials(search_methods,number_of_trials,datasets)
    plot_time_based_on_the_number_of_trials(search_methods,number_of_trials,datasets)

def plot_time_based_exp_resutls():
    search_methods = ['tpe', 'random', 'GridSearch', 'evolution']
    duratrions=  ["300s","600s","1200s","2400s","3600s"]
    datasets= ['dataset2', 'dataset1']
    # plot_metrics_based_on_exp_duration(search_methods,duratrions,datasets)
    plot_trials_based_on_exp_duration(search_methods,duratrions,datasets)

plot_time_based_exp_resutls()

def get_expiriments_ids_list(base_path):
    ids= [p.name for p in Path(base_path).iterdir() if p.is_dir() ]
    try:
        ids.remove('_latest')
    except ValueError:
        pass
    return ids  
def change_files_names(base_path,ids):
    new_names = []   
    for exp_id in ids:
        print("Changing files names for experiment ID: " + exp_id)
        f=open(base_path + '/' + exp_id + '.csv', 'r') 
        reader = csv.reader(f)
        reader. __next__()  # Skip the header row
        row1=next(reader)  # Skip the header row

        new_name={
                "id": exp_id,
                "experiment_name": row1[7],
            }
        new_names.append(new_name)
        f.close()

    print('=============================================================')
    for name in new_names:
        f=open("./csv/exp_profiles/" + name['id'] + '.csv', 'r')
        content= f.read()
        f2= open( "./csv/new_exp_profiles/" + name['experiment_name'] + '.csv', 'w')
        f2.write(content)
        f.close()
        f2.close()
        f=open("./csv/metric_data/" + name['id'] + '.csv', 'r')
        content= f.read()
        f2= open( "./csv/new_metric_data/" + name['experiment_name'] + '.csv', 'w')
        f2.write(content)
        f.close()
        f2.close()
        f=open("./csv/trial_job_event/" + name['id'] + '.csv', 'r')
        content= f.read()
        f2= open( "./csv/new_trial_job_event/" + name['experiment_name'] + '.csv', 'w')
        f2.write(content)
        f.close()
        f2.close()
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
                        print(f"Invalid JSON for trial_job_id {trial_job_id} at timestamp {timestamp}")
                        metrics_json = json.loads(metrics_json)
                    print(metrics_json)
                    
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
                    print(csv_row)
                    
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