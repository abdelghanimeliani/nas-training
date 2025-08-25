import csv
import json
from pathlib import Path
import os

def get_file_name(base_dir, dataset, steps, method):
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




 

def plot_time_based_on_the_number_of_trials(search_methods,number_of_trials):
    
    
    return
def plot_trials_based_exp_resutls():
    '''
    this methods is used to plot the resutlts of the trial-based expiriments
    on the 4 search methods:
    TPE, Random, gridSearch, Evolution
    the plots are,  the time token for each exp for each database based algorithm, and the defined number of trials
    
    '''
    search_methods = ['TPE', 'Random', 'gridSearch', 'Evolution']
    return
    
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