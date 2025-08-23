import csv
import json

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
                    print("======================================================")
                    print(default_metric)
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