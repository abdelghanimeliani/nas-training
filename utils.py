import csv
import json
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