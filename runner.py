'''
this script runs experiments with different optimizers on specified datasets.
it takes the port number and optimizer type as command line arguments.
the plan is to use this sctipt from inside another bash script that will run the experiments with the specified parameters.
'''

import argparse
import subprocess
from pathlib import Path


def run_experiment(optimizer, port,duration, max_trials, experiment_name):
    template_path = Path("config_template.yml")
    with open(template_path) as f:
        config = f.read()

    for dataset in ["./dataset1_8000.csv", "./dataset2_8000.csv"]:
        dataset_name = Path(dataset).stem
        # Replace placeholders
        new_config = config.replace("{OPTIMIZER}", optimizer)
        new_config = new_config.replace("{DATA_PATH}", dataset)
        new_config = new_config.replace("{MAX_TRIALS}", str(max_trials))
        new_config = new_config.replace("{MAX_DURATION}", duration)
        new_config = new_config.replace("{EXPERIMENT_NAME}", experiment_name+f"_{dataset_name}")
        temp_config_path = Path(f"temp_{dataset_name}.yml")
        with open(temp_config_path, "w") as f:
            f.write(new_config)

        print(f"Running {optimizer} on {dataset} at port {port}...")
        print(f"Using config: {temp_config_path}")
        print("Experiment name:", experiment_name)
        print("Optimizer:", optimizer)
        print("this experiment will run for", duration, "seconds")
        print("Maximum number of trials:", max_trials)
        subprocess.run(["nnictl", "create", "--config", str(temp_config_path), "--port", str(port)])
        port += 1  
        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, required=True)
    parser.add_argument("--port", type=int, default=8080, help="Starting port number")
    parser.add_argument("--max-trials", type=int, default=100, help="Maximum number of trials")
    parser.add_argument("--max-duration", type=str, default="3600s", help="Maximum experiment duration in seconds")
    parser.add_argument("--experiment-name", type=str, default="dataset-test", help="Name of the experiment")
    args = parser.parse_args()

    run_experiment(args.optimizer, args.port, args.max_duration, args.max_trials, args.experiment_name)