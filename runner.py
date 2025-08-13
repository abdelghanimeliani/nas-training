'''
this script runs experiments with different optimizers on specified datasets.
it takes the port number and optimizer type as command line arguments.
the plan is to use this sctipt from inside another bash script that will run the experiments with the specified parameters.
'''

import argparse
import subprocess
from pathlib import Path


def run_experiment(optimizer, port):
    template_path = Path("config_template.yml")
    with open(template_path) as f:
        config = f.read()

    for dataset in ["./dataset1_8000.csv", "./dataset2_8000.csv"]:
        dataset_name = Path(dataset).stem
        # Replace placeholders
        new_config = config.replace("{OPTIMIZER}", optimizer).replace("{DATA_PATH}", dataset)
        temp_config_path = Path(f"temp_{dataset_name}.yml")
        with open(temp_config_path, "w") as f:
            f.write(new_config)

        print(f"Running {optimizer} on {dataset} at port {port}...")
        subprocess.run(["nnictl", "create", "--config", str(temp_config_path), "--port", str(port)])
        port += 1  
        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, required=True)
    parser.add_argument("--port", type=int, default=8080, help="Starting port number")
    args = parser.parse_args()

    run_experiment(args.optimizer, args.port)