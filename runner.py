import argparse
import subprocess
from pathlib import Path
import re
import sys

def run_single_experiment(optimizer, port, duration, max_trials, experiment_name, dataset):
    template_path = Path("config_template.yml")
    with open(template_path) as f:
        config = f.read()

    dataset_name = Path(dataset).stem

    # Fill template
    new_config = (
        config.replace("{OPTIMIZER}", optimizer)
              .replace("{DATA_PATH}", dataset)
              .replace("{MAX_TRIALS}", str(max_trials))
              .replace("{MAX_DURATION}", duration)
              .replace("{EXPERIMENT_NAME}", experiment_name + f"_{dataset_name}")
    )

    temp_config_path = Path(f"temp_{dataset_name}.yml")
    with open(temp_config_path, "w") as f:
        f.write(new_config)

    # Print info for the bash script
    print(f"DATASET_USED: {dataset}")
    print(f"CONFIG_FILE: {temp_config_path}")
    print(f"PORT_USED: {port}")
    sys.stdout.flush()

    # Run nni
    proc = subprocess.run(
            ["nnictl", "create", "--config", str(temp_config_path), "--port", str(port)],
            capture_output=True, text=True
        )

    match = re.search(r'Experiment ID:\s*(\S+)', proc.stdout)
    if match:
        print("Experiment ID:", match.group(1))
    else:
        print("ERROR: Experiment ID not found!")
        print("===== NNI ERROR LOGS =====")
        print("STDOUT:", proc.stdout)
        print("STDERR:", proc.stderr)
        print("==========================")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--max-trials", type=int, required=True)
    parser.add_argument("--max-duration", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    run_single_experiment(
        args.optimizer,
        args.port,
        args.max_duration,
        args.max_trials,
        args.experiment_name,
        args.dataset
    )
