#!/bin/bash
set -euo pipefail  # safer bash: exit on error, unset vars, pipe errors


# Run the experiment creator and extract clean IDs
EXP_IDS=$(python3 runner.py  --experiment-name test  --max-trials 2  --optimizer random --max-duration 360000s | sed -r 's/\x1B\[[0-9;]*[A-Za-z]//g' | grep -oP "(?<=Experiment ID: )\S+")           # extract only IDs )

# Put IDs into an array (one per line)
mapfile -t ids_array <<< "$EXP_IDS"

# Trim whitespace from each ID
id1="${ids_array[0]//[[:space:]]/}"
id2="${ids_array[1]//[[:space:]]/}"

echo "=========================="
echo "===== Running Tests ====="
echo "=========================="
echo "Experiment ID 1: $id1"
echo "Experiment ID 2: $id2"

status1=$(nnictl experiment status "$id1" 2>/dev/null | grep -oP '"status":"\K[^"]+')
status2=$(nnictl experiment status "$id2" 2>/dev/null | grep -oP '"status":"\K[^"]+')

echo "Status for $id1: $status1"
echo "Status for $id2: $status2"
nnictl experiment list
nnictl stop --all