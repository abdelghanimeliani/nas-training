#!/bin/bash

SEARCH_METHODS=("random" "GridSearch" "tpe" "anneal" "evolution")
TRIALS=(5 20 50 80 100)
DATASETS=("./dataset1_140.csv" "./dataset2_140.csv" "./dataset3_140.csv")

BASE_PORT=8100

for trials in "${TRIALS[@]}"; do
    for method in "${SEARCH_METHODS[@]}"; do

        echo "================================================"
        echo "Running optimizer: $method with $trials trials"
        echo "================================================"

        for dataset in "${DATASETS[@]}"; do
            PORT=$BASE_PORT

            EXP_NAME="exp_${method}_${trials}_${dataset}"

            echo "Running dataset: $dataset on port $PORT"

            PY_OUTPUT=$(python3 runner.py \
                --experiment-name "$EXP_NAME" \
                --port $PORT \
                --max-trials $trials \
                --optimizer $method \
                --max-duration 360000s \
                --dataset "$dataset"
            )

            echo "$PY_OUTPUT"

            EXP_ID=$(echo "$PY_OUTPUT" | sed -r 's/\x1B\[[0-9;]*[A-Za-z]//g' | grep -oP "(?<=Experiment ID: )\S+")

            echo "Waiting for $EXP_ID to finish..."

            while true; do
                STATUS=$(nnictl experiment status "$EXP_ID" 2>/dev/null | grep -oP '"status":"\K[^"]+')
                echo "Experiment $EXP_ID → $STATUS"

                if [[ "$STATUS" == "DONE" || "$STATUS" == "STOPPED" || "$STATUS" == "ERROR" || "$STATUS" == "NO_MORE_TRIAL" ]]; then
                    nnictl stop "$EXP_ID" 2>/dev/null
                    echo "Finished dataset $dataset"
                    break
                fi

                sleep 10
            done

            BASE_PORT=$((BASE_PORT+1))
            sleep 2

        done

    done
done
