#!/bin/bash

SEARCH_METHODS=("random" "GridSearch" "tpe" "anneal" "evolution")
DURATIONS=("300s" "600s" "1200s" "2400s" "3600s")
DATASETS=("./dataset1_140.csv" "./dataset2_140.csv" "./dataset3_140.csv")

BASE_PORT=8300

for duration in "${DURATIONS[@]}"; do
    for method in "${SEARCH_METHODS[@]}"; do

        echo "================================================"
        echo "Running optimizer: $method with duration: $duration"
        echo "================================================"

        for dataset in "${DATASETS[@]}"; do
            PORT=$BASE_PORT

            EXP_NAME="exp_${method}_${duration}_${dataset}"

            echo "Running dataset: $dataset on port $PORT"

            PY_OUTPUT=$(python3 runner.py \
                --experiment-name "$EXP_NAME" \
                --port $PORT \
                --max-trials 10000000000 \
                --optimizer $method \
                --max-duration $duration \
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
