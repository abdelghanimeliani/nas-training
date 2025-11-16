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

        PORT=$BASE_PORT

        for dataset in "${DATASETS[@]}"; do

            EXP_NAME="exp_${method}_${duration}_${dataset}"

            echo ""
            echo "-----------------------------------------------"
            echo "Experiment Name: $EXP_NAME"
            echo "DATASET_USED: $dataset"
            echo "PORT_USED: $PORT"
            echo "OPTIMIZER: $method"
            echo "DURATION: $duration"
            echo "-----------------------------------------------"

            # Run Python runner
            PY_OUTPUT=$(python3 runner.py \
                --experiment-name "$EXP_NAME" \
                --port $PORT \
                --max-trials 9999999999 \
                --optimizer $method \
                --max-duration $duration \
                --dataset "$dataset"
            )

            echo "$PY_OUTPUT"

            echo "Waiting for experiment on port $PORT ..."

            while true; do
                # Read status using PORT (much more reliable)
                STATUS_RAW=$(nnictl experiment status $PORT 2>/dev/null)
                STATUS_CLEAN=$(echo "$STATUS_RAW" | tr -cd '\11\12\15\40-\176')
                STATUS=$(echo "$STATUS_CLEAN" | grep -oP '"status":"\K[^"]+')

                echo "Experiment on port $PORT → '$STATUS'"

                # STOP CONDITIONS
                if [[ "$STATUS" == "DONE" || "$STATUS" == "STOPPED" || "$STATUS" == "ERROR" || "$STATUS" == "NO_MORE_TRIAL" ]]; then
                    echo "Experiment on port $PORT finished → $STATUS"
                    nnictl stop $PORT 2>/dev/null
                    break
                fi

                sleep 10
            done

            PORT=$((PORT+1))
            sleep 2

        done

        BASE_PORT=$((BASE_PORT+3))
    done
done

echo "All duration-based experiments finished."
