#!/bin/bash

# Configuration settings
SEARCH_METHODS=("random" "GridSearch" "tpe" "anneal" "evolution")
DATA_SIZES=(10 20 50 100 200 300 400 500) # Replaced horizons loop array with size tracking arrays
FIXED_HORIZON=1                            # Locking horizon to 1 step for this specific test
FIXED_DURATION="10m"                       # Explicitly telling NNI to run for 10 minutes max
FIXED_MAX_TRIALS=25                        

DATASETS=(
    "./mat_only.csv"
)
BASE_PORT=8400
CURRENT_EXP=0
SKIP_UNTIL=0 

for size in "${DATA_SIZES[@]}"; do
    for method in "${SEARCH_METHODS[@]}"; do
        FIRST_PRINT=true

        for dataset in "${DATASETS[@]}"; do
            CURRENT_EXP=$((CURRENT_EXP + 1))
            
            if [ "$CURRENT_EXP" -le "$SKIP_UNTIL" ]; then
                BASE_PORT=$((BASE_PORT + 1))
                if [ "$BASE_PORT" -gt 9000 ]; then BASE_PORT=8700; fi
                continue
            fi

            if [ "$FIRST_PRINT" = true ]; then
                echo "================================================================="
                echo "Data Size: $size | Optimizer: $method | Max Duration: $FIXED_DURATION | Max Trials: $FIXED_MAX_TRIALS"
                echo "================================================================="
                FIRST_PRINT=false
            fi

            PORT=$BASE_PORT
            DATASET_NAME=$(basename "$dataset" .csv)
            EXP_NAME="exp_${method}_sz${size}_${DATASET_NAME}"

            echo "Running data size: $size (Scenario: $DATASET_NAME) Method: $method on port $PORT [Exp #$CURRENT_EXP]"

            # Launch NNI experiment setup via runner.py with the new --data-size flag
            PY_OUTPUT=$(python3 runner.py \
                --experiment-name "$EXP_NAME" \
                --port $PORT \
                --max-trials $FIXED_MAX_TRIALS \
                --optimizer "$method" \
                --max-duration "$FIXED_DURATION" \
                --dataset "$dataset" \
                --horizon $FIXED_HORIZON \
                --data-size "$size"
            )

            echo "$PY_OUTPUT"

            EXP_ID=$(echo "$PY_OUTPUT" | sed -r 's/\x1B\[[0-9;]*[A-Za-z]//g' | grep -oP "(?<=Experiment ID: )\S+")

            if [ -z "$EXP_ID" ]; then
                echo "Error: Could not extract Experiment ID for $EXP_NAME"
                continue
            fi

            echo "Waiting for $EXP_ID to finish..."

            while true; do
                STATUS=$(nnictl experiment status "$EXP_ID" 2>/dev/null | grep -oP '"status":"\K[^"]+')
                
                if [ -z "$STATUS" ]; then
                    echo "Warning: Status empty for $EXP_ID. Checking again..."
                    sleep 5
                    continue
                fi

                echo "Experiment $EXP_ID ($DATASET_NAME) → $STATUS"

                if [[ "$STATUS" == "DONE" || "$STATUS" == "STOPPED" || "$STATUS" == "ERROR" || "$STATUS" == "NO_MORE_TRIAL" ]]; then
                    nnictl stop "$EXP_ID" 2>/dev/null
                    echo "Finished data size $size for Optimizer $method"
                    break
                fi

                sleep 10
            done

            BASE_PORT=$((BASE_PORT + 1))
            if [ "$BASE_PORT" -gt 9000 ]; then BASE_PORT=8700; fi
            sleep 2
        done
    done
done