#!/bin/bash

# Configuration settings
SEARCH_METHODS=("random" "GridSearch" "tpe" "anneal" "evolution")
HORIZONS=(1 2 3 4 5)
FIXED_DURATION="600s"  # Fixed to 10 minutes for NNI configuration
FIXED_MAX_TRIALS=25   # Fixed to 25 maximum steps/trials

DATASETS=(
    "./local_only.csv"
    "./tt_only.csv"
    "./local_plus_tt.csv"
    "./mat_only.csv"
    "./ali_only.csv"
    "./local_plus_mat.csv"
    "./local_plus_ali.csv"
)

BASE_PORT=8700
CURRENT_EXP=0
SKIP_UNTIL=0 # Adjust this if you ever need to resume an interrupted run

for horizon in "${HORIZONS[@]}"; do
    for method in "${SEARCH_METHODS[@]}"; do
        FIRST_PRINT=true

        for dataset in "${DATASETS[@]}"; do
            CURRENT_EXP=$((CURRENT_EXP + 1))
            
            # --- Resume Skip Logic ---
            if [ "$CURRENT_EXP" -le "$SKIP_UNTIL" ]; then
                BASE_PORT=$((BASE_PORT + 1))
                if [ "$BASE_PORT" -gt 9000 ]; then BASE_PORT=8700; fi
                continue
            fi

            if [ "$FIRST_PRINT" = true ]; then
                echo "================================================================="
                echo "Horizon: $horizon | Optimizer: $method | Max Duration: $FIXED_DURATION | Max Trials: $FIXED_MAX_TRIALS"
                echo "================================================================="
                FIRST_PRINT=false
            fi

            PORT=$BASE_PORT
            DATASET_NAME=$(basename "$dataset" .csv)
            EXP_NAME="exp_${method}_h${horizon}_${DATASET_NAME}"

            echo "Running dataset: $dataset (Scenario: $DATASET_NAME) Horizon: $horizon on port $PORT [Exp #$CURRENT_EXP]"

            # Launch NNI experiment setup via runner.py
            PY_OUTPUT=$(python3 runner.py \
                --experiment-name "$EXP_NAME" \
                --port $PORT \
                --max-trials $FIXED_MAX_TRIALS \
                --optimizer "$method" \
                --max-duration "$FIXED_DURATION" \
                --dataset "$dataset" \
                --horizon $horizon
            )

            echo "$PY_OUTPUT"

            # Extract Experiment ID while cleaning ANSI logs
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
                    echo "Finished dataset $dataset for Horizon $horizon"
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