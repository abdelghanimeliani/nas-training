#!/bin/bash

# Updated Search Methods and Durations
SEARCH_METHODS=("random" "GridSearch" "tpe" "anneal" "evolution")
DURATIONS=( "300s" "600s" "900s" "1200s" "1500s" "1800s" )

# New Dataset List based on the 7 evaluation scenarios
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

# ==============================================================================
# RESUME CONFIGURATION
# Set this to 146 to safely rerun the experiment that was interrupted when the laptop died.
# If you are 100% sure the 147th folder fully finished, change this to 147.
# ==============================================================================
SKIP_UNTIL=147 

for duration in "${DURATIONS[@]}"; do
    for method in "${SEARCH_METHODS[@]}"; do

        # Used to prevent printing headers for skipped optimizers
        FIRST_PRINT=true

        for dataset in "${DATASETS[@]}"; do
            CURRENT_EXP=$((CURRENT_EXP + 1))
            
            # --- RESUME SKIP LOGIC ---
            if [ "$CURRENT_EXP" -le "$SKIP_UNTIL" ]; then
                # We must simulate the port increments so the port remains synced
                BASE_PORT=$((BASE_PORT + 1))
                if [ "$BASE_PORT" -gt 9000 ]; then
                    BASE_PORT=8700
                fi
                continue
            fi
            # -------------------------

            # Print the optimizer header only when an experiment actually runs
            if [ "$FIRST_PRINT" = true ]; then
                echo "================================================"
                echo "Running optimizer: $method with duration: $duration"
                echo "================================================"
                FIRST_PRINT=false
            fi

            PORT=$BASE_PORT
            
            # Extract just the filename without extension for a cleaner Experiment Name
            DATASET_NAME=$(basename "$dataset" .csv)
            EXP_NAME="exp_${method}_${duration}_${DATASET_NAME}"

            echo "Running dataset: $dataset (Scenario: $DATASET_NAME) on port $PORT [Exp #$CURRENT_EXP]"

            # Launch the NNI Experiment via your runner.py
            PY_OUTPUT=$(python3 runner.py \
                --experiment-name "$EXP_NAME" \
                --port $PORT \
                --max-trials 100000000 \
                --optimizer "$method" \
                --max-duration "$duration" \
                --dataset "$dataset"
            )

            echo "$PY_OUTPUT"

            # Extract Experiment ID, stripping ANSI colors (handling bold/colored output)
            EXP_ID=$(echo "$PY_OUTPUT" | sed -r 's/\x1B\[[0-9;]*[A-Za-z]//g' | grep -oP "(?<=Experiment ID: )\S+")

            if [ -z "$EXP_ID" ]; then
                echo "Error: Could not extract Experiment ID for $EXP_NAME"
                continue
            fi

            echo "Waiting for $EXP_ID to finish..."

            while true; do
                # Query NNI status
                STATUS=$(nnictl experiment status "$EXP_ID" 2>/dev/null | grep -oP '"status":"\K[^"]+')
                
                # If STATUS is empty, the command might have failed or experiment closed prematurely
                if [ -z "$STATUS" ]; then
                    echo "Warning: Status empty for $EXP_ID. Checking again..."
                    sleep 5
                    continue
                fi

                echo "Experiment $EXP_ID ($DATASET_NAME) → $STATUS"

                # Check for completion states
                if [[ "$STATUS" == "DONE" || "$STATUS" == "STOPPED" || "$STATUS" == "ERROR" || "$STATUS" == "NO_MORE_TRIAL" ]]; then
                    nnictl stop "$EXP_ID" 2>/dev/null
                    echo "Finished dataset $dataset"
                    break
                fi

                sleep 10
            done

            # Increment port for the next dataset to avoid potential socket conflicts
            BASE_PORT=$((BASE_PORT+1))
            
            # Reset port if it gets too high
            if [ "$BASE_PORT" -gt 9000 ]; then
                BASE_PORT=8700
            fi
            
            sleep 2
        done
    done
done