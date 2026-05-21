SEARCH_METHODS=("random" "GridSearch" "tpe" "anneal" "evolution")
TRIALS=(5 10 20 50 80 100)

# Updated Dataset List based on the previous evaluation scenarios
DATASETS=(
    "./local_only.csv"
    "./tt_only.csv"
    "./local_plus_tt.csv"
    "./mat_only.csv"
    "./ali_only.csv"
    "./local_plus_mat.csv"
    "./local_plus_ali.csv"
)

BASE_PORT=8100

for trials in "${TRIALS[@]}"; do
    for method in "${SEARCH_METHODS[@]}"; do

        echo "================================================"
        echo "Running optimizer: $method with $trials trials"
        echo "================================================"

        for dataset in "${DATASETS[@]}"; do
            PORT=$BASE_PORT
            
            # Extract just the filename without extension for a cleaner Experiment Name
            DATASET_NAME=$(basename "$dataset" .csv)
            EXP_NAME="exp_${method}_${trials}_${DATASET_NAME}"

            echo "Running dataset: $dataset (Scenario: $DATASET_NAME) on port $PORT"

            # Launch the NNI Experiment via your runner.py
            PY_OUTPUT=$(python3 runner.py \
                --experiment-name "$EXP_NAME" \
                --port $PORT \
                --max-trials "$trials" \
                --optimizer "$method" \
                --max-duration 360000s \
                --dataset "$dataset"
            )

            echo "$PY_OUTPUT"

            # Extract Experiment ID, stripping ANSI colors (handling bold/colored output)
            EXP_ID=$(echo "$PY_OUTPUT" | sed -r 's/\x1B\[[0-9;]*[A-Za-z]//g' | grep -oP "(?<=Experiment ID: )\S+")

            # Ensure we actually got an ID before waiting
            if [ -z "$EXP_ID" ]; then
                echo "Error: Could not extract Experiment ID for $EXP_NAME"
                continue
            fi

            echo "Waiting for $EXP_ID to finish..."

            while true; do
                # Query NNI status
                STATUS=$(nnictl experiment status "$EXP_ID" 2>/dev/null | grep -oP '"status":"\K[^"]+')
                
                # Check for empty status to prevent loop crashing
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

            # Increment port and protect against high port numbers
            BASE_PORT=$((BASE_PORT+1))
            if [ "$BASE_PORT" -gt 9000 ]; then
                BASE_PORT=8100
            fi
            
            sleep 2

        done

    done
done