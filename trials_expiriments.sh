#!/bin/bash
SEARCH_METHODS=("random" "GridSearch" "tpe" "anneal" "evolution")
TRIALS=(5 20 50 80 100)

BASE_PORT=8080
PORT=$BASE_PORT

for trials in "${TRIALS[@]}"; do
    for method in "${SEARCH_METHODS[@]}"; do
        EXP_NAME="exp_{$method}_optimizer_with_{$trials}_steps_with_"
        echo running trials experiments with methods: "${SEARCH_METHODS[@]}" and trials: "${TRIALS[@]}"
        echo "Running: $EXP_NAME on port $PORT with $trials trials and method $method"
        # Run the Python runner and get experiment IDs
        EXP_IDS=$(python3 runner.py  --experiment-name $EXP_NAME  --port $PORT    --max-trials $trials  --optimizer $method --max-duration 360000s | sed -r 's/\x1B\[[0-9;]*[A-Za-z]//g' | grep -oP "(?<=Experiment ID: )\S+") 
        mapfile -t ids_array <<< "$EXP_IDS"

        for EXP_ID in "${ids_array[@]}"; do
           echo "Waiting for experiment $EXP_ID to finish..."
           while true; do
             STATUS=$(nnictl experiment status "$EXP_ID" 2>/dev/null | grep -oP '"status":"\K[^"]+')
             echo exp with id $EXP_ID is now: $STATUS
             if [[ "$STATUS" == "DONE" || "$STATUS" == "STOPPED" || "$STATUS" == "ERROR" || "$STATUS" == "NO_MORE_TRIAL" ]]; then
               echo "Experiment $EXP_ID finished with status: $STATUS"
               nnictl stop "$EXP_ID" 
               break
             fi
             sleep 10
           done
        done

        PORT=$((PORT+1))
        sleep 2
    done
done

echo "All trial-based experiments finished."