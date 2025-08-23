#!/bin/bash
SEARCH_METHODS=("random" "grid" "tpe" "anneal" "evolution")
DURATIONS=("300s" "600s" "1200s" "2400s" "3600s")

BASE_PORT=8300
PORT=$BASE_PORT

for duration in "${DURATIONS[@]}"; do
    for method in "${SEARCH_METHODS[@]}"; do
        EXP_NAME="exp_{$method}_optimizer_in_{$duration}_duration_with_"
        echo "Running: $EXP_NAME on port $PORT with $trials trials and method $method"
        # Run the Python runner and get experiment IDs
        EXP_IDS=$(python3 runner.py  --experiment-name $EXP_NAME  --max-trials 10000000000000  --port $PORT --optimizer $method --max-duration $duration | sed -r 's/\x1B\[[0-9;]*[A-Za-z]//g' | grep -oP "(?<=Experiment ID: )\S+") 
        mapfile -t ids_array <<< "$EXP_IDS"

        for EXP_ID in "${ids_array[@]}"; do
           echo "Waiting for experiment $EXP_ID to finish..."
           while true; do
             STATUS=$(nnictl experiment status "$EXP_ID" 2>/dev/null | grep -oP '"status":"\K[^"]+')
             echo exp with id $EXP_ID is now: $STATUS
             if [[ "$STATUS" == "DONE" || "$STATUS" == "STOPPED" || "$STATUS" == "ERROR" || "$STATUS" == "NO_MORE_TRIAL" ]]; then
               echo "Experiment $EXP_ID finished with status: $STATUS"
               nnictl  stop "$EXP_ID"
               break
             fi
             sleep 30
           done
        done

        PORT=$((PORT+1))
        sleep 2
    done
done

echo "All trial-based experiments finished."