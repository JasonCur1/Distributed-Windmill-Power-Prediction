# !/bin/bash

MACHINES=("anchovy" "mackerel" "marlin" "perch" "pollock" "sardine" "shark" "sole" "swordfish" "tarpon")

echo "Launching distributed training on ${#MACHINES[@]} machbines..."

echo "Starting coordinator on ${MACHINES[0]}"
ssh ${MACHINES[0]} "bash ~/cs555/term-project/scripts/start_coordinator.sh" &

sleep 5

for i in $(seq 1 $((${#MACHINES[@]} - 1))); do
    echo "Starting worker $i on ${MACHINES[$i]}"
    ssh ${MACHINES[$i]} "bash ~/cs555/term-project/scripts/start_worker.sh $i" &
done

echo "All machines launched. Training in progress..."
wait