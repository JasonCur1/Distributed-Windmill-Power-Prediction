#!/bin/bash

# This script multiplexes your terminal for easy navigation across multiple computers.

# Useful keyboard shortcuts within tmux:
   # `Ctrl+b, d` -- detaches from tmux
   # `Ctrl+b, [arrow_key]` -- navigates between panes (up arrow to go up, left to go left, etc.)
   # `Ctrl+b, :` -- opens the command line; the `setw synchronize-panes` command synchs and desynchs the panes

# Useful shell commands
   # `tmux attach -t [session_name]` -- attaches to the specified session
   # `tmux ls` -- lists active tmux sessions
   # `man tmux` -- opens the tmux manual

# list your hosts here, e.g., HOSTS=("carrot" "broccoli")
# visit https://www.cs.colostate.edu/machinestats/ for a list of machines you can use
# if you see the same hostname in multiple tmux windows, it is likely that one your hosts is down
HOSTS=("venus" "mackerel" "marlin" "perch" "pollock" "sardine" "shark" "sole" "swordfish" "tarpon") # "turbot" "tuna" "herring" "wahoo" "grouper" "barracuda" "blowfish" "bonito" "brill" "bullhead"

# path to your project directory
DIR="~/cs555/term-project"

# session name
SESSION="csx55-term-project"

# Master address (first host - coordinator)
MASTER_ADDR="venus"  # Change this to the actual IP if needed
MASTER_PORT="29500"
WORLD_SIZE="${#HOSTS[@]}"

# IT IS RECCOMMENDED THAT YOU SET UP PASSWORDLESS ssh: https://sna.cs.colostate.edu/remote-connection/ssh/keybased/

tmux kill-session -t $SESSION 2> /dev/null
tmux new-session -d -s $SESSION

FIRST_HOST=true
RANK=0
for HOST in "${HOSTS[@]}"
do
    if $FIRST_HOST; then
        FIRST_HOST=false
    else
        tmux split-window -t $SESSION
    fi

    # SSH into host, activate venv, set environment variables, run training
    tmux send-keys -t $SESSION "ssh $HOST" C-m
    tmux send-keys -t $SESSION "cd $DIR" C-m
    tmux send-keys -t $SESSION "source venv/bin/activate" C-m
    tmux send-keys -t $SESSION "export MASTER_ADDR=$MASTER_ADDR" C-m
    tmux send-keys -t $SESSION "export MASTER_PORT=$MASTER_PORT" C-m
    tmux send-keys -t $SESSION "export WORLD_SIZE=$WORLD_SIZE" C-m
    tmux send-keys -t $SESSION "export RANK=$RANK" C-m
    tmux send-keys -t $SESSION "clear" C-m
    tmux send-keys -t $SESSION "echo 'Worker Rank $RANK on $HOST - Ready to start'" C-m

    # Don't auto-start yet - let user verify all are ready
    # User can manually run: python src/train.py
    # Or uncomment the line below to auto-start:
    tmux send-keys -t $SESSION "python src/train.py" C-m

    tmux select-layout -t $SESSION tiled

    RANK=$((RANK + 1))
done

tmux attach -t $SESSION