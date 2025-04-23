#!/bin/bash

SESSION_NAME="gz_session"

# Start a new tmux session in detached mode
tmux new-session -d -s "$SESSION_NAME"

# Top pane: Launch Gazebo
tmux send-keys -t "$SESSION_NAME" 'gz sim -v4 -r delivery_sim.sdf' C-m

# Bottom pane: Launch ArduPilot
tmux send-keys -t "$SESSION_NAME" 'sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --map --console' C-m

# Attach to session
tmux attach -t "$SESSION_NAME"