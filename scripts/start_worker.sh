#!/bin/bash

#RANK=$1 # rank passed as arg

export MASTER_ADDR="129.82.44.125" # coordinator's ip
export MASTER_PORT="29500" # coordinator's port
export WORLD_SIZE="2"
export RANK="1"

cd ~/cs555/term-project
python src/train.py