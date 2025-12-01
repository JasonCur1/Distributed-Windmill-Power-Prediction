#!/bin/bash

# Coord is always rank 0
export MASTER_ADDR="129.82.44.125" # venus' address. Use 'hostname -I'
export MASTER_PORT="29500" # random port num
export WORLD_SIZE="10" # total machines for program
export RANK="0"

cd ~/cs555/term-project # repo directory
python src/train.py