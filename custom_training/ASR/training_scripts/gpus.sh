#!/bin/bash

# Needed variables to be input from the user
GPUS=$1 # Number of gpus to use for the training. Look into the hyperparameter file for optimal setting
YAML=$2 # Hyperparameter file specifying all needed hyperparams (you can find it in hparams/transformer_multi_gpu.yaml)
VENV=$3 # Path to your virtual environement with SpeechBrain and all other libraries installed (not sure if conda environments work better to stick with "python -m venv")
        # this variable should have the form of "path/to/venv/bin/activate"

# Activates your virtual environment
source $VENV

# Run 26th January 2023
# torchrun --nproc_per_node=$GPUS train.py $YAML

# Correct Speechbrain Launch
python -m torch.distributed.launch --nproc_per_node=$GPUS train.py $YAML --distributed_launch --distributed_backend='nccl'
