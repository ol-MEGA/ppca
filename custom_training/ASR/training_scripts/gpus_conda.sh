#!/bin/bash

# Needed variables to be input from the user
GPUS=$1 # Number of gpus to use for the training. Look into the hyperparameter file for optimal setting
VENV=$2 # virtual environement with SpeechBrain and all other libraries installed 
YAML=hparams/param_reduction/transformer_multi_gpu.yaml # Hyperparameter file specifying all needed hyperparams

# Activates your virtual environment
eval "$(conda shell.bash hook)"
conda activate $VENV

#param configs
sample_rate=16000
n_mels=10 #80, 40, 20, 10
time_factor=1 #1, 4, 8, 12
win_length=$((25*time_factor))
hop_length=$((10*time_factor))
n_fft=$((sample_rate*win_length/1000))

echo Training ASR with $n_mels mels, win_length $win_length ms, hop_length $hop_length ms, nfft $n_fft ...

# Correct Speechbrain Launch
python -m torch.distributed.launch --nproc_per_node=$GPUS train.py $YAML --batch_size=16 \
    --n_mels=$n_mels --win_length=$win_length --hop_length=$hop_length --n_fft=$n_fft \
    --distributed_launch --distributed_backend='nccl'