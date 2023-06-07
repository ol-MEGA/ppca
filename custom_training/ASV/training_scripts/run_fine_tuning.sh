#!/bin/bash

VENV="/fs/scratch/users/francesco_nespoli/SpeechBrain/venv/bin/python"
CONFIG=$1
SCRIPT="/fs/scratch/users/francesco_nespoli/SpeechBrain/train_on_voxceleb/training/train_speaker_embeddings.py"
CONTAINER=$2

export PYTHONPATH="${PYTHONPATH}:/fs/scratch/users/francesco_nespoli/SpeechBrain/speechbrain"

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

export CUDA_VISIBLE_DEVICES=0,1,2,3


singularity exec ${CONTAINER} $VENV $SCRIPT $CONFIG --device='cuda' --data_parallel_backend

#singularity exec ${CONTAINER} $VENV -m torch.distributed.launch --nproc_per_node=2 $SCRIPT $CONFIG --distributed_launch --distributed_backend='nccl' 
