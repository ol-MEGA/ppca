#!/bin/bash

#===== begin config =======
data_folder=/home/jule/datasets/LibriParty/dataset
musan_folder=/home/jule/datasets/musan
commonlanguage_folder=/home/jule/datasets/common_voice_kpd

#VAD configs
skip_prep=True
sample_rate=16000
n_mels=80 #80, 40, 20, 10
time_factor=1 #1, 4, 8, 12
win_length=$((25*time_factor))
hop_length=$((10*time_factor))
n_fft=$((sample_rate*win_length/1000))

#=========== end config ===========
echo Running VAD with win_length $win_length ms, hop_length $hop_length ms, nfft $n_fft ...

python train.py hparams/train.yaml --data_folder=$data_folder --musan_folder=$musan_folder --commonlanguage_folder=$commonlanguage_folder \
        --skip_prep=$skip_prep --n_mels=$n_mels --win_length=$win_length --hop_length=$hop_length --n_fft=$n_fft --device=cuda:1

echo Done