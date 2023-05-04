#!/bin/bash

# Needed variables to be input from the user
VENV=$1 # virtual environement with SpeechBrain and all other libraries installed 

# Activates your virtual environment
eval "$(conda shell.bash hook)"
conda activate $VENV

#===== begin config =======
data_root=/home/jule/datasets
#data_root=/media/Daten/datasets
data_folder=$data_root/LibriParty/dataset
musan_folder=$data_root/musan
commonlanguage_folder=$data_root/common_voice_kpd
open_rir_folder=$data_root/RIRS_NOISES

#VAD configs
skip_prep=True
sample_rate=16000
n_mels=40 #80, 40, 20, 10
time_factor=1 #1, 4, 8, 12
smoothPSD=true
win_length=$((25*time_factor))
hop_length=$((10*time_factor))
n_fft=$((sample_rate*win_length/1000))

#=========== end config ===========

if $smoothPSD; then
    echo Running VAD with $n_mels mels and smoothed PSD ...

    python train.py hparams/train_olMEGA.yaml --data_folder=$data_folder --musan_folder=$musan_folder --commonlanguage_folder=$commonlanguage_folder --open_rir_folder=$open_rir_folder \
        --skip_prep=$skip_prep --n_mels=$n_mels --device=cuda:1
else
    echo Running VAD with $n_mels mels, win_length $win_length ms, hop_length $hop_length ms, nfft $n_fft ...

    python train.py hparams/train.yaml --data_folder=$data_folder --musan_folder=$musan_folder --commonlanguage_folder=$commonlanguage_folder --open_rir_folder=$open_rir_folder \
        --skip_prep=$skip_prep --n_mels=$n_mels --win_length=$win_length --hop_length=$hop_length --n_fft=$n_fft --device=cuda:1
fi
echo Done