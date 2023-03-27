#!/bin/bash

#===== begin config =======
data_folder=/home/jule/datasets/LibriParty/dataset
musan_folder=/home/jule/datasets/musan
commonlanguage_folder=/home/jule/datasets/common_voice_kpd

#VAD configs
skip_prep=True
smoothPSD=True
repeatPSD=True

#=========== end config ===========
echo Running VAD on $data_folder ...

python train.py hparams/train_olMEGA.yaml --data_folder=$data_folder --musan_folder=$musan_folder --commonlanguage_folder=$commonlanguage_folder --skip_prep=$skip_prep --smoothPSD=$smoothPSD --repeatPSD=$repeatPSD --device=cuda:1

echo Done