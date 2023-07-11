#!/bin/bash

data_root=$1
json_folder=$2
py=$3

for dset in vctk_dev_{trials_f,trials_m,trials_f_common,trials_m_common} \
            libri_test_{trials_f,trials_m} \
            vctk_test_{trials_f,trials_m,trials_f_common,trials_m_common}; do

   # Get basedir for enrollment data
   dataset=${dset%_trials*} 

   $py speaker_verification_cosine_similarity.py hparams/verification_ecapa.yaml \
   --data_folder_tr=$data_root --data_folder_test=$data_root \
   --subset=$dset --subset_enrol=${dataset}_enrolls \
   --json_folder=$json_folder --separator=$sep

done