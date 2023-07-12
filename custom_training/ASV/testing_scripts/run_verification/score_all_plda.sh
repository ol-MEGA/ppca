#!/bin/bash

data_root=$1
json_folder=$2
py=$3

for dset in libri_dev_{trials_f,trials_m} \
            vctk_dev_{trials_f,trials_m,trials_f_common,trials_m_common} \
            libri_test_{trials_f,trials_m} \
            vctk_test_{trials_f,trials_m,trials_f_common,trials_m_common}; do

   # Get basedir name for enrollment data
   dataset=${dset%_trials*} 
   
   # Define Separator
   if [[ ${dset%%_*} == "vctk"  ]];then
      sep="_"
   else
      sep="'-'"
   fi
   
   $py speaker_verification_plda.py hparams/verification_ecapa.yaml \
   --data_folder_tr=$data_root --data_folder_test=$data_root --json_folder=$json_folder \
   --subset=$dset --subset_enrol=${dataset}_enrolls --separator=$sep 

done