#!/bin/bash

YAML=$1
json_folder=$2
pretrain=$3
result_folder=$4
data_folder_tr=$5
data_folder_test=$6

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
   
   python speaker_verification_cosine_similarity.py $YAML \
   --data_folder_tr=$data_folder_tr --data_folder_test=$data_folder_test --json_folder=$json_folder \
   --subset=$dset --subset_enrol=${dataset}_enrolls --separator=$sep \
   --pretrain_path=$pretrain --result_folder=$result_folder

done