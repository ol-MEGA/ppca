#!/bin/bash

datadir=$1
py=$2

for dset in libri_dev_{enrolls_anon,trials_f_anon,trials_m_anon} \
            vctk_dev_{enrolls_anon,trials_f_anon,trials_m_anon,trials_f_common_anon,trials_m_common_anon} \
            libri_test_{enrolls_anon,trials_f_anon,trials_m_anon} \
            vctk_test_{enrolls_anon,trials_f_anon,trials_m_anon,trials_f_common_anon,trials_m_common_anon}; do


   # Wavscp file
   wavscp=$datadir/data/${dset}/wav.scp

   # Separator
   dataset="$(cut -d'_' -f1 <<<$dset)"
   if [[ $dataset == "vctk"  ]];then
      sep="_"
   else
      sep="-"
   fi


   $py generate_plda_data_manifest.py --data-dir $datadir \
   --wav-scp-file $wavscp \
   --save-json ${dset}.json \
   --data-type  anon \
   --spk-sep sep


done