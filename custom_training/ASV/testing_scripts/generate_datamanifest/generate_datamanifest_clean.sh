#!/bin/bash

datadir=$1
py=$2

for dset in libri_dev_{enrolls,trials_f,trials_m} \
            vctk_dev_{enrolls,trials_f,trials_m,trials_f_common,trials_m_common} \
            libri_test_{enrolls,trials_f,trials_m} \
            vctk_test_{enrolls,trials_f,trials_m,trials_f_common,trials_m_common}; do

   # Wavscp file
   wavscp=$datadir/data/${dset}/wav.scp

   # Separator
   dataset="$(cut -d'_' -f1 <<<$dset)"
   if [[ $dataset == "vctk"  ]];then
      sep="_"
   else
      sep="-"
   fi


   $py generate_datamanifest.py --data-dir $datadir \
   --wav-scp-file $wavscp \
   --save-json ${dset}.json \
   --data-type  clean \
   --spk-sep sep


done