#!/bin/bash

PYDIR="/raid/francesco_nespoli/SpeechBrain/plda_scoring"

VENV="/raid/francesco_nespoli/SpeechBrain/venv/bin/activate"
source $VENV

for dset in libri_dev_{enrolls,trials_f,trials_m,enrolls_anon,trials_f_anon,trials_m_anon} \
            vctk_dev_{enrolls,trials_f,trials_m,enrolls_anon,trials_f_anon,trials_m_anon,trials_f_common,trials_m_common,trials_f_common_anon,trials_m_common_anon} \
            libri_test_{enrolls,trials_f,trials_m,enrolls_anon,trials_f_anon,trials_m_anon} \
            vctk_test_{enrolls,trials_f,trials_m,enrolls_anon,trials_f_anon,trials_m_anon,trials_f_common,trials_m_common,trials_f_common_anon,trials_m_common_anon}; do

   mkdir /raid/francesco_nespoli/SpeechBrain/plda_scoring/verification_data/test/$dset

   # Mode
   mode=$(echo $dset | rev | cut -d "_" -f1 | rev)
   if [[  $mode == "anon"  ]];then
     mode="anon"
   else
     mode="clean"
   fi 

   # Separator
   dataset="$(cut -d'_' -f1 <<<$dset)"
   if [[ $dataset == "vctk"  ]];then
      sep="_" 
   else
      sep="-"
   fi


   # Python
   wav_scp=/raid/francesco_nespoli/SpeechBrain/plda_scoring/verification_data/vst_input_norm_gp/wav_scp_files_anon/$dset.scp  #/raid/francesco_nespoli/Voice-Privacy-Challenge-2022/baseline/data/$dset/wav.scp
   python $PYDIR/generate_plda_data_manifest.py --data-dir /raid/francesco_nespoli/Voice-Privacy-Challenge-2022/baseline --wav-scp-file $wav_scp --save-json /raid/francesco_nespoli/SpeechBrain/plda_scoring/verification_data/test/$dset/$dset.json --data-type $mode --spk-sep $sep

done
