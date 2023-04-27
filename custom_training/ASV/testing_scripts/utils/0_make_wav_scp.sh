
#!/bin/bash

PYDIR="/raid/francesco_nespoli/SpeechBrain/plda_scoring"

VENV="/raid/francesco_nespoli/SpeechBrain/venv/bin/activate"
source $VENV

wavscp_dir="/raid/francesco_nespoli/SpeechBrain/plda_scoring/verification_data/vst_input_norm_gp/wav_scp_files_anon/new/corrected"

datadir="/raid/francesco_nespoli/SpeechBrain/plda_scoring/verification_data/vst_input_norm_gp/wav_scp_files_anon/new"

for dset in libri_dev_{enrolls,trials_f,trials_m,enrolls_anon,trials_f_anon,trials_m_anon} \
            vctk_dev_{enrolls,trials_f,trials_m,enrolls_anon,trials_f_anon,trials_m_anon,trials_f_common,trials_m_common,trials_f_common_anon,trials_m_common_anon} \
            train-clean-360-asv_anon \
            libri_test_{enrolls,trials_f,trials_m,enrolls_anon,trials_f_anon,trials_m_anon} \
            vctk_test_{enrolls,trials_f,trials_m,enrolls_anon,trials_f_anon,trials_m_anon,trials_f_common,trials_m_common,trials_f_common_anon,trials_m_common_anon}; do

   # Mode
   mode=$(echo $dset | rev | cut -d "_" -f1 | rev)
   if [[  $mode == "anon"  ]];then
     wav_scp=$datadir/$dset/wav.scp
     python modify_wav_scp.py --file $wav_scp --new-loc "/raid/francesco_nespoli/VST_pipeline/data/anon_test_datasets/INPUT_NORMALIZATION/vst_gender_preserv/" --set-name $dset --outfolder $wavscp_dir
   else
     echo "NOT WORKING ON: $dset"
   fi 



done
