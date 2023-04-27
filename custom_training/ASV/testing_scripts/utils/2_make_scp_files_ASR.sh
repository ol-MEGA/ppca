#!/bin/bash

basedir="/raid/francesco_nespoli/SpeechBrain/plda_scoring/verification_data"

for dset in vst_input_norm_gp  \
            ; do

mkdir $basedir/$dset/wav_scp_files_anon/ASR
scp_dir=$basedir/$dset/wav_scp_files_anon

cat $scp_dir/libri_dev_trials_f_anon.scp $scp_dir/libri_dev_trials_m_anon.scp >> $basedir/$dset/wav_scp_files_anon/ASR/libri_dev_asr_anon.scp
cat $scp_dir/libri_test_trials_f_anon.scp $scp_dir/libri_test_trials_m_anon.scp >> $basedir/$dset/wav_scp_files_anon/ASR/libri_test_asr_anon.scp

cat $scp_dir/vctk_dev_trials_f_anon.scp $scp_dir/vctk_dev_trials_f_common_anon.scp >> $scp_dir/vctk_dev_trials_f_all.scp
cat $scp_dir/vctk_dev_trials_m_anon.scp $scp_dir/vctk_dev_trials_m_common_anon.scp >> $scp_dir/vctk_dev_trials_m_all.scp
cat $scp_dir/vctk_dev_trials_f_all.scp $scp_dir/vctk_dev_trials_m_all.scp >> $basedir/$dset/wav_scp_files_anon/ASR/vctk_dev_asr_anon.scp

cat $scp_dir/vctk_test_trials_f_anon.scp $scp_dir/vctk_test_trials_f_common_anon.scp >> $scp_dir/vctk_test_trials_f_all.scp
cat $scp_dir/vctk_test_trials_m_anon.scp $scp_dir/vctk_test_trials_m_common_anon.scp >> $scp_dir/vctk_test_trials_m_all.scp
cat $scp_dir/vctk_test_trials_f_all.scp $scp_dir/vctk_test_trials_m_all.scp >> $basedir/$dset/wav_scp_files_anon/ASR/vctk_test_asr_anon.scp


done
