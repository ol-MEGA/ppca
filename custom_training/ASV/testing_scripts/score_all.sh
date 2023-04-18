#!/bin/bash

veridir="/raid/francesco_nespoli/SpeechBrain/plda_scoring/verification_config/x-vector/anon/VC"
scoring_files=$(find $veridir -type f)

source /raid/francesco_nespoli/SpeechBrain/venv/bin/activate

for verifile in $scoring_files; do

   python 2.1_speaker_verification_PLDA_VPC2022.py $verifile

done
