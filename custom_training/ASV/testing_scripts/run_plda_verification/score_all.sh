#!/bin/bash

veridir=$1
py=$2

scoring_files=$(find $veridir -type f)

for verifile in $scoring_files; do

   $py 2.1_speaker_verification_PLDA_VPC2022.py $verifile

done
