#!/bin/bash

veridir=$1
py=$2

scoring_files=$(find $veridir -type f)

for verifile in $scoring_files; do

   $py speaker_verification_plda.py $verifile

done
