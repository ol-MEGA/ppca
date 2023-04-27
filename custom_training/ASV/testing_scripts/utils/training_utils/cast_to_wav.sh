#!/bin/bash

files=$(find "/raid/francesco_nespoli/SpeechBrain/plda_scoring/training_utils/train-clean-100" -name "*.flac")

for file in $files;do

 filez=${file%.*}

 sox $file $filez.wav

 rm -rf $file

done
