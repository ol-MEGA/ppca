#!/bin/bash

datadir_clean=$1
py=$2
jsoname=$3
sep="-"


declare -a datadirs=( $datadir_clean )

for datadir in ${datadirs[@]}; do

   $py generate_datamanifest.py --data-dir $datadir \
   --save-json ${jsoname}.json \
   --data-type  clean \
   --spk-sep $sep

done