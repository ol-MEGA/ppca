#!/bin/bash

datadir_clean=$1
extension=$2
py=$3
jsoname=$4
sep="-"


declare -a datadirs=( $datadir_clean )

for datadir in ${datadirs[@]}; do

   $py generate_plda_data_manifest.py --data-dir $datadir \
   --save-json ${jsoname}.json \
   --data-type  clean \
   --spk-sep $sep

done