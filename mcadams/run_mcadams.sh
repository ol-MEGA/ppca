#!/bin/bash
# Script for The First VoicePrivacy Challenge 2020
#
#
# Copyright (C) 2020  <Brij Mohan Lal Srivastava, Natalia Tomashenko, Xin Wang, Jose Patino,...>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

set -e

#===== begin config =======
dset=/home/jule/datasets/amicorpus
anon_data_suffix=_anon

#McAdams anonymisation configs
n_lpc=20
mcadams=0.8
mcadams_rand=False

#=========== end config ===========

#write wav.scp file with all wav files
ls data/$dset/*/*.wav | awk -F'[/.]' '{print $5 " sox " $0 " -t wav -r 16000 -b 16 - |"}' > $dset/wav.scp


#create folder that will contain the anonymised wav files
mkdir -p $dset$anon_data_suffix

#anonymise dataset 
#python mcadams/anonymise_dir_mcadams.py --data_dir=$dset/wav.scp --anon_suffix=$anon_data_suffix --n_coeffs=$n_lpc --mc_coeff=$mcadams --mc_rand=$mcadams_rand    

echo $dset
#overwrite wav.scp file with new anonymised content
#note sox is inclued to by-pass that files written by local/anon/anonymise_dir_mcadams.py were in float32 format and not pcm
#ls data/$dset$anon_data_suffix/wav/*/*.wav | awk -F'[/.]' '{print $5 " sox " $0 " -t wav -r 16000 -b 16 - |"}' > data/$dset$anon_data_suffix/wav.scp

echo Done