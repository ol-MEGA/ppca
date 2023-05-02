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
anon_data_suffix=_mcadams

#McAdams anonymisation configs
n_lpc=20
mcadams=0.8
mcadams_rand=false

if $mcadams_rand; then
    anon_data_suffix=_mcadams_rand
fi

#=========== end config ===========
echo Anonymizing $dset ...

#write wav.scp file with all wav files
if [ ! -f $dset/wav.scp ]; then
    ls $dset/*/*/*.wav | awk -F'[/]' '{print $NF " sox " $0 " -t wav -r 16000 -b 16 - |"}' > $dset/wav.scp
fi

#copy content of the folder to the new folder
#cp -r $dset $dset$anon_data_suffix

#create folder that will contain the anonymised wav files
mkdir -p $dset$anon_data_suffix

#anonymise dataset 
if $mcadams_rand; then
    python mcadams/anonymise_dir_mcadams.py --data_dir=$dset --anon_suffix=$anon_data_suffix --n_coeffs=$n_lpc --mc_coeff=$mcadams --mc_rand
else
    python mcadams/anonymise_dir_mcadams.py --data_dir=$dset --anon_suffix=$anon_data_suffix --n_coeffs=$n_lpc --mc_coeff=$mcadams --no-mc_rand
fi

#overwrite wav.scp file with new anonymised content
#note sox is inclued to by-pass that files written by local/anon/anonymise_dir_mcadams.py were in float32 format and not pcm
ls $dset$anon_data_suffix/*/*/*.wav | awk -F'[/]' '{print $NF " sox " $0 " -t wav -r 16000 -b 16 - |"}' > $dset$anon_data_suffix/wav.scp

echo Done