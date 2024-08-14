#!/bin/bash
# Script for The First VoicePrivacy Challenge 2020
#
#
# Copyright (C) 2020  <Brij Mohan Lal Srivastava, Natalia Tomashenko, Xin Wang, Jose Patino,...>
# Modified by Jule Pohlhausen, 2023
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
dset=$1 # path/to/VPC_mixed_meetings
json_path=$2 # path/to/json/of/iput/dataset
anon_data_suffix=_mcadams

#McAdams anonymisation configs
n_lpc=20
mcadams=0.8
mcadams_rand=true

if $mcadams_rand; then
    anon_data_suffix=_mcadams_rand
fi

#=========== end config ===========
echo Anonymizing $dset ...

#create folder that will contain the anonymised wav files
mkdir -p $dset$anon_data_suffix

#anonymise dataset 
if $mcadams_rand; then
    python mcadams/anonymise_mcadams_vpc.py --data_dir=$dset --json_path=$json_path --anon_suffix=$anon_data_suffix --n_coeffs=$n_lpc --mc_rand
else
    python mcadams/anonymise_mcadams_vpc.py --data_dir=$dset --json_path=$json_path --anon_suffix=$anon_data_suffix --n_coeffs=$n_lpc --mc_coeff=$mcadams --no-mc_rand
fi

echo Done