#!/bin/bash

#===== begin config =======
dset=/home/jule/datasets/amicorpus

#diarization configs
oracle_n_spkrs=True
skip_prep=True
repeatPSD=True

#=========== end config ===========
echo Diarizing $dset ...

if [[ "$repeatPSD" == "False" ]]; then
    python experiment.py hparams/ecapa_tdnn_olMEGA.yaml --data_folder=$dset --manual_annot_folder=$dset/ami_public_manual --oracle_n_spkrs=$oracle_n_spkrs --skip_prep=$skip_prep
else
    echo ... repeating smoothed PSDs ...
    python experiment.py hparams/ecapa_tdnn_olMEGA.yaml --data_folder=$dset --manual_annot_folder=$dset/ami_public_manual --oracle_n_spkrs=$oracle_n_spkrs --skip_prep=$skip_prep --repeatPSD=$repeatPSD --output_folder=results/ami/ecapa/olMEGArep --device=cuda:0
fi

echo Done