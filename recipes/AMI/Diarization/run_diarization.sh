#!/bin/bash

#===== begin config =======
dset=/home/jule/datasets/amicorpus

#diarization configs
oracle_n_spkrs=True
skip_prep=True

#=========== end config ===========
echo Diarizing $dset ...

python experiment.py hparams/ecapa_tdnn_olMEGA.yaml --data_folder=$dset --manual_annot_folder=$dset/ami_public_manual --oracle_n_spkrs=$oracle_n_spkrs --skip_prep=$skip_prep

echo Done