# Privacy preserving conversation analysis (PPCA)
We conducted our experiments with the open-source speechprocessing toolkit [SpeechBrain](http://speechbrain.github.io/).
We consider the two main privacy aspects for speech recordings: the linguistic speech content and the speaker identity, hence Automatic Speech Recognition (ASR) model and  Automatic Speaker Verification (ASV). Our utility tasks are Voice Activity
detection (VAD) and Speaker Diarization (SD).

## ASR Pipeline 
This pipeline has been derived from the [SpeechBrain](http://speechbrain.github.io/) LibriSpeech ASR recipe. First it creates the data manifests in a json format and then uses them for the training of an ASR (transformer + CTC) model.

### Create the data manifest

- go to custom_training/ASR/create_data_manifest/ and you will find create_data_manifest_ASR.py
- the function can be run as follows 

   `python create_data_manifest_ASR.py --data-folder /path/to/dataset  --save-json-train path/wheretosave/train.json --save-json-valid path/wheretosave/valid.json --save-json-test  path/wheretosave/test.json --extension extension/of/audio/files (wav, flac ex.) --transcripts-folder path/to/folder/with/transcripts `

this function generate 3 json files with the training, validation and test sets.

### ASR training

- to run the training of the model, custom_training/ASR/training_script/gpu.sh
- there are different parameters you need to adjust, specifically in the hparams/.yaml file



## ASV Pipeline 
This pipeline has been derived from the [SpeechBrain](http://speechbrain.github.io/) VoxCeleb SpeakerRec recipe. First it creates the data manifests in a json format and then uses them for the training of an ASV model using ECAPA-TDNN embeddings.

### Create the data manifest

- go to custom_training/ASV/create_data_manifest/ and you will find 1_create_data_manifest.py
- the function can be run as follows 

   `python 1_create_data_manifest.py --data-folder /path/to/dataset  --save-json-train path/wheretosave/train.json --save-json-valid path/wheretosave/valid.json --save-json-test  path/wheretosave/test.json --extension extension/of/audio/files(wav, flac ex.) --transcripts-folder path/to/folder/with/transcripts `

   	this function generate 3 json files with the training, validation and test sets.

### ASV training
- to run the training of the model, custom_training/ASV/training_script/submit_training.sh
- there are different parameters you need to adjust, specifically in the hparams/.yaml file


## VAD Pipeline 
The voice activity detection is based on the [SpeechBrain](http://speechbrain.github.io/) LibriParty VAD recipe. 
Run the following command to train the model:
`python train.py hparams/train.yaml --data_folder=/localscratch/LibriParty/dataset/ --musan_folder=/localscratch/musan/ --commonlanguage_folder=/localscratch/common_voice_kpd`
(change the paths with your local ones)


## SD Pipeline 
The speaker diarization is based on the [SpeechBrain](http://speechbrain.github.io/) AMI Diarization recipe. 
Use the following command to run diarization on AMI corpus `python experiment.py hparams/ecapa_tdnn.yaml` or on VPC simulated conversations data `python experiment_vpc.py hparams/ecapa_tdnn_vpc.yaml`. There are different parameters you need to adjust, specifically in the hparams/.yaml file.
