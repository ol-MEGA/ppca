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

this function generates 3 json files with the training, validation and test sets.

### ASR training

- to run the training of the model use custom_training/ASR/training_script/gpu.sh
- there are different parameters you need to adjust, specifically in the hparams/.yaml file



## ASV Pipeline 
This pipeline has been derived from the [SpeechBrain](http://speechbrain.github.io/) VoxCeleb [SpeakerRec](recipes/VoxCeleb/SpeakerRec) recipe. 
For creating the data manifests in a json format please refer to [`voxceleb_prepare.py`](recipes/VoxCeleb/voxceleb_prepare.py) in the [SpeechBrain](http://speechbrain.github.io/) VoxCeleb recipe. The data manifets are used for the training of an ASV model using [ECAPA-TDNN](https://arxiv.org/abs/2005.07143) embeddings.

### ASV training
- Run the following command to train speaker embeddings using [ECAPA-TDNN](https://arxiv.org/abs/2005.07143):

   `python train_speaker_embeddings.py hparams.yaml`

- there are different parameters you need to adjust, specifically in the hparams.yaml file


## VAD Pipeline 
The voice activity detection is based on the [SpeechBrain](http://speechbrain.github.io/) LibriParty VAD recipe. Please refer to the corresponding [README.md](recipes/LibriParty/VAD/README.md) for more informations on the preparation of the datasets. 

### VAD training
Run the following command to train the CRDNN model:

`python train.py hparams/train.yaml --data_folder=/data/LibriParty/dataset/ --musan_folder=/data/musan/ --commonlanguage_folder=/data/common_voice_kpd`
(change the paths with your local ones)


## SD Pipeline 
The speaker diarization is based on the [SpeechBrain](http://speechbrain.github.io/) AMI Diarization recipe. Please refer to the corresponding [README.md](recipes/AMI/Diarization/README.md) for more informations on the preparation of the datasets. The SD is based on the same [ECAPA-TDNN](https://arxiv.org/abs/2005.07143) embeddings as the ASV model.

### SD experiment
Use the following command to run diarization on AMI corpus `python experiment.py hparams/ecapa_tdnn.yaml` or on VPC simulated conversations data `python experiment_vpc.py hparams/ecapa_tdnn_vpc.yaml`. There are different parameters you need to adjust, specifically in the hparams/.yaml files.



## Simulated Conversations
In order to compare the ASV and SD performance on the same dataset, conversations were simulated based on the [VoicePrivacy Challenge](https://www.voiceprivacychallenge.org) eval and test data. Please refer to the branch simulate_conversations, where the submodule mms-msg is based on the Multipurpose Multi Speaker Mixture Signal Generator ([MMS-MSG](https://github.com/fgnt/mms_msg)).
