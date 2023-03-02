# Codes for privacy preserving features

## ASR Pipeline 
This pipeline has been derived trom the LibriSpeech recipe. First it creates the data manifests in a json format and the uses them for the training of an ASR (transformer + CTC) model.

### Create the data manifest

- go to custom_training/ASR/create_data_manifest/ and you will find create_data_manifest_ASR.py
- the function can be run as follows 

   python create_data_manifest_ASR.py --data-folder /path/to/dataset \ 
                                      --save-json-train path/wheretosave/train.json \ 
                                      --save-json-valid path/wheretosave/valid.json \ 
                                      --save-json-test  path/wheretosave/test.json \
                                      --extension extension/of/audio/files (wav, flac ex.) \
                                      --transcripts-folder path/to/folder/with/transcripts \
