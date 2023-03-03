---
language: "en"
thumbnail:
tags:
- speechbrain
- VAD
- SAD
- Voice Activity Detection
- Speech Activity Detection
- Speaker Diarization
- pytorch
- CRDNN
- LibriSpeech
- LibryParty
datasets:
- Urbansound8k
metrics:
- Accuracy

---

<iframe src="https://ghbtns.com/github-btn.html?user=speechbrain&repo=speechbrain&type=star&count=true&size=large&v=2" frameborder="0" scrolling="0" width="170" height="30" title="GitHub"></iframe>
<br/><br/>

# Voice Activity Detection with a (small) CRDNN model trained on Libriparty

This repository provides all the necessary tools to perform voice activity detection with SpeechBrain using a model pretrained on Libriparty.

The pre-trained system can process short and long speech recordings and outputs the segments where speech activity is detected. 
The output of the system looks like this:

```
segment_001  0.00  2.57 NON_SPEECH
segment_002  2.57  8.20 SPEECH
segment_003  8.20  9.10 NON_SPEECH
segment_004  9.10  10.93 SPEECH
segment_005  10.93  12.00 NON_SPEECH
segment_006  12.00  14.40 SPEECH
segment_007  14.40  15.00 NON_SPEECH
segment_008  15.00  17.70 SPEECH
```

The system expects input recordings sampled at 16kHz (single channel).
If your signal has a different sample rate, resample it (e.g., using torchaudio or sox) before using the interface.

For a better experience, we encourage you to learn more about
[SpeechBrain](https://speechbrain.github.io).

# Results
The model performance on the LibriParty test set is:

| Release | hyperparams file | n_mels | Test Precision | Test Recall | Test F-Score | Model link | GPUs |
|:-------------:|:----------------------------:| -------:| -------:| -------:| --------:| :-----------:|  :-----------:|
| 2021-09-09 | train.yaml | 40 |  0.9518 | 0.9437 | 0.9477 | [Model](https://drive.google.com/drive/folders/1YLYGuiyuTH0D7fXOOp6cMddfQoM74o-Y?usp=sharing) | 1xV100 16GB
| 2023-02-10 | train.yaml | 40 |  0.9471 | 0.9503 | 0.9487 | JP |
| 2023-03-02 | train_olMEGA.yaml | 40 |  0.9431 | 0.8645 | 0.9021 | JP |


## Pipeline description
This system is composed of a CRDNN that outputs posteriors probabilities with a value close to one for speech frames and close to zero for non-speech segments. 
A threshold is applied on top of the posteriors to detect candidate speech boundaries. 

Depending on the active options, these boundaries can be post-processed  (e.g, merging close segments, removing short segments, etc) to further improve the performance. See more details below.

## Install SpeechBrain

```
pip install speechbrain
```

Please notice that we encourage you to read our tutorials and learn more about
[SpeechBrain](https://speechbrain.github.io).

### Perform Voice Activity Detection

```
from speechbrain.pretrained import VAD

VAD = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="pretrained_models/vad-crdnn-libriparty")
boundaries = VAD.get_speech_segments("speechbrain/vad-crdnn-libriparty/example_vad.wav")

# Print the output
VAD.save_boundaries(boundaries)
```
The output is a tensor that contains the beginning/end second of each
detected speech segment. You can save the boundaries on a file with:

```
VAD.save_boundaries(boundaries, save_path='VAD_file.txt')
```

Sometimes it is useful to jointly visualize the VAD output with the input signal itself. This is helpful to quickly figure out if the VAD is doing or not a good job.  

To do it:

```
import torchaudio
upsampled_boundaries = VAD.upsample_boundaries(boundaries, 'pretrained_model_checkpoints/example_vad.wav')    
torchaudio.save('vad_final.wav', upsampled_boundaries.cpu(), 16000) 
```  

This creates a "VAD signal" with the same dimensionality as the original signal. 

You can now open *vad_final.wav* and *pretrained_model_checkpoints/example_vad.wav* with software like audacity to visualize them jointly. 


### VAD pipeline details
The pipeline for detecting the speech segments is the following:
1. Compute posteriors probabilities at the frame level.
2. Apply a threshold on the posterior probability.
3. Derive candidate speech segments on top of that.
4. Apply energy VAD within each candidate segment (optional). This might break down long sentences into short one based on the energy content.
5. Merge segments that are too close.
6. Remove segments that are too short.
7. Double-check speech segments (optional). This could is a final check to make sure the detected segments are actually speech ones.

We designed the VAD such that you can have access to all of these steps (this might help to debug):


```python
from speechbrain.pretrained import VAD
VAD = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="pretrained_models/vad-crdnn-libriparty")

# 1- Let's compute frame-level posteriors first
audio_file = 'pretrained_model_checkpoints/example_vad.wav'
prob_chunks = VAD.get_speech_prob_file(audio_file)

# 2- Let's apply a threshold on top of the posteriors
prob_th = VAD.apply_threshold(prob_chunks).float()

# 3- Let's now derive the candidate speech segments
boundaries = VAD.get_boundaries(prob_th)

# 4- Apply energy VAD within each candidate speech segment (optional)

boundaries = VAD.energy_VAD(audio_file,boundaries)

# 5- Merge segments that are too close
boundaries = VAD.merge_close_segments(boundaries, close_th=0.250)

# 6- Remove segments that are too short
boundaries = VAD.remove_short_segments(boundaries, len_th=0.250)

# 7- Double-check speech segments (optional).
boundaries = VAD.double_check_speech_segments(boundaries, audio_file,  speech_th=0.5)
``` 


### Inference on GPU
To perform inference on the GPU, add  `run_opts={"device":"cuda"}`  when calling the `from_hparams` method.

### Training
The model was trained with SpeechBrain (ea17d22).
To train it from scratch follows these steps:
1. Clone SpeechBrain:
```bash
git clone https://github.com/speechbrain/speechbrain/
```
2. Install it:
```
cd speechbrain
pip install -r requirements.txt
pip install -e .
```

3. Run Training:
Training heavily relies on data augmentation.  Make sure you have downloaded all the datasets needed:

		- LibriParty: https://drive.google.com/file/d/1--cAS5ePojMwNY5fewioXAv9YlYAWzIJ/view?usp=sharing
		- Musan: https://www.openslr.org/resources/17/musan.tar.gz
		- CommonLanguage: https://zenodo.org/record/5036977/files/CommonLanguage.tar.gz?download=1

```
cd recipes/LibriParty/VAD
python train.py hparams/train.yaml --data_folder=/path/to/LibriParty --musan_folder=/path/to/musan/ --commonlanguage_folder=/path/to/common_voice_kpd
```

### Limitations
The SpeechBrain team does not provide any warranty on the performance achieved by this model when used on other datasets.

# **Citing SpeechBrain**
Please, cite SpeechBrain if you use it for your research or business.


```bibtex
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```
