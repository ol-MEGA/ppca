import glob
from pathlib import Path
import json
import pandas as pd
import os
import torchaudio

def parse_to_json(dev_clean_root):

    flac_files = glob.glob(os.path.join(dev_clean_root, "**/*.flac"),
                           recursive=True)

    # in this dataset files names are spk_id-chapter_id-utterance_id.flac
    text_files = glob.glob(os.path.join(dev_clean_root, "**/*.txt"),
                           recursive=True)
    # we build a dictionary with words for each utterance
    words_dict = {}
    for txtf in text_files:
        with open(txtf, "r") as f:
            lines = f.readlines()
        for l in lines:
            l = l.strip("\n")
            utt_id = l.split(" ")[0]
            words = " ".join(l.split(" ")[1:])
            words_dict[utt_id] = words

    # we now build JSON examples

    examples = {}
    for utterance in flac_files:
        utt_id = Path(utterance).stem
        examples[utt_id] = {"file_path": utterance,
                            "words": words_dict[utt_id],
                            "spkID": "speaker_" + utt_id.split("-")[0],
                            "length": torchaudio.info(utterance).num_frames}

    with open("data.json", "w") as f:
        json.dump(examples, f, indent=4)


def parse_to_csv(dev_clean_root):
    flac_files = glob.glob(os.path.join(dev_clean_root, "**/*.flac"),
                           recursive=True)

    # in this dataset files names are spk_id-chapter_id-utterance_id.flac
    text_files = glob.glob(os.path.join(dev_clean_root, "**/*.txt"),
                           recursive=True)
    # we build a dictionary with words for each utterance
    words_dict = {}
    for txtf in text_files:
        with open(txtf, "r") as f:
            lines = f.readlines()
        for l in lines:
            l = l.strip("\n")
            utt_id = l.split(" ")[0]
            words = " ".join(l.split(" ")[1:])
            words_dict[utt_id] = words

    # we now build JSON examples

    examples = []
    for utterance in flac_files:
        utt_id = Path(utterance).stem
        examples.append({"id": utt_id,
                            "file_path": utterance,
                            "words": words_dict[utt_id],
                            "spkID": "speaker_" + utt_id.split("-")[0],
                            "duration": torchaudio.info(utterance).num_frames})

    dataframe = pd.DataFrame(examples)
    dataframe.to_csv("data.csv")