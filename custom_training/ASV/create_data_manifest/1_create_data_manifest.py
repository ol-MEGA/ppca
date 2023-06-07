"""
Downloads and creates data manifest files for Mini LibriSpeech (spk-id).
For speaker-id, different sentences of the same speaker must appear in train,
validation, and test sets. In this case, these sets are thus derived from
splitting the original training set intothree chunks.
Authors:
 * Mirco Ravanelli, 2021
"""
import argparse
import os
import json
import shutil
import random
import logging
import numpy as np
from speechbrain.utils.data_utils import get_all_files, download_file
from speechbrain.dataio.dataio import read_audio

logger = logging.getLogger(__name__)
MINILIBRI_TRAIN_URL = "http://www.openslr.org/resources/31/train-clean-5.tar.gz"
SAMPLERATE = 16000

def create_json(wav_list, json_file):
    """
    Creates the json file given a list of wav files.
    Arguments
    ---------
    wav_list : list of str
        The list of wav files.
    json_file : str
        The path of the output json file
    """
    # Processing all the wav files in the list
    json_dict = {}
    i = 0
    for wav_file in wav_list:

#        print("Utterance ID {}".format(i))
        # Reading the signal (to retrieve duration in seconds)
        signal = read_audio(wav_file)
        duration = signal.shape[0] / SAMPLERATE

        # Manipulate path to get relative path and uttid
        path_parts = wav_file.split(os.path.sep)
        uttid = i #os.path.splitext(path_parts[])
 
        relative_path = os.path.join("{data_root}", *path_parts[-5:])

        # Getting speaker-id from utterance-id
        spk_id, _ = os.path.splitext(path_parts[-1])
        spk_id = spk_id.split("-")[0]

        # Create entry for this utterance
        json_dict[uttid] = {
            "wav": relative_path,
            "length": duration,
            "spk_id": spk_id,
        }
        i+=1

    # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file} successfully created!")


def skip(*filenames):
    """
    Detects if the data preparation has been already done.
    If the preparation has been done, we can skip it.
    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True

def check_folders(*folders):
    """Returns False if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True


def split_sets(wav_list, split_ratio):
    """Randomly splits the wav list into training, validation, and test lists.
    Note that a better approach is to make sure that all the classes have the
    same proportion of samples (e.g, spk01 should have 80% of samples in
    training, 10% validation, 10% test, the same for speaker2 etc.). This
    is the approach followed in some recipes such as the Voxceleb one. For
    simplicity, we here simply split the full list without necessarily respecting
    the split ratio within each class.
    Arguments
    ---------
    wav_lst : list
        list of all the signals in the dataset
    split_ratio: list
        List composed of three integers that sets split ratios for train, valid,
        and test sets, respectively. For instance split_ratio=[80, 10, 10] will
        assign 80% of the sentences to training, 10% for validation, and 10%
        for test.
    Returns
    ------
    dictionary containing train, valid, and test splits.
    """
    # Random shuffle of the list
    random.shuffle(wav_list)
    tot_split = sum(split_ratio)
    tot_snts = len(wav_list)
    data_split = {}
    splits = ["train", "valid"]
    for i, split in enumerate(splits):
        n_snts = int(tot_snts * split_ratio[i] / tot_split)
        data_split[split] = wav_list[0:n_snts]
        del wav_list[0:n_snts]
    data_split["test"] = wav_list

    return data_split

def download_mini_librispeech(destination):
    """Download dataset and unpack it.
    Arguments
    ---------
    destination : str
        Place to put dataset.
    """
    train_archive = os.path.join(destination, "train-clean-5.tar.gz")
    download_file(MINILIBRI_TRAIN_URL, train_archive)
    shutil.unpack_archive(train_archive, destination)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process input data.')
    parser.add_argument('--save-json-train', type=str, help='Path where the train data specification file will be saved')
    parser.add_argument('--save-json-valid', type=str, help='Path where the validation data specification file will be saved.')
    parser.add_argument('--save-json-test', type=str, help='Path where the test data specification file will be saved.')
    parser.add_argument('--split-ratio', type=list, default=[95, 5, 0], help='List composed of three integers that sets split ratios for train, valid, and test sets, respectively. For instance split_ratio')
    parser.add_argument('--filelist', type=str, help='List of files to be processed')


    args = parser.parse_args()

    """
    Prepares the json files for the Mini Librispeech dataset.
    Downloads the dataset if it is not found in the `data_folder`.
    Arguments
    ---------
    data_folder : str
        Path to the folder where the Mini Librispeech dataset is stored.
    save_json_train : str
        Path where the train data specification file will be saved.
    save_json_valid : str
        Path where the validation data specification file will be saved.
    save_json_test : str
        Path where the test data specification file will be saved.
    split_ratio: list
        List composed of three integers that sets split ratios for train, valid,
        and test sets, respectively. For instance split_ratio=[80, 10, 10] will
        assign 80% of the sentences to training, 10% for validation, and 10%
        for test.
    Example
    -------
    >>> data_folder = '/path/to/mini_librispeech'
    >>> prepare_mini_librispeech(data_folder, 'train.json', 'valid.json', 'test.json')
    """

    # Check if this phase is already done (if so, skip it)
    if skip(args.save_json_train, args.save_json_valid, args.save_json_test):
        logger.info("Preparation completed in previous run, skipping.")
        exit()


    # List files and create manifest from list
    logger.info(
        f"Creating {args.save_json_train}, {args.save_json_valid}, and {args.save_json_test}"
    )
   
    # Get all files
    with open(args.filelist) as file:
       lines = [line.rstrip() for line in file]
 
    # Get only wavs (From Dushyant file!) 
    wav_list = []
    for line in lines:
       fields = line.split(" ")
       wav_list.append(fields[0])


    spks = [i.split("/")[-1].split("-")[0] for i in wav_list]
    print(len(np.unique(spks)) )
    print(len(wav_list))
    exit()
    # Random split the signal list into train, valid, and test sets.
    data_split = split_sets(wav_list, args.split_ratio)
#    print(len(data_split["train"]))
#    print(len(data_split["valid"]))
#    print(len(data_split["test"]))
#    exit()

    # Creating json files
    create_json(data_split["train"], args.save_json_train)
    create_json(data_split["valid"], args.save_json_valid)
    create_json(data_split["test"], args.save_json_test)

