from os import listdir
from os.path import isfile, join
import argparse
import os
import json
from speechbrain.utils.data_utils import get_all_files, download_file
from speechbrain.dataio.dataio import read_audio

SAMPLERATE = 16000

def create_json(wav_list, spks, json_file):
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
    for wav_file, spk in zip(wav_list, spks):

        # Reading the signal (to retrieve duration in seconds)
        signal = read_audio(wav_file)
        duration = signal.shape[0] / SAMPLERATE

        # Manipulate path to get relative path and uttid
        path_parts = wav_file.split(os.path.sep)
        uttid, _ = os.path.splitext(path_parts[-1])
        relative_path = os.path.join("{data_root}", *path_parts[-5:])

        # Getting speaker-id from utterance-id
        spk_id = uttid.split("-")[0]

        # Create entry for this utterance
        json_dict[uttid] = {
            "wav": relative_path,
            "length": duration,
            "spk_id": spk,
        }

    # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)


def read_scp(filename, data_dir, data_type, sep):
    wavs, spks = [], []
    with open(filename) as file:
        for line in file:
            if data_type == "anon":
               spk, _, wav, _, _, _, _, _, _, _ = line.split(" ")
#               print(spk)
            elif data_type == "clean":
               spk, wav = line.split(" ")

            spk = spk.split("{}".format(sep))
            wavs.append(os.path.join(data_dir, wav.rstrip()))
            spks.append(spk[0])

    return wavs, spks

def get_spk_data(data_dir, sep):
    onlyfiles = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
    wavs, spks = [], []
    for file in onlyfiles:
        wavs.append(file)  
        spks.append(file.split(sep)[0])
    return wavs, spks

if __name__ == "__main__":
    '''
    This code creates the SpeechBrain data manifest starting from a Kaldi formatting from the VPC 2022
    '''

    parser = argparse.ArgumentParser(description='Process input data.')
    parser.add_argument('--data-dir', type=str, help=' Path to the wav files')
    parser.add_argument('--wav-scp-file', default=None, type=str, help=' Path to the wav.scp file')
    parser.add_argument('--save-json', type=str, help='Path where the validation data specification file will be saved.')   
    parser.add_argument('--data-type', type=str, help=' Input data type:orig or anonymized.')
    parser.add_argument('--spk-sep', type=str, help='Spk separator')

    args = parser.parse_args()

    if args.wav_scp_file:
       data, spks = read_scp(args.wav_scp_file, args.data_dir, args.data_type, args.spk_sep)
    else:
       data, spks = get_spk_data(args.data_dir, args.spk_sep)
 
    create_json(data, spks, args.save_json)
