'''
Apply pretrained VAD to VPC data
25.04.2023
'''
import os
from pathlib import Path
import click
import shutil
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from mms_msg.databases.single_speaker.vpc.database import VPC

SAMPLERATE = 16000

def save_boundaries(boundaries, save_path=None, print_boundaries=True, audio_file=None
):
    """Saves the boundaries on a file (and/or prints them)  in a readable format.

    Arguments
    ---------
    boundaries: torch.Tensor
        Tensor containing the speech boundaries. It can be derived using the
        get_boundaries method.
    save_path: path
        When to store the text file containing the speech/non-speech intervals.
    print_boundaries: Bool
        Prints the speech/non-speech intervals in the standard outputs.
    audio_file: path
        Path of the audio file containing the recording. The file is read
        with torchaudio. It is used here to detect the length of the
        signal.
    """
    # Create a new file if needed
    if save_path is not None:
        f = open(save_path, mode="a", encoding="utf-8")

    # Getting the total size of the input file
    if audio_file is not None:
        filename = audio_file.split('/')[-1]
        audio_info = sf.info(audio_file)
        audio_len = audio_info.duration
    else:
        filename = ''

    # Setting the rights format for second- or sample-based boundaries
    if boundaries.dtype == torch.int:
        value_format = "% i"
    else:
        value_format = "% .2f "

    # check for empty boundaries
    if boundaries.shape[0] == 0:
        boundaries = torch.empty((1, 2))
        boundaries[0, 0] = 0
        boundaries[0, 1] = audio_len

    # Printing speech and non-speech intervals
    last_end = 0
    cnt_seg = 0
    for i in range(boundaries.shape[0]):
        begin_value = boundaries[i, 0]
        end_value = boundaries[i, 1]

        if last_end != begin_value:
            cnt_seg = cnt_seg + 1
            print_str = (
                filename + "  segment_%03d " + value_format + value_format + " NON_SPEECH"
            )
            if print_boundaries:
                print(print_str % (cnt_seg, last_end, begin_value))
            if save_path is not None:
                f.write(print_str % (cnt_seg, last_end, begin_value) + "\n")

        cnt_seg = cnt_seg + 1
        print_str = filename + "  segment_%03d " + value_format + value_format + " SPEECH"
        if print_boundaries:
            print(print_str % (cnt_seg, begin_value, end_value))
        if save_path is not None:
            f.write(print_str % (cnt_seg, begin_value, end_value) + "\n")

        last_end = end_value

    # Managing last segment
    if audio_file is not None:
        if last_end < audio_len:
            cnt_seg = cnt_seg + 1
            print_str = (
                filename + "  segment_%03d " + value_format + value_format + " NON_SPEECH"
            )
            if print_boundaries:
                print(print_str % (cnt_seg, end_value, audio_len))
            if save_path is not None:
                f.write(print_str % (cnt_seg, end_value, audio_len) + "\n")

    if save_path is not None:
        f.close()


@click.command()
@click.option(
    '--json-path', '-j', default='vpc.json',
    help='Path to json file with all infos about dataset (cf. create_json.py)'
)
@click.option(
    '--database-path', '-d',
    help='Path to the folder containing the VPC test data',
)
def main(json_path, database_path):
    # get VPC test data
    db = VPC(json_path=json_path)
    dataset_names = db.dataset_names

    # init pretrained VAD
    from speechbrain.pretrained import VAD
    VAD = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="pretrained_models/vad-crdnn-libriparty")

    show_plots = False
    if show_plots:
        fig, axs = plt.subplots(4, 4, figsize=(16, 12))

    for s, subset in enumerate(dataset_names):
        dset = db.get_dataset(subset)
        print('*** running VAD on: ', subset)
        
        save_path = Path(os.path.join(database_path, subset, 'vad_info.txt'))

        if os.path.exists(save_path):
            os.remove(save_path)

        for f, file in enumerate(range(len(dset))):
            filename = dset[file]['audio_path']['observation']

            boundaries = VAD.get_speech_segments(
                audio_file=filename, 
                apply_energy_VAD=True, 
                large_chunk_size=40, 
                small_chunk_size=40
                )

            # Print the output
            #save_boundaries(boundaries, save_path=save_path, print_boundaries=False, audio_file=filename)

            # cut speech signal and write to new file
            output_path, filename_out = os.path.split(filename)
            output_path = output_path.replace('VPC', 'VPC_cutted')
            Path(output_path).mkdir(parents=True, exist_ok=True)
            if boundaries.shape[0] == 0:
                shutil.copyfile(filename, os.path.join(output_path, filename_out))
            else:
                start = int(np.floor(boundaries[0, 0].numpy()*SAMPLERATE))
                end = int(np.ceil(boundaries[-1, 1].numpy()*SAMPLERATE))
                speech, _ = sf.read(filename, start=start, stop=end)
                sf.write(os.path.join(output_path, filename_out), speech, SAMPLERATE)

            if show_plots:
                signal, fs = sf.read(filename)
                time = np.linspace(0, signal.shape[0]/fs, signal.shape[0])
                upsampled_boundaries = VAD.upsample_boundaries(boundaries, filename)  

                plt.sca(axs[s, f])
                plt.plot(time, signal)  
                plt.plot(time, upsampled_boundaries.squeeze())
        
    if show_plots:
        plt.savefig('VAD_VPC.png')


if __name__ == '__main__':
    main()