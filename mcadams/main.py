
# !/usr/bin/env python3.0
# -*- coding: utf-8 -*-
"""
@original authors: Jose Patino, Massimiliano Todisco, Pramod Bachhav, Nicholas Evans
Audio Security and Privacy Group, EURECOM
modified by Francesco Nespoli
"""
import os
import librosa
import numpy as np
import scipy
from scipy.io import wavfile
import wave
import argparse
import matplotlib.pyplot as plt
import random



def load_utt2spk(path):
    assert os.path.isfile(path), f'File does not exist {path}'
    table = np.genfromtxt(path, dtype='U')
    utt2spk = {utt: spk for utt, spk in table}
    return utt2spk


def anonym(freq, samples, winLengthinms=20, shiftLengthinms=10, lp_order=20, mcadams=0.8):
    # print(mcadams)
    eps = np.finfo(np.float32).eps
    samples = samples + eps

    # simulation parameters
    winlen = np.floor(winLengthinms * 0.001 * freq).astype(int)
    shift = np.floor(shiftLengthinms * 0.001 * freq).astype(int)
    length_sig = len(samples)

    # fft processing parameters
    NFFT = 2 ** (np.ceil((np.log2(winlen)))).astype(int)
    # anaysis and synth window which satisfies the constraint
    wPR = np.hanning(winlen)
    K = np.sum(wPR) / shift
    win = np.sqrt(wPR / K)
    Nframes = 1 + np.floor((length_sig - winlen) / shift).astype(int)  # nr of complete frames

    # carry out the overlap - add FFT processing
    sig_rec = np.zeros([length_sig])  # allocate output+'ringing' vector

    for m in np.arange(1, Nframes):
        # indices of the mth frame
        index = np.arange(m * shift, np.minimum(m * shift + winlen, length_sig))
        # windowed mth frame (other than rectangular window)
        frame = samples[index] * win
        # get lpc coefficients
        a_lpc = librosa.core.lpc(frame + eps, order=lp_order)
        # get poles
        poles = scipy.signal.tf2zpk(np.array([1]), a_lpc)[1]
        # index of imaginary poles
        ind_imag = np.where(np.isreal(poles) == False)[0]
        # index of first imaginary poles
        ind_imag_con = ind_imag[np.arange(0, np.size(ind_imag), 2)]

        # here we define the new angles of the poles, shifted accordingly to the mcadams coefficient
        # values >1 expand the spectrum, while values <1 constract it for angles>1
        # values >1 constract the spectrum, while values <1 expand it for angles<1
        # the choice of this value is strongly linked to the number of lpc coefficients
        # a bigger lpc coefficients number constraints the effect of the coefficient to very small variations
        # a smaller lpc coefficients number allows for a bigger flexibility
        new_angles = np.angle(poles[ind_imag_con]) ** mcadams
        # new_angles = np.angle(poles[ind_imag_con])**path[m]

        # make sure new angles stay between 0 and pi
        new_angles[np.where(new_angles >= np.pi)] = np.pi
        new_angles[np.where(new_angles <= 0)] = 0

        # copy of the original poles to be adjusted with the new angles
        new_poles = poles
        for k in np.arange(np.size(ind_imag_con)):
            # compute new poles with the same magnitued and new angles
            new_poles[ind_imag_con[k]] = np.abs(poles[ind_imag_con[k]]) * np.exp(1j * new_angles[k])
            # applied also to the conjugate pole
            new_poles[ind_imag_con[k] + 1] = np.abs(poles[ind_imag_con[k] + 1]) * np.exp(-1j * new_angles[k])

            # recover new, modified lpc coefficients
        a_lpc_new = np.real(np.poly(new_poles))
        # get residual excitation for reconstruction
        res = scipy.signal.lfilter(a_lpc, np.array(1), frame)
        # reconstruct frames with new lpc coefficient
        frame_rec = scipy.signal.lfilter(np.array([1]), a_lpc_new, res)
        frame_rec = frame_rec * win

        outindex = np.arange(m * shift, m * shift + len(frame_rec))
        # overlap add
        sig_rec[outindex] = sig_rec[outindex] + frame_rec
    sig_rec = (sig_rec / np.max(np.abs(sig_rec)) * (np.iinfo(np.int16).max - 1)).astype(np.int16)
    return sig_rec
    # scipy.io.wavfile.write(output_file, freq, np.float32(sig_rec))
    # awk -F'[/.]' '{print $5 " sox " $0 " -t wav -R -b 16 - |"}' > data/$dset$anon_data_suffix/wav.scp


if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str)
    #parser.add_argument('--anon_suffix', type=str, default='_anon')
    parser.add_argument('--n_coeffs', type=int, default=20)
    parser.add_argument('--mc_coeff_min', type=float, default=0.5)
    parser.add_argument('--mc_coeff_max', type=float, default=0.9)
    parser.add_argument('--winLengthinms', type=int, default=20)
    parser.add_argument('--shiftLengthinms', type=int, default=10)
    #parser.add_argument('--subset', type=str, default='vctk_dev_enrolls')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--outfile', type=str, default='out_mcadams.wav')
    config = parser.parse_args()

    samplerate, data = wavfile.read(config.file)

    rand_mc_coeff = random.uniform(config.mc_coeff_min, config.mc_coeff_max)

    samples = anonym(freq=samplerate, samples=data, winLengthinms=config.winLengthinms,
                     shiftLengthinms=config.shiftLengthinms, lp_order=config.n_coeffs,
                     mcadams=rand_mc_coeff)

    with wave.open(config.outfile, 'wb') as stream:
        stream.setframerate(samplerate)
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.writeframes(samples)
    # print(f'{utid} {config.outfile}', file=writer)

    print("Done")