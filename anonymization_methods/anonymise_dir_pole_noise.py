#!/usr/bin/env python3.0
# -*- coding: utf-8 -*-
"""
@author: Jule Pohlhausen, 2024
"""
import os
import librosa
import numpy as np
import scipy
import soundfile
import copy
import argparse
from tqdm import tqdm

def add_pole_noise(file, output_file, winLengthinms=20, shiftLengthinms=10, lp_order=20, emph_coef=0.97, 
            angles_shift=0, radii_gain=0, angle_min=np.pi/80):        
    sig, fs = librosa.load(file,sr=None)    
    eps = np.finfo(np.float32).eps
    sig = sig+eps

    # pre-emphasize signal
    sig = librosa.effects.preemphasis(sig, coef=emph_coef)
    
    # simulation parameters
    winlen = np.floor(winLengthinms*0.001*fs).astype(int)
    shift = np.floor(shiftLengthinms*0.001*fs).astype(int)
    length_sig = len(sig)
    
    # analysis and synth window which satisfies the constraint
    wPR = np.hanning(winlen)
    K = np.sum(wPR)/shift
    win = np.sqrt(wPR/K)
    Nframes = 1+np.floor((length_sig-winlen)/shift).astype(int) # nr of complete frames
    
    # carry out the overlap - add FFT processing
    sig_rec = np.zeros([length_sig]) # allocate output+'ringing' vector
    
    for m in np.arange(Nframes):
        # indices of the mth frame
        index = np.arange(m*shift,np.minimum(m*shift+winlen,length_sig))    
        # windowed mth frame (other than rectangular window)
        frame = sig[index]*win 
        # get lpc coefficients
        a_lpc = librosa.core.lpc(frame+eps, order=lp_order)
        # get poles
        poles = scipy.signal.tf2zpk(np.array([1]), a_lpc)[1]
        #index of imaginary poles
        ind_imag = np.where(np.isreal(poles)==False)[0]
        #index of first imaginary poles
        ind_imag_con = ind_imag[::2]
        
        # get radii and angles
        radii = np.abs(poles[ind_imag_con])
        new_angles = np.angle(poles[ind_imag_con])
        
        # shift radii and angles (optional per pole)
        new_radii = 1 - ((1 - radii) * radii_gain[ind_imag_con])
        new_angles += angles_shift[ind_imag_con]

        # make sure new radius stay between 0 and 0.99
        new_radii[np.where(new_radii >= 1)] = 0.99
        new_radii[np.where(new_radii <= 0)] = 0

        # make sure new angles stay between min and pi
        new_angles[np.where(new_angles >= np.pi)] = np.pi
        new_angles[np.where(new_angles < angle_min)] = angle_min
        
        # copy of the original poles to be adjusted with the new angles
        new_poles = copy.deepcopy(poles)
        for k in np.arange(np.size(ind_imag_con)):
            # compute new poles with the same magnitued and new angles
            new_poles[ind_imag_con[k]] = new_radii[k] * np.exp(1j * new_angles[k]) 
            # applied also to the conjugate pole
            new_poles[ind_imag_con[k] + 1] = new_radii[k] * np.exp(-1j * new_angles[k])

        # recover new, modified lpc coefficients
        a_lpc_new = np.real(np.poly(new_poles))
        # get residual excitation for reconstruction
        res = scipy.signal.lfilter(a_lpc,np.array(1),frame)
        # reconstruct frames with new lpc coefficient
        frame_rec = scipy.signal.lfilter(np.array([1]),a_lpc_new,res)
        frame_rec = frame_rec*win    
 
        outindex = np.arange(m*shift,m*shift+len(frame_rec))
        # overlap add
        sig_rec[outindex] += frame_rec
    # de-emphasize signal
    sig_rec = librosa.effects.deemphasis(sig_rec, coef=emph_coef)
    soundfile.write(output_file, np.float32(sig_rec), fs) 
    return []

if __name__ == "__main__":
    #Parse args    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=str)    
    parser.add_argument('--dataset',type=str,default='')
    parser.add_argument('--wav_scp',type=str,default=None)
    parser.add_argument('--enrolls',type=int,default=0)
    parser.add_argument('--anon_suffix',type=str,default='_pole_noise')
    parser.add_argument('--lp_order',type=int,default=20)
    parser.add_argument('--rate',type=int,default=16000)
    parser.add_argument('--angle_max_shift_hz',type=int,default=250)
    parser.add_argument('--angle_min', type=float, default=np.pi/80)
    parser.add_argument('--radius_max_gain', type=int, default=10)
    parser.add_argument('--mode', type=str,default='utt')
    parser.add_argument('--winLengthinms',type=int,default=20)
    parser.add_argument('--shiftLengthinms',type=int,default=10)
    config = parser.parse_args()

    if config.enrolls:
        np.random.seed(5678)
        rng = np.random.default_rng(5678)
    else:
        np.random.seed(1234)
        rng = np.random.default_rng()

    #Load protocol file
    if config.wav_scp is None:
        list_name = os.path.join(config.data_dir, 'wav.scp')
    else:
        list_name = config.wav_scp
    list_files = np.genfromtxt(list_name,dtype='U')

    if 'amicorpus' in config.data_dir or 'VPC' in config.data_dir:
        dirname = config.data_dir
        basename = ''
    else:
        dirname = os.path.dirname(config.data_dir)
        basename = os.path.basename(config.data_dir)
    output_dir = os.path.join(dirname, f'a_max_{config.angle_max_shift_hz}hz', f'r_max_{config.radius_max_gain}db', basename)
    output_dir = output_dir.replace(config.dataset, config.dataset+config.anon_suffix)
    os.makedirs(output_dir, exist_ok=True)

    if config.enrolls:
        f = open(os.path.join(output_dir, config.wav_scp.split("/")[-2], 'pole_noise_enrolls.txt'), 'w')
    else:
        f = open(os.path.join(output_dir, basename, 'pole_noise.txt'), 'w')
    
    if "per_speaker" in config.mode or "per_rec" in config.mode:
        old_id = ""

    for file in tqdm(list_files): 
        if config.enrolls:
            if "vctk" in config.data_dir: 
                input_file = os.path.join(config.data_dir, config.wav_scp.split("/")[-2], "wav", file.split("_")[0], file + ".wav")
            else:
                input_file = os.path.join(config.data_dir, config.wav_scp.split("/")[-2], "wav", file, file + ".wav")  
        else:
            if "VPC" in config.wav_scp:
                input_file = os.path.join(config.data_dir, file[1].replace('data/', ''))
            else:
                input_file = file[2]

        # change outpt dirname and create folders
        output_file = input_file.replace(config.data_dir, output_dir)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        angle_max_shift = config.angle_max_shift_hz / (config.rate / 2) * np.pi
        if not os.path.exists(output_file) or config.enrolls:
            if config.mode == "per_speaker":
                spk_id = output_file.split("/")[-1].split(".")[0].split("-")[0].split("_")[0]
                if not spk_id == old_id:
                    radii_gain = 10**(- rng.uniform(low=-config.radius_max_gain, high=config.radius_max_gain) / 20) * np.ones(config.lp_order)
                    angles_shift = rng.uniform(low=-angle_max_shift, high=angle_max_shift) * np.ones(config.lp_order)
                old_id = spk_id
            elif config.mode == "utt":
                radii_gain = 10**(- rng.uniform(low=-config.radius_max_gain, high=config.radius_max_gain) / 20) * np.ones(config.lp_order)
                angles_shift = rng.uniform(low=-angle_max_shift, high=angle_max_shift) * np.ones(config.lp_order)
            elif config.mode == "utt_per_pole":
                radii_gain = 10**(- rng.uniform(low=-config.radius_max_gain, high=config.radius_max_gain, size=config.lp_order) / 20)
                angles_shift = rng.uniform(low=-angle_max_shift, high=angle_max_shift, size=config.lp_order)
            elif config.mode == "rad_per_pole":
                radii_gain = 10**(- rng.uniform(low=-config.radius_max_gain, high=config.radius_max_gain, size=config.lp_order) / 20)
                angles_shift = angle_max_shift * np.ones(config.lp_order)
            elif config.mode == "fixed":
                radii_gain = 10**(- config.radius_max_gain / 20) * np.ones(config.lp_order)
                angles_shift = angle_max_shift * np.ones(config.lp_order)
            
            if "per_pole" in config.mode:
                f.write(f'{output_file}, {radii_gain}, {angles_shift}\n')
            else:
                f.write(f'{output_file}, {radii_gain[0]}, {angles_shift[0]}\n')
            
            add_pole_noise(input_file, output_file, winLengthinms=config.winLengthinms, shiftLengthinms=config.shiftLengthinms, 
                lp_order=config.lp_order, radii_gain=radii_gain, angles_shift=angles_shift)       
    f.close()