#!/usr/bin/env python3.0
# -*- coding: utf-8 -*-
"""
@author: Jose Patino, Massimiliano Todisco, Pramod Bachhav, Nicholas Evans
Audio Security and Privacy Group, EURECOM
Modified by Jule Pohlhausen, 2023
"""
import os
import librosa
import numpy as np
import scipy
import soundfile
import argparse
from tqdm import tqdm

def anonym(file, output_file, winLengthinms=20, shiftLengthinms=10, lp_order=20, mcadams=0.8, emph_coef=0.97):    
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
        ind_imag_con = ind_imag[np.arange(0,np.size(ind_imag),2)]
        
        # here we define the new angles of the poles, shifted accordingly to the mcadams coefficient
        # values >1 expand the spectrum, while values <1 constract it for angles>1
        # values >1 constract the spectrum, while values <1 expand it for angles<1
        # the choice of this value is strongly linked to the number of lpc coefficients
        # a bigger lpc coefficients number constraints the effect of the coefficient to very small variations
        # a smaller lpc coefficients number allows for a bigger flexibility
        new_angles = np.angle(poles[ind_imag_con]) ** mcadams

        # make sure new angles stay between 0 and pi
        new_angles[np.where(new_angles>=np.pi)] = np.pi        
        new_angles[np.where(new_angles<=0)] = 0  
        
        # copy of the original poles to be adjusted with the new angles
        new_poles = poles
        for k in np.arange(np.size(ind_imag_con)):
            # compute new poles with the same magnitued and new angles
            new_poles[ind_imag_con[k]] = np.abs(poles[ind_imag_con[k]])*np.exp(1j*new_angles[k])
            # applied also to the conjugate pole
            new_poles[ind_imag_con[k]+1] = np.abs(poles[ind_imag_con[k]+1])*np.exp(-1j*new_angles[k])            
        
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
    sig_rec = sig_rec/np.max(np.abs(sig_rec))
    soundfile.write(output_file, np.float32(sig_rec), fs) 
    return []

if __name__ == "__main__":
    #Parse args    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=str)    
    parser.add_argument('--dataset',type=str,default='')
    parser.add_argument('--wav_scp',type=str,default=None)
    parser.add_argument('--anon_suffix',type=str,default='_anon')
    parser.add_argument('--n_coeffs',type=int,default=20)
    parser.add_argument('--mc_coeff',type=float,default=0.8)
    parser.add_argument('--mc_coeff_min', type=float, default=0.5)
    parser.add_argument('--mc_coeff_max', type=float, default=0.9)
    parser.add_argument('--mode', type=str,default='rand')
    parser.add_argument('--winLengthinms',type=int,default=20)
    parser.add_argument('--shiftLengthinms',type=int,default=10)
    config = parser.parse_args()

    np.random.seed(1234)
    
    #Load protocol file
    if config.wav_scp is None:
        list_name = os.path.join(config.data_dir, 'wav.scp')
    else:
        list_name = config.wav_scp
    list_files = np.genfromtxt(list_name,dtype='U')
    
    output_dir = config.data_dir.replace(config.dataset, config.dataset+config.anon_suffix)
    os.makedirs(output_dir, exist_ok=True)

    if "rand" in config.mode:
        f = open(output_dir + '/mcadams_rand.txt', 'w')
    
    if "per_speaker" in config.mode or "per_rec" in config.mode:
        old_id = ""

    for idx,file in enumerate(tqdm(list_files)):   
        if "VPC" in config.wav_scp:
            input_file = os.path.join(config.data_dir, file[1].replace('data/', ''))
        else:
            input_file = file[2]

        # change outpt dirname and create folders
        output_file = input_file.replace(config.data_dir, output_dir)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        if not os.path.exists(output_file):
            if config.mode == "rand":
                config.mc_coeff = np.random.uniform(config.mc_coeff_min, config.mc_coeff_max)
                f.write(output_file + ', ' + str(config.mc_coeff) + '\n')
            elif config.mode == "rand_per_speaker":
                spk_id = output_file.split("/")[-1].split(".")[0].split("-")[0].split("_")[0]
                if not spk_id == old_id:
                    config.mc_coeff = np.random.uniform(config.mc_coeff_min, config.mc_coeff_max)
                old_id = spk_id
                f.write(output_file + ', ' + str(config.mc_coeff) + '\n')
            elif config.mode == "rand_per_rec_session":
                rec_id = "-".join(output_file.split("/")[-1].split(".")[0].split("-")[:2])
                if not rec_id == old_id:
                    config.mc_coeff = np.random.uniform(config.mc_coeff_min, config.mc_coeff_max)
                old_id = rec_id
                f.write(output_file + ', ' + str(config.mc_coeff) + '\n')
            
            anonym(input_file, output_file, winLengthinms=config.winLengthinms, shiftLengthinms=config.shiftLengthinms, lp_order=config.n_coeffs, mcadams=config.mc_coeff, mode=config.mode)
       
    if "rand" in config.mode:
        f.close()