#!/usr/bin/env python3.0
# -*- coding: utf-8 -*-
"""
@author: Jule Pohlhausen
October 2023
"""
import librosa
import numpy as np
import scipy
import soundfile

def quantize_midtread(x, bits, range_max=1, range_min=-1, limit=True):
    '''
    quantizes the input sig x with a certain bit depth 
    using the midtread quantizing line 

    Parameters:
    -----------
    x : np.ndarray
        vector to quantize
    bits : int
        quantizing depth in bits

    Returns:
    out : np.ndarray
        quantized input vector
    '''
    stepsize = (range_max - range_min)/(2**int(bits))
    index = np.round(x / stepsize)
    out = index * stepsize
    if limit:
        out_max = range_max - stepsize
        out[np.where(out > out_max)] = out_max
    return out   

def quantize_midrise (x, bits, range_max=1, range_min=-1):
    '''
    quantizes the input samples x with a certain bit depth 
    using the midrise quantizing line 

    Parameters:
    -----------
    x : np.ndarray
        vector to quantize
    bits : int
        quantizing depth in bits

    Returns:
    out : np.ndarray
        quantized input vector
    '''
    stepsize = (range_max - range_min)/(2**bits)
    index = np.floor(x / stepsize)
    out = (index + 0.5) * stepsize
    out_max = range_max - stepsize/2
    out[np.where(out > out_max)] = out_max
    return out  
       
def quantize_lpc(
        input_file, 
        output_file, 
        winLengthinms=20, 
        shiftLengthinms=10, 
        lp_order=20, 
        emph_coef=0.97, 
        a_quantize=True,
        a_bits=3, 
        a_uniform=True,  
        r_quantize=True,
        r_bits=2,
        r_uniform=True, 
        exp=1, 
        e_quantize=False,
        e_bits=2,   
        e_uniform=False,  
        unique_poles=False, 
        ):
    '''
    Quantize LPC poles in angle and radius

    Parameters:
    -----------
    input_file : string
        Path to input audio signal
    output_file : string
        Path to quantized output audio signal
    winLengthinms : float
        Length in ms of the sliding window used to compute the STFT
    shiftLengthinms : float
        Length in ms of the shift of the sliding window used to compute
        the STFT.
    lp_order : int
        Order of the linear filter during LPC computation
    emph_coef : positive number
        Pre-emphasis coefficient, typically between 0 and 1
    a_bits : int
        Quantizing depth in bits for angles
    r_bits : int
        Quantizing depth in bits for radius
    e_bits : int
        Quantizing depth in bits for excitation signal
    r_uniform : boolean
        If True (default) quantizes the radius uniformly, 
        if False applies exponential function with exp before quantization
    exp : int
        Exponent for expanding the input before quantization
    '''
    sig, rate = librosa.load(input_file, sr=None) 
    eps = np.finfo(np.float32).eps
    sig = sig + eps

    # pre-emphasize signal
    sig = librosa.effects.preemphasis(sig, coef=emph_coef)

    # simulation parameters
    winlen = np.floor(winLengthinms * 0.001 * rate).astype(int)
    shift = np.floor(shiftLengthinms * 0.001 * rate).astype(int)
    length_sig = len(sig)

    # analysis and synth window which satisfies the constraint
    wPR = np.hanning(winlen)
    K = np.sum(wPR) / shift
    win = np.sqrt(wPR / K)
    Nframes = 1 + np.floor((length_sig - winlen) / shift).astype(int)  # nr of complete frames

    if not a_uniform:
        min_angle = 400/(rate/2) * np.pi
        max_angle = 7000/(rate/2) * np.pi
        a_bins = np.logspace(np.log10(min_angle), np.log10(max_angle), num=int(2**a_bits))
        a_step = np.diff(np.append(0, a_bins))
        a_bins_mid = a_bins - a_step/2

    if not e_uniform:
        min_ampl = 0.01
        max_ampl = 1
        e_bins = np.logspace(np.log10(min_ampl), np.log10(max_ampl), num=int(2**e_bits))
        e_step = np.diff(np.append(0, e_bins))
        e_bins_mid = e_bins - e_step/2

    # carry out the overlap - add FFT processing
    sig_quantized = np.zeros([length_sig])  # allocate output vector
    for m in np.arange(Nframes):
        # indices of the mth frame
        index = np.arange(m * shift, np.minimum(m * shift + winlen, length_sig))
        # windowed mth frame (other than rectangular window)
        frame = sig[index] * win
        # get lpc coefficients
        a_lpc = librosa.core.lpc(frame + eps, order=lp_order)
        # get residual excitation for reconstruction
        res = scipy.signal.lfilter(a_lpc, np.array(1), frame)
        # get poles
        poles = scipy.signal.tf2zpk(np.array([1]), a_lpc)[1]
        # get angles
        angles = np.angle(poles)

        # quantize residual signal
        if e_quantize:
            if e_uniform:
                res_quan = quantize_midrise(res, e_bits)
            else:
                e_sign = np.sign(res)
                e_idx = np.digitize(np.abs(res), e_bins_mid)
                e_idx[e_idx == 0] = 1
                res_quan = e_sign * e_bins[e_idx-1]
        else:
            res_quan = res

        # quantize angles 
        if a_quantize:
            a_sign = np.sign(angles)
            idx_zero = np.abs(angles) == 0
            idx_pi = np.abs(angles) == np.pi
            if a_uniform:
                angles_norm = 2 * np.abs(angles) / np.pi - 1
                new_angles = quantize_midtread(angles_norm, a_bits, limit=False)
                new_angles = np.pi * (new_angles + 1) / 2
            else:
                a_idx = np.digitize(np.abs(angles), a_bins_mid)
                a_idx[a_idx == 0] = 1
                new_angles = a_bins[a_idx-1]
            new_angles[idx_zero] = 0
            new_angles[idx_pi] = np.pi
            new_angles *= a_sign
        else:
            new_angles = angles

        # quantize radii
        if r_quantize:
            radii = np.abs(poles)
            if not r_uniform:
                radii = radii ** exp
            radii_norm = 2 * radii - 1
            new_radii = quantize_midtread(radii_norm, r_bits)
            new_radii = (new_radii + 1) / 2
            if not r_uniform:
                new_radii =  new_radii ** (1/exp)
        else:
            new_radii = np.abs(poles)

        # compute new poles with the new magnitude and new angles
        new_poles = np.zeros_like(poles)
        for k in np.arange(np.size(new_poles)):
            new_poles[k] = new_radii[k] * np.exp(1j * new_angles[k])
        
        # optional: unique poles
        if unique_poles:
            _, idx = np.unique(new_poles, return_index=True)
            new_poles = new_poles[np.sort(idx)]

        # recover new, modified lpc coefficients
        a_lpc_new = np.real(np.poly(new_poles))
        # reconstruct frames with new lpc coefficient
        frame_rec = scipy.signal.lfilter(np.array([1]), a_lpc_new, res_quan)
        frame_rec = frame_rec * win

        # overlap add
        outindex = np.arange(m * shift, m * shift + winlen)
        sig_quantized[outindex] = sig_quantized[outindex] + frame_rec

    # de-emphasize signal
    sig_quantized = librosa.effects.deemphasis(sig_quantized, coef=emph_coef)
    soundfile.write(output_file, np.float32(sig_quantized), rate) 
    return []
