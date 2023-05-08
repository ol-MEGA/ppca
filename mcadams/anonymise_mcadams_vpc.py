#!/usr/bin/env python3.0
# -*- coding: utf-8 -*-
"""
Anonymise VPC meeting mix data
saves infos: utteranceID, speakerID, duration in samples, offset in samples, Scaling factor, McAdams Coeff., ConversationID

@author: Jose Patino, Massimiliano Todisco, Pramod Bachhav, Nicholas Evans
Audio Security and Privacy Group, EURECOM
Jule Pohlhausen 2023
"""
import os
import numpy as np
import argparse
from tqdm import tqdm
from lazy_dataset.database import JsonDatabase
from anonymise_dir_mcadams import anonym

if __name__ == "__main__":
    #Parse args    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=str)
    parser.add_argument('--json_path',type=str)
    parser.add_argument('--anon_suffix',type=str,default='_mcadams')
    parser.add_argument('--n_coeffs',type=int,default=20)
    parser.add_argument('--mc_coeff',type=float,default=0.8)
    parser.add_argument('--mc_coeff_min', type=float, default=0.5)
    parser.add_argument('--mc_coeff_max', type=float, default=0.9)
    parser.add_argument('--mc_rand', action=argparse.BooleanOptionalAction)
    parser.add_argument('--winLengthinms',type=int,default=20)
    parser.add_argument('--shiftLengthinms',type=int,default=10)
    config = parser.parse_args()
            
    np.random.seed(1234)

    # get VPC meeting mix data
    db = JsonDatabase(config.json_path)
    dataset_names = db.dataset_names

    # init summary output file    
    savefile = 'vpc_mix_anon_mcadams.txt'
    if config.mc_rand:
        savefile = savefile.replace('mcadams', 'mcadams_rand')
    f = open(os.path.join(config.data_dir, savefile), 'w')
    f.write("#utteranceID, speakerID, duration, offset, scalingFactor, McAdamsCoeff, conversationID\n")

    for sub_set in dataset_names:
        dset = db.get_dataset(sub_set)

        output_dir = os.path.join(config.data_dir, sub_set + config.anon_suffix)
        os.makedirs(output_dir, exist_ok=True)

        for idx in tqdm(range(len(dset))): 
            ex = dset[idx]
            conversationID = ex['example_id']
            filename = os.path.join(config.data_dir, conversationID + '.wav')

            # change outpt dirname and create folders
            output_file = os.path.join(output_dir, conversationID + '.wav')

            if config.mc_rand:
                config.mc_coeff = np.random.uniform(config.mc_coeff_min, config.mc_coeff_max)
            
            # saves infos: utteranceID, speakerID, duration, offset, Scaling factor, McAdams Coeff., ConversationID
            utteranceIDs = ex['source_id']
            speakerIDs = ex['speaker_id']
            scale_factors = ex['log_weights']
            num_samples = ex['num_samples']['original_source']
            offset = ex['offset']['original_source']
            for u, s, n, o, w in zip(utteranceIDs, speakerIDs, num_samples, offset, scale_factors):
                line =  (
                    u + ", " + s + ", " + str(n) + ", " + str(o) + ", " + str(w) 
                    + ", " +
                    str(config.mc_coeff)
                    + ", " +
                    conversationID
                    + '\n'
                )
                f.write(line)
            
            #anonym(filename, output_file, winLengthinms=config.winLengthinms, shiftLengthinms=config.shiftLengthinms, lp_order=config.n_coeffs, mcadams=config.mc_coeff)
       
    f.close()