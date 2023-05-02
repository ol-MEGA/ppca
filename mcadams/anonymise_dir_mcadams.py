"""
Data preparation: VPC meeting mix

Prepares metadata files (JSON) from VAD annotations in "vpc_mix.json" using RTTM format.
"""

import os
import logging
import xml.etree.ElementTree as et
import glob
import json
from lazy_dataset.database import JsonDatabase

from speechbrain.dataio.dataio import (
    load_pkl,
    save_pkl,
)

logger = logging.getLogger(__name__)
SAMPLERATE = 16000


def prepare_vpc(
    json_path,
    save_folder,
    ref_rttm_dir,
    meta_data_dir,
    max_subseg_dur=3.0,
    overlap=1.5,
):
    """
    Prepares reference RTTM and JSON files for the AMI dataset.

    Arguments
    ---------
    json_path : str
        Path where the VPC meeting mix json is stored.
    save_folder : str
        The save directory in results.
    ref_rttm_dir : str
        Directory to store reference RTTM files.
    meta_data_dir : str
        Directory to store the meta data (json) files.
    max_subseg_dur : float
        Duration in seconds of a subsegments to be prepared from larger segments.
    overlap : float
        Overlap duration in seconds between adjacent subsegments

    Example
    -------
    >>> from recipes.AMI.ami_prepare import prepare_ami
    >>> data_folder = '/network/datasets/ami/amicorpus/'
    >>> manual_annot_folder = '/home/mila/d/dawalatn/nauman/ami_public_manual/'
    >>> save_folder = 'results/save/'
    >>> split_type = 'full_corpus_asr'
    >>> mic_type = 'Lapel'
    >>> prepare_ami(data_folder, manual_annot_folder, save_folder, split_type, mic_type)
    """

    # Meta files
    meta_files = [
        os.path.join(meta_data_dir, "libri_dev.subsegs.json"),
        os.path.join(meta_data_dir, "libri_test.subsegs.json"),
        os.path.join(meta_data_dir, "vctk_dev.subsegs.json"),
        os.path.join(meta_data_dir, "vctk_test.subsegs.json"),
    ]

    # Create configuration for easily skipping data_preparation stage
    conf = {
        "save_folder": save_folder,
        "ref_rttm_dir": ref_rttm_dir,
        "meta_data_dir": meta_data_dir,
        "max_subseg_dur": max_subseg_dur,
        "overlap": overlap,
        "meta_files": meta_files,
    }

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Setting output option files.
    opt_file = "opt_vpc_prepare.pkl"

    # Check if this phase is already done (if so, skip it)
    if skip(save_folder, conf, meta_files, opt_file):
        logger.info(
            "Skipping data preparation, as it was completed in previous run."
        )
        return

    msg = "\tCreating meta-data file for the VPC meeting mix Dataset.."
    logger.debug(msg)

    # Prepare RTTM from XML(manual annot) and store are groundtruth
    # Create ref_RTTM directory
    if not os.path.exists(ref_rttm_dir):
        os.makedirs(ref_rttm_dir)

    # get VPC meeting mix data
    db = JsonDatabase(json_path)
    dataset_names = db.dataset_names

    # Create reference RTTM files
    for subset in dataset_names:
        # get subset data
        dset = db.get_dataset(subset)

        rttm_file = ref_rttm_dir + "/fullref_vpc_" + subset + ".rttm"
        prepare_RTTM(dset, rttm_file)
    
    save_opt_file = os.path.join(save_folder, opt_file)
    save_pkl(conf, save_opt_file)


def get_RTTM_per_rec(ex):
    """Prepares rttm for each recording
    """
    rec_id = ex['example_id']
    spkrs_list = ex['speaker_id']
    num_samples = ex['num_samples']['original_source']
    offset = ex['offset']['original_source']

    rttm = []

    # Prepare header
    for spkr_id in set(spkrs_list):
        # e.g. SPKR-INFO ES2008c 0 <NA> <NA> <NA> unknown ES2008c.A_PM <NA> <NA>
        line = (
            "SPKR-INFO "
            + rec_id
            + " 0 <NA> <NA> <NA> unknown "
            + spkr_id
            + " <NA> <NA>"
        )
        rttm.append(line)

    # Append lines
    for spkr_id, s, o in zip(spkrs_list, num_samples, offset):
        # e.g. SPEAKER ES2008c 0 37.880 0.590 <NA> <NA> ES2008c.A_PM <NA> <NA>
        start = float(o)/SAMPLERATE
        dur = float(s)/SAMPLERATE

        line = (
            "SPEAKER "
            + rec_id
            + " 0 "
            + str(round(start, 4))
            + " "
            + str(round(dur, 4))
            + " <NA> <NA> "
            + spkr_id
            + " <NA> <NA>"
        )
        rttm.append(line)

    return rttm


def prepare_RTTM(dset, out_rttm_file):

    RTTM = []  # Stores all RTTMs clubbed together for a given dataset split

    for idx in range(len(dset)):
        ex = dset[idx]
        rttm_per_rec = get_RTTM_per_rec(ex)
        RTTM = RTTM + rttm_per_rec

    # Write one RTTM as groundtruth
    with open(out_rttm_file, "w") as f:
        for item in RTTM:
            f.write("%s\n" % item)


def is_overlapped(end1, start2):
    """Returns True if the two segments overlap

    Arguments
    ---------
    end1 : float
        End time of the first segment.
    start2 : float
        Start time of the second segment.
    """

    if start2 > end1:
        return False
    else:
        return True


def merge_rttm_intervals(rttm_segs):
    """Merges adjacent segments in rttm if they overlap.
    """
    # For one recording
    # rec_id = rttm_segs[0][1]
    rttm_segs.sort(key=lambda x: float(x[3]))

    # first_seg = rttm_segs[0] # first interval.. as it is
    merged_segs = [rttm_segs[0]]
    strt = float(rttm_segs[0][3])
    end = float(rttm_segs[0][3]) + float(rttm_segs[0][4])

    for row in rttm_segs[1:]:
        s = float(row[3])
        e = float(row[3]) + float(row[4])

        if is_overlapped(end, s):
            # Update only end. The strt will be same as in last segment
            # Just update last row in the merged_segs
            end = max(end, e)
            merged_segs[-1][3] = str(round(strt, 4))
            merged_segs[-1][4] = str(round((end - strt), 4))
            merged_segs[-1][7] = "overlap"  # previous_row[7] + '-'+ row[7]
        else:
            # Add a new disjoint segment
            strt = s
            end = e
            merged_segs.append(row)  # this will have 1 spkr ID

    return merged_segs


def get_subsegments(merged_segs, max_subseg_dur=3.0, overlap=1.5):
    """Divides bigger segments into smaller sub-segments
    """

    shift = max_subseg_dur - overlap
    subsegments = []

    # These rows are in RTTM format
    for row in merged_segs:
        seg_dur = float(row[4])
        rec_id = row[1]

        if seg_dur > max_subseg_dur:
            num_subsegs = int(seg_dur / shift)
            # Taking 0.01 sec as small step
            seg_start = float(row[3])
            seg_end = seg_start + seg_dur

            # Now divide this segment (new_row) in smaller subsegments
            for i in range(num_subsegs):
                subseg_start = seg_start + i * shift
                subseg_end = min(subseg_start + max_subseg_dur - 0.01, seg_end)
                subseg_dur = subseg_end - subseg_start

                new_row = [
                    "SPEAKER",
                    rec_id,
                    "0",
                    str(round(float(subseg_start), 4)),
                    str(round(float(subseg_dur), 4)),
                    "<NA>",
                    "<NA>",
                    row[7],
                    "<NA>",
                    "<NA>",
                ]

                subsegments.append(new_row)

                # Break if exceeding the boundary
                if subseg_end >= seg_end:
                    break
        else:
            subsegments.append(row)

    return subsegments


def skip(save_folder, conf, meta_files, opt_file):
    """
    Detects if the VPC data_preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    # Checking if meta (json) files are available
    skip = True
    for file_path in meta_files:
        if not os.path.isfile(file_path):
            skip = False

    # Checking saved options
    save_opt_file = os.path.join(save_folder, opt_file)
    if skip is True:
        if os.path.isfile(save_opt_file):
            opts_old = load_pkl(save_opt_file)
            if opts_old == conf:
                skip = True
            else:
                skip = False
        else:
            skip = False

    return skip
