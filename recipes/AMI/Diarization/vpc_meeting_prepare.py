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
    data_folder,
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
    data_folder : str
        Path to the folder where the mixed VPC meetings are stored.
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
    >>> from recipes.AMI.vpc_prepare import prepare_vpc
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
        "json_path": json_path,
        "data_folder": data_folder,
        "save_folder": save_folder,
        "ref_rttm_dir": ref_rttm_dir,
        "meta_data_dir": meta_data_dir,
        "max_subseg_dur": max_subseg_dur,
        "overlap": overlap,
        "meta_files": meta_files,
    }

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(meta_data_dir):
        os.makedirs(meta_data_dir)

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

        # Create meta_files for subset
        data_dir = os.path.join(data_folder, subset)
        meta_filename_prefix = "vpc_" + subset
        prepare_metadata(
            rttm_file,
            meta_data_dir,
            data_dir,
            meta_filename_prefix,
            max_subseg_dur,
            overlap,
        )
    
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
    Takes rounding inaccuricies into account

    Arguments
    ---------
    end1 : float
        End time of the first segment.
    start2 : float
        Start time of the second segment.
    """

    if start2 > end1:
        return False
    elif (end1 - start2) <= 0.0002:
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
            # update previous segment end, i.e. duration
            merged_segs[-1][4] = str(round((s - strt), 4))

            # add overlap region between 2nd start and 1st end
            row_ov = row
            row_ov[3] = str(round(s, 4))
            row_ov[4] = str(round(end - s, 4))
            row_ov[7] = "overlap"  # previous_row[7] + '-'+ row[7]
            merged_segs.append(row_ov)

            # update current segment start
            row[3] = str(round(end, 4))

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

def prepare_metadata(
    rttm_file, save_dir, data_dir, filename, max_subseg_dur, overlap
):
    # Read RTTM, get unique meeting_IDs (from RTTM headers)
    # For each MeetingID. select that meetID -> merge -> subsegment -> json -> append

    # Read RTTM
    RTTM = []
    with open(rttm_file, "r") as f:
        for line in f:
            entry = line[:-1]
            RTTM.append(entry)

    spkr_info = filter(lambda x: x.startswith("SPKR-INFO"), RTTM)
    rec_ids = list(set([row.split(" ")[1] for row in spkr_info]))
    rec_ids.sort()  # sorting just to make JSON look in proper sequence

    # For each recording merge segments and then perform subsegmentation
    MERGED_SEGMENTS = []
    SUBSEGMENTS = []
    for rec_id in rec_ids:
        segs_iter = filter(
            lambda x: x.startswith("SPEAKER " + str(rec_id)), RTTM
        )
        gt_rttm_segs = [row.split(" ") for row in segs_iter]

        # Merge, subsegment and then convert to json format.
        merged_segs = merge_rttm_intervals(
            gt_rttm_segs
        )  # We lose speaker_ID after merging
        MERGED_SEGMENTS = MERGED_SEGMENTS + merged_segs

        # Divide segments into smaller sub-segments
        subsegs = get_subsegments(merged_segs, max_subseg_dur, overlap)
        SUBSEGMENTS = SUBSEGMENTS + subsegs

    # Write segment AND sub-segments (in RTTM format)
    segs_file = os.path.join(save_dir, filename + ".segments.rttm")
    subsegment_file = os.path.join(save_dir, filename + ".subsegments.rttm")

    with open(segs_file, "w") as f:
        for row in MERGED_SEGMENTS:
            line_str = " ".join(row)
            f.write("%s\n" % line_str)

    with open(subsegment_file, "w") as f:
        for row in SUBSEGMENTS:
            line_str = " ".join(row)
            f.write("%s\n" % line_str)

    # Create JSON from subsegments
    json_dict = {}
    for row in SUBSEGMENTS:
        rec_id = row[1]
        strt = str(round(float(row[3]), 4))
        end = str(round((float(row[3]) + float(row[4])), 4))
        subsegment_ID = rec_id + "_" + strt + "_" + end
        dur = row[4]
        start_sample = int(float(strt) * SAMPLERATE)
        end_sample = int(float(end) * SAMPLERATE)

        # Single mic audio
        wav_file_path = os.path.join(data_dir, rec_id + ".wav")

        # Note: key "file" without 's' is used for single-mic
        json_dict[subsegment_ID] = {
            "wav": {
                "file": wav_file_path,
                "duration": float(dur),
                "start": int(start_sample),
                "stop": int(end_sample),
            },
        }

    out_json_file = os.path.join(save_dir, filename + ".subsegs.json")
    with open(out_json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    msg = "%s JSON prepared" % (out_json_file)
    logger.debug(msg)

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
