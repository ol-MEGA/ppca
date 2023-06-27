#!/usr/bin/python3
"""Recipe for training a speaker verification system based on PLDA using any dataset.
The system employs a pre-trained model followed by a PLDA transformation.

To run this recipe, run the following command:
    >  python 2_speaker_verification_plda.py /home/francesco/Documents/Python_Project/SpeechBrain/speechbrain/recipes/VoxCeleb/SpeakerRec/hparams/verification_plda_xvector.yaml

Authors
    * Nauman Dawalatabad 2020
    * Mirco Ravanelli 2020
    * Modified by Francesco Nespoli October 2022
"""

import os
import sys
import torch
import torchaudio
import random
import logging
import speechbrain as sb
import numpy
import pickle
from tqdm.contrib import tqdm
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.metric_stats import EER, minDCF
from speechbrain.processing.PLDA_LDA import StatObject_SB
from speechbrain.processing.PLDA_LDA import Ndx
from speechbrain.processing.PLDA_LDA import fast_PLDA_scoring
from speechbrain.utils.data_utils import download_file
from speechbrain.utils.distributed import run_on_main


################################################################################## Compute embeddings from the waveforms
def compute_embeddings(wavs, wav_lens):
    """Compute speaker embeddings.

    Arguments
    ---------
    wavs : Torch.Tensor
        Tensor containing the speech waveform (batch, time).
        Make sure the sample rate is fs=16000 Hz.
    wav_lens: Torch.Tensor
        Tensor containing the relative length for each sentence
        in the length (e.g., [0.8 0.6 1.0])
    """
    wavs = wavs.to(params["device"])
    wav_lens = wav_lens.to(params["device"])
    with torch.no_grad():
        feats = params["compute_features"](wavs)
        feats = params["mean_var_norm"](feats, wav_lens)
        embeddings = params["embedding_model"](feats, wav_lens)
    return embeddings.squeeze(1)

def emb_computation_loop(split, set_loader, stat_file):
    """Computes the embeddings and saves the in a stat file"""
    # Extract embeddings (skip if already done)
    if not os.path.isfile(stat_file):
        embeddings = numpy.empty(
            shape=[0, params["emb_dim"]], dtype=numpy.float64
        )
        modelset = []
        segset = []
        with tqdm(set_loader, dynamic_ncols=True) as t:
            for batch in t:
                ids = batch.id
                wavs, lens = batch.sig
                mod = [x for x in ids]
                seg = [x for x in ids]
                modelset = modelset + mod
                segset = segset + seg

                # Enrollment and test embeddings
                embs = compute_embeddings(wavs, lens)
                xv = embs.squeeze().cpu().numpy()
                embeddings = numpy.concatenate((embeddings, xv), axis=0)

        modelset = numpy.array(modelset, dtype="|O")
        segset = numpy.array(segset, dtype="|O")

        # Intialize variables for start, stop and stat0
        s = numpy.array([None] * embeddings.shape[0])
        b = numpy.array([[1.0]] * embeddings.shape[0])

        # Stat object (used to collect embeddings)
        stat_obj = StatObject_SB(
            modelset=modelset,
            segset=segset,
            start=s,
            stop=s,
            stat0=b,
            stat1=embeddings,
        )
        logger.info(f"Saving stat obj for {split}")
        stat_obj.save_stat_object(stat_file)

    else:
        logger.info(f"Skipping embedding Extraction for {split}")
        logger.info(f"Loading previously saved stat_object for {split}")

        with open(stat_file, "rb") as input:
            stat_obj = pickle.load(input)

    # Average x-vectors from the same speaker for enrollmet
    if split == "enrol":
        embeddings = numpy.empty(shape=[0, params["emb_dim"]], dtype=numpy.float64)
        spks = numpy.unique(numpy.array([i.split(params["separator"])[0] for i in stat_obj.modelset]))
        for spk in spks:
            xv_spk = numpy.zeros(shape=(1,params["emb_dim"]), dtype=float)
            N = 0
            for idx,utt in enumerate(stat_obj.modelset):
                if spk == utt.split(params["separator"])[0]:
                    xv_spk = xv_spk + stat_obj.stat1[idx]
                    N +=1
            embeddings = numpy.concatenate((embeddings, xv_spk/N), axis=0)

        modelset = numpy.array(spks, dtype="|O")
        segset = numpy.array(spks, dtype="|O")

        # Intialize variables for start, stop and stat0
        s = numpy.array([None] * embeddings.shape[0])
        b = numpy.array([[1.0]] * embeddings.shape[0])

        # Stat object (used to collect embeddings)
        stat_obj = StatObject_SB(
            modelset=modelset,
            segset=segset,
            start=s,
            stop=s,
            stat0=b,
            stat1=embeddings,
        )
        logger.info(f"Saving stat obj for {split}")
        stat_obj.save_stat_object(stat_file)

    return stat_obj

################################################################################################# Verification Functions
def verification_performance_mean_spkemb(scores_plda):
    """Computes the Equal Error Rate give the PLDA scores"""

    # Create ids, labels, and scoring list for EER evaluation
    ids = []
    labels = []
    positive_scores = []
    negative_scores = []
    for line in open(veri_file_path):

        lab = line.split(" ")[0].rstrip().split(".")[0].strip() # speaker from the enrollmet data
        enrol_id = line.split(" ")[1].rstrip().split(".")[0].strip() # test_id
        test_id = line.split(" ")[2].rstrip().split(".")[0].strip() # target/non_target
#        print(lab)
#        print(enrol_id)
#        print(test_id)
#        print(scores_plda.segset)

        # Find the column ID for the test utterance
        trial_index = int(numpy.where(scores_plda.segset == enrol_id)[0][0])
#        print(trial_index)

        # Find row IDs for all the utterances of the enrollment speaker
        enroll_idxs = []
#        print(scores_plda.modelset)
        
        for idx,elem in enumerate(scores_plda.modelset):
            if elem.split(params["separator"])[0] == lab:
                    enroll_idxs.append(idx)
#        print(elem)
#        print(elem.split(params["separator"]))
#        print(enroll_idxs)

        # # Assuming enrol_id and test_id are unique
        # i = int(numpy.where(scores_plda.modelset == enrol_id)[0][0])
        # j = int(numpy.where(scores_plda.segset == test_id)[0][0])
        s = []
        for j in enroll_idxs:
            s.append(float(scores_plda.scoremat[j,trial_index]))
        labels.append(test_id)
        ids.append(enrol_id + "<>" + lab)
        if test_id == "target":
            for S in s:
                positive_scores.append(S)
        else:
            for S in s:
                negative_scores.append(S)

    # Clean variable
    del scores_plda

    # Final EER computation
#    print(len(positive_scores), len(negative_scores)) # both empty!!!
    eer, th = EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
    min_dcf, th = minDCF(torch.tensor(positive_scores), torch.tensor(negative_scores))

    return eer , min_dcf

############################################################################################################## Data Prep
def get_utt_ids_for_test(ids, data_dict):
    mod = [data_dict[x]["wav1"]["data"] for x in ids]
    seg = [data_dict[x]["wav2"]["data"] for x in ids]

    return mod, seg


def dataio_prep(params): # TODO === Need to change the data IO function ===
    "Creates the dataloaders and their data processing pipelines."

    data_folder_tr = params["data_folder_tr"]
    data_folder_test = params["data_folder_test"]

    # Train data (used for normalization)
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=params["train_data"], replacements={"data_root": data_folder_tr},
    )
    # train_data = train_data.filtered_sorted(
    #     sort_key="duration", select_n=params["n_train_snts"]
    # )

    # Enrol data
    enrol_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=params["enrol_data"], replacements={"data_root": data_folder_test},
    )
    # enrol_data = enrol_data.filtered_sorted(sort_key="duration")

    # Test data
    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=params["test_data"], replacements={"data_root": data_folder_test},
    )
    # test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, enrol_data, test_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")#, "start", "stop")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav): #, start, stop):
        #start = int(start)
        #stop = int(stop)
        #num_frames = stop - start
        sig, fs = torchaudio.load(wav) #, num_frames=num_frames, frame_offset=start)
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "spk_id"])

    # 4 Create dataloaders
    train_dataloader = sb.dataio.dataloader.make_dataloader(
        train_data, **params["train_dataloader_opts"]
    )
    enrol_dataloader = sb.dataio.dataloader.make_dataloader(
        enrol_data, **params["enrol_dataloader_opts"]
    )
    test_dataloader = sb.dataio.dataloader.make_dataloader(
        test_data, **params["test_dataloader_opts"]
    )

    return train_dataloader, enrol_dataloader, test_dataloader






##################################################### MAIN ############################################################

if __name__ == "__main__":

    # Logger setup
    logger = logging.getLogger(__name__)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(current_dir))

    # Load hyperparameters file with command-line overrides
    params_file, run_opts, overrides = sb.core.parse_arguments(sys.argv[1:])
    with open(params_file) as fin:
        params = load_hyperpyyaml(fin, overrides)

    # This file has to be created before: it basically pairs enrolment and trial utterances for PLDA scoring
    veri_file_path = params["verification_file"]

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=params["output_folder"],
        hyperparams_to_save=params_file,
        overrides=overrides,
    )

    # Prepare data
    logger.info("Data preparation")

    # Here we create the datasets
    train_dataloader, enrol_dataloader, test_dataloader = dataio_prep(params)

    # Initialize PLDA vars
    modelset, segset = [], []
    embeddings = numpy.empty(shape=[0, params["emb_dim"]], dtype=numpy.float64)

    # Embedding file for train data
    xv_file = os.path.join(params["save_folder"], "my_training.pkl")
    if not(os.path.isdir(params["save_folder"])):
        os.mkdir(params["save_folder"])

    # We download the pretrained LM from HuggingFace (or elsewhere depending on
    # the path given in the YAML file). The tokenizer is loaded at the same time.
    run_on_main(params["pretrainer"].collect_files)
    params["pretrainer"].load_collected()

    params["embedding_model"].eval()
    params["embedding_model"].to(params["device"])

    # Computing training embeddings (skip it of if already extracted)
    if not os.path.exists(xv_file):
        logger.info("Extracting embeddings from Training set..")
        with tqdm(train_dataloader, dynamic_ncols=True) as t:
            for batch in t:
                snt_id = batch.id
                wav, lens = batch.sig
                spk_ids = batch.spk_id

                # Flattening speaker ids
                modelset = modelset + spk_ids

                # For segset
                segset = segset + snt_id

                # Compute embeddings
                emb = compute_embeddings(wav, lens)
                xv = emb.squeeze(1).cpu().numpy()
                embeddings = numpy.concatenate((embeddings, xv), axis=0)

        # Speaker IDs and utterance IDs
        modelset = numpy.array(modelset, dtype="|O")
        segset = numpy.array(segset, dtype="|O")

        # Intialize variables for start, stop and stat0
        s = numpy.array([None] * embeddings.shape[0])
        b = numpy.array([[1.0]] * embeddings.shape[0])

        embeddings_stat = StatObject_SB(
            modelset=modelset,
            segset=segset,
            start=s,
            stop=s,
            stat0=b,
            stat1=embeddings,
        )

        del embeddings

        # Save TRAINING embeddings in StatObject_SB object
        embeddings_stat.save_stat_object(xv_file)

    else:
        # Load the saved stat object for train embedding
        logger.info("Skipping embedding Extraction for training set")
        logger.info(
            "Loading previously saved stat_object for train embeddings.."
        )
        with open(xv_file, "rb") as input:
            embeddings_stat = pickle.load(input)

    # Training Gaussian PLDA model
    logger.info("Training PLDA model")
    params["compute_plda"].plda(embeddings_stat)
    logger.info("PLDA training completed")

    # Set paths for enrol/test embeddings
    enrol_stat_file = os.path.join(params["save_folder"], "stat_enrol.pkl")
    test_stat_file = os.path.join(params["save_folder"], "stat_test.pkl")
    ndx_file = os.path.join(params["save_folder"], "ndx.pkl")

    # Compute enrol and Test embeddings
    enrol_obj = emb_computation_loop("enrol", enrol_dataloader, enrol_stat_file)
    test_obj = emb_computation_loop("test", test_dataloader, test_stat_file)


    # Prepare Ndx Object
    if not os.path.isfile(ndx_file):
        models = enrol_obj.modelset
        testsegs = test_obj.modelset

        logger.info("Preparing Ndx")
        ndx_obj = Ndx(models=models, testsegs=testsegs)
        logger.info("Saving ndx obj...")
        ndx_obj.save_ndx_object(ndx_file)
    else:
        logger.info("Skipping Ndx preparation")
        logger.info("Loading Ndx from disk")
        with open(ndx_file, "rb") as input:
            ndx_obj = pickle.load(input)

    # PLDA scoring
    logger.info("PLDA scoring...")
    scores_plda = fast_PLDA_scoring(
        enrol_obj,
        test_obj,
        ndx_obj,
        params["compute_plda"].mean,
        params["compute_plda"].F,
        params["compute_plda"].Sigma,
    )

    logger.info("Computing EER... ")

    # Cleaning variable
    del enrol_dataloader
    del test_dataloader
    del enrol_obj
    del test_obj
    del embeddings_stat

    # Final EER computation
    # eer, min_dcf = verification_performance(scores_plda, single_random_comparison=
    eer, min_dcf = verification_performance_mean_spkemb(scores_plda)
    logger.info("EER(%%)=%f", eer * 100)
    logger.info("min_dcf=%f", min_dcf * 100)
