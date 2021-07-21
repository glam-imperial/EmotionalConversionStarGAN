"""
Author - Max Elliott

Script completes three task:
    1) refile the IEMOCAP dataset
    2) Generates the WORLD features needed for training EmotionalConversionStarGAN
    3) Generates f0 look up dictionaries needs for producing converted audio files
"""

import os
import numpy as np
import pickle
from shutil import copyfile
import argparse
from utils.data_preprocessing_utils import get_wav_and_labels
from utils.preprocess_world import world_features, cal_mcep, get_f0_stats


def copy_files(iemocap_dir, output_dir):

    """
    Make initial directory structure needed for preprocessing. Takes IEMOCAP
    and puts all audio files in one folder, and all annotations in another.
    """
    audio_output_dir = os.path.join(output_dir, 'audio')
    annotations_output_dir = os.path.join(output_dir, 'annotations')

    if not os.path.exists(audio_output_dir):
        os.mkdir(audio_output_dir)
    if not os.path.exists(annotations_output_dir):
        os.mkdir(annotations_output_dir)

    for session in os.listdir(iemocap_dir):
        if not session.startswith("Session"):
            continue

        session_dir = os.path.join(iemocap_dir, session)

        annotations_dir = os.path.join(session_dir, "dialog", "EmoEvaluation")
        for filename in os.listdir(annotations_dir):
            if not filename.endswith(".txt"):
                continue

            src_file = os.path.join(annotations_dir, filename)
            dest_file = os.path.join(annotations_output_dir, filename)
            if not os.path.exists(dest_file):
                copyfile(src_file, dest_file)

        wav_dir = os.path.join(session_dir, "sentences", "wav")
        for foldername in os.listdir(wav_dir):
            if not foldername.startswith("Ses"):
                continue

            subsession_dir = os.path.join(wav_dir, foldername)
            for filename in os.listdir(subsession_dir):
                if not filename.endswith(".wav"):
                    continue

                src_file = os.path.join(subsession_dir, filename)
                dest_file = os.path.join(audio_output_dir, filename)
                if not os.path.exists(dest_file):
                    copyfile(src_file, dest_file)

        print(session + " completed.")


def generate_world_features(filenames, data_dir):
    """Code for creating and saving world features and sample labels"""

    world_dir = os.path.join(data_dir, 'world')
    f0_dir = os.path.join(data_dir, 'f0')
    labels_dir = os.path.join(data_dir, "labels")

    if not os.path.exists(world_dir):
        os.mkdir(world_dir)
    if not os.path.exists(f0_dir):
        os.mkdir(f0_dir)
    if not os.path.exists(labels_dir):
        os.mkdir(labels_dir)

    MIN_LENGTH = 0 # actual is 59
    MAX_LENGTH = 1719
    worlds_made = 0

    for i, f in enumerate(filenames):

        wav, labels = get_wav_and_labels(f, data_dir)
        wav = np.array(wav, dtype=np.float64)
        labels = np.array(labels)

        coded_sp_name = os.path.join(world_dir, f[:-4] + ".npy")
        label_name = os.path.join(labels_dir, f[:-4] + ".npy")
        f0_name = os.path.join(f0_dir, f[:-4] + ".npy")
        if os.path.exists(coded_sp_name) and os.path.exists(label_name) and os.path.exists(f0_name):
            worlds_made += 1
            continue

        # Ignores data sample if wrong emotion
        if labels[0] != -1:
            f0, ap, sp, coded_sp = cal_mcep(wav)

            # Ignores data sample sample is too long
            if coded_sp.shape[1] < MAX_LENGTH:

                np.save(os.path.join(world_dir, f[:-4] + ".npy"), coded_sp)
                np.save(os.path.join(labels_dir, f[:-4] + ".npy"), labels)
                np.save(os.path.join(f0_dir, f[:-4] + ".npy"), f0)

                worlds_made += 1

        if i % 10 == 0:
            print(i, " complete.")
            print(worlds_made, "worlds made.")


def generate_f0_stats(filenames, data_dir):
    """Generate absolute and relative f0 dictionary"""

    NUM_SPEAKERS = 10
    NUM_EMOTIONS = 4
    f0_dir = os.path.join(data_dir, 'f0')

    # CALCULATE ABSOLUTE F0 STATS

    emo_stats = {}
    for e in range(NUM_EMOTIONS):
        spk_dict = {}
        for s in range(NUM_SPEAKERS):
            f0s = []
            for f in filenames:
                wav, labels = get_wav_and_labels(f, data_dir)
                wav = np.array(wav, dtype=np.float64)
                labels = np.array(labels)
                if labels[0] == e and labels[1] == s:
                    f0_file = os.path.join(f0_dir, f[:-4] + ".npy")
                    if os.path.exists(f0_file):
                        f0 = np.load(f0_file)
                        f0s.append(f0)

            log_f0_mean, f0_std = get_f0_stats(f0s)
            spk_dict[s] = (log_f0_mean, f0_std)
            print(f"Done emotion {e}, speaker {s}.")
        emo_stats[e] = spk_dict

    with open('f0_dict.pkl', 'wb') as absolute_file:
        pickle.dump(emo_stats, absolute_file, pickle.HIGHEST_PROTOCOL)

    print(" ---- Absolute f0 stats completed ----")

    for tag, val in emo_stats.items():
        print(f'Emotion {tag} stats:')
        for tag2, val2 in val.items():
            print(f'{tag2} = {val2[0]}, {val2[1]}')

    # CALCULATE RELATIVE F0 STATS

    emo2emo_dict = {}

    for e1 in range(NUM_EMOTIONS):

        emo2emo_dict[e1] = {}

        for e2 in range(NUM_EMOTIONS):

            mean_list = []
            std_list = []

            for s in range(NUM_SPEAKERS):
                mean_diff = emo_stats[e2][s][0] - emo_stats[e1][s][0]
                std_diff = emo_stats[e2][s][1] - emo_stats[e1][s][1]
                mean_list.append(mean_diff)
                std_list.append(std_diff)

            mean_mean = np.mean(mean_list)
            std_mean = np.mean(std_list)
            emo2emo_dict[e1][e2] = (mean_mean, std_mean)

    print(" ---- Relative f0 stats completed ----")
    for tag, val in emo2emo_dict.items():
        print(f'Emotion {tag} stats:')
        for tag2, val2 in val.items():
            print(f'{tag2} = {val2[0]}, {val2[1]}')

    with open('f0_relative_dict.pkl', 'wb') as relative_file:
        pickle.dump(emo2emo_dict, relative_file, pickle.HIGHEST_PROTOCOL)


def run_preprocessing(args):

    print(f"--------------- Copying and restructuring IEMOCAP dataset in {args.data_dir} ---------------")
    copy_files(args.iemocap_dir, args.data_dir)

    data_dir = args.data_dir
    audio_dir = os.path.join(data_dir, 'audio')

    audio_filenames = [f for f in os.listdir(audio_dir) if '.wav' in f]

    print("----------------- Producing WORLD features data -----------------")
    generate_world_features(audio_filenames, data_dir)

    print("--------------- Producing relative f0 dictionaries ---------------")
    generate_f0_stats(audio_filenames, data_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Main preprocessing pipeline')
    parser.add_argument("--iemocap_dir", type=str, help="Directory of IEMOCAP dataset")
    parser.add_argument("--data_dir", type=str, default='./processed_data',
                        help="Directory to copy audio and annotation files to.")

    args = parser.parse_args()

    run_preprocessing(args)
