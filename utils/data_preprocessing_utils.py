"""
data_preprocessing2.py

Author - Max Elliott

Functions for pre-processing the IEMOCAP dataset. Can make mel-specs, WORLD
features, and labels for each audio clip.
"""

import torch

from utils import audio_utils

import numpy as np
import os
from librosa.util import find_files


def get_speaker_from_filename(filename):
    code = filename[4] + filename[-8]

    conversion = {'1F': 0, '1M': 1, '2F': 2, '2M': 3, '3F': 4, '3M': 5, '4F': 6, '4M': 7, '5F': 8, '5M': 9}

    label = conversion[code]

    return label


def get_emotion_from_label(category):

    if category == 'xxx' or category == 'dis' or category == 'fea' or category == 'oth':
        return -1
    if category == 'exc' or category == 'fru' or category == 'sur':
        return -1

    conversion = {'ang': 0, 'sad': 1, 'hap': 2, 'neu': 3}

    label = conversion[category]

    return label


def getOneHot(label, n_labels):

    onehot = np.zeros(n_labels)
    onehot[label] = 1

    return onehot


def cont2list(cont, binned=False):

    list = [0,0,0]
    list[0] = float(cont[1:6])
    list[1] = float(cont[9:14])
    list[2] = float(cont[17:22])

    #Option to make the values discrete: low(0), med(1) or high(2)
    if binned:
        for i, val in enumerate(list):
            if val <= 2:
                list[i] = 0
            elif val < 4:
                list[i] = 1
            else:
                list[i] = 2
        return list
    else:
        return list


def concatenate_labels(emo, speaker, dims, dims_dis):

    all_labels = torch.zeros(8)

    # for i, row in enumerate(all_labels):
    all_labels[0] = emo
    all_labels[1] = speaker
    all_labels[2] = dims[0]
    all_labels[3] = dims[1]
    all_labels[4] = dims[2]
    all_labels[5] = dims_dis[0]
    all_labels[6] = dims_dis[1]
    all_labels[7] = dims_dis[2]

    return all_labels


def get_wav_and_labels(filename, data_dir):

    wav_path = os.path.join(data_dir, "audio", filename)
    label_path = os.path.join(data_dir, "annotations", filename[:-9] + ".txt")

    with open(label_path, 'r') as label_file:

        category = ""
        dimensions = ""
        speaker = ""

        for row in label_file:
            if row[0] == '[':
                split = row.split("\t")
                if split[1] == filename[:-4]:
                    category = get_emotion_from_label(split[2])
                    dimensions = cont2list(split[3])
                    dimensions_dis = cont2list(split[3], binned = True)
                    speaker = get_speaker_from_filename(filename)

    audio = audio_utils.load_wav(wav_path)
    audio = np.array(audio, dtype = np.float32)
    labels = concatenate_labels(category, speaker, dimensions, dimensions_dis)

    return audio, labels


def get_samples_and_labels(filename, config):

    wav_path = config['data']['sample_set_dir'] + "/" + filename
    folder = filename[:-9]
    label_path = config['data']['dataset_dir'] + "/Annotations/" + folder + ".txt"

    with open(label_path, 'r') as label_file:

        category = ""
        dimensions = ""
        speaker = ""

        for row in label_file:
            if row[0] == '[':
                split = row.split("\t")
                if split[1] == filename[:-4]:
                    category = get_emotion_from_label(split[2])
                    dimensions = cont2list(split[3])
                    dimensions_dis = cont2list(split[3], binned = True)
                    speaker = get_speaker_from_filename(filename)

    audio = audio_utils.load_wav(wav_path)
    audio = np.array(audio, dtype = np.float32)
    labels = concatenate_labels(category, speaker, dimensions, dimensions_dis)

    return audio, labels


def get_filenames(data_dir):

    files = find_files(data_dir, ext = 'wav')
    filenames = []

    for f in files:
        f = os.path.basename(f)[:-4]
        filenames.append(f)

    return filenames


if __name__ == '__main__':

    min_length = 0 # actual is 59
    max_length = 688

    data_dir = '/Users/Max/MScProject/data'
    annotations_dir = os.path.join(data_dir, "audio")
    files = find_files(annotations_dir, ext = 'wav')

    filenames = []
    for f in files:
        f = os.path.basename(f)
        filenames.append(f)



    ############################################
    #      Code for making mels and labels     #
    ############################################
    i = 0
    found = 0
    lengths = []
    longest_lensgth = 0
    longest_name = ""
    for f in filenames:
        if i > 10000:
            print(f)
        wav, labels = get_wav_and_labels(f, data_dir)
        # mel = audio_utils.wav2melspectrogram(wav)
        labels = np.array(labels)
        if labels[0] in range(0,4) and f[0:3] == 'Ses':

            length = wav.shape[0]/16000.
            lengths.append(length)
            # np.save(data_dir + "/mels/" + f[:-4] + ".npy", mel)
            # np.save(data_dir + "/labels/" + f[:-4] + ".npy", labels)
            found += 1

            if length > longest_length:
                longest_length = length
                longest_name = f

        i += 1
        if i % 100 == 0:
            print(i, " complete.")
            print(found, "found.")

    print(found, "found.")
    print(f"longest + {longest_name}")

    lengths.sort()
    lengths = lengths[:int(len(lengths)*0.9)]
    print("Total seconds =", np.sum(lengths))

    # n, bins, patches = plt.hist(lengths, bins = 32)
    # plt.xlabel('Sequence length / seconds')
    # plt.xlim(0, 32)
    # plt.ylabel('Count')
    # plt.title(r'Histogram of sequence lengths for 4 emotional categories')
    # plt.show()

    ############################################
    #      Loop through mels for analysis      #
    ############################################
    # files = find_files(data_dir + "/mels", ext = 'npy')
    # lengths = []
    # for f in files:
    #
    #     mel = np.load(f)
    #     lengths.append(mel.shape[1])
    #     # print(mel.shape)
    #
    # n, bins, patches = plt.hist(lengths, bins = 22)
    # plt.xlabel('Sequence length')
    # plt.ylabel('Count')
    # plt.title(r'New histogram of sequence lengths for 4 emotional categories')
    # plt.show()

    ############################################
    #     Loop through labels for analysis     #
    ############################################
    # files = find_files(data_dir + "/labels", ext = 'npy')
    # category_counts = np.zeros((4))
    # speaker_counts = np.zeros((10))
    # for f in files:
    #
    #     labels = np.load(f)
    #     cat = int(labels[0])
    #     speaker = int(labels[1])
    #     category_counts[cat] += 1
    #     speaker_counts[speaker] += 1
    #
    # print(category_counts)
    # print(speaker_counts)
    # #### RESULTS ####
    # # [ 549.  890.  996. 1605.] 4040 total
    # # [416. 425. 353. 364. 448. 480. 342. 370. 473. 369.]
    # #### # # # # ####
    #
    # def make_autopct(values):
    #
    #     def my_autopct(pct):
    #         total = sum(values)
    #         val = int(round(pct*total/100.0))
    #         return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    #
    #     return my_autopct
    #
    # plt.pie(category_counts, labels = ['Happy','Sad','Angry','Neutral'],
    #         autopct =make_autopct(category_counts), shadow=False)
    # plt.show()
    #
    # plt.pie(speaker_counts, labels = ['Ses01F','Ses01M','Ses02F','Ses02M','Ses03F',
    #                                 'Ses03M','Ses04F','Ses04M','Ses05F','Ses05M'],
    #         autopct ='%1.1f%%', shadow=False)
    # plt.show()

    # 1.34591066837310


    ############################################
    #   Finding min and max intensity of mels  #
    ############################################
    # i = 0
    # mels_made = 0
    # mel_lengths = []
    #
    # max_intensity = 0
    # min_intensity = 99999999
    #
    # for f in filenames:
    #
    #     wav, labels = get_wav_and_labels(f, data_dir)
    #     mel = audio_utils.wav2melspectrogram(wav)
    #     labels = np.array(labels)
    #     if labels[0] != -1:
    #
    #         # mel_lengths.append(mel.shape[1])
    #         max_val = np.max(mel)
    #         min_val = np.min(mel)
    #
    #         if max_val > max_intensity:
    #             max_intensity = max_val
    #         if min_val < min_intensity:
    #             min_intensity = min_val
    #         mels_made += 1
    #
    #     i += 1
    #     if i % 100 == 0:
    #         # print(mel_lengths[mels_made-1])
    #         print(mel[:, 45])
    #         print(max_intensity, ", ", min_intensity)
    #         print(i, " complete.")
    #         print(mels_made, "mels made.")
    #
    # print("max = {}".format(max_intensity))
    # print("min = {}".format(min_intensity))
    #
    # np.save('./stats/all_mel_lengths', np.array(mel_lengths))
    #
    # n, bins, patches = plt.hist(mel_lengths, bins = 22)
    # plt.xlabel('Sequence length')
    # plt.ylabel('Count')
    # plt.title(r'Histogram of sequence lengths for 4 emotional categories')
    # plt.show()
    #
    # mel_lengths = sorted(mel_lengths)
    # print(mel_lengths[0:30])
    # split_index = int(len(mel_lengths)*0.9)
    # print(mel_lengths[split_index])  # IS MAX LENGTH OF mels
    # print(mel_lengths[0])  # IS MIN LENGTH OF mels
