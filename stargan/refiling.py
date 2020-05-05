'''
refiling.py

Author - Max Elliott

Script for refiling the IEMOCAP dataset.
'''

from IPython.display import Audio
from scipy.io import wavfile
import os
import librosa
import numpy as np
import torch
import torch.nn as nn
import audio_utils
import pickle
from shutil import copyfile

dataset_dir = "/Users/Max/MScProject/datasets/IEMOCAP"

# dataset_dir = "/Users/Max/MScProject/datasets/test_dir"

def copy_files(data_dir):

    # print(data_dir)
    os.chdir(data_dir)

    filenames = []
    specs = []
    mels = []
    labels = []
    conts = []
    conts_dis = []
    speakers = []
    for session in os.listdir(data_dir):

        if not (session == "Processed_data" or session == ".DS_Store" or session == "All"):

            for foldername in os.listdir(data_dir + "/" + session):

                if not (foldername == "Annotations" or foldername == ".DS_Store"):

                    for filename in os.listdir(data_dir + "/" + session + "/" + foldername):

                        if not filename == ".DS_Store":
                            src_dir = data_dir + "/" + session + "/" + foldername + "/" + filename
                            dest_dir = data_dir + "/All/" + filename
                            copyfile(src_dir, dest_dir)

                print(foldername + " completed.")

if __name__ == '__main__':

    copy_files(dataset_dir)
