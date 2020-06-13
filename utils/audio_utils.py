'''
audio_utils.py

Author - Max Elliott

Helper functions for reading and writing wav files.
Hyperparameters are stored in the hyperparams class
'''

from scipy.io import wavfile
import os
import yaml
import copy
import pickle

import librosa
import librosa.display

from pyworld import decode_spectral_envelope, synthesize

import numpy as np
import torch

import matplotlib.pyplot as plt

# dataset_dir = "/Users/Max/MScProject/datasets/IEMOCAP"
# dataset_dir = "/Users/Max/MScProject/test_dir"


class hyperparams(object):
    def __init__(self):
        self.sr = 16000  # Sampling rate. Paper => 24000
        self.n_fft = 1024  # fft points (samples)
        self.frame_shift = 0.0125  # seconds
        self.frame_length = 0.05   # seconds
        self.hop_length = int(self.sr*self.frame_shift)  # samples  This is dependent on the frame_shift.
        self.win_length = int(self.sr*self.frame_length)  # samples This is dependent on the frame_length.
        self.n_mels = 80  # Number of Mel banks to generate
        self.power = 1.2 # Exponent for amplifying the predicted magnitude
        self.n_iter = 100  # Number of inversion iterations
        self.use_log_magnitude = True  # if False, use magnitude
        self.preemph = 0.97

        self.config = yaml.load(open('./config.yaml', 'r'))
        self.sample_set_dir = self.config['logs']['sample_dir']

        self.normalise = self.config['data']['normalise']
        self.max_norm_value = 3226.99139880277
        self.min_norm_value = 3.8234146815389095e-10

        self.sp_max_norm_value = 6.482182376067761
        self.sp_min_norm_value = -18.50642857581744

        # Store dictionaries used for f0 pitch transformations
        if os.path.exists('./f0_dict.pkl'):
            with open('./f0_dict.pkl', 'rb') as fp:
                self.f0_dict = pickle.load(fp)

        if os.path.exists('./f0_relative_dict.pkl'):
            with open('./f0_relative_dict.pkl', 'rb') as fp:
                self.f0_relative_dict = pickle.load(fp)

hp = hyperparams()


def load_wav(path):
    wav = wavfile.read(path)[1]
    wav = copy.deepcopy(wav)/32767.0
    return wav


def save_wav(wav, path):
    # wav *= 32767 / max(0.01, np.max(np.abs(wav)))

    wav *= 48000
    wav = np.clip(wav, -32767, 32767)
    wavfile.write(path, hp.sr, wav.astype(np.int16))


def wav2spectrogram(y, sr=hp.sr):

    '''
    Produces log-magnitude spectrogram of audio data y
    '''

    spec = librosa.core.stft(y, n_fft=hp.n_fft, hop_length=hp.hop_length,
                                                win_length=hp.win_length)
    spec_mag = amp_to_db(np.abs(spec))

    return spec_mag


def _normalise_mel(mel):
    mel = (mel - hp.min_norm_value)/(hp.max_norm_value - hp.min_norm_value)
    return mel


def _unnormalise_mel(mel):
    mel = (hp.max_norm_value - hp.min_norm_value) * mel + hp.min_norm_value
    return mel


def _normalise_coded_sp(sp):
    sp = (sp - hp.sp_min_norm_value)/(hp.sp_max_norm_value - hp.sp_min_norm_value)

    return sp


def _unnormalise_coded_sp(sp):
    sp = (hp.sp_max_norm_value - hp.sp_min_norm_value) * sp + hp.sp_min_norm_value
    np.clip(sp, hp.sp_min_norm_value, hp.sp_max_norm_value)
    return sp


def wav2melspectrogram(y, sr = hp.sr, n_mels = hp.n_mels):
    '''
    y = input wav file
    '''

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
        n_fft=hp.n_fft, hop_length=hp.hop_length)
    # mel_spec = librosa.core.amplitude_to_db(y)
    if hp.normalise:
        mel_spec = _normalise_mel(mel_spec)

    return mel_spec


def spectrogram2melspectrogram(spec, n_fft=hp.n_fft, n_mels=hp.n_mels):

    if isinstance(spec, torch.Tensor):
        spec = spec.numpy()

    mels = librosa.filters.mel(hp.sr, n_fft, n_mels=n_mels)
    return mels.dot(spec**hp.power)


def melspectrogram2wav(mel):
    '''
    Not implemented
    '''
    return 0


def spectrogram2wav(spectrogram):
    '''
    Griffin-Lim Algorithm
    spectrogram: [t, f], i.e. [t, nfft // 2 + 1]
    '''
    if isinstance(spectrogram, torch.Tensor):
        spectrogram = spectrogram.cpu().numpy()

    spectrogram = db_to_amp(spectrogram)#**hp.power

    X_best = copy.deepcopy(spectrogram)  # [f, t]
    for i in range(hp.n_iter):

        X_t = invert_spectrogram(X_best)
        # print(X_t.shape())
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)  # [f, t]
        phase = est / np.maximum(1e-8, np.abs(est))  # [f, t]
        X_best = spectrogram * phase  # [f, t]
    X_t = invert_spectrogram(X_best)

    return np.real(X_t)


def invert_spectrogram(spectrogram):
    '''
    spectrogram: [f, t]
    '''
    return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")


def amp_to_db(spec):
    return librosa.core.amplitude_to_db(spec)


def db_to_amp(spec):
    return librosa.core.db_to_amplitude(spec)


def plot_spec(spec, type = 'mel'):

    plt.figure(figsize=(6, 4))

    if isinstance(spec, torch.Tensor):
        spec = spec.cpu().numpy()

    if hp.normalise:
        spec = _unnormalise_mel(spec)

    librosa.display.specshow(librosa.power_to_db(spec), y_axis=type, sr=hp.sr,
                                hop_length=hp.hop_length)
                                                    # fmin=None, fmax=4000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Power spectrogram')
    plt.show()


def save_spec(spec, model_name, filename, type = 'mel'):
    '''
    spec: [n_feats, seq_len] - np.array or torch.Tensor
    model_name: str - just the basename, no directory
    filename: str
    '''
    fig = plt.figure(figsize=(6,4))

    if isinstance(spec, torch.Tensor):
        spec = spec.cpu().numpy()
    if hp.normalise:
        spec = _unnormalise_mel(spec)

    path = os.path.join(hp.sample_set_dir, model_name)

    if not os.path.exists(path):
        os.makedirs(path)

    path = os.path.join(path, filename)

    np.save(path, spec)

    # print("Saved.")


def save_spec_plot(spec, model_name, filename, type = 'mel'):
    '''
    spec: [n_feats, seq_len] - np.array or torch.Tensor
    model_name: str - just the basename, no directory
    filename: str
    '''
    fig = plt.figure(figsize=(6,4))

    if isinstance(spec, torch.Tensor):
        spec = spec.cpu().numpy()
    if hp.normalise:
        spec = _unnormalise_mel(spec)

    librosa.display.specshow(librosa.power_to_db(spec), y_axis=type, sr=hp.sr,
                            hop_length=hp.hop_length)
                                                    # fmin=None, fmax=4000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Power spectrogram')

    path = os.path.join(hp.sample_set_dir, model_name)

    if not os.path.exists(path):
        os.makedirs(path)

    path = os.path.join(path, filename)

    plt.savefig(path)
    plt.close(fig)
    plt.close("all")
    # print("Saved.")


def save_world_wav(feats, filename):

    # feats = [f0, sp, ap, sp_coded, labels]

    if isinstance(feats[3], torch.Tensor):
        feats[3] = feats[3].cpu().numpy()
    if hp.normalise:
        feats[3] = _unnormalise_coded_sp(feats[3])

    # path = os.path.join(hp.sample_set_dir, model_name)

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    # path = os.path.join(path, filename)

    feats[3] = np.ascontiguousarray(feats[3], dtype=np.float64)
    decoded_sp = decode_spectral_envelope(feats[3], hp.sr, fft_size=hp.n_fft)
    wav = synthesize(feats[0], decoded_sp, feats[1], hp.sr)

    save_wav(wav, filename)


def f0_pitch_conversion(f0, source_labels, target_labels):
    '''
    Logarithm Gaussian normalization for Pitch Conversions
    (np.array) f0 - array to be converted
    (tuple) source_labels - (emo, speaker) discrete labels
    (tuple) target_labels - (emo, speaker) discrete labels
    If doing relative-LGNT, then speaker can be anything becuase its not used
    '''
    src_emo = int(source_labels[0])
    src_spk = int(source_labels[1])
    trg_emo = int(target_labels[0])
    trg_spk = int(target_labels[1])

    # ----- Absolute transformation ----- #
    # mean_log_src = hp.f0_dict[src_emo][src_spk][0]
    # std_log_src = hp.f0_dict[src_emo][src_spk][1]
    # mean_log_target = hp.f0_dict[trg_emo][src_spk][0]
    # std_log_target = hp.f0_dict[trg_emo][src_spk][1]
    #
    # f0_converted = np.exp((np.ma.log(f0) - mean_log_src) / std_log_src * std_log_target + mean_log_target)

    # ----- Proposed relative transformation ----- #
    logf0 = np.ma.log(f0)
    mean = np.mean(logf0)
    var = np.var(logf0)
    f0_converted = np.exp((logf0-mean)/var * (hp.f0_relative_dict[src_emo][trg_emo][1]+var) + mean + hp.f0_relative_dict[src_emo][trg_emo][0])

    return f0_converted


if __name__ == '__main__':

    #####################################
    #  PLOTTING CONVERTED SPECTROGRAMS  #
    #####################################

    file = '../data/audio/Ses01F_impro02_F014.wav' #8

    wav = load_wav(file)
    spec = wav2melspectrogram(wav)
    print("Max = ", np.max(librosa.power_to_db(spec)))
    print("Min = ", np.min(librosa.power_to_db(spec)))

    spec = spec[:, 16:-16]
    print("Original size =", spec.shape)

    fig = plt.figure(figsize=(9, 13))
    ax1 = fig.add_subplot(4, 2, 1)

    plt.subplot(4, 2, 1)
    librosa.display.specshow(librosa.power_to_db(spec), y_axis='mel', sr=hp.sr,
                            hop_length=hp.hop_length, vmax=-8.47987, vmin=-100.0)
                                                    # fmin=None, fmax=4000)
    # plt.colorbar(format='%+2.0f dB')
    plt.title('1) Original (sad)')

    # world3_4d_newloader_cont_080_testSet
    file = './samples/final/3-emo_spec_100/Ses01F_impro02_F014_1to2.wav'

    wav = load_wav(file)
    spec = wav2melspectrogram(wav)
    print("Max = ", np.max(librosa.power_to_db(spec)))
    print("Min = ", np.min(librosa.power_to_db(spec)))

    ax2 = fig.add_subplot(4, 2, 2)

    librosa.display.specshow(librosa.power_to_db(spec), y_axis='mel', sr=hp.sr,
                            hop_length=hp.hop_length, vmax = -8.47987, vmin= -100.0)
                                                    # fmin=None, fmax=4000)
    # plt.colorbar(format='%+2.0f dB')
    plt.title('2) 3 Emotion (happy)')

    file = './samples/f0s/world2_crop_4d_200_200_testSet/Ses01F_impro02_F014_1to1.wav'

    wav = load_wav(file)
    spec = wav2melspectrogram(wav)
    print("Max = ", np.max(librosa.power_to_db(spec)))
    print("Min = ", np.min(librosa.power_to_db(spec)))


    ax3 = fig.add_subplot(4, 2, 3, sharey=ax1)
    librosa.display.specshow(librosa.power_to_db(spec), y_axis='mel', sr=hp.sr,
                            hop_length=hp.hop_length, vmax = -8.47987, vmin= -100.0)
                                                    # fmin=None, fmax=4000)
    # plt.colorbar(format='%+2.0f dB')
    plt.title('3) 2 Emotion (sad)')

    file = './samples/final/3-emo_spec_100/Ses01F_impro02_F014_1to1.wav'
    # file = './samples/final/3-emo_spec_100/Ses01F_impro02_F014_1to2.wav'

    wav = load_wav(file)
    spec = wav2melspectrogram(wav)
    print("Max = ", np.max(librosa.power_to_db(spec)))
    print("Min = ", np.min(librosa.power_to_db(spec)))

    # if hp.normalise:
    #     spec = _unnormalise_mel(spec)

    ax4 = fig.add_subplot(4, 2, 4, sharey=ax1)
    librosa.display.specshow(librosa.power_to_db(spec), y_axis='mel', sr=hp.sr,
                            hop_length=hp.hop_length, vmax = -8.47987, vmin= -100.0)
                                                    # fmin=None, fmax=4000)
    # plt.colorbar(format='%+2.0f dB')
    plt.title('4) 3 Emotion (sad)')
    #

    file = './samples/f0s/world2_crop_4d_200_200_testSet/Ses01F_impro02_F014_1to0.wav'

    wav = load_wav(file)
    spec = wav2melspectrogram(wav)
    print("Max = ", np.max(librosa.power_to_db(spec)))
    print("Min = ", np.min(librosa.power_to_db(spec)))

    # if hp.normalise:
    #     spec = _unnormalise_mel(spec)

    ax4 = fig.add_subplot(4, 2, 5, sharey=ax1)
    librosa.display.specshow(librosa.power_to_db(spec), y_axis='mel', sr=hp.sr,
                            hop_length=hp.hop_length, vmax = -8.47987, vmin= -100.0)
                                                    # fmin=None, fmax=4000)
    # plt.colorbar(format='%+2.0f dB')
    plt.title('5) 2 Emotion (angry)')


    file = './samples/final/3-emo_spec_100/Ses01F_impro02_F014_1to0.wav'

    wav = load_wav(file)
    spec = wav2melspectrogram(wav)
    print("Max = ", np.max(librosa.power_to_db(spec)))
    print("Min = ", np.min(librosa.power_to_db(spec)))

    # if hp.normalise:
    #     spec = _unnormalise_mel(spec)

    ax4 = fig.add_subplot(4, 2, 6, sharey=ax1)
    librosa.display.specshow(librosa.power_to_db(spec), y_axis='mel', sr=hp.sr,
                            hop_length=hp.hop_length, vmax = -8.47987, vmin= -100.0)
                                                    # fmin=None, fmax=4000)
    # plt.colorbar(format='%+2.0f dB',orientation='horizontal')
    plt.title('6) 3 Emotion (angry)')

    cbaxes = fig.add_subplot(4,2,8)
    # cbaxes = fig.add_axes([0.8, 0.1, 0.03, 0.8])
    # cb = plt.colorbar(ax4, format='%+2.0f dB')
    plt.colorbar(format='%+2.0f dB',orientation='horizontal')


    plt.savefig('../graphs/specs/All+cb.png')
    plt.close(fig)
    plt.close("all")
    ######################################
    ##   CALCULATE RELATIVE F0 STATS   ###
    ######################################
    #
    # emo2emo_dict = {}
    #
    # for e1 in range(0,4):
    #
    #     emo2emo_dict[e1] = {}
    #
    #     for e2 in range(0,4):
    #
    #         mean_list = []
    #         std_list = []
    #
    #         for s in range(0,10):
    #             mean_diff = hp.f0_dict[e2][s][0] - hp.f0_dict[e1][s][0]
    #             std_diff = hp.f0_dict[e2][s][1] - hp.f0_dict[e1][s][1]
    #             mean_list.append(mean_diff)
    #             std_list.append(std_diff)
    #
    #         mean_mean = np.mean(mean_list)
    #         std_mean = np.mean(std_list)
    #         emo2emo_dict[e1][e2] = (mean_mean, std_mean)
    #
    # for tag, val in emo2emo_dict.items():
    #     print(f'Emotion {tag} stats:')
    #     for tag2, val2 in val.items():
    #         print(f'{tag2} = {val2[0]}, {val2[1]}')
    #
    # with open('f0_relative_dict2.pkl', 'wb') as f:
    #     pickle.dump(emo2emo_dict, f, pickle.HIGHEST_PROTOCOL)

    #######################################
    ###     CODE FOR F0 EXPERIMENTS     ###
    #######################################

    # for tag, val in hp.f0_dict.items():
    #     print(f'Emotion {tag} stats:')
    #     for tag2, val2 in val.items():
    #         print(f'{tag2} & {val2[0]:.3f} & {val2[1]:.3f} \\\\ \hline')
    #
    # for tag, val in hp.f0_relative_dict.items():
    #     print(f'Emotion {tag} stats:')
    #     for tag2, val2 in val.items():
    #         print(f'{tag2} & {val2[0]:.3f} & {val2[1]:.3f}')
    #
    # files = librosa.util.find_files("/Users/Max/MScProject/data/f0", ext = "npy")
    # # basenames = [os.path.basename(f) for f in files]
    # print(len(files))
    # f0s = [(np.load(f), np.load(os.path.join("/Users/Max/MScProject/data/labels",os.path.basename(f)))[0]) \
    #         for f in files \
    #         if np.load(os.path.join("/Users/Max/MScProject/data/labels",os.path.basename(f)))[1]==0]
    # print(len(f0s))
    # print(f0s[0][0].shape)
    #
    # labels = [x[1] for x in f0s]
    # f0s = [x[0] for x in f0s]

    # angry = []
    # for i,f0 in enumerate(f0s):
    #     if labels[i] == 0:
    #         angry.append(f0)
    #
    # conv_cat = np.ma.log(np.concatenate(angry))
    # # conv_cat = np.concatenate(converted)
    # conv_cat_no0s = []
    #
    # for i,v in enumerate(conv_cat):
    #     if v != 0:
    #         conv_cat_no0s.append(v)
    # # print(conv_cat_no0s[0:300])
    # log_f0s_mean = np.nanmean(conv_cat_no0s)
    # log_f0s_std = np.nanvar(conv_cat_no0s)
    #
    # print("Mean =", log_f0s_mean)
    # print("STD =", log_f0s_std)

    # for i in range(0,4):
    #     # f0s_copy = [np.copy(f0) for f0 in f0s]
    #     print("Emotion: ", i)
    #     originals = []
    #     converted = []
    #     for j, f0 in enumerate(f0s):
    #
    #         # if labels[j] == i:
    #             # originals.append(f0)
    #         converted.append(f0_pitch_conversion(f0,(labels[j],0),(i,0)))
    #
    #     # for i,val in enumerate(originals[25]):
    #         # print(np.ma.log(val), ", ", np.ma.log((converted[25][i])))
    #
    #     # for i,val in enumerate(converted[0]):
    #         # print(np.ma.log(val), ", ", np.ma.log((f0s[0][i])))
    #
    #     # print(len(converted))
    #
    #     conv_cat = np.ma.log(np.concatenate(converted))
    #     # conv_cat = np.concatenate(converted)
    #     conv_cat_no0s = []
    #
    #     for i,v in enumerate(conv_cat):
    #         if v != 0:
    #             conv_cat_no0s.append(v)
    #     # print(conv_cat_no0s[0:300])
    #     log_f0s_mean = np.nanmean(conv_cat_no0s)
    #     log_f0s_std = np.nanvar(conv_cat_no0s)

        # sum=0
        # for val in conv_cat:
        #     sum += val
        #
        # print(sum)
        # print(sum/len(conv_cat))

        # print(log_f0s_mean)
        # print(log_f0s_std)

    # for i,val in enumerate(f0s_copy[0]):
        # print(val, ", ", converted[0][i])
    # print("{:.2f}".format(f0s_copy[0]))
    # print("{:.2f}".format(converted[0]))


    # # files = [os.path.basename(f) for f in files]
    # print(files)
    # numbers = []
    # for f in files:
    #     # numbers.append(f[-5])
    #     if f[-8] != '3':
    #         # if f[-5] != '0':
    #
    #         name = os.path.basename(f)[:-10] + os.path.basename(f)[-8:]
    #         numbers.append(name)
    #         os.rename(f, "/Users/Max/MScProject/StarGAN-Emotional-VC/samples/DA/all/3-emo-augmented/" + name)
    #         # os.rename(f, "/Users/Max/MScProject/StarGAN-Emotional-VC/samples/DA/all/actual/" + name)
    #
    # print(numbers)
