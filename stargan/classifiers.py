"""
classifiers.py

Author - Max Elliott

All classifier models used in this thesis. Emotion_Classifier is the main
auxiliary classifier for categorical emotion recognition. Dimension_Classifier
is the auxiliary classifier for dimensional emotion recognition (didn't end up
being used in this thesis).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stargan.average_weighted_attention import Average_Weighted_Attention


class Emotion_Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, bi=False,
                 device='cuda'):
        """
        NOTE: input size must be directly divisible by 4
        Is also used for speaker classifier
        """
        super(Emotion_Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size # == n_mels/world feats
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.num_outchannels = 32
        self.m_factor = 2 if bi else 1

        self.device = device  # Legacy now, never actually used

        kernel = 7
        padding = int((kernel-1)/2)
        self.conv1 = nn.Conv2d(1, 16, kernel, padding=padding)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(16, 24, kernel, padding=padding)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(24, self.num_outchannels, kernel, padding=padding)
        self.maxpool3 = nn.MaxPool2d(2, stride=2)

        self.lstm1 = nn.LSTM(input_size=self.num_outchannels*(self.input_size//8),
                             hidden_size=self.hidden_size, num_layers=self.num_layers,
                             batch_first=True, bidirectional=bi)
        self.att = Average_Weighted_Attention(self.hidden_size*self.m_factor)

        self.fc = nn.Linear(self.m_factor*hidden_size, 64)
        self.drop = nn.Dropout(p=0.2)
        self.out = nn.Linear(64, self.num_classes)

    def forward(self, x_data, x_lens):
        """
        x is size (batch_size, 1, max_seq_length, feature_dim)
        x_lens is size (batch_size, 1), contains seq_lens
        batch is in descending seq_len order
        """

        batch_size = x_data.size(0)
        no_features = x_data.size(3)
        curr_device = x_data.device

        # Convolutional layers
        x_data = self.maxpool1(F.relu(self.conv1(x_data)))
        x_data = self.maxpool2(F.relu(self.conv2(x_data)))
        x_data = self.maxpool3(F.relu(self.conv3(x_data)))
        x_lens = x_lens//8    # seq_len have got ~4 times shorted
        # x = (B, channels, max_l//4, n_mels//4)

        # Recurrent layers
        x_data = x_data.permute(0,2,1,3)
        x_data = x_data.contiguous().view(batch_size, -1, self.num_outchannels*(no_features//8))
        # Now x = (B, max_l//8, channels*(n_mels//8))

        x_data = nn.utils.rnn.pack_padded_sequence(x_data, x_lens,
                                                   batch_first=True,
                                                   enforce_sorted=True)

        h0 = torch.zeros(self.m_factor*self.num_layers, batch_size,
                         self.hidden_size).to(device=curr_device, dtype=torch.float)

        c0 = torch.zeros(self.m_factor*self.num_layers, batch_size,
                         self.hidden_size).to(device=curr_device, dtype=torch.float)

        # LSTM returns: (seq_len, batch, num_directions * hidden_size),
        #               ((num_layers * num_directions, batch, hidden_size), c_n)
        x_data, _ = self.lstm1(x_data, (h0, c0))

        x_data, x_lens = torch.nn.utils.rnn.pad_packed_sequence(x_data, batch_first=True)

        x_data = self.att(x_data)

        # Alternate non-attention based method: take the final hidden layer for each sequence
        # x_data = torch.stack([row[x_lens[i]-1] for (i,row) in enumerate(x_data)]) #(B, m_factor*hidden_size)

        x_data = self.drop(F.relu(self.fc(x_data)))

        x_data = self.out(x_data)

        return x_data


class Dimension_Classifier(nn.Module):
    """
    Model for multi-task classification of valence and arousal.
    Uses three conv2d->maxpooling layers, into two separate sequential modelling
    networks for prediction of valence and arousal.
    """
    def __init__(self, input_size, hidden_size, num_layers, bi=False, device='cuda'):
        """
        NOTE: input size must be directly divisible by 4
        """
        super(Dimension_Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size  # == n_mels
        self.num_layers = num_layers
        self.num_outchannels = 32
        self.m_factor = 2 if bi else 1

        kernel = 7
        padding = int((kernel-1)/2)
        self.conv1 = nn.Conv2d(1, 16, kernel, padding=padding)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(16, 24, kernel, padding=padding)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(24, self.num_outchannels, kernel, padding=padding)
        self.maxpool3 = nn.MaxPool2d(2, stride=2)

        self.valence_predictor = Single_Dimension_Classifier(
                        input_size=self.input_size*(self.num_outchannels//8),
                        hidden_size=self.hidden_size,
                        num_layers=self.num_layers,
                        bi=bi, device=device)
        self.arousal_predictor = Single_Dimension_Classifier(
                        input_size=self.input_size*(self.num_outchannels//8),
                        hidden_size=self.hidden_size,
                        num_layers=self.num_layers,
                        bi=bi, device=device)
        # self.dominance_predictor = Single_Dimension_Classifier(
        #                 input_size = (self.input_size*self.num_outchannels)//8,
        #                 hidden_size = self.hidden_size,
        #                 num_layers = self.num_layers,
        #                 bi = bi, device = device)

    def forward(self, x_data, x_lens):
        """
        x[0] is size (batch_size, max_seq_length, feature_dim)
        x[1] is size (batch_size, 1), contains seq_lens
        batch is in descending seq_len order
        """

        batch_size = x_data.size(0)
        no_features = x_data.size(2)

        # Convolutional layers
        x_data = self.maxpool1(F.relu(self.conv1(x_data)))
        x_data = self.maxpool2(F.relu(self.conv2(x_data)))
        x_data = self.maxpool3(F.relu(self.conv3(x_data)))
        x_lens = x_lens//8    # seq_len have got ~4 times shorted
        # x = (B, channels, max_l//4, n_mels//4)

        # Recurrent layers

        x_data = x_data.permute(0,2,1,3)
        x_data = x_data.contiguous().view(batch_size, -1, self.num_outchannels*no_features//8)
        # Now x = (B, max_l//4, channels*n_mels//4)

        x_data = nn.utils.rnn.pack_padded_sequence(x_data, x_lens,
                                                   batch_first=True,
                                                   enforce_sorted=True)

        # Pass into 3 single_dim_predictors
        x_val = self.valence_predictor(x_data, batch_size)
        x_aro = self.arousal_predictor(x_data, batch_size)
        # x_dom = self.dominance_predictor(x_data, batch_size)


        return x_val, x_aro #, x_dom


class Single_Dimension_Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bi=False, device='cuda'):
        super(Single_Dimension_Classifier, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size  # == n_mels
        self.num_layers = num_layers
        self.num_outchannels = 32
        self.m_factor = 2 if bi else 1

        self.device = device  # Now legacy as isn't used

        self.lstm1 = nn.LSTM(input_size=self.input_size,
                           hidden_size=self.hidden_size, num_layers=self.num_layers,
                           batch_first=True, bidirectional=bi)
        self.att = Average_Weighted_Attention(self.hidden_size*self.m_factor)

        self.fc1 = nn.Linear(self.hidden_size*self.m_factor, (self.hidden_size*self.m_factor)//2)
        self.fc2 = nn.Linear((self.hidden_size*self.m_factor)//2, 3)

    def forward(self, x, batch_size):

        curr_device = x.device

        h0 = torch.zeros(self.m_factor*self.num_layers, batch_size,
                         self.hidden_size).to(device=curr_device, dtype=torch.float)

        c0 = torch.zeros(self.m_factor*self.num_layers, batch_size,
                         self.hidden_size).to(device=curr_device, dtype=torch.float)

        # LSTM returns: (seq_len, batch, num_directions * hidden_size),
        #               ((num_layers * num_directions, batch, hidden_size), c_n)
        x_data, _ = self.lstm1(x, (h0, c0))

        x_data, x_lens = torch.nn.utils.rnn.pad_packed_sequence(x_data, batch_first=True)

        x_data = self.att(x_data)

        x_data = F.relu(self.fc1(x_data))
        x_data = self.fc2(x_data)

        return x_data


if __name__ == '__main__':
    pass
