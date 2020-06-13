"""
my_dataset.py

Author - Max Elliott

The custom dataset and collate function described in the report.
"""

import torch
import torch.utils.data as data_utils

import numpy as np
from librosa.util import find_files
import random
import os


def get_filenames(dir):

    files = find_files(dir, ext='npy')
    filenames = []

    for f in files:
        f = os.path.basename(f)[:-4]
        filenames.append(f)

    return filenames


def shuffle(in_list):
    """
    in_list: list to be shuffled
    """

    indices = list(range(len(in_list)))
    random.shuffle(indices)

    shuffled_list = []

    for i in indices:
        shuffled_list.append(in_list[i])

    return shuffled_list


def _pad_sequence(seq, length, pad_value=0):
    new_seq = torch.zeros((length,seq.size(1)))
    if seq.size(0) <= length:
        new_seq[:seq.size(0), :] = seq
    else:
        new_seq[:seq.size(0), :] = seq[:length, :]
    return new_seq


def crop_sequences(seq_list, labels, segment_len):
    """
    seq_list = ([(seq_len, n_feats)])
    labels = ([label])
    """
    new_seqs = []
    new_labels = []

    for i, seq in enumerate(seq_list):

        while seq.size(0) >= segment_len:

            new_seq = seq[0:segment_len, :]
            new_seqs.append(new_seq)
            new_labels.append(labels[i])

            seq = torch.Tensor(seq[segment_len:, :])
            if new_seq.size(0) != segment_len:
                print(i, new_seq.size(0))

        if seq.size(0) > segment_len // 2:

            new_seq = _pad_sequence(seq, segment_len)
            new_seqs.append(new_seq)
            new_labels.append(labels[i])

    return new_seqs, new_labels


class MyDataset(data_utils.Dataset):

    def __init__(self, config, filenames):
        super(MyDataset, self).__init__()

        self.config = config
        self.dataset_dir = config['data']['dataset_dir']

        if config['data']['type'] == 'mel':
            self.feat_dir = os.path.join(self.dataset_dir, "mels")
        else:
            self.feat_dir = os.path.join(self.dataset_dir, "world")

        self.labels_dir = os.path.join(self.dataset_dir, "labels")

        self.filenames = filenames

    def __getitem__(self, index):

        f = self.filenames[index]
        mel = np.load(self.feat_dir + "/" + f + ".npy")
        label = np.load(self.labels_dir + "/" + f + ".npy")

        mel = torch.FloatTensor(mel).t()
        label = torch.Tensor(label).long()

        return mel, label

    def __len__(self):
        return len(self.filenames)


def collate_length_order(batch):
    """
    batch: Batch elements are tuples ((Tensor)sequence, target)

    Sorts batch by sequence length

    returns:
        (FloatTensor) sequence_padded: seqs in length order, padded to max_len
        (LongTensor) lengths: lengths of seqs in sequence_padded
        (LongTensor) labels: corresponding targets, in correct order
    """
    # assume that each element in "batch" is a tuple (data, label).
    # Sort the batch in the descending order
    sorted_batch = sorted(batch, key=lambda x: x[0].size(0), reverse=True)

    # Get each sequence and pad it
    sequences = [x[0] for x in sorted_batch]

    #################################################
    #            FOR FIXED LENGTH INPUTS            #
    #################################################
    for i,seq in enumerate(sequences):
        if seq.size(0) > 512:
            start_index = random.randint(0, seq.size(0)-512)
            sequences[i] = seq[start_index:start_index+512, :]

    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    current_len = sequences_padded.size(1)
    if current_len < 512:
        pad_len = 512 - current_len
        new_tensor = torch.zeros((sequences_padded.size(0),pad_len,sequences_padded.size(2)))
        sequences_padded = torch.cat([sequences_padded, new_tensor], dim =1)
    # else:
    #     sequences_padded = sequences_padded[:,:512,:]
    # print(f"Padded length: {sequences_padded.size(1)}")

    # sequences_padded = [_pad_sequence(x, 512) for x in sequences]
    # sequences_padded = torch.stack(sequences_padded)

    #################################################
    #          FOR VARIABLE LENGTH INPUTS           #
    #################################################
    # sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    #
    # max_len = sequences_padded.size(1)
    # print("Original size = ", max_len)
    # div8 = max_len%8==0
    # div5 = max_len%5==0
    # div9 = max_len%3==0
    # if not (div8 and div5 and div9):
    #     pad_len = max_len + 1
    #     # print("Current pad:", pad_len)
    #     while (pad_len%8 !=0 or pad_len%5!=0 or pad_len%3!=0):
    #         pad_len += 1
    #         # print("Current pad:", pad_len%9)
    # div16 = max_len%16==0
    #
    # if not div16:
    #     pad_len = max_len + 1
    #     # print("Current pad:", pad_len)
    #     while pad_len%16 !=0:
    #         pad_len += 1
    #         # print("Current pad:", pad_len%9)
    #     pad_len = pad_len - max_len
    #     new_tensor = torch.zeros((sequences_padded.size(0),pad_len,sequences_padded.size(2)))
    #     sequences_padded = torch.cat([sequences_padded, new_tensor], dim =1)

    # print("New size = ", max_len+pad_len)
    # print("Pad size = ", pad_len)

    # Also need to store the length of each sequence
    # This is later needed in order to unpad the sequences
    lengths = [len(x) for x in sequences]
    for i, l in enumerate(lengths):
        if l > 512:
            lengths[i] = 512
    lengths = torch.LongTensor([len(x) for x in sequences])

    # Don't forget to grab the labels of the *sorted* batch
    targets = torch.stack([x[1] for x in sorted_batch]).long()

    return [sequences_padded, lengths], targets


def make_variable_dataloader(train_set, test_set, batch_size=64):

    train_loader = data_utils.DataLoader(train_set, batch_size=batch_size,
                                         collate_fn=collate_length_order,
                                         num_workers=0, shuffle=True)

    test_loader = data_utils.DataLoader(test_set, batch_size=batch_size,
                                        collate_fn=collate_length_order,
                                        num_workers=0, shuffle=True)

    return train_loader, test_loader


if __name__ == '__main__':
    pass
