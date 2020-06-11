"""
average_weighted_attentions.py

Author: Max Elliott

Implementation of the average weighted attention layers proposed in
ATTENTION-AUGMENTED END-TO-END MULTI-TASK LEARNING FOR EMOTION PREDICTION FROM
SPEECH by Zixing Zhang.
"""

import torch
import torch.nn as nn
import numpy as np


class Average_Weighted_Attention(nn.Module):
    def __init__(self, vector_size):
        super(Average_Weighted_Attention, self).__init__()
        self.vector_size = vector_size
        self.weights = nn.Parameter(torch.randn(self.vector_size, 1, requires_grad=True)/np.sqrt(self.vector_size),
                                    requires_grad=True)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x):
        """
        x.size() = (Batch, max_seq_len, n_feats)
        """

        original_sizes = x.size()
        x = x.contiguous().view(original_sizes[0]*original_sizes[1], -1)

        x_dot_w = x.mm(self.weights)

        x_dot_w = x_dot_w.view(original_sizes[0], original_sizes[1])

        softmax = nn.Softmax(dim=1)
        alphas = softmax(x_dot_w)
        alphas = alphas.view(-1, 1)

        x = x.mul(alphas)
        x = x.view(original_sizes)
        x = torch.sum(x, dim=1)

        return x
