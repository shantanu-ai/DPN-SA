"""
MIT License

Copyright (c) 2020 Shantanu Ghosh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from Utils import Utils


class DCN(nn.Module):
    def __init__(self, training_flag, input_nodes):
        super(DCN, self).__init__()
        self.training = training_flag

        # shared layer
        self.shared1 = nn.Linear(in_features=input_nodes, out_features=200)
        nn.init.xavier_uniform_(self.shared1.weight)

        self.shared2 = nn.Linear(in_features=200, out_features=200)
        nn.init.xavier_uniform_(self.shared2.weight)

        # potential outcome1 Y(1)
        self.hidden1_Y1 = nn.Linear(in_features=200, out_features=200)
        nn.init.xavier_uniform_(self.hidden1_Y1.weight)

        self.hidden2_Y1 = nn.Linear(in_features=200, out_features=200)
        nn.init.xavier_uniform_(self.hidden2_Y1.weight)

        self.out_Y1 = nn.Linear(in_features=200, out_features=2)
        nn.init.xavier_uniform_(self.out_Y1.weight)

        # potential outcome1 Y(0)
        self.hidden1_Y0 = nn.Linear(in_features=200, out_features=200)
        nn.init.xavier_uniform_(self.hidden1_Y0.weight)

        self.hidden2_Y0 = nn.Linear(in_features=200, out_features=200)
        nn.init.xavier_uniform_(self.hidden2_Y0.weight)

        self.out_Y0 = nn.Linear(in_features=200, out_features=2)
        nn.init.xavier_uniform_(self.out_Y0.weight)

    def forward(self, x, ps_score):
        if torch.cuda.is_available():
            x = x.float().cuda()
        else:
            x = x.float()

        if self.training:
            y1, y0 = self.__train_net(x, ps_score)
        else:
            y1, y0 = self.__eval_net(x)

        return y1, y0

    def __train_net(self, x, ps_score):
        entropy = Utils.get_shanon_entropy(ps_score.item())
        dropout_prob = Utils.get_dropout_probability(entropy, gama=1)

        # shared layers
        shared_mask = Utils.get_dropout_mask(dropout_prob, self.shared1(x))
        x = F.relu(shared_mask * self.shared1(x))
        x = F.relu(shared_mask * self.shared2(x))

        # potential outcome1 Y(1)
        y1_mask = Utils.get_dropout_mask(dropout_prob, self.hidden1_Y1(x))
        y1 = F.relu(y1_mask * self.hidden1_Y1(x))
        y1 = F.relu(y1_mask * self.hidden2_Y1(y1))
        y1 = self.out_Y1(y1)

        # potential outcome1 Y(0)
        y0_mask = Utils.get_dropout_mask(dropout_prob, self.hidden1_Y0(x))
        y0 = F.relu(y0_mask * self.hidden1_Y0(x))
        y0 = F.relu(y0_mask * self.hidden2_Y0(y0))
        y0 = self.out_Y0(y0)

        return y1, y0

    def __eval_net(self, x):
        # shared layers
        x = F.relu(self.shared1(x))
        x = F.relu(self.shared2(x))

        # potential outcome1 Y(1)
        y1 = F.relu(self.hidden1_Y1(x))
        y1 = F.relu(self.hidden2_Y1(y1))
        y1 = self.out_Y1(y1)

        # potential outcome1 Y(0)
        y0 = F.relu(self.hidden1_Y0(x))
        y0 = F.relu(self.hidden2_Y0(y0))
        y0 = self.out_Y0(y0)

        return y1, y0
