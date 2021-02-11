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

import torch.nn as nn
import torch.nn.functional as F


# phase = ["train", "eval"]
class Propensity_net_NN(nn.Module):
    def __init__(self, phase, input_nodes):
        super(Propensity_net_NN, self).__init__()
        self.phase = phase
        self.fc1 = nn.Linear(in_features=input_nodes, out_features=25)
        # nn.init.xavier_uniform_(self.fc1.weight)

        self.fc2 = nn.Linear(in_features=25, out_features=25)
        # nn.init.xavier_uniform_(self.fc2.weight)

        self.ps_out = nn.Linear(in_features=25, out_features=2)

    def forward(self, x):
        # if torch.cuda.is_available():
        #     x = x.float().cuda()
        # else:
        #     x = x.float()

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.ps_out(x)
        if self.phase == "eval":
            return F.softmax(x, dim=1)
        else:
            return x
