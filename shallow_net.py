import torch
import torch.nn as nn
import torch.utils.data


class shallow_net(nn.Module):
    def __init__(self, training_mode, device):
        print("Training mode: {0}".format(training_mode))
        super(shallow_net, self).__init__()
        self.training_mode = training_mode

        # encoder
        self.encoder = nn.Sequential(nn.Linear(in_features=25, out_features=10),
                                     nn.Tanh()
                                     )

        if self.training_mode == "train":
            # decoder
            self.decoder = nn.Sequential(nn.Linear(in_features=10, out_features=25),
                                         nn.Tanh(),
                                         nn.Linear(in_features=25, out_features=25))

    def forward(self, x):
        if torch.cuda.is_available():
            x = x.float().cuda()
        else:
            x = x.float()

        # encoding
        x = self.encoder(x)

        if self.training_mode == "train":
            # decoding
            x = self.decoder(x)
        return x
