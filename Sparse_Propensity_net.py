import torch
import torch.nn as nn
import torch.utils.data


class Sparse_Propensity_net(nn.Module):
    def __init__(self, training_mode, device):
        print("Training mode: {0}".format(training_mode))
        super(Sparse_Propensity_net, self).__init__()
        self.training_mode = training_mode

        # encoder
        self.encoder = nn.Sequential(nn.Linear(in_features=25, out_features=20),
                                     nn.Tanh(),
                                     nn.Linear(in_features=20, out_features=10),
                                     nn.Tanh())

        # nn.init.xavier_uniform_(self.encoder[0].weight)
        # nn.init.xavier_uniform_(self.encoder[2].weight)
        # self.encoder1 = nn.Linear(in_features=25, out_features=10)
        # nn.init.xavier_uniform_(self.encoder1.weight)
        #
        # self.encoder2 = nn.Linear(in_features=10, out_features=5)
        # nn.init.xavier_uniform_(self.encoder2.weight)

        # self.encoder3 = nn.Linear(in_features=20, out_features=10)
        # nn.init.xavier_uniform_(self.encoder3.weight)

        if self.training_mode == "train":
            # decoder
            self.decoder = nn.Sequential(nn.Linear(in_features=10, out_features=20),
                                         nn.Tanh(),
                                         nn.Linear(in_features=20, out_features=25),
                                         nn.Tanh(), nn.Linear(in_features=25, out_features=25))
            # nn.init.xavier_uniform_(self.decoder[0].weight)
            # nn.init.xavier_uniform_(self.decoder[2].weight)
            # self.decoder1 = nn.Linear(in_features=5, out_features=10)
            # # nn.init.xavier_uniform_(self.decoder1.weight)
            # self.decoder2 = nn.Linear(in_features=10, out_features=25)
            # nn.init.xavier_uniform_(self.decoder2.weight)
            # self.decoder3 = nn.Linear(in_features=25, out_features=25)
            # nn.init.xavier_uniform_(self.decoder3.weight)
        # elif self.training_mode == "eval":
        #     # classifier
        #     self.classifier = nn.Linear(in_features=10, out_features=2)
        #     nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, x):
        if torch.cuda.is_available():
            x = x.float().cuda()
        else:
            x = x.float()

        # encoding
        x = self.encoder(x)
        # x = F.sigmoid(self.encoder1(x))
        # x = F.sigmoid(self.encoder2(x))
        # x = F.relu(self.encoder3(x))
        if self.training_mode == "train":
            # decoding
            x = self.decoder(x)
            # x = F.sigmoid(self.decoder1(x))
            # x = F.sigmoid(self.decoder2(x))
            # x = F.relu(self.decoder3(x))
        # elif self.training_mode == "eval":
        #     # classifier
        #     x = self.classifier(x)
        return x
