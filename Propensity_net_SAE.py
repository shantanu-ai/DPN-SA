import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class Propensity_netSAE(nn.Module):
    def __init__(self, training_mode, device, init_weight_dict=None):
        """
        Constructor
        training_mode: training type of the network
                       Can hold two values
                       1. SAE - for sparse autoencoder based training
                       2. Classifier - for calculating propensity score
        """
        super(Propensity_netSAE, self).__init__()
        self.training_mode = training_mode

        # encoder
        self.encoder1 = nn.Linear(in_features=25, out_features=25)
        self.encoder2 = nn.Linear(in_features=25, out_features=25)
        self.encoder3 = nn.Linear(in_features=25, out_features=10)

        if self.training_mode == "SAE":
            # encoder initialization
            nn.init.xavier_uniform_(self.encoder1.weight)
            nn.init.xavier_uniform_(self.encoder2.weight)
            nn.init.xavier_uniform_(self.encoder3.weight)
            # decoder
            self.decoder1 = nn.Linear(in_features=10, out_features=25)
            nn.init.xavier_uniform_(self.decoder1.weight)
            self.decoder2 = nn.Linear(in_features=25, out_features=25)
            nn.init.xavier_uniform_(self.decoder2.weight)
            self.decoder3 = nn.Linear(in_features=25, out_features=25)
            nn.init.xavier_uniform_(self.decoder3.weight)
        elif self.training_mode == "Classifier":
            # encoder initialization with the autoencoder parameters
            self.initialize_parameters(init_weight_dict, device)

            # classifier
            self.classifier = nn.Linear(in_features=10, out_features=2)
            nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, x):
        if torch.cuda.is_available():
            x = x.float().cuda()
        else:
            x = x.float()

        # encoding
        x = F.relu(self.encoder1(x))
        x = F.relu(self.encoder2(x))
        x = F.relu(self.encoder3(x))
        if self.training_mode == "SAE":
            # decoding
            x = F.relu(self.decoder1(x))
            x = F.relu(self.decoder2(x))
            x = F.relu(self.decoder3(x))
        elif self.training_mode == "Classifier":
            # classifier
            x = self.classifier(x)
        return x

    def initialize_parameters(self, init_weight_dict, device):
        if init_weight_dict:
            print("Initializing parameters from SAE..")
            self.encoder1.weight = torch.nn.Parameter(init_weight_dict["encoder1_weight"]).to(device)
            self.encoder1.bias = torch.nn.Parameter(init_weight_dict["encoder1_bias"]).to(device)
            self.encoder2.weight = torch.nn.Parameter(init_weight_dict["encoder2_weight"]).to(device)
            self.encoder2.bias = torch.nn.Parameter(init_weight_dict["encoder2_bias"]).to(device)
            self.encoder3.weight = torch.nn.Parameter(init_weight_dict["encoder3_weight"]).to(device)
            self.encoder3.bias = torch.nn.Parameter(init_weight_dict["encoder3_bias"]).to(device)
