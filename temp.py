import statistics
from collections import OrderedDict

import numpy as np

from DCN_network import DCN_network
from Propensity_score_LR import Propensity_socre_LR
from Propensity_socre_network import Propensity_socre_network
from Sparse_Propensity_score import Sparse_Propensity_score
from Utils import Utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloader import DataLoader

device = Utils.get_device()
dL = DataLoader()
csv_path = "Dataset/ihdp_sample.csv"
split_size = 0.8
np_covariates_X_train, np_covariates_X_test, np_covariates_Y_train, np_covariates_Y_test = \
                dL.preprocess_data_from_csv(csv_path, split_size)

phase = "train"

ps_train_set = dL.convert_to_tensor(np_covariates_X_train, np_covariates_Y_train)
train_parameters = {
            "epochs": 200,
            "lr": 0.0001,
            "batch_size": 32,
            "shuffle": True,
            "train_set": ps_train_set,
            "sparsity_probability": 0.08,
            "weight_decay": 0.0003,
            "BETA": 0.4
        }


ps_net_SAE = Sparse_Propensity_score()
sparse_classifier, sae_classifier_stacked_all_layer_active, \
        sae_classifier_stacked_cur_layer_active = ps_net_SAE.train(train_parameters, device, phase="train")

np_covariates_X, np_treatment_Y = \
                dL.prep_process_all_data(csv_path)


ps_train_set = dL.convert_to_tensor(np_covariates_X, np_treatment_Y)

ps_score_list_SAE = ps_net_SAE.eval(ps_train_set, device, phase="eval",
                                            sparse_classifier=sparse_classifier)

print(len(ps_score_list_SAE))