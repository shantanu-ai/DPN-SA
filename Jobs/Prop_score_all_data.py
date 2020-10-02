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

from Utils import Utils
from dataloader import DataLoader
from shallow_train import shallow_train

device = Utils.get_device()
dL = DataLoader()
csv_path = "Dataset/ihdp_sample.csv"
split_size = 0.8
np_covariates_X, np_treatment_Y = \
    dL.prep_process_all_data(csv_path)

ps_train_set = dL.convert_to_tensor(np_covariates_X, np_treatment_Y)

phase = "train"

# LR Lasso
# ps_score_list_LR, LR_model = Propensity_socre_LR.train(np_covariates_X,
#                                                        np_treatment_Y,
#                                                        regularized=True)

# LR
# ps_score_list_LR, LR_model = Propensity_socre_LR.train(np_covariates_X,
#                                                        np_treatment_Y)


# 25-1-25

train_parameters_SAE = {
    'epochs': 2000,
    'lr': 0.001,
    "batch_size": 32,
    "shuffle": True,
    "train_set": ps_train_set,
    "sparsity_probability": 0.8,
    "weight_decay": 0.0003,
    "BETA": 0.1,
    "input_nodes": 25
}
ps_net_SAE = shallow_train()
print("############### Propensity Score SAE net Training ###############")
sparse_classifier = ps_net_SAE.train(train_parameters_SAE, device, phase="train")

ps_score_list_SAE = ps_net_SAE.eval(ps_train_set, device, phase="eval",
                                            sparse_classifier=sparse_classifier)

print("25-10-25 Prop list")
print("--" * 20)
for ps in ps_score_list_SAE:
    print(ps)


# SAE

train_parameters_SAE = {
    "epochs": 2000,
    "lr": 0.001,
    "batch_size": 32,
    "shuffle": True,
    "train_set": ps_train_set,
    "sparsity_probability": 0.8,
    "weight_decay": 0.0003,
    "BETA": 0.1,
    "input_nodes": 25
}
# ps_net_SAE = Sparse_Propensity_score()
#
# sparse_classifier, sae_classifier_stacked_all_layer_active, \
# sae_classifier_stacked_cur_layer_active = ps_net_SAE.train(train_parameters_SAE, device, phase="train")
#
# ps_score_list_train_SAE_e2e = ps_net_SAE.eval(ps_train_set, device, phase="eval",
#                                               sparse_classifier=sparse_classifier)
#
# ps_score_list_train_SAE_all_layer_active = ps_net_SAE.eval(ps_train_set, device, phase="eval",
#                                                            sparse_classifier=sae_classifier_stacked_all_layer_active)
#
# ps_score_list_train_SAE_cur_layer_active = ps_net_SAE.eval(ps_train_set, device, phase="eval",
#                                                            sparse_classifier=sae_classifier_stacked_cur_layer_active)
#
# print("e2e Prop list")
# print("--" * 20)
# for ps in ps_score_list_train_SAE_e2e:
#     print(ps)
#
# print("all layer Prop list")
# print("--" * 20)
# for ps in ps_score_list_train_SAE_all_layer_active:
#     print(ps)
#
# print("cur layer  Prop list")
# print("--" * 20)
# for ps in ps_score_list_train_SAE_cur_layer_active:
#     print(ps)

# NN

train_parameters_NN = {
    "epochs": 50,
    "lr": 0.001,
    "batch_size": 32,
    "shuffle": True,
    "train_set": ps_train_set,
    "model_save_path": "./Propensity_Model/NN_PS_model_iter_id_"
                       + str(1) + "_epoch_{0}_lr_{1}.pth",
    "input_nodes": 25
}
# # ps using NN
# ps_net_NN = Propensity_socre_network()
# print("############### Propensity Score neural net Training ###############")
# ps_net_NN.train(train_parameters_NN, device, phase="train")
# # eval
# eval_parameters_train_NN = {
#     "eval_set": ps_train_set,
#     "model_path": "./Propensity_Model/NN_PS_model_iter_id_{0}_epoch_{1}_lr_0.001.pth"
#         .format(1, train_parameters_NN["epochs"]),
#     "input_nodes": 25
# }
# ps_score_list_train_NN = ps_net_NN.eval(eval_parameters_train_NN, device, phase="eval")
#
# for ps in ps_score_list_train_NN:
#     print(ps)
