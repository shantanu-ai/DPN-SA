import numpy
from matplotlib import pyplot
import pandas as pd
import os
from Propensity_socre_network import Propensity_socre_network
from Utils import Utils
from dataloader import DataLoader


def draw(treated_ps_list, control_ps_list, bins1):
    pyplot.hist(treated_ps_list, bins1, alpha=0.5, label='treated')
    pyplot.hist(control_ps_list, bins1, alpha=0.5, label='control')

    pyplot.legend(loc='upper right')
    pyplot.show()


csv_path = "Dataset/ihdp_sample.csv"

# 139 treated
# 747 - 139 =  608 control
# 747 total

split_size = 0.8
device = Utils.get_device()

dL = DataLoader()
np_covariates_X_train, np_covariates_X_test, np_covariates_Y_train, np_covariates_Y_test = \
    dL.preprocess_data_from_csv(csv_path, split_size)

ps_train_set = dL.convert_to_tensor(np_covariates_X_train, np_covariates_Y_train)

train_parameters_NN = {
    "epochs": 75,
    "lr": 0.001,
    "batch_size": 32,
    "shuffle": True,
    "train_set": ps_train_set,
    "model_save_path": "./Propensity_Model/NN_PS_model_iter_id_"
                       + str(1) + "_epoch_{0}_lr_{1}.pth"
}
# ps using NN
ps_net_NN = Propensity_socre_network()
print("############### Propensity Score neural net Training ###############")
ps_net_NN.train(train_parameters_NN, device, phase="train")

# eval
eval_parameters_NN = {
    "eval_set": ps_train_set,
    "model_path": "./Propensity_Model/NN_PS_model_iter_id_{0}_epoch_75_lr_0.001.pth"
        .format(1)
}

ps_score_list_NN = ps_net_NN.eval_return_complete_list(eval_parameters_NN, device, phase="eval")
treated_ps_list = [d["prop_score"] for d in ps_score_list_NN if d['treatment'] == 1]
control_ps_list = [d["prop_score"] for d in ps_score_list_NN if d['treatment'] == 0]
for ps_dict in treated_ps_list:
    print(ps_dict)

print("--------------")
for ps_dict in control_ps_list:
    print(ps_dict)


print("treated: " + str(len(treated_ps_list)))
print("control: " + str(len(control_ps_list)))
print("total: " + str(len(treated_ps_list) + len(control_ps_list)))

bins1 = numpy.linspace(0, 1, 100)
bins2 = numpy.linspace(0, 0.2, 100)
bins3 = numpy.linspace(0.2, 0.5, 100)
bins4 = numpy.linspace(0.5, 1, 100)

draw(treated_ps_list, control_ps_list, bins1)
draw(treated_ps_list, control_ps_list, bins2)
draw(treated_ps_list, control_ps_list, bins3)
draw(treated_ps_list, control_ps_list, bins4)
