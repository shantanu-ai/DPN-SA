import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from Utils import Utils

csv_path = "/Users/shantanughosh/Desktop/Shantanu_MS/Research/Mattia_Prosperi/Propensity_Dropout/Dataset/ihdp_sample.csv"

from dataloader import DataLoader

dL = DataLoader()
split_size = 0.8
np_covariates_X_train, np_covariates_X_test, \
np_covariates_Y_train, np_covariates_Y_test = dL.preprocess_data_from_csv(csv_path, split_size=0.8)
norm_X_train = np_covariates_X_train / np.linalg.norm(np_covariates_X_train)
ps_train_set = dL.convert_to_tensor(np_covariates_X_train, np_covariates_Y_train)
device = Utils.get_device()

print(device)

from Sparse_Propensity_score import Sparse_Propensity_score

train_parameters_SAE = {
    "epochs": 100,
    "lr": 0.001,
    "batch_size": 32,
    "shuffle": True,
    "train_set": ps_train_set,
    "sparsity_probability": 0.5,
    "weight_decay": 0.0003,
    "BETA": 3,
    "model_save_path": "./Propensity_Model/SAE_PS_model_iter_id_" + str(1) + "_epoch_{0}_lr_{1}.pth"
}

ps_net_SAE = Sparse_Propensity_score()
print("############### Propensity Score SAE net Training ###############")
network = ps_net_SAE.train(train_parameters_SAE, device, phase="train")
data_loader = torch.utils.data.DataLoader(ps_train_set, shuffle=False, num_workers=4)
for batch in data_loader:
    n = network
    n.eval()
    covariates, treatment = batch
    print(covariates)
    covariates = covariates[:, :-2]
    _c1 = n(covariates)
    print(_c1)
    break

classifier = nn.Sequential(*list(network.children())[:-1])
classifier.add_module('classifier',
                      nn.Sequential(nn.Linear(in_features=10, out_features=2), nn.LogSoftmax()))
print(classifier)


def train(eval_parameters, device, phase):
    print(".. Propensity score evaluation started using Sparse AE..")
    train_set = eval_parameters["eval_set"]
    for n in classifier.parameters():
        print(n)
    # classifier.train()
    # classifier[0][0].weight.requires_grad = False
    # classifier[0][0].bias.requires_grad = False
    # classifier[0][2].weight.requires_grad = False
    # classifier[0][2].bias.requires_grad = False

    print("chutiya")
    total_correct = 0
    eval_set_size = 0
    prop_score_list = []

    low_dim_list = []
    data_loader = torch.utils.data.DataLoader(train_set, batch_size=32,
                                              shuffle=True, num_workers=4)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.01)
    for epoch in range(100):
        classifier.train()
        total_loss = 0
        total_correct = 0
        train_set_size = 0

        for batch in data_loader:
            covariates, treatment = batch
            covariates = covariates.to(device)
            train_set_size += covariates.size(0)

            #             print(covariates.size())
            #             print(treatment.size())

            treatment = treatment.squeeze().to(device)
            covariates = covariates[:, :-2]

            treatment_pred = classifier(covariates)
            # print(treatment_pred)
            # print(F.softmax(treatment_pred))

            loss = criterion(treatment_pred, treatment).to(device)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(treatment_pred.data, 1)
            total_correct += torch.sum(preds == treatment.data)

        pred_accuracy = total_correct.item() / train_set_size
        print("Epoch: {0}, loss: {1}, correct: {2}/{3}, accuracy: {4}".
              format(epoch, total_loss, total_correct, train_set_size, pred_accuracy))
    #     for batch in eval_set:
    #         covariates, treatment = batch
    #         covariates = covariates.to(device)
    #         eval_set_size += covariates.size(0)
    #         treatment_pred = network(covariates)
    #         total_correct += Utils.get_num_correct(treatment_pred, treatment)

    #         treatment_pred_prob = F.softmax(treatment_pred, dim=1)
    #         treatment_pred_prob = treatment_pred_prob.squeeze()
    #         prop_score_list.append(treatment_pred_prob[1].item())

    return classifier


eval_parameters_SAE_1 = {
    # "eval_set": ps_low
    "eval_set": ps_train_set

}
print(device)
classifier = train(eval_parameters_SAE_1, device, phase="eval")


