import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Propensity_net_SAE import Propensity_netSAE
from Utils import Utils


class Propensity_socre_SAE:
    def train(self, train_parameters, device, phase):
        print(".. Training started ..")
        epochs = train_parameters["epochs"]
        batch_size = train_parameters["batch_size"]
        lr = train_parameters["lr"]
        shuffle = train_parameters["shuffle"]
        model_save_path = train_parameters["model_save_path"].format(epochs, lr)
        train_set = train_parameters["train_set"]

        sparsity_probability = train_parameters["sparsity_probability"],
        weight_decay = train_parameters["weight_decay"]
        BETA = train_parameters["weight_decay"]
        print("Saved model path: {0}".format(model_save_path))

        data_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                  shuffle=shuffle, num_workers=4)
        sae_network = self.train_SAE(phase, epochs, device, data_loader, lr, weight_decay, sparsity_probability, BETA,
                                     "SAE")
        init_weights_dict = {
            "encoder1_weight": copy.deepcopy(sae_network.encoder1.weight.data),
            "encoder1_bias": copy.deepcopy(sae_network.encoder1.bias.data),
            "encoder2_weight": copy.deepcopy(sae_network.encoder2.weight.data),
            "encoder2_bias": copy.deepcopy(sae_network.encoder2.bias.data),
            "encoder3_weight": copy.deepcopy(sae_network.encoder3.weight.data),
            "encoder3_bias": copy.deepcopy(sae_network.encoder3.bias.data)
        }

        classifier_network = self.train_classifier(init_weights_dict, epochs, device, data_loader, lr, "Classifier")

        print("Saving model..")
        torch.save(classifier_network.state_dict(), model_save_path)

    def train_SAE(self, phase, epochs, device, data_loader, lr, weight_decay=0.0005,
                  sparsity_probability=0.05,
                  BETA=0.001, training_mode="SAE"):
        """
        :param phase:
        :param epochs:
        :param device:
        :param data_loader:
        :param lr:
        :param weight_decay:
        :param sparsity_probability:
        :param BETA:
        :param training_mode: "SAE" for autoencoder training
        :return:
        """
        print("----- Training SAE -----")
        network = Propensity_netSAE(training_mode=training_mode, device=device).to(device)
        criterion = nn.MSELoss()
        # the optimizer
        optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)
        model_children = list(network.children())

        for epoch in range(epochs):
            network.train()
            total_loss = 0
            counter = 0
            train_set_size = 0

            for batch in data_loader:
                counter += 1
                covariates, _ = batch
                covariates = covariates.to(device)
                covariates = covariates[:, :-2]
                train_set_size += covariates.size(0)

                treatment_pred = network(covariates)
                mse_loss = criterion(treatment_pred, covariates)
                sparsity = self.sparse_loss(sparsity_probability, covariates, model_children, device)
                loss = mse_loss + BETA * sparsity

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            epoch_loss = total_loss / counter
            print("Epoch: {0}, loss: {1}".
                  format(epoch, epoch_loss))

            return network

    @staticmethod
    def sparse_loss(sparsity_probability, covariates, model_children, device):
        values = covariates
        loss = 0
        for i in range(len(model_children)):
            values = model_children[i](values)
            loss += Utils.KL_divergence(sparsity_probability, values, device)
        return loss

    def train_classifier(self, init_weights, epochs, device, data_loader, lr, training_mode="Classifier"):
        """

        :param init_weights:
        :param epochs:
        :param device:
        :param data_loader:
        :param lr:
        :param training_mode: "Classifier" for classifier training
        :return:
        """
        print("----- Training classifier -----")
        network = Propensity_netSAE(training_mode=training_mode, device=device,
                                    init_weight_dict=init_weights).to(device)
        optimizer = optim.Adam(network.parameters(), lr=lr)
        for epoch in range(epochs):
            network.train()
            total_loss = 0
            total_correct = 0
            train_set_size = 0

            for batch in data_loader:
                covariates, treatment = batch
                covariates = covariates.to(device)
                treatment = treatment.squeeze().to(device)

                covariates = covariates[:, :-2]
                train_set_size += covariates.size(0)

                treatment_pred = network(covariates)
                loss = F.cross_entropy(treatment_pred, treatment).to(device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_correct += Utils.get_num_correct(treatment_pred, treatment)

            pred_accuracy = total_correct / train_set_size
            print("Epoch: {0}, loss: {1}, correct: {2}/{3}, accuracy: {4}".
                  format(epoch, total_loss, total_correct, train_set_size, pred_accuracy))
            return network

    @staticmethod
    def eval(eval_parameters, device, phase, training_mode="Classifier"):
        print(".. Propensity score evaluation started using SAE ..")
        eval_set = eval_parameters["eval_set"]
        model_path = eval_parameters["model_path"]
        network = Propensity_netSAE(training_mode=training_mode, device=device).to(device)
        network.load_state_dict(torch.load(model_path, map_location=device))
        network.eval()
        data_loader = torch.utils.data.DataLoader(eval_set, shuffle=False, num_workers=4)
        total_correct = 0
        eval_set_size = 0
        prop_score_list = []
        for batch in data_loader:
            covariates, treatment = batch
            covariates = covariates.to(device)
            covariates = covariates[:, :-2]
            treatment = treatment.squeeze().to(device)

            treatment_pred = network(covariates)
            treatment_pred = F.softmax(treatment_pred, dim=1)
            treatment_pred = treatment_pred.squeeze()
            prop_score_list.append(treatment_pred[1].item())

        print(".. Propensity score evaluation completed using SAE ..")
        return prop_score_list
