import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Sparse_Propensity_net import Sparse_Propensity_net
from Sparse_Propensity_net_shallow import Sparse_Propensity_net_shallow
from Utils import Utils


class Sparse_Propensity_score:
    def __init__(self):
        self.sparse_classifier_e2e = None

    @staticmethod
    def eval(eval_set, device, phase, sparse_classifier):
        print(".. Propensity score evaluation started using Sparse AE ..")
        sparse_classifier.eval()
        data_loader = torch.utils.data.DataLoader(eval_set, shuffle=False, num_workers=1)
        total_correct = 0
        eval_set_size = 0
        prop_score_list = []
        total_correct = 0

        for batch in data_loader:
            covariates, treatment = batch
            covariates = covariates.to(device)
            covariates = covariates[:, :-2]
            treatment = treatment.squeeze().to(device)

            eval_set_size += covariates.size(0)

            treatment_pred = sparse_classifier(covariates).to(device)
            treatment_pred_prob = F.softmax(treatment_pred, dim=1)
            # print(treatment_pred_prob)
            treatment_pred_prob = treatment_pred_prob.squeeze()
            prop_score_list.append(treatment_pred_prob[1].item())

            _, preds = torch.max(treatment_pred.data, 1)
            total_correct += torch.sum(preds == treatment.data)

        pred_accuracy = total_correct.item() / eval_set_size
        # print("Accuracy: {0}".format(total_correct))
        # print("correct: {0}, accuracy: {1}".format(total_correct, pred_accuracy))
        print(".. Propensity score evaluation completed Sparse AE ..")
        return prop_score_list

    def train(self, train_parameters, device, phase):
        print(".. Training started ..")
        epochs = train_parameters["epochs"]
        batch_size = train_parameters["batch_size"]
        lr = train_parameters["lr"]
        shuffle = train_parameters["shuffle"]
        train_set = train_parameters["train_set"]
        input_nodes = train_parameters["input_nodes"]

        sparsity_probability = train_parameters["sparsity_probability"],
        weight_decay = train_parameters["weight_decay"]
        BETA = train_parameters["weight_decay"]

        data_loader_train = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                        shuffle=shuffle, num_workers=1)

        print("##### train e2e #########")
        sparse_classifier = self.__end_to_end_train_SAE(phase, device, epochs,
                                                        data_loader_train,
                                                        lr,
                                                        weight_decay,
                                                        sparsity_probability, BETA,
                                                        train_set,
                                                        input_nodes)

        print("########## train layer wise all layer active ############")
        sae_classifier_stacked_all_layer_active = self.__layer_wise_train_SAE(phase, device, epochs,
                                                                              data_loader_train,
                                                                              lr, weight_decay,
                                                                              sparsity_probability, BETA,
                                                                              train_set, input_nodes,
                                                                              train_cur_layer=False)

        print("########## train layer wise only newly stacked layer active ############")
        sae_classifier_stacked_cur_layer_active = self.__layer_wise_train_SAE(phase, device, epochs,
                                                                              data_loader_train,
                                                                              lr, weight_decay,
                                                                              sparsity_probability, BETA,
                                                                              train_set, input_nodes,
                                                                              train_cur_layer=True)
        print("Training completed..")
        return sparse_classifier, sae_classifier_stacked_all_layer_active, sae_classifier_stacked_cur_layer_active

    def __end_to_end_train_SAE(self, phase, device, epochs, data_loader_train,
                               lr, weight_decay,
                               sparsity_probability, BETA,
                               train_set, input_nodes):
        model = Sparse_Propensity_net(training_mode=phase, device=device,
                                      input_nodes=input_nodes).to(device)
        sae_network_e2e = self.__train_SAE(epochs, device, data_loader_train,
                                           model, lr, weight_decay,
                                           sparsity_probability, BETA)

        sparse_classifier = nn.Sequential(*list(sae_network_e2e.children())[:-1])
        sparse_classifier.add_module('classifier',
                                     nn.Sequential(nn.Linear(in_features=10, out_features=2),
                                                   nn.LogSoftmax(dim=1)))
        sparse_classifier = sparse_classifier.to(device)
        sparse_classifier = self.__train_classifier(train_set, device, phase, sparse_classifier)
        return sparse_classifier

    def __layer_wise_train_SAE(self, phase, device, epochs, data_loader, lr, weight_decay,
                               sparsity_probability, BETA,
                               train_set, input_nodes, train_cur_layer=False):
        model = Sparse_Propensity_net_shallow(training_mode=phase, device=device, input_nodes=input_nodes).to(device)
        sae_network = self.__train_SAE(epochs, device, data_loader, model, lr, weight_decay,
                                       sparsity_probability, BETA)
        sae_network.encoder.add_module('New_Encoder_Layer',
                                       nn.Sequential(nn.Linear(in_features=20,
                                                               out_features=10)
                                                     , nn.Tanh()
                                                     # , nn.BatchNorm1d(num_features=10)
                                                     ))
        sae_network.encoder.add_module('New_Decoder_Layer',
                                       nn.Sequential(nn.Linear(in_features=10,
                                                               out_features=20)
                                                     , nn.Tanh()
                                                     # , nn.BatchNorm1d(num_features=20)
                                                     ))
        sae_network = sae_network.to(device)
        # print(sae_network)
        if train_cur_layer:
            sae_network.encoder[0].weight.requires_grad = False
            sae_network.encoder[0].bias.requires_grad = False
        sae_network = self.__train_SAE(epochs, device, data_loader,
                                       sae_network, lr, weight_decay,
                                       sparsity_probability, BETA)

        classifier = nn.Sequential(*list(sae_network.children())[:-1])
        net = classifier
        classifier = nn.Sequential(*list(net[0].children())[:-1])
        sae_classifier = classifier
        sae_classifier.add_module('classifier',
                                  nn.Sequential(nn.Linear(in_features=10, out_features=2),
                                                nn.LogSoftmax(dim=1)))
        sae_classifier = sae_classifier.to(device)
        if train_cur_layer:
            sae_classifier[0].weight.requires_grad = True
            sae_classifier[0].bias.requires_grad = True
        sae_classifier = self.__train_classifier(train_set, device, phase,
                                                 sae_classifier)

        # print(sae_classifier)
        return sae_classifier

    def __train_SAE(self, epochs, device, data_loader, network, lr, weight_decay=0.0005,
                    sparsity_probability=0.03,
                    BETA=0.001):
        print("----- Training SAE -----")

        criterion = nn.MSELoss()
        # the optimizer
        optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)
        model_children = list(network.children())

        for epoch in range(epochs):
            epoch += 1
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
                sparsity = self.__sparse_loss(sparsity_probability, covariates, model_children, device)
                loss = mse_loss + BETA * sparsity
                # loss = mse_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            epoch_loss = total_loss / counter
            if epoch % 100 == 0:
                print("Epoch: {0}, loss: {1}".
                      format(epoch, epoch_loss))

        return network

    def __train_classifier(self, train_set, device, phase, sparse_classifier):
        # print(device)
        print(".. Propensity score evaluation started using Sparse AE..")

        data_loader_train = torch.utils.data.DataLoader(train_set, batch_size=32,
                                                        shuffle=True, num_workers=1)

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(sparse_classifier.parameters(), lr=0.01)
        for epoch in range(50):
            epoch += 1
            sparse_classifier.train()
            total_loss = 0
            total_correct = 0
            train_set_size = 0

            for batch in data_loader_train:
                covariates, treatment = batch
                covariates = covariates.to(device)
                train_set_size += covariates.size(0)

                treatment = treatment.squeeze().to(device)
                covariates = covariates[:, :-2]
                treatment_pred = sparse_classifier(covariates).to(device)

                loss = criterion(treatment_pred, treatment)

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                total_loss += loss.item()
                _, preds = torch.max(treatment_pred.data, 1)
                total_correct += torch.sum(preds == treatment.data)

            pred_accuracy = total_correct.item() / train_set_size
            if epoch % 25 == 0:
                print("Epoch: {0}, loss: {1}, correct: {2}/{3}, accuracy: {4}".
                      format(epoch, total_loss, total_correct, train_set_size, pred_accuracy))

        return sparse_classifier

    @staticmethod
    def __sparse_loss(sparsity_probability, covariates, model_children, device):
        values = covariates
        loss = 0
        encoder = list(model_children[0].children())
        if len(encoder) == 2:
            # only last encoder layer
            values = model_children[0](values)
        elif len(encoder) == 4:
            # only last encoder layer
            values = encoder[2](encoder[1](encoder[0](values)))

        loss += Utils.KL_divergence(sparsity_probability, values, device)
        return loss
