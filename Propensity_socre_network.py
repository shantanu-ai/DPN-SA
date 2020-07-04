import torch
import torch.nn.functional as F
import torch.optim as optim

from Propensity_net_NN import Propensity_net_NN
from Utils import Utils


class Propensity_socre_network:
    def train(self, train_parameters, device, phase):
        print(".. Training started ..")
        epochs = train_parameters["epochs"]
        batch_size = train_parameters["batch_size"]
        lr = train_parameters["lr"]
        shuffle = train_parameters["shuffle"]
        model_save_path = train_parameters["model_save_path"].format(epochs, lr)
        train_set = train_parameters["train_set"]
        input_nodes = train_parameters["input_nodes"]
        print("Saved model path: {0}".format(model_save_path))

        network = Propensity_net_NN(phase, input_nodes).to(device)

        data_loader = torch.utils.data.DataLoader(train_set, batch_size=32,
                                                  shuffle=shuffle, num_workers=4)

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

        print("Saving model..")
        torch.save(network.state_dict(), model_save_path)

    @staticmethod
    def eval(eval_parameters, device, phase):
        print(".. Propensity score evaluation started using NN..")
        eval_set = eval_parameters["eval_set"]
        model_path = eval_parameters["model_path"]
        input_nodes = eval_parameters["input_nodes"]

        network = Propensity_net_NN(phase, input_nodes).to(device)
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

            eval_set_size += covariates.size(0)

            treatment_pred = network(covariates)
            total_correct += Utils.get_num_correct(treatment_pred, treatment)

            treatment_pred = treatment_pred.squeeze()
            prop_score_list.append(treatment_pred[1].item())

        # pred_accuracy = total_correct / eval_set_size
        # print("correct: {0}/{1}, accuracy: {2}".
        #           format(total_correct, eval_set_size, pred_accuracy))

        print(".. Propensity score evaluation completed using NN..")
        return prop_score_list

    @staticmethod
    def eval_return_complete_list(eval_parameters, device, phase):
        print(".. Propensity score evaluation started using NN..")
        eval_set = eval_parameters["eval_set"]
        input_nodes = eval_parameters["input_nodes"]
        model_path = eval_parameters["model_path"]
        network = Propensity_net_NN(phase, input_nodes).to(device)
        network.load_state_dict(torch.load(model_path, map_location=device))
        network.eval()
        data_loader = torch.utils.data.DataLoader(eval_set, shuffle=False, num_workers=4)
        total_correct = 0
        eval_set_size = 0
        prop_score_list = []
        for batch in data_loader:
            prop_dict = {}
            covariates, treatment = batch
            covariates = covariates.to(device)
            covariates = covariates[:, :-2]
            treatment = treatment.squeeze().to(device)

            eval_set_size += covariates.size(0)

            treatment_pred = network(covariates)
            total_correct += Utils.get_num_correct(treatment_pred, treatment)

            treatment_pred = treatment_pred.squeeze()
            prop_dict["treatment"] = treatment.item()
            prop_dict["prop_score"] = treatment_pred[1].item()
            prop_score_list.append(prop_dict)

        # pred_accuracy = total_correct / eval_set_size
        # print("correct: {0}/{1}, accuracy: {2}".
        #           format(total_correct, eval_set_size, pred_accuracy))

        print(".. Propensity score evaluation completed using NN..")
        return prop_score_list
