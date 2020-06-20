import matplotlib.pyplot as plt

from Propensity_score_LR import Propensity_socre_LR
from Propensity_socre_network import Propensity_socre_network
from Sparse_Propensity_score import Sparse_Propensity_score
from Utils import Utils
from dataloader import DataLoader


class Graphs:
    def draw_scatter_plots(self):
        device = Utils.get_device()
        csv_path = "Dataset/ihdp_sample.csv"
        dL = DataLoader()
        split_size = 0.8
        # np_covariates_X, np_covariates_Y = dL.preprocess_for_graphs(csv_path)
        np_covariates_X_train, np_covariates_X_test, np_covariates_Y_train, np_covariates_Y_test = \
            dL.preprocess_data_from_csv(csv_path, split_size)
        print(np_covariates_Y_test.shape)

        # ps_train_set = dL.convert_to_tensor(np_covariates_X, np_covariates_Y)
        ps_train_set = dL.convert_to_tensor(np_covariates_X_train, np_covariates_Y_train)
        ps_test_set = dL.convert_to_tensor(np_covariates_X_test, np_covariates_Y_test)
        ps_list_nn = self.__train_propensity_net_NN(ps_train_set, device, ps_test_set)
        ps_list_SAE = self.__train_propensity_net_SAE(ps_train_set, device, ps_test_set)
        ps_list_LR = self.__train_propensity_net_LR(np_covariates_X_train, np_covariates_Y_train,
                                                    np_covariates_X_test, np_covariates_Y_test)
        ps_list_LR_lasso = self.__train_propensity_net_LR_Lasso(np_covariates_X_train, np_covariates_Y_train,
                                                                np_covariates_X_test, np_covariates_Y_test)

        self.draw_ps_scatter_plots_all(ps_list_nn, ps_list_SAE, ps_list_LR, ps_list_LR_lasso)

        self.draw_ps_scatter_plots(ps_list_nn, "PD")
        self.draw_ps_scatter_plots(ps_list_SAE, "SAE")
        self.draw_ps_scatter_plots(ps_list_LR, "LR")
        self.draw_ps_scatter_plots(ps_list_LR_lasso, "LR Lasso")

        self.draw_ps_scatter_plots_sae(ps_list_nn, ps_list_SAE, x_label="PD", y_label="SAE")
        self.draw_ps_scatter_plots_sae(ps_list_LR, ps_list_SAE, x_label="LR", y_label="SAE")
        self.draw_ps_scatter_plots_sae(ps_list_LR_lasso, ps_list_SAE, x_label="LR_Lasso", y_label="SAE")

    @staticmethod
    def __train_propensity_net_NN(ps_train_set, device, ps_test_set):
        train_parameters_NN = {
            "epochs": 100,
            "lr": 0.001,
            "batch_size": 32,
            "shuffle": True,
            "train_set": ps_train_set,
            "model_save_path": "./Propensity_Model/Graph_NN_PS_model_epoch_{0}_lr_{1}.pth"
        }
        # ps using NN
        ps_net_NN = Propensity_socre_network()
        print("############### Propensity Score neural net Training ###############")
        ps_net_NN.train(train_parameters_NN, device, phase="train")

        # eval
        eval_parameters_NN = {
            "eval_set": ps_test_set,
            "model_path": "./Propensity_Model/Graph_NN_PS_model_epoch_100_lr_0.001.pth"
        }

        ps_score_list_NN = ps_net_NN.eval(eval_parameters_NN, device, phase="eval")
        return ps_score_list_NN

    @staticmethod
    def __train_propensity_net_SAE(ps_train_set, device, ps_test_set):
        # !!! best parameter list
        train_parameters_SAE = {
            "epochs": 200,
            "lr": 0.0001,
            "batch_size": 32,
            "shuffle": True,
            "train_set": ps_train_set,
            "sparsity_probability": 0.08,
            "weight_decay": 0.0003,
            "BETA": 0.4,
            "model_save_path": "./Propensity_Model/SAE_PS_model_iter_id_epoch_{0}_lr_{1}.pth"
        }

        ps_net_SAE = Sparse_Propensity_score()
        print("############### Propensity Score SAE net Training ###############")
        sparse_classifier = ps_net_SAE.train(train_parameters_SAE, device, phase="train")

        # eval propensity network using SAE
        ps_score_list_SAE = ps_net_SAE.eval(ps_test_set, device, phase="eval",
                                            sparse_classifier=sparse_classifier)
        return ps_score_list_SAE

    @staticmethod
    def __train_propensity_net_LR(np_covariates_X_train, np_covariates_Y_train,
                                  np_covariates_X_test, np_covariates_Y_test):
        # eval propensity network using Logistic Regression
        _, LR_model = Propensity_socre_LR.train(np_covariates_X_train,
                                                np_covariates_Y_train)
        ps_score_list_LR = Propensity_socre_LR.test(np_covariates_X_test,
                                                    np_covariates_Y_test,
                                                    log_reg=LR_model)
        return ps_score_list_LR

    @staticmethod
    def __train_propensity_net_LR_Lasso(np_covariates_X_train, np_covariates_Y_train,
                                        np_covariates_X_test, np_covariates_Y_test):
        # eval propensity network using Logistic Regression Lasso
        _, LR_model_lasso = Propensity_socre_LR.train(np_covariates_X_train,
                                                      np_covariates_Y_train,
                                                      regularized=True)
        ps_score_list_LR_lasso = Propensity_socre_LR.test(np_covariates_X_test,
                                                          np_covariates_Y_test,
                                                          log_reg=LR_model_lasso)
        return ps_score_list_LR_lasso

    @staticmethod
    def draw_ps_scatter_plots_all(ps_list_nn, ps_list_SAE, ps_list_LR, ps_list_LR_lasso):
        X = list(range(1, len(ps_list_nn) + 1))
        Y_nn = ps_list_nn
        y_SAE = ps_list_SAE
        y_LR = ps_list_LR
        y_LR_lasso = ps_list_LR_lasso

        fig = plt.figure()
        plt.scatter(X, Y_nn, c="green", alpha=1, label="PD")
        plt.scatter(X, y_LR, c="yellow", alpha=1, label="LR")
        plt.scatter(X, y_LR_lasso, c="blue", alpha=1, label="LR Lasso")
        plt.scatter(X, y_SAE, c="red", alpha=1, label="SAE")
        plt.legend(loc=1, facecolor='white', framealpha=0.85)
        fig.savefig("all.jpg")
        plt.show()

    @staticmethod
    def draw_ps_scatter_plots(ps_list, title):
        fig = plt.figure()
        X = list(range(1, len(ps_list) + 1))
        Y = ps_list
        plt.scatter(X, Y, c="black", alpha=1, label=title)
        plt.legend(loc=1, facecolor='white', framealpha=0.85)
        fig.savefig(title + ".jpg")
        plt.show()

    @staticmethod
    def draw_ps_scatter_plots_sae(ps_list_X, ps_list_Y, x_label="", y_label="SAE"):
        fig = plt.figure()
        plt.scatter(ps_list_X, ps_list_Y, marker="x", c="black", alpha=1)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        fig.savefig(x_label + " vs " + y_label + ".jpg")
        plt.show()
