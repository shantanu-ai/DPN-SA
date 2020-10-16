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

import matplotlib.pyplot as plt
import numpy as np

from Propensity_score_LR import Propensity_socre_LR
from Propensity_socre_network import Propensity_socre_network
from Sparse_Propensity_score import Sparse_Propensity_score
from Utils import Utils
from dataloader import DataLoader


class Graphs:
    def draw_scatter_plots(self):
        device = Utils.get_device()
        train_path = "Dataset/jobs_DW_bin.new.10.train.npz"
        dL = DataLoader()
        np_covariates_X, np_covariates_Y = dL.preprocess_for_graphs(train_path)
        ps_train_set = dL.convert_to_tensor(np_covariates_X, np_covariates_Y)
        # ps_list_nn = self.__train_propensity_net_NN(ps_train_set, device)
        ps_list_SAE = self.__train_propensity_net_SAE(ps_train_set, device)

        # ps_list_LR = self.__train_propensity_net_LR(np_covariates_X, np_covariates_Y)
        # ps_list_LR_lasso = self.__train_propensity_net_LR_Lasso(np_covariates_X, np_covariates_Y)

        # print(len(ps_list_nn))
        # print(len(ps_list_SAE))
        # print(len(ps_list_LR))
        # print(len(ps_list_LR_lasso))
        #
        # self.draw_ps_scatter_plots_all(ps_list_nn, ps_list_SAE, ps_list_LR, ps_list_LR_lasso)
        #
        # self.draw_ps_scatter_plots(ps_list_nn, "PD")
        # self.draw_ps_scatter_plots(ps_list_SAE, "SAE")
        # self.draw_ps_scatter_plots(ps_list_LR, "LR")
        # self.draw_ps_scatter_plots(ps_list_LR_lasso, "LR Lasso")
        #
        # self.draw_ps_scatter_plots_sae(ps_list_nn, ps_list_SAE, x_label="PD", y_label="SAE")
        # self.draw_ps_scatter_plots_sae(ps_list_LR, ps_list_SAE, x_label="LR", y_label="SAE")
        # self.draw_ps_scatter_plots_sae(ps_list_LR_lasso, ps_list_SAE, x_label="LR_Lasso", y_label="SAE")

    def __train_propensity_net_NN(self, ps_train_set, device):
        train_parameters_NN = {
            "epochs": 50,
            "lr": 0.001,
            "batch_size": 32,
            "shuffle": True,
            "train_set": ps_train_set,
            "input_nodes": 17,
            "model_save_path": "./Propensity_Model/Graph_NN_PS_model_epoch_{0}_lr_{1}.pth"
        }
        # ps using NN
        ps_net_NN = Propensity_socre_network()
        print("############### Propensity Score neural net Training ###############")
        ps_net_NN.train(train_parameters_NN, device, phase="train")

        # eval
        eval_parameters_NN = {
            "eval_set": ps_train_set,
            "input_nodes": 17,
            "model_path": "./Propensity_Model/Graph_NN_PS_model_epoch_50_lr_0.001.pth"
        }

        # ps_score_list_NN = ps_net_NN.eval(eval_parameters_NN, device, phase="eval")
        ps_score_list_NN = ps_net_NN.eval_return_complete_list(eval_parameters_NN, device, phase="eval")
        treated_ps_list = [d["prop_score"] for d in ps_score_list_NN if d['treatment'] == 1]
        control_ps_list = [d["prop_score"] for d in ps_score_list_NN if d['treatment'] == 0]
        print("treated: " + str(len(treated_ps_list)))
        print("control: " + str(len(control_ps_list)))
        print("total: " + str(len(treated_ps_list) + len(control_ps_list)))
        self.draw(treated_ps_list, control_ps_list,
                  label_treated="Treated", label_control="Control",
                  fig_name="./Plots/Fig_NN",
                  title="Jobs: DCN-PD", max_limit=500)
        return ps_score_list_NN

    def __train_propensity_net_SAE(self, ps_train_set, device):
        # !!! best parameter list
        train_parameters_SAE = {
            "epochs": 400,
            "lr": 0.001,
            "batch_size": 32,
            "shuffle": True,
            "train_set": ps_train_set,
            "sparsity_probability": 0.8,
            "weight_decay": 0.0003,
            "BETA": 0.1,
            "input_nodes": 17,
            "model_save_path": "./Propensity_Model/SAE_PS_model_iter_id_epoch_{0}_lr_{1}.pth"
        }

        ps_net_SAE = Sparse_Propensity_score()
        print("############### Propensity Score SAE net Training ###############")
        sparse_classifier, sae_classifier_stacked_all_layer_active, \
        sae_classifier_stacked_cur_layer_active = ps_net_SAE.train(train_parameters_SAE, device, phase="train")

        ps_score_list_SAE = ps_net_SAE.eval_return_complete_list(ps_train_set, device, phase="eval",
                                                                 sparse_classifier=sparse_classifier)
        treated_ps_list = [d["prop_score"] for d in ps_score_list_SAE if d['treatment'] == 1]
        control_ps_list = [d["prop_score"] for d in ps_score_list_SAE if d['treatment'] == 0]
        print("treated: " + str(len(treated_ps_list)))
        print("control: " + str(len(control_ps_list)))
        print("total: " + str(len(treated_ps_list) + len(control_ps_list)))
        self.draw(treated_ps_list, control_ps_list,
                  label_treated="Treated", label_control="Control",
                  fig_name="./Plots/Fig_SAE",
                  title="Jobs: DPN-SA End to End", max_limit=500)
        return ps_score_list_SAE

    @staticmethod
    def __train_propensity_net_LR(np_covariates_X_train, np_covariates_Y_train):
        # eval propensity network using Logistic Regression
        ps_score_list_LR, _ = Propensity_socre_LR.train(np_covariates_X_train,
                                                        np_covariates_Y_train)
        return ps_score_list_LR

    def __train_propensity_net_LR_Lasso(self, np_covariates_X_train, np_covariates_Y_train):
        # eval propensity network using Logistic Regression Lasso
        treated_ps_list, control_ps_list = Propensity_socre_LR.train_graph(np_covariates_X_train,
                                                                           np_covariates_Y_train,
                                                                           regularized=True)
        self.draw(treated_ps_list, control_ps_list,
                  label_treated="Treated", label_control="Control",
                  fig_name="./Plots/Fig_LR_Lasso",
                  title="Jobs: LR Lasso", max_limit=500)
        return treated_ps_list

    @staticmethod
    def draw(treated_ps_list, control_ps_list, label_treated, label_control, fig_name, title, max_limit):
        bins1 = np.linspace(0, 1, 10)
        print(treated_ps_list)
        plt.hist(treated_ps_list, bins1, alpha=0.5, label=label_treated, color='#B60E0E', histtype="bar",
                 edgecolor='r')
        plt.hist(control_ps_list, bins1, alpha=0.5, label=label_control, color='g', histtype="bar",
                 edgecolor='g')
        plt.xlabel('Propensity scores', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(title)
        plt.ylim(0, max_limit)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        plt.legend(loc='upper right')
        # plt.show()
        plt.draw()
        plt.savefig(fig_name, dpi=220)
        plt.clf()

    @staticmethod
    def draw_ps_scatter_plots_all(ps_list_nn, ps_list_SAE, ps_list_LR, ps_list_LR_lasso):
        X = list(range(1, 2571))
        print(len(X))
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
        X = list(range(1, 2571))
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


Graphs().draw_scatter_plots()
