from collections import OrderedDict

import numpy as np
import pandas as pd

from DCN_network import DCN_network
from Propensity_score_LR import Propensity_socre_LR
from Propensity_socre_network import Propensity_socre_network
from Sparse_Propensity_score import Sparse_Propensity_score
from Utils import Utils
from dataloader import DataLoader


class DCN_PD_Deep:
    def execute_DCN_PD(self):
        csv_path = "Dataset/ihdp_sample.csv"
        split_size = 0.8
        device = Utils.get_device()
        print(device)
        results_list = []

        train_parameters_SAE = {
            "epochs": 200,
            "lr": 0.0001,
            "batch_size": 32,
            "shuffle": True,
            "sparsity_probability": 0.08,
            "weight_decay": 0.0003,
            "BETA": 0.4
        }

        print(str(train_parameters_SAE))
        file1 = open("Details.txt", "a")
        file1.write(str(train_parameters_SAE))
        file1.write("\n")

        for iter_id in range(1):
            iter_id += 1
            print("--" * 20)
            print("iter_id: {0}".format(iter_id))
            print("--" * 20)
            # load data for propensity network
            dL = DataLoader()
            np_covariates_X_train, np_covariates_X_test, np_covariates_Y_train, np_covariates_Y_test = \
                dL.preprocess_data_from_csv(csv_path, split_size)

            sparse_classifier, LR_model, LR_model_lasso = self.__train_eval_DCN(iter_id,
                                                                                np_covariates_X_train,
                                                                                np_covariates_Y_train,
                                                                                dL, device)

            # test DCN network
            reply = self.__test_DCN(iter_id,
                                    np_covariates_X_test,
                                    np_covariates_Y_test, dL,
                                    sparse_classifier,
                                    LR_model, LR_model_lasso,
                                    device)
            MSE_SAE = reply["MSE_SAE"]
            MSE_NN = reply["MSE_NN"]
            MSE_LR = reply["MSE_LR"]
            MSE_LR_lasso = reply["MSE_LR_Lasso"]

            true_ATE_NN = reply["true_ATE_NN"]
            true_ATE_SAE = reply["true_ATE_SAE"]
            true_ATE_LR = reply["true_ATE_LR"]
            true_ATE_LR_Lasso = reply["true_ATE_LR_Lasso"]

            predicted_ATE_NN = reply["predicted_ATE_NN"]
            predicted_ATE_SAE = reply["predicted_ATE_SAE"]
            predicted_ATE_LR = reply["predicted_ATE_LR"]
            predicted_ATE_LR_Lasso = reply["predicted_ATE_LR_Lasso"]

            file1.write("Iter: {0}, MSE_Sparse: {1}, MSE_NN: {2}, MSE_LR: {3}, MSE_LR_Lasso: {4}\n"
                        .format(iter_id, MSE_SAE, MSE_NN, MSE_LR, MSE_LR_lasso))
            result_dict = OrderedDict()
            result_dict["iteration"] = iter_id
            result_dict["MSE_NN"] = MSE_NN
            result_dict["MSE_SAE"] = MSE_SAE
            result_dict["MSE_LR"] = MSE_LR
            result_dict["MSE_LR_lasso"] = MSE_LR_lasso

            result_dict["true_ATE_NN"] = true_ATE_NN
            result_dict["true_ATE_SAE"] = true_ATE_SAE
            result_dict["true_ATE_LR"] = true_ATE_LR
            result_dict["true_ATE_LR_Lasso"] = true_ATE_LR_Lasso

            result_dict["predicted_ATE_NN"] = predicted_ATE_NN
            result_dict["predicted_ATE_SAE"] = predicted_ATE_SAE
            result_dict["predicted_ATE_LR"] = predicted_ATE_LR
            result_dict["predicted_ATE_LR_Lasso"] = predicted_ATE_LR_Lasso

            results_list.append(result_dict)

        MSE_set_NN = []
        MSE_set_SAE = []
        MSE_set_LR = []
        MSE_set_LR_Lasso = []
        true_ATE_NN_set = []
        true_ATE_SAE_set = []
        true_ATE_LR_set = []
        true_ATE_LR_Lasso_set = []
        predicted_ATE_NN_set = []
        predicted_ATE_SAE_set = []
        predicted_ATE_LR_set = []
        predicted_ATE_LR_Lasso_set = []

        for result in results_list:
            MSE_set_NN.append(result["MSE_NN"])
            MSE_set_SAE.append(result["MSE_SAE"])
            MSE_set_LR.append(result["MSE_LR"])
            MSE_set_LR_Lasso.append(result["MSE_LR_lasso"])

            true_ATE_NN_set.append(result["true_ATE_NN"])
            true_ATE_SAE_set.append(result["true_ATE_SAE"])
            true_ATE_LR_set.append(result["true_ATE_LR"])
            true_ATE_LR_Lasso_set.append(result["true_ATE_LR_Lasso"])

            predicted_ATE_NN_set.append(result["predicted_ATE_NN"])
            predicted_ATE_SAE_set.append(result["predicted_ATE_SAE"])
            predicted_ATE_LR_set.append(result["predicted_ATE_LR"])
            predicted_ATE_LR_Lasso_set.append(result["predicted_ATE_LR_Lasso"])

        MSE_total_NN = np.mean(np.array(MSE_set_NN))
        std_MSE_NN = np.std(MSE_set_NN, dtype=np.float64)
        Mean_ATE_NN_true = np.mean(np.array(true_ATE_NN_set))
        std_ATE_NN_true = np.std(true_ATE_NN_set, dtype=np.float64)
        Mean_ATE_NN_predicted = np.mean(np.array(predicted_ATE_NN_set))
        std_ATE_NN_predicted = np.std(predicted_ATE_NN_set, dtype=np.float64)

        print("Using NN, MSE: {0}, SD: {1}".format(MSE_total_NN, std_MSE_NN))
        print("Using NN, true ATE: {0}, SD: {1}".format(Mean_ATE_NN_true, std_ATE_NN_true))
        print("Using NN, predicted ATE: {0}, SD: {1}".format(Mean_ATE_NN_predicted, std_ATE_NN_predicted))

        MSE_total_SAE = np.mean(np.array(MSE_set_SAE))
        std_MSE_SAE = np.std(MSE_set_SAE)
        Mean_ATE_SAE_true = np.mean(np.array(true_ATE_SAE_set))
        std_ATE_SAE_true = np.std(true_ATE_SAE_set)
        Mean_ATE_SAE_predicted = np.mean(np.array(predicted_ATE_SAE_set))
        std_ATE_SAE_predicted = np.std(predicted_ATE_SAE_set)

        print("Using SAE, MSE: {0}, SD: {1}".format(MSE_total_SAE, std_MSE_SAE))
        print("Using SAE, true ATE: {0}, SD: {1}".format(Mean_ATE_SAE_true, std_ATE_SAE_true))
        print("Using SAE, predicted ATE: {0}, SD: {1}".format(Mean_ATE_SAE_predicted, std_ATE_SAE_predicted))

        MSE_total_LR = np.mean(np.array(MSE_set_LR))
        std_MSE_LR = np.std(MSE_set_LR)
        Mean_ATE_LR_true = np.mean(np.array(true_ATE_LR_set))
        std_ATE_LR_true = np.std(true_ATE_LR_set)
        Mean_ATE_LR_predicted = np.mean(np.array(predicted_ATE_LR_set))
        std_ATE_LR_predicted = np.std(predicted_ATE_LR_set)
        print("Using Logistic Regression, MSE: {0}, SD: {1}".format(MSE_total_LR, std_MSE_LR))
        print("Using Logistic Regression, true ATE: {0}, SD: {1}".format(Mean_ATE_LR_true, std_ATE_LR_true))
        print("Using Logistic Regression, predicted ATE: {0}, SD: {1}".format(Mean_ATE_LR_predicted,
                                                                              std_ATE_LR_predicted))

        MSE_total_LR_lasso = np.mean(np.array(MSE_set_LR_Lasso))
        std_MSE_LR_lasso = np.std(np.array(MSE_set_LR_Lasso))
        Mean_ATE_LR_lasso_true = np.mean(np.array(true_ATE_LR_Lasso_set))
        std_ATE_LR_lasso_true = np.std(np.array(true_ATE_LR_Lasso_set))
        Mean_ATE_LR_lasso_predicted = np.mean(np.array(predicted_ATE_LR_Lasso_set))
        std_ATE_LR_lasso_predicted = np.std(np.array(predicted_ATE_LR_Lasso_set))
        print("Using Lasso Logistic Regression, MSE: {0}, SD: {1}".format(MSE_total_LR_lasso, std_MSE_LR_lasso))
        print("Using Lasso Logistic Regression, true ATE: {0}, SD: {1}".format(Mean_ATE_LR_lasso_true,
                                                                               std_ATE_LR_lasso_true))
        print("Using Lasso Logistic Regression, predicted ATE: {0}, SD: {1}".format(Mean_ATE_LR_lasso_predicted,
                                                                                    std_ATE_LR_lasso_predicted))
        print("--" * 20)

        file1.write("\n##################################################")
        file1.write("\n")
        file1.write("\nUsing NN, MSE: {0}, SD: {1}".format(MSE_total_NN, std_MSE_NN))
        file1.write("\nUsing NN, true ATE: {0}, SD: {1}".format(Mean_ATE_NN_true, std_ATE_NN_true))
        file1.write("\nUsing NN, predicted ATE: {0}, SD: {1}".format(Mean_ATE_NN_predicted, std_ATE_NN_predicted))
        file1.write("\n-------------------------------------------------\n")
        file1.write("Using SAE, MSE: {0}, SD: {1}".format(MSE_total_SAE, std_MSE_SAE))
        file1.write("\nUsing SAE, true ATE: {0}, SD: {1}".format(Mean_ATE_SAE_true, std_ATE_SAE_true))
        file1.write("\nUsing SAE, predicted ATE: {0}, SD: {1}".format(Mean_ATE_SAE_predicted, std_ATE_SAE_predicted))
        file1.write("\n-------------------------------------------------\n")
        file1.write("Using Logistic Regression, MSE: {0}, SD: {1}".format(MSE_total_LR, std_MSE_LR))
        file1.write("\nUsing Logistic Regression, true ATE: {0}, SD: {1}".format(Mean_ATE_LR_true, std_ATE_LR_true))
        file1.write("\nUsing Logistic Regression, predicted ATE: {0}, SD: {1}".format(Mean_ATE_LR_predicted,
                                                                                      std_ATE_LR_predicted))
        file1.write("\n-------------------------------------------------\n")
        file1.write("Using Lasso Logistic Regression, MSE: {0}, SD: {1}".format(MSE_total_LR_lasso, std_MSE_LR_lasso))
        file1.write("\nUsing Lasso Logistic Regression, true ATE: {0}, SD: {1}".format(Mean_ATE_LR_lasso_true,
                                                                                       std_ATE_LR_lasso_true))
        file1.write("\nUsing Lasso Logistic Regression, predicted ATE: {0}, SD: {1}".format(Mean_ATE_LR_lasso_predicted,
                                                                                            std_ATE_LR_lasso_predicted))
        file1.write("\n##################################################")

        pd.DataFrame.from_dict(
            results_list,
            orient='columns'
        ).to_csv("./MSE/Results_consolidated.csv")

    def __train_eval_DCN(self, iter_id, np_covariates_X_train, np_covariates_Y_train, dL, device):
        print("----------- Training and evaluation phase ------------")
        ps_train_set = dL.convert_to_tensor(np_covariates_X_train, np_covariates_Y_train)

        # using NN
        self.__train_propensity_net_NN(ps_train_set, np_covariates_X_train, np_covariates_Y_train, dL,
                                       iter_id, device)

        # using SAE
        sparse_classifier = self.__train_propensity_net_SAE(ps_train_set, np_covariates_X_train,
                                                            np_covariates_Y_train, dL,
                                                            iter_id, device)
        # using Logistic Regression
        LR_model = self.__train_propensity_net_LR(np_covariates_X_train, np_covariates_Y_train,
                                                  dL,
                                                  iter_id, device)

        # using Logistic Regression Lasso
        LR_model_lasso = self.__train_propensity_net_LR_Lasso(np_covariates_X_train,
                                                              np_covariates_Y_train,
                                                              dL,
                                                              iter_id, device)

        return sparse_classifier, LR_model, LR_model_lasso

    def __train_propensity_net_NN(self, ps_train_set, np_covariates_X_train,
                                  np_covariates_Y_train, dL,
                                  iter_id, device):
        train_parameters_NN = {
            "epochs": 75,
            "lr": 0.001,
            "batch_size": 32,
            "shuffle": True,
            "train_set": ps_train_set,
            "model_save_path": "./Propensity_Model/NN_PS_model_iter_id_"
                               + str(iter_id) + "_epoch_{0}_lr_{1}.pth"
        }
        # ps using NN
        ps_net_NN = Propensity_socre_network()
        print("############### Propensity Score neural net Training ###############")
        ps_net_NN.train(train_parameters_NN, device, phase="train")

        # eval
        eval_parameters_NN = {
            "eval_set": ps_train_set,
            "model_path": "./Propensity_Model/NN_PS_model_iter_id_{0}_epoch_75_lr_0.001.pth"
                .format(iter_id)
        }

        ps_score_list_NN = ps_net_NN.eval(eval_parameters_NN, device, phase="eval")

        # train DCN
        print("############### DCN Training using NN ###############")
        data_loader_dict_NN = dL.prepare_tensor_for_DCN(np_covariates_X_train,
                                                        np_covariates_Y_train,
                                                        ps_score_list_NN)
        model_path = "./DCNModel/NN_DCN_model_iter_id_" + str(iter_id) + "_epoch_{0}_lr_{1}.pth"
        self.__train_DCN(data_loader_dict_NN, model_path, dL, device)

    def __train_propensity_net_SAE(self, ps_train_set, np_covariates_X_train, np_covariates_Y_train, dL,
                                   iter_id, device):
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
            "model_save_path": "./Propensity_Model/SAE_PS_model_iter_id_"
                               + str(1) + "_epoch_{0}_lr_{1}.pth"
        }

        ps_net_SAE = Sparse_Propensity_score()
        print("############### Propensity Score SAE net Training ###############")
        sparse_classifier = ps_net_SAE.train(train_parameters_SAE, device, phase="train")

        # eval propensity network using SAE
        ps_score_list_SAE = ps_net_SAE.eval(ps_train_set, device, phase="eval",
                                            sparse_classifier=sparse_classifier)

        # load data for ITE network using SAE
        print("############### DCN Training using SAE ###############")
        data_loader_dict_SAE = dL.prepare_tensor_for_DCN(np_covariates_X_train,
                                                         np_covariates_Y_train,
                                                         ps_score_list_SAE)

        model_path = "./DCNModel/SAE_DCN_model_iter_id_" + str(iter_id) + "_epoch_{0}_lr_{1}.pth"
        self.__train_DCN(data_loader_dict_SAE, model_path, dL, device)

        return sparse_classifier

    def __train_propensity_net_LR(self, np_covariates_X_train, np_covariates_Y_train,
                                  dL, iter_id, device):
        # eval propensity network using Logistic Regression
        ps_score_list_LR, LR_model = Propensity_socre_LR.train(np_covariates_X_train,
                                                               np_covariates_Y_train)

        # load data for ITE network using Logistic Regression
        print("############### DCN Training using Logistic Regression ###############")
        data_loader_dict_LR = dL.prepare_tensor_for_DCN(np_covariates_X_train,
                                                        np_covariates_Y_train,
                                                        ps_score_list_LR)
        model_path = "./DCNModel/LR_DCN_model_iter_id_" + str(iter_id) + "_epoch_{0}_lr_{1}.pth"
        self.__train_DCN(data_loader_dict_LR, model_path, dL, device)

        return LR_model

    def __train_propensity_net_LR_Lasso(self, np_covariates_X_train, np_covariates_Y_train,
                                        dL, iter_id, device):
        # eval propensity network using Logistic Regression Lasso
        ps_score_list_LR_lasso, LR_model_lasso = Propensity_socre_LR.train(np_covariates_X_train,
                                                                           np_covariates_Y_train,
                                                                           regularized=True)
        # load data for ITE network using Logistic Regression Lasso
        print("############### DCN Training using Logistic Regression Lasso ###############")
        data_loader_dict_LR_lasso = dL.prepare_tensor_for_DCN(np_covariates_X_train,
                                                              np_covariates_Y_train,
                                                              ps_score_list_LR_lasso)
        model_path = "./DCNModel/LR_Lasso_DCN_model_iter_id_" + str(iter_id) + "_epoch_{0}_lr_{1}.pth"
        self.__train_DCN(data_loader_dict_LR_lasso, model_path, dL, device)

        return LR_model_lasso

    @staticmethod
    def __train_DCN(data_loader_dict, model_path, dL, device):
        treated_group = data_loader_dict["treated_data"]
        np_treated_df_X = treated_group[0]
        np_treated_ps_score = treated_group[1]
        np_treated_df_Y_f = treated_group[2]
        np_treated_df_Y_cf = treated_group[3]
        tensor_treated = dL.convert_to_tensor_DCN(np_treated_df_X, np_treated_ps_score,
                                                  np_treated_df_Y_f, np_treated_df_Y_cf)

        control_group = data_loader_dict["control_data"]
        np_control_df_X = control_group[0]
        np_control_ps_score = control_group[1]
        np_control_df_Y_f = control_group[2]
        np_control_df_Y_cf = control_group[3]
        tensor_control = dL.convert_to_tensor_DCN(np_control_df_X, np_control_ps_score,
                                                  np_control_df_Y_f, np_control_df_Y_cf)

        DCN_train_parameters = {
            "epochs": 100,
            "lr": 0.001,
            "treated_batch_size": 1,
            "control_batch_size": 1,
            "shuffle": True,
            "treated_set": tensor_treated,
            "control_set": tensor_control,
            "model_save_path": model_path
        }

        # train DCN network
        dcn = DCN_network()
        dcn.train(DCN_train_parameters, device)

    def __test_DCN(self, iter_id, np_covariates_X_test, np_covariates_Y_test, dL, sparse_classifier,
                   LR_model, LR_model_lasso, device):
        print("----------- Testing phase ------------")
        ps_test_set = dL.convert_to_tensor(np_covariates_X_test, np_covariates_Y_test)
        # using NN
        MSE_NN, true_ATE_NN, predicted_ATE_NN = self.__test_DCN_NN(iter_id, np_covariates_X_test, np_covariates_Y_test,
                                                                   dL, device, ps_test_set)

        # using SAE
        MSE_SAE, true_ATE_SAE, predicted_ATE_SAE = self.__test_DCN_SAE(iter_id, np_covariates_X_test,
                                                                       np_covariates_Y_test, dL, device,
                                                                       ps_test_set, sparse_classifier)

        # using LR
        MSE_LR, true_ATE_LR, predicted_ATE_LR = self.__test_DCN_LR(np_covariates_X_test, np_covariates_Y_test,
                                                                   LR_model,
                                                                   iter_id, dL, device)

        # using LR Lasso
        MSE_LR_Lasso, true_ATE_LR_Lasso, predicted_ATE_LR_Lasso = self.__test_DCN_LR_Lasso(np_covariates_X_test,
                                                                                           np_covariates_Y_test,
                                                                                           LR_model_lasso,
                                                                                           iter_id, dL, device)

        return {
            "MSE_NN": MSE_NN,
            "true_ATE_NN": true_ATE_NN,
            "predicted_ATE_NN": predicted_ATE_NN,
            "MSE_SAE": MSE_SAE,
            "true_ATE_SAE": true_ATE_SAE,
            "predicted_ATE_SAE": predicted_ATE_SAE,
            "MSE_LR": MSE_LR,
            "true_ATE_LR": true_ATE_LR,
            "predicted_ATE_LR": predicted_ATE_LR,
            "MSE_LR_Lasso": MSE_LR_Lasso,
            "true_ATE_LR_Lasso": true_ATE_LR_Lasso,
            "predicted_ATE_LR_Lasso": predicted_ATE_LR_Lasso
        }

    def __test_DCN_NN(self, iter_id, np_covariates_X_test, np_covariates_Y_test, dL, device, ps_test_set):
        # testing using NN
        ps_net_NN = Propensity_socre_network()
        ps_eval_parameters_NN = {
            "eval_set": ps_test_set,
            "model_path": "./Propensity_Model/NN_PS_model_iter_id_{0}_epoch_75_lr_0.001.pth".format(iter_id)
        }
        ps_score_list_NN = ps_net_NN.eval(ps_eval_parameters_NN, device, phase="eval")
        Utils.write_to_csv("./MSE/NN_Prop_score_{0}.csv".format(iter_id), ps_score_list_NN)

        # load data for ITE network using vanilla network
        print("############### DCN Testing using NN ###############")
        data_loader_dict_NN = dL.prepare_tensor_for_DCN(np_covariates_X_test,
                                                        np_covariates_Y_test,
                                                        ps_score_list_NN)
        model_path = "./DCNModel/NN_DCN_model_iter_id_{0}_epoch_100_lr_0.001.pth".format(iter_id)
        MSE_NN, true_ATE_NN, predicted_ATE_NN = self.__do_test_DCN(data_loader_dict_NN, dL, device, model_path)

        return MSE_NN, true_ATE_NN, predicted_ATE_NN

    def __test_DCN_SAE(self, iter_id, np_covariates_X_test, np_covariates_Y_test, dL, device,
                       ps_test_set, sparse_classifier):
        # testing using SAE
        ps_net_SAE = Sparse_Propensity_score()
        ps_score_list_SAE = ps_net_SAE.eval(ps_test_set, device, phase="eval",
                                            sparse_classifier=sparse_classifier)
        Utils.write_to_csv("./MSE/SAE_Prop_score_{0}.csv".format(iter_id), ps_score_list_SAE)

        # load data for ITE network using SAE
        print("############### DCN Testing using SAE ###############")
        data_loader_dict_SAE = dL.prepare_tensor_for_DCN(np_covariates_X_test,
                                                         np_covariates_Y_test,
                                                         ps_score_list_SAE)
        model_path = "./DCNModel/SAE_DCN_model_iter_id_{0}_epoch_100_lr_0.001.pth".format(iter_id)
        MSE_SAE, true_ATE_SAE, predicted_ATE_SAE = self.__do_test_DCN(data_loader_dict_SAE, dL, device, model_path)

        return MSE_SAE, true_ATE_SAE, predicted_ATE_SAE

    def __test_DCN_LR(self, np_covariates_X_test, np_covariates_Y_test, LR_model, iter_id, dL, device):
        # testing using Logistic Regression
        ps_score_list_LR = Propensity_socre_LR.test(np_covariates_X_test,
                                                    np_covariates_Y_test,
                                                    log_reg=LR_model)
        Utils.write_to_csv("./MSE/LR_Prop_score_{0}.csv".format(iter_id), ps_score_list_LR)

        # load data for ITE network using Logistic Regression
        data_loader_dict_SAE = dL.prepare_tensor_for_DCN(np_covariates_X_test,
                                                         np_covariates_Y_test,
                                                         ps_score_list_LR)
        print("############### DCN Testing using LR ###############")
        model_path = "./DCNModel/LR_DCN_model_iter_id_{0}_epoch_100_lr_0.001.pth".format(iter_id)
        MSE_LR, true_ATE_LR, predicted_ATE_LR = self.__do_test_DCN(data_loader_dict_SAE, dL,
                                                                   device, model_path)
        return MSE_LR, true_ATE_LR, predicted_ATE_LR

    def __test_DCN_LR_Lasso(self, np_covariates_X_test, np_covariates_Y_test, LR_model_lasso,
                            iter_id, dL, device):
        # testing using Logistic Regression Lasso
        ps_score_list_LR_lasso = Propensity_socre_LR.test(np_covariates_X_test,
                                                          np_covariates_Y_test,
                                                          log_reg=LR_model_lasso)

        Utils.write_to_csv("./MSE/LR_lasso_Prop_score_{0}.csv".format(iter_id), ps_score_list_LR_lasso)

        # load data for ITE network using Logistic Regression Lasso
        data_loader_dict_SAE = dL.prepare_tensor_for_DCN(np_covariates_X_test,
                                                         np_covariates_Y_test,
                                                         ps_score_list_LR_lasso)
        print("############### DCN Testing using LR Lasso ###############")
        model_path = "./DCNModel/LR_Lasso_DCN_model_iter_id_{0}_epoch_100_lr_0.001.pth".format(iter_id)

        MSE_LR_Lasso, true_ATE_LR_Lasso, predicted_ATE_LR_Lasso = self.__do_test_DCN(data_loader_dict_SAE, dL,
                                                                                     device, model_path)
        return MSE_LR_Lasso, true_ATE_LR_Lasso, predicted_ATE_LR_Lasso

    @staticmethod
    def __do_test_DCN(data_loader_dict, dL, device, model_path):
        treated_group = data_loader_dict["treated_data"]
        np_treated_df_X = treated_group[0]
        np_treated_ps_score = treated_group[1]
        np_treated_df_Y_f = treated_group[2]
        np_treated_df_Y_cf = treated_group[3]
        tensor_treated = dL.convert_to_tensor_DCN(np_treated_df_X, np_treated_ps_score,
                                                  np_treated_df_Y_f, np_treated_df_Y_cf)

        control_group = data_loader_dict["control_data"]
        np_control_df_X = control_group[0]
        np_control_ps_score = control_group[1]
        np_control_df_Y_f = control_group[2]
        np_control_df_Y_cf = control_group[3]
        tensor_control = dL.convert_to_tensor_DCN(np_control_df_X, np_control_ps_score,
                                                  np_control_df_Y_f, np_control_df_Y_cf)

        DCN_test_parameters = {
            "treated_set": tensor_treated,
            "control_set": tensor_control,
            "model_save_path": model_path
        }

        dcn = DCN_network()
        response_dict = dcn.eval(DCN_test_parameters, device)
        err_treated = [ele ** 2 for ele in response_dict["treated_err"]]
        err_control = [ele ** 2 for ele in response_dict["control_err"]]

        true_ATE = sum(response_dict["true_ITE"]) / len(response_dict["true_ITE"])
        predicted_ATE = sum(response_dict["predicted_ITE"]) / len(response_dict["predicted_ITE"])

        total_sum = sum(err_treated) + sum(err_control)
        total_item = len(err_treated) + len(err_control)
        MSE = total_sum / total_item
        print("MSE: {0}".format(MSE))
        max_treated = max(err_treated)
        max_control = max(err_control)
        max_total = max(max_treated, max_control)

        min_treated = min(err_treated)
        min_control = min(err_control)
        min_total = min(min_treated, min_control)

        print("Max: {0}, Min: {1}".format(max_total, min_total))
        return MSE, true_ATE, predicted_ATE
        # np.save("treated_err.npy", err_treated)
        # np.save("control_err.npy", err_control)
