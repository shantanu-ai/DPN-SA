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

import statistics
from collections import OrderedDict

import numpy as np

from DCN_network import DCN_network
from Utils import Utils
from dataloader import DataLoader
from shallow_train import shallow_train


class Model_25_1_25:
    def run_all_expeiments(self):
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
        file1 = open("Details_original_1.txt", "a")
        file1.write(str(train_parameters_SAE))
        file1.write("\n")
        file1.write("Without batch norm")
        file1.write("\n")
        for iter_id in range(100):
            iter_id += 1
            print("--" * 20)
            print("iter_id: {0}".format(iter_id))
            print("--" * 20)
            # load data for propensity network
            dL = DataLoader()
            np_covariates_X_train, np_covariates_X_test, np_covariates_Y_train, np_covariates_Y_test = \
                dL.preprocess_data_from_csv_augmented(csv_path, split_size)

            trained_models = self.__train_eval_DCN(iter_id,
                                                   np_covariates_X_train,
                                                   np_covariates_Y_train,
                                                   dL, device)

            sparse_classifier = trained_models["sparse_classifier"]

            # test DCN network
            reply = self.__test_DCN(iter_id,
                                    np_covariates_X_test,
                                    np_covariates_Y_test,
                                    dL,
                                    sparse_classifier,
                                    device)

            MSE_SAE_e2e = reply["MSE_SAE_e2e"]
            true_ATE_SAE_e2e = reply["true_ATE_SAE_e2e"]
            predicted_ATE_SAE_e2e = reply["predicted_ATE_SAE_e2e"]

            file1.write("Iter: {0}, MSE_Sparse_e2e: {1}, MSE_Sparse_stacked_all_layer_active: {2}, "
                        "MSE_Sparse_stacked_cur_layer_active: {3},"
                        " MSE_NN: {4}, MSE_LR: {5}, MSE_LR_Lasso: {6}\n"
                        .format(iter_id, MSE_SAE_e2e, 0,
                                0, 0, 0, 0))
            result_dict = OrderedDict()
            result_dict["iter_id"] = iter_id
            result_dict["MSE_SAE_e2e"] = MSE_SAE_e2e
            result_dict["true_ATE_SAE_e2e"] = true_ATE_SAE_e2e
            result_dict["predicted_ATE_SAE_e2e"] = predicted_ATE_SAE_e2e

            results_list.append(result_dict)

        MSE_set_SAE_e2e = []

        true_ATE_SAE_set_e2e = []
        predicted_ATE_SAE_set_e2e = []

        for result in results_list:
            MSE_set_SAE_e2e.append(result["MSE_SAE_e2e"])
            true_ATE_SAE_set_e2e.append(result["true_ATE_SAE_e2e"])

            predicted_ATE_SAE_set_e2e.append(result["predicted_ATE_SAE_e2e"])

        print("\n-------------------------------------------------\n")

        MSE_total_SAE_e2e = np.mean(np.array(MSE_set_SAE_e2e))
        std_MSE_SAE_e2e = statistics.pstdev(MSE_set_SAE_e2e)
        Mean_ATE_SAE_true_e2e = np.mean(np.array(true_ATE_SAE_set_e2e))
        std_ATE_SAE_true_e2e = statistics.pstdev(true_ATE_SAE_set_e2e)
        Mean_ATE_SAE_predicted_e2e = np.mean(np.array(predicted_ATE_SAE_set_e2e))
        std_ATE_SAE_predicted_e2e = statistics.pstdev(predicted_ATE_SAE_set_e2e)

        print("Using SAE E2E, MSE: {0}, SD: {1}".format(MSE_total_SAE_e2e, std_MSE_SAE_e2e))
        print("Using SAE E2E, true ATE: {0}, SD: {1}".format(Mean_ATE_SAE_true_e2e, std_ATE_SAE_true_e2e))
        print("Using SAE E2E, predicted ATE: {0}, SD: {1}".format(Mean_ATE_SAE_predicted_e2e,
                                                                  std_ATE_SAE_predicted_e2e))
        print("--" * 20)

        file1.write("\n##################################################")
        file1.write("\n")
        file1.write("Using SAE E2E, MSE: {0}, SD: {1}".format(MSE_total_SAE_e2e, std_MSE_SAE_e2e))
        file1.write("\nUsing SAE E2E, true ATE: {0}, SD: {1}".format(Mean_ATE_SAE_true_e2e, std_ATE_SAE_true_e2e))
        file1.write("\nUsing SAE E2E, predicted ATE: {0}, SD: {1}".format(Mean_ATE_SAE_predicted_e2e,
                                                                          std_ATE_SAE_predicted_e2e))
        file1.write("\n-------------------------------------------------\n")
        file1.write("\n-------------------------------------------------\n")
        file1.write("\n##################################################")

        Utils.write_to_csv("./MSE/Results_consolidated.csv", results_list)

    def __train_eval_DCN(self, iter_id, np_covariates_X_train, np_covariates_Y_train, dL, device):
        print("----------- Training and evaluation phase ------------")
        ps_train_set = dL.convert_to_tensor(np_covariates_X_train, np_covariates_Y_train)

        # using SAE
        sparse_classifier = \
            self.__train_propensity_net_SAE(ps_train_set, np_covariates_X_train,
                                            np_covariates_Y_train, dL,
                                            iter_id, device)

        return {
            "sparse_classifier": sparse_classifier
        }

    def __train_propensity_net_SAE(self, ps_train_set, np_covariates_X_train, np_covariates_Y_train, dL,
                                   iter_id, device):
        # !!! best parameter list
        train_parameters_SAE = {
            'epochs': 400,
            'lr': 0.001,
            "batch_size": 32,
            "shuffle": True,
            "train_set": ps_train_set,
            "sparsity_probability": 0.08,
            "weight_decay": 0.0003,
            "BETA": 0.4
        }

        ps_net_SAE = shallow_train()
        print("############### Propensity Score SAE net Training ###############")
        sparse_classifier = ps_net_SAE.train(train_parameters_SAE, device, phase="train")

        # eval propensity network using SAE
        model_path_e2e = "./DCNModel/SAE_E2E_DCN_model_iter_id_" + str(iter_id) + "_epoch_{0}_lr_{1}.pth"
        print("----------End to End SAE training----------")

        self.__train_DCN_SAE(ps_net_SAE, ps_train_set, device, np_covariates_X_train,
                             np_covariates_Y_train, iter_id, dL, sparse_classifier, model_path_e2e)

        return sparse_classifier

    def __train_DCN_SAE(self, ps_net_SAE, ps_train_set, device, np_covariates_X_train,
                        np_covariates_Y_train, iter_id, dL, sparse_classifier, model_path):
        # eval propensity network using SAE
        ps_score_list_SAE = ps_net_SAE.eval(ps_train_set, device, phase="eval",
                                            sparse_classifier=sparse_classifier)

        # load data for ITE network using SAE
        print("############### DCN Training using SAE ###############")
        data_loader_dict_SAE = dL.prepare_tensor_for_DCN(np_covariates_X_train,
                                                         np_covariates_Y_train,
                                                         ps_score_list_SAE,
                                                         is_synthetic=True)

        self.__train_DCN(data_loader_dict_SAE, model_path, dL, device)

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
            "treated_set_train": tensor_treated,
            "control_set_train": tensor_control,
            "model_save_path": model_path,
            "input_nodes": 100
        }

        # train DCN network
        dcn = DCN_network()
        dcn.train(DCN_train_parameters, device)

    def __test_DCN(self, iter_id, np_covariates_X_test, np_covariates_Y_test, dL, sparse_classifier, device):
        print("----------- Testing phase ------------")
        ps_test_set = dL.convert_to_tensor(np_covariates_X_test, np_covariates_Y_test)

        # using SAE
        model_path_e2e = "./DCNModel/SAE_E2E_DCN_model_iter_id_{0}_epoch_100_lr_0.001.pth".format(iter_id)

        propensity_score_save_path_e2e = "./MSE/SAE_E2E_Prop_score_{0}.csv"

        ITE_save_path_e2e = "./MSE/ITE/ITE_SAE_E2E_iter_{0}.csv"

        print("############### DCN Testing using SAE E2E ###############")
        MSE_SAE_e2e, true_ATE_SAE_e2e, predicted_ATE_SAE_e2e = \
            self.__test_DCN_SAE(iter_id, np_covariates_X_test,
                                np_covariates_Y_test, dL, device,
                                ps_test_set, sparse_classifier, model_path_e2e,
                                propensity_score_save_path_e2e, ITE_save_path_e2e)

        return {

            "MSE_SAE_e2e": MSE_SAE_e2e,
            "true_ATE_SAE_e2e": true_ATE_SAE_e2e,
            "predicted_ATE_SAE_e2e": predicted_ATE_SAE_e2e,
        }

    def __test_DCN_SAE(self, iter_id, np_covariates_X_test, np_covariates_Y_test, dL, device,
                       ps_test_set, sparse_classifier, model_path, propensity_score_csv_path,
                       ite_csv_path):
        # testing using SAE
        ps_net_SAE = shallow_train()
        ps_score_list_SAE = ps_net_SAE.eval(ps_test_set, device, phase="eval",
                                            sparse_classifier=sparse_classifier)
        Utils.write_to_csv(propensity_score_csv_path.format(iter_id), ps_score_list_SAE)

        # load data for ITE network using SAE
        data_loader_dict_SAE = dL.prepare_tensor_for_DCN(np_covariates_X_test,
                                                         np_covariates_Y_test,
                                                         ps_score_list_SAE,
                                                         is_synthetic=True)
        MSE_SAE, true_ATE_SAE, predicted_ATE_SAE, ITE_dict_list = self.__do_test_DCN(data_loader_dict_SAE,
                                                                                     dL, device,
                                                                                     model_path)

        Utils.write_to_csv(ite_csv_path.format(iter_id), ITE_dict_list)
        return MSE_SAE, true_ATE_SAE, predicted_ATE_SAE

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
            "model_save_path": model_path,
            "input_nodes": 100
        }

        dcn = DCN_network()
        response_dict = dcn.eval(DCN_test_parameters, device, input_nodes=100)
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

        return MSE, true_ATE, predicted_ATE, response_dict["ITE_dict_list"]
        # np.save("treated_err.npy", err_treated)
        # np.save("control_err.npy", err_control)
