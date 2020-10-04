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

from collections import OrderedDict

import numpy as np

from DPN_SA_Deep import DPN_SA_Deep
from Utils import Utils
from dataloader import DataLoader


class Experiments:
    def run_all_experiments(self, iterations, running_mode):

        train_path = "Dataset/jobs_DW_bin.new.10.train.npz"
        test_path = "Dataset/jobs_DW_bin.new.10.test.npz"
        split_size = 0.8
        device = Utils.get_device()
        print(device)
        results_list = []

        train_parameters_SAE = {
            "epochs": 400,
            "lr": 0.001,
            "batch_size": 32,
            "shuffle": True,
            "sparsity_probability": 0.8,
            "weight_decay": 0.0003,
            "BETA": 0.1,
        }
        run_parameters = self.__get_run_parameters(running_mode)

        print(str(train_parameters_SAE))
        file1 = open(run_parameters["summary_file_name"], "a")
        file1.write(str(train_parameters_SAE))
        file1.write("\n")
        file1.write("\n")
        for iter_id in range(iterations):
            print("########### 400 epochs ###########")
            print("--" * 20)
            print("iter_id: {0}".format(iter_id))
            print("--" * 20)
            # load data for propensity network
            dL = DataLoader()

            np_covariates_X_train, np_covariates_X_test, np_covariates_Y_train, np_covariates_Y_test \
                = self.load_data(running_mode, dL, train_path, test_path, iter_id)

            dp_sa = DPN_SA_Deep()
            trained_models = dp_sa.train_eval_DCN(iter_id,
                                                  np_covariates_X_train,
                                                  np_covariates_Y_train,
                                                  dL, device, run_parameters,
                                                  is_synthetic=run_parameters["is_synthetic"])

            sparse_classifier = trained_models["sparse_classifier"]
            LR_model = trained_models["LR_model"]
            LR_model_lasso = trained_models["LR_model_lasso"]
            sae_classifier_stacked_all_layer_active = trained_models["sae_classifier_stacked_all_layer_active"]
            sae_classifier_stacked_cur_layer_active = trained_models["sae_classifier_stacked_cur_layer_active"]

            # test DCN network
            reply = dp_sa.test_DCN(iter_id,
                                   np_covariates_X_test,
                                   np_covariates_Y_test,
                                   dL,
                                   sparse_classifier,
                                   sae_classifier_stacked_all_layer_active,
                                   sae_classifier_stacked_cur_layer_active,
                                   LR_model,
                                   LR_model_lasso,
                                   device, run_parameters)

            # MSE_SAE_e2e = reply["MSE_SAE_e2e"]
            # MSE_SAE_stacked_all_layer_active = reply["MSE_SAE_stacked_all_layer_active"]
            # MSE_SAE_stacked_cur_layer_active = reply["MSE_SAE_stacked_cur_layer_active"]
            # MSE_NN = reply["MSE_NN"]
            # MSE_LR = reply["MSE_LR"]
            # MSE_LR_lasso = reply["MSE_LR_Lasso"]
            #
            # true_ATE_NN = reply["true_ATE_NN"]
            # true_ATE_SAE_e2e = reply["true_ATE_SAE_e2e"]
            # true_ATE_SAE_stacked_all_layer_active = reply["true_ATE_SAE_stacked_all_layer_active"]
            # true_ATE_SAE_stacked_cur_layer_active = reply["true_ATE_SAE_stacked_cur_layer_active"]
            # true_ATE_LR = reply["true_ATE_LR"]
            # true_ATE_LR_Lasso = reply["true_ATE_LR_Lasso"]
            #
            # predicted_ATE_NN = reply["predicted_ATE_NN"]
            # predicted_ATE_SAE_e2e = reply["predicted_ATE_SAE_e2e"]
            # predicted_ATE_SAE_stacked_all_layer_active = reply["predicted_ATE_SAE_stacked_all_layer_active"]
            # predicted_ATE_SAE_stacked_cur_layer_active = reply["predicted_ATE_SAE_stacked_cur_layer_active"]
            # predicted_ATE_LR = reply["predicted_ATE_LR"]
            # predicted_ATE_LR_Lasso = reply["predicted_ATE_LR_Lasso"]

            NN_ate_pred = reply["NN_ate_pred"]
            NN_att_pred = reply["NN_att_pred"]
            NN_bias_att = reply["NN_bias_att"]
            NN_atc_pred = reply["NN_atc_pred"]
            NN_policy_value = reply["NN_policy_value"]
            NN_policy_risk = reply["NN_policy_risk"]
            NN_err_fact = reply["NN_err_fact"]

            SAE_e2e_ate_pred = reply["SAE_e2e_ate_pred"]
            SAE_e2e_att_pred = reply["SAE_e2e_att_pred"]
            SAE_e2e_bias_att = reply["SAE_e2e_bias_att"]
            SAE_e2e_atc_pred = reply["SAE_e2e_atc_pred"]
            SAE_e2e_policy_value = reply["SAE_e2e_policy_value"]
            SAE_e2e_policy_risk = reply["SAE_e2e_policy_risk"]
            SAE_e2e_err_fact = reply["SAE_e2e_err_fact"]

            SAE_stacked_all_layer_active_ate_pred = reply["SAE_stacked_all_layer_active_ate_pred"]
            SAE_stacked_all_layer_active_att_pred = reply["SAE_stacked_all_layer_active_att_pred"]
            SAE_stacked_all_layer_active_bias_att = reply["SAE_stacked_all_layer_active_bias_att"]
            SAE_stacked_all_layer_active_atc_pred = reply["SAE_stacked_all_layer_active_atc_pred"]
            SAE_stacked_all_layer_active_policy_value = reply["SAE_stacked_all_layer_active_policy_value"]
            SAE_stacked_all_layer_active_policy_risk = reply["SAE_stacked_all_layer_active_policy_risk"]
            SAE_stacked_all_layer_active_err_fact = reply["SAE_stacked_all_layer_active_err_fact"]

            SAE_stacked_cur_layer_active_ate_pred = reply["SAE_stacked_cur_layer_active_ate_pred"]
            SAE_stacked_cur_layer_active_att_pred = reply["SAE_stacked_cur_layer_active_att_pred"]
            SAE_stacked_cur_layer_active_bias_att = reply["SAE_stacked_cur_layer_active_bias_att"]
            SAE_stacked_cur_layer_active_atc_pred = reply["SAE_stacked_cur_layer_active_atc_pred"]
            SAE_stacked_cur_layer_active_policy_value = reply["SAE_stacked_cur_layer_active_policy_value"]
            SAE_stacked_cur_layer_active_policy_risk = reply["SAE_stacked_cur_layer_active_policy_risk"]
            SAE_stacked_cur_layer_active_err_fact = reply["SAE_stacked_cur_layer_active_err_fact"]

            LR_ate_pred = reply["LR_ate_pred"]
            LR_att_pred = reply["LR_att_pred"]
            LR_bias_att = reply["LR_bias_att"]
            LR_atc_pred = reply["LR_atc_pred"]
            LR_policy_value = reply["LR_policy_value"]
            LR_policy_risk = reply["LR_policy_risk"]
            LR_err_fact = reply["LR_err_fact"]

            LR_Lasso_ate_pred = reply["LR_Lasso_ate_pred"]
            LR_Lasso_att_pred = reply["LR_Lasso_att_pred"]
            LR_Lasso_bias_att = reply["LR_Lasso_bias_att"]
            LR_Lasso_atc_pred = reply["LR_Lasso_atc_pred"]
            LR_Lasso_policy_value = reply["LR_Lasso_policy_value"]
            LR_Lasso_policy_risk = reply["LR_Lasso_policy_risk"]
            LR_Lasso_err_fact = reply["LR_Lasso_err_fact"]

            file1.write("Iter: {0}, "
                        "NN_bias_att: {1},  "
                        "SAE_e2e_bias_att: {2},  "
                        "SAE_stacked_all_layer_active_bias_att: {3},  "
                        "SAE_stacked_cur_layer_active_bias_att: {4}, "
                        "LR_bias_att: {5}, "
                        "LR_Lasso_bias_att: {6}, "
                        "NN_policy_risk: {7}, "
                        "SAE_e2e_policy_risk: {8},  "
                        "SAE_stacked_all_layer_active_policy_risk: {9}, "
                        "SAE_stacked_cur_layer_active_policy_risk: {10}, "
                        "LR_policy_risk: {11}, "
                        "LR_Lasso_policy_risk: {12}"

                        .format(iter_id, NN_bias_att,
                                SAE_e2e_bias_att,
                                SAE_stacked_all_layer_active_bias_att,
                                SAE_stacked_cur_layer_active_bias_att,
                                LR_bias_att,
                                LR_Lasso_bias_att,
                                NN_policy_risk,
                                SAE_e2e_policy_risk,
                                SAE_stacked_all_layer_active_policy_risk,
                                SAE_stacked_cur_layer_active_policy_risk,
                                LR_policy_risk,
                                LR_Lasso_policy_risk))
            result_dict = OrderedDict()
            result_dict["iter_id"] = iter_id
            result_dict["NN_ate_pred"] = NN_ate_pred
            result_dict["NN_att_pred"] = NN_att_pred
            result_dict["NN_bias_att"] = NN_bias_att
            result_dict["NN_atc_pred"] = NN_atc_pred
            result_dict["NN_policy_value"] = NN_policy_value
            result_dict["NN_policy_risk"] = NN_policy_risk
            result_dict["NN_err_fact"] = NN_err_fact

            result_dict["SAE_e2e_ate_pred"] = SAE_e2e_ate_pred
            result_dict["SAE_e2e_att_pred"] = SAE_e2e_att_pred
            result_dict["SAE_e2e_bias_att"] = SAE_e2e_bias_att
            result_dict["SAE_e2e_atc_pred"] = SAE_e2e_atc_pred
            result_dict["SAE_e2e_policy_value"] = SAE_e2e_policy_value
            result_dict["SAE_e2e_policy_risk"] = SAE_e2e_policy_risk
            result_dict["SAE_e2e_err_fact"] = SAE_e2e_err_fact

            result_dict["SAE_stacked_all_layer_active_ate_pred"] = SAE_stacked_all_layer_active_ate_pred
            result_dict["SAE_stacked_all_layer_active_att_pred"] = SAE_stacked_all_layer_active_att_pred
            result_dict["SAE_stacked_all_layer_active_bias_att"] = SAE_stacked_all_layer_active_bias_att
            result_dict["SAE_stacked_all_layer_active_atc_pred"] = SAE_stacked_all_layer_active_atc_pred
            result_dict["SAE_stacked_all_layer_active_policy_value"] = SAE_stacked_all_layer_active_policy_value
            result_dict["SAE_stacked_all_layer_active_policy_risk"] = SAE_stacked_all_layer_active_policy_risk
            result_dict["SAE_stacked_all_layer_active_err_fact"] = SAE_stacked_all_layer_active_err_fact

            result_dict["SAE_stacked_cur_layer_active_ate_pred"] = SAE_stacked_cur_layer_active_ate_pred
            result_dict["SAE_stacked_cur_layer_active_att_pred"] = SAE_stacked_cur_layer_active_att_pred
            result_dict["SAE_stacked_cur_layer_active_bias_att"] = SAE_stacked_cur_layer_active_bias_att
            result_dict["SAE_stacked_cur_layer_active_atc_pred"] = SAE_stacked_cur_layer_active_atc_pred
            result_dict["SAE_stacked_cur_layer_active_policy_value"] = SAE_stacked_cur_layer_active_policy_value
            result_dict["SAE_stacked_cur_layer_active_policy_risk"] = SAE_stacked_cur_layer_active_policy_risk
            result_dict["SAE_stacked_cur_layer_active_err_fact"] = SAE_stacked_cur_layer_active_err_fact

            result_dict["LR_ate_pred"] = LR_ate_pred
            result_dict["LR_att_pred"] = LR_att_pred
            result_dict["LR_bias_att"] = LR_bias_att
            result_dict["LR_atc_pred"] = LR_atc_pred
            result_dict["LR_policy_value"] = LR_policy_value
            result_dict["LR_policy_risk"] = LR_policy_risk
            result_dict["LR_err_fact"] = LR_err_fact

            result_dict["LR_Lasso_ate_pred"] = LR_Lasso_ate_pred
            result_dict["LR_Lasso_att_pred"] = LR_Lasso_att_pred
            result_dict["LR_Lasso_bias_att"] = LR_Lasso_bias_att
            result_dict["LR_Lasso_atc_pred"] = LR_Lasso_atc_pred
            result_dict["LR_Lasso_policy_value"] = LR_Lasso_policy_value
            result_dict["LR_Lasso_policy_risk"] = LR_Lasso_policy_risk
            result_dict["LR_Lasso_err_fact"] = LR_Lasso_err_fact

            results_list.append(result_dict)

        bias_att_set_NN = []
        policy_risk_set_NN = []

        bias_att_set_SAE_E2E = []
        policy_risk_set_SAE_E2E = []

        bias_att_set_SAE_stacked_all_layer_active = []
        policy_risk_set_SAE_stacked_all_layer_active = []

        bias_att_set_SAE_stacked_cur_layer = []
        policy_risk_set_SAE_stacked_cur_layer = []

        bias_att_set_LR = []
        policy_risk_set_LR = []

        bias_att_set_LR_Lasso = []
        policy_risk_set_LR_Lasso = []

        for result in results_list:
            bias_att_set_NN.append(result["NN_bias_att"])
            policy_risk_set_NN.append(result["NN_policy_risk"])

            bias_att_set_SAE_E2E.append(result["SAE_e2e_bias_att"])
            policy_risk_set_SAE_E2E.append(result["SAE_e2e_policy_risk"])

            bias_att_set_SAE_stacked_all_layer_active.append(result["SAE_stacked_all_layer_active_bias_att"])
            policy_risk_set_SAE_stacked_all_layer_active.append(result["SAE_stacked_all_layer_active_policy_risk"])

            bias_att_set_SAE_stacked_cur_layer.append(result["SAE_stacked_cur_layer_active_bias_att"])
            policy_risk_set_SAE_stacked_cur_layer.append(result["SAE_stacked_cur_layer_active_policy_risk"])

            bias_att_set_LR.append(result["LR_bias_att"])
            policy_risk_set_LR.append(result["LR_policy_risk"])

            bias_att_set_LR_Lasso.append(result["LR_Lasso_bias_att"])
            policy_risk_set_LR_Lasso.append(result["LR_Lasso_policy_risk"])

        bias_att_set_NN_mean = np.mean(np.array(bias_att_set_NN))
        bias_att_set_NN_std = np.std(bias_att_set_NN)
        policy_risk_set_NN_mean = np.mean(np.array(policy_risk_set_NN))
        policy_risk_set_NN_std = np.std(policy_risk_set_NN)

        print("\n-------------------------------------------------\n")
        print("Using NN, bias_att: {0}, SD: {1}".format(bias_att_set_NN_mean, bias_att_set_NN_std))
        print("Using NN, policy_risk: {0}, SD: {1}".format(policy_risk_set_NN_mean, policy_risk_set_NN_std))
        print("\n-------------------------------------------------\n")

        bias_att_set_SAE_E2E_mean = np.mean(np.array(bias_att_set_SAE_E2E))
        bias_att_set_SAE_E2E_std = np.std(bias_att_set_SAE_E2E)
        policy_risk_set_SAE_E2E_mean = np.mean(np.array(policy_risk_set_SAE_E2E))
        policy_risk_set_SAE_E2E_std = np.std(policy_risk_set_SAE_E2E)

        print("Using SAE E2E, bias_att: {0}, SD: {1}".format(bias_att_set_SAE_E2E_mean, bias_att_set_SAE_E2E_std))
        print("Using SAE E2E, policy_risk: {0}, SD: {1}".format(policy_risk_set_SAE_E2E_mean,
                                                                policy_risk_set_SAE_E2E_std))
        print("\n-------------------------------------------------\n")

        bias_att_set_SAE_stacked_all_layer_active_mean = np.mean(np.array(bias_att_set_SAE_stacked_all_layer_active))
        bias_att_set_SAE_stacked_all_layer_active_std = np.std(bias_att_set_SAE_stacked_all_layer_active)
        policy_risk_set_SAE_stacked_all_layer_active_mean = np.mean(
            np.array(policy_risk_set_SAE_stacked_all_layer_active))
        policy_risk_set_SAE_stacked_all_layer_active_std = np.std(policy_risk_set_SAE_stacked_all_layer_active)

        print(
            "Using SAE stacked all layer active, bias_att: {0}, SD: {1}".format(
                bias_att_set_SAE_stacked_all_layer_active_mean,
                bias_att_set_SAE_stacked_all_layer_active_std))
        print("Using SAE stacked all layer active, policy_risk: {0}, SD: {1}".format(
            policy_risk_set_SAE_stacked_all_layer_active_mean,
            policy_risk_set_SAE_stacked_all_layer_active_std))
        print("\n-------------------------------------------------\n")

        bias_att_set_SAE_stacked_cur_layer_mean = np.mean(np.array(bias_att_set_SAE_stacked_cur_layer))
        bias_att_set_SAE_stacked_cur_layer_std = np.std(bias_att_set_SAE_stacked_cur_layer)
        policy_risk_set_SAE_stacked_cur_layer_mean = np.mean(np.array(policy_risk_set_SAE_stacked_cur_layer))
        policy_risk_set_SAE_stacked_cur_layer_std = np.std(policy_risk_set_SAE_stacked_cur_layer)

        print(
            "Using SAE stacked cur layer active, bias_att: {0}, SD: {1}".format(bias_att_set_SAE_stacked_cur_layer_mean,
                                                                                bias_att_set_SAE_stacked_cur_layer_std))
        print("Using SAE stacked cur layer active, policy_risk: {0}, SD: {1}".format(
            policy_risk_set_SAE_stacked_cur_layer_mean,
            policy_risk_set_SAE_stacked_cur_layer_std))

        print("\n-------------------------------------------------\n")

        bias_att_set_LR_mean = np.mean(np.array(bias_att_set_LR))
        bias_att_set_LR_std = np.std(bias_att_set_LR)
        policy_risk_set_LR_mean = np.mean(np.array(policy_risk_set_LR))
        policy_risk_set_LR_std = np.std(policy_risk_set_LR)
        print("Using Logistic Regression, bias_att: {0}, SD: {1}".format(bias_att_set_LR_mean, bias_att_set_LR_std))
        print("Using Logistic Regression, policy_risk: {0}, SD: {1}".format(policy_risk_set_LR_mean,
                                                                            policy_risk_set_LR_std))
        print("\n-------------------------------------------------\n")

        bias_att_set_LR_Lasso_mean = np.mean(np.array(bias_att_set_LR_Lasso))
        bias_att_set_LR_Lasso_std = np.std(bias_att_set_LR_Lasso)
        policy_risk_set_LR_Lasso_mean = np.mean(np.array(policy_risk_set_LR_Lasso))
        policy_risk_set_LR_Lasso_std = np.std(policy_risk_set_LR_Lasso)
        print("Using Lasso Logistic Regression, bias_att: {0}, SD: {1}".format(bias_att_set_LR_Lasso_mean,
                                                                               bias_att_set_LR_Lasso_std))
        print("Using Lasso Logistic Regression, policy_risk: {0}, SD: {1}".format(policy_risk_set_LR_Lasso_mean,
                                                                                  policy_risk_set_LR_Lasso_std))
        print("--" * 20)

        file1.write("\n##################################################")
        file1.write("\n")
        file1.write("\nUsing NN, bias att: {0}, SD: {1}".format(bias_att_set_NN_mean, bias_att_set_NN_std))
        file1.write("\nUsing NN, policy risk: {0}, SD: {1}".format(policy_risk_set_NN_mean, policy_risk_set_NN_std))
        file1.write("\n-------------------------------------------------\n")

        file1.write("Using SAE E2E, bias att: {0}, SD: {1}".format(bias_att_set_SAE_E2E_mean, bias_att_set_SAE_E2E_std))
        file1.write("\nUsing SAE E2E, policy risk: {0}, SD: {1}".format(policy_risk_set_SAE_E2E_mean,
                                                                        policy_risk_set_SAE_E2E_std))
        file1.write("\n-------------------------------------------------\n")

        file1.write(
            "Using SAE stacked all layer active, bias att: {0}, SD: {1}"
                .format(bias_att_set_SAE_stacked_all_layer_active_mean,
                        bias_att_set_SAE_stacked_all_layer_active_std))
        file1.write("\nUsing SAE stacked all layer active,  policy risk: {0}, SD: {1}"
                    .format(policy_risk_set_SAE_stacked_all_layer_active_mean,
                            policy_risk_set_SAE_stacked_all_layer_active_std))
        file1.write("\n-------------------------------------------------\n")

        file1.write("Using SAE stacked cur layer active, bias att: {0}, SD: {1}"
                    .format(bias_att_set_SAE_stacked_cur_layer_mean,
                            bias_att_set_SAE_stacked_cur_layer_std))
        file1.write("\nUsing SAE stacked cur layer active, policy risk: {0}, SD: {1}"
                    .format(policy_risk_set_SAE_stacked_cur_layer_mean,
                            policy_risk_set_SAE_stacked_cur_layer_mean))
        file1.write("\n-------------------------------------------------\n")
        file1.write("Using Logistic Regression, bias att: {0}, SD: {1}".format(bias_att_set_LR_mean,
                                                                               bias_att_set_LR_std))
        file1.write("\nUsing Logistic Regression, policy risk: {0}, SD: {1}".format(policy_risk_set_LR_mean,
                                                                                    policy_risk_set_LR_std))

        file1.write("\n-------------------------------------------------\n")
        file1.write(
            "Using Lasso Logistic Regression, bias att: {0}, SD: {1}".format(bias_att_set_LR_Lasso_mean,
                                                                             bias_att_set_LR_Lasso_std))
        file1.write("\nUsing Lasso Logistic Regression, policy risk: {0}, SD: {1}".format(policy_risk_set_LR_Lasso_mean,
                                                                                          policy_risk_set_LR_Lasso_std))
        file1.write("\n##################################################")

        Utils.write_to_csv(run_parameters["consolidated_file_path"], results_list)

    @staticmethod
    def __get_run_parameters(running_mode):
        run_parameters = {}
        if running_mode == "original_data":
            run_parameters["input_nodes"] = 17
            run_parameters["consolidated_file_path"] = "./MSE/Results_consolidated.csv"

            # NN
            run_parameters["nn_prop_file"] = "./MSE/NN_Prop_score_{0}.csv"
            run_parameters["nn_iter_file"] = "./MSE/ITE/ITE_NN_iter_{0}.csv"
            # SAE
            run_parameters["sae_e2e_prop_file"] = "./MSE/SAE_E2E_Prop_score_{0}.csv"
            run_parameters["sae_stacked_all_prop_file"] = "./MSE/SAE_stacked_all_Prop_score_{0}.csv"
            run_parameters["sae_stacked_cur_prop_file"] = "./MSE/SAE_stacked_cur_Prop_score_{0}.csv"

            run_parameters["sae_e2e_iter_file"] = "./MSE/ITE/ITE_SAE_E2E_iter_{0}.csv"
            run_parameters["sae_stacked_all_iter_file"] = "./MSE/ITE/ITE_SAE_stacked_all_iter_{0}.csv"
            run_parameters["sae_stacked_cur_iter_file"] = "./MSE/ITE/ITE_SAE_stacked_cur_Prop_iter_{0}.csv"

            # LR
            run_parameters["lr_prop_file"] = "./MSE/LR_Prop_score_{0}.csv"
            run_parameters["lr_iter_file"] = "./MSE/ITE/ITE_LR_iter_{0}.csv"

            # LR Lasso
            run_parameters["lr_lasso_prop_file"] = "./MSE/LR_lasso_Prop_score_{0}.csv"
            run_parameters["lr_lasso_iter_file"] = "./MSE/ITE/ITE_LR_Lasso_iter_{0}.csv"
            run_parameters["summary_file_name"] = "Details_original.txt"
            run_parameters["is_synthetic"] = False

        elif running_mode == "synthetic_data":
            run_parameters["input_nodes"] = 225
            run_parameters["consolidated_file_path"] = "./MSE_Augmented/Results_consolidated.csv"

            # NN
            run_parameters["nn_prop_file"] = "./MSE_Augmented/NN_Prop_score_{0}.csv"
            run_parameters["nn_iter_file"] = "./MSE_Augmented/ITE/ITE_NN_iter_{0}.csv"
            # SAE
            run_parameters["sae_e2e_prop_file"] = "./MSE_Augmented/SAE_E2E_Prop_score_{0}.csv"
            run_parameters["sae_stacked_all_prop_file"] = "./MSE_Augmented/SAE_stacked_all_Prop_score_{0}.csv"
            run_parameters["sae_stacked_cur_prop_file"] = "./MSE_Augmented/SAE_stacked_cur_Prop_score_{0}.csv"

            run_parameters["sae_e2e_iter_file"] = "./MSE_Augmented/ITE/ITE_SAE_E2E_iter_{0}.csv"
            run_parameters["sae_stacked_all_iter_file"] = "./MSE_Augmented/ITE/ITE_SAE_stacked_all_iter_{0}.csv"
            run_parameters["sae_stacked_cur_iter_file"] = "./MSE_Augmented/ITE/ITE_SAE_stacked_cur_Prop_iter_{0}.csv"

            # LR
            run_parameters["lr_prop_file"] = "./MSE_Augmented/LR_Prop_score_{0}.csv"
            run_parameters["lr_iter_file"] = "./MSE_Augmented/ITE/ITE_LR_iter_{0}.csv"

            # LR Lasso
            run_parameters["lr_lasso_prop_file"] = "./MSE_Augmented/LR_lasso_Prop_score_{0}.csv"
            run_parameters["lr_lasso_iter_file"] = "./MSE_Augmented/ITE/ITE_LR_Lasso_iter_{0}.csv"

            run_parameters["summary_file_name"] = "Details_augmented.txt"
            run_parameters["is_synthetic"] = True

        return run_parameters

    @staticmethod
    def load_data(running_mode, dL, train_path, test_path, iter_id):
        if running_mode == "original_data":
            return dL.preprocess_data_from_csv(train_path, test_path, iter_id)

        elif running_mode == "synthetic_data":
            return dL.preprocess_data_from_csv_augmented(train_path, test_path, iter_id)
