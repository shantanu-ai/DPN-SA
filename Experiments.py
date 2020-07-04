import statistics
from collections import OrderedDict

import numpy as np

from DCN_network import DCN_network
from DPN_SA_Deep import DPN_SA_Deep
from Propensity_score_LR import Propensity_socre_LR
from Propensity_socre_network import Propensity_socre_network
from Sparse_Propensity_score import Sparse_Propensity_score
from Utils import Utils
from dataloader import DataLoader


class Experiments:
    def run_all_experiments(self, iterations, running_mode):
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
        run_parameters = self.__get_run_parameters(running_mode)

        print(str(train_parameters_SAE))
        file1 = open(run_parameters["summary_file_name"], "a")
        file1.write(str(train_parameters_SAE))
        file1.write("\n")
        file1.write("Without batch norm")
        file1.write("\n")
        for iter_id in range(iterations):
            iter_id += 1
            print("--" * 20)
            print("iter_id: {0}".format(iter_id))
            print("--" * 20)
            # load data for propensity network
            dL = DataLoader()
            if running_mode == "original_data":
                np_covariates_X_train, np_covariates_X_test, np_covariates_Y_train, np_covariates_Y_test = \
                    dL.preprocess_data_from_csv(csv_path, split_size)
            elif running_mode == "synthetic_data":
                np_covariates_X_train, np_covariates_X_test, np_covariates_Y_train, np_covariates_Y_test = \
                    dL.preprocess_data_from_csv_augmented(csv_path, split_size)
            dp_sa = DPN_SA_Deep()
            trained_models = dp_sa.train_eval_DCN(iter_id,
                                                        np_covariates_X_train,
                                                        np_covariates_Y_train,
                                                        dL, device, run_parameters)

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

            MSE_SAE_e2e = reply["MSE_SAE_e2e"]
            MSE_SAE_stacked_all_layer_active = reply["MSE_SAE_stacked_all_layer_active"]
            MSE_SAE_stacked_cur_layer_active = reply["MSE_SAE_stacked_cur_layer_active"]
            MSE_NN = reply["MSE_NN"]
            MSE_LR = reply["MSE_LR"]
            MSE_LR_lasso = reply["MSE_LR_Lasso"]

            true_ATE_NN = reply["true_ATE_NN"]
            true_ATE_SAE_e2e = reply["true_ATE_SAE_e2e"]
            true_ATE_SAE_stacked_all_layer_active = reply["true_ATE_SAE_stacked_all_layer_active"]
            true_ATE_SAE_stacked_cur_layer_active = reply["true_ATE_SAE_stacked_cur_layer_active"]
            true_ATE_LR = reply["true_ATE_LR"]
            true_ATE_LR_Lasso = reply["true_ATE_LR_Lasso"]

            predicted_ATE_NN = reply["predicted_ATE_NN"]
            predicted_ATE_SAE_e2e = reply["predicted_ATE_SAE_e2e"]
            predicted_ATE_SAE_stacked_all_layer_active = reply["predicted_ATE_SAE_stacked_all_layer_active"]
            predicted_ATE_SAE_stacked_cur_layer_active = reply["predicted_ATE_SAE_stacked_cur_layer_active"]
            predicted_ATE_LR = reply["predicted_ATE_LR"]
            predicted_ATE_LR_Lasso = reply["predicted_ATE_LR_Lasso"]

            file1.write("Iter: {0}, MSE_Sparse_e2e: {1}, MSE_Sparse_stacked_all_layer_active: {2}, "
                        "MSE_Sparse_stacked_cur_layer_active: {3},"
                        " MSE_NN: {4}, MSE_LR: {5}, MSE_LR_Lasso: {6}\n"
                        .format(iter_id, MSE_SAE_e2e, MSE_SAE_stacked_all_layer_active,
                                MSE_SAE_stacked_cur_layer_active, MSE_NN, MSE_LR, MSE_LR_lasso))
            result_dict = OrderedDict()
            result_dict["iter_id"] = iter_id
            result_dict["MSE_NN"] = MSE_NN
            result_dict["MSE_SAE_e2e"] = MSE_SAE_e2e
            result_dict["MSE_SAE_stacked_all_layer_active"] = MSE_SAE_stacked_all_layer_active
            result_dict["MSE_SAE_stacked_cur_layer_active"] = MSE_SAE_stacked_cur_layer_active
            result_dict["MSE_LR"] = MSE_LR
            result_dict["MSE_LR_lasso"] = MSE_LR_lasso

            result_dict["true_ATE_NN"] = true_ATE_NN
            result_dict["true_ATE_SAE_e2e"] = true_ATE_SAE_e2e
            result_dict["true_ATE_SAE_stacked_all_layer_active"] = true_ATE_SAE_stacked_all_layer_active
            result_dict["true_ATE_SAE_stacked_cur_layer_active"] = true_ATE_SAE_stacked_cur_layer_active
            result_dict["true_ATE_LR"] = true_ATE_LR
            result_dict["true_ATE_LR_Lasso"] = true_ATE_LR_Lasso

            result_dict["predicted_ATE_NN"] = predicted_ATE_NN
            result_dict["predicted_ATE_SAE_e2e"] = predicted_ATE_SAE_e2e
            result_dict["predicted_ATE_SAE_stacked_all_layer_active"] = predicted_ATE_SAE_stacked_all_layer_active
            result_dict["predicted_ATE_SAE_stacked_cur_layer_active"] = predicted_ATE_SAE_stacked_cur_layer_active
            result_dict["predicted_ATE_LR"] = predicted_ATE_LR
            result_dict["predicted_ATE_LR_Lasso"] = predicted_ATE_LR_Lasso

            results_list.append(result_dict)

        MSE_set_NN = []
        MSE_set_SAE_e2e = []
        MSE_set_SAE_stacked_all_layer_active = []
        MSE_set_SAE_stacked_cur_layer_active = []
        MSE_set_LR = []
        MSE_set_LR_Lasso = []

        true_ATE_NN_set = []
        true_ATE_SAE_set_e2e = []
        true_ATE_SAE_set_stacked_all_layer_active = []
        true_ATE_SAE_set_stacked_cur_layer_active = []
        true_ATE_LR_set = []
        true_ATE_LR_Lasso_set = []

        predicted_ATE_NN_set = []
        predicted_ATE_SAE_set_e2e = []
        predicted_ATE_SAE_set_all_layer_active = []
        predicted_ATE_SAE_set_cur_layer_active = []
        predicted_ATE_LR_set = []
        predicted_ATE_LR_Lasso_set = []

        for result in results_list:
            MSE_set_NN.append(result["MSE_NN"])
            MSE_set_SAE_e2e.append(result["MSE_SAE_e2e"])
            MSE_set_SAE_stacked_all_layer_active.append(result["MSE_SAE_stacked_all_layer_active"])
            MSE_set_SAE_stacked_cur_layer_active.append(result["MSE_SAE_stacked_cur_layer_active"])
            MSE_set_LR.append(result["MSE_LR"])
            MSE_set_LR_Lasso.append(result["MSE_LR_lasso"])

            true_ATE_NN_set.append(result["true_ATE_NN"])
            true_ATE_SAE_set_e2e.append(result["true_ATE_SAE_e2e"])
            true_ATE_SAE_set_stacked_all_layer_active.append(result["true_ATE_SAE_stacked_all_layer_active"])
            true_ATE_SAE_set_stacked_cur_layer_active.append(result["true_ATE_SAE_stacked_all_layer_active"])
            true_ATE_LR_set.append(result["true_ATE_LR"])
            true_ATE_LR_Lasso_set.append(result["true_ATE_LR_Lasso"])

            predicted_ATE_NN_set.append(result["predicted_ATE_NN"])
            predicted_ATE_SAE_set_e2e.append(result["predicted_ATE_SAE_e2e"])
            predicted_ATE_SAE_set_all_layer_active.append(result["predicted_ATE_SAE_stacked_all_layer_active"])
            predicted_ATE_SAE_set_cur_layer_active.append(result["predicted_ATE_SAE_stacked_cur_layer_active"])
            predicted_ATE_LR_set.append(result["predicted_ATE_LR"])
            predicted_ATE_LR_Lasso_set.append(result["predicted_ATE_LR_Lasso"])

        MSE_total_NN = np.mean(np.array(MSE_set_NN))
        std_MSE_NN = statistics.pstdev(MSE_set_NN)
        Mean_ATE_NN_true = np.mean(np.array(true_ATE_NN_set))
        std_ATE_NN_true = statistics.pstdev(true_ATE_NN_set)
        Mean_ATE_NN_predicted = np.mean(np.array(predicted_ATE_NN_set))
        std_ATE_NN_predicted = statistics.pstdev(predicted_ATE_NN_set)

        print("\nWith Batch norm\n")
        print("\n-------------------------------------------------\n")
        print("Using NN, MSE: {0}, SD: {1}".format(MSE_total_NN, std_MSE_NN))
        print("Using NN, true ATE: {0}, SD: {1}".format(Mean_ATE_NN_true, std_ATE_NN_true))
        print("Using NN, predicted ATE: {0}, SD: {1}".format(Mean_ATE_NN_predicted, std_ATE_NN_predicted))
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
        print("\n-------------------------------------------------\n")

        MSE_total_SAE_stacked_all_layer_active = np.mean(np.array(MSE_set_SAE_stacked_all_layer_active))
        std_MSE_SAE_stacked_all_layer_active = statistics.pstdev(MSE_set_SAE_stacked_all_layer_active)
        Mean_ATE_SAE_true_stacked_all_layer_active = np.mean(np.array(true_ATE_SAE_set_stacked_all_layer_active))
        std_ATE_SAE_true_stacked_all_layer_active = statistics.pstdev(true_ATE_SAE_set_stacked_all_layer_active)
        Mean_ATE_SAE_predicted_all_layer_active = np.mean(np.array(predicted_ATE_SAE_set_all_layer_active))
        std_ATE_SAE_predicted_all_layer_active = statistics.pstdev(predicted_ATE_SAE_set_all_layer_active)

        print("Using SAE stacked all layer active, MSE: {0}, SD: {1}".format(MSE_total_SAE_stacked_all_layer_active,
                                                                             std_MSE_SAE_stacked_all_layer_active))
        print("Using SAE stacked all layer active, true ATE: {0}, SD: {1}".format(
            Mean_ATE_SAE_true_stacked_all_layer_active,
            std_ATE_SAE_true_stacked_all_layer_active))

        print("Using SAE stacked all layer active, predicted ATE: {0}, SD: {1}".format(
            Mean_ATE_SAE_predicted_all_layer_active,
            std_ATE_SAE_predicted_all_layer_active))
        print("\n-------------------------------------------------\n")

        MSE_total_SAE_stacked_cur_layer_active = np.mean(np.array(MSE_set_SAE_stacked_cur_layer_active))
        std_MSE_SAE_stacked_cur_layer_active = statistics.pstdev(MSE_set_SAE_stacked_cur_layer_active)
        Mean_ATE_SAE_true_stacked_cur_layer_active = np.mean(np.array(true_ATE_SAE_set_stacked_cur_layer_active))
        std_ATE_SAE_true_stacked_cur_layer_active = statistics.pstdev(true_ATE_SAE_set_stacked_cur_layer_active)
        Mean_ATE_SAE_predicted_cur_layer_active = np.mean(np.array(predicted_ATE_SAE_set_cur_layer_active))
        std_ATE_SAE_predicted_cur_layer_active = statistics.pstdev(predicted_ATE_SAE_set_cur_layer_active)

        print("Using SAE stacked cur layer active, MSE: {0}, SD: {1}".format(MSE_total_SAE_stacked_cur_layer_active,
                                                                             std_MSE_SAE_stacked_cur_layer_active))
        print("Using SAE stacked cur layer active, true ATE: {0}, SD: {1}".format(
            Mean_ATE_SAE_true_stacked_cur_layer_active,
            std_ATE_SAE_true_stacked_cur_layer_active))

        print("Using SAE stacked cur layer active, predicted ATE: {0}, SD: {1}".format(
            Mean_ATE_SAE_predicted_cur_layer_active,
            std_ATE_SAE_predicted_cur_layer_active))

        print("\n-------------------------------------------------\n")

        MSE_total_LR = np.mean(np.array(MSE_set_LR))
        std_MSE_LR = statistics.pstdev(MSE_set_LR)
        Mean_ATE_LR_true = np.mean(np.array(true_ATE_LR_set))
        std_ATE_LR_true = statistics.pstdev(true_ATE_LR_set)
        Mean_ATE_LR_predicted = np.mean(np.array(predicted_ATE_LR_set))
        std_ATE_LR_predicted = statistics.pstdev(predicted_ATE_LR_set)
        print("Using Logistic Regression, MSE: {0}, SD: {1}".format(MSE_total_LR, std_MSE_LR))
        print("Using Logistic Regression, true ATE: {0}, SD: {1}".format(Mean_ATE_LR_true, std_ATE_LR_true))
        print("Using Logistic Regression, predicted ATE: {0}, SD: {1}".format(Mean_ATE_LR_predicted,
                                                                              std_ATE_LR_predicted))
        print("\n-------------------------------------------------\n")

        MSE_total_LR_lasso = np.mean(np.array(MSE_set_LR_Lasso))
        std_MSE_LR_lasso = statistics.pstdev(MSE_set_LR_Lasso)
        Mean_ATE_LR_lasso_true = np.mean(np.array(true_ATE_LR_Lasso_set))
        std_ATE_LR_lasso_true = statistics.pstdev(true_ATE_LR_Lasso_set)
        Mean_ATE_LR_lasso_predicted = np.mean(np.array(predicted_ATE_LR_Lasso_set))
        std_ATE_LR_lasso_predicted = statistics.pstdev(predicted_ATE_LR_Lasso_set)
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
        file1.write("Using SAE E2E, MSE: {0}, SD: {1}".format(MSE_total_SAE_e2e, std_MSE_SAE_e2e))
        file1.write("\nUsing SAE E2E, true ATE: {0}, SD: {1}".format(Mean_ATE_SAE_true_e2e, std_ATE_SAE_true_e2e))
        file1.write("\nUsing SAE E2E, predicted ATE: {0}, SD: {1}".format(Mean_ATE_SAE_predicted_e2e,
                                                                          std_ATE_SAE_predicted_e2e))
        file1.write("\n-------------------------------------------------\n")
        file1.write("\n-------------------------------------------------\n")
        file1.write(
            "Using SAE stacked all layer active, MSE: {0}, SD: {1}".format(MSE_total_SAE_stacked_all_layer_active,
                                                                           std_MSE_SAE_stacked_all_layer_active))
        file1.write("\nUsing SAE stacked all layer active, true ATE: {0}, SD: {1}"
                    .format(Mean_ATE_SAE_true_stacked_all_layer_active,
                            std_ATE_SAE_true_stacked_all_layer_active))
        file1.write(
            "\nUsing SAE, stacked all layer active predicted ATE: {0}, SD: {1}"
                .format(Mean_ATE_SAE_predicted_all_layer_active,
                        std_ATE_SAE_predicted_all_layer_active))
        file1.write("\n-------------------------------------------------\n")
        file1.write("\n-------------------------------------------------\n")
        file1.write("Using SAE stacked cur layer active, MSE: {0}, SD: {1}"
                    .format(MSE_total_SAE_stacked_cur_layer_active,
                            std_MSE_SAE_stacked_cur_layer_active))
        file1.write("\nUsing SAE stacked cur layer active, true ATE: {0}, SD: {1}"
                    .format(Mean_ATE_SAE_true_stacked_cur_layer_active,
                            std_ATE_SAE_true_stacked_cur_layer_active))
        file1.write("\nUsing SAE stacked cur layer active, predicted ATE: {0}, SD: {1}"
                    .format(Mean_ATE_SAE_predicted_cur_layer_active,
                            std_ATE_SAE_predicted_cur_layer_active))
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

        Utils.write_to_csv(run_parameters["consolidated_file_path"], results_list)

    @staticmethod
    def __get_run_parameters(running_mode):
        run_parameters = {}
        if running_mode == "original_data":
            run_parameters["input_nodes"] = 25
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
            run_parameters["lr_prop_file"] = "./MSE/LR_lasso_Prop_score_{0}.csv"
            run_parameters["lr_iter_file"] = "./MSE/ITE/ITE_LR_Lasso_iter_{0}.csv"
            run_parameters["summary_file_name"] = "Details_original.txt"

        elif running_mode == "synthetic_data":
            run_parameters["input_nodes"] = 75
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
            run_parameters["lr_prop_file"] = "./MSE_Augmented/LR_lasso_Prop_score_{0}.csv"
            run_parameters["lr_iter_file"] = "./MSE_Augmented/ITE/ITE_LR_Lasso_iter_{0}.csv"
            run_parameters["summary_file_name"] = "Details_augmented.txt"

        return run_parameters


