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

from DCN_network import DCN_network
from Utils import Utils
from dataloader import DataLoader
from shallow_train import shallow_train


class Model_25_1_25:
    def run_all_expeiments(self):
        train_path = "Dataset/jobs_DW_bin.new.10.train.npz"
        test_path = "Dataset/jobs_DW_bin.new.10.test.npz"
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
        for iter_id in range(9):
            print("--" * 20)
            print("iter_id: {0}".format(iter_id))
            print("--" * 20)
            # load data for propensity network
            dL = DataLoader()
            np_covariates_X_train, np_covariates_X_test, np_covariates_Y_train, np_covariates_Y_test = \
                dL.preprocess_data_from_csv(train_path, test_path, iter_id)

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

            SAE_e2e_ate_pred = reply["SAE_e2e_ate_pred"]
            SAE_e2e_att_pred = reply["SAE_e2e_att_pred"]
            SAE_e2e_bias_att = reply["SAE_e2e_bias_att"]
            SAE_e2e_atc_pred = reply["SAE_e2e_atc_pred"]
            SAE_e2e_policy_value = reply["SAE_e2e_policy_value"]
            SAE_e2e_policy_risk = reply["SAE_e2e_policy_risk"]
            SAE_e2e_err_fact = reply["SAE_e2e_err_fact"]

            file1.write("Iter: {0}, "
                        "SAE_e2e_bias_att: {1},  "
                        "SAE_e2e_policy_risk: {2}, "
                        .format(iter_id,
                                SAE_e2e_bias_att,
                                SAE_e2e_policy_risk))
            result_dict = OrderedDict()
            result_dict["SAE_e2e_ate_pred"] = SAE_e2e_ate_pred
            result_dict["SAE_e2e_att_pred"] = SAE_e2e_att_pred
            result_dict["SAE_e2e_bias_att"] = SAE_e2e_bias_att
            result_dict["SAE_e2e_atc_pred"] = SAE_e2e_atc_pred
            result_dict["SAE_e2e_policy_value"] = SAE_e2e_policy_value
            result_dict["SAE_e2e_policy_risk"] = SAE_e2e_policy_risk
            result_dict["SAE_e2e_err_fact"] = SAE_e2e_err_fact

            results_list.append(result_dict)

        bias_att_set_SAE_E2E = []
        policy_risk_set_SAE_E2E = []

        for result in results_list:
            bias_att_set_SAE_E2E.append(result["SAE_e2e_bias_att"])
            policy_risk_set_SAE_E2E.append(result["SAE_e2e_policy_risk"])

        print("\n-------------------------------------------------\n")

        bias_att_set_SAE_E2E_mean = np.mean(np.array(bias_att_set_SAE_E2E))
        bias_att_set_SAE_E2E_std = np.std(bias_att_set_SAE_E2E)
        policy_risk_set_SAE_E2E_mean = np.mean(np.array(policy_risk_set_SAE_E2E))
        policy_risk_set_SAE_E2E_std = np.std(policy_risk_set_SAE_E2E)

        print("Using SAE E2E, bias_att: {0}, SD: {1}".format(bias_att_set_SAE_E2E_mean, bias_att_set_SAE_E2E_std))
        print("Using SAE E2E, policy_risk: {0}, SD: {1}".format(policy_risk_set_SAE_E2E_mean,
                                                                policy_risk_set_SAE_E2E_std))
        print("\n-------------------------------------------------\n")
        print("--" * 20)

        file1.write("Using SAE E2E, bias att: {0}, SD: {1}".format(bias_att_set_SAE_E2E_mean, bias_att_set_SAE_E2E_std))
        file1.write("\nUsing SAE E2E, policy risk: {0}, SD: {1}".format(policy_risk_set_SAE_E2E_mean,
                                                                        policy_risk_set_SAE_E2E_std))
        file1.write("\n-------------------------------------------------\n")

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
                                                         is_synthetic=False)

        self.__train_DCN(data_loader_dict_SAE, model_path, dL, device)

    def __train_DCN(self, data_loader_dict_train, model_path, dL, device):
        tensor_treated_train = self.create_tensors_from_tuple(data_loader_dict_train["treated_data"])
        tensor_control_train = self.create_tensors_from_tuple(data_loader_dict_train["control_data"])

        DCN_train_parameters = {
            "epochs": 100,
            "lr": 0.001,
            "treated_batch_size": 1,
            "control_batch_size": 1,
            "shuffle": True,
            "treated_set_train": tensor_treated_train,
            "control_set_train": tensor_control_train,
            "model_save_path": model_path,
            "input_nodes": 17
        }

        # train DCN network
        dcn = DCN_network()
        dcn.train(DCN_train_parameters, device)

    @staticmethod
    def create_tensors_from_tuple(group, test_set_flag=False):
        np_df_X = group[0]
        np_ps_score = group[1]
        np_df_Y_f = group[2]
        if test_set_flag:
            np_df_e = group[3]
            tensor = Utils.convert_to_tensor_DCN_test(np_df_X, np_ps_score,
                                                      np_df_Y_f, np_df_e)
        else:
            tensor = Utils.convert_to_tensor_DCN(np_df_X, np_ps_score,
                                                 np_df_Y_f)
        return tensor

    def __test_DCN(self, iter_id, np_covariates_X_test, np_covariates_Y_test, dL, sparse_classifier, device):
        print("----------- Testing phase ------------")
        ps_test_set = dL.convert_to_tensor(np_covariates_X_test, np_covariates_Y_test)

        # using SAE
        model_path_e2e = "./DCNModel/SAE_E2E_DCN_model_iter_id_{0}_epoch_100_lr_0.001.pth".format(iter_id)

        propensity_score_save_path_e2e = "./MSE/SAE_E2E_Prop_score_{0}.csv"

        ITE_save_path_e2e = "./MSE/ITE/ITE_SAE_E2E_iter_{0}.csv"

        print("############### DCN Testing using SAE E2E ###############")
        ate_pred, att_pred, bias_att, atc_pred, policy_value, \
        policy_risk, err_fact = \
            self.__test_DCN_SAE(iter_id, np_covariates_X_test,
                                np_covariates_Y_test, dL, device,
                                ps_test_set, sparse_classifier, model_path_e2e,
                                propensity_score_save_path_e2e, ITE_save_path_e2e)

        return {

            "SAE_e2e_ate_pred": ate_pred,
            "SAE_e2e_att_pred": att_pred,
            "SAE_e2e_bias_att": bias_att,
            "SAE_e2e_atc_pred": atc_pred,
            "SAE_e2e_policy_value": policy_value,
            "SAE_e2e_policy_risk": policy_risk,
            "SAE_e2e_err_fact": err_fact

        }

    def __test_DCN_SAE(self, iter_id, np_covariates_X_test, np_covariates_Y_test, dL, device,
                       ps_test_set, sparse_classifier, model_path, propensity_score_csv_path,
                       iter_file):
        # testing using SAE
        ps_net_SAE = shallow_train()
        ps_score_list_SAE = ps_net_SAE.eval(ps_test_set, device, phase="eval",
                                            sparse_classifier=sparse_classifier)
        Utils.write_to_csv(propensity_score_csv_path.format(iter_id), ps_score_list_SAE)

        # load data for ITE network using SAE
        data_loader_dict_SAE = dL.prepare_tensor_for_DCN(np_covariates_X_test,
                                                         np_covariates_Y_test,
                                                         ps_score_list_SAE,
                                                         is_synthetic=False)
        ate_pred, att_pred, bias_att, atc_pred, policy_value, \
        policy_risk, err_fact = \
            self.__do_test_DCN(data_loader_dict_SAE, dL,
                               device, model_path, 17, iter_file, iter_id)

        return ate_pred, att_pred, bias_att, atc_pred, policy_value, \
               policy_risk, err_fact

    def __do_test_DCN(self, data_loader_dict, dL, device, model_path, input_nodes, iter_file, iter_id):
        t_1 = np.ones(data_loader_dict["treated_data"][0].shape[0])

        t_0 = np.zeros(data_loader_dict["control_data"][0].shape[0])

        tensor_treated = \
            Utils.create_tensors_from_tuple_test(data_loader_dict["treated_data"], t_1)
        tensor_control = \
            Utils.create_tensors_from_tuple_test(data_loader_dict["control_data"], t_0)

        DCN_test_parameters = {
            "treated_set": tensor_treated,
            "control_set": tensor_control,
            "model_save_path": model_path
        }

        dcn = DCN_network()
        dcn_pd_models_eval_dict = dcn.eval(DCN_test_parameters, device, input_nodes)

        ate_pred, att_pred, bias_att, atc_pred, policy_value, \
        policy_risk, err_fact = \
            self.__process_evaluated_metric(
                dcn_pd_models_eval_dict["yf_list"],
                dcn_pd_models_eval_dict["e_list"],
                dcn_pd_models_eval_dict["T_list"],
                dcn_pd_models_eval_dict["y1_hat_list"],
                dcn_pd_models_eval_dict["y0_hat_list"],
                dcn_pd_models_eval_dict["ITE_dict_list"],
                dcn_pd_models_eval_dict["predicted_ITE"],
                iter_file,
                iter_id)

        return ate_pred, att_pred, bias_att, atc_pred, policy_value, \
               policy_risk, err_fact

    def __process_evaluated_metric(self, y_f, e, T,
                                   y1_hat, y0_hat,
                                   ite_dict, predicted_ITE_list,
                                   ite_csv_path,
                                   iter_id):
        y1_hat_np = np.array(y1_hat)
        y0_hat_np = np.array(y0_hat)
        e_np = np.array(e)
        t_np = np.array(T)
        np_y_f = np.array(y_f)

        y1_hat_np_b = 1.0 * (y1_hat_np > 0.5)
        y0_hat_np_b = 1.0 * (y0_hat_np > 0.5)

        err_fact = np.mean(np.abs(y1_hat_np_b - np_y_f))
        att = np.mean(np_y_f[t_np > 0]) - np.mean(np_y_f[(1 - t_np + e_np) > 1])

        eff_pred = y0_hat_np - y1_hat_np
        eff_pred[t_np > 0] = -eff_pred[t_np > 0]

        ate_pred = np.mean(eff_pred[e_np > 0])
        atc_pred = np.mean(eff_pred[(1 - t_np + e_np) > 1])

        att_pred = np.mean(eff_pred[(t_np + e_np) > 1])
        bias_att = np.abs(att_pred - att)

        policy_value = self.cal_policy_val(t_np[e_np > 0], np_y_f[e_np > 0],
                                           eff_pred[e_np > 0])

        print("bias_att: " + str(bias_att))
        print("policy_value: " + str(policy_value))
        print("Risk: " + str(1 - policy_value))
        print("atc_pred: " + str(atc_pred))
        print("att_pred: " + str(att_pred))
        print("err_fact: " + str(err_fact))

        Utils.write_to_csv(ite_csv_path.format(iter_id), ite_dict)
        return ate_pred, att_pred, bias_att, atc_pred, policy_value, 1 - policy_value, err_fact

    @staticmethod
    def cal_policy_val(t, yf, eff_pred):
        #  policy_val(t[e>0], yf[e>0], eff_pred[e>0], compute_policy_curve)

        if np.any(np.isnan(eff_pred)):
            return np.nan, np.nan

        policy = eff_pred > 0
        treat_overlap = (policy == t) * (t > 0)
        control_overlap = (policy == t) * (t < 1)

        if np.sum(treat_overlap) == 0:
            treat_value = 0
        else:
            treat_value = np.mean(yf[treat_overlap])

        if np.sum(control_overlap) == 0:
            control_value = 0
        else:
            control_value = np.mean(yf[control_overlap])

        pit = np.mean(policy)
        policy_value = pit * treat_value + (1 - pit) * control_value

        return policy_value
