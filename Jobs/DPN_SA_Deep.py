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
from datetime import datetime

import numpy as np

from DCN_network import DCN_network
from Propensity_score_LR import Propensity_socre_LR
from Propensity_socre_network import Propensity_socre_network
from Sparse_Propensity_score import Sparse_Propensity_score
from Utils import Utils


class DPN_SA_Deep:
    def train_eval_DCN(self, iter_id, np_covariates_X_train,
                       np_covariates_Y_train,
                       dL, device,
                       run_parameters,
                       is_synthetic=False):
        print("----------- Training and evaluation phase ------------")
        ps_train_set = dL.convert_to_tensor(np_covariates_X_train, np_covariates_Y_train)

        # using NN
        start = datetime.now()
        self.__train_propensity_net_NN(ps_train_set,
                                       np_covariates_X_train,
                                       np_covariates_Y_train,
                                       dL,
                                       iter_id, device, run_parameters["input_nodes"],
                                       is_synthetic)
        end = datetime.now()
        print("Neural Net start time: =", start)
        print("Neural Net end time: =", end)
        # diff = start - end
        # diff_minutes = divmod(diff.seconds, 60)
        # print('Time to train: ', diff_minutes[0], 'minutes',
        #       diff_minutes[1], 'seconds')

        # using SAE
        start = datetime.now()
        sparse_classifier, \
        sae_classifier_stacked_all_layer_active, sae_classifier_stacked_cur_layer_active = \
            self.__train_propensity_net_SAE(ps_train_set,
                                            np_covariates_X_train,
                                            np_covariates_Y_train,
                                            dL,
                                            iter_id, device,
                                            run_parameters["input_nodes"],
                                            is_synthetic)

        # using Logistic Regression
        start = datetime.now()
        LR_model = self.__train_propensity_net_LR(np_covariates_X_train, np_covariates_Y_train,
                                                  dL,
                                                  iter_id, device,
                                                  run_parameters["input_nodes"],
                                                  is_synthetic)
        end = datetime.now()
        print("Logistic Regression start time: =", start)
        print("Logistic Regression end time: =", end)
        diff = start - end
        # diff_minutes = divmod(diff.seconds, 60)
        # print('Time to train: ', diff_minutes[0], 'minutes',
        #       diff_minutes[1], 'seconds')

        # using Logistic Regression Lasso
        start = datetime.now()
        LR_model_lasso = self.__train_propensity_net_LR_Lasso(np_covariates_X_train,
                                                              np_covariates_Y_train,
                                                              dL,
                                                              iter_id, device,
                                                              run_parameters["input_nodes"],
                                                              is_synthetic)
        end = datetime.now()
        print("Logistic Regression Lasso start time: =", start)
        print("Logistic Regression Lasso end time: =", end)
        diff = start - end
        # diff_minutes = divmod(diff.seconds, 60)
        # print('Time to train: ', diff_minutes[0], 'minutes',
        #       diff_minutes[1], 'seconds')

        return {
            "sparse_classifier": sparse_classifier,
            "sae_classifier_stacked_all_layer_active": sae_classifier_stacked_all_layer_active,
            "sae_classifier_stacked_cur_layer_active": sae_classifier_stacked_cur_layer_active,
            "LR_model": LR_model,
            "LR_model_lasso": LR_model_lasso
        }

    def test_DCN(self, iter_id, np_covariates_X_test, np_covariates_Y_test, dL,
                 sparse_classifier,
                 sae_classifier_stacked_all_layer_active,
                 sae_classifier_stacked_cur_layer_active,
                 LR_model, LR_model_lasso, device,
                 run_parameters):
        print("----------- Testing phase ------------")
        ps_test_set = dL.convert_to_tensor(np_covariates_X_test,
                                           np_covariates_Y_test)
        print("############### DCN Testing using NN ###############")
        # using NN
        # MSE_NN, true_ATE_NN, predicted_ATE_NN \
        NN_ate_pred, NN_att_pred, NN_bias_att, NN_atc_pred, NN_policy_value, \
        NN_policy_risk, NN_err_fact = self.__test_DCN_NN(iter_id,
                                                         np_covariates_X_test,
                                                         np_covariates_Y_test,
                                                         dL, device,
                                                         ps_test_set,
                                                         run_parameters["nn_prop_file"],
                                                         run_parameters["nn_iter_file"],
                                                         run_parameters["is_synthetic"],
                                                         run_parameters["input_nodes"])

        # using SAE
        model_path_e2e = "./DCNModel/SAE_E2E_DCN_model_iter_id_{0}_epoch_100_lr_0.0001.pth".format(iter_id)
        model_path_stacked_all = "./DCNModel/SAE_stacked_all_DCN_model_iter_id_{0}_epoch_100_lr_0.0001.pth".format(
            iter_id)
        model_path_stacked_cur = "./DCNModel/SAE_stacked_cur_DCN_model_iter_id_{0}_epoch_100_lr_0.0001.pth".format(
            iter_id)

        propensity_score_save_path_e2e = run_parameters["sae_e2e_prop_file"]
        propensity_score_save_path_stacked_all = run_parameters["sae_stacked_all_prop_file"]
        propensity_score_save_path_stacked_cur = run_parameters["sae_stacked_cur_prop_file"]

        ITE_save_path_e2e = run_parameters["sae_e2e_iter_file"]
        ITE_save_path_stacked_all = run_parameters["sae_stacked_all_iter_file"]
        ITE_save_path_stacked_cur = run_parameters["sae_stacked_cur_iter_file"]

        # MSE_SAE_e2e = 0
        # true_ATE_SAE_e2e = 0
        # predicted_ATE_SAE_e2e = 0
        #
        # MSE_SAE_stacked_all_layer_active = 0
        # true_ATE_SAE_stacked_all_layer_active = 0
        # predicted_ATE_SAE_stacked_all_layer_active = 0
        #
        # MSE_SAE_stacked_cur_layer_active = 0
        # true_ATE_SAE_stacked_cur_layer_active = 0
        # predicted_ATE_SAE_stacked_cur_layer_active = 0
        #
        # MSE_LR = 0
        # true_ATE_LR = 0
        # predicted_ATE_LR = 0
        # MSE_LR_Lasso = 0
        # true_ATE_LR_Lasso = 0
        # predicted_ATE_LR_Lasso = 0

        print("############### DCN Testing using SAE E2E ###############")

        SAE_e2e_ate_pred, SAE_e2e_att_pred, SAE_e2e_bias_att, SAE_e2e_atc_pred, SAE_e2e_policy_value, \
        SAE_e2e_policy_risk, SAE_e2e_err_fact = \
            self.__test_DCN_SAE(iter_id, np_covariates_X_test,
                                np_covariates_Y_test, dL, device,
                                ps_test_set, sparse_classifier, model_path_e2e,
                                propensity_score_save_path_e2e, ITE_save_path_e2e,
                                run_parameters["is_synthetic"],
                                run_parameters["input_nodes"])

        print("############### DCN Testing using SAE Stacked all layer active ###############")
        SAE_stacked_all_layer_active_ate_pred, SAE_stacked_all_layer_active_att_pred, \
        SAE_stacked_all_layer_active_bias_att, SAE_stacked_all_layer_active_atc_pred, \
        SAE_stacked_all_layer_active_policy_value, \
        SAE_stacked_all_layer_active_policy_risk, SAE_stacked_all_layer_active_err_fact = \
            self.__test_DCN_SAE(iter_id, np_covariates_X_test,
                                np_covariates_Y_test, dL, device,
                                ps_test_set,
                                sae_classifier_stacked_all_layer_active, model_path_stacked_all,
                                propensity_score_save_path_stacked_all,
                                ITE_save_path_stacked_all,
                                run_parameters["is_synthetic"],
                                run_parameters["input_nodes"])

        print("############### DCN Testing using SAE cur layer active ###############")
        SAE_stacked_cur_layer_active_ate_pred, SAE_stacked_cur_layer_active_att_pred, \
        SAE_stacked_cur_layer_active_bias_att, SAE_stacked_cur_layer_active_atc_pred, \
        SAE_stacked_cur_layer_active_policy_value, \
        SAE_stacked_cur_layer_active_policy_risk, SAE_stacked_cur_layer_active_err_fact = \
            self.__test_DCN_SAE(iter_id, np_covariates_X_test,
                                np_covariates_Y_test, dL, device,
                                ps_test_set,
                                sae_classifier_stacked_cur_layer_active, model_path_stacked_cur,
                                propensity_score_save_path_stacked_cur,
                                ITE_save_path_stacked_cur,
                                run_parameters["is_synthetic"],
                                run_parameters["input_nodes"])

        # using LR
        LR_ate_pred, LR_att_pred, \
        LR_bias_att, LR_atc_pred, \
        LR_policy_value, \
        LR_policy_risk, LR_err_fact = self.__test_DCN_LR(np_covariates_X_test, np_covariates_Y_test,
                                                         LR_model,
                                                         iter_id, dL, device,
                                                         run_parameters["lr_prop_file"],
                                                         run_parameters["lr_iter_file"],
                                                         run_parameters["is_synthetic"],
                                                         run_parameters["input_nodes"])

        # using LR Lasso
        LR_Lasso_ate_pred, LR_Lasso_att_pred, \
        LR_Lasso_bias_att, LR_Lasso_atc_pred, \
        LR_Lasso_policy_value, \
        LR_Lasso_policy_risk, LR_Lasso_err_fact = self.__test_DCN_LR_Lasso(np_covariates_X_test,
                                                                           np_covariates_Y_test,
                                                                           LR_model_lasso,
                                                                           iter_id, dL, device,
                                                                           run_parameters[
                                                                               "lr_lasso_prop_file"],
                                                                           run_parameters[
                                                                               "lr_lasso_iter_file"],
                                                                           run_parameters[
                                                                               "is_synthetic"],
                                                                           run_parameters[
                                                                               "input_nodes"])

        return {
            "NN_ate_pred": NN_ate_pred,
            "NN_att_pred": NN_att_pred,
            "NN_bias_att": NN_bias_att,
            "NN_atc_pred": NN_atc_pred,
            "NN_policy_value": NN_policy_value,
            "NN_policy_risk": NN_policy_risk,
            "NN_err_fact": NN_err_fact,

            "SAE_e2e_ate_pred": SAE_e2e_ate_pred,
            "SAE_e2e_att_pred": SAE_e2e_att_pred,
            "SAE_e2e_bias_att": SAE_e2e_bias_att,
            "SAE_e2e_atc_pred": SAE_e2e_atc_pred,
            "SAE_e2e_policy_value": SAE_e2e_policy_value,
            "SAE_e2e_policy_risk": SAE_e2e_policy_risk,
            "SAE_e2e_err_fact": SAE_e2e_policy_risk,

            "SAE_stacked_all_layer_active_ate_pred": SAE_stacked_all_layer_active_ate_pred,
            "SAE_stacked_all_layer_active_att_pred": SAE_stacked_all_layer_active_ate_pred,
            "SAE_stacked_all_layer_active_bias_att": SAE_stacked_all_layer_active_bias_att,
            "SAE_stacked_all_layer_active_atc_pred": SAE_stacked_all_layer_active_bias_att,
            "SAE_stacked_all_layer_active_policy_value": SAE_stacked_all_layer_active_policy_value,
            "SAE_stacked_all_layer_active_policy_risk": SAE_stacked_all_layer_active_policy_risk,
            "SAE_stacked_all_layer_active_err_fact": SAE_stacked_all_layer_active_err_fact,

            "SAE_stacked_cur_layer_active_ate_pred": SAE_stacked_cur_layer_active_ate_pred,
            "SAE_stacked_cur_layer_active_att_pred": SAE_stacked_cur_layer_active_att_pred,
            "SAE_stacked_cur_layer_active_bias_att": SAE_stacked_cur_layer_active_bias_att,
            "SAE_stacked_cur_layer_active_atc_pred": SAE_stacked_cur_layer_active_atc_pred,
            "SAE_stacked_cur_layer_active_policy_value": SAE_stacked_cur_layer_active_policy_value,
            "SAE_stacked_cur_layer_active_policy_risk": SAE_stacked_cur_layer_active_policy_risk,
            "SAE_stacked_cur_layer_active_err_fact": SAE_stacked_cur_layer_active_err_fact,

            "LR_ate_pred": LR_ate_pred,
            "LR_att_pred": LR_att_pred,
            "LR_bias_att": LR_bias_att,
            "LR_atc_pred": LR_atc_pred,
            "LR_policy_value": LR_policy_value,
            "LR_policy_risk": LR_policy_risk,
            "LR_err_fact": LR_err_fact,

            "LR_Lasso_ate_pred": LR_Lasso_ate_pred,
            "LR_Lasso_att_pred": LR_Lasso_att_pred,
            "LR_Lasso_bias_att": LR_Lasso_bias_att,
            "LR_Lasso_atc_pred": LR_Lasso_atc_pred,
            "LR_Lasso_policy_value": LR_Lasso_policy_value,
            "LR_Lasso_policy_risk": LR_Lasso_policy_risk,
            "LR_Lasso_err_fact": LR_Lasso_err_fact

        }

    def __train_propensity_net_NN(self, ps_train_set,
                                  np_covariates_X_train,
                                  np_covariates_Y_train, dL,
                                  iter_id, device, input_nodes, is_synthetic):
        train_parameters_NN = {
            "epochs": 50,
            "lr": 0.001,
            "batch_size": 32,
            "shuffle": True,
            "train_set": ps_train_set,
            "model_save_path": "./Propensity_Model/NN_PS_model_iter_id_"
                               + str(iter_id) + "_epoch_{0}_lr_{1}.pth",
            "input_nodes": input_nodes
        }
        # ps using NN
        ps_net_NN = Propensity_socre_network()
        print("############### Propensity Score neural net Training ###############")
        ps_net_NN.train(train_parameters_NN, device, phase="train")

        # eval
        eval_parameters_train_NN = {
            "eval_set": ps_train_set,
            "model_path": "./Propensity_Model/NN_PS_model_iter_id_{0}_epoch_{1}_lr_0.001.pth"
                .format(iter_id, train_parameters_NN["epochs"]),
            "input_nodes": input_nodes
        }

        ps_score_list_train_NN = ps_net_NN.eval(eval_parameters_train_NN, device, phase="eval")

        # train DCN
        print("############### DCN Training using NN ###############")
        data_loader_dict_train_NN = dL.prepare_tensor_for_DCN(np_covariates_X_train,
                                                              np_covariates_Y_train,
                                                              ps_score_list_train_NN,
                                                              is_synthetic)

        model_path = "./DCNModel/NN_DCN_model_iter_id_" + str(iter_id) + "_epoch_{0}_lr_{1}.pth"
        self.__train_DCN(data_loader_dict_train_NN, model_path, dL, device,
                         input_nodes)

    def __train_propensity_net_SAE(self,
                                   ps_train_set, np_covariates_X_train, np_covariates_Y_train,
                                   dL,
                                   iter_id, device, input_nodes, is_synthetic):
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
            "input_nodes": input_nodes
        }

        # train_parameters_SAE = {
        #     "epochs": 2000,
        #     "lr": 0.001,
        #     "batch_size": 32,
        #     "shuffle": True,
        #     "train_set": ps_train_set,
        #     "sparsity_probability": 0.8,
        #     "weight_decay": 0.0003,
        #     "BETA": 0.1,
        #     "input_nodes": input_nodes
        # }

        print(str(train_parameters_SAE))
        ps_net_SAE = Sparse_Propensity_score()
        print("############### Propensity Score SAE net Training ###############")
        sparse_classifier, sae_classifier_stacked_all_layer_active, \
        sae_classifier_stacked_cur_layer_active = ps_net_SAE.train(train_parameters_SAE, device, phase="train")

        # eval propensity network using SAE
        model_path_e2e = "./DCNModel/SAE_E2E_DCN_model_iter_id_" + str(iter_id) + "_epoch_{0}_lr_{1}.pth"
        model_path_stacked_all = "./DCNModel/SAE_stacked_all_DCN_model_iter_id_" + \
                                 str(iter_id) + "_epoch_{0}_lr_{1}.pth"
        model_path_stacked_cur = "./DCNModel/SAE_stacked_cur_DCN_model_iter_id_" + \
                                 str(iter_id) + "_epoch_{0}_lr_{1}.pth"
        print("---" * 25)
        print("End to End SAE training")
        print("---" * 25)

        start = datetime.now()
        self.__train_DCN_SAE(ps_net_SAE, ps_train_set, device, np_covariates_X_train,
                             np_covariates_Y_train,
                             iter_id, dL, sparse_classifier,
                             model_path_e2e, input_nodes, is_synthetic)
        end = datetime.now()
        print("SAE E2E start time: =", start)
        print("SAE E2E end time: =", end)
        # diff = start - end
        # diff_minutes = divmod(diff.seconds, 60)
        # print('Time to train: ', diff_minutes[0], 'minutes',
        #       diff_minutes[1], 'seconds')

        print("---" * 25)
        print("----------Layer wise greedy stacked SAE training - All layers----------")
        print("---" * 25)
        start = datetime.now()
        self.__train_DCN_SAE(ps_net_SAE, ps_train_set, device, np_covariates_X_train,
                             np_covariates_Y_train, iter_id, dL,
                             sae_classifier_stacked_all_layer_active,
                             model_path_stacked_all, input_nodes, is_synthetic)

        end = datetime.now()
        print("SAE all layer active start time: =", start)
        print("SAE all layer active end time: =", end)
        # diff = start - end
        # diff_minutes = divmod(diff.seconds, 60)
        # print('Time to train: ', diff_minutes[0], 'minutes',
        #       diff_minutes[1], 'seconds')

        print("---" * 25)
        print("Layer wise greedy stacked SAE training - Current layers")
        print("---" * 25)
        start = datetime.now()
        self.__train_DCN_SAE(ps_net_SAE, ps_train_set, device, np_covariates_X_train,
                             np_covariates_Y_train, iter_id, dL,
                             sae_classifier_stacked_cur_layer_active,
                             model_path_stacked_cur, input_nodes, is_synthetic)

        end = datetime.now()
        print("SAE cur layer active start time: =", start)
        print("SAE all layer active end time: =", end)
        # diff = start - end
        # diff_minutes = divmod(diff.seconds, 60)
        # print('Time to train: ', diff_minutes[0], 'minutes',
        #       diff_minutes[1], 'seconds')

        return sparse_classifier, sae_classifier_stacked_all_layer_active, sae_classifier_stacked_cur_layer_active

    def __train_DCN_SAE(self, ps_net_SAE, ps_train_set,
                        device, np_covariates_X_train,
                        np_covariates_Y_train,
                        iter_id, dL, sparse_classifier,
                        model_path,
                        input_nodes, is_synthetic):
        # eval propensity network using SAE
        ps_score_list_train_SAE = ps_net_SAE.eval(ps_train_set, device, phase="eval",
                                                  sparse_classifier=sparse_classifier)

        # load data for ITE network using SAE
        print("############### DCN Training using SAE ###############")
        data_loader_dict_train_SAE = dL.prepare_tensor_for_DCN(np_covariates_X_train,
                                                               np_covariates_Y_train,
                                                               ps_score_list_train_SAE,
                                                               is_synthetic)

        self.__train_DCN(data_loader_dict_train_SAE,
                         model_path, dL, device, input_nodes)

    def __train_propensity_net_LR(self, np_covariates_X_train, np_covariates_Y_train,
                                  dL, iter_id, device, input_nodes, is_synthetic):
        # eval propensity network using Logistic Regression
        ps_score_list_LR, LR_model = Propensity_socre_LR.train(np_covariates_X_train,
                                                               np_covariates_Y_train)

        # load data for ITE network using Logistic Regression
        print("############### DCN Training using Logistic Regression ###############")
        data_loader_dict_LR = dL.prepare_tensor_for_DCN(np_covariates_X_train,
                                                        np_covariates_Y_train,
                                                        ps_score_list_LR,
                                                        is_synthetic)
        model_path = "./DCNModel/LR_DCN_model_iter_id_" + str(iter_id) + "_epoch_{0}_lr_{1}.pth"
        self.__train_DCN(data_loader_dict_LR, model_path, dL, device, input_nodes)

        return LR_model

    def __train_propensity_net_LR_Lasso(self, np_covariates_X_train, np_covariates_Y_train,
                                        dL, iter_id, device, input_nodes, is_synthetic):
        # eval propensity network using Logistic Regression Lasso
        ps_score_list_LR_lasso, LR_model_lasso = Propensity_socre_LR.train(np_covariates_X_train,
                                                                           np_covariates_Y_train,
                                                                           regularized=True)
        # load data for ITE network using Logistic Regression Lasso
        print("############### DCN Training using Logistic Regression Lasso ###############")
        data_loader_dict_LR_lasso = dL.prepare_tensor_for_DCN(np_covariates_X_train,
                                                              np_covariates_Y_train,
                                                              ps_score_list_LR_lasso,
                                                              is_synthetic)
        model_path = "./DCNModel/LR_Lasso_DCN_model_iter_id_" + str(iter_id) + "_epoch_{0}_lr_{1}.pth"
        self.__train_DCN(data_loader_dict_LR_lasso, model_path, dL, device, input_nodes)

        return LR_model_lasso

    def __train_DCN(self, data_loader_dict_train, model_path, dL, device, input_nodes):
        tensor_treated_train = self.create_tensors_from_tuple(data_loader_dict_train["treated_data"])
        tensor_control_train = self.create_tensors_from_tuple(data_loader_dict_train["control_data"])

        DCN_train_parameters = {
            "epochs": 100,
            "lr": 0.0001,
            "treated_batch_size": 1,
            "control_batch_size": 1,
            "shuffle": True,
            "treated_set_train": tensor_treated_train,
            "control_set_train": tensor_control_train,
            "model_save_path": model_path,
            "input_nodes": input_nodes
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

    def __test_DCN_NN(self, iter_id, np_covariates_X_test, np_covariates_Y_test, dL, device,
                      ps_test_set,
                      prop_score_file, iter_file, is_synthetic, input_nodes):
        # testing using NN
        ps_net_NN = Propensity_socre_network()
        ps_eval_parameters_NN = {
            "eval_set": ps_test_set,
            "model_path": "./Propensity_Model/NN_PS_model_iter_id_{0}_epoch_50_lr_0.001.pth".format(iter_id),
            "input_nodes": input_nodes
        }
        ps_score_list_NN = ps_net_NN.eval(ps_eval_parameters_NN, device, phase="eval")
        Utils.write_to_csv(prop_score_file.format(iter_id), ps_score_list_NN)

        # load data for ITE network using vanilla network

        data_loader_dict_NN = dL.prepare_tensor_for_DCN(np_covariates_X_test,
                                                        np_covariates_Y_test,
                                                        ps_score_list_NN,
                                                        is_synthetic)
        model_path = "./DCNModel/NN_DCN_model_iter_id_{0}_epoch_100_lr_0.0001.pth".format(iter_id)
        ate_pred, att_pred, bias_att, atc_pred, policy_value, \
        policy_risk, err_fact = self.__do_test_DCN(data_loader_dict_NN,
                                                   dL, device,
                                                   model_path,
                                                   input_nodes,
                                                   iter_file,
                                                   iter_id)

        return ate_pred, att_pred, bias_att, atc_pred, policy_value, \
               policy_risk, err_fact

    def __test_DCN_SAE(self, iter_id, np_covariates_X_test, np_covariates_Y_test, dL, device,
                       ps_test_set, sparse_classifier, model_path, propensity_score_csv_path,
                       iter_file, is_synthetic, input_nodes):
        # testing using SAE
        ps_net_SAE = Sparse_Propensity_score()
        ps_score_list_SAE = ps_net_SAE.eval(ps_test_set, device, phase="eval",
                                            sparse_classifier=sparse_classifier)
        Utils.write_to_csv(propensity_score_csv_path.format(iter_id), ps_score_list_SAE)

        # load data for ITE network using SAE
        data_loader_dict_SAE = dL.prepare_tensor_for_DCN(np_covariates_X_test,
                                                         np_covariates_Y_test,
                                                         ps_score_list_SAE,
                                                         is_synthetic)
        ate_pred, att_pred, bias_att, atc_pred, policy_value, \
        policy_risk, err_fact \
            = self.__do_test_DCN(data_loader_dict_SAE,
                                 dL, device,
                                 model_path,
                                 input_nodes,
                                 iter_file,
                                 iter_id)
        return ate_pred, att_pred, bias_att, atc_pred, policy_value, \
               policy_risk, err_fact

    def __test_DCN_LR(self, np_covariates_X_test, np_covariates_Y_test, LR_model, iter_id, dL, device,
                      prop_score_file, iter_file, is_synthetic, input_nodes):
        # testing using Logistic Regression
        ps_score_list_LR = Propensity_socre_LR.test(np_covariates_X_test,
                                                    np_covariates_Y_test,
                                                    log_reg=LR_model)
        Utils.write_to_csv(prop_score_file.format(iter_id), ps_score_list_LR)

        # load data for ITE network using Logistic Regression
        data_loader_dict_SAE = dL.prepare_tensor_for_DCN(np_covariates_X_test,
                                                         np_covariates_Y_test,
                                                         ps_score_list_LR,
                                                         is_synthetic)
        print("############### DCN Testing using LR ###############")
        model_path = "./DCNModel/LR_DCN_model_iter_id_{0}_epoch_100_lr_0.0001.pth".format(iter_id)
        ate_pred, att_pred, bias_att, atc_pred, policy_value, \
        policy_risk, err_fact = \
            self.__do_test_DCN(data_loader_dict_SAE, dL,
                               device, model_path,
                               input_nodes, iter_file, iter_id)
        return ate_pred, att_pred, bias_att, atc_pred, policy_value, \
               policy_risk, err_fact

    def __test_DCN_LR_Lasso(self, np_covariates_X_test, np_covariates_Y_test, LR_model_lasso,
                            iter_id, dL, device, prop_score_file, iter_file,
                            is_synthetic, input_nodes):
        # testing using Logistic Regression Lasso
        ps_score_list_LR_lasso = Propensity_socre_LR.test(np_covariates_X_test,
                                                          np_covariates_Y_test,
                                                          log_reg=LR_model_lasso)

        Utils.write_to_csv(prop_score_file.format(iter_id), ps_score_list_LR_lasso)

        # load data for ITE network using Logistic Regression Lasso
        data_loader_dict_SAE = dL.prepare_tensor_for_DCN(np_covariates_X_test,
                                                         np_covariates_Y_test,
                                                         ps_score_list_LR_lasso,
                                                         is_synthetic)
        print("############### DCN Testing using LR Lasso ###############")
        model_path = "./DCNModel/LR_Lasso_DCN_model_iter_id_{0}_epoch_100_lr_0.0001.pth".format(iter_id)

        ate_pred, att_pred, bias_att, atc_pred, policy_value, \
        policy_risk, err_fact = \
            self.__do_test_DCN(data_loader_dict_SAE, dL,
                               device, model_path, input_nodes, iter_file, iter_id)
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
