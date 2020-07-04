from DCN_network import DCN_network
from Propensity_score_LR import Propensity_socre_LR
from Propensity_socre_network import Propensity_socre_network
from Sparse_Propensity_score import Sparse_Propensity_score
from Utils import Utils


class DPN_SA_Deep:
    def test_DCN(self, iter_id, np_covariates_X_test, np_covariates_Y_test, dL,
                 sparse_classifier,
                 sae_classifier_stacked_all_layer_active,
                 sae_classifier_stacked_cur_layer_active,
                 LR_model, LR_model_lasso, device,
                 run_parameters):
        print("----------- Testing phase ------------")
        ps_test_set = dL.convert_to_tensor(np_covariates_X_test,
                                           np_covariates_Y_test)

        # using NN
        MSE_NN, true_ATE_NN, predicted_ATE_NN = self.__test_DCN_NN(iter_id,
                                                                   np_covariates_X_test,
                                                                   np_covariates_Y_test,
                                                                   dL, device,
                                                                   ps_test_set,
                                                                   run_parameters["nn_prop_file"],
                                                                   run_parameters["nn_iter_file"],
                                                                   run_parameters["input_nodes"])

        # using SAE
        model_path_e2e = "./DCNModel/SAE_E2E_DCN_model_iter_id_{0}_epoch_100_lr_0.001.pth".format(iter_id)
        model_path_stacked_all = "./DCNModel/SAE_stacked_all_DCN_model_iter_id_{0}_epoch_100_lr_0.001.pth".format(
            iter_id)
        model_path_stacked_cur = "./DCNModel/SAE_stacked_cur_DCN_model_iter_id_{0}_epoch_100_lr_0.001.pth".format(
            iter_id)

        propensity_score_save_path_e2e = run_parameters["sae_e2e_prop_file"]
        propensity_score_save_path_stacked_all = run_parameters["sae_stacked_all_prop_file"]
        propensity_score_save_path_stacked_cur = run_parameters["sae_stacked_cur_prop_file"]

        ITE_save_path_e2e = run_parameters["sae_e2e_iter_file"]
        ITE_save_path_stacked_all = run_parameters["sae_stacked_all_iter_file"]
        ITE_save_path_stacked_cur = run_parameters["sae_stacked_cur_iter_file"]

        print("############### DCN Testing using SAE E2E ###############")
        MSE_SAE_e2e, true_ATE_SAE_e2e, predicted_ATE_SAE_e2e = \
            self.__test_DCN_SAE(iter_id, np_covariates_X_test,
                                np_covariates_Y_test, dL, device,
                                ps_test_set, sparse_classifier, model_path_e2e,
                                propensity_score_save_path_e2e, ITE_save_path_e2e)

        print("############### DCN Testing using SAE Stacked all layer active ###############")
        MSE_SAE_stacked_all_layer_active, true_ATE_SAE_stacked_all_layer_active, \
        predicted_ATE_SAE_stacked_all_layer_active = \
            self.__test_DCN_SAE(iter_id, np_covariates_X_test,
                                np_covariates_Y_test, dL, device,
                                ps_test_set,
                                sae_classifier_stacked_all_layer_active, model_path_stacked_all,
                                propensity_score_save_path_stacked_all,
                                ITE_save_path_stacked_all)

        print("############### DCN Testing using SAE cur layer active ###############")
        MSE_SAE_stacked_cur_layer_active, true_ATE_SAE_stacked_cur_layer_active, \
        predicted_ATE_SAE_stacked_cur_layer_active = \
            self.__test_DCN_SAE(iter_id, np_covariates_X_test,
                                np_covariates_Y_test, dL, device,
                                ps_test_set,
                                sae_classifier_stacked_cur_layer_active, model_path_stacked_cur,
                                propensity_score_save_path_stacked_cur,
                                ITE_save_path_stacked_cur)

        # using LR
        MSE_LR, true_ATE_LR, predicted_ATE_LR = self.__test_DCN_LR(np_covariates_X_test, np_covariates_Y_test,
                                                                   LR_model,
                                                                   iter_id, dL, device,
                                                                   run_parameters["lr_prop_file"],
                                                                   run_parameters["lr_iter_file"])

        # using LR Lasso
        MSE_LR_Lasso, true_ATE_LR_Lasso, predicted_ATE_LR_Lasso = self.__test_DCN_LR_Lasso(np_covariates_X_test,
                                                                                           np_covariates_Y_test,
                                                                                           LR_model_lasso,
                                                                                           iter_id, dL, device,
                                                                                           run_parameters["lr_prop_file"],
                                                                                           run_parameters["lr_iter_file"])

        return {
            "MSE_NN": MSE_NN,
            "true_ATE_NN": true_ATE_NN,
            "predicted_ATE_NN": predicted_ATE_NN,

            "MSE_SAE_e2e": MSE_SAE_e2e,
            "true_ATE_SAE_e2e": true_ATE_SAE_e2e,
            "predicted_ATE_SAE_e2e": predicted_ATE_SAE_e2e,

            "MSE_SAE_stacked_all_layer_active": MSE_SAE_stacked_all_layer_active,
            "true_ATE_SAE_stacked_all_layer_active": true_ATE_SAE_stacked_all_layer_active,
            "predicted_ATE_SAE_stacked_all_layer_active": predicted_ATE_SAE_stacked_all_layer_active,

            "MSE_SAE_stacked_cur_layer_active": MSE_SAE_stacked_cur_layer_active,
            "true_ATE_SAE_stacked_cur_layer_active": true_ATE_SAE_stacked_cur_layer_active,
            "predicted_ATE_SAE_stacked_cur_layer_active": predicted_ATE_SAE_stacked_cur_layer_active,

            "MSE_LR": MSE_LR,
            "true_ATE_LR": true_ATE_LR,
            "predicted_ATE_LR": predicted_ATE_LR,
            "MSE_LR_Lasso": MSE_LR_Lasso,
            "true_ATE_LR_Lasso": true_ATE_LR_Lasso,
            "predicted_ATE_LR_Lasso": predicted_ATE_LR_Lasso
        }

    def train_eval_DCN(self, iter_id, np_covariates_X_train, np_covariates_Y_train, dL, device, run_parameters):
        print("----------- Training and evaluation phase ------------")
        ps_train_set = dL.convert_to_tensor(np_covariates_X_train, np_covariates_Y_train)

        # using NN
        self.__train_propensity_net_NN(ps_train_set, np_covariates_X_train, np_covariates_Y_train, dL,
                                       iter_id, device, run_parameters["input_nodes"])

        # using SAE
        sparse_classifier, sae_classifier_stacked_all_layer_active, sae_classifier_stacked_cur_layer_active = \
            self.__train_propensity_net_SAE(ps_train_set, np_covariates_X_train,
                                            np_covariates_Y_train, dL,
                                            iter_id, device, run_parameters["input_nodes"])
        # using Logistic Regression
        LR_model = self.__train_propensity_net_LR(np_covariates_X_train, np_covariates_Y_train,
                                                  dL,
                                                  iter_id, device)

        # using Logistic Regression Lasso
        LR_model_lasso = self.__train_propensity_net_LR_Lasso(np_covariates_X_train,
                                                              np_covariates_Y_train,
                                                              dL,
                                                              iter_id, device)

        return {
            "sparse_classifier": sparse_classifier,
            "sae_classifier_stacked_all_layer_active": sae_classifier_stacked_all_layer_active,
            "sae_classifier_stacked_cur_layer_active": sae_classifier_stacked_cur_layer_active,
            "LR_model": LR_model,
            "LR_model_lasso": LR_model_lasso
        }

    def __train_propensity_net_NN(self, ps_train_set, np_covariates_X_train,
                                  np_covariates_Y_train, dL,
                                  iter_id, device, input_nodes):
        train_parameters_NN = {
            "epochs": 75,
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
        eval_parameters_NN = {
            "eval_set": ps_train_set,
            "model_path": "./Propensity_Model/NN_PS_model_iter_id_{0}_epoch_75_lr_0.001.pth"
                .format(iter_id),
            "input_nodes": input_nodes
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
                                   iter_id, device, input_nodes):
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
            "input_nodes": input_nodes
        }

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
        print("----------End to End SAE training----------")

        self.__train_DCN_SAE(ps_net_SAE, ps_train_set, device, np_covariates_X_train,
                             np_covariates_Y_train, iter_id, dL, sparse_classifier, model_path_e2e)
        print("----------Layer wise greedy stacked SAE training - All layers----------")

        self.__train_DCN_SAE(ps_net_SAE, ps_train_set, device, np_covariates_X_train,
                             np_covariates_Y_train, iter_id, dL, sae_classifier_stacked_all_layer_active,
                             model_path_stacked_all)
        print("----------Layer wise greedy stacked SAE training - Current layers----------")

        self.__train_DCN_SAE(ps_net_SAE, ps_train_set, device, np_covariates_X_train,
                             np_covariates_Y_train, iter_id, dL, sae_classifier_stacked_cur_layer_active,
                             model_path_stacked_cur)

        return sparse_classifier, sae_classifier_stacked_all_layer_active, sae_classifier_stacked_cur_layer_active

    def __train_DCN_SAE(self, ps_net_SAE, ps_train_set, device, np_covariates_X_train,
                        np_covariates_Y_train, iter_id, dL, sparse_classifier, model_path):
        # eval propensity network using SAE
        ps_score_list_SAE = ps_net_SAE.eval(ps_train_set, device, phase="eval",
                                            sparse_classifier=sparse_classifier)

        # load data for ITE network using SAE
        print("############### DCN Training using SAE ###############")
        data_loader_dict_SAE = dL.prepare_tensor_for_DCN(np_covariates_X_train,
                                                         np_covariates_Y_train,
                                                         ps_score_list_SAE)

        self.__train_DCN(data_loader_dict_SAE, model_path, dL, device)

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

    def __test_DCN_NN(self, iter_id, np_covariates_X_test, np_covariates_Y_test, dL, device, ps_test_set,
                      prop_score_file, iter_file, input_nodes):
        # testing using NN
        ps_net_NN = Propensity_socre_network()
        ps_eval_parameters_NN = {
            "eval_set": ps_test_set,
            "model_path": "./Propensity_Model/NN_PS_model_iter_id_{0}_epoch_75_lr_0.001.pth".format(iter_id),
            "input_nodes": input_nodes
        }
        ps_score_list_NN = ps_net_NN.eval(ps_eval_parameters_NN, device, phase="eval")
        Utils.write_to_csv(prop_score_file.format(iter_id), ps_score_list_NN)

        # load data for ITE network using vanilla network
        print("############### DCN Testing using NN ###############")
        data_loader_dict_NN = dL.prepare_tensor_for_DCN(np_covariates_X_test,
                                                        np_covariates_Y_test,
                                                        ps_score_list_NN)
        model_path = "./DCNModel/NN_DCN_model_iter_id_{0}_epoch_100_lr_0.001.pth".format(iter_id)
        MSE_NN, true_ATE_NN, predicted_ATE_NN, ITE_dict_list = self.__do_test_DCN(data_loader_dict_NN,
                                                                                  dL, device, model_path)
        Utils.write_to_csv(iter_file.format(iter_id), ITE_dict_list)

        return MSE_NN, true_ATE_NN, predicted_ATE_NN

    def __test_DCN_SAE(self, iter_id, np_covariates_X_test, np_covariates_Y_test, dL, device,
                       ps_test_set, sparse_classifier, model_path, propensity_score_csv_path,
                       ite_csv_path):
        # testing using SAE
        ps_net_SAE = Sparse_Propensity_score()
        ps_score_list_SAE = ps_net_SAE.eval(ps_test_set, device, phase="eval",
                                            sparse_classifier=sparse_classifier)
        Utils.write_to_csv(propensity_score_csv_path.format(iter_id), ps_score_list_SAE)

        # load data for ITE network using SAE
        data_loader_dict_SAE = dL.prepare_tensor_for_DCN(np_covariates_X_test,
                                                         np_covariates_Y_test,
                                                         ps_score_list_SAE)
        MSE_SAE, true_ATE_SAE, predicted_ATE_SAE, ITE_dict_list = self.__do_test_DCN(data_loader_dict_SAE,
                                                                                     dL, device,
                                                                                     model_path)

        Utils.write_to_csv(ite_csv_path.format(iter_id), ITE_dict_list)
        return MSE_SAE, true_ATE_SAE, predicted_ATE_SAE

    def __test_DCN_LR(self, np_covariates_X_test, np_covariates_Y_test, LR_model, iter_id, dL, device,
                      prop_score_file, iter_file):
        # testing using Logistic Regression
        ps_score_list_LR = Propensity_socre_LR.test(np_covariates_X_test,
                                                    np_covariates_Y_test,
                                                    log_reg=LR_model)
        Utils.write_to_csv(prop_score_file.format(iter_id), ps_score_list_LR)

        # load data for ITE network using Logistic Regression
        data_loader_dict_SAE = dL.prepare_tensor_for_DCN(np_covariates_X_test,
                                                         np_covariates_Y_test,
                                                         ps_score_list_LR)
        print("############### DCN Testing using LR ###############")
        model_path = "./DCNModel/LR_DCN_model_iter_id_{0}_epoch_100_lr_0.001.pth".format(iter_id)
        MSE_LR, true_ATE_LR, predicted_ATE_LR, ITE_dict_list = self.__do_test_DCN(data_loader_dict_SAE, dL,
                                                                                  device, model_path)
        Utils.write_to_csv(iter_file.format(iter_id), ITE_dict_list)
        return MSE_LR, true_ATE_LR, predicted_ATE_LR

    def __test_DCN_LR_Lasso(self, np_covariates_X_test, np_covariates_Y_test, LR_model_lasso,
                            iter_id, dL, device, prop_score_file, iter_file):
        # testing using Logistic Regression Lasso
        ps_score_list_LR_lasso = Propensity_socre_LR.test(np_covariates_X_test,
                                                          np_covariates_Y_test,
                                                          log_reg=LR_model_lasso)

        Utils.write_to_csv(prop_score_file.format(iter_id), ps_score_list_LR_lasso)

        # load data for ITE network using Logistic Regression Lasso
        data_loader_dict_SAE = dL.prepare_tensor_for_DCN(np_covariates_X_test,
                                                         np_covariates_Y_test,
                                                         ps_score_list_LR_lasso)
        print("############### DCN Testing using LR Lasso ###############")
        model_path = "./DCNModel/LR_Lasso_DCN_model_iter_id_{0}_epoch_100_lr_0.001.pth".format(iter_id)

        MSE_LR_Lasso, true_ATE_LR_Lasso, predicted_ATE_LR_Lasso, ITE_dict_list = \
            self.__do_test_DCN(data_loader_dict_SAE, dL,
                               device, model_path)
        Utils.write_to_csv(iter_file.format(iter_id), ITE_dict_list)
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

        return MSE, true_ATE, predicted_ATE, response_dict["ITE_dict_list"]
