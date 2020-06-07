from collections import OrderedDict

import pandas as pd

from DCN_network import DCN_network
from Propensity_score_SAE import Propensity_socre_SAE
from Propensity_socre_network import Propensity_socre_network
from Utils import Utils
from dataloader import DataLoader


def train_eval_DCN(iter_id, np_covariates_X_train, np_covariates_Y_train, dL, device):
    print("----------- Training and evaluation phase ------------")
    ps_train_set = dL.convert_to_tensor(np_covariates_X_train, np_covariates_Y_train)

    # test set -> np_covariates_Y_train, np_covariates
    # train propensity network
    train_parameters = {
        "epochs": 100,
        "lr": 0.001,
        "batch_size": 32,
        "shuffle": True,
        "train_set": ps_train_set,
        "model_save_path": "./Propensity_Model/NN_PS_model_iter_id_" + str(iter_id) + "_epoch_{0}_lr_{1}.pth"
    }
    # ps using NN
    ps_net_NN = Propensity_socre_network()
    print("############### Propensity Score neural net Training ###############")
    ps_net_NN.train(train_parameters, device, phase="train")

    # ps using SAE
    train_parameters = {
        "epochs": 100,
        "lr": 0.001,
        "batch_size": 32,
        "shuffle": True,
        "train_set": ps_train_set,
        "sparsity_probability": 0.05,
        "weight_decay": 0.0005,
        "BETA": 0.001,
        "model_save_path": "./Propensity_Model/SAE_PS_model_iter_id_" + str(iter_id) + "_epoch_{0}_lr_{1}.pth",
    }
    ps_net_SAE = Propensity_socre_SAE()
    print("############### Propensity Score SAE net Training ###############")
    ps_net_SAE.train(train_parameters, device, phase="train")

    eval_parameters_NN = {
        "eval_set": ps_train_set,
        "model_path": "./Propensity_Model/NN_PS_model_iter_id_{0}_epoch_100_lr_0.001.pth".format(iter_id)
    }
    # eval propensity network usig NN
    ps_score_list_NN = ps_net_NN.eval(eval_parameters_NN, device, phase="eval")

    # eval propensity network usig SAE
    eval_parameters_SAE = {
        "eval_set": ps_train_set,
        "model_path": "./Propensity_Model/SAE_PS_model_iter_id_{0}_epoch_100_lr_0.001.pth".format(iter_id)
    }
    ps_score_list_SAE = ps_net_SAE.eval(eval_parameters_SAE, device, phase="eval")

    # load data for ITE network
    print("############### DCN Training using NN ###############")
    data_loader_dict_NN = dL.prepare_tensor_for_DCN(np_covariates_X_train,
                                                    np_covariates_Y_train,
                                                    ps_score_list_NN)
    model_path = "./DCNModel/NN_DCN_model_iter_id_" + str(iter_id) + "_epoch_{0}_lr_{1}.pth"
    train_DCN(data_loader_dict_NN, model_path, dL, device)

    print("############### DCN Training using SAE ###############")
    data_loader_dict_SAE = dL.prepare_tensor_for_DCN(np_covariates_X_train,
                                                     np_covariates_Y_train,
                                                     ps_score_list_SAE)
    model_path = "./DCNModel/SAE_DCN_model_iter_id_" + str(iter_id) + "_epoch_{0}_lr_{1}.pth"
    train_DCN(data_loader_dict_SAE, model_path, dL, device)


def train_DCN(data_loader_dict, model_path, dL, device):
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


def test_DCN(iter_id, np_covariates_X_test, np_covariates_Y_test, dL, device):
    print("----------- Testing phase ------------")
    ps_test_set = dL.convert_to_tensor(np_covariates_X_test, np_covariates_Y_test)
    # testing using NN
    ps_net_NN = Propensity_socre_network()
    ps_eval_parameters_NN = {
        "eval_set": ps_test_set,
        "model_path": "./Propensity_Model/NN_PS_model_iter_id_{0}_epoch_100_lr_0.001.pth".format(iter_id)
    }
    ps_score_list_NN = ps_net_NN.eval(ps_eval_parameters_NN, device, phase="eval")
    pd.DataFrame.from_dict(
        ps_score_list_NN,
        orient='columns'
    ).to_csv("./MSE/Prop_score_{0}.csv".format(iter_id))

    # testing using SAE
    ps_net_SAE = Propensity_socre_SAE()
    ps_eval_parameters_SAE = {
        "eval_set": ps_test_set,
        "model_path": "./Propensity_Model/SAE_PS_model_iter_id_{0}_epoch_100_lr_0.001.pth".format(iter_id)
    }
    ps_score_list_SAE = ps_net_SAE.eval(ps_eval_parameters_SAE, device, phase="eval")
    pd.DataFrame.from_dict(
        ps_score_list_SAE,
        orient='columns'
    ).to_csv("./MSE/Prop_score_{0}.csv".format(iter_id))

    # load data for ITE network
    print("############### DCN Testing using NN ###############")
    data_loader_dict_NN = dL.prepare_tensor_for_DCN(np_covariates_X_test,
                                                    np_covariates_Y_test,
                                                    ps_score_list_NN)
    model_path = "./DCNModel/NN_DCN_model_iter_id_{0}_epoch_100_lr_0.001.pth".format(iter_id)
    MSE_NN = do_test_DCN(data_loader_dict_NN, dL, device, model_path)

    print("############### DCN Testing using SAE ###############")
    data_loader_dict_SAE = dL.prepare_tensor_for_DCN(np_covariates_X_test,
                                                     np_covariates_Y_test,
                                                     ps_score_list_SAE)
    model_path = "./DCNModel/SAE_DCN_model_iter_id_{0}_epoch_100_lr_0.001.pth".format(iter_id)
    MSE_SAE = do_test_DCN(data_loader_dict_SAE, dL, device, model_path)

    return MSE_NN, MSE_SAE


def do_test_DCN(data_loader_dict, dL, device, model_path):
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
    err_dict = dcn.eval(DCN_test_parameters, device)
    err_treated = [ele ** 2 for ele in err_dict["treated_err"]]
    err_control = [ele ** 2 for ele in err_dict["control_err"]]

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
    return MSE
    # np.save("treated_err.npy", err_treated)
    # np.save("control_err.npy", err_control)


def main_propensity_dropout_BL():
    csv_path = "Dataset/ihdp_sample.csv"
    split_size = 0.8
    device = Utils.get_device()
    print(device)
    MSE_list_NN = []
    MSE_set_NN = []

    MSE_list_SAE = []
    MSE_set_SAE = []
    for iter_id in range(1):
        iter_id += 1
        print("--" * 20)
        print("iter_id: {0}".format(iter_id))
        print("--" * 20)
        # load data for propensity network
        dL = DataLoader()
        np_covariates_X_train, np_covariates_X_test, np_covariates_Y_train, np_covariates_Y_test = \
            dL.preprocess_data_from_csv(csv_path, split_size)

        train_eval_DCN(iter_id, np_covariates_X_train, np_covariates_Y_train, dL, device)

        # test DCN network
        MSE_NN, MSE_SAE = test_DCN(iter_id, np_covariates_X_test, np_covariates_Y_test, dL, device)

        MSE_dict_NN = OrderedDict()
        MSE_dict_NN[iter_id] = MSE_NN
        MSE_set_NN.append(MSE_NN)
        MSE_list_NN.append(MSE_dict_NN)

        MSE_dict_SAE = OrderedDict()
        MSE_dict_SAE[iter_id] = MSE_SAE
        MSE_set_SAE.append(MSE_SAE)
        MSE_list_SAE.append(MSE_dict_SAE)

    print("---> NN statistics: <--- ")
    print("--" * 20)
    for _dict in MSE_list_NN:
        print(_dict)
    print("--" * 20)

    print("---> SAE statistics: <--- ")
    print("--" * 20)
    for _dict in MSE_list_SAE:
        print(_dict)
    print("--" * 20)

    print("---> Overall statistics: <--- ")
    print("--" * 20)
    MSE_total_NN = sum(MSE_set_NN) / len(MSE_set_NN)
    print("Mean squared error using NN: {0}".format(MSE_total_NN))

    MSE_total_SAE = sum(MSE_set_SAE) / len(MSE_set_SAE)
    print("Mean squared error using SAE: {0}".format(MSE_total_SAE))
    print("--" * 20)

    pd.DataFrame.from_dict(
        MSE_set_NN,
        orient='columns'
    ).to_csv("./MSE/MSE_dict_NN.csv")

    pd.DataFrame.from_dict(
        MSE_set_NN,
        orient='columns'
    ).to_csv("./MSE/MSE_dict_SAE.csv")


if __name__ == '__main__':
    main_propensity_dropout_BL()
