import numpy as np

from DCN_network import DCN_network
from Propensity_socre_network import Propensity_socre_network
from Utils import Utils
from dataloader import DataLoader


def train_eval_DCN(np_covariates_X_train, np_covariates_Y_train, dL, device):
    ps_train_set = dL.convert_to_tensor(np_covariates_X_train, np_covariates_Y_train)

    # test set -> np_covariates_Y_train, np_covariates
    # train propensity network
    train_parameters = {
        "epochs": 250,
        "lr": 0.001,
        "batch_size": 32,
        "shuffle": True,
        "train_set": ps_train_set,
        "model_save_path": "./Propensity_Model/PS_model_epoch_{0}_lr_{1}.pth"
    }
    ps_net = Propensity_socre_network()
    ps_net.train(train_parameters, device, phase="train")

    # eval propensity network
    eval_parameters = {
        "eval_set": ps_train_set,
        "model_path": "./Propensity_Model/PS_model_epoch_250_lr_0.001.pth"
    }
    ps_score_list = ps_net.eval(eval_parameters, device, phase="eval")

    # load data for ITE network
    print("############### DCN Training ###############")
    data_loader_dict = dL.prepare_tensor_for_DCN(np_covariates_X_train,
                                                 np_covariates_Y_train,
                                                 ps_score_list)
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
        "epochs": 500,
        "lr": 0.001,
        "treated_batch_size": 1,
        "control_batch_size": 1,
        "shuffle": True,
        "treated_set": tensor_treated,
        "control_set": tensor_control,
        "model_save_path": "./DCNModel/DCN_model_epoch_{0}_lr_{1}.pth"
    }

    # train DCN network
    dcn = DCN_network()
    dcn.train(DCN_train_parameters, device)


def test_DCN(np_covariates_X_test, np_covariates_Y_test, dL, device):
    ps_test_set = dL.convert_to_tensor(np_covariates_X_test, np_covariates_Y_test)
    ps_net = Propensity_socre_network()
    eval_parameters = {
        "eval_set": ps_test_set,
        "model_path": "./Propensity_Model/PS_model_epoch_250_lr_0.001.pth"
    }
    ps_score_list = ps_net.eval(eval_parameters, device, phase="eval")
    # load data for ITE network
    print("############### DCN Testing ###############")
    data_loader_dict = dL.prepare_tensor_for_DCN(np_covariates_X_test,
                                                 np_covariates_Y_test,
                                                 ps_score_list)
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
        "model_save_path": "./DCNModel/DCN_model_epoch_500_lr_0.001.pth"
    }

    dcn = DCN_network()
    err_dict = dcn.eval(DCN_test_parameters, device)
    err_treated = [abs(ele) for ele in err_dict["treated_err"]]
    err_control = [abs(ele) for ele in err_dict["control_err"]]

    total_sum = sum(err_treated) + sum(err_control)
    total_item = len(err_treated) + len(err_control)
    print("MSE: {0}".format(total_sum / total_item))

    np.save("treated_err.npy", err_treated)
    np.save("control_err.npy", err_control)


def main_propensity_dropout_BL():
    csv_path = "Dataset/ihdp_sample.csv"
    split_size = 0.8
    device = Utils.get_device()
    print(device)
    # do this for #100 iterations
    # # load data for propensity network
    dL = DataLoader()
    # np_covariates_X_train, np_covariates_X_test, np_covariates_Y_train, np_covariates_Y_test = \
    #     dL.preprocess_data_from_csv(csv_path, split_size)
    #
    # print(".. Saving test file ..")
    # np.save("np_covariates_X_test.npy", np_covariates_X_test)
    # np.save("np_covariates_Y_test.npy", np_covariates_Y_test)
    #
    # train_eval_DCN(np_covariates_X_train, np_covariates_Y_train, dL, device)

    # test DCN network
    np_covariates_X_test = np.load("np_covariates_X_test.npy")
    np_covariates_Y_test = np.load("np_covariates_Y_test.npy")
    test_DCN(np_covariates_X_test, np_covariates_Y_test, dL, device)


if __name__ == '__main__':
    main_propensity_dropout_BL()
