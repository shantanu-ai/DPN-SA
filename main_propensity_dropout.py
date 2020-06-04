from Propensity_socre_network import Propensity_socre_network
from Utils import Utils
from dataloader import DataLoader


def main_propensity_dropout_BL():
    csv_path = "Dataset/ihdp_sample.csv"
    split_size = 0.8
    device = Utils.get_device()
    print(device)
    # do this for #100 iterations
    # # load data for propensity network
    dL = DataLoader()
    np_covariates_X_train, np_covariates_X_test, np_covariates_Y_train, np_covariates_Y_test = \
        dL.preprocess_data_from_csv(csv_path, split_size)
    ps_train_set = dL.convert_to_tensor(np_covariates_X_train, np_covariates_Y_train)

    # test set -> np_covariates_Y_train, np_covariates_Y_test

    # train propensity network
    ps_net = Propensity_socre_network()
    train_parameters = {
        "epochs": 250,
        "lr": 0.001,
        "batch_size": 32,
        "shuffle": True,
        "train_set": ps_train_set,
        "model_save_path": "./Propensity_Model/PS_model_epoch_{0}_lr_{1}.pth"
    }

    # ps_net.train(train_parameters, device, phase="train")

    # eval propensity network
    eval_parameters = {
        "eval_set": ps_train_set,
        "model_path": "./Propensity_Model/PS_model_epoch_250_lr_0.001.pth"
    }
    ps_score_list = ps_net.eval(eval_parameters, device, phase="eval")

    # load data for ITE network
    print("############### DCN ###############")
    tensor_treated, tensor_control = dL.prepare_tensor_for_DCN(np_covariates_X_train,
                                                               np_covariates_Y_train,
                                                               ps_score_list)
    # train ITE network

    # test ITE network


if __name__ == '__main__':
    main_propensity_dropout_BL()
