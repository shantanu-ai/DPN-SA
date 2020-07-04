from Propensity_score_LR import Propensity_socre_LR
from Utils import Utils
from dataloader import DataLoader

device = Utils.get_device()
dL = DataLoader()
csv_path = "Dataset/ihdp_sample.csv"
split_size = 0.8
np_covariates_X, np_treatment_Y = \
    dL.prep_process_all_data(csv_path)

ps_train_set = dL.convert_to_tensor(np_covariates_X, np_treatment_Y)

phase = "train"

ps_score_list_LR, LR_model = Propensity_socre_LR.train(np_covariates_X,
                                                       np_treatment_Y,
                                                       regularized=True)

for ps in ps_score_list_LR:
    print(ps)
