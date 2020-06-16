from sklearn.linear_model import LogisticRegression


class Propensity_socre_LR:
    @staticmethod
    def train(np_covariates_X_train, np_covariates_Y_train, regularized=False):
        # print(np_covariates_X_train.shape)
        # print(np_covariates_Y_train.shape)
        lr_model = None
        if not regularized:
            lr_model = LogisticRegression(solver='liblinear')
        elif regularized:
            lr_model = LogisticRegression(penalty="l1", solver="liblinear")

        # fit the model with data
        lr_model.fit(np_covariates_X_train, np_covariates_Y_train.ravel())
        proba = lr_model.predict_proba(np_covariates_X_train)[:, -1].tolist()
        return proba, lr_model

    @staticmethod
    def test(np_covariates_X_test, np_covariates_Y_test, log_reg):
        # print(np_covariates_X_train.shape)
        # print(np_covariates_Y_train.shape)

        # fit the model with data
        proba = log_reg.predict_proba(np_covariates_X_test)[:, -1].tolist()
        return proba
