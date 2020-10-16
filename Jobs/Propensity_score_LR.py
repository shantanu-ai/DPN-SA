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

from sklearn.linear_model import LogisticRegression
import numpy as np

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

    @staticmethod
    def train_graph(np_covariates_X_train, np_covariates_Y_train, regularized=False):
        print(np_covariates_X_train.shape)
        # print(np_covariates_Y_train.shape)
        lr_model = None
        if not regularized:
            lr_model = LogisticRegression(solver='liblinear')
        elif regularized:
            lr_model = LogisticRegression(penalty="l1", C=1.25, solver="liblinear")

        # fit the model with data
        lr_model.fit(np_covariates_X_train, np_covariates_Y_train.ravel())
        treatment = np_covariates_Y_train.ravel()
        print(treatment.shape)
        treated_index = np.where(treatment == 1)[0]
        control_index = np.where(treatment == 0)[0]
        treated_X = np_covariates_X_train[treated_index]
        control_X = np_covariates_X_train[control_index]

        print(treated_X.shape)
        print(control_X.shape)
        proba_treated = lr_model.predict_proba(treated_X)[:, -1].tolist()
        proba_control = lr_model.predict_proba(control_X)[:, -1].tolist()
        return proba_treated, proba_control

