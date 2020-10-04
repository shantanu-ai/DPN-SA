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

import os

import numpy as np
import pandas as pd

from Utils import Utils


class DataLoader:
    def preprocess_for_graphs(self, csv_path):
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), csv_path), header=None)
        return self.__convert_to_numpy(df)

    def prep_process_all_data(self, csv_path):
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), csv_path), header=None)
        np_covariates_X, np_treatment_Y = self.__convert_to_numpy(df)
        return np_covariates_X, np_treatment_Y

    def preprocess_data_from_csv(self, train_path, test_path, iter_id):
        train_arr = np.load(train_path)
        test_arr = np.load(test_path)
        np_train_X = train_arr['x'][:, :, iter_id]
        np_train_T = Utils.convert_to_col_vector(train_arr['t'][:, iter_id])
        np_train_e = Utils.convert_to_col_vector(train_arr['e'][:, iter_id])
        np_train_yf = Utils.convert_to_col_vector(train_arr['yf'][:, iter_id])

        train_X = np.concatenate((np_train_X, np_train_e, np_train_yf), axis=1)

        np_test_X = test_arr['x'][:, :, iter_id]
        np_test_T = Utils.convert_to_col_vector(test_arr['t'][:, iter_id])
        np_test_e = Utils.convert_to_col_vector(test_arr['e'][:, iter_id])
        np_test_yf = Utils.convert_to_col_vector(test_arr['yf'][:, iter_id])

        test_X = np.concatenate((np_test_X, np_test_e, np_test_yf), axis=1)

        print("Numpy Train Statistics:")
        print(train_X.shape)
        print(np_train_T.shape)

        print(" Numpy Test Statistics:")
        print(test_X.shape)
        print(np_test_T.shape)

        # X -> x1.. x17, e, yf -> (19, 1)
        return train_X, test_X, np_train_T, np_test_T

    def preprocess_data_from_csv_augmented(self, csv_path, split_size):
        # print(".. Data Loading synthetic..")
        # data load
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), csv_path), header=None)
        np_covariates_X, np_treatment_Y = self.__convert_to_numpy_augmented(df)
        # print("ps_np_covariates_X: {0}".format(np_covariates_X.shape))
        # print("ps_np_treatment_Y: {0}".format(np_treatment_Y.shape))

        np_covariates_X_train, np_covariates_X_test, np_covariates_Y_train, np_covariates_Y_test = \
            Utils.test_train_split(np_covariates_X, np_treatment_Y, split_size)
        # print("np_covariates_X_train: {0}".format(np_covariates_X_train.shape))
        # print("np_covariates_Y_train: {0}".format(np_covariates_Y_train.shape))
        # print("---" * 20)
        # print("np_covariates_X_test: {0}".format(np_covariates_X_test.shape))
        # print("np_covariates_Y_test: {0}".format(np_covariates_Y_test.shape))
        return np_covariates_X_train, np_covariates_X_test, np_covariates_Y_train, np_covariates_Y_test

    @staticmethod
    def convert_to_tensor(ps_np_covariates_X, ps_np_treatment_Y):
        return Utils.convert_to_tensor(ps_np_covariates_X, ps_np_treatment_Y)

    @staticmethod
    def convert_to_tensor_DCN(np_df_X,
                              np_ps_score,
                              np_df_Y_f,
                              np_df_Y_cf):
        return Utils.convert_to_tensor_DCN(np_df_X,
                                           np_ps_score,
                                           np_df_Y_f,
                                           np_df_Y_cf)

    def prepare_tensor_for_DCN(self, ps_np_covariates_X, ps_np_treatment_Y, ps_list,
                               is_synthetic):
        # print("ps_np_covariates_X: {0}".format(ps_np_covariates_X.shape))
        # print("ps_np_treatment_Y: {0}".format(ps_np_treatment_Y.shape))
        X = Utils.concat_np_arr(ps_np_covariates_X, ps_np_treatment_Y, axis=1)

        # col of X -> x1 .. x25, Y_f, Y_cf, T, Ps
        X = Utils.concat_np_arr(X, np.array([ps_list]).T, axis=1)
        # print("Big X: {0}".format(X.shape))
        df_X = pd.DataFrame(X)
        treated_df_X, treated_ps_score, treated_df_Y_f, treated_df_e = \
            self.__preprocess_data_for_DCN(df_X, treatment_index=1,
                                           is_synthetic=is_synthetic)

        control_df_X, control_ps_score, control_df_Y_f, control_df_e = \
            self.__preprocess_data_for_DCN(df_X, treatment_index=0,
                                           is_synthetic=is_synthetic)

        np_treated_df_X, np_treated_ps_score, np_treated_df_Y_f, np_treated_df_e = \
            self.__convert_to_numpy_DCN(treated_df_X, treated_ps_score, treated_df_Y_f, treated_df_e)

        np_control_df_X, np_control_ps_score, np_control_df_Y_f, np_control_df_e = \
            self.__convert_to_numpy_DCN(control_df_X, control_ps_score, control_df_Y_f, control_df_e)

        # np_treated_df_Y_f = Utils.convert_to_col_vector()

        print(" Treated Statistics ==>")
        print(np_treated_df_X.shape)
        print(" Control Statistics ==>")
        print(np_control_df_X.shape)

        return {
            "treated_data": (np_treated_df_X, np_treated_ps_score,
                             np_treated_df_Y_f, np_treated_df_e),
            "control_data": (np_control_df_X, np_control_ps_score,
                             np_control_df_Y_f, np_control_df_e)
        }

    @staticmethod
    def __convert_to_numpy(df):
        covariates_X = df.iloc[:, 5:]
        treatment_Y = df.iloc[:, 0:1]
        outcomes_Y = df.iloc[:, 1:3]

        np_covariates_X = Utils.convert_df_to_np_arr(covariates_X)
        np_outcomes_Y = Utils.convert_df_to_np_arr(outcomes_Y)
        np_X = Utils.concat_np_arr(np_covariates_X, np_outcomes_Y, axis=1)

        np_treatment_Y = Utils.convert_df_to_np_arr(treatment_Y)

        return np_X, np_treatment_Y

    @staticmethod
    def __convert_to_numpy_augmented(df):
        covariates_X = df.iloc[:, 5:]
        treatment_Y = df.iloc[:, 0:1]
        outcomes_Y = df.iloc[:, 1:3]

        np_covariates_X = Utils.convert_df_to_np_arr(covariates_X)
        np_std = np.std(np_covariates_X, axis=0)
        np_outcomes_Y = Utils.convert_df_to_np_arr(outcomes_Y)

        noise = np.empty([747, 25])
        id = -1
        for std in np_std:
            id += 1
            noise[:, id] = np.random.normal(0, 1.96 * std)

        random_correlated = np_covariates_X + noise

        random_X = np.random.permutation(np.random.random((747, 175)) * 10)
        np_covariates_X = np.concatenate((np_covariates_X, random_X), axis=1)
        np_covariates_X = np.concatenate((np_covariates_X, random_correlated), axis=1)
        np_X = Utils.concat_np_arr(np_covariates_X, np_outcomes_Y, axis=1)

        np_treatment_Y = Utils.convert_df_to_np_arr(treatment_Y)

        return np_X, np_treatment_Y

    @staticmethod
    def __preprocess_data_for_DCN(df_X, treatment_index, is_synthetic):
        df = df_X[df_X.iloc[:, -2] == treatment_index]
        # col of X -> x1 .. x17, Y_f, T, Ps
        if is_synthetic:
            # for synthetic dataset #covariates: 75
            df_X = df.iloc[:, 0:75]
        else:
            # for original dataset #covariates: 17
            df_X = df.iloc[:, 0:17]

        ps_score = df.iloc[:, -1]
        df_Y_f = df.iloc[:, -3]
        df_e = df.iloc[:, -4]

        return df_X, ps_score, df_Y_f, df_e

    @staticmethod
    def __convert_to_numpy_DCN(df_X, ps_score, df_Y_f, df_Y_cf):
        np_df_X = Utils.convert_df_to_np_arr(df_X)
        np_ps_score = Utils.convert_df_to_np_arr(ps_score)
        np_df_Y_f = Utils.convert_df_to_np_arr(df_Y_f)
        np_df_Y_cf = Utils.convert_df_to_np_arr(df_Y_cf)

        # print("np_df_X: {0}".format(np_df_X.shape))
        # print("np_ps_score: {0}".format(np_ps_score.shape))
        # print("np_df_Y_f: {0}".format(np_df_Y_f.shape))
        # print("np_df_Y_cf: {0}".format(np_df_Y_cf.shape))

        return np_df_X, np_ps_score, np_df_Y_f, np_df_Y_cf
