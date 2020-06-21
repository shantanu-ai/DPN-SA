# from math import sqrt
#
# import numpy as np
from numpy import genfromtxt
#
# sae_np = genfromtxt('./MSE/25-1-25/Stat/MSE_dict_SAE.csv')
# lr_lasso_np = genfromtxt('./MSE/25-1-25/Stat/MSE_dict_LR_lasso.csv')
# # lr_lasso_np = genfromtxt('./MSE/[!!!best]epoch_200_lr_0001_bs_32_sparse_prob_0.08 _wt_decay_0003_beta_0_4/STAT/MSE_dict_LR_lasso.csv')
#
#
# # Compute the difference between the results
# diff = [y - x for y, x in zip(sae_np, lr_lasso_np)]
# # Compute the mean of differences
# d_bar = np.mean(diff)
# # compute the variance of differences
# sigma2 = np.var(diff)
# # compute the number of data points used for training
# n1 = 597
# # compute the number of data points used for testing
# n2 = 150
# # compute the total number of data points
# n = 747
# # compute the modified variance
# sigma2_mod = sigma2 * (1 / n + n2 / n1)
# # compute the t_static
# t_static = d_bar / np.sqrt(sigma2_mod)
#
# m = 0.2  # test set fraction
# n = 0.8  # training test fraction
# r = 100  # number of runs
# k = 2  # folds
#
# from scipy.stats import t
#
# tcorr = sqrt(m / n + 1 / (r * k))  # correction for t-value
# tc = t_static / (tcorr * sqrt(r * k))
# # Compute p-value and plot the results
# Pvalue = (1 - t.cdf(tc, (r * k - 1)))
# print(Pvalue)


# import seaborn as sns
from scipy.stats import ttest_ind


def compare_2_groups(arr_1, arr_2, alpha, sample_size):
    stat, p = ttest_ind(arr_1, arr_2)
    print('Statistics=%.8f, p=%.53f' % (stat, p))
    if p > alpha:
        print('Same distributions (fail to reject H0)')
    else:
        print('Different distributions (reject H0)')


sample_size = 15

lr_lasso_np = genfromtxt('./MSE/25-1-25/Stat/MSE_dict_LR_lasso.csv')

sae_np = [2.11293416954517,
               1.9156299745241283,
               1.8842843368988917,
               1.893848187427988,
               1.8814272118341138,
               1.9498656435072526,
               1.8385509063701446,
               1.784295601581781,
               1.8906715143911443,
               1.7984839576240301,
               1.9855445212754206,
               1.922287783536811,
               1.9529705071354768,
               1.934987494015969,
               1.904106636810697,
               1.7819165650167441,
               1.8205612752938767,
               1.901332986060631,
               1.7422743499441704,
               1.8164239063044825,
               2.087779985200049,
               1.7829459509706003,
               1.8972510660216988,
               1.8553627635685037,
               1.9102683810956758,
               1.977649856862145,
               1.902091059245989,
               1.8874142524775481,
               1.8819616488161475,
               1.807336384846191,
               1.9967226229827952,
               1.8770784585332292,
               1.8320086372302145,
               2.179897433098271,
               2.0403194334340613,
               1.7276678140275559,
               1.9124045428857535,
               1.7285418353255972,
               1.7709564883884328,
               1.9670928399347005,
               1.9248270315749092,
               2.167043522634095,
               1.7927596637362684,
               1.9945597483610686,
               1.992416843386299,
               1.8653060397353136,
               2.001015623387842,
               2.152627871057743,
               1.9331372884889633,
               2.092476258198252,
               1.81910418750855,
               1.874189190176827,
               1.94650201948475,
               2.021550623239348,
               2.01522650407869,
               1.9609355638088555,
               1.9202879261674521,
               2.1491613968878607,
               1.7872945445906108,
               1.8650543354595295,
               1.8313055993317517,
               1.9021371542730916,
               1.9899250530441142,
               2.0362934396403407,
               2.0118420204104086,
               2.016954418875641,
               1.907001817047127,
               1.8596417933081497,
               1.988396976035294,
               2.144904123458625,
               1.9230424078375234,
               1.8411370075639328,
               1.7842589850505333,
               1.8191776671571744,
               1.8270969732546245,
               1.9611715895588246,
               1.8428176209391354,
               2.041641130762317,
               1.906901264487265,
               1.9462743179542052,
               1.9188950089089902,
               1.8922349071805986,
               1.8765617115663955,
               1.8715240756734821,
               1.7452168996989281,
               1.9250158678100888,
               2.0361777792961013,
               1.8395116504916968,
               1.9214140684011143,
               1.8376124232282545,
               1.8549079286326848,
               1.8778482467504565,
               1.7891770453374298,
               2.0161996043679498,
               1.930085625755654,
               1.9677190537178202,
               1.7949951977635366,
               1.9237657238958967,
               1.8900560887042888,
               2.063163972776114
               ]

# sae_np = genfromtxt('./MSE/25-1-25/Stat/MSE_dict_SAE.csv')
compare_2_groups(sae_np, lr_lasso_np, 0.05, sample_size)
