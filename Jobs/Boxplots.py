import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)

# fake up some data
dcn_pd = np.array([0.19285714285714300,
                   0.42380952380952400,
                   0.06473079249848760,
                   0.06655974338412180,
                   0.20809614168247900,
                   0.030291109362706500,
                   0.06878306878306880,
                   0.10527355322931100,
                   0.059980525803310500,
                   0.12460468058191000])

dpn_sa = np.array([
    0.20952380952381000,
    0.29047619047619000,
    0.005444646098003660,
    0.014835605453087300,
    0.2277039848197340,
    0.06254917387883560,
    0.005291005291005350,
    0.0069128974916058400,
    0.01382667964946440,
    0.14421252371916500
])

lr_lasso = np.array([0.24285714285714300,
                     0.34047619047619000,
                     0.005444646098003660,
                     0.20449077786688000,
                     0.2473118279569890,
                     0.014162077104642000,
                     0.1481481481481480,
                     0.03969978273750750,
                     0.001557935735151010,
                     0.2814674256799490])
data = [dcn_pd, dpn_sa, lr_lasso]

fig1, ax1 = plt.subplots()
ax1.set_title('Basic Plot')
ax1.set_xticklabels(['DCN-PD', 'DPN-SA End to End', 'LR Lasso'])
ax1.boxplot(data)
# plt.show()

plt.draw()
plt.savefig("./Plots/boxplot_jobs_biass_att.jpg", dpi=220)
plt.clf()
