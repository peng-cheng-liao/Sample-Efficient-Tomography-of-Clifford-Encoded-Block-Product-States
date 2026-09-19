import matplotlib.pyplot as plt
import numpy as np

from main import *

n_list = [2, 3, 4, 5, 6, 7, 9, 10]  #
M_list = [i * 10 ** j for j in [0, 1, 2, 3, 4, 5] for i in [1, 4, 7]]
N_list = [i * 10 ** j for j in [0, 1, 2, 3, 4, 5] for i in [1, 4, 7]]
seed_list = np.arange(20)

error_matrix = np.empty((len(n_list), len(M_list), len(N_list), len(seed_list)))

for i, n in enumerate(n_list):
    x_list = []
    y_list = []
    for j, M in enumerate(M_list):
        for k, N in enumerate(N_list):
            errors = np.load(f"Data/01/n={n}/M={M}_N={N}_errors.npy")
            errors[errors>1] = np.nan
            errors[errors <0] = np.nan
            error = np.nanmean(errors[1, :])
            if error>1:
                print(errors)
            error_matrix[i, j, k] = error
            x = M*2+N*3
            y = error
            x_list.append(x)
            y_list.append(y)
    plt.scatter(x_list,y_list,label=f"n={n}")

plt.yscale("log")
plt.xscale("log")
plt.xlabel("number of samples")
plt.ylabel("error")
plt.legend()
plt.show()

