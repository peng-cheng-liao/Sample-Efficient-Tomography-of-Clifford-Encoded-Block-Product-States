import numpy as np

from main import *
import matplotlib.pyplot as plt


M_list = np.linspace(1, 7, 50)
M_list = np.power(10, M_list)

n = 3
rho = qt.tensor([qt.qeye(2)] * n) / (2 ** n)
P = "XII"
error_list = []
for M in M_list:
    B = bell_sampling(n=n, M=int(M), rho=rho)
    _, _, error = tr2_true_est_error(B, rho, P)
    error_list.append(error)
    print(M, error)

print(error_list)

plt.plot(M_list, error_list, marker="o")
plt.xscale("log")
plt.yscale("log")
plt.show()