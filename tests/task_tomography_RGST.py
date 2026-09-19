from main_3 import *
from datetime import datetime
import os

"""M1 = 10000
M2 = 10000
M3 = 10000"""


def tomography(n, M1, M2, M3, num_seed=20):
    os.makedirs(f"Data/04/n={n}", exist_ok=True)
    error_list = []
    for seed in range(num_seed):
        time0 = datetime.now()
        rho = random_clifford_encoded_product_pure_state(
            steps=steps,
            delta=delta,
            n=n,
            S=S,
            A1=A1,
            seed=seed,
            return_details=False)
        error = full_recovery_infidelity(
            rho,
            n,
            lam,
            kappa,
            M1=M1,
            M2=M2,
            M3=M3,
            seed1=seed,
            seed2=seed,
            return_details=False
        )
        time1 = datetime.now()
        error_list.append(error)
        print(n, M1, M2, M3, seed, time1 - time0, error)
        np.save(f"Data/04/n={n}/Error_n={n}_M1={M1}_M2={M2}_M3={M3}.npy", error_list)


delta = 0.02
lam = 0.01
kappa = 0.01

M2_list = [i * 10 ** j for j in [0, 1, 2, 3, 4, 5] for i in [1, 3, 5, 7, 9]]
M3_list = [i * 10 ** j for j in [0, 1, 2, 3, 4, 5] for i in [1, 3, 5, 7, 9]]

for n in [6]:
    S = int(n / 4)
    A1 = n - S
    steps = 20 * n
    for M2 in M2_list:
        for M3 in M3_list:
            M1 = int((M2 + 1) / 2)
            tomography(n, M1, M2, M3)
