from main_4 import *
from datetime import datetime
import os


delta = 0.02
lam = 0.01
kappa = 0.01



def tomography(n, M, num_seed=1):
    #os.makedirs(f"Data/05/n={n}", exist_ok=True)
    error_list = []
    for seed in range(num_seed):
        time0 = datetime.now()
        psi = random_clifford_encoded_product_pure_state(
            steps=steps,
            delta=delta,
            n=n,
            S=S,
            A1=A1,
            seed=seed,
            return_details=False)
        error = generic_pure_state_local_pauli_tomography(
        n,
        psi,
        M,
        seed,
        return_details= False)
        time1 = datetime.now()
        error_list.append(error[1:])
        print(n, M, seed, time1 - time0, error[1:])
        #np.save(f"Data/05/n={n}/Error_n={n}_M={M}.npy", error_list)




tomography()

"""
M_list = [i * 10 ** j for j in [0, 1, 2, 3, 4, 5] for i in [1, 3, 5, 7, 9]]

for n in [4]:
    S = int(n / 4)
    A1 = n - S
    steps = 20 * n
    for M in M_list:
            tomography(n, M)
"""
