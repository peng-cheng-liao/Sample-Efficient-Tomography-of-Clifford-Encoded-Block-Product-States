import numpy as np

from main import *
from datetime import datetime


def delta_verification(n, M, delta, seed=0):
    time0 = datetime.now()
    rho_prod = random_product_state_with_bloch_constraints(n, delta, seed=seed)
    Uc = random_clifford_gate(n, steps=20 * n)[2]
    Uc = qt.Qobj(Uc, dims=[[2] * n, [2] * n])
    rho = Uc * rho_prod * Uc.dag()
    B = bell_sampling(n, M, rho)
    ranking1 = rank_all_sP_from_B(B)
    Q1 = recover_Q_from_ranking(ranking1)
    # print(Q1)
    # print(Q2)
    # result1 = check_recovered_Q(Q1, n)
    # result2 = check_recovered_Q(Q2, n)
    # print(result1)
    # print(result2)

    F1 = symplectic_matrix_from_recover_output(Q1)
    # print(is_symplectic(F1), is_symplectic(F2))
    Gates1, U_hat_1 = synthesize_and_unitary(F1)
    # print(Gates1)
    # print(Gates2)

    U_hat_1 = qt.Qobj(U_hat_1, dims=[[2] * n, [2] * n])
    rho_prod_prime1 = U_hat_1.dag() * rho * U_hat_1

    rho_prod_prime1_list = [qt.ptrace(rho_prod_prime1, [j]) for j in range(n)]

    rho_recoverd_1 = U_hat_1 * qt.tensor(rho_prod_prime1_list) * U_hat_1.dag()
    print(datetime.now() - time0)
    return qt.fidelity(rho_recoverd_1, rho)


import numpy as np

n = 4
M_list = [i * 10 ** j for j in [0, 1, 2, 3, 4, 5] for i in [1, 4, 7]]
delta_list = [i * 10 ** j for j in [-1, -2, -3, -4, -5] for i in [1, 4, 7]]
seed_list = np.arange(20)

for M in M_list:
    for delta in delta_list:
        error_list = []
        for seed in seed_list:
            print(M, delta, seed)
            error = delta_verification(n, M, delta, seed=seed)
            error_list.append(error)

        M_str = f"{M:.0e}".replace("e+0", "e").replace("e+", "e")
        delta_str = f"{delta:.0e}".replace("e-0", "e-").replace("e+0", "e").replace("e+", "e")

        np.save(
            f"Data/02/delta_verification_n={n}_M={M_str}_delta={delta_str}.npy",
            np.array(error_list))
