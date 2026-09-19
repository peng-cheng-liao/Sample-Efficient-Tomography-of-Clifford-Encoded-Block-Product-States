from main import *
from datetime import datetime
n = 5
M = int(200)


time0 = datetime.now()
num_false = 0
for i in range(1):
    rho_prod = random_product_state_with_bloch_constraints(n, 0.01,)
    #Uc = random_clifford_gate(n, steps=10*n)[2]
    #Uc = qt.Qobj(Uc, dims=[[2]*n, [2]*n])
    #rho = Uc * rho_prod * Uc.dag()
    rho = rho_prod
    B = bell_sampling(n, M, rho)
    ranking1 = rank_all_sP_from_B(B)
    ranking2 = rank_all_true_sP(rho, n)
    Q1 = recover_Q_from_ranking(ranking1)
    Q2 = recover_Q_from_ranking(ranking2)
    print(Q1)
    print(Q2)
    #result1 = check_recovered_Q(Q1, n)
    #result2 = check_recovered_Q(Q2, n)
    #print(result1)
    #print(result2)

    F1 = symplectic_matrix_from_recover_output(Q1)
    F2 = symplectic_matrix_from_recover_output(Q2)
    print(is_symplectic(F1), is_symplectic(F2))
    Gates1, U_hat_1 = synthesize_and_unitary(F1)
    Gates2, U_hat_2 = synthesize_and_unitary(F2)
    print(Gates1)
    print(Gates2)

    U_hat_1 = qt.Qobj(U_hat_1, dims=[[2]*n, [2]*n])
    U_hat_2 = qt.Qobj(U_hat_2, dims=[[2]*n, [2]*n])
    rho_prod_prime1 = U_hat_1.dag() * rho * U_hat_1
    rho_prod_prime2 = U_hat_2.dag() * rho * U_hat_2

    rho_prod_prime1_list = [qt.ptrace(rho_prod_prime1, [j]) for j in range(n)]
    rho_prod_prime2_list = [qt.ptrace(rho_prod_prime2, [j]) for j in range(n)]

    rho_recoverd_1 = U_hat_1 * qt.tensor(rho_prod_prime1_list) * U_hat_1.dag()
    rho_recoverd_2 = U_hat_2 * qt.tensor(rho_prod_prime2_list) * U_hat_2.dag()
    print(qt.fidelity(rho_recoverd_1, rho))
    print(qt.fidelity(rho_recoverd_2, rho))
    print(datetime.now()-time0)

