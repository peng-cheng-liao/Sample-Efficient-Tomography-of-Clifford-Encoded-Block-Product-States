from main_2 import *

n = 3
M = int(2000)

rho = random_product_state(n)
B = bell_sampling(n, M, rho)
ranking1 = rank_all_sP_from_B(B)
ranking2 = rank_all_true_sP(rho, n)

print(ranking1)
Q1 = recover_Q_from_ranking(ranking1, disp=True)
print(Q1)


print(ranking2)
Q2 = recover_Q_from_ranking(ranking2, disp=True)
print(Q2)


result1 = check_recovered_Q_product_state(Q1, n)
result2 = check_recovered_Q_product_state(Q2, n)





