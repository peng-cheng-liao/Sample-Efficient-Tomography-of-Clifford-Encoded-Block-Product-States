from main_2 import *

n = 3
M = int(2000)

num_false = 0
for i in range(1):
    rho = random_product_state(n)
    B = bell_sampling(n, M, rho)
    ranking1 = rank_all_sP_from_B(B)
    ranking2 = rank_all_true_sP(rho, n)
    Q1 = recover_Q_from_ranking(ranking1)
    Q2 = recover_Q_from_ranking(ranking2)
    print(i, ranking2[0], ranking2[-1],Q1)
    result1 = check_recovered_Q_product_state(Q1, n)
    result2 = check_recovered_Q_product_state(Q2, n)
    if not result1["ok"]:
        num_false += 1
        print(i)
        print(ranking1)
        print(ranking2)
        print(Q1)
        print(Q2)
    #print(i, result1["ok"], result2["ok"])

print(num_false)
