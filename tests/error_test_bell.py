from main import *


N = 10000
psi_prod = random_product_state(1, pure=True)
rho = qt.ket2dm(psi_prod)
counts = pauli_shot_counts_from_state(rho.full(), N, N, N)
rho_prime1 = mle_qubit_from_pauli_counts(counts)
rho_prime1 = qt.Qobj(rho_prime1)
rho_prime2 = linear_inversion_qubit_from_pauli_counts(counts)
print(qt.fidelity(rho, rho_prime1))
print(qt.fidelity(rho, rho_prime2))
