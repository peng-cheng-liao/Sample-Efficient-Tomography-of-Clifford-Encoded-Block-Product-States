from main import *
from qutip_qip.operations import cnot, snot

n = 4
S = 1
A1 = 1
A2 = 1
B = 1
steps = 100
delta = 0.02
lam = 0.01
kappa = 0.01


rho = random_clifford_encoded_product_state(
    steps=steps,
    delta=delta,
    n=n,
    S=S,
    A1=A1,
    A2=A2,
    B=B,
    seed=0,
    return_details=False)

error = full_recovery_infidelity(rho,n,
        lam,
        kappa,
        M1=500,
        M2=5000,
        M3=5000,
        return_details=False)

print(error)

