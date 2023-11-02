import qutip as qt
import numpy as np
from math import gcd

def qft(n):
    N = 2**n
    omega = np.exp(2*np.pi*1j/N)
    qft_matrix = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            qft_matrix[i,j] = omega**(i*j)
    return qft_matrix / np.sqrt(N)

def shors_algorithm(N, M, num_qubits):
    # Initialize the state |0>|0>
    state = qt.tensor([qt.basis(2,0) for i in range(num_qubits*2)])

    # Apply Hadamard gates to first num_qubits qubits
    for i in range(num_qubits):
        state = qt.tensor(qt.hadamard_transform(num_qubits, [i]), qt.qeye(2**(num_qubits+i))) * state

    # Apply quantum function f(x) = M^x mod N
    for i in range(num_qubits):
        state = qt.tensor(qt.phasegate(2*np.pi*M*2**i/N, num_qubits+i), qt.qeye(2**(num_qubits+i))) * state

    # Apply inverse QFT to first num_qubits qubits
    qft_matrix = qft(num_qubits)
    for i in range(num_qubits):
        state = qt.tensor(qt.qeye(2**i), qft_matrix * qt.qeye(2**(num_qubits-i))) * state

    # Measure the first num_qubits qubits
    qubits_to_measure = list(range(num_qubits))
    results = qt.measure_all(state, qubits_to_measure, num_shots=1)
    measured_state = qt.tensor([qt.basis(2,int(results[i]), "qubit") for i in range(num_qubits)])

    # Apply continued fraction algorithm to determine the period r
    x = int(''.join([str(int(results[i])) for i in range(num_qubits)]), 2)
    r = qt.continued_fractions(x/N, max_denominator=100)[0][1]

    # Check if r is even or gcd(M^(r/2) +/- 1, N) != 1
    if r % 2 == 0:
        return gcd(int(M**(r/2) + 1), N), gcd(int(M**(r/2) - 1), N)
    else:
        return None

N = 15
M = 4
num_qubits = 4

factors = shors_algorithm(N, M, num_qubits)
print(factors)