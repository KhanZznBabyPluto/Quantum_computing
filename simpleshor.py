import numpy as np
import qiskit as q
from qiskit.visualization import plot_histogram
from qutip import *
from qutip.qip.operations import *
from qutip.qip.circuit import *

def qft(n):
    """Creates an n-qubit QFT circuit"""
    circ = QubitCircuit(n)
    for i in range(n):
        for j in range(i):
            circ.add_gate(swapalpha(np.pi/2**(i-j)), targets=[j], controls=[i])
        circ.add_gate(hadamard_transform(), targets=[i])
    return circ

def shor(N, a):
    """Shor's algorithm"""
    # Step 1: Find period r using quantum period finding
    qft_circ = qft(4)
    qc = q.QuantumCircuit(8, 4)
    qc.x(7)
    for i in range(4):
        qc.h(i)
    for i in range(4):
        qc.append(qft_circ.to_instruction(), [i])
    qc.barrier()
    for i in range(3):
        qc.measure(i, i)
    backend = q.Aer.get_backend('qasm_simulator')
    result = q.execute(qc, backend=backend, shots=1).result()
    measured = int(list(result.get_counts())[0], 2)
    r = measured + 1
    while r % 2 == 0:
        r //= 2
    print("r =", r)

    # Step 2: Factor N using r
    if r % 2 != 0:
        x = (a**(r//2) + 1) % N
        if x == 0:
            return N
        p = np.gcd(x - 1, N)
        if p > 1 and p < N:
            return p
    return N

# Test Shor's algorithm on N = 15, a = 2
result = shor(15, 2)
print("Result =", result)