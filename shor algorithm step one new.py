import numpy as np
from qutip import *
import qdi


def is_quantum_object(obj):
    return isinstance(obj, (Qobj, QobjEvo))


def tensor(*args):
    if not all(is_quantum_object(arg) for arg in args):
        raise TypeError("One of inputs is not a quantum object")
    return qutip.tensor(*args)


def qubit_states(n):
    return [tensor(basis(2, 0), basis(2, 0)) for i in range(n)]


def hadamard_transform(n):
    hadamard = qutip.qeye(2)
    for i in range(n):
        hadamard = tensor(qutip.hadamard_transform(), hadamard)
    return hadamard


def controlled_gate(U, control, targets):
    I = qeye(2)
    # Define the controlled-U operator
    Cu = tensor([I]*control + [U] + [I]*(len(targets)-control-1))
    # Define the target operator
    T = tensor([I]*control + [sigmax()]*len(targets) + [I]*(len(U.dims[0])-len(targets)-control-1))
    # Combine the two operators
    return T * Cu * T.dag()


def gate(U, targets):
    """
    Returns the tensor product of U applied to the target qubits.
    """
    I = qeye(2)
    T = tensor([I]*targets[0] + [U] + [I]*(len(U.dims[0])-targets[0]-1))
    for i in range(1, len(targets)):
        T = tensor([I]*targets[i] + [U] + [I]*(len(U.dims[0])-targets[i]-1)) * T
    return T


# Set parameters
N = 15  # composite number to be factored
a = 7  # random number between 1 and N-1

# Create quantum registers
n_qubits = int(np.ceil(np.log2(N)))  # number of qubits needed for N
qr = qubit_states(n_qubits)  # quantum register for input state
cr = qubit_states(1)  # classical register for measurement outcome

## Create quantum circuit for QPE
circuit = tensor(qr + [basis(2, 0)])  # initialize circuit with |0> control qubit
for i in range(n_qubits):
    circuit = tensor(circuit, hadamard_transform(1))

# Apply controlled-not gate to the first n_qubits
for i in range(1, n_qubits + 1):
    circuit = controlled_gate(qutip.cnot, control=0, targets=[i])(circuit)

for i in range(1, n_qubits + 1):
    for j in range(2**i):
        phase = np.exp(2j*np.pi*a*j/2**i)
        circuit = gate(qdi.phase_shift_gate(phase), targets=[i])(circuit)

# Invert the circuit
circuit = circuit.dag()

# Apply controlled-not gates
for i in range(n_qubits):
    for j in range(2**i):
        control = i+1
        target = (control+j)%(n_qubits+1)
        if target != control:
            circuit = controlled_gate(qdi.cnot, control=control, targets=[target])(circuit)

# Apply Hadamard gates
for i in range(n_qubits):
    circuit = gate(qdi.hadamard_transform(), targets=[i+1])(circuit)

# Simulate circuit
psi = circuit * basis(2**n_qubits, 0) # apply circuit to initial state |0>^(n+1)
rho = psi * psi.dag() # compute density matrix of final state
eigenvalues, eigenvectors = rho.eigenstates() # compute eigensystem of density matrix
phase_estimate = eigenvalues.max() # extract phase estimate from largest eigenvalue

# Compute period
r = int(round(2**n_qubits * phase_estimate))

print(f"Period of {a} modulo {N} is {r}")