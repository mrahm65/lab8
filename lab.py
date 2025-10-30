"""
Phase Estimation & Quantum Fourier Transform Lab
CSC 4700/7700 - Quantum Computing

Instructions:
- Implement all methods in the QuantumLab class below
- Do NOT modify method signatures or class name
- You may add helper methods if needed
- Submit this file to Gradescope
- Total Points: 10

Required installations:
pip install qiskit qiskit-aer numpy

qiskit-aer doesn't work on Python 3.14.

You may need to use the format include full version number.
For example to use a virtual environment:
python3.13 -m venv .venv
source .venv/bin/activate
pip install qiskit qiskit-aer numpy

Always check your Python version with:
python3 --version
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZGate, SGate, TGate, IGate, QFTGate
from qiskit.quantum_info import Statevector, Operator
from qiskit_aer import AerSimulator


class QuantumLab:
    """
    Lab implementation for Phase Estimation and QFT.
    Implement all methods marked with 'pass'.
    """

    def __init__(self):
        """Initialize the lab class."""
        self.simulator = AerSimulator()

    # =========================================================================
    # Problem 1: Spectral Decomposition Verifier (2 points)
    # =========================================================================

    def verify_spectral_theorem(self, U):
        """
        Verify the Spectral Theorem for a 2x2 unitary matrix.

        The spectral theorem states that any unitary matrix U can be written as:
        U = Σ e^(2πiθ_k) |ψ_k⟩⟨ψ_k|

        Args:
            U: 2x2 numpy array (unitary matrix)

        Returns:
            tuple: (eigenvalues, eigenvectors, reconstructed_U)
            - eigenvalues: numpy array of 2 complex eigenvalues
            - eigenvectors: 2x2 numpy array where columns are normalized eigenvectors
            - reconstructed_U: 2x2 numpy array reconstructed using spectral decomposition

        Hints:
            - Use np.linalg.eig() to find eigenvalues and eigenvectors
            - Reconstruct U using: U = Σ λ_k |ψ_k⟩⟨ψ_k|
            - Use np.outer() for outer products
        """
        # Get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(U)
        
        # Reconstruct U using spectral decomposition: U = Σ λ_k |ψ_k⟩⟨ψ_k|
        reconstructed_U = np.zeros((2, 2), dtype=complex)
        for k in range(2):
            # Extract k-th eigenvector (column from eigenvectors matrix)
            psi_k = eigenvectors[:, k]
            # Add contribution: λ_k * |ψ_k⟩⟨ψ_k|
            reconstructed_U += eigenvalues[k] * np.outer(psi_k, psi_k.conj())
        
        return eigenvalues, eigenvectors, reconstructed_U

    # =========================================================================
    # Problem 2: Phase Kickback Circuit (2 points)
    # =========================================================================

    def phase_kickback_circuit(self, target_state):
        """
        Build a phase kickback circuit demonstrating quantum interference.

        Circuit structure:
        1. Apply Hadamard to control qubit (qubit 0)
        2. Initialize target qubit (qubit 1) to specified state
        3. Apply controlled-Z gate

        Args:
            target_state: string '0' or '1' for target qubit initialization

        Returns:
            QuantumCircuit with 2 qubits

        Hints:
            - Use qc.h(0) for Hadamard on control
            - Use qc.x(1) to initialize target to |1⟩
            - Use qc.cz(0, 1) for controlled-Z
        """
        # Create a 2-qubit circuit
        qc = QuantumCircuit(2)
        
        # Initialize target qubit to specified state
        if target_state == "1":
            qc.x(1)
        
        # Apply Hadamard to control qubit
        qc.h(0)
        
        # Apply controlled-Z gate
        qc.cz(0, 1)
        
        return qc

    # =========================================================================
    # Problem 3: Two-Qubit QFT Circuit (2.5 points)
    # =========================================================================

    def build_qft_2qubit(self):
        """
        Build a 2-qubit Quantum Fourier Transform circuit.

        The QFT circuit pattern for 2 qubits:
        - Apply H to qubit 0
        - Apply controlled-phase rotation between qubits
        - Apply H to qubit 1
        - Apply SWAP to reverse qubit order (if needed)

        Returns:
            QuantumCircuit implementing 2-qubit QFT

        Hints:
            - Use qc.h(i) for Hadamard gates
            - Use qc.cp(angle, control, target) for controlled-phase
            - For 2 qubits, the controlled-phase angle is π/2
            - Use qc.swap(0, 1) if needed for qubit ordering
        """
        qc = QuantumCircuit(2)
        
        # Apply H to qubit 0
        qc.h(0)
        
        # Apply controlled-phase gate: CP(π/2) with control=1, target=0
        qc.cp(np.pi / 2, 1, 0)
        
        # Apply H to qubit 1
        qc.h(1)
        
        # Apply SWAP to reverse qubit order
        qc.swap(0, 1)
        
        return qc

    # =========================================================================
    # Problem 4: Using QFT from Circuit Library (1 point)
    # =========================================================================

    def apply_qft_library(self, n_qubits, initial_state_bits):
        """
        Use Qiskit's built-in QFTGate from the circuit library.

        Create a circuit that:
        1. Initializes qubits to the specified classical bit string
        2. Applies Qiskit's QFTGate from circuit library
        3. Returns the resulting statevector

        Args:
            n_qubits: int, number of qubits (1 <= n <= 4)
            initial_state_bits: string of '0's and '1's, e.g., "01" or "101"

        Returns:
            numpy array: the statevector after applying QFT

        Hints:
            - Use: from qiskit.circuit.library import QFTGate
            - Create QFT: qft_gate = QFTGate(n_qubits)
            - Apply with: qc.append(qft_gate, qubits)
            - Initialize state: use qc.x(i) for each '1' in initial_state_bits
            - Get statevector: Statevector.from_instruction(qc)
        """
        # Create quantum circuit with n_qubits
        qc = QuantumCircuit(n_qubits)
        
        # Initialize qubits based on initial_state_bits
        for i, bit in enumerate(initial_state_bits):
            if bit == "1":
                qc.x(i)
        
        # Create and apply QFT gate
        qft_gate = QFTGate(n_qubits)
        qc.append(qft_gate, range(n_qubits))
        
        # Get and return statevector
        sv = Statevector.from_instruction(qc)
        return sv.data

    # =========================================================================
    # Problem 5: Inverse QFT Circuit (0.5 points)
    # =========================================================================

    def build_inverse_qft_2qubit(self):
        """
        Build the inverse of the 2-qubit QFT circuit.

        The inverse QFT reverses the operations of QFT:
        - Reverse the order of gates
        - Negate the angles of controlled-phase gates

        Returns:
            QuantumCircuit implementing inverse 2-qubit QFT

        Hints:
            - Can use qc.inverse() on your QFT circuit, OR
            - Manually reverse: SWAP, H, CP(-angle), H
            - Both approaches are valid
        """
        # Get the QFT circuit
        qft_circuit = self.build_qft_2qubit()
        
        # Use the inverse() method to get the inverse circuit
        iqft_circuit = qft_circuit.inverse()
        
        return iqft_circuit

    # =========================================================================
    # Problem 6: Two-Qubit Phase Estimation (2 points)
    # =========================================================================

    def phase_estimation_2qubit(self, phase_gate, eigenstate="0"):
        """
        Build a 2-qubit phase estimation circuit.

        Circuit structure:
        1. Initialize 2 counting qubits with Hadamard gates
        2. Initialize target qubit (qubit 2) to eigenstate
        3. Apply controlled-U operations with appropriate powers:
           - Controlled-U^1 from qubit 1 to qubit 2
           - Controlled-U^2 from qubit 0 to qubit 2
        4. Apply inverse QFT to counting qubits
        5. Measure counting qubits

        Args:
            phase_gate: Qiskit gate (e.g., ZGate(), SGate(), TGate())
            eigenstate: '0' or '1' - eigenstate to initialize target qubit

        Returns:
            QuantumCircuit with 3 qubits and 2 classical bits
            - Qubits 0-1: counting qubits
            - Qubit 2: target qubit (eigenstate)
            - Classical bits 0-1: for measurement results

        Hints:
            - Apply H to qubits 0 and 1
            - Use phase_gate.control() to create controlled version
            - For U^2, apply the controlled gate twice
            - Apply inverse QFT to qubits [0, 1]
            - Measure qubits 0 and 1 to classical bits 0 and 1
        """
        # Create quantum circuit with 3 qubits and 2 classical bits
        qc = QuantumCircuit(3, 2)
        
        # Initialize target qubit to eigenstate if needed
        if eigenstate == "1":
            qc.x(2)
        
        # Apply Hadamard gates to counting qubits (0 and 1)
        qc.h(0)
        qc.h(1)
        
        # Apply controlled-U^1 from qubit 1 to qubit 2
        controlled_gate = phase_gate.control()
        qc.append(controlled_gate, [1, 2])
        
        # Apply controlled-U^2 from qubit 0 to qubit 2
        controlled_gate_squared = (phase_gate.power(2)).control()
        qc.append(controlled_gate_squared, [0, 2])
        
        # Apply inverse QFT to counting qubits
        iqft = self.build_inverse_qft_2qubit()
        qc.compose(iqft, qubits=[0, 1], inplace=True)
        
        # Measure counting qubits
        qc.measure([0, 1], [0, 1])
        
        return qc


# =============================================================================
# BASIC SANITY TESTS BUT DOESN'T INVOLVE ALL TEST CASES. FEEL FREE TO EXTEND.
# =============================================================================


def main():
    """
    Simple test to verify your implementation works.
    This is NOT the full autograder - just a sanity check.
    """
    print("=" * 70)
    print("Quantum Lab - Sanity Check")
    print("=" * 70)

    lab = QuantumLab()

    # Test 1: Spectral Theorem
    print("\n[Test 1] Spectral Theorem with Pauli Z...")
    try:
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        evals, evecs, U_recon = lab.verify_spectral_theorem(Z)
        if np.allclose(U_recon, Z):
            print("✓ Passed: Matrix reconstructed correctly")
        else:
            print("✗ Failed: Reconstruction does not match original")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test 2: Phase Kickback
    print("\n[Test 2] Phase Kickback with target |1⟩...")
    try:
        circuit = lab.phase_kickback_circuit("1")
        if circuit.num_qubits == 2:
            print("✓ Passed: Circuit has correct number of qubits")
        else:
            print("✗ Failed: Wrong number of qubits")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test 3: QFT
    print("\n[Test 3] 2-Qubit QFT...")
    try:
        circuit = lab.build_qft_2qubit()
        sv = Statevector.from_instruction(circuit)
        expected = np.array([0.5, 0.5, 0.5, 0.5])
        if np.allclose(np.abs(sv.data), expected):
            print("✓ Passed: QFT produces equal superposition from |00⟩")
        else:
            print("✗ Failed: QFT output incorrect")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test 4: Using QFT Library
    print("\n[Test 4] QFT from Circuit Library...")
    try:
        sv = lab.apply_qft_library(2, "00")
        expected = np.array([0.5, 0.5, 0.5, 0.5])
        if np.allclose(np.abs(sv), expected):
            print("✓ Passed: QFT library gate applied correctly")
        else:
            print("✗ Failed: QFT library output incorrect")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test 5: Inverse QFT
    print("\n[Test 5] Inverse QFT...")
    try:
        qft = lab.build_qft_2qubit()
        iqft = lab.build_inverse_qft_2qubit()
        combined = QuantumCircuit(2)
        combined.compose(qft, inplace=True)
        combined.compose(iqft, inplace=True)
        op = Operator(combined)
        if np.allclose(op.data, np.eye(4), atol=1e-10):
            print("✓ Passed: QFT followed by inverse gives identity")
        else:
            print("✗ Failed: Not a proper inverse")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test 6: Phase Estimation
    print("\n[Test 6] Phase Estimation with Z gate...")
    try:
        circuit = lab.phase_estimation_2qubit(ZGate(), eigenstate="1")
        result = lab.simulator.run(circuit, shots=1000).result()
        counts = result.get_counts()
        if counts.get("01", 0) > 800:
            print("✓ Passed: Correctly estimates phase 0.5 → measures '01'")
        else:
            print(f"✗ Failed: Expected '01', got distribution: {counts}")
    except Exception as e:
        print(f"✗ Error: {e}")

    print("\n" + "=" * 70)
    print("Sanity check complete!")
    print("Note: Full autograder on Gradescope has more comprehensive tests.")
    print("=" * 70)


if __name__ == "__main__":
    main()
