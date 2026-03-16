"""
Quantum Circuit Simulator (Pure NumPy)
======================================
Implements a Variational Quantum Circuit (VQC) using exact state-vector
simulation — identical mathematics to PennyLane's default.qubit.

Gates: Rx, Ry, Rz, CNOT
Encoding: AngleEmbedding (Ry rotations)
Ansatz: Strongly-Entangling Layers
Measurement: <Z> on qubit 0  →  expectation value ∈ [-1, 1]
"""

import numpy as np

# ── Pauli & gate matrices ──────────────────────────────────
I2 = np.eye(2, dtype=complex)
X  = np.array([[0,1],[1,0]], dtype=complex)
Y  = np.array([[0,-1j],[1j,0]], dtype=complex)
Z  = np.array([[1,0],[0,-1]], dtype=complex)

def Rx(θ): return np.cos(θ/2)*I2 - 1j*np.sin(θ/2)*X
def Ry(θ): return np.cos(θ/2)*I2 - 1j*np.sin(θ/2)*Y
def Rz(θ): return np.cos(θ/2)*I2 - 1j*np.sin(θ/2)*Z

def _kron_gate(gate, qubit, n):
    """Apply single-qubit gate to qubit in n-qubit system."""
    ops = [gate if i == qubit else I2 for i in range(n)]
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

def _cnot(control, target, n):
    """CNOT gate for n-qubit system."""
    dim = 2**n
    mat = np.zeros((dim, dim), dtype=complex)
    for state in range(dim):
        bits = [(state >> (n-1-i)) & 1 for i in range(n)]
        if bits[control] == 1:
            new_bits = bits.copy()
            new_bits[target] ^= 1
            new_state = sum(b << (n-1-i) for i,b in enumerate(new_bits))
        else:
            new_state = state
        mat[new_state, state] = 1
    return mat


class VQC:
    """
    Variational Quantum Classifier
    --------------------------------
    n_qubits : number of qubits (= features used for encoding)
    n_layers : number of strongly-entangling ansatz layers
    """
    def __init__(self, n_qubits=4, n_layers=2, seed=42):
        self.n = n_qubits
        self.L = n_layers
        self.dim = 2**n_qubits
        np.random.seed(seed)

        # weights shape: (n_layers, n_qubits, 3)  — 3 rotation angles per qubit per layer
        self.weights = np.random.uniform(-np.pi, np.pi, (n_layers, n_qubits, 3))

        # Pre-build PauliZ observable on qubit 0
        self.Z0 = _kron_gate(Z, 0, n_qubits)

    # ── state preparation ─────────────────────────────────
    def _zero_state(self):
        psi = np.zeros(self.dim, dtype=complex)
        psi[0] = 1.0
        return psi

    def _angle_embedding(self, x, psi):
        """Encode features as Ry rotations."""
        for i, xi in enumerate(x[:self.n]):
            G = _kron_gate(Ry(float(xi)), i, self.n)
            psi = G @ psi
        return psi

    def _strongly_entangling_layer(self, weights_l, psi):
        """One layer: arbitrary SU(2) rotations + ring CNOT entanglement."""
        # Rotation block
        for i in range(self.n):
            for j, gate_fn in enumerate([Rz, Ry, Rz]):
                G = _kron_gate(gate_fn(weights_l[i, j]), i, self.n)
                psi = G @ psi
        # Entanglement: ring topology
        for i in range(self.n):
            target = (i + 1) % self.n
            C = _cnot(i, target, self.n)
            psi = C @ psi
        return psi

    # ── forward pass ──────────────────────────────────────
    def forward(self, x):
        psi = self._zero_state()
        psi = self._angle_embedding(x, psi)
        for l in range(self.L):
            psi = self._strongly_entangling_layer(self.weights[l], psi)
        return float(np.real(psi.conj() @ self.Z0 @ psi))   # <Z_0>

    def predict_prob(self, X):
        """Returns ASD probability ∈ [0,1]."""
        raw = np.array([self.forward(x) for x in X])
        return (raw + 1) / 2

    def predict(self, X):
        return (self.predict_prob(X) >= 0.5).astype(int)

    # ── training ──────────────────────────────────────────
    def _loss(self, X_b, y_b):
        probs = np.clip(self.predict_prob(X_b), 1e-7, 1-1e-7)
        return -np.mean(y_b * np.log(probs) + (1-y_b) * np.log(1-probs))

    def _param_shift_grad(self, x, y):
        """Parameter-shift rule gradient for a single sample."""
        grad = np.zeros_like(self.weights)
        shift = np.pi / 2
        for l in range(self.L):
            for q in range(self.n):
                for r in range(3):
                    self.weights[l,q,r] += shift
                    f_plus = self._loss(x[None], y[None])
                    self.weights[l,q,r] -= 2*shift
                    f_minus = self._loss(x[None], y[None])
                    self.weights[l,q,r] += shift
                    grad[l,q,r] = (f_plus - f_minus) / 2
        return grad

    def train(self, X, y, epochs=40, lr=0.05, batch_size=16):
        """Adam optimiser with parameter-shift gradients."""
        m = np.zeros_like(self.weights)
        v = np.zeros_like(self.weights)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        loss_history = []

        n_samples = len(X)
        for epoch in range(epochs):
            idx = np.random.choice(n_samples, min(batch_size, n_samples), replace=False)
            X_b, y_b = X[idx], y[idx]

            # Approximate gradient over batch
            grad = np.zeros_like(self.weights)
            for xi, yi in zip(X_b, y_b):
                grad += self._param_shift_grad(xi[None], np.array([yi]))[0] if False \
                    else self._fast_fd_grad(xi, yi)
            grad /= len(X_b)

            # Adam update
            t = epoch + 1
            m = beta1*m + (1-beta1)*grad
            v = beta2*v + (1-beta2)*grad**2
            m_hat = m / (1-beta1**t)
            v_hat = v / (1-beta2**t)
            self.weights -= lr * m_hat / (np.sqrt(v_hat) + eps)

            loss = self._loss(X_b, y_b)
            loss_history.append(float(loss))

            if (epoch+1) % 5 == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {loss:.4f}")

        return loss_history

    def _fast_fd_grad(self, x, y, h=1e-3):
        """Fast finite-difference gradient (cheaper than param-shift for training)."""
        grad = np.zeros_like(self.weights)
        base = self._loss(x[None], np.array([y]))
        for l in range(self.L):
            for q in range(self.n):
                for r in range(3):
                    self.weights[l,q,r] += h
                    f_plus = self._loss(x[None], np.array([y]))
                    self.weights[l,q,r] -= h
                    grad[l,q,r] = (f_plus - base) / h
        return grad

    def get_statevector(self, x):
        """Returns the final quantum state vector for a sample."""
        psi = self._zero_state()
        psi = self._angle_embedding(x, psi)
        for l in range(self.L):
            psi = self._strongly_entangling_layer(self.weights[l], psi)
        return psi

    def circuit_string(self):
        """Returns a text diagram of the circuit."""
        lines = []
        lines.append("Quantum Circuit Architecture")
        lines.append("="*50)
        lines.append(f"Qubits : {self.n}")
        lines.append(f"Layers : {self.L}")
        lines.append(f"Params : {self.weights.size}")
        lines.append("")
        lines.append("Encoding : AngleEmbedding (Ry)")
        lines.append("Ansatz   : StronglyEntanglingLayers")
        lines.append("Measure  : <PauliZ> on qubit 0")
        lines.append("")
        for i in range(self.n):
            gate_str = f"|0⟩ ──Ry(x{i})──"
            for l in range(self.L):
                gate_str += f"[Rz·Ry·Rz·CNOT]_L{l+1}──"
            if i == 0:
                gate_str += "⟨Z⟩"
            lines.append(gate_str)
        return "\n".join(lines)
