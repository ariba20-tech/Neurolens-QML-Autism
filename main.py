"""
From Black Box to Glass Box: Explainable Quantum Machine Learning
Framework for Autism Spectrum Analysis
=================================================================
Main pipeline: data loading → preprocessing → QML model → explainability
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. DATA MODULE
# ─────────────────────────────────────────────

def load_asd_dataset():
    """
    Loads the UCI ASD Screening dataset.
    If not available locally, generates a realistic synthetic version
    with the same feature structure.
    """
    try:
        from ucimlrepo import fetch_ucirepo
        asd = fetch_ucirepo(id=426)
        X = asd.data.features
        y = asd.data.targets
        print("[✓] Loaded UCI ASD Screening dataset")
        return X, y
    except Exception:
        print("[~] UCI repo unavailable — generating synthetic ASD dataset...")
        return _generate_synthetic_asd()


def _generate_synthetic_asd():
    """
    Generates a synthetic dataset matching the UCI ASD Screening structure.
    10 behavioural Q-CHAT-10 items + demographic features.
    """
    np.random.seed(42)
    n = 800

    feature_names = [
        "A1_Score", "A2_Score", "A3_Score", "A4_Score", "A5_Score",
        "A6_Score", "A7_Score", "A8_Score", "A9_Score", "A10_Score",
        "age", "gender", "jaundice", "autism_family", "result"
    ]

    # Q-CHAT binary scores (0 or 1)
    qtchat = np.random.randint(0, 2, (n, 10))

    # ASD label: high score → higher probability of ASD
    score_sum = qtchat.sum(axis=1)
    prob_asd = 1 / (1 + np.exp(-(score_sum - 5)))
    labels = (np.random.rand(n) < prob_asd).astype(int)

    # Demographics
    age = np.random.randint(18, 65, n)
    gender = np.random.randint(0, 2, n)
    jaundice = np.random.randint(0, 2, n)
    autism_family = np.random.randint(0, 2, n)
    result = score_sum

    X = pd.DataFrame(
        np.column_stack([qtchat, age, gender, jaundice, autism_family, result]),
        columns=feature_names
    )
    y = pd.Series(labels, name="ASD_class")
    return X, y


def preprocess(X, y):
    """Encode, scale, and split the dataset."""
    X = X.copy()

    # Encode any string columns
    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = X.fillna(X.median())

    if hasattr(y, "values"):
        y_arr = y.values.ravel()
    else:
        y_arr = np.array(y).ravel()

    le = LabelEncoder()
    y_enc = le.fit_transform(y_arr)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.25, random_state=42, stratify=y_enc
    )
    return X_train, X_test, y_train, y_test, scaler, list(X.columns)


# ─────────────────────────────────────────────
# 2. QUANTUM MACHINE LEARNING MODEL
# ─────────────────────────────────────────────

def build_qml_model(n_qubits=4, n_layers=2):
    """
    Builds a Variational Quantum Circuit (VQC) classifier using PennyLane.
    Returns the quantum node and parameter count.
    """
    import pennylane as qml

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(inputs, weights):
        # Angle embedding of classical features into quantum states
        qml.AngleEmbedding(inputs[:n_qubits], wires=range(n_qubits))

        # Strongly entangling layers
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

        # Measure expectation value of PauliZ on qubit 0
        return qml.expval(qml.PauliZ(0))

    weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    return circuit, weight_shape, dev


def train_qml(X_train, y_train, n_qubits=4, n_layers=2, epochs=30, lr=0.05):
    """
    Trains the VQC using gradient descent via PennyLane + NumPy optimiser.
    Returns trained weights and loss history.
    """
    import pennylane as qml
    from pennylane import numpy as pnp

    circuit, weight_shape, _ = build_qml_model(n_qubits, n_layers)

    # Initialise weights
    weights = pnp.random.uniform(-np.pi, np.pi, weight_shape, requires_grad=True)

    opt = qml.AdamOptimizer(stepsize=lr)
    loss_history = []

    def cost(w, X_batch, y_batch):
        preds = pnp.array([circuit(x[:n_qubits], w) for x in X_batch])
        # Map [-1,1] → [0,1] probabilities
        probs = (preds + 1) / 2
        probs = pnp.clip(probs, 1e-7, 1 - 1e-7)
        y_f = pnp.array(y_batch, dtype=float)
        loss = -pnp.mean(y_f * pnp.log(probs) + (1 - y_f) * pnp.log(1 - probs))
        return loss

    batch_size = min(32, len(X_train))

    for epoch in range(epochs):
        idx = np.random.choice(len(X_train), batch_size, replace=False)
        X_b, y_b = X_train[idx], y_train[idx]
        weights, loss_val = opt.step_and_cost(cost, weights, X_b, y_b)
        loss_history.append(float(loss_val))

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {loss_val:.4f}")

    return weights, loss_history


def predict_qml(circuit, weights, X, n_qubits=4):
    """Returns binary predictions from the trained VQC."""
    raw = np.array([circuit(x[:n_qubits], weights) for x in X])
    return (raw >= 0.0).astype(int)   # expval ≥ 0 → class 1


def get_quantum_probs(circuit, weights, X, n_qubits=4):
    """Returns soft probabilities ∈ [0,1] for explainability."""
    raw = np.array([circuit(x[:n_qubits], weights) for x in X])
    return (raw + 1) / 2


# ─────────────────────────────────────────────
# 3. EXPLAINABILITY MODULE
# ─────────────────────────────────────────────

def compute_shap_values(circuit, weights, X_train, X_test, feature_names, n_qubits=4):
    """
    Computes SHAP values for the quantum model using KernelExplainer.
    Returns shap_values array and explainer object.
    """
    import shap

    def predict_fn(X_arr):
        return get_quantum_probs(circuit, weights, X_arr, n_qubits)

    # Use a small background sample for speed
    bg = shap.kmeans(X_train, 20)
    explainer = shap.KernelExplainer(predict_fn, bg)

    # Explain first 50 test samples
    n_explain = min(50, len(X_test))
    shap_values = explainer.shap_values(X_test[:n_explain], nsamples=80)
    return shap_values, explainer, X_test[:n_explain]


def sensitivity_analysis(circuit, weights, X_test, feature_names, n_qubits=4, delta=0.1):
    """
    Perturbs each feature by ±delta and measures prediction change.
    Returns a feature sensitivity score array.
    """
    base_probs = get_quantum_probs(circuit, weights, X_test, n_qubits)
    sensitivities = []

    for f_idx in range(min(n_qubits, len(feature_names))):
        X_plus = X_test.copy()
        X_minus = X_test.copy()
        X_plus[:, f_idx] += delta
        X_minus[:, f_idx] -= delta

        probs_plus = get_quantum_probs(circuit, weights, X_plus, n_qubits)
        probs_minus = get_quantum_probs(circuit, weights, X_minus, n_qubits)

        sensitivity = np.mean(np.abs(probs_plus - base_probs) + np.abs(probs_minus - base_probs)) / 2
        sensitivities.append(sensitivity)

    # Pad remaining features with 0
    sensitivities += [0.0] * (len(feature_names) - len(sensitivities))
    return np.array(sensitivities)


# ─────────────────────────────────────────────
# 4. QUANTUM CIRCUIT DIAGRAM
# ─────────────────────────────────────────────

def get_circuit_diagram(n_qubits=4, n_layers=2):
    """Returns a text/ASCII diagram of the VQC structure."""
    import pennylane as qml
    circuit, weight_shape, _ = build_qml_model(n_qubits, n_layers)
    weights_demo = np.zeros(weight_shape)
    inputs_demo = np.zeros(n_qubits)
    return qml.draw(circuit)(inputs_demo, weights_demo)


# ─────────────────────────────────────────────
# 5. FULL PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(n_qubits=4, n_layers=2, epochs=30):
    print("\n" + "="*60)
    print("  Explainable QML Framework — ASD Analysis")
    print("="*60)

    # Load & preprocess
    X_raw, y_raw = load_asd_dataset()
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(X_raw, y_raw)
    print(f"[✓] Dataset: {len(X_train)} train / {len(X_test)} test samples")
    print(f"[✓] Features: {feature_names}")

    # Train QML
    print("\n[→] Training Variational Quantum Circuit...")
    circuit, weight_shape, _ = build_qml_model(n_qubits, n_layers)
    weights, loss_history = train_qml(X_train, y_train, n_qubits, n_layers, epochs)

    # Evaluate
    y_pred = predict_qml(circuit, weights, X_test, n_qubits)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[✓] Test Accuracy: {acc*100:.1f}%")
    print(classification_report(y_test, y_pred, target_names=["Non-ASD", "ASD"]))

    # Explainability
    print("\n[→] Computing SHAP values (quantum KernelExplainer)...")
    shap_vals, explainer, X_explain = compute_shap_values(
        circuit, weights, X_train, X_test, feature_names, n_qubits
    )

    print("[→] Running sensitivity analysis...")
    sensitivities = sensitivity_analysis(circuit, weights, X_test, feature_names, n_qubits)

    results = {
        "accuracy": acc,
        "loss_history": loss_history,
        "y_test": y_test,
        "y_pred": y_pred,
        "shap_values": shap_vals,
        "X_explain": X_explain,
        "feature_names": feature_names,
        "sensitivities": sensitivities,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "weights": weights,
        "n_qubits": n_qubits,
    }
    print("\n[✓] Pipeline complete.")
    return results


if __name__ == "__main__":
    results = run_pipeline(n_qubits=4, n_layers=2, epochs=30)
