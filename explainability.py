"""
Explainability Module (Pure NumPy)
====================================
Implements SHAP-style feature attribution via:
  1. Kernel SHAP approximation (weighted linear model on coalitions)
  2. Sensitivity / gradient analysis (finite difference)
  3. LIME-style local linear approximation

No external explainability libraries required.
"""

import numpy as np
from itertools import combinations


# ─────────────────────────────────────────────────────────────
# 1. Kernel SHAP (subset-sampling approximation)
# ─────────────────────────────────────────────────────────────

def kernel_shap(predict_fn, x, X_background, n_samples=200):
    """
    Approximates SHAP values for a single instance x.

    predict_fn : callable(X) → 1-D array of predictions
    x          : 1-D feature vector (1 sample)
    X_background: 2-D background dataset (used as baseline)
    n_samples  : number of random coalitions to sample
    """
    n_feat = len(x)
    baseline = X_background.mean(axis=0)
    f_baseline = float(np.squeeze(predict_fn(baseline[None])))
    f_full = float(np.squeeze(predict_fn(x[None])))

    # Sample random coalitions
    coalitions = []
    for _ in range(n_samples):
        # Random subset of features
        size = np.random.randint(1, n_feat)
        subset = np.random.choice(n_feat, size, replace=False)
        mask = np.zeros(n_feat, dtype=bool)
        mask[subset] = True
        coalitions.append(mask)

    # Shapley kernel weights  w(|S|) = (n-1) / (C(n,|S|) * |S| * (n-|S|))
    W, Y = [], []
    for mask in coalitions:
        sz = mask.sum()
        if sz == 0 or sz == n_feat:
            continue
        w = (n_feat - 1) / (_binomial(n_feat, sz) * sz * (n_feat - sz) + 1e-9)

        # Interventional prediction: replace masked-out features with baseline
        x_masked = x.copy()
        x_masked[~mask] = baseline[~mask]
        y_pred = float(np.squeeze(predict_fn(x_masked[None])))

        W.append(w)
        Y.append(y_pred - f_baseline)

    # Weighted least squares: φ = (Z^T W Z)^-1 Z^T W y
    Z = np.array([m.astype(float) for m, _ in
                  zip(coalitions, range(len(W)))], dtype=float)
    Z = Z[:len(W)]
    W_diag = np.diag(W)
    Y_arr = np.array(Y)

    try:
        ZtW = Z.T @ W_diag
        phi = np.linalg.lstsq(ZtW @ Z, ZtW @ Y_arr, rcond=None)[0]
    except Exception:
        phi = np.zeros(n_feat)

    return phi


def batch_kernel_shap(predict_fn, X_explain, X_background, n_samples=150):
    """Compute SHAP values for multiple instances."""
    shap_vals = []
    for i, x in enumerate(X_explain):
        phi = kernel_shap(predict_fn, x, X_background, n_samples)
        shap_vals.append(phi)
        if (i + 1) % 10 == 0:
            print(f"  SHAP: {i+1}/{len(X_explain)} samples processed")
    return np.array(shap_vals)


def _binomial(n, k):
    from math import comb
    return max(comb(n, k), 1)


# ─────────────────────────────────────────────────────────────
# 2. Sensitivity Analysis
# ─────────────────────────────────────────────────────────────

def sensitivity_analysis(predict_fn, X_test, feature_names, delta=0.15):
    """
    Measures how much each feature's perturbation changes the output.
    Returns per-feature mean absolute gradient.
    """
    n_feat = len(feature_names)
    base_probs = predict_fn(X_test)
    sensitivities = []

    for fi in range(n_feat):
        X_plus  = X_test.copy(); X_plus[:, fi]  += delta
        X_minus = X_test.copy(); X_minus[:, fi] -= delta

        grad = (predict_fn(X_plus) - predict_fn(X_minus)) / (2 * delta)
        sensitivities.append(float(np.mean(np.abs(grad))))

    return np.array(sensitivities)


# ─────────────────────────────────────────────────────────────
# 3. LIME-style local linear explanation
# ─────────────────────────────────────────────────────────────

def lime_explain(predict_fn, x, X_background, n_samples=300, kernel_width=0.75):
    """
    Local Interpretable Model-agnostic Explanations for one sample.
    Returns feature coefficients of the local linear model.
    """
    n_feat = len(x)
    std = X_background.std(axis=0) + 1e-9

    # Perturb around x
    noise = np.random.normal(0, 1, (n_samples, n_feat)) * std * 0.3
    X_pert = x[None] + noise

    # Kernel weights based on distance
    dists = np.sqrt(((X_pert - x)**2).sum(axis=1)) / (n_feat * kernel_width)
    weights = np.exp(-dists**2)

    # Get predictions
    y_pert = predict_fn(X_pert)

    # Weighted linear regression
    W = np.diag(weights)
    try:
        coefs = np.linalg.lstsq(X_pert.T @ W @ X_pert,
                                 X_pert.T @ W @ y_pert, rcond=None)[0]
    except Exception:
        coefs = np.zeros(n_feat)

    return coefs


# ─────────────────────────────────────────────────────────────
# 4. Waterfall values for a single prediction
# ─────────────────────────────────────────────────────────────

def waterfall_values(shap_row, feature_names, base_value):
    """Formats a single row of SHAP values for waterfall plotting."""
    order = np.argsort(np.abs(shap_row))[::-1]
    cumulative = base_value
    steps = [("baseline", base_value, 0, base_value)]
    for fi in order[:8]:
        prev = cumulative
        cumulative += shap_row[fi]
        steps.append((feature_names[fi], shap_row[fi], prev, cumulative))
    return steps
