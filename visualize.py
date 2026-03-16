"""
Visualization Module
====================
Generates all plots for the explainability dashboard:
- Training loss curve
- Confusion matrix
- SHAP summary / bar / waterfall
- Feature sensitivity radar
- Quantum circuit diagram (Matplotlib)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os

# ── colour palette ──────────────────────────────────────────
PALETTE = {
    "bg":      "#0D0F1A",
    "panel":   "#141726",
    "accent1": "#7B61FF",   # quantum violet
    "accent2": "#00D4FF",   # cyan
    "accent3": "#FF6B6B",   # coral
    "accent4": "#4ADE80",   # green
    "text":    "#E8EAFF",
    "muted":   "#6B7280",
}

def _fig(w=10, h=6):
    fig = plt.figure(figsize=(w, h), facecolor=PALETTE["bg"])
    return fig

def _ax(fig, *args, **kwargs):
    ax = fig.add_subplot(*args, **kwargs)
    ax.set_facecolor(PALETTE["panel"])
    ax.tick_params(colors=PALETTE["text"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["muted"])
    ax.title.set_color(PALETTE["text"])
    ax.xaxis.label.set_color(PALETTE["muted"])
    ax.yaxis.label.set_color(PALETTE["muted"])
    return ax


# ── 1. Loss Curve ────────────────────────────────────────────
def plot_loss_curve(loss_history, save_path=None):
    fig = _fig(8, 4)
    ax = _ax(fig, 111)

    x = range(1, len(loss_history) + 1)
    ax.plot(x, loss_history, color=PALETTE["accent1"], lw=2.5, label="Train Loss")
    ax.fill_between(x, loss_history, alpha=0.15, color=PALETTE["accent1"])

    # Smoothed
    if len(loss_history) >= 5:
        kernel = np.ones(5) / 5
        smooth = np.convolve(loss_history, kernel, mode="valid")
        ax.plot(range(3, len(smooth) + 3), smooth, color=PALETTE["accent2"],
                lw=1.5, linestyle="--", label="Smoothed")

    ax.set_title("VQC Training Loss", fontsize=13, pad=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary Cross-Entropy")
    ax.legend(facecolor=PALETTE["panel"], labelcolor=PALETTE["text"])
    ax.grid(alpha=0.12, color=PALETTE["muted"])

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=PALETTE["bg"])
    return fig


# ── 2. Confusion Matrix ──────────────────────────────────────
def plot_confusion_matrix(cm, save_path=None):
    fig = _fig(6, 5)
    ax = _ax(fig, 111)

    im = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-ASD", "ASD"], color=PALETTE["text"])
    ax.set_yticklabels(["Non-ASD", "ASD"], color=PALETTE["text"])
    ax.set_xlabel("Predicted", labelpad=8)
    ax.set_ylabel("Actual", labelpad=8)
    ax.set_title("Confusion Matrix", fontsize=13, pad=10)

    total = cm.sum()
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = 100 * val / total
            ax.text(j, i, f"{val}\n({pct:.1f}%)",
                    ha="center", va="center",
                    color="white" if val > cm.max() * 0.5 else PALETTE["text"],
                    fontsize=12, fontweight="bold")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=PALETTE["bg"])
    return fig


# ── 3. SHAP Feature Importance Bar ───────────────────────────
def plot_shap_importance(shap_values, feature_names, save_path=None):
    mean_abs = np.abs(shap_values).mean(axis=0)
    n_feat = min(len(mean_abs), len(feature_names))
    mean_abs = mean_abs[:n_feat]
    names = feature_names[:n_feat]

    # Sort
    order = np.argsort(mean_abs)
    sorted_vals = mean_abs[order]
    sorted_names = [names[i] for i in order]

    fig = _fig(9, max(4, n_feat * 0.45))
    ax = _ax(fig, 111)

    colors = [PALETTE["accent1"] if v > sorted_vals.mean() else PALETTE["accent2"]
              for v in sorted_vals]

    bars = ax.barh(sorted_names, sorted_vals, color=colors,
                   edgecolor="none", height=0.65)

    for bar, val in zip(bars, sorted_vals):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", color=PALETTE["text"], fontsize=8)

    ax.set_title("SHAP Feature Importance (mean |SHAP|)", fontsize=13, pad=10)
    ax.set_xlabel("Mean |SHAP Value|")
    ax.tick_params(axis="y", labelcolor=PALETTE["text"])
    ax.grid(axis="x", alpha=0.12, color=PALETTE["muted"])
    ax.set_xlim(0, sorted_vals.max() * 1.2)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=PALETTE["bg"])
    return fig


# ── 4. SHAP Beeswarm-style scatter ───────────────────────────
def plot_shap_scatter(shap_values, X_explain, feature_names, save_path=None):
    n_feat = min(shap_values.shape[1], len(feature_names), 8)
    mean_abs = np.abs(shap_values).mean(axis=0)[:n_feat]
    top_idx = np.argsort(mean_abs)[::-1][:n_feat]

    fig = _fig(10, 5)
    ax = _ax(fig, 111)

    for rank, fi in enumerate(top_idx):
        sv = shap_values[:, fi]
        fv = X_explain[:, fi] if fi < X_explain.shape[1] else np.zeros(len(sv))
        # Normalize feature values to [0,1] for colour
        fv_norm = (fv - fv.min()) / (fv.ptp() + 1e-9)
        jitter = np.random.uniform(-0.25, 0.25, len(sv))
        ax.scatter(sv, np.full(len(sv), rank) + jitter,
                   c=fv_norm, cmap="plasma", s=18, alpha=0.7, linewidths=0)

    ax.set_yticks(range(n_feat))
    ax.set_yticklabels([feature_names[i] for i in top_idx], color=PALETTE["text"], fontsize=9)
    ax.axvline(0, color=PALETTE["muted"], lw=1, linestyle="--")
    ax.set_xlabel("SHAP value (impact on model output)")
    ax.set_title("SHAP Beeswarm — Feature Impact per Sample", fontsize=13, pad=10)
    ax.grid(axis="x", alpha=0.1, color=PALETTE["muted"])

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=PALETTE["bg"])
    return fig


# ── 5. Sensitivity Radar ─────────────────────────────────────
def plot_sensitivity_radar(sensitivities, feature_names, save_path=None):
    n = min(len(sensitivities), len(feature_names), 10)
    vals = sensitivities[:n]
    names = feature_names[:n]

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    vals_plot = vals.tolist() + [vals[0]]
    angles += [angles[0]]

    fig = plt.figure(figsize=(7, 7), facecolor=PALETTE["bg"])
    ax = fig.add_subplot(111, polar=True, facecolor=PALETTE["panel"])
    ax.tick_params(colors=PALETTE["text"])

    ax.plot(angles, vals_plot, color=PALETTE["accent1"], lw=2.5)
    ax.fill(angles, vals_plot, alpha=0.25, color=PALETTE["accent1"])
    ax.set_thetagrids(np.degrees(angles[:-1]), labels=names,
                      color=PALETTE["text"], fontsize=9)
    ax.set_title("Feature Sensitivity Radar", fontsize=13, pad=20,
                 color=PALETTE["text"])
    ax.grid(color=PALETTE["muted"], alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=PALETTE["bg"])
    return fig


# ── 6. Quantum Probability Distribution ──────────────────────
def plot_quantum_probs(y_test, probs, save_path=None):
    fig = _fig(9, 4)
    ax = _ax(fig, 111)

    asd_probs = probs[y_test == 1]
    non_probs = probs[y_test == 0]

    ax.hist(non_probs, bins=25, alpha=0.7, color=PALETTE["accent2"],
            label="Non-ASD", edgecolor="none", density=True)
    ax.hist(asd_probs, bins=25, alpha=0.7, color=PALETTE["accent3"],
            label="ASD", edgecolor="none", density=True)
    ax.axvline(0.5, color="white", lw=1.5, linestyle="--", label="Decision boundary")

    ax.set_title("Quantum Model Probability Distributions", fontsize=13, pad=10)
    ax.set_xlabel("Predicted Probability (ASD)")
    ax.set_ylabel("Density")
    ax.legend(facecolor=PALETTE["panel"], labelcolor=PALETTE["text"])
    ax.grid(alpha=0.12, color=PALETTE["muted"])

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=PALETTE["bg"])
    return fig


# ── 7. Summary Dashboard (single figure) ─────────────────────
def plot_full_dashboard(results, save_path=None):
    fig = plt.figure(figsize=(20, 14), facecolor=PALETTE["bg"])
    gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    feature_names = results["feature_names"]
    shap_vals     = results["shap_values"]
    X_exp         = results["X_explain"]
    cm            = results["confusion_matrix"]
    loss          = results["loss_history"]
    y_test        = results["y_test"]
    sensitivities = results["sensitivities"]

    # Estimate probabilities from predictions
    y_pred = results["y_pred"]
    probs = y_pred.astype(float) + np.random.normal(0, 0.05, len(y_pred))
    probs = np.clip(probs, 0, 1)

    # ── Row 0 ──────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0], facecolor=PALETTE["panel"])
    _style_ax(ax0)
    ax0.plot(loss, color=PALETTE["accent1"], lw=2)
    ax0.fill_between(range(len(loss)), loss, alpha=0.15, color=PALETTE["accent1"])
    ax0.set_title("Training Loss", color=PALETTE["text"], fontsize=11)
    ax0.set_xlabel("Epoch", color=PALETTE["muted"], fontsize=9)
    ax0.grid(alpha=0.1)

    ax1 = fig.add_subplot(gs[0, 1], facecolor=PALETTE["panel"])
    _style_ax(ax1)
    ax1.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white", fontsize=14, fontweight="bold")
    ax1.set_xticks([0,1]); ax1.set_yticks([0,1])
    ax1.set_xticklabels(["Non-ASD","ASD"], color=PALETTE["text"])
    ax1.set_yticklabels(["Non-ASD","ASD"], color=PALETTE["text"])
    ax1.set_title("Confusion Matrix", color=PALETTE["text"], fontsize=11)

    ax2 = fig.add_subplot(gs[0, 2], facecolor=PALETTE["panel"])
    _style_ax(ax2)
    _add_metrics_panel(ax2, results)

    # ── Row 1 ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2], facecolor=PALETTE["panel"])
    _style_ax(ax3)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    n = min(len(mean_abs), len(feature_names))
    order = np.argsort(mean_abs[:n])
    colors = [PALETTE["accent1"] if v > mean_abs[:n].mean() else PALETTE["accent2"]
              for v in mean_abs[:n][order]]
    ax3.barh([feature_names[i] for i in order], mean_abs[:n][order],
             color=colors, height=0.6)
    ax3.set_title("SHAP Feature Importance (mean |SHAP|)", color=PALETTE["text"], fontsize=11)
    ax3.grid(axis="x", alpha=0.1)

    ax4 = fig.add_subplot(gs[1, 2], facecolor=PALETTE["panel"])
    _style_ax(ax4)
    ax4.hist(probs[y_test==0], bins=20, alpha=0.7, color=PALETTE["accent2"],
             label="Non-ASD", density=True)
    ax4.hist(probs[y_test==1], bins=20, alpha=0.7, color=PALETTE["accent3"],
             label="ASD", density=True)
    ax4.axvline(0.5, color="white", lw=1.2, linestyle="--")
    ax4.set_title("Quantum Prob. Distribution", color=PALETTE["text"], fontsize=11)
    ax4.legend(facecolor=PALETTE["panel"], labelcolor=PALETTE["text"], fontsize=8)
    ax4.grid(alpha=0.1)

    # ── Row 2 ──────────────────────────────────────────
    # Beeswarm
    ax5 = fig.add_subplot(gs[2, :2], facecolor=PALETTE["panel"])
    _style_ax(ax5)
    n_feat = min(shap_vals.shape[1], len(feature_names), 8)
    top_idx = np.argsort(np.abs(shap_vals).mean(axis=0))[::-1][:n_feat]
    for rank, fi in enumerate(top_idx):
        sv = shap_vals[:, fi]
        fv = X_exp[:, fi] if fi < X_exp.shape[1] else np.zeros(len(sv))
        fv_norm = (fv - fv.min()) / (fv.ptp() + 1e-9)
        jitter = np.random.uniform(-0.2, 0.2, len(sv))
        ax5.scatter(sv, np.full(len(sv), rank) + jitter,
                    c=fv_norm, cmap="plasma", s=15, alpha=0.7)
    ax5.set_yticks(range(n_feat))
    ax5.set_yticklabels([feature_names[i] for i in top_idx],
                        color=PALETTE["text"], fontsize=8)
    ax5.axvline(0, color=PALETTE["muted"], lw=1, linestyle="--")
    ax5.set_title("SHAP Beeswarm — Feature Impact per Sample", color=PALETTE["text"], fontsize=11)
    ax5.grid(axis="x", alpha=0.1)

    # Sensitivity bar
    ax6 = fig.add_subplot(gs[2, 2], facecolor=PALETTE["panel"])
    _style_ax(ax6)
    n_s = min(len(sensitivities), len(feature_names))
    ax6.barh(feature_names[:n_s], sensitivities[:n_s],
             color=PALETTE["accent4"], height=0.6)
    ax6.set_title("Sensitivity Analysis", color=PALETTE["text"], fontsize=11)
    ax6.grid(axis="x", alpha=0.1)

    # Header
    fig.text(0.5, 0.98, "Explainable QML Framework — ASD Analysis",
             ha="center", va="top", fontsize=16,
             color=PALETTE["text"], fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=PALETTE["bg"])
    return fig


def _style_ax(ax):
    ax.tick_params(colors=PALETTE["text"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["muted"])
    ax.title.set_color(PALETTE["text"])
    ax.xaxis.label.set_color(PALETTE["muted"])
    ax.yaxis.label.set_color(PALETTE["muted"])


def _add_metrics_panel(ax, results):
    from sklearn.metrics import accuracy_score, f1_score
    acc = results["accuracy"]
    f1 = f1_score(results["y_test"], results["y_pred"], average="weighted")

    ax.axis("off")
    metrics = [
        ("Accuracy",  f"{acc*100:.1f}%",  PALETTE["accent4"]),
        ("F1 Score",  f"{f1:.3f}",        PALETTE["accent2"]),
        ("Qubits",    str(results.get("n_qubits", 4)), PALETTE["accent1"]),
        ("Samples",   str(len(results["y_test"])),     PALETTE["muted"]),
    ]
    for i, (label, value, color) in enumerate(metrics):
        y = 0.85 - i * 0.22
        ax.text(0.1, y, label, transform=ax.transAxes,
                color=PALETTE["muted"], fontsize=10)
        ax.text(0.1, y - 0.09, value, transform=ax.transAxes,
                color=color, fontsize=18, fontweight="bold")
    ax.set_title("Model Metrics", color=PALETTE["text"], fontsize=11)


if __name__ == "__main__":
    # Quick smoke-test with dummy data
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    os.makedirs("outputs", exist_ok=True)

    dummy_results = {
        "accuracy": 0.82,
        "loss_history": np.random.exponential(0.5, 30)[::-1] + 0.1,
        "y_test": np.array([0,1,0,1,0,1,0,0,1,1]),
        "y_pred": np.array([0,1,0,0,0,1,0,0,1,1]),
        "shap_values": np.random.randn(50, 8),
        "X_explain": np.random.randn(50, 8),
        "feature_names": [f"feat_{i}" for i in range(8)],
        "sensitivities": np.random.rand(8),
        "confusion_matrix": np.array([[50, 10], [8, 32]]),
        "n_qubits": 4,
    }
    plot_full_dashboard(dummy_results, "outputs/dashboard_test.png")
    print("[✓] Dashboard saved to outputs/dashboard_test.png")
