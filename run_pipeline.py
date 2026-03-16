"""
run_pipeline.py
================
Full end-to-end pipeline:
  1. Load / generate ASD dataset
  2. Preprocess
  3. Train VQC
  4. Evaluate
  5. Run explainability (SHAP approx + sensitivity + LIME)
  6. Save all visualisation plots
"""

import numpy as np
import pandas as pd
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score,
                              classification_report, confusion_matrix)

from quantum_sim import VQC
from explainability import batch_kernel_shap, sensitivity_analysis, lime_explain

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

os.makedirs("outputs", exist_ok=True)

PALETTE = {
    "bg":      "#0D0F1A",
    "panel":   "#141726",
    "accent1": "#7B61FF",
    "accent2": "#00D4FF",
    "accent3": "#FF6B6B",
    "accent4": "#4ADE80",
    "text":    "#E8EAFF",
    "muted":   "#6B7280",
}

# ─────────────────────────────────────────────
# 1. DATA
# ─────────────────────────────────────────────

def generate_asd_dataset():
    np.random.seed(42)
    n = 700
    feature_names = [
        "A1_Score","A2_Score","A3_Score","A4_Score","A5_Score",
        "A6_Score","A7_Score","A8_Score","A9_Score","A10_Score",
        "age","gender","jaundice","autism_family","result"
    ]
    qtchat = np.random.randint(0, 2, (n, 10))
    score_sum = qtchat.sum(axis=1)
    prob_asd = 1 / (1 + np.exp(-(score_sum - 5)))
    labels = (np.random.rand(n) < prob_asd).astype(int)
    age = np.random.randint(18, 65, n)
    gender = np.random.randint(0, 2, n)
    jaundice = np.random.randint(0, 2, n)
    autism_family = np.random.randint(0, 2, n)
    result = score_sum
    X = np.column_stack([qtchat, age, gender, jaundice, autism_family, result]).astype(float)
    return X, labels, feature_names

X_raw, y_raw, FEAT_NAMES = generate_asd_dataset()
print(f"[✓] Dataset: {X_raw.shape[0]} samples, {X_raw.shape[1]} features")

scaler = StandardScaler()
X_sc = scaler.fit_transform(X_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X_sc, y_raw, test_size=0.25, random_state=42, stratify=y_raw
)
print(f"[✓] Train: {len(X_train)}  Test: {len(X_test)}")

# ─────────────────────────────────────────────
# 2. TRAIN VQC
# ─────────────────────────────────────────────

N_QUBITS = 4
N_LAYERS = 2
EPOCHS   = 40

print(f"\n[→] Training VQC ({N_QUBITS} qubits, {N_LAYERS} layers, {EPOCHS} epochs)...")
vqc = VQC(n_qubits=N_QUBITS, n_layers=N_LAYERS, seed=7)
loss_history = vqc.train(X_train, y_train, epochs=EPOCHS, lr=0.04, batch_size=24)

# ─────────────────────────────────────────────
# 3. EVALUATE
# ─────────────────────────────────────────────

y_pred  = vqc.predict(X_test)
y_probs = vqc.predict_prob(X_test)
acc  = accuracy_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred, average="weighted")
cm   = confusion_matrix(y_test, y_pred)

print(f"\n[✓] Accuracy : {acc*100:.1f}%")
print(f"[✓] F1 Score : {f1:.3f}")
print(classification_report(y_test, y_pred, target_names=["Non-ASD","ASD"]))

# ─────────────────────────────────────────────
# 4. EXPLAINABILITY
# ─────────────────────────────────────────────

print("\n[→] Computing SHAP values (KernelSHAP approx)...")
n_explain = 60
predict_fn = lambda X: vqc.predict_prob(X)
shap_vals = batch_kernel_shap(predict_fn, X_test[:n_explain], X_train, n_samples=120)
X_explain = X_test[:n_explain]

print("[→] Sensitivity analysis...")
sensitivities = sensitivity_analysis(predict_fn, X_test, FEAT_NAMES)

print("[→] LIME local explanation (first test sample)...")
lime_coefs = lime_explain(predict_fn, X_test[0], X_train, n_samples=400)

base_val = float(np.mean(y_probs))

# ─────────────────────────────────────────────
# 5. PLOTS
# ─────────────────────────────────────────────

def style_ax(ax):
    ax.set_facecolor(PALETTE["panel"])
    ax.tick_params(colors=PALETTE["text"], labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(PALETTE["muted"])
    ax.title.set_color(PALETTE["text"])
    ax.xaxis.label.set_color(PALETTE["muted"])
    ax.yaxis.label.set_color(PALETTE["muted"])


# ── Fig 1: Training Loss ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(9,4), facecolor=PALETTE["bg"])
style_ax(ax)
x = range(1, len(loss_history)+1)
ax.plot(x, loss_history, color=PALETTE["accent1"], lw=2.5, label="Train Loss")
ax.fill_between(x, loss_history, alpha=0.15, color=PALETTE["accent1"])
if len(loss_history) >= 5:
    k = np.ones(5)/5
    sm = np.convolve(loss_history, k, mode="valid")
    ax.plot(range(3, len(sm)+3), sm, color=PALETTE["accent2"], lw=1.8,
            linestyle="--", label="Smoothed (window=5)")
ax.set_title("VQC Training Loss", fontsize=13, pad=10)
ax.set_xlabel("Epoch"); ax.set_ylabel("Binary Cross-Entropy")
ax.legend(facecolor=PALETTE["panel"], labelcolor=PALETTE["text"])
ax.grid(alpha=0.12, color=PALETTE["muted"])
fig.tight_layout()
fig.savefig("outputs/01_training_loss.png", dpi=130, bbox_inches="tight", facecolor=PALETTE["bg"])
plt.close(); print("[✓] Saved 01_training_loss.png")

# ── Fig 2: Confusion Matrix ──────────────────────────────────
fig, ax = plt.subplots(figsize=(6,5), facecolor=PALETTE["bg"])
style_ax(ax)
im = ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        v = cm[i,j]; pct = 100*v/cm.sum()
        ax.text(j, i, f"{v}\n({pct:.1f}%)", ha="center", va="center",
                color="white" if v > cm.max()*0.5 else PALETTE["text"],
                fontsize=13, fontweight="bold")
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["Non-ASD","ASD"], color=PALETTE["text"])
ax.set_yticklabels(["Non-ASD","ASD"], color=PALETTE["text"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix", fontsize=13, pad=10)
fig.tight_layout()
fig.savefig("outputs/02_confusion_matrix.png", dpi=130, bbox_inches="tight", facecolor=PALETTE["bg"])
plt.close(); print("[✓] Saved 02_confusion_matrix.png")

# ── Fig 3: SHAP Feature Importance ──────────────────────────
mean_abs = np.abs(shap_vals).mean(axis=0)
n_f = min(len(mean_abs), len(FEAT_NAMES))
order = np.argsort(mean_abs[:n_f])
sorted_v = mean_abs[:n_f][order]
sorted_n = [FEAT_NAMES[i] for i in order]

fig, ax = plt.subplots(figsize=(10, max(4, n_f*0.48)), facecolor=PALETTE["bg"])
style_ax(ax)
colors = [PALETTE["accent1"] if v > sorted_v.mean() else PALETTE["accent2"] for v in sorted_v]
bars = ax.barh(sorted_n, sorted_v, color=colors, height=0.62, edgecolor="none")
for bar, val in zip(bars, sorted_v):
    ax.text(val+0.0005, bar.get_y()+bar.get_height()/2,
            f"{val:.4f}", va="center", color=PALETTE["text"], fontsize=8)
ax.set_title("SHAP Feature Importance (mean |SHAP value|)", fontsize=13, pad=10)
ax.set_xlabel("Mean |SHAP Value|")
ax.grid(axis="x", alpha=0.12, color=PALETTE["muted"])
ax.set_xlim(0, sorted_v.max()*1.25)
fig.tight_layout()
fig.savefig("outputs/03_shap_importance.png", dpi=130, bbox_inches="tight", facecolor=PALETTE["bg"])
plt.close(); print("[✓] Saved 03_shap_importance.png")

# ── Fig 4: SHAP Beeswarm ────────────────────────────────────
n_feat_show = min(shap_vals.shape[1], len(FEAT_NAMES), 10)
top_idx = np.argsort(np.abs(shap_vals).mean(axis=0))[::-1][:n_feat_show]

fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=PALETTE["bg"])
style_ax(ax)
for rank, fi in enumerate(top_idx):
    sv = shap_vals[:, fi]
    fv = X_explain[:, fi] if fi < X_explain.shape[1] else np.zeros(len(sv))
    fv_norm = (fv - fv.min()) / ((fv.max()-fv.min())+1e-9)
    jitter = np.random.uniform(-0.22, 0.22, len(sv))
    sc = ax.scatter(sv, np.full(len(sv), rank)+jitter,
                    c=fv_norm, cmap="plasma", s=22, alpha=0.75, linewidths=0)

ax.set_yticks(range(n_feat_show))
ax.set_yticklabels([FEAT_NAMES[i] for i in top_idx], color=PALETTE["text"], fontsize=9)
ax.axvline(0, color=PALETTE["muted"], lw=1.2, linestyle="--")
ax.set_xlabel("SHAP value  (impact on model output → ASD probability)")
ax.set_title("SHAP Beeswarm — Feature Impact per Sample", fontsize=13, pad=10)
ax.grid(axis="x", alpha=0.1, color=PALETTE["muted"])
cbar = plt.colorbar(sc, ax=ax, pad=0.01)
cbar.set_label("Feature value (low→high)", color=PALETTE["muted"], fontsize=8)
cbar.ax.yaxis.set_tick_params(color=PALETTE["text"])
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=PALETTE["text"])
fig.tight_layout()
fig.savefig("outputs/04_shap_beeswarm.png", dpi=130, bbox_inches="tight", facecolor=PALETTE["bg"])
plt.close(); print("[✓] Saved 04_shap_beeswarm.png")

# ── Fig 5: Sensitivity Radar ────────────────────────────────
n_r = min(len(sensitivities), len(FEAT_NAMES), 10)
vals_r = sensitivities[:n_r]
names_r = FEAT_NAMES[:n_r]
angles = np.linspace(0, 2*np.pi, n_r, endpoint=False)
vals_p = np.append(vals_r, vals_r[0])
angs_p = np.append(angles, angles[0])

fig = plt.figure(figsize=(7,7), facecolor=PALETTE["bg"])
ax = fig.add_subplot(111, polar=True, facecolor=PALETTE["panel"])
ax.tick_params(colors=PALETTE["text"])
ax.plot(angs_p, vals_p, color=PALETTE["accent1"], lw=2.5)
ax.fill(angs_p, vals_p, alpha=0.25, color=PALETTE["accent1"])
ax.scatter(angles, vals_r, color=PALETTE["accent2"], s=50, zorder=5)
ax.set_thetagrids(np.degrees(angles), labels=names_r,
                  color=PALETTE["text"], fontsize=9)
ax.set_title("Feature Sensitivity Radar", fontsize=13, pad=20, color=PALETTE["text"])
ax.grid(color=PALETTE["muted"], alpha=0.3)
fig.tight_layout()
fig.savefig("outputs/05_sensitivity_radar.png", dpi=130, bbox_inches="tight", facecolor=PALETTE["bg"])
plt.close(); print("[✓] Saved 05_sensitivity_radar.png")

# ── Fig 6: Quantum Probability Distribution ──────────────────
fig, ax = plt.subplots(figsize=(9,4), facecolor=PALETTE["bg"])
style_ax(ax)
ax.hist(y_probs[y_test==0], bins=28, alpha=0.72, color=PALETTE["accent2"],
        label="Non-ASD (actual)", edgecolor="none", density=True)
ax.hist(y_probs[y_test==1], bins=28, alpha=0.72, color=PALETTE["accent3"],
        label="ASD (actual)", edgecolor="none", density=True)
ax.axvline(0.5, color="white", lw=2, linestyle="--", label="Decision boundary (0.5)")
ax.set_title("Quantum Model — Predicted Probability Distributions", fontsize=13, pad=10)
ax.set_xlabel("P(ASD) from VQC")
ax.set_ylabel("Density")
ax.legend(facecolor=PALETTE["panel"], labelcolor=PALETTE["text"])
ax.grid(alpha=0.1, color=PALETTE["muted"])
fig.tight_layout()
fig.savefig("outputs/06_quantum_probs.png", dpi=130, bbox_inches="tight", facecolor=PALETTE["bg"])
plt.close(); print("[✓] Saved 06_quantum_probs.png")

# ── Fig 7: LIME Waterfall ────────────────────────────────────
n_lime = min(len(lime_coefs), len(FEAT_NAMES))
lc = lime_coefs[:n_lime]
order_l = np.argsort(np.abs(lc))[::-1][:10]
vals_l = lc[order_l]
names_l = [FEAT_NAMES[i] for i in order_l]

# Waterfall
running = base_val
xs = [running]
for v in vals_l:
    running += v
    xs.append(running)

fig, ax = plt.subplots(figsize=(10, 5), facecolor=PALETTE["bg"])
style_ax(ax)
bottoms = [min(xs[i], xs[i+1]) for i in range(len(vals_l))]
heights = np.abs(vals_l)
bar_colors = [PALETTE["accent4"] if v >= 0 else PALETTE["accent3"] for v in vals_l]

ax.bar(range(len(vals_l)), heights, bottom=bottoms, color=bar_colors,
       width=0.6, edgecolor="none")
ax.step(range(-1, len(vals_l)+1),
        [base_val] + [xs[i+1] for i in range(len(vals_l))] + [xs[-1]],
        where="post", color=PALETTE["muted"], lw=1.2, linestyle="--")
ax.axhline(base_val, color=PALETTE["muted"], lw=1, linestyle=":")
ax.set_xticks(range(len(vals_l)))
ax.set_xticklabels(names_l, rotation=30, ha="right", color=PALETTE["text"], fontsize=9)
ax.set_title(f"LIME Waterfall — Local Explanation (Sample 0) | Base={base_val:.3f}", fontsize=12, pad=10)
ax.set_ylabel("Cumulative Effect on P(ASD)")
ax.grid(axis="y", alpha=0.1, color=PALETTE["muted"])

pos_patch = mpatches.Patch(color=PALETTE["accent4"], label="Increases P(ASD)")
neg_patch = mpatches.Patch(color=PALETTE["accent3"], label="Decreases P(ASD)")
ax.legend(handles=[pos_patch, neg_patch], facecolor=PALETTE["panel"], labelcolor=PALETTE["text"])
fig.tight_layout()
fig.savefig("outputs/07_lime_waterfall.png", dpi=130, bbox_inches="tight", facecolor=PALETTE["bg"])
plt.close(); print("[✓] Saved 07_lime_waterfall.png")

# ── Fig 8: Quantum Circuit Architecture ──────────────────────
fig, ax = plt.subplots(figsize=(12, 5), facecolor=PALETTE["bg"])
ax.set_facecolor(PALETTE["panel"])
ax.set_xlim(0, 10); ax.set_ylim(-0.5, N_QUBITS-0.5)
ax.axis("off")
ax.set_title(f"VQC Architecture — {N_QUBITS} Qubits, {N_LAYERS} Layers, {vqc.weights.size} Parameters",
             color=PALETTE["text"], fontsize=13, pad=12)

# Wire lines
for q in range(N_QUBITS):
    ax.axhline(q, color=PALETTE["muted"], lw=1.5, alpha=0.5, xmin=0.05, xmax=0.95)
    ax.text(-0.3, q, f"|0⟩ q{q}", color=PALETTE["text"], va="center", fontsize=10, ha="right",
            transform=ax.transData)

# Encoding gates
for q in range(N_QUBITS):
    rect = plt.Rectangle((0.5, q-0.3), 0.8, 0.6,
                          color=PALETTE["accent2"], alpha=0.85, zorder=3)
    ax.add_patch(rect)
    ax.text(0.9, q, f"Ry(x{q})", color="#000", fontsize=8, ha="center", va="center",
            zorder=4, fontweight="bold")

# Ansatz layers
layer_x_start = 1.8
for l in range(N_LAYERS):
    lx = layer_x_start + l * 3.5

    # Rotation block
    for q in range(N_QUBITS):
        rect = plt.Rectangle((lx, q-0.28), 1.2, 0.56,
                              color=PALETTE["accent1"], alpha=0.85, zorder=3)
        ax.add_patch(rect)
        ax.text(lx+0.6, q, "Rz·Ry·Rz", color="white", fontsize=7.5,
                ha="center", va="center", zorder=4)

    # CNOT entanglement (ring)
    for q in range(N_QUBITS):
        tgt = (q+1) % N_QUBITS
        cx = lx + 1.5
        ax.annotate("", xy=(cx, tgt), xytext=(cx, q),
                    arrowprops=dict(arrowstyle="->", color=PALETTE["accent3"],
                                   lw=1.5, connectionstyle="arc3,rad=0.2"))
        ax.plot(cx, q, "o", color=PALETTE["accent3"], ms=6, zorder=5)
        ax.text(cx+0.05, (q+tgt)/2, "⊕", color=PALETTE["accent3"], fontsize=10,
                ha="left", va="center", zorder=6)

    # Layer label
    ax.text(lx + 1.2, N_QUBITS - 0.1,
            f"Layer {l+1}", color=PALETTE["accent1"], fontsize=9, ha="center")

# Measurement
mx = layer_x_start + N_LAYERS * 3.5 + 0.2
rect = plt.Rectangle((mx, -0.3), 0.9, 0.6,
                      color=PALETTE["accent4"], alpha=0.9, zorder=3)
ax.add_patch(rect)
ax.text(mx+0.45, 0, "⟨Z₀⟩", color="#000", fontsize=9, ha="center", va="center",
        zorder=4, fontweight="bold")
ax.text(mx+0.45, -0.48, "output", color=PALETTE["muted"], fontsize=7.5, ha="center")

fig.tight_layout()
fig.savefig("outputs/08_circuit_diagram.png", dpi=130, bbox_inches="tight", facecolor=PALETTE["bg"])
plt.close(); print("[✓] Saved 08_circuit_diagram.png")

# ── Fig 9: Full Summary Dashboard ───────────────────────────
fig = plt.figure(figsize=(22, 16), facecolor=PALETTE["bg"])
gs = GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38)

# Loss
ax = fig.add_subplot(gs[0,0]); style_ax(ax)
ax.plot(loss_history, color=PALETTE["accent1"], lw=2)
ax.fill_between(range(len(loss_history)), loss_history, alpha=0.15, color=PALETTE["accent1"])
ax.set_title("Training Loss", color=PALETTE["text"], fontsize=11)
ax.set_xlabel("Epoch"); ax.grid(alpha=0.1)

# Confusion
ax = fig.add_subplot(gs[0,1]); style_ax(ax)
ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                color="white", fontsize=16, fontweight="bold")
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["Non-ASD","ASD"], color=PALETTE["text"])
ax.set_yticklabels(["Non-ASD","ASD"], color=PALETTE["text"])
ax.set_title("Confusion Matrix", color=PALETTE["text"], fontsize=11)

# Metrics panel
ax = fig.add_subplot(gs[0,2]); ax.set_facecolor(PALETTE["panel"]); ax.axis("off")
for sp in ax.spines.values(): sp.set_edgecolor(PALETTE["muted"])
ax.set_title("Model Metrics", color=PALETTE["text"], fontsize=11)
metrics = [("Accuracy", f"{acc*100:.1f}%", PALETTE["accent4"]),
           ("F1 Score", f"{f1:.3f}", PALETTE["accent2"]),
           ("Qubits",   f"{N_QUBITS}", PALETTE["accent1"]),
           ("Params",   f"{vqc.weights.size}", PALETTE["muted"])]
for i,(lbl,val,col) in enumerate(metrics):
    y = 0.83 - i*0.22
    ax.text(0.1, y, lbl, transform=ax.transAxes, color=PALETTE["muted"], fontsize=10)
    ax.text(0.1, y-0.09, val, transform=ax.transAxes, color=col, fontsize=20, fontweight="bold")

# SHAP bar
ax = fig.add_subplot(gs[1,:2]); style_ax(ax)
ax.barh(sorted_n, sorted_v, color=colors, height=0.62)
ax.set_title("SHAP Feature Importance", color=PALETTE["text"], fontsize=11)
ax.grid(axis="x", alpha=0.1)

# Prob distribution
ax = fig.add_subplot(gs[1,2]); style_ax(ax)
ax.hist(y_probs[y_test==0], bins=22, alpha=0.7, color=PALETTE["accent2"], label="Non-ASD", density=True)
ax.hist(y_probs[y_test==1], bins=22, alpha=0.7, color=PALETTE["accent3"], label="ASD", density=True)
ax.axvline(0.5, color="white", lw=1.5, linestyle="--")
ax.set_title("Quantum Prob. Dist.", color=PALETTE["text"], fontsize=11)
ax.legend(facecolor=PALETTE["panel"], labelcolor=PALETTE["text"], fontsize=8)
ax.grid(alpha=0.1)

# Beeswarm
ax = fig.add_subplot(gs[2,:2]); style_ax(ax)
for rank, fi in enumerate(top_idx):
    sv = shap_vals[:,fi]
    fv = X_explain[:,fi] if fi < X_explain.shape[1] else np.zeros(len(sv))
    fv_n = (fv-fv.min())/((fv.max()-fv.min())+1e-9)
    j = np.random.uniform(-0.2, 0.2, len(sv))
    ax.scatter(sv, np.full(len(sv), rank)+j, c=fv_n, cmap="plasma", s=18, alpha=0.7)
ax.set_yticks(range(n_feat_show))
ax.set_yticklabels([FEAT_NAMES[i] for i in top_idx], color=PALETTE["text"], fontsize=8)
ax.axvline(0, color=PALETTE["muted"], lw=1, linestyle="--")
ax.set_title("SHAP Beeswarm", color=PALETTE["text"], fontsize=11)
ax.grid(axis="x", alpha=0.1)

# Sensitivity bar
ax = fig.add_subplot(gs[2,2]); style_ax(ax)
n_s = min(len(sensitivities), len(FEAT_NAMES))
ax.barh(FEAT_NAMES[:n_s], sensitivities[:n_s], color=PALETTE["accent4"], height=0.62)
ax.set_title("Sensitivity Analysis", color=PALETTE["text"], fontsize=11)
ax.grid(axis="x", alpha=0.1)

fig.text(0.5, 0.98, "From Black Box to Glass Box — Explainable QML for ASD Analysis",
         ha="center", va="top", fontsize=16, color=PALETTE["text"], fontweight="bold")

fig.savefig("outputs/09_full_dashboard.png", dpi=130, bbox_inches="tight", facecolor=PALETTE["bg"])
plt.close(); print("[✓] Saved 09_full_dashboard.png")

print(f"\n{'='*55}")
print(f"  Pipeline Complete!")
print(f"  Accuracy : {acc*100:.1f}%   F1 : {f1:.3f}")
print(f"  All 9 plots saved to outputs/")
print(f"{'='*55}")

# Save results summary
summary = {
    "accuracy": acc, "f1": f1,
    "loss_history": loss_history,
    "shap_mean_abs": mean_abs[:n_f].tolist(),
    "feature_names": FEAT_NAMES,
    "sensitivities": sensitivities.tolist(),
}
import json
with open("outputs/results_summary.json","w") as fp:
    json.dump(summary, fp, indent=2)
print("[✓] Saved results_summary.json")
