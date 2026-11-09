# -*- coding: utf-8 -*-
# Joint confusion matrix + dual ROC (shock & recovery) using D(FSI(-k)) to predict GDP direction
# Set PRED_LAG = 1 or 2
# TURNING_RULE = True -> actual turning points require sign change in GDP:
#   Shock:    dRGDP < 0 and dRGDP(-1) >= 0
#   Recovery: dRGDP > 0 and dRGDP(-1) <= 0
# TURNING_RULE = False -> simple direction:
#   Shock:    dRGDP < 0
#   Recovery: dRGDP > 0
#
# Requires: pandas, numpy, matplotlib, scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, classification_report

# ============================
# Config
# ============================
CSV_PATH     = "fsi_rgdp_trf.csv"
PRED_LAG     = 2         # choose 1 or 2
TURNING_RULE = False       # True = turning points, False = simple sign

# ============================
# Load and prep
# ============================
df = pd.read_csv(CSV_PATH)
for c in ["FSI", "RGDP_SA3"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["dFSI"]  = df["FSI"].diff()
df["dRGDP"] = df["RGDP_SA3"].diff()

# Lags we may need
df["dFSI_l1"]  = df["dFSI"].shift(1)
df["dFSI_l2"]  = df["dFSI"].shift(2)
df["dRGDP_l1"] = df["dRGDP"].shift(1)

# Select predictor lag
if PRED_LAG == 1:
    pred_series = df["dFSI_l1"]
    pred_label  = "D(FSI(-1))"
elif PRED_LAG == 2:
    pred_series = df["dFSI_l2"]
    pred_label  = "D(FSI(-2))"
else:
    raise ValueError("PRED_LAG must be 1 or 2")

# ============================
# Define masks for CM and ROC
# ============================
# For CM, we need predictor and actual labels; if TURNING_RULE, we also need dRGDP_l1
need_cols_cm = ["dRGDP", "dRGDP_l1"] if TURNING_RULE else ["dRGDP"]
need_cols_cm = need_cols_cm + [pred_series.name]
mask_cm = df[need_cols_cm].notna().all(axis=1)
df_cm = df.loc[mask_cm].copy()

# For ROC, same requirement as CM so AUC matches the same sample/definition
mask_roc = mask_cm
df_roc = df.loc[mask_roc].copy()

# ============================
# Actual labels (ground truth)
# ============================
if TURNING_RULE:
    # true turning points
    actual_shock   = (df_cm["dRGDP"] < 0) & (df_cm["dRGDP_l1"] >= 0)
    actual_recover = (df_cm["dRGDP"] > 0) & (df_cm["dRGDP_l1"] <= 0)
else:
    # simple sign
    actual_shock   = (df_cm["dRGDP"] < 0)
    actual_recover = (df_cm["dRGDP"] > 0)

# Joint 2-class truth: Shock=1, Recover=0; drop zeros of dRGDP (no move)
y_actual = np.where(actual_shock, 1,
             np.where(actual_recover, 0, np.nan))
valid_actual = ~np.isnan(y_actual)
df_cm = df_cm.loc[valid_actual].copy()
y_actual = y_actual[valid_actual].astype(int)

# ============================
# Predicted labels from D(FSI(-k)) sign
# ============================
pred_vals = pred_series.loc[df_cm.index]
pred_shock   = pred_vals > 0
pred_recover = pred_vals < 0

y_pred = np.where(pred_shock, 1,
          np.where(pred_recover, 0, np.nan))

valid_pred = ~np.isnan(y_pred)
y_pred = y_pred[valid_pred].astype(int)
y_actual_cm = y_actual[valid_pred]

# ============================
# Confusion matrix & report
# ============================
cm = confusion_matrix(y_actual_cm, y_pred, labels=[0, 1])
print(f"Joint Confusion Matrix (Actual GDP turning vs Pred by {pred_label} sign):")
cm_df = pd.DataFrame(
    cm,
    index=["Actual_Recover", "Actual_Shock"],
    columns=["Pred_Recover", "Pred_Shock"]
)
print(cm_df)

print("\nClassification report (Shock=1):")
print(classification_report(
    y_actual_cm, y_pred,
    target_names=["Recover","Shock"],
    digits=3
))

# ============================
# ROC curves (two one-vs-rest tasks)
# ============================
# Build actual labels for ROC on the same sample/definition
if TURNING_RULE:
    y_shock   = ((df_roc["dRGDP"] < 0) & (df_roc["dRGDP_l1"] >= 0)).astype(int).values
    y_recover = ((df_roc["dRGDP"] > 0) & (df_roc["dRGDP_l1"] <= 0)).astype(int).values
else:
    y_shock   = (df_roc["dRGDP"] < 0).astype(int).values
    y_recover = (df_roc["dRGDP"] > 0).astype(int).values

score_shock   = pred_series.loc[df_roc.index].values          # higher => shock
score_recover = (-pred_series.loc[df_roc.index].values)       # higher => recovery

fpr_s, tpr_s, thr_s = roc_curve(y_shock, score_shock)
auc_s = auc(fpr_s, tpr_s)

fpr_r, tpr_r, thr_r = roc_curve(y_recover, score_recover)
auc_r = auc(fpr_r, tpr_r)

# ============================
# Plot ROC + joint CM
# ============================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ROC panel with both curves
ax = axes[0]
ax.plot(fpr_s, tpr_s, lw=2, color="crimson", label=f"Shock ROC (AUC={auc_s:.3f})")
ax.plot(fpr_r, tpr_r, lw=2, color="seagreen", label=f"Recovery ROC (AUC={auc_r:.3f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.6)
title_suffix = "turning points" if TURNING_RULE else "simple sign"
ax.set_title(f"ROC — GDP Shock/Recovery vs {pred_label} ({title_suffix})")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()

# Joint confusion matrix
ConfusionMatrixDisplay(confusion_matrix=cm,
                       display_labels=["Recover","Shock"]).plot(
    ax=axes[1], cmap="Blues", colorbar=False
)
axes[1].set_title(f"Joint Confusion Matrix\nActual vs Predicted by {pred_label} sign\n({title_suffix})")

plt.tight_layout()

# save as jpg
plt.savefig("roc_joint_cm.jpg", dpi=300, format="jpeg")

plt.show()