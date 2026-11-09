# -*- coding: utf-8 -*-
# Joint confusion matrix + dual ROC (shock & recovery) using D(FSI(-2)) to predict GDP direction
# Requires: pandas, numpy, matplotlib, scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, classification_report

# -------------------------------------------------
# Load data
# -------------------------------------------------
df = pd.read_csv("fsi_rgdp_trf.csv")

# Ensure numeric
for c in ["FSI", "RGDP_SA3"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# -------------------------------------------------
# Differences and lags
# -------------------------------------------------
df["dFSI"] = df["FSI"].diff()
df["dRGDP"] = df["RGDP_SA3"].diff()

df["dFSI_l1"] = df["dFSI"].shift(1)
df["dFSI_l2"] = df["dFSI"].shift(2)  # predictor we test
df["dRGDP_l1"] = df["dRGDP"].shift(1)

# Rows needed for confusion matrix (need predictor lag + actual)
mask_cm = df[["dFSI_l2", "dRGDP", "dRGDP_l1"]].notna().all(axis=1)  # CHANGED: uses dFSI_l2
df_cm = df.loc[mask_cm].copy()

# Rows needed for ROC (need predictor lag + actual)
mask_roc = df[["dFSI_l2", "dRGDP"]].notna().all(axis=1)  # CHANGED: uses dFSI_l2
df_roc = df.loc[mask_roc].copy()

# -------------------------------------------------
# JOINT confusion matrix:
#   Actual "Shock"  (1) = dRGDP < 0
#   Actual "Recover"(0) = dRGDP > 0
#   Pred "Shock"    (1) = dFSI(-2) > 0              # CHANGED
#   Pred "Recover"  (0) = dFSI(-2) < 0              # CHANGED
#   Neutral (dFSI(-2)==0 or dRGDP==0) rows dropped
# -------------------------------------------------
y_actual = np.where(df_cm["dRGDP"] < 0, 1,
            np.where(df_cm["dRGDP"] > 0, 0, np.nan))
valid_actual = ~np.isnan(y_actual)
df_cm = df_cm.loc[valid_actual].copy()
y_actual = y_actual[valid_actual].astype(int)

pred_shock = df_cm["dFSI_l2"] > 0      # CHANGED
pred_recover = df_cm["dFSI_l2"] < 0     # CHANGED

y_pred = np.where(pred_shock, 1,
          np.where(pred_recover, 0, np.nan))

valid_pred = ~np.isnan(y_pred)
y_pred = y_pred[valid_pred].astype(int)
y_actual_cm = y_actual[valid_pred]

cm = confusion_matrix(y_actual_cm, y_pred, labels=[0, 1])
print("Joint Confusion Matrix (Actual GDP sign vs Predicted by D(FSI(-2)) sign):")
cm_df = pd.DataFrame(cm,
                     index=["Actual_Recover(Up)", "Actual_Shock(Down)"],
                     columns=["Pred_Recover(FSI↓, -2)", "Pred_Shock(FSI↑, -2)"])
print(cm_df)

print("\nClassification report (Shock as positive class = 1):")
print(classification_report(y_actual_cm, y_pred,
                            target_names=["Recover(Up)","Shock(Down)"],
                            digits=3))

# -------------------------------------------------
# DUAL ROC curves (using dFSI(-2) as continuous score)
#   Shock ROC:    y_shock = 1 if dRGDP<0, score = dFSI(-2)
#   Recovery ROC: y_rec   = 1 if dRGDP>0, score = -dFSI(-2)
# -------------------------------------------------
y_shock = (df_roc["dRGDP"] < 0).astype(int).values
y_recover = (df_roc["dRGDP"] > 0).astype(int).values
score_shock = df_roc["dFSI_l2"].values           # CHANGED
score_recover = (-df_roc["dFSI_l2"]).values      # CHANGED

fpr_s, tpr_s, thr_s = roc_curve(y_shock, score_shock)
auc_s = auc(fpr_s, tpr_s)

fpr_r, tpr_r, thr_r = roc_curve(y_recover, score_recover)
auc_r = auc(fpr_r, tpr_r)

# -------------------------------------------------
# Plot ROC + Joint Confusion Matrix
# -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ROC panel with both curves
ax = axes[0]
ax.plot(fpr_s, tpr_s, lw=2, color="crimson", label=f"Shock ROC (AUC={auc_s:.3f})")
ax.plot(fpr_r, tpr_r, lw=2, color="seagreen", label=f"Recovery ROC (AUC={auc_r:.3f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.6)
ax.set_title("ROC — GDP Shock and Recovery vs D(FSI(-2))")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()

# Joint confusion matrix
ConfusionMatrixDisplay(confusion_matrix=cm,
                       display_labels=["Recover","Shock"]).plot(
    ax=axes[1], cmap="Blues", colorbar=False
)
axes[1].set_title("Joint Confusion Matrix\nActual GDP sign vs Pred by D(FSI(-2)) sign")

plt.tight_layout()
plt.show()