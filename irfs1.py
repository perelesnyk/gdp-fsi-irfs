# -*- coding: utf-8 -*-
# Extended version: includes IS regime min/median/max G analysis
# Requirements: pandas, numpy, matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ----------------------------
# Inputs
# ----------------------------
CSV_PATH = "rgdp_str_table.csv"
OUT_JPG  = Path("irf_by_regime_with_IS_range.jpg")
H  = 20
H1 = 4
H5 = 20

# Regime cutoffs
G_CUT_LOW  = 0.3422
G_CUT_HIGH = 0.9655
FSI_CUT_LOW  = 0.1187
FSI_CUT_HIGH = 0.1704

# STR coefficients
cL, bL, aL = 0.008409,  -0.240669, -0.290350
cN, bN, aN = -0.008686,  0.221148,  0.848650

# HAC covariance matrix for [cL, bL, aL, cN, bN, aN]
HAC_matrix = np.array([
    [8.507526398719898e-06, -2.481778447058299e-06, 7.966160692733066e-06,
     -7.867716334907634e-06, 1.209793606002429e-07, -6.058187308713758e-05],
    [-2.481778447058299e-06, 0.0002222572618298202, -5.083520951131941e-05,
     1.030834129704509e-05, -0.0002267137873032802, -0.0002898724110253803],
    [7.966160692733066e-06, -5.083520951131941e-05, 0.0004538704845508114,
     -1.539527789330668e-05, 8.75283613440443e-05, 0.0002569638379923147],
    [-7.867716334907634e-06, 1.030834129704509e-05, -1.539527789330668e-05,
     1.029641781553392e-05, -7.276883740540779e-06, -1.977877749943097e-05],
    [1.209793606002429e-07, -0.0002267137873032802, 8.75283613440443e-05,
     -7.276883740540779e-06, 0.000259886179521454, 0.0003731099645247996],
    [-6.058187308713758e-05, -0.0002898724110253803, 0.0002569638379923147,
     -1.977877749943097e-05, 0.0003731099645247996, 0.003484345146219229]
])

# ----------------------------
# Column mapping
# ----------------------------
COLS = {
    "date": "dates",
    "g":    "G",
    "fsi":  "FSI",
    "dfsi": "D(FSI)",
}

# ----------------------------
# Helpers
# ----------------------------
def parse_date_quarter(x: str):
    x = str(x).strip()
    if "Q" in x.upper():
        return pd.Period(x, freq="Q").asfreq("Q", "end").to_timestamp()
    return pd.to_datetime(x, errors="coerce")

def regime_from_g(g):
    if pd.isna(g): return "NA"
    if g < G_CUT_LOW:   return "LS"
    elif g < G_CUT_HIGH:return "IS"
    else:               return "HS"

def band_from_fsi_level(f):
    if pd.isna(f): return "NA"
    if f <= FSI_CUT_LOW:        return "LS"
    elif f <= FSI_CUT_HIGH:     return "IS"
    else:                       return "HS"

def ar1_irf(beta, phi, sigma, H):
    irf = np.zeros(H+1)
    if not (np.isfinite(sigma) and np.isfinite(beta) and np.isfinite(phi)):
        return irf
    irf[1] = beta * sigma
    for h in range(2, H+1):
        irf[h] = irf[h-1] * phi
    return irf

def compute_irf_with_ci(G, sigma, H, z_68=1.0, z_95=1.96):
    """
    Compute IRF with 68% and 95% confidence intervals using HAC variance.
    Returns: irf, irf_lower_68, irf_upper_68, irf_lower_95, irf_upper_95
    """
    beta = bL + bN * G
    phi = aL + aN * G
    
    # Jacobian vectors w.r.t. [cL, bL, aL, cN, bN, aN]
    J_beta = np.array([0, 1, 0, 0, G, 0])
    J_phi = np.array([0, 0, 1, 0, 0, G])
    
    # Variance via delta method
    var_beta = J_beta @ HAC_matrix @ J_beta
    var_phi = J_phi @ HAC_matrix @ J_phi
    
    se_beta = np.sqrt(var_beta)
    se_phi = np.sqrt(var_phi)
    
    # Point estimate
    irf = ar1_irf(beta, phi, sigma, H)
    
    # Variance propagation
    irf_var = np.zeros(H+1)
    if np.isfinite(sigma) and np.isfinite(beta) and np.isfinite(phi):
        irf_var[1] = (se_beta * sigma)**2
        for h in range(2, H+1):
            irf_var[h] = (phi**2) * irf_var[h-1] + (irf[h-1]**2) * (se_phi**2)
    
    # Confidence bands
    irf_se = np.sqrt(irf_var)
    irf_lower_68 = irf - z_68 * irf_se
    irf_upper_68 = irf + z_68 * irf_se
    irf_lower_95 = irf - z_95 * irf_se
    irf_upper_95 = irf + z_95 * irf_se
    
    return irf, irf_lower_68, irf_upper_68, irf_lower_95, irf_upper_95

def log_to_pct(x):
    return 100.0 * (np.exp(x) - 1.0)

# ----------------------------
# Load & prep
# ----------------------------
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]
for k in ["g", "fsi", "dfsi"]:
    if COLS[k] in df.columns:
        df[COLS[k]] = (
            df[COLS[k]]
            .astype(str).str.replace(",", ".", regex=False)
            .replace({"": np.nan})
            .astype(float)
        )

df["__date"] = df[COLS["date"]].apply(parse_date_quarter)
df = df.sort_values("__date").reset_index(drop=True)
df["regime_G"] = df[COLS["g"]].apply(regime_from_g)
df["band_FSI"] = df[COLS["fsi"]].apply(band_from_fsi_level)

# ----------------------------
# Regime stats
# ----------------------------
mean_G      = df.groupby("regime_G")[COLS["g"]].mean()
sigma_dFSI  = df.groupby("band_FSI")[COLS["dfsi"]].std(ddof=1)

# Additional IS distribution stats
IS_values = df.loc[df["regime_G"] == "IS", COLS["g"]].dropna()
G_IS_mean   = IS_values.mean()
G_IS_median = IS_values.median()
G_IS_min    = IS_values.min()
G_IS_max    = IS_values.max()

# ----------------------------
# Build table + IRFs with confidence intervals
# ----------------------------
regimes = ["LS", "IS", "HS"]
rows = []
irf_pct_dict = {}
cum_pct_dict = {}
irf_ci_dict = {}  # Store confidence intervals
cum_ci_dict = {}

for reg in regimes:
    gbar = mean_G.get(reg, np.nan)
    sigma = sigma_dFSI.get(reg, np.nan)

    beta = bL + bN * gbar
    phi  = aL + aN * gbar

    # Compute IRF with confidence intervals
    irf_log, irf_l68, irf_u68, irf_l95, irf_u95 = compute_irf_with_ci(gbar, sigma, H)
    cum_log = np.cumsum(irf_log)
    cum_l68 = np.cumsum(irf_l68)
    cum_u68 = np.cumsum(irf_u68)
    cum_l95 = np.cumsum(irf_l95)
    cum_u95 = np.cumsum(irf_u95)
    
    irf_pct = log_to_pct(irf_log)
    cum_pct = log_to_pct(cum_log)

    irf_pct_dict[reg] = irf_pct
    cum_pct_dict[reg] = cum_pct
    irf_ci_dict[reg] = {
        'lower_68': log_to_pct(irf_l68), 'upper_68': log_to_pct(irf_u68),
        'lower_95': log_to_pct(irf_l95), 'upper_95': log_to_pct(irf_u95)
    }
    cum_ci_dict[reg] = {
        'lower_68': log_to_pct(cum_l68), 'upper_68': log_to_pct(cum_u68),
        'lower_95': log_to_pct(cum_l95), 'upper_95': log_to_pct(cum_u95)
    }

    rows.append({
        "Regime": reg,
        "G_mean": gbar,
        "σ_dFSI": sigma,
        "β": beta,
        "φ": phi,
        "cum_4q_%": cum_pct[H1],
        "cum_20q_%": cum_pct[H5]
    })

# Compute IRFs for IS min/median/max
IS_extra_stats = {"min": G_IS_min, "median": G_IS_median, "max": G_IS_max}
sigma_IS = sigma_dFSI.get("IS", np.nan)
for label, gval in IS_extra_stats.items():
    beta = bL + bN * gval
    phi  = aL + aN * gval
    irf_log = ar1_irf(beta, phi, sigma_IS, H)
    cum_log = np.cumsum(irf_log)
    irf_pct_dict[f"IS_{label}"] = log_to_pct(irf_log)
    cum_pct_dict[f"IS_{label}"] = log_to_pct(cum_log)

# Append stats to table
rows.append({
    "Regime": "IS_min/median/max",
    "G_min": G_IS_min,
    "G_median": G_IS_median,
    "G_max": G_IS_max,
    "σ_dFSI": sigma_IS
})

table = pd.DataFrame(rows).set_index("Regime")
print("\nExtended regime table (G-defined means; FSI-band σ):")
print(table.round(4).to_string())

# ----------------------------
# Plot (IRF left, Cumulative right)
# ----------------------------
quarters = np.arange(H+1)
colors = {"LS": "#2E7D32", "IS": "#F9A825", "HS": "#C62828"}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

# --- IRF ---
for reg in regimes:
    # Shaded 68% CI
    ax1.fill_between(quarters, irf_ci_dict[reg]['lower_68'], irf_ci_dict[reg]['upper_68'],
                     alpha=0.2, color=colors[reg])
    # Main line
    ax1.plot(quarters, irf_pct_dict[reg], lw=2, label=reg, color=colors[reg])
    # Dashed 95% CI
    ax1.plot(quarters, irf_ci_dict[reg]['lower_95'], ls='--', lw=1, color=colors[reg], alpha=0.7)
    ax1.plot(quarters, irf_ci_dict[reg]['upper_95'], ls='--', lw=1, color=colors[reg], alpha=0.7)
# IS range lines
# ax1.plot(quarters, irf_pct_dict["IS_min"], ls="--", lw=1.5, color=colors["IS"], label="IS min G")
# ax1.plot(quarters, irf_pct_dict["IS_median"], ls=":", lw=2, color=colors["IS"], label="IS median G")
# ax1.plot(quarters, irf_pct_dict["IS_max"], ls="-.", lw=1.5, color=colors["IS"], label="IS max G")

ax1.axhline(0, color="black", lw=0.8)
ax1.set_title("IRF of 1σ D(FSI) shock (quarterly, %) with 68%/95% CI")
ax1.set_xlabel("Quarters after shock")
ax1.set_ylabel("Percent change (quarter)")
ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax1.grid(True, alpha=0.25)
ax1.legend(title="Regime", frameon=True)

# --- Cumulative ---
for reg in regimes:
    # Shaded 68% CI
    ax2.fill_between(quarters, cum_ci_dict[reg]['lower_68'], cum_ci_dict[reg]['upper_68'],
                     alpha=0.2, color=colors[reg])
    # Main line
    ax2.plot(quarters, cum_pct_dict[reg], lw=2, label=reg, color=colors[reg])
    # Dashed 95% CI
    ax2.plot(quarters, cum_ci_dict[reg]['lower_95'], ls='--', lw=1, color=colors[reg], alpha=0.7)
    ax2.plot(quarters, cum_ci_dict[reg]['upper_95'], ls='--', lw=1, color=colors[reg], alpha=0.7)
# ax2.plot(quarters, cum_pct_dict["IS_min"], ls="--", lw=1.5, color=colors["IS"], label="IS min G")
# ax2.plot(quarters, cum_pct_dict["IS_median"], ls=":", lw=2, color=colors["IS"], label="IS median G")
# ax2.plot(quarters, cum_pct_dict["IS_max"], ls="-.", lw=1.5, color=colors["IS"], label="IS max G")

ax2.axhline(0, color="black", lw=0.8)
for q, lab in [(H1, "1y"), (H5, "5y")]:
    ax2.axvline(q, color="gray", lw=0.8, ls="--")
    ytop = ax2.get_ylim()[1]
    ax2.text(q, ytop - 0.05*(ytop - ax2.get_ylim()[0]), lab, ha="center", va="top", fontsize=9, color="gray")

ax2.set_title("Cumulative effect (%, exact exp transform) with 68%/95% CI")
ax2.set_xlabel("Quarters after shock")
ax2.set_ylabel("Percent change (level)")
ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax2.grid(True, alpha=0.25)
ax2.legend(title="Regime", frameon=True)

plt.savefig(OUT_JPG, dpi=300, format="jpeg")
print(f"\nSaved plot → {OUT_JPG.resolve()}")