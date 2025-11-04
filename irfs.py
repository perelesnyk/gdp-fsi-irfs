# -*- coding: utf-8 -*-
# IRF for STR model by regimes (LS/IS/HS) with:
#  - Regimes DEFINED BY G (transition index)
#  - Shock sizes σ estimated FROM FSI BANDS
# Requirements: pandas, numpy, matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ----------------------------
# Inputs & settings
# ----------------------------
CSV_PATH = "rgdp_str_table.csv"    # <- your data table
OUT_PNG  = Path("irf_regime_byG_sigmas_byFSI.png")

# Regime cutoffs
#   G cuts define regimes (used for mean_G and the STR coefficients)
G_CUT_LOW  = 0.3422      # LS / IS
G_CUT_HIGH = 0.9655      # IS / HS
#   FSI cuts define bands for shock-size estimation only
FSI_CUT_LOW  = 0.1187    # LS / IS
FSI_CUT_HIGH = 0.1704    # IS / HS

H1 = 4      # 1-year horizon
H5 = 20     # 5-year horizon
H  = 20     # plot horizon (quarters)

# STR coefficients (from your printout)
# dlog y_t = (cL + bL*dFSI_{t-1} + aL*dlogy_{t-1})
#          + (cN + bN*dFSI_{t-1} + aN*dlogy_{t-1}) * G
# (dummies set to zero for IRF)
cL, bL, aL = 0.008409,  -0.240669, -0.290350
cN, bN, aN = -0.008686,  0.221148,  0.848650

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

def regime_from_G(g):
    if pd.isna(g):
        return "NA"
    if g < G_CUT_LOW:
        return "LS"
    elif g < G_CUT_HIGH:
        return "IS"
    else:
        return "HS"

def band_from_FSI(fsi):
    if pd.isna(fsi):
        return "NA"
    if fsi < FSI_CUT_LOW:
        return "LS"
    elif fsi < FSI_CUT_HIGH:
        return "IS"
    else:
        return "HS"

def ar1_irf(beta, phi, sigma, H):
    """
    One-time shock to dFSI at t=0 -> first impact appears at t=1 (because regressor is D(FSI(-1))).
    Returns array of length H+1 with IRF for t=0..H in LOG POINTS.
    """
    irf = np.zeros(H + 1)
    if sigma is None or np.isnan(sigma) or sigma == 0:
        return irf
    # first non-zero impact at t=1
    irf[1] = beta * sigma
    for h in range(2, H + 1):
        irf[h] = irf[h-1] * phi
    return irf

def log_to_percent(log_series):
    """Convert log change to percent change exactly: 100*(exp(x)-1)."""
    return 100.0 * (np.exp(log_series) - 1.0)

# ----------------------------
# Load & prep
# ----------------------------
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]

# coerce numbers (commas -> dots)
for k in ["g", "fsi", "dfsi"]:
    if COLS[k] in df.columns:
        df[COLS[k]] = (
            df[COLS[k]].astype(str).str.replace(",", ".", regex=False)
                        .replace({"": np.nan})
                        .astype(float)
        )

df["__date"] = df[COLS["date"]].apply(parse_date_quarter)
df = df.sort_values("__date").reset_index(drop=True)

# Regimes by G (for mean_G and coefficients)
df["regime_G"] = df[COLS["g"]].apply(regime_from_G)
# Bands by FSI (for σ estimation)
df["band_FSI"] = df[COLS["fsi"]].apply(band_from_FSI)

# ----------------------------
# 1) Regime stats (by G): mean_G (and counts)
# ----------------------------
meanG = (
    df.loc[df["regime_G"].isin(["LS","IS","HS"]), [COLS["g"], "regime_G"]]
      .groupby("regime_G")[COLS["g"]]
      .mean()
      .rename("mean_G")
)

n_by_G = (
    df.loc[df["regime_G"].isin(["LS","IS","HS"]), "regime_G"]
      .value_counts()
      .reindex(["LS","IS","HS"])
      .rename("n_G")
)

# ----------------------------
# 2) Shock-size stats (by FSI band): σ(dFSI) and σ(FSI)
# ----------------------------
sigma_dFSI = (
    df.loc[df["band_FSI"].isin(["LS","IS","HS"]), [COLS["dfsi"], "band_FSI"]]
      .groupby("band_FSI")[COLS["dfsi"]]
      .std(ddof=1)
      .rename("std_dFSI")
)

sigma_FSI = (
    df.loc[df["band_FSI"].isin(["LS","IS","HS"]), [COLS["fsi"], "band_FSI"]]
      .groupby("band_FSI")[COLS["fsi"]]
      .std(ddof=1)
      .rename("std_FSI")
)

n_by_FSI = (
    df.loc[df["band_FSI"].isin(["LS","IS","HS"]), "band_FSI"]
      .value_counts()
      .reindex(["LS","IS","HS"])
      .rename("n_FSI")
)

# Combine into a single table (note: mean_G from G-regimes, stds from FSI-bands)
stats_df = pd.concat([meanG, sigma_FSI, sigma_dFSI, n_by_G, n_by_FSI], axis=1).reindex(["LS","IS","HS"])
print("\nRegime statistics (G-defined mean_G; FSI-band stds):")
print(stats_df.round(6).fillna(""))

# ----------------------------
# 3) IRFs & cumulative (log space -> percent)
# ----------------------------
results = {}
for reg in ["LS", "IS", "HS"]:
    gbar = stats_df.loc[reg, "mean_G"]
    s_d  = stats_df.loc[reg, "std_dFSI"]

    # if either is missing, skip gracefully
    if pd.isna(gbar) or pd.isna(s_d):
        results[reg] = None
        continue

    # regime-specific coefficients
    beta = bL + bN * gbar         # elasticity on D(FSI-1)
    phi  = aL + aN * gbar         # AR(1) on DLOG(-1)

    irf_log = ar1_irf(beta, phi, s_d, H)      # in log points
    cum_log = np.cumsum(irf_log)              # cumulative log change

    irf_pct = log_to_percent(irf_log)         # %
    cum_pct = log_to_percent(cum_log)         # %

    results[reg] = {
        "gbar": gbar, "sigma_dFSI": s_d, "beta": beta, "phi": phi,
        "irf_log": irf_log, "cum_log": cum_log,
        "irf_pct": irf_pct, "cum_pct": cum_pct
    }

# Report 1y & 5y cumulative effects
def line(reg):
    r = results[reg]
    if r is None:
        return {"G_mean": np.nan, "sigma_dFSI": np.nan, "beta": np.nan, "phi": np.nan,
                "cum_4q_%": np.nan, "cum_20q_%": np.nan}
    return {
        "G_mean": r["gbar"],
        "sigma_dFSI": r["sigma_dFSI"],
        "beta": r["beta"], "phi": r["phi"],
        "cum_4q_%": r["cum_pct"][H1],
        "cum_20q_%": r["cum_pct"][H5]
    }

report = pd.DataFrame({reg: line(reg) for reg in ["LS", "IS", "HS"]}).T
print("\nCumulative effect of a 1σ D(FSI) shock (dummies=0), percent change in RGDP level:")
print(report[["G_mean", "sigma_dFSI", "beta", "phi", "cum_4q_%", "cum_20q_%"]].round(4))

# ----------------------------
# 4) Plot: IRF & Cumulative side-by-side
# ----------------------------
quarters = np.arange(H + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

ax1, ax2 = axes
colors = {"LS": "#2E7D32", "IS": "#F9A825", "HS": "#C62828"}

# IRF (quarter-by-quarter)
for reg in ["LS", "IS", "HS"]:
    if results[reg] is None: 
        continue
    ax1.plot(quarters, results[reg]["irf_pct"], label=f"{reg}", lw=2, color=colors[reg])
ax1.axhline(0, color="black", lw=0.8)
ax1.set_title("IRF of 1σ D(FSI) shock (quarterly, %)")
ax1.set_xlabel("Quarters after shock")
ax1.set_ylabel("Percent change (quarter)")
ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax1.grid(True, alpha=0.25)
ax1.legend(title="Regime", frameon=True)

# Cumulative
for reg in ["LS", "IS", "HS"]:
    if results[reg] is None: 
        continue
    ax2.plot(quarters, results[reg]["cum_pct"], label=f"{reg}", lw=2, color=colors[reg])
ax2.axhline(0, color="black", lw=0.8)
# markers at 4q and 20q
for q, lab in [(H1, "1y"), (H5, "5y")]:
    ax2.axvline(q, color="gray", lw=0.8, ls="--")
    ax2.text(q, ax2.get_ylim()[1]*0.95 if ax2.get_ylim()[1]>0 else 0.95*max(0.1, ax2.get_ylim()[1]),
             lab, ha="center", va="top", fontsize=9, color="gray")
ax2.set_title("Cumulative effect (%, exact exp transform)")
ax2.set_xlabel("Quarters after shock")
ax2.set_ylabel("Percent change (level)")
ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax2.grid(True, alpha=0.25)
ax2.legend(title="Regime", frameon=True)

plt.savefig(OUT_PNG, dpi=180)
print(f"\nSaved plot → {OUT_PNG.resolve()}")