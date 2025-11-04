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
OUT_PNG  = Path("irf_by_regime_with_IS_range.png")
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
# Build table + IRFs
# ----------------------------
regimes = ["LS", "IS", "HS"]
rows = []
irf_pct_dict = {}
cum_pct_dict = {}

for reg in regimes:
    gbar = mean_G.get(reg, np.nan)
    sigma = sigma_dFSI.get(reg, np.nan)

    beta = bL + bN * gbar
    phi  = aL + aN * gbar

    irf_log = ar1_irf(beta, phi, sigma, H)
    cum_log = np.cumsum(irf_log)
    irf_pct = log_to_pct(irf_log)
    cum_pct = log_to_pct(cum_log)

    irf_pct_dict[reg] = irf_pct
    cum_pct_dict[reg] = cum_pct

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
    ax1.plot(quarters, irf_pct_dict[reg], lw=2, label=reg, color=colors[reg])
# IS range lines
ax1.plot(quarters, irf_pct_dict["IS_min"], ls="--", lw=1.5, color=colors["IS"], label="IS min G")
ax1.plot(quarters, irf_pct_dict["IS_median"], ls=":", lw=2, color=colors["IS"], label="IS median G")
ax1.plot(quarters, irf_pct_dict["IS_max"], ls="-.", lw=1.5, color=colors["IS"], label="IS max G")

ax1.axhline(0, color="black", lw=0.8)
ax1.set_title("IRF of 1σ D(FSI) shock (quarterly, %)")
ax1.set_xlabel("Quarters after shock")
ax1.set_ylabel("Percent change (quarter)")
ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax1.grid(True, alpha=0.25)
ax1.legend(title="Regime", frameon=True)

# --- Cumulative ---
for reg in regimes:
    ax2.plot(quarters, cum_pct_dict[reg], lw=2, label=reg, color=colors[reg])
ax2.plot(quarters, cum_pct_dict["IS_min"], ls="--", lw=1.5, color=colors["IS"], label="IS min G")
ax2.plot(quarters, cum_pct_dict["IS_median"], ls=":", lw=2, color=colors["IS"], label="IS median G")
ax2.plot(quarters, cum_pct_dict["IS_max"], ls="-.", lw=1.5, color=colors["IS"], label="IS max G")

ax2.axhline(0, color="black", lw=0.8)
for q, lab in [(H1, "1y"), (H5, "5y")]:
    ax2.axvline(q, color="gray", lw=0.8, ls="--")
    ytop = ax2.get_ylim()[1]
    ax2.text(q, ytop - 0.05*(ytop - ax2.get_ylim()[0]), lab, ha="center", va="top", fontsize=9, color="gray")

ax2.set_title("Cumulative effect (%, exact exp transform)")
ax2.set_xlabel("Quarters after shock")
ax2.set_ylabel("Percent change (level)")
ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax2.grid(True, alpha=0.25)
ax2.legend(title="Regime / IS variants", frameon=True)

plt.savefig(OUT_PNG, dpi=180)
print(f"\nSaved plot → {OUT_PNG.resolve()}")