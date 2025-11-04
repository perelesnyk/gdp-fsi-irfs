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
OUT_PNG  = Path("irf_regime_byG_sigmas_byFSI_trend.png")

# Regime cutoffs (G -> regimes; FSI -> bands for σ only)
G_CUT_LOW  = 0.3422      # LS / IS
G_CUT_HIGH = 0.9655      # IS / HS
FSI_CUT_LOW  = 0.1187    # LS / IS (FSI level)
FSI_CUT_HIGH = 0.1704    # IS / HS (FSI level)

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

def regime_from_g(g):
    if pd.isna(g): return "NA"
    if g < G_CUT_LOW:   return "LS"
    elif g < G_CUT_HIGH:return "IS"
    else:               return "HS"

def fsi_band_from_level(fsi):
    """FSI band for estimating σ (based on FSI level, not G)."""
    if pd.isna(fsi): return "NA"
    if fsi <= FSI_CUT_LOW:        return "LS"
    elif fsi <= FSI_CUT_HIGH:     return "IS"
    else:                         return "HS"

def exact_pct(x):
    """log-points -> exact percent: 100*(exp(x)-1)."""
    return 100.0 * (np.exp(x) - 1.0)

def simulate_paths(cG, phi, beta, sigma, H, y0_log=0.0):
    """
    Simulate baseline and shocked cumulative level indexes (100 at t=0).
    Shock is +σ to D(FSI) at t=0, impacting dlog y at t=1 via beta*D(FSI_{t-1}).
    Intercept cG is added each step.
    """
    dlogy_base  = np.zeros(H+1)
    dlogy_shock = np.zeros(H+1)
    ylog_base   = np.zeros(H+1)
    ylog_shock  = np.zeros(H+1)

    ylog_base[0] = y0_log
    ylog_shock[0]= y0_log

    for t in range(1, H+1):
        # baseline (no D(FSI) shock)
        dlogy_base[t]  = cG + phi * dlogy_base[t-1]
        ylog_base[t]   = ylog_base[t-1] + dlogy_base[t]

        # shocked path: one-time D(FSI) shock at t=0 -> enters at t=1
        x_lag = sigma if t == 1 else 0.0
        dlogy_shock[t] = cG + phi * dlogy_shock[t-1] + beta * x_lag
        ylog_shock[t]  = ylog_shock[t-1] + dlogy_shock[t]

    lvl_base  = np.exp(ylog_base)
    lvl_shock = np.exp(ylog_shock)
    idx_base  = 100 * lvl_base  / lvl_base[0]
    idx_shock = 100 * lvl_shock / lvl_shock[0]
    loss_pct  = 100 * (idx_shock - idx_base) / idx_base
    # Also return IRF and cumulative in percent for convenience:
    irf_pct = exact_pct(dlogy_shock)
    cum_pct = exact_pct(ylog_shock)
    return idx_base, idx_shock, loss_pct, irf_pct, cum_pct

# ----------------------------
# Load & prep data
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

# Regimes by G and FSI bands for σ
df["regime_G"] = df[COLS["g"]].apply(regime_from_g)
df["fsi_band"] = df[COLS["fsi"]].apply(fsi_band_from_level)

# ----------------------------
# Compute stats:
#   - mean_G per G-regime (drives coefficients)
#   - std of FSI and D(FSI) per FSI-band (drives σ)
# Also count n for transparency.
# ----------------------------
mean_G  = (df.dropna(subset=[COLS["g"]])
             .groupby("regime_G")[COLS["g"]]
             .mean())

n_G     = (df.dropna(subset=[COLS["g"]])
             .groupby("regime_G")[COLS["g"]]
             .size())

std_FSI = (df.dropna(subset=[COLS["fsi"]])
             .groupby("fsi_band")[COLS["fsi"]]
             .std(ddof=1))
std_dFSI= (df.dropna(subset=[COLS["dfsi"]])
             .groupby("fsi_band")[COLS["dfsi"]]
             .std(ddof=1))

n_FSI   = (df.dropna(subset=[COLS["fsi"]])
             .groupby("fsi_band")[COLS["fsi"]]
             .size())

# Build regime table (G-defined mean_G; FSI-band stds)
regimes = ["LS","IS","HS"]
rows_stats = []
for reg in regimes:
    rows_stats.append({
        "regime":   reg,
        "mean_G":   mean_G.get(reg, np.nan),
        "std_FSI":  std_FSI.get(reg, np.nan),
        "std_dFSI": std_dFSI.get(reg, np.nan),
        "n_G":      int(n_G.get(reg, 0)),
        "n_FSI":    int(n_FSI.get(reg, 0)),
    })
stats_df = pd.DataFrame(rows_stats).set_index("regime")
print("\nRegime statistics (G-defined mean_G; FSI-band stds):")
print(stats_df.round(6))

# ----------------------------
# Parameters & simulations by regime
# ----------------------------
rows_params = []
sims = {}
for reg in regimes:
    gbar = stats_df.loc[reg, "mean_G"]
    sigma= stats_df.loc[reg, "std_dFSI"]

    if pd.isna(gbar) or pd.isna(sigma) or stats_df.loc[reg, "n_G"] < 1 or stats_df.loc[reg, "n_FSI"] < 2:
        rows_params.append({
            "regime": reg, "G_mean": gbar, "sigma_dFSI": sigma,
            "beta": np.nan, "phi": np.nan, "cG": np.nan,
            "cum_4q_%": np.nan, "cum_20q_%": np.nan
        })
        continue

    # Regime-specific coefficients at G = gbar
    cG   = cL + cN * gbar
    beta = bL + bN * gbar
    phi  = aL + aN * gbar

    # Simulate baseline vs +1σ shock
    idx_base, idx_shock, loss_pct, irf_pct, cum_pct = simulate_paths(cG, phi, beta, sigma, H)

    sims[reg] = {
        "base": idx_base, "shock": idx_shock, "loss": loss_pct,
        "irf_pct": irf_pct, "cum_pct": cum_pct
    }

    rows_params.append({
        "regime": reg, "G_mean": gbar, "sigma_dFSI": sigma,
        "beta": beta, "phi": phi, "cG": cG,
        "cum_4q_%": cum_pct[H1],
        "cum_20q_%": cum_pct[H5]
    })

params_df = pd.DataFrame(rows_params).set_index("regime")
print("\nCumulative effect of a 1σ D(FSI) shock (dummies=0), percent change in RGDP level:")
print(params_df[["G_mean","sigma_dFSI","beta","phi","cum_4q_%","cum_20q_%"]].round(4))

# ----------------------------
# Plot: 3×2 (LS/IS/HS × (IRF, cumulative))
# ----------------------------
quarters = np.arange(H+1)
fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
color_irf = {"LS": "#2E7D32", "IS": "#F9A825", "HS": "#C62828"}
color_cum = {"LS": "#66BB6A", "IS": "#FFD54F", "HS": "#EF5350"}

for i, reg in enumerate(regimes):
    ax_irf = axes[i, 0]
    ax_cum = axes[i, 1]

    if reg not in sims:
        ax_irf.text(0.5, 0.5, f"{reg}: insufficient data", ha="center", va="center")
        ax_cum.text(0.5, 0.5, f"{reg}: insufficient data", ha="center", va="center")
        for ax in (ax_irf, ax_cum):
            ax.axis("off")
        continue

    # IRF (%)
    ax_irf.plot(quarters, sims[reg]["irf_pct"], lw=2, color=color_irf[reg])
    ax_irf.axhline(0, color="black", lw=0.8)
    ax_irf.set_title(f"{reg} — IRF (quarterly, %)")
    ax_irf.set_xlabel("Quarters after shock")
    ax_irf.set_ylabel("Percent")
    ax_irf.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax_irf.grid(True, alpha=0.25)

    # Cumulative (%)
    ax_cum.plot(quarters, sims[reg]["cum_pct"], lw=2, color=color_cum[reg])
    ax_cum.axhline(0, color="black", lw=0.8)
    for q, lab in [(H1, "1y"), (H5, "5y")]:
        ax_cum.axvline(q, color="gray", lw=0.8, ls="--")
        ax_cum.text(q, ax_cum.get_ylim()[1]*0.9, lab, ha="center", va="top", fontsize=9, color="gray")
    ax_cum.set_title(f"{reg} — Cumulative (level, %)")
    ax_cum.set_xlabel("Quarters after shock")
    ax_cum.set_ylabel("Percent")
    ax_cum.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax_cum.grid(True, alpha=0.25)

plt.suptitle("IRF & Cumulative (STR): Regimes by G, σ from FSI bands\nExact exp transform; intercept added each step", y=1.02, fontsize=12)
plt.savefig(OUT_PNG, dpi=180)
print(f"\nSaved plot → {OUT_PNG.resolve()}")