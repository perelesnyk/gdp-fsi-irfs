# -*- coding: utf-8 -*-
# STR regime switches: plot ONLY baseline-subtracted IRFs and cumulative paths
# Transitions: LS→HS, HS→IS, IS→LS
# Regimes defined by G (for coefficients), FSI means used for jump size
# Requirements: pandas, numpy, matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------
# Inputs & settings
# ----------------------------
CSV_PATH = "rgdp_str_table.csv"
OUT_PNG  = Path("irf_regime_switching_dev_only.png")

# Regime definitions
G_CUT_LOW  = 0.3422
G_CUT_HIGH = 0.9655
FSI_CUT_LOW  = 0.1187
FSI_CUT_HIGH = 0.1704

# Simulation parameters
H = 20            # horizon (quarters)
SWITCH_LAG = 3    # regime switch after this many quarters
H1 = 4            # 1-year horizon
H5 = 20           # 5-year horizon

# STR coefficients (from your estimation)
# dlog y_t = (cL + bL*dFSI_{t-1} + aL*dlogy_{t-1})
#          + (cN + bN*dFSI_{t-1} + aN*dlogy_{t-1}) * G
# (dummies = 0 for IRFs)
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
    if pd.isna(fsi): return "NA"
    if fsi <= FSI_CUT_LOW:        return "LS"
    elif fsi <= FSI_CUT_HIGH:     return "IS"
    else:                         return "HS"

def exact_pct(x):
    """Exact log->percent: 100*(exp(x)-1)."""
    return 100.0 * (np.exp(x) - 1.0)

def debug_transition(seq, par_dict, H=8, switch_lag=3, label="DEBUG"):
    from_reg, to_reg = seq
    pf, pt = par_dict[from_reg], par_dict[to_reg]
    dFSI_jump = pt["mean_FSI"] - pf["mean_FSI"]

    dlog_b = np.zeros(H+1)
    dlog_s = np.zeros(H+1)
    ylog_b = np.zeros(H+1)
    ylog_s = np.zeros(H+1)

    print(f"\n[{label}] {from_reg} → {to_reg}")
    print(f"ΔFSI = mean_FSI[{to_reg}] - mean_FSI[{from_reg}] = {dFSI_jump:.6f}")
    hdr = ("t  "
           "reg   "
           "cG       beta       phi     x_lag     "
           "dlog_base   dlog_shock   IRF_log   cum_IRF_log")
    print(hdr)
    print("-"*len(hdr))

    cum_irf = 0.0
    for t in range(1, H+1):
        # baseline
        cG, beta, phi = pf["cG"], pf["beta"], pf["phi"]
        dlog_b[t] = cG + phi*dlog_b[t-1]
        ylog_b[t] = ylog_b[t-1] + dlog_b[t]

        # switching
        reg = from_reg if t < switch_lag else to_reg
        pr = par_dict[reg]
        x_lag = dFSI_jump if t == 1 else 0.0
        dlog_s[t] = pr["cG"] + pr["phi"]*dlog_s[t-1] + pr["beta"]*x_lag
        ylog_s[t] = ylog_s[t-1] + dlog_s[t]

        irf_log = dlog_s[t] - dlog_b[t]
        cum_irf += irf_log

        print(f"{t:<2d} {reg:<3s}  "
              f"{pr['cG']:+0.6f} {pr['beta']:+0.6f} {pr['phi']:+0.6f} "
              f"{x_lag:+0.6f}  "
              f"{dlog_b[t]:+0.6f}   {dlog_s[t]:+0.6f}   "
              f"{irf_log:+0.6f}   {cum_irf:+0.6f}")

# ----------------------------
# Simulation (baseline-subtracted)
# ----------------------------
def simulate_regime_switch(seq, par_dict, H, switch_lag=3):
    """
    Simulate baseline (constant 'from' regime) vs. switching to 'to' regime.
    Shock is a one-off jump in FSI LEVEL equal to mean_FSI[to] - mean_FSI[from],
    appearing at t=1 and entering via dFSI_{t-1}.
    Returns IRF_t (%) and cumulative_t (%) as deviations vs baseline.
    """
    from_reg, to_reg = seq
    pars_from = par_dict[from_reg]
    pars_to   = par_dict[to_reg]

    fsi_jump = pars_to["mean_FSI"] - pars_from["mean_FSI"]

    dlog_base  = np.zeros(H+1)
    dlog_shock = np.zeros(H+1)
    ylog_base  = np.zeros(H+1)
    ylog_shock = np.zeros(H+1)

    for t in range(1, H+1):
        # baseline (stay in 'from' regime)
        cG, beta, phi = pars_from["cG"], pars_from["beta"], pars_from["phi"]
        dlog_base[t] = cG + phi * dlog_base[t-1]
        ylog_base[t] = ylog_base[t-1] + dlog_base[t]

        # switching path
        current = from_reg if t < switch_lag else to_reg
        cG_s, beta_s, phi_s = par_dict[current]["cG"], par_dict[current]["beta"], par_dict[current]["phi"]
        x_lag = fsi_jump if t == 1 else 0.0  # one-off jump shows up with a lag
        dlog_shock[t] = cG_s + phi_s * dlog_shock[t-1] + beta_s * x_lag
        ylog_shock[t] = ylog_shock[t-1] + dlog_shock[t]

    irf_log = dlog_shock - dlog_base
    cum_log = ylog_shock - ylog_base
    irf_pct = exact_pct(irf_log)
    cum_pct = exact_pct(cum_log)
    return irf_pct, cum_pct

# ----------------------------
# Load data & compute regime params
# ----------------------------
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]

for k in ["g", "fsi", "dfsi"]:
    if COLS[k] in df.columns:
        df[COLS[k]] = (
            df[COLS[k]].astype(str).str.replace(",", ".", regex=False)
                       .replace({"": np.nan})
                       .astype(float)
        )

df["__date"] = df[COLS["date"]].apply(parse_date_quarter)
df = df.sort_values("__date").reset_index(drop=True)
df["regime_G"] = df[COLS["g"]].apply(regime_from_g)
df["fsi_band"] = df[COLS["fsi"]].apply(fsi_band_from_level)

# Regime-mean G (for coefficients) and mean FSI (for jump size)
mean_G   = df.groupby("regime_G")[COLS["g"]].mean()
mean_FSI = df.groupby("fsi_band")[COLS["fsi"]].mean()

regimes = ["LS","IS","HS"]
par_dict = {}
for reg in regimes:
    gbar = mean_G.get(reg, np.nan)
    mean_fsi = mean_FSI.get(reg, np.nan)
    cG   = cL + cN * gbar
    beta = bL + bN * gbar
    phi  = aL + aN * gbar
    par_dict[reg] = {"cG": cG, "beta": beta, "phi": phi, "mean_FSI": mean_fsi}

print("\nParameters per regime (using mean G and mean FSI):")
print(pd.DataFrame(par_dict).T.round(5))

# ----------------------------
# Simulate transitions (Δ vs baseline only)
# ----------------------------
transitions = [("LS","HS"), ("HS","IS"), ("IS","LS")]
paths_dev = {}
summary_rows = []

for (r1, r2) in transitions:
    irf_dev, cum_dev = simulate_regime_switch([r1, r2], par_dict, H, SWITCH_LAG)
    debug_transition([r1, r2], par_dict, H=8, switch_lag=SWITCH_LAG, label=f"{r1}→{r2}")
    paths_dev[(r1, r2)] = {"irf": irf_dev, "cum": cum_dev}

    fsi_from = par_dict[r1]["mean_FSI"]
    fsi_to   = par_dict[r2]["mean_FSI"]
    print(f"\nTransition {r1} → {r2}:")
    print(f"   mean FSI from = {fsi_from:.4f}, to = {fsi_to:.4f}, Δ = {fsi_to - fsi_from:.4f}")

    summary_rows.append({
        "Transition": f"{r1}→{r2}",
        "Δ FSI": fsi_to - fsi_from,
        "1y_cum_%": cum_dev[H1],
        "5y_cum_%": cum_dev[H5],
    })

summary_df = pd.DataFrame(summary_rows)
print("\nCumulative GDP effect (Δ vs baseline):")
print(summary_df.round(4).to_string(index=False))

# ----------------------------
# Plot: ONLY IRF and cumulative (Δ vs baseline)
# ----------------------------
quarters = np.arange(H+1)
fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
colors = {"LS":"#2E7D32", "IS":"#F9A825", "HS":"#C62828"}

for i, (r1, r2) in enumerate(transitions):
    ax_irf, ax_cum = axes[i]

    # --- IRF plot ---
    ax_irf.plot(quarters, paths_dev[(r1,r2)]["irf"], lw=2, color=colors[r2])
    ax_irf.axhline(0, color="black", lw=0.8)
    ax_irf.axvline(SWITCH_LAG, ls="--", color="gray", lw=1)
    ax_irf.set_xticks(np.arange(0, H+1, 2))  # tick every 2 quarters
    ax_irf.set_title(f"{r1}→{r2}: IRF (Δ vs baseline), %")
    ax_irf.set_xlabel("Quarters")
    ax_irf.set_ylabel("Percent")
    ax_irf.grid(True, alpha=0.25)

    # --- Cumulative plot ---
    ax_cum.plot(quarters, paths_dev[(r1,r2)]["cum"], lw=2, color=colors[r2])
    ax_cum.axhline(0, color="black", lw=0.8)
    ax_cum.axvline(SWITCH_LAG, ls="--", color="gray", lw=1)
    ax_cum.set_xticks(np.arange(0, H+1, 2))  # tick every 2 quarters
    ymax = ax_cum.get_ylim()[1]
    ax_cum.text(SWITCH_LAG, ymax*0.9, "Regime switch", ha="center", color="gray")
    ax_cum.set_title(f"{r1}→{r2}: Cumulative (Δ vs baseline), %")
    ax_cum.set_xlabel("Quarters")
    ax_cum.set_ylabel("Percent")
    ax_cum.grid(True, alpha=0.25)

# plt.suptitle(
#     f"Regime-switching IRF & Cumulative (Δ vs baseline only)\n"
#     f"(switch after {SWITCH_LAG} quarters; FSI jump = mean(FSI_target) − mean(FSI_origin))",
#     y=1.02
# )
plt.savefig(OUT_PNG, dpi=180)
print(f"\nSaved figure → {OUT_PNG.resolve()}")