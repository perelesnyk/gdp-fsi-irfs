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
OUT_JPG  = Path("irf_regime_switching_dev_only.jpg")

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

def fsi_band_from_level(fsi):
    if pd.isna(fsi): return "NA"
    if fsi <= FSI_CUT_LOW:        return "LS"
    elif fsi <= FSI_CUT_HIGH:     return "IS"
    else:                         return "HS"

def exact_pct(x):
    """Exact log->percent: 100*(exp(x)-1)."""
    return 100.0 * (np.exp(x) - 1.0)

def coeffs_with_variance(G):
    """
    Compute effective coefficients and their variances using HAC matrix.
    Returns: dict with c, beta, phi, se_c, se_beta, se_phi
    """
    c = cL + cN * G
    beta = bL + bN * G
    phi = aL + aN * G
    
    # Jacobian vectors w.r.t. [cL, bL, aL, cN, bN, aN]
    J_c = np.array([1, 0, 0, G, 0, 0])
    J_beta = np.array([0, 1, 0, 0, G, 0])
    J_phi = np.array([0, 0, 1, 0, 0, G])
    
    # Variance via delta method: J' * HAC * J
    var_c = J_c @ HAC_matrix @ J_c
    var_beta = J_beta @ HAC_matrix @ J_beta
    var_phi = J_phi @ HAC_matrix @ J_phi
    
    return {
        'c': c, 'beta': beta, 'phi': phi,
        'se_c': np.sqrt(var_c),
        'se_beta': np.sqrt(var_beta),
        'se_phi': np.sqrt(var_phi)
    }

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
# Simulation (baseline-subtracted with confidence intervals)
# ----------------------------
def simulate_regime_switch(seq, par_dict, H, switch_lag=3):
    """
    Simulate baseline (constant 'from' regime) vs. switching to 'to' regime.
    Shock is a one-off jump in FSI LEVEL equal to mean_FSI[to] - mean_FSI[from],
    appearing at t=1 and entering via dFSI_{t-1}.
    Returns IRF_t (%), cumulative_t (%), and confidence intervals (68% and 95%) as deviations vs baseline.
    """
    from_reg, to_reg = seq
    pars_from = par_dict[from_reg]
    pars_to   = par_dict[to_reg]

    fsi_jump = pars_to["mean_FSI"] - pars_from["mean_FSI"]
    
    # Get G values for variance calculation
    G_from = par_dict[from_reg]["G"]
    G_to = par_dict[to_reg]["G"]

    dlog_base  = np.zeros(H+1)
    dlog_shock = np.zeros(H+1)
    ylog_base  = np.zeros(H+1)
    ylog_shock = np.zeros(H+1)
    
    # Variance tracking for switching path
    dlog_shock_var = np.zeros(H+1)

    for t in range(1, H+1):
        # baseline (stay in 'from' regime)
        cG, beta, phi = pars_from["cG"], pars_from["beta"], pars_from["phi"]
        dlog_base[t] = cG + phi * dlog_base[t-1]
        ylog_base[t] = ylog_base[t-1] + dlog_base[t]

        # switching path - point estimate
        current = from_reg if t < switch_lag else to_reg
        current_G = G_from if t < switch_lag else G_to
        cG_s, beta_s, phi_s = par_dict[current]["cG"], par_dict[current]["beta"], par_dict[current]["phi"]
        x_lag = fsi_jump if t == 1 else 0.0  # one-off jump shows up with a lag
        dlog_shock[t] = cG_s + phi_s * dlog_shock[t-1] + beta_s * x_lag
        ylog_shock[t] = ylog_shock[t-1] + dlog_shock[t]
        
        # Variance computation using HAC matrix
        cf = coeffs_with_variance(current_G)
        var_intercept = cf['se_c']**2
        var_ar = (phi_s**2) * dlog_shock_var[t-1]
        var_shock = (cf['se_beta'] * x_lag)**2 if x_lag != 0 else 0
        var_phi_term = (cf['se_phi'] * dlog_shock[t-1])**2
        
        dlog_shock_var[t] = var_intercept + var_ar + var_shock + var_phi_term

    # Confidence bands (68% and 95%)
    dlog_shock_se = np.sqrt(dlog_shock_var)
    dlog_shock_upper_68 = dlog_shock + 1.0 * dlog_shock_se
    dlog_shock_lower_68 = dlog_shock - 1.0 * dlog_shock_se
    dlog_shock_upper_95 = dlog_shock + 1.96 * dlog_shock_se
    dlog_shock_lower_95 = dlog_shock - 1.96 * dlog_shock_se
    
    ylog_shock_upper_68 = np.cumsum(dlog_shock_upper_68)
    ylog_shock_lower_68 = np.cumsum(dlog_shock_lower_68)
    ylog_shock_upper_95 = np.cumsum(dlog_shock_upper_95)
    ylog_shock_lower_95 = np.cumsum(dlog_shock_lower_95)

    # Deviations vs baseline
    irf_log = dlog_shock - dlog_base
    irf_log_upper_68 = dlog_shock_upper_68 - dlog_base
    irf_log_lower_68 = dlog_shock_lower_68 - dlog_base
    irf_log_upper_95 = dlog_shock_upper_95 - dlog_base
    irf_log_lower_95 = dlog_shock_lower_95 - dlog_base
    
    cum_log = ylog_shock - ylog_base
    cum_log_upper_68 = ylog_shock_upper_68 - ylog_base
    cum_log_lower_68 = ylog_shock_lower_68 - ylog_base
    cum_log_upper_95 = ylog_shock_upper_95 - ylog_base
    cum_log_lower_95 = ylog_shock_lower_95 - ylog_base
    
    # Convert to exact percent
    irf_pct = exact_pct(irf_log)
    irf_pct_upper_68 = exact_pct(irf_log_upper_68)
    irf_pct_lower_68 = exact_pct(irf_log_lower_68)
    irf_pct_upper_95 = exact_pct(irf_log_upper_95)
    irf_pct_lower_95 = exact_pct(irf_log_lower_95)
    
    cum_pct = exact_pct(cum_log)
    cum_pct_upper_68 = exact_pct(cum_log_upper_68)
    cum_pct_lower_68 = exact_pct(cum_log_lower_68)
    cum_pct_upper_95 = exact_pct(cum_log_upper_95)
    cum_pct_lower_95 = exact_pct(cum_log_lower_95)
    
    return {
        'irf': irf_pct, 
        'irf_upper_68': irf_pct_upper_68, 'irf_lower_68': irf_pct_lower_68,
        'irf_upper_95': irf_pct_upper_95, 'irf_lower_95': irf_pct_lower_95,
        'cum': cum_pct,
        'cum_upper_68': cum_pct_upper_68, 'cum_lower_68': cum_pct_lower_68,
        'cum_upper_95': cum_pct_upper_95, 'cum_lower_95': cum_pct_lower_95
    }

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
    par_dict[reg] = {"cG": cG, "beta": beta, "phi": phi, "mean_FSI": mean_fsi, "G": gbar}

print("\nParameters per regime (using mean G and mean FSI):")
print(pd.DataFrame(par_dict).T.round(5))

# ----------------------------
# Simulate transitions (Δ vs baseline only)
# ----------------------------
transitions = [("LS","HS"), ("HS","IS"), ("IS","LS")]
paths_dev = {}
summary_rows = []

for (r1, r2) in transitions:
    result = simulate_regime_switch([r1, r2], par_dict, H, SWITCH_LAG)
    debug_transition([r1, r2], par_dict, H=8, switch_lag=SWITCH_LAG, label=f"{r1}→{r2}")
    paths_dev[(r1, r2)] = result

    fsi_from = par_dict[r1]["mean_FSI"]
    fsi_to   = par_dict[r2]["mean_FSI"]
    print(f"\nTransition {r1} → {r2}:")
    print(f"   mean FSI from = {fsi_from:.4f}, to = {fsi_to:.4f}, Δ = {fsi_to - fsi_from:.4f}")

    summary_rows.append({
        "Transition": f"{r1}→{r2}",
        "Δ FSI": fsi_to - fsi_from,
        "1y_cum_%": result['cum'][H1],
        "5y_cum_%": result['cum'][H5],
    })

summary_df = pd.DataFrame(summary_rows)
print("\nCumulative GDP effect (Δ vs baseline):")
print(summary_df.round(4).to_string(index=False))

# ----------------------------
# Plot: ONLY IRF and cumulative (Δ vs baseline) with confidence intervals
# ----------------------------
quarters = np.arange(H+1)
fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
colors = {"LS":"#2E7D32", "IS":"#F9A825", "HS":"#C62828"}

for i, (r1, r2) in enumerate(transitions):
    ax_irf, ax_cum = axes[i]
    result = paths_dev[(r1, r2)]

    # --- IRF plot with confidence intervals (68% shaded + 95% dashed) ---
    fsi_jump = par_dict[r2]["mean_FSI"] - par_dict[r1]["mean_FSI"]
    ax_irf.fill_between(quarters, result['irf_lower_68'], result['irf_upper_68'], 
                         alpha=0.2, color=colors[r2], label='68% CI')
    ax_irf.plot(quarters, result['irf'], lw=2, color=colors[r2], label='Point Estimate')
    ax_irf.plot(quarters, result['irf_lower_95'], ls='--', lw=1, color=colors[r2], alpha=0.7, label='95% CI')
    ax_irf.plot(quarters, result['irf_upper_95'], ls='--', lw=1, color=colors[r2], alpha=0.7)
    ax_irf.axhline(0, color="black", lw=0.8)
    ax_irf.axvline(SWITCH_LAG, ls="--", color="gray", lw=1, label=f'Regime Switch (t={SWITCH_LAG})')
    ax_irf.set_xticks(np.arange(0, H+1, 2))  # tick every 2 quarters
    ax_irf.set_title(f"Regime Transition {r1}→{r2}: Quarterly GDP Response (ΔFSI={fsi_jump:+.3f})")
    ax_irf.set_xlabel("Quarters After Shock")
    ax_irf.set_ylabel("GDP Growth Deviation (%)")
    ax_irf.grid(True, alpha=0.25)
    ax_irf.legend(loc='best', fontsize=8)

    # --- Cumulative plot with confidence intervals (68% shaded + 95% dashed) ---
    ax_cum.fill_between(quarters, result['cum_lower_68'], result['cum_upper_68'],
                         alpha=0.2, color=colors[r2], label='68% CI')
    ax_cum.plot(quarters, result['cum'], lw=2, color=colors[r2], label='Point Estimate')
    ax_cum.plot(quarters, result['cum_lower_95'], ls='--', lw=1, color=colors[r2], alpha=0.7, label='95% CI')
    ax_cum.plot(quarters, result['cum_upper_95'], ls='--', lw=1, color=colors[r2], alpha=0.7)
    ax_cum.axhline(0, color="black", lw=0.8)
    ax_cum.axvline(SWITCH_LAG, ls="--", color="gray", lw=1, label=f'Regime Switch (t={SWITCH_LAG})')
    ax_cum.set_xticks(np.arange(0, H+1, 2))  # tick every 2 quarters
    ymin, ymax = ax_cum.get_ylim()
    # Position label lower for LS->HS (negative cumulative), higher for others
    label_pos = ymin*0.85 if r1 == "LS" and r2 == "HS" else ymax*0.9
    ax_cum.text(SWITCH_LAG, label_pos, f"Regime switch\n(t={SWITCH_LAG})", ha="center", color="gray", fontsize=8)
    ax_cum.set_title(f"Regime Transition {r1}→{r2}: Cumulative GDP Effect")
    ax_cum.set_xlabel("Quarters After Shock")
    ax_cum.set_ylabel("Cumulative GDP Level Deviation (%)")
    ax_cum.grid(True, alpha=0.25)
    ax_cum.legend(loc='best', fontsize=8)

# plt.suptitle(
#     f"Regime-switching IRF & Cumulative (Δ vs baseline only)\n"
#     f"(switch after {SWITCH_LAG} quarters; FSI jump = mean(FSI_target) − mean(FSI_origin))",
#     y=1.02
# )
plt.savefig(OUT_JPG, dpi=300, format="jpeg")
print(f"\nSaved figure → {OUT_JPG.resolve()}")