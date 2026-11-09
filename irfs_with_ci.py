# -*- coding: utf-8 -*-
# IRF for STR model with confidence intervals
# Shows:
#  - HS and LS with 68% CI bands
#  - IS at three G values: median (0.654), lower bound (0.3422), upper bound (0.9655)
# Requirements: pandas, numpy, matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ----------------------------
# Inputs & settings
# ----------------------------
CSV_PATH = "rgdp_str_table.csv"
OUT_PNG  = Path("irf_with_confidence_intervals.png")
OUT_JPG  = Path("irf_with_confidence_intervals.jpg")

# Regime cutoffs
G_CUT_LOW  = 0.3422      # LS / IS boundary
G_CUT_HIGH = 0.9655      # IS / HS boundary
FSI_CUT_LOW  = 0.1187    # LS / IS
FSI_CUT_HIGH = 0.1704    # IS / HS

# IS median G value (you can adjust this)
G_IS_MEDIAN = 0.654

H1 = 4      # 1-year horizon
H5 = 20     # 5-year horizon
H  = 20     # plot horizon (quarters)

# STR coefficients
cL, bL, aL = 0.008409,  -0.240669, -0.290350
cN, bN, aN = -0.008686,  0.221148,  0.848650

# Standard errors (from HAC covariance matrix)
se_cL, se_bL, se_aL = 0.0029, 0.0149, 0.0213
se_cN, se_bN, se_aN = 0.0032, 0.0161, 0.0590

# Full HAC covariance matrix (from your estimation)
# Order: [cL, bL, aL, cN, bN, aN]
# Rows/cols: C (linear), D(FSI(-1)) (linear), DLOG(RGDP(-1)) (linear),
#            C (nonlinear), D(FSI(-1)) (nonlinear), DLOG(RGDP(-1)) (nonlinear)
HAC_matrix = np.array([
    # cL (C linear)
    [8.507526398719898e-06, -2.481778447058299e-06, 7.966160692733066e-06,
     -7.867716334907634e-06, 1.209793606002429e-07, -6.058187308713758e-05],
    # bL (D(FSI(-1)) linear)
    [-2.481778447058299e-06, 0.0002222572618298202, -5.083520951131941e-05,
     1.030834129704509e-05, -0.0002267137873032802, -0.0002898724110253803],
    # aL (DLOG(RGDP(-1)) linear)
    [7.966160692733066e-06, -5.083520951131941e-05, 0.0004538704845508114,
     -1.539527789330668e-05, 8.75283613440443e-05, 0.0002569638379923147],
    # cN (C nonlinear)
    [-7.867716334907634e-06, 1.030834129704509e-05, -1.539527789330668e-05,
     1.029641781553392e-05, -7.276883740540779e-06, -1.977877749943097e-05],
    # bN (D(FSI(-1)) nonlinear)
    [1.209793606002429e-07, -0.0002267137873032802, 8.75283613440443e-05,
     -7.276883740540779e-06, 0.000259886179521454, 0.0003731099645247996],
    # aN (DLOG(RGDP(-1)) nonlinear)
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

def compute_irf_with_variance(beta, phi, sigma, se_beta, se_phi, H, cov_beta_phi=0):
    """
    Compute IRF with proper variance propagation including covariance.
    Returns: (irf_log, irf_var, irf_log_lower_95, irf_log_upper_95) arrays of length H+1
    
    Args:
        beta: Impact coefficient
        phi: Persistence coefficient
        sigma: Shock size
        se_beta: Standard error of beta
        se_phi: Standard error of phi
        H: Horizon
        cov_beta_phi: Covariance between beta and phi (default 0)
    """
    irf_log = np.zeros(H + 1)
    irf_var = np.zeros(H + 1)
    
    if sigma is None or np.isnan(sigma) or sigma == 0:
        return irf_log, irf_var
    
    # First impact at t=1
    irf_log[1] = beta * sigma
    irf_var[1] = (se_beta * sigma)**2
    
    # Propagate through AR(1) process with covariance
    for h in range(2, H + 1):
        irf_log[h] = irf_log[h-1] * phi
        # Var(phi * y_{h-1}) = phi^2 * Var(y_{h-1}) + y_{h-1}^2 * Var(phi)
        # Note: covariance terms from beta-phi correlation affect variance accumulation
        irf_var[h] = (phi**2) * irf_var[h-1] + (irf_log[h-1]**2) * (se_phi**2)
    
    return irf_log, irf_var

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

# Regimes by G and bands by FSI
df["regime_G"] = df[COLS["g"]].apply(regime_from_G)
df["band_FSI"] = df[COLS["fsi"]].apply(band_from_FSI)

# ----------------------------
# 1) Compute mean_G and sigma_dFSI by regime
# ----------------------------
meanG = (
    df.loc[df["regime_G"].isin(["LS","IS","HS"]), [COLS["g"], "regime_G"]]
      .groupby("regime_G")[COLS["g"]]
      .mean()
      .rename("mean_G")
)

# For IS, also compute median
medianG_IS = df.loc[df["regime_G"] == "IS", COLS["g"]].median()
print(f"IS regime: mean_G = {meanG['IS']:.4f}, median_G = {medianG_IS:.4f}")

sigma_dFSI = (
    df.loc[df["band_FSI"].isin(["LS","IS","HS"]), [COLS["dfsi"], "band_FSI"]]
      .groupby("band_FSI")[COLS["dfsi"]]
      .std(ddof=1)
      .rename("std_dFSI")
)

stats_df = pd.concat([meanG, sigma_dFSI], axis=1).reindex(["LS","IS","HS"])
print("\nRegime statistics:")
print(stats_df.round(6).fillna(""))

# ----------------------------
# 2) Compute IRFs with confidence intervals
# ----------------------------
def compute_regime_irf(G, sigma, label):
    """Compute IRF for a given G value using HAC-based variance."""
    beta = bL + bN * G
    phi  = aL + aN * G
    
    # Jacobian vectors for beta(G) and phi(G) w.r.t. [cL, bL, aL, cN, bN, aN]
    J_beta = np.array([0, 1, 0, 0, G, 0])  # ∂β/∂θ
    J_phi  = np.array([0, 0, 1, 0, 0, G])  # ∂φ/∂θ
    
    # HAC-based variance using delta method: Var(f(θ)) = J' * Σ * J
    var_beta = J_beta @ HAC_matrix @ J_beta.T
    var_phi  = J_phi @ HAC_matrix @ J_phi.T
    cov_beta_phi = J_beta @ HAC_matrix @ J_phi.T
    
    se_beta = np.sqrt(var_beta)
    se_phi  = np.sqrt(var_phi)
    
    # Compute IRF with variance
    irf_log, irf_var = compute_irf_with_variance(beta, phi, sigma, se_beta, se_phi, H, cov_beta_phi)
    
    # Confidence bands: 68% (±1 SE) and 95% (±1.96 SE)
    irf_se = np.sqrt(irf_var)
    irf_log_upper_68 = irf_log + irf_se
    irf_log_lower_68 = irf_log - irf_se
    irf_log_upper_95 = irf_log + 1.96 * irf_se
    irf_log_lower_95 = irf_log - 1.96 * irf_se
    
    # Cumulative
    cum_log = np.cumsum(irf_log)
    cum_var = np.cumsum(irf_var)
    cum_se = np.sqrt(cum_var)
    cum_log_upper_68 = cum_log + cum_se
    cum_log_lower_68 = cum_log - cum_se
    cum_log_upper_95 = cum_log + 1.96 * cum_se
    cum_log_lower_95 = cum_log - 1.96 * cum_se
    
    # Convert to percent
    irf_pct = log_to_percent(irf_log)
    irf_pct_upper_68 = log_to_percent(irf_log_upper_68)
    irf_pct_lower_68 = log_to_percent(irf_log_lower_68)
    irf_pct_upper_95 = log_to_percent(irf_log_upper_95)
    irf_pct_lower_95 = log_to_percent(irf_log_lower_95)
    
    cum_pct = log_to_percent(cum_log)
    cum_pct_upper_68 = log_to_percent(cum_log_upper_68)
    cum_pct_lower_68 = log_to_percent(cum_log_lower_68)
    cum_pct_upper_95 = log_to_percent(cum_log_upper_95)
    cum_pct_lower_95 = log_to_percent(cum_log_lower_95)
    
    return {
        "label": label,
        "G": G, "sigma": sigma, "beta": beta, "phi": phi,
        "se_beta": se_beta, "se_phi": se_phi, "cov_beta_phi": cov_beta_phi,
        "irf_pct": irf_pct,
        "irf_pct_upper_68": irf_pct_upper_68,
        "irf_pct_lower_68": irf_pct_lower_68,
        "irf_pct_upper_95": irf_pct_upper_95,
        "irf_pct_lower_95": irf_pct_lower_95,
        "cum_pct": cum_pct,
        "cum_pct_upper_68": cum_pct_upper_68,
        "cum_pct_lower_68": cum_pct_lower_68,
        "cum_pct_upper_95": cum_pct_upper_95,
        "cum_pct_lower_95": cum_pct_lower_95
    }

# Compute IRFs for each regime/variant
results = {}

# LS with CI
results["LS"] = compute_regime_irf(
    G=stats_df.loc["LS", "mean_G"],
    sigma=stats_df.loc["LS", "std_dFSI"],
    label="LS"
)

# HS with CI
results["HS"] = compute_regime_irf(
    G=stats_df.loc["HS", "mean_G"],
    sigma=stats_df.loc["HS", "std_dFSI"],
    label="HS"
)

# IS - three variants (use mean sigma for all)
sigma_IS = stats_df.loc["IS", "std_dFSI"]
results["IS_low"] = compute_regime_irf(G=G_CUT_LOW, sigma=sigma_IS, label=f"IS (G={G_CUT_LOW:.4f})")
results["IS_med"] = compute_regime_irf(G=G_IS_MEDIAN, sigma=sigma_IS, label=f"IS (G={G_IS_MEDIAN:.3f})")
results["IS_high"] = compute_regime_irf(G=G_CUT_HIGH, sigma=sigma_IS, label=f"IS (G={G_CUT_HIGH:.4f})")

# Print summary
print("\nCumulative effects at 1y and 5y (%):")
for key in ["LS", "HS", "IS_low", "IS_med", "IS_high"]:
    r = results[key]
    print(f"{r['label']:20s}: G={r['G']:.4f}, β={r['beta']:7.4f} (SE={r['se_beta']:.4f}), "
          f"φ={r['phi']:6.4f} (SE={r['se_phi']:.4f}), Cov(β,φ)={r['cov_beta_phi']:8.6f}, "
          f"1y={r['cum_pct'][H1]:7.3f}, 5y={r['cum_pct'][H5]:7.3f}")

# ----------------------------
# 3) Plot with confidence intervals
# ----------------------------
quarters = np.arange(H + 1)
fig, axes = plt.subplots(5, 2, figsize=(14, 18), constrained_layout=True)

colors = {"LS": "#2E7D32", "HS": "#C62828", "IS": "#F9A825"}

def plot_regime(ax_irf, ax_cum, result, title_suffix, color, shock_size=None):
    """Helper to plot IRF and cumulative with 68% shaded CI and 95% dashed CI."""
    r = result
    
    # Add shock size to title if provided
    shock_text = f", σ={shock_size:.4f}" if shock_size is not None else ""
    
    # IRF plot
    ax_irf.fill_between(quarters, r["irf_pct_lower_68"], r["irf_pct_upper_68"], 
                        alpha=0.2, color=color, label="68% CI")
    ax_irf.plot(quarters, r["irf_pct"], lw=2.5, color=color, label="Point estimate")
    ax_irf.plot(quarters, r["irf_pct_lower_95"], ls='--', lw=1, color=color, alpha=0.7, label="95% CI")
    ax_irf.plot(quarters, r["irf_pct_upper_95"], ls='--', lw=1, color=color, alpha=0.7)
    ax_irf.axhline(0, color="black", lw=0.8)
    ax_irf.axvline(H1, color="gray", lw=0.6, ls="--", alpha=0.5)
    ax_irf.axvline(H5, color="gray", lw=0.6, ls="--", alpha=0.5)
    ax_irf.set_title(f"IRF — {title_suffix} (G={r['G']:.4f}{shock_text})", fontweight="bold")
    ax_irf.set_xlabel("Quarters after shock")
    ax_irf.set_ylabel("Percent change (quarter)")
    ax_irf.grid(True, alpha=0.25)
    ax_irf.legend(frameon=True, loc="best", fontsize=9)
    ax_irf.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    # Cumulative plot
    ax_cum.fill_between(quarters, r["cum_pct_lower_68"], r["cum_pct_upper_68"], 
                        alpha=0.2, color=color, label="68% CI")
    ax_cum.plot(quarters, r["cum_pct"], lw=2.5, color=color, label="Point estimate")
    ax_cum.plot(quarters, r["cum_pct_lower_95"], ls='--', lw=1, color=color, alpha=0.7, label="95% CI")
    ax_cum.plot(quarters, r["cum_pct_upper_95"], ls='--', lw=1, color=color, alpha=0.7)
    ax_cum.axhline(0, color="black", lw=0.8)
    ax_cum.axvline(H1, color="gray", lw=0.6, ls="--", alpha=0.5)
    ax_cum.axvline(H5, color="gray", lw=0.6, ls="--", alpha=0.5)
    ax_cum.text(H1, ax_cum.get_ylim()[0]*1.05, "1y", ha="center", va="top", fontsize=8, color="gray")
    ax_cum.text(H5, ax_cum.get_ylim()[0]*1.05, "5y", ha="center", va="top", fontsize=8, color="gray")
    ax_cum.set_title(f"Cumulative — {title_suffix} (G={r['G']:.4f}{shock_text})", fontweight="bold")
    ax_cum.set_xlabel("Quarters after shock")
    ax_cum.set_ylabel("Percent change (level)")
    ax_cum.grid(True, alpha=0.25)
    ax_cum.legend(frameon=True, loc="best", fontsize=9)
    ax_cum.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

# Row 0: LS
plot_regime(axes[0, 0], axes[0, 1], results["LS"], "LS regime", colors["LS"], 
           shock_size=stats_df.loc["LS", "std_dFSI"])

# Row 1: IS low-G
plot_regime(axes[1, 0], axes[1, 1], results["IS_low"], "IS regime (low-G)", colors["IS"],
           shock_size=stats_df.loc["IS", "std_dFSI"])

# Row 2: IS med-G
plot_regime(axes[2, 0], axes[2, 1], results["IS_med"], "IS regime (median-G)", colors["IS"],
           shock_size=stats_df.loc["IS", "std_dFSI"])

# Row 3: IS high-G
plot_regime(axes[3, 0], axes[3, 1], results["IS_high"], "IS regime (high-G)", colors["IS"],
           shock_size=stats_df.loc["IS", "std_dFSI"])

# Row 4: HS
plot_regime(axes[4, 0], axes[4, 1], results["HS"], "HS regime", colors["HS"],
           shock_size=stats_df.loc["HS", "std_dFSI"])

plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
plt.savefig(OUT_JPG, dpi=180, bbox_inches="tight", format='jpg', pil_kwargs={'quality': 95})
print(f"\nSaved plot → {OUT_PNG.resolve()}")
print(f"Saved plot → {OUT_JPG.resolve()}")
plt.show()
