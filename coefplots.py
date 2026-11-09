# plot_effective_coeffs_from_STR.py (robust to 's_lag3' etc.)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from pathlib import Path

FSI_CSV = "fsi_lag3_quarterly.csv"
OUT_FSI_JPG  = Path("regime_spans.jpg")
OUT_COEF_JPG = Path("effective_coeffs.jpg")

# STR params
alpha0 =  0.008409; alpha1 = -0.240669; alpha2 = -0.290350
beta0  = -0.008686; beta1  =  0.221148; beta2  =  0.848650
gamma  = 77.16299;  c_thr  =  0.127169

G_CUT_LOW, G_CUT_HIGH = 0.3422, 0.9655

REGION_COLOR = {
    "LS": (0.80, 0.93, 0.80, 0.45),  # softer but clearer green
    "IS": (0.99, 0.92, 0.70, 0.45),  # brighter yellow
    "HS": (0.96, 0.70, 0.70, 0.45),  # more distinct red
    "NA": (0.90, 0.90, 0.90, 0.35),  # neutral gray for missing data
}

def parse_date(x: str) -> pd.Timestamp:
    x = str(x).strip()
    if "Q" in x.upper():
        p = pd.Period(x, freq="Q")
        return p.asfreq("Q", "end").to_timestamp()
    return pd.to_datetime(x, errors="coerce")

def pick_col(df: pd.DataFrame, preferred_names, fallback_contains=None):
    # exact preferred
    for n in preferred_names:
        if n in df.columns: return n
    # case-insensitive contains
    low = {c.lower(): c for c in df.columns}
    if fallback_contains:
        for token in fallback_contains:
            for lc, orig in low.items():
                if token in lc: return orig
    raise ValueError(f"Could not find any of {preferred_names} or tokens {fallback_contains} in {list(df.columns)}")

def normalize_numeric(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace(",", ".", regex=False)
         .replace({"": np.nan})
         .astype(float)
    )

def logistic_G(s: pd.Series, gamma: float, c: float) -> pd.Series:
    return 1.0 / (1.0 + np.exp(-gamma * (s - c)))

def regimes_from_G(dates, G, cut_low: float, cut_high: float):
    """
    Build contiguous regime spans from G(t) with quarterly alignment.
    Accepts dates as Series, list, DatetimeIndex — anything pd.to_datetime handles.
    """
    # Ensure 1-D Series
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    G     = pd.Series(G).reset_index(drop=True)

    # Map each timestamp to its quarter START timestamp
    q_start = (pd.PeriodIndex(dates, freq="Q")
                 .asfreq("Q-DEC")
                 .to_timestamp(how="start"))

    # Series indexed by quarter starts
    s = pd.Series(G.values, index=q_start).sort_index()

    # Fill to all quarters in range (forward-fill across gaps)
    full_q = pd.period_range(q_start.min(), q_start.max(), freq="Q")
    idx    = full_q.asfreq("Q-DEC").to_timestamp(how="start")
    s      = s.reindex(idx).ffill()

    # Label regimes from G
    def lab(v):
        if pd.isna(v): return "NA"
        return "LS" if v < cut_low else ("IS" if v < cut_high else "HS")

    labs = s.map(lab)

    # Build contiguous spans
    spans = []
    cur_label = None
    seg_start = None
    for t, L in labs.items():
        if L != cur_label:
            if cur_label is not None:
                spans.append((seg_start, t, cur_label))
            cur_label = L
            seg_start = t
    if cur_label is not None:
        # end at next quarter start
        t_end = (labs.index[-1].to_period("Q") + 1).asfreq("Q-DEC").to_timestamp(how="start")
        spans.append((seg_start, t_end, cur_label))

    # Small epsilon to avoid white seams at boundaries
    eps = pd.Timedelta(hours=12)
    return [(a - eps, b + eps, r) for a, b, r in spans]
def add_year_ticks(ax, tmin, tmax):
    periods = pd.period_range(tmin, tmax, freq="Q")
    years   = periods.to_timestamp(how="start")
    is_q1   = periods.quarter == 1
    major   = years[is_q1]
    ax.set_xlim(tmin - pd.Timedelta(days=15), tmax + pd.Timedelta(days=15))
    ax.xaxis.set_major_locator(mticker.FixedLocator(mdates.date2num(major)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", which="major", length=8, width=1.1)

# ---- Load ----
fsi = pd.read_csv(FSI_CSV)
date_col = pick_col(fsi, ["date", "quarter", "period", "dates"], fallback_contains=["date"])
# include 's_lag3' in the preferred list; also robust contains tokens:
fsi_col  = pick_col(
    fsi,
    ["FSI", "FSI_lag3", "fsi_lag3", "fsi", "s_lag3", "s"],
    fallback_contains=["fsi", "lag3", "s_lag3", "s_lag", "s("]
)

fsi["__date"] = fsi[date_col].apply(parse_date)
fsi = fsi.dropna(subset=["__date"]).sort_values("__date").reset_index(drop=True)
fsi[fsi_col] = normalize_numeric(fsi[fsi_col])

# use existing G if present, else compute
g_col = next((c for c in fsi.columns if c.strip().upper() == "G"), None)
fsi["G"] = normalize_numeric(fsi[g_col]) if g_col else logistic_G(fsi[fsi_col], gamma, c_thr)

# effective coefficients
fsi["const_eff"] = alpha0 + beta0 * fsi["G"]
fsi["dfsi_eff"]  = alpha1 + beta1 * fsi["G"]
fsi["dy_eff"]    = alpha2 + beta2 * fsi["G"]

spans = regimes_from_G(fsi["__date"], fsi["G"], G_CUT_LOW, G_CUT_HIGH)

# ---- FIG 1: FSI with regime spans ----
fig1, ax1 = plt.subplots(figsize=(18, 7))
for t0, t1, lab in spans:
    ax1.axvspan(t0, t1, facecolor=REGION_COLOR.get(lab, REGION_COLOR["NA"]), edgecolor="none", zorder=0)
ax1.plot(fsi["__date"], fsi[fsi_col], color="black", lw=2.0, zorder=2)
ax1.set_title(r"Regime Spans over $FSI_{t-3}$", pad=10)
ax1.set_ylabel(r"$FSI_{t-3}$ (level)")
add_year_ticks(ax1, fsi["__date"].min(), fsi["__date"].max())
ax1.grid(True, axis="y", alpha=0.25, zorder=1)
fig1.tight_layout()
fig1.savefig(OUT_FSI_JPG, dpi=300, format="jpeg")

# ---- FIG 2: Effective coefficients (3 panels) ----
fig2, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
titles = [
    r"Effective Constant in $\Delta y_t$",
    r"Effective Coefficient on $\Delta s_{t-1}$ (FSI change)",
    r"Effective Coefficient on $\Delta y_{t-1}$ (Lagged GDP growth)"
]
cols = ["const_eff", "dfsi_eff", "dy_eff"]

for ax, ttl, col in zip(axes, titles, cols):
    for t0, t1, lab in spans:
        ax.axvspan(t0, t1, facecolor=REGION_COLOR.get(lab, REGION_COLOR["NA"]), edgecolor="none", zorder=0)
    ax.axhline(0.0, ls="--", lw=1.0, color="gray", alpha=0.7, zorder=1)
    ax.plot(fsi["__date"], fsi[col], color="black", lw=2.0, zorder=2)
    ax.set_title(ttl, pad=6)
    ax.grid(True, axis="y", alpha=0.25, zorder=1)

add_year_ticks(axes[-1], fsi["__date"].min(), fsi["__date"].max())
axes[-1].set_xlabel("Quarter")
fig2.tight_layout()
fig2.savefig(OUT_COEF_JPG, dpi=300, format="jpeg")

print(f"Saved:\n - {OUT_FSI_JPG.resolve()}\n - {OUT_COEF_JPG.resolve()}")