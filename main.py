# ====== RGDP fit + decomposition figure (STR model) ======
# Requirements: pandas, numpy, matplotlib
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.ticker as mticker

# ----------------------------
# User settings
# ----------------------------
CSV_PATH = "rgdp_str_table.csv"   # <-- put your CSV path here

# Regime cutoffs for G (FSI level lagged 3q) — customize to your latest thresholds
G_CUT_LOW  = 0.3422  # LS / IS boundary
G_CUT_HIGH = 0.9655   # IS / HS boundary

# y-limits (optional). Set to None for auto.
YLIM_TOP    = None
YLIM_BOTTOM = None

# **NEW**: bar width in days (affects BOTH panes)
BAR_WIDTH_DAYS = 75  # try 60–90 if you want even thicker

# Save output
OUT_PNG = Path("rgdp_fit_decomp.png")

# ----------------------------
# Column mapping (adjust if your headers differ)
# ----------------------------
COLS = {
    "date": "dates",
    "rgdp": "RGDP",                  # UAH bn (levels)
    "dlog": "DLOG(RGDP)",            # log-delta (model dep.var)
    "g": "G",                        # transition index in [0,1]
    "fsi": "FSI",
    "dfsi": "D(FSI)",
    "fsi_input": "FSI Input",        # model contribution in log-units
    "inertia": "GDP Inertia",        # model contribution in log-units
    "intercept": "Intercept",        # model contribution in log-units
    "crisis": "Crises dummies",      # model contribution in log-units (sum of WAR/COVID dummies effect)
    "total": "total input"           # sum of contributions (log-units), i.e., predicted DLOG
}

# ----------------------------
# Load & prepare
# ----------------------------
df = pd.read_csv(CSV_PATH)

# Normalize headers (strip spaces)
df.columns = [c.strip() for c in df.columns]

# Dates → quarterly period ends (so Q1 ticks land cleanly)
def parse_date(x):
    x = str(x).strip()
    if "Q" in x.upper():
        p = pd.Period(x, freq="Q")
        return p.asfreq("Q", "end").to_timestamp()
    else:
        return pd.to_datetime(x, errors="coerce")

print("Parsing dates...")
df["__date"] = df[COLS["date"]].apply(parse_date)
df = df.sort_values("__date").reset_index(drop=True)

# Numeric columns (coerce commas if present)
for key in ["rgdp","dlog","g","fsi","dfsi","fsi_input","inertia","intercept","crisis","total"]:
    if COLS[key] in df.columns:
        df[COLS[key]] = (
            df[COLS[key]]
            .astype(str).str.replace(",", ".", regex=False)
            .replace({"": np.nan})
            .astype(float)
        )

# Drop rows without RGDP or date
df = df.dropna(subset=["__date", COLS["rgdp"]]).copy()

# Previous RGDP level (for offsetting bars and converting log-deltas to level deltas)
df["RGDP_prev"] = df[COLS["rgdp"]].shift(1)

# Convert log-delta contributions to level deltas using exact exponential form
for part in ["fsi_input","inertia","intercept","crisis","total"]:
    if COLS[part] in df.columns:
        # exact nonlinear transformation
        df[f"{part}_bn"] = df["RGDP_prev"] * (np.exp(df[COLS[part]]) - 1.0)
    else:
        df[f"{part}_bn"] = 0.0

# Predicted ΔRGDP (UAH bn) and fitted RGDP level
df["dRGDP_pred_bn"] = df["total_bn"]
df["RGDP_pred"]     = df["RGDP_prev"] + df["dRGDP_pred_bn"]

# ----------------------------
# Regime classification by G
# ----------------------------
def regime_from_g(g):
    if pd.isna(g):
        return "NA"
    if g < G_CUT_LOW:
        return "LS"
    elif g < G_CUT_HIGH:
        return "IS"
    else:
        return "HS"

df["regime"] = df[COLS["g"]].apply(regime_from_g)

# Find contiguous spans by regime for shading
spans = []
if not df.empty:
    cur_reg = df["regime"].iloc[0]
    start = df["__date"].iloc[0]
    for i in range(1, len(df)):
        if df["regime"].iloc[i] != cur_reg:
            spans.append((start, df["__date"].iloc[i], cur_reg))
            cur_reg = df["regime"].iloc[i]
            start = df["__date"].iloc[i]
    spans.append((start, df["__date"].iloc[-1] + pd.Timedelta(days=90), cur_reg))

REGION_COLOR = {
    "LS": (0.85, 0.95, 0.85, 0.35),  # soft green
    "IS": (0.99, 0.95, 0.80, 0.35),  # soft yellow
    "HS": (0.98, 0.85, 0.85, 0.35),  # soft red
    "NA": (0.92, 0.92, 0.92, 0.30),
}

# ----------------------------
# Figure
# ----------------------------
fig = plt.figure(figsize=(16, 10))
gs  = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1, 2], hspace=0.20)

ax_top = fig.add_subplot(gs[0, 0])
ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)

# Background regime spans
for (t0, t1, reg) in spans:
    ax_top.axvspan(t0, t1, facecolor=REGION_COLOR.get(reg, REGION_COLOR["NA"]), edgecolor="none", zorder=0)
    ax_bot.axvspan(t0, t1, facecolor=REGION_COLOR.get(reg, REGION_COLOR["NA"]), edgecolor="none", zorder=0)

# ----------------------------
# TOP: Actual vs Fitted (levels), bars are ΔRGDP offset by RGDP(t-1)
# ----------------------------
t = df["__date"]
actual = df[COLS["rgdp"]]

# bar bottoms = previous level if predicted delta >= 0, else previous level + delta
d_pred = df["dRGDP_pred_bn"].fillna(0.0)
prev   = df["RGDP_prev"]
pred  = df["RGDP_pred"]
print("PREV", df[["dates", "RGDP_prev"]].tail(15))
print("PRED", df[["dates", "RGDP_pred"]].tail(15))
print("D_PRED", df[["dates", "dRGDP_pred_bn"]].tail(15))

bar_bottom = np.where(d_pred >= 0, prev, pred)
bar_height = np.abs(d_pred)

# **Use a real time width so it scales on a datetime x-axis**
BAR_WIDTH = pd.Timedelta(days=BAR_WIDTH_DAYS)

# Bars
pos = d_pred >= 0
neg = ~pos
ax_top.bar(t[pos], bar_height[pos], bottom=bar_bottom[pos], width=BAR_WIDTH, align="center",
           linewidth=0.8, edgecolor="black", color="#4CAF50", label="Predicted ΔRGDP (↑)")
ax_top.bar(t[neg], bar_height[neg], bottom=bar_bottom[neg], width=BAR_WIDTH, align="center",
           linewidth=0.8, edgecolor="black", color="#E53935", label="Predicted ΔRGDP (↓)")

# Lines
ax_top.plot(t, actual, color="black", linewidth=2.0, label="Actual RGDP (UAH bn)")
if "RGDP_pred" in df:
    ax_top.plot(t, df["RGDP_pred"], linestyle="--", linewidth=2.0, label="Predicted RGDP (line)")

# # Add value markers on top pane for Actual and Predicted
# ax_top.scatter(t, actual, color="black", s=30, zorder=5, label="_nolegend_")        # actual markers
# ax_top.scatter(t, df["RGDP_pred"], color="deepskyblue", s=30, zorder=5, label="_nolegend_")  # predicted markers


# Aesthetics (top)
ax_top.set_ylabel("RGDP (UAH bn)")
if YLIM_TOP:
    ax_top.set_ylim(*YLIM_TOP)
ax_top.legend(loc="lower left", frameon=True)
ax_top.grid(True, axis="y", alpha=0.25)

# Auto-extend the lower bound slightly to avoid bar clipping
ymin, ymax = ax_top.get_ylim()
ax_top.set_ylim(ymin * 0.95, ymax * 1.02)  # 5% padding below, 2% above

# ----------------------------
# BOTTOM: Decomposition (stacked, UAH bn) + dashed total
# ----------------------------
parts = [
    ("fsi_input_bn",  "FSI input"),
    ("inertia_bn",    "Inertia"),
    ("intercept_bn",  "Intercept"),
    ("crisis_bn",     "Crisis shocks"),
]
colors = ["#FFC107", "#1E88E5", "#7CB342", "#E53935"]

# Build 4 × T matrix (rows = components)
data = np.vstack([df[col].fillna(0.0).to_numpy() for col, _ in parts])
n_rows, n_cols = data.shape

def cum_bottom(arr, *, clip_kwargs):
    """Cumulative baseline for stacking (pos/neg treated separately)."""
    clipped = arr.clip(**clip_kwargs)
    csum = np.cumsum(clipped, axis=0)
    out = np.zeros_like(arr)
    out[1:] = csum[:-1]
    return out

# Separate positive and negative baselines
bottom_pos = cum_bottom(data, clip_kwargs=dict(min=0))
bottom_neg = cum_bottom(data, clip_kwargs=dict(max=0))
bottom_stack = np.where(data >= 0, bottom_pos, bottom_neg)

# Draw each stacked component — no black edge
for i, ((col, lab), colr) in enumerate(zip(parts, colors)):
    ax_bot.bar(
        t, data[i], bottom=bottom_stack[i],
        width=BAR_WIDTH, align="center",
        color=colr, label=lab, zorder=3
    )

# Overlay dashed total ΔRGDP
ax_bot.plot(
    t, df["total_bn"], linestyle=(0, (1, 1)), linewidth=2.0,
    color="black", label="Predicted ΔRGDP (total)", zorder=4
)

# Aesthetics
ax_bot.set_ylabel("ΔRGDP decomposition (UAH bn)")
if YLIM_BOTTOM:
    ax_bot.set_ylim(*YLIM_BOTTOM)
ax_bot.grid(True, axis="y", alpha=0.25, zorder=0)
ax_bot.legend(loc="lower left", ncol=2, frameon=True)

# ----------------------------
# X-axis ticks: major at Q1 every year; minor at each quarter
# ----------------------------
periods = pd.period_range(df["__date"].min(), df["__date"].max(), freq="Q")
years = periods.to_timestamp(how="start")
is_q1 = periods.quarter == 1
major_locs = years[is_q1]
minor_locs = years

for ax in (ax_top, ax_bot):
    ax.set_xlim(df["__date"].min() - pd.Timedelta(days=30),
                df["__date"].max() + pd.Timedelta(days=30))
    ax.xaxis.set_major_locator(mticker.FixedLocator(mdates.date2num(major_locs)))
    ax.xaxis.set_minor_locator(mticker.FixedLocator(mdates.date2num(minor_locs)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", which="major", length=8, width=1.25)
    ax.tick_params(axis="x", which="minor", length=3, width=0.8)
    for label in ax.get_xticklabels(minor=True):
        label.set_visible(False)

# Titles
ax_top.set_title("RGDP: Actual vs Fitted (STR), with regime shading", pad=6)
ax_bot.set_title("Predicted ΔRGDP decomposition (FSI / Inertia / Intercept / Crisis)", pad=6)

# Add yearly vertical grid lines (at major tick positions)
for ax in (ax_top, ax_bot):
    for year in major_locs:
        ax.axvline(year, color="gray", linestyle="--", linewidth=0.8, alpha=0.4, zorder=0)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=180)
print(f"Saved: {OUT_PNG.resolve()}")