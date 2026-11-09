# fsi_decomp.py
# Works with older Matplotlib: quarterly ticks handled manually

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from matplotlib.ticker import FixedLocator

CSV_FILE = "FSI_decomp.csv"
YLABEL = "Contribution"
OUT_JPG = "stacked_area_primary_quarters.jpg"
FIGSIZE = (11, 5)
ALPHA = 0.9
LEGEND_LOC = "upper left"

def parse_dates(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", dayfirst=True)
    if dt.notna().any():
        return dt
    def q_to_ts(x):
        if not isinstance(x, str): return pd.NaT
        v = x.strip().upper()
        m = re.match(r"^(\d{4})[-_ ]?Q([1-4])$", v)
        if not m: return pd.NaT
        yr, q = int(m.group(1)), int(m.group(2))
        try: return pd.Period(year=yr, quarter=q, freq="Q").end_time
        except: return pd.NaT
    qparsed = series.apply(q_to_ts)
    if qparsed.notna().any():
        return qparsed
    return dt

def coerce_numeric_col(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace("\u00A0", " ").str.strip()
    s = s.str.replace(r"\s+", "", regex=True)
    s = s.str.replace(r"(?<=\d),(?=\d)", ".", regex=True)
    s = s.str.replace(r",(?!\d)", "", regex=True)
    s = s.str.replace(r"[^0-9\.\-eE]", "", regex=True)
    s = s.replace({"": np.nan, ".": np.nan, "-": np.nan})
    return pd.to_numeric(s, errors="coerce")

def main():
    path = Path(CSV_FILE)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path.resolve()}")

    df = pd.read_csv(path, dtype=str)
    date_col = df.columns[0]
    df[date_col] = parse_dates(df[date_col])
    if df[date_col].notna().sum() == 0:
        raise ValueError(f"Could not parse dates from '{date_col}'.")

    comp_cols = [c for c in df.columns if c != date_col]
    for c in comp_cols:
        df[c] = coerce_numeric_col(df[c])
    comp_cols = [c for c in comp_cols if df[c].notna().sum() >= max(2, int(0.3 * len(df)))]
    if len(comp_cols) < 2:
        raise ValueError(f"Need at least two numeric component columns; found {comp_cols}.")

    df = df.dropna(subset=[date_col]).sort_values(by=date_col).reset_index(drop=True)
    x = df[date_col]
    df_pos = df[comp_cols].clip(lower=0)
    df_neg = df[comp_cols].clip(upper=0)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # 🎨 Bright base colors (red, green, blue, purple, orange, cyan)
    base_colors = ["#e41a1c", "#4daf4a", "#377eb8", "#984ea3", "#ff7f00", "#c0c0c0"]
    colors = [base_colors[i % len(base_colors)] for i in range(len(comp_cols))]

    ax.stackplot(x, df_pos.T, labels=comp_cols, colors=colors, alpha=ALPHA)
    ax.stackplot(x, df_neg.T, colors=colors, alpha=ALPHA)

    # === Quarterly ticks manually ===
    quarters = pd.period_range(start=x.min(), end=x.max(), freq="Q")
    tick_dates = quarters.to_timestamp(how="end")
    ax.xaxis.set_major_locator(FixedLocator(tick_dates.map(lambda d: d.toordinal())))
    ax.set_xticks(tick_dates)
    ax.set_xticklabels([f"{p.year} Q{p.quarter}" for p in quarters], rotation=90, ha="right")

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Quarter")
    ax.set_ylabel(YLABEL)
    ax.grid(True, linewidth=0.5, alpha=0.5)
    ax.legend(loc=LEGEND_LOC, ncol=2, frameon=False)
    plt.tight_layout()

    fig.savefig(OUT_JPG, dpi=300, format="jpeg")
    plt.show()
    print(f"Saved: {OUT_JPG}")

if __name__ == "__main__":
    main()