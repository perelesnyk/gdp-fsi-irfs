# fsi_historical_stair_safe.py
import textwrap
import datetime as _dt
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.offsetbox import AnnotationBbox, TextArea
import matplotlib.lines as mlines

# ------------------- INPUTS -------------------
DATA_FILE = "FSI_daily.csv"   # must have one date-like column + one FSI column

EVENTS = [
    ("2008-09-15", "Lehman Brothers bankruptcy"),
    ("2008-11-05", "Adoption of anti-crisis legislation"),
    ("2014-01-19", "First violent clashes during\nthe Revolution of Dignity"),
    ("2014-03-27", "Agreement under IMF program"),
    ("2014-04-12", "Russian invasion of Donbas"),
    ("2015-03-02", "Delta Bank insolvency"),
    ("2015-03-12", "Start of official talks on external debt restructuring"),
    ("2018-11-25", "Russian armed aggression in the Kerch Strait"),
    ("2020-03-12", "Start of COVID-19 lockdown"),
    ("2022-02-24", "Russian full-scale invasion"),
    ("2022-10-10", "Start of major attacks on Ukrainian energy infrastructure"),
    ("2024-08-09", "Ukraine external debt \n restructuring 2024"),
]
OVERRIDES = {
    "2008-09-15": dict(row=0, side="left",  nudge=30),  # Lehman (bottom)
    "2008-11-05": dict(row=2, side="left",  nudge=30),  # SBA on a higher tier, shifted right
    # 2014–15 “stair” (topmost among the below-area rows)
    "2014-01-19": dict(row=4, side="right", nudge=30),
    "2014-03-27": dict(row=0, side="left", nudge=-500),
    "2014-04-12": dict(row=2, side="left", nudge=30),
    "2015-03-02": dict(row=4, side="left", nudge=30),
    "2015-03-12": dict(row=6, side="left", nudge=30),
    # rest
    "2018-11-25": dict(row=0, side="left",  nudge=30),
    "2020-03-12": dict(row=4, side="left",  nudge=-400),
    "2022-02-24": dict(row=6, side="left",  nudge=-500),
    "2022-10-10": dict(row=2, side="right", nudge=0),
    "2024-08-09": dict(row=4, side="right",  nudge=-300),
}
# ------------------------------------------------

# Colors
LINE_COLOR_RAW    = "#0b3d91"  # dark blue
LINE_COLOR_SMOOTH = "#d97706"  # orange
MARKER_COLOR      = "#741a2c"  # maroon
ARROW_COLOR       = "#777777"

def guess_cols(df: pd.DataFrame):
    d = [c for c in df.columns if c.lower() in {"date","time","period"} or "date" in c.lower()]
    if not d:
        raise ValueError("Could not find a date column (name must contain 'date').")
    date_col = d[0]
    v = [c for c in df.columns if c != date_col]
    v_fsi = [c for c in v if "fsi" in c.lower()]
    value_col = v_fsi[0] if v_fsi else v[0]
    return date_col, value_col

def load_series():
    df = pd.read_csv(DATA_FILE)
    date_col, value_col = guess_cols(df)
    print(f"Using date column: '{date_col}', value column: '{value_col}'")

    # Parse dates (prefer EU day-first)
    dt = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    if dt.isna().all():
        dt = pd.to_datetime(df[date_col], errors="coerce", infer_datetime_format=True)
    df["_date"] = dt
    df = df.dropna(subset=["_date"]).sort_values("_date")

    # Decimal comma → dot, strip stray chars
    df[value_col] = (
        df[value_col]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace(r"[^0-9\.\-]", "", regex=True)
    )
    s = pd.to_numeric(df[value_col], errors="coerce")
    s = pd.Series(s.values, index=df["_date"]).dropna().sort_index()
    return s

def nearest_point(series: pd.Series, t: pd.Timestamp):
    if t in series.index:
        return t, float(series.loc[t])
    pos = series.index.get_indexer([t], method="nearest")[0]
    t0 = series.index[pos]
    return t0, float(series.loc[t0])

def _lookup_override(date_str):
    # exact-date match first; else None
    return OVERRIDES.get(pd.to_datetime(date_str).strftime("%Y-%m-%d"))

def annotate_events_below_edges(ax, s, events, rows=8, wrap_width=34,
                                arrow_color="#888888", box_edge="#666666"):
    """
    Place labels BELOW the plot. Arrow starts from the *true top edge* of each
    AnnotationBbox, computed from its display-space bbox and transformed back
    to data coords.
    """
    events_sorted = sorted(events, key=lambda e: pd.to_datetime(e[0]))

    y_min = float(np.nanmin(s.values))
    y_max = float(np.nanmax(s.values))
    span  = max(1e-9, y_max - y_min)
    top_pad = 0.08 * span
    bot_pad = 1.60 * span
    ax.set_ylim(y_min - bot_pad, y_max + top_pad)

    row_ys = np.linspace(y_min - bot_pad * 0.95, y_min - bot_pad * 0.15, rows)

    for i, (dstr, label) in enumerate(events_sorted):
        t_req = pd.to_datetime(dstr)
        t0 = s.index[s.index.get_indexer([t_req], method="nearest")[0]]
        y0 = float(s.loc[t0])

        hint = OVERRIDES.get(pd.to_datetime(dstr).strftime("%Y-%m-%d"), {})
        row  = int(hint.get("row", i % rows))
        side = hint.get("side", "left")           # "left" or "right"
        nudge_days = int(hint.get("nudge", 80))

        y_box = float(row_ys[max(0, min(rows - 1, row))])

        # Box anchor and placement (in DATA coords)
        if side == "left":
            arrow_x = t0
            box_x   = t0 + pd.Timedelta(days=nudge_days)
            align   = (0.0, 0.5)  # left edge anchored
        else:
            arrow_x = t0
            box_x   = t0 - pd.Timedelta(days=nudge_days)
            align   = (1.0, 0.5)  # right edge anchored

        # 1) Create the annotation box
        txt = TextArea(textwrap.fill(label, width=wrap_width),
                       textprops=dict(color="black", fontsize=9.5))
        ab = AnnotationBbox(
            txt,
            (arrow_x, y0),                 # reference (not used for the line)
            xybox=(box_x, y_box),          # actual box position (data coords)
            xycoords="data",
            boxcoords="data",
            box_alignment=align,
            frameon=True,
            bboxprops=dict(
                boxstyle="round,pad=0.35,rounding_size=0.0",
                fc="white", ec=box_edge, lw=1.0, alpha=0.98
            ),
            arrowprops=None,
            zorder=6,
        )
        ax.add_artist(ab)

        # 2) Force a draw so bbox is computed, then measure box top in DISPLAY coords
        fig = ax.figure
        fig.canvas.draw()  # ensure a renderer exists and layout is realized
        renderer = fig.canvas.get_renderer()
        bbox_disp = ab.get_window_extent(renderer=renderer)

        # 3) Convert the box top (display y) back to DATA y at the arrow's x
        #    (transform an x at arrow_x, and y at bbox.top)
        x_disp_at_arrow = ax.transData.transform((mdates.date2num(arrow_x), y0))[0]
        inv = ax.transData.inverted()
        # Pick a tiny padding above the box in pixels (e.g. 2 px)
        pad_px = 2.0
        y_top_data = inv.transform((x_disp_at_arrow, bbox_disp.y1 + pad_px))[1]

        # 4) Draw the vertical connector from box top to the curve
        ax.add_line(mlines.Line2D(
            [arrow_x, arrow_x], [y_top_data, y0],
            lw=1.2, color=arrow_color, zorder=5, solid_capstyle="round",
        ))

        # 5) Big dot at the box top, arrowhead near the curve
        ax.plot(arrow_x, y_top_data, "o",
                ms=6, color=arrow_color, markeredgecolor="black", zorder=7)

        ax.annotate("",
            xy=(arrow_x, y0),                             # tip at the curve
            xytext=(arrow_x, y_top_data + (y0 - y_top_data) * 0.85),
            arrowprops=dict(
                arrowstyle="-|>", color=arrow_color, lw=1.5,
                shrinkA=0, shrinkB=0, mutation_scale=10,
            ),
            zorder=7,
        )
def make_plot():
    s = load_series()
    s_smooth = s.rolling(30, min_periods=1, center=True).mean()  # optional soft reference

    fig, ax = plt.subplots(figsize=(12.0, 4.6))

    # Quarterly ticks (Jan/Apr/Jul/Oct), vertical labels
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
    ax.tick_params(axis="x", labelsize=9, rotation=90)
    ax.tick_params(axis="y", labelsize=9)

    ax.margins(x=0.01)
    for sp in ax.spines.values():
        sp.set_linewidth(1.0)
    ax.grid(False)
    ax.set_ylabel("")
    ax.set_xlabel("")

   # colors for the two lines
    ax.plot(s.index, s.values, lw=1, color="#0B3C89", label="FSI (daily)")
        # Add horizontal gridlines only above 0
    ax.yaxis.grid(True, which="major", linestyle="--", color="#dddddd", zorder=0)
    ax.axhline(0, color="black", lw=1.0)

    # Remove y-axis tick labels below 0
    yticks = ax.get_yticks()
    ax.set_yticks([y for y in yticks if y >= 0])

    annotate_events_below_edges(ax, s, EVENTS, rows=8) 

    ax.legend(loc="upper right", fontsize=9, frameon=False)

    fig.tight_layout()
    out = "fsi_breakpoints_stair.jpg"
    fig.savefig(out, dpi=300, format="jpeg")
    plt.show()
    print(f"Saved: {out}")

if __name__ == "__main__":
    make_plot()