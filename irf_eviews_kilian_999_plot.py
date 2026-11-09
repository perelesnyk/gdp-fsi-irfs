# -*- coding: utf-8 -*-
# IRF plot from EViews table with Kilian unbiased bootstrap S.E. (999 reps)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

OUT_PNG = Path("irf_eviews_kilian_999.png")
OUT_JPG = Path("irf_eviews_kilian_999.jpg")

# Horizon (periods 1..10)
periods = np.arange(1, 11)

# ----------------------------
# Response of DLOG(FSI)
# ----------------------------
resp_fsi_to_fsi = np.array([
    0.7468343288944346,
    -0.09009094575836439,
    -0.05114855622987277,
    -0.1051706897506362,
    0.02170433706654839,
    0.01434430825194058,
    0.01137451487086941,
    -0.006272565626513196,
    -0.003787164434304365,
    0.0003891482801664978,
])

se_fsi_to_fsi = np.array([
    0.005675837273440066,
    0.00685343576627756,
    0.006262555772350266,
    0.0041641635389987,
    0.004501071283670959,
    0.001416034624402953,
    0.0007052912919579732,
    0.0006861328625358534,
    0.0003814469834054076,
    0.0001527683522831506,
])

resp_fsi_to_rgdp = np.array([
    0.0,
    -0.02667536909806051,
    0.005955549056132425,
    -0.01127059222768194,
    0.02780516839622913,
    -0.002975530279268566,
    -0.003074867120355616,
    -0.003666921497853361,
    -0.000006706539831111765,
    0.001171767243297905,
])

se_fsi_to_rgdp = np.array([
    0.0,
    0.001686560095258328,
    0.001722730994305369,
    0.001436906507444738,
    0.001274582941386044,
    0.0002251948439630857,
    0.0001210735192438471,
    0.00008976826212124925,
    0.00004238333972322526,
    0.00002003821015561783,
])

# ----------------------------
# Response of DLOG(RGDP_SA3)
# ----------------------------
resp_rgdp_to_fsi = np.array([
    -0.002965648775116871,
    -0.004373773192457128,
    -0.002297783205209463,
    -0.0009881152731903396,
    0.0002547712736611871,
    0.001354037683618243,
    0.0003062549324756067,
    -0.0001577265225852801,
    -0.0002418407177537356,
    -0.0000679302036533894,
])

se_rgdp_to_fsi = np.array([
    0.000005688959453653872,
    0.00000470904607801158,
    0.000004625335509693047,
    0.000003299588576958057,
    0.00000287129040224887,
    0.0000007791785447935234,
    0.0000003143349343523781,
    0.0000002080972920155701,
    0.0000001087679274834211,
    0.00000005763550166049805,
])

resp_rgdp_to_rgdp = np.array([
    0.01873923064651505,
    -0.002607755676767437,
    -0.002280991697156504,
    -0.0001548327945276272,
    -0.0005343034248233995,
    0.0003060926802036527,
    0.00009045576223766399,
    -0.00006541683475698118,
    -0.000007411127692425441,
    0.00002601664317079966,
])

se_rgdp_to_rgdp = np.array([
    0.000003766852873124809,
    0.000001257233968045383,
    0.000001160387168282307,
    0.00000094226978888675,
    0.000000814543682749584,
    0.0000001808402878305767,
    0.0000001064170240393039,
    0.000000041868828742225,
    0.00000002523919862622095,
    0.00000001241260631673264,
])


def plot_irf(ax, y, se, title, color):
    lower_68 = y - se
    upper_68 = y + se
    lower_95 = y - 1.96 * se
    upper_95 = y + 1.96 * se

    ax.fill_between(periods, lower_68, upper_68, alpha=0.2, color=color, label="68% CI")
    ax.plot(periods, y, lw=2.5, color=color, label="Point estimate")
    ax.plot(periods, lower_95, ls="--", lw=1, color=color, alpha=0.7, label="95% CI")
    ax.plot(periods, upper_95, ls="--", lw=1, color=color, alpha=0.7)

    ax.axhline(0, color="black", lw=0.8)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlim(1, 10)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Period")
    ax.set_ylabel("Response")
    ax.legend(frameon=True, fontsize=9, loc="best")


fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)

plot_irf(
    axes[0, 0],
    resp_fsi_to_fsi,
    se_fsi_to_fsi,
    "Response of DLOG(FSI) to DLOG(FSI) shock",
    "#2E7D32",
)

plot_irf(
    axes[0, 1],
    resp_fsi_to_rgdp,
    se_fsi_to_rgdp,
    "Response of DLOG(FSI) to DLOG(RGDP_SA3) shock",
    "#2E7D32",
)

plot_irf(
    axes[1, 0],
    resp_rgdp_to_fsi,
    se_rgdp_to_fsi,
    "Response of DLOG(RGDP_SA3) to DLOG(FSI) shock",
    "#C62828",
)

plot_irf(
    axes[1, 1],
    resp_rgdp_to_rgdp,
    se_rgdp_to_rgdp,
    "Response of DLOG(RGDP_SA3) to DLOG(RGDP_SA3) shock",
    "#C62828",
)

fig.suptitle(
    "VAR Impulse Responses with Kilian Unbiased Bootstrap S.E. (999 reps)",
    fontsize=13,
    fontweight="bold",
)

plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
plt.savefig(OUT_JPG, dpi=180, bbox_inches="tight", format="jpg", pil_kwargs={"quality": 95})
print(f"Saved plot → {OUT_PNG.resolve()}")
print(f"Saved plot → {OUT_JPG.resolve()}")
plt.close(fig)
