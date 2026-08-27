#!/usr/bin/env python
"""
plot_causal_rollouts.py
=======================
Reproduce the two causal SAE-dial rollout figures for the Climate-XAI paper:

  1. Concept 99   (Matryoshka L8, GLOBAL AR driver)   -> dial_99_reversion.png
  2. Concept 1592 (Standard   L8, W. N. America AR)    -> dial_1592_panels.png

Each figure shows a 5-day GraphCast rollout in which one SAE concept is dialed by
a feature multiplier (beta) at processor layer 8 at every autoregressive step,
tracking physically linked AR variables (IVT, water vapor, wind, precipitation).

RUN (on the HPC, where PATCH_DIR lives):
  module load cuda/12.8.0
  source .../graphcast_sae_env/bin/activate
  python src/patching/plot_causal_rollouts.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
from datetime import datetime, timedelta

PATCH_DIR = "/scratch/euh7ys/climate_xai/patching"
CMAP = cm.RdBu_r
STEP_HOURS = 6
LANDFALL = (datetime(2021, 11, 14, 12), datetime(2021, 11, 16, 0))

plt.rcParams.update({
    "font.size": 16, "axes.labelsize": 17, "xtick.labelsize": 14,
    "ytick.labelsize": 14, "legend.fontsize": 15,
})


def _load(npz_name):
    d = np.load(f"{PATCH_DIR}/{npz_name}")
    b = d["betas"]; m = d["m"]
    cols = [str(x) for x in d["cols"]]
    init = datetime.strptime(str(d["init"]), "%Y-%m-%dT%H:%M")
    dates = [init + timedelta(hours=STEP_HOURS * (t + 1)) for t in range(m.shape[1])]
    return b, m, cols, dates


def _col(cols, key):
    for i, c in enumerate(cols):
        if key in c.lower():
            return i
    raise ValueError(f"no column matching '{key}' in {cols}")


def _precip_scale(m, pcol):
    return 1000.0 if np.nanmax(m[:, :, pcol]) < 0.1 else 1.0


def _style_panel(a, letter, ylabel, bottom_row):
    a.set_ylabel(ylabel)
    a.grid(alpha=0.3)
    a.text(-0.06, 1.05, f"{letter}.", transform=a.transAxes,
           fontweight="bold", fontsize=19, ha="right", va="bottom")
    a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    if bottom_row:
        a.set_xlabel("Valid Date")
        a.tick_params(axis="x", rotation=30)
    else:
        a.tick_params(labelbottom=False)


def _colorbar(fig, norm):
    sm = cm.ScalarMappable(norm=norm, cmap=CMAP)
    sm.set_array([])
    cax = fig.add_axes([0.895, 0.27, 0.02, 0.46])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Feature Multiplier β")
    cb.ax.axhline(1.0, color="k", ls="--", lw=2)
    return cb


def _draw_lines(ax, dates, b, m, COL, scale, norm):
    base = None
    for j, col in enumerate(COL):
        for i, bb in enumerate(b):
            y = m[i, :, col] * scale[j]
            if abs(bb - 1.0) < 1e-9:
                base, = ax[j].plot(dates, y, "k--", lw=2.4, zorder=6)
            else:
                ax[j].plot(dates, y, color=CMAP(norm(bb)), lw=2.1)
    return base


def plot_99_global(out="dial_99_reversion.png", letters="ABCD"):
    b, m, cols, dates = _load("dial_99_rollout.npz")
    ylab = ["Within-AR IVT (kg m$^{-1}$ s$^{-1}$)",
            "Within-AR Water Vapor (kg m$^{-2}$)",
            "Within-AR 10 m Wind (m s$^{-1}$)",
            "Within-AR Precipitation (mm)"]
    COL = [_col(cols, k) for k in ("ivt", "iwv", "wind", "precip")]
    scale = [1, 1, 1, _precip_scale(m, _col(cols, "precip"))]
    norm = TwoSlopeNorm(vmin=float(b.min()), vcenter=1.0, vmax=float(b.max()))
    fig, ax = plt.subplots(2, 2, figsize=(14, 11)); ax = ax.ravel()
    base = _draw_lines(ax, dates, b, m, COL, scale, norm)
    for j in range(4):
        _style_panel(ax[j], letters[j], ylab[j], bottom_row=j in (2, 3))
    fig.subplots_adjust(left=0.08, right=0.87, top=0.88, bottom=0.15, hspace=0.24, wspace=0.22)
    _colorbar(fig, norm)
    fig.legend([base], ["β = 1 (Baseline)"], loc="lower center",
               bbox_to_anchor=(0.5, -0.02), frameon=True)
    fig.suptitle("Concept 99's Handle on Global ARs", fontsize=22, fontweight="bold", y=0.985)
    fig.savefig(f"{PATCH_DIR}/{out}", dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig); print("saved", out)


def plot_1592_regional(out="dial_1592_panels.png"):
    b, m, cols, dates = _load("dial_1592_rollout.npz")
    ylab = ["Peak IVT (kg m$^{-1}$ s$^{-1}$)",
            "Integrated Water Vapor (kg m$^{-2}$)",
            "Max 10 m Wind (m s$^{-1}$)",
            "6 h Precipitation (mm)"]
    COL = [_col(cols, k) for k in ("peak", "iwv", "wind", "precip")]
    scale = [1, 1, 1, _precip_scale(m, _col(cols, "precip"))]
    norm = TwoSlopeNorm(vmin=float(b.min()), vcenter=1.0, vmax=float(b.max()))
    fig, ax = plt.subplots(2, 2, figsize=(14, 11)); ax = ax.ravel()
    base = _draw_lines(ax, dates, b, m, COL, scale, norm)
    for j in range(4):
        ax[j].axvspan(*LANDFALL, color="0.85", zorder=0)
        _style_panel(ax[j], "ABCD"[j], ylab[j], bottom_row=j in (2, 3))
    fig.subplots_adjust(left=0.08, right=0.87, top=0.88, bottom=0.15, hspace=0.24, wspace=0.22)
    _colorbar(fig, norm)
    lf_patch = Patch(facecolor="0.85", edgecolor="none")
    fig.legend([base, lf_patch], ["β = 1 (Baseline)", "AR Landfall Window"],
               loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=True)
    fig.suptitle("Concept 1592 Dials a Western North America AR", fontsize=23, fontweight="bold", y=0.985)
    fig.savefig(f"{PATCH_DIR}/{out}", dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig); print("saved", out)


if __name__ == "__main__":
    plot_99_global()
    plot_1592_regional()
