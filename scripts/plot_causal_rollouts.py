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

INPUT (produced by the dial-rollout SLURM jobs), in PATCH_DIR:
  dial_99_rollout.npz     within-AR (global) means; no landfall window
  dial_1592_rollout.npz   W. N. America regional metrics; AR-landfall window shaded
  each with keys:
    betas (nbeta,)              feature multipliers, includes 1.0 (baseline)
    m     (nbeta, nstep, nvar)  metric trajectories
    cols  (nvar,)               column names (looked up by substring, not index)
    init  (str, "%Y-%m-%dT%H:%M")

OUTPUT (written back into PATCH_DIR):
  dial_99_reversion.png
  dial_1592_panels.png

SHARED STYLE CONVENTIONS (keep all rollout figures consistent):
  - diverging RdBu_r colormap centered on the baseline via TwoSlopeNorm(vcenter=1.0):
    beta<1 (removed) -> blue, beta=1 (baseline) -> white, beta>1 (amplified) -> red.
  - beta=1 baseline drawn as a black dashed line, AND marked on the colorbar with a
    dashed line at 1.0 (so the colorbar's neutral point == the dashed baseline).
  - panel letters bold, uppercase, above-left of each panel (matches all other figures).
  - one baseline legend below the whole figure; dates only on the bottom row.
  - precipitation auto-converted to mm if stored in metres.

RUN (on the HPC, where PATCH_DIR lives):
  module load cuda/12.8.0
  source .../graphcast_sae_env/bin/activate   # any env with numpy + matplotlib
  python scripts/plot_causal_rollouts.py
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

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
PATCH_DIR = "/scratch/euh7ys/climate_xai/patching"
CMAP = cm.RdBu_r
STEP_HOURS = 6                      # GraphCast autoregressive step
LANDFALL = (datetime(2021, 11, 14, 12), datetime(2021, 11, 16, 0))  # 1592 event

plt.rcParams.update({
    "font.size": 16, "axes.labelsize": 17, "xtick.labelsize": 14,
    "ytick.labelsize": 14, "legend.fontsize": 15,
})


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _load(npz_name):
    """Load a rollout npz -> (betas, metrics, column-names, valid-datetimes)."""
    d = np.load(f"{PATCH_DIR}/{npz_name}")
    b = d["betas"]
    m = d["m"]
    cols = [str(x) for x in d["cols"]]
    init = datetime.strptime(str(d["init"]), "%Y-%m-%dT%H:%M")
    dates = [init + timedelta(hours=STEP_HOURS * (t + 1)) for t in range(m.shape[1])]
    return b, m, cols, dates


def _col(cols, key):
    """Index of the first column whose name contains `key` (case-insensitive)."""
    for i, c in enumerate(cols):
        if key in c.lower():
            return i
    raise ValueError(f"no column matching '{key}' in {cols}")


def _precip_scale(m, pcol):
    """1000 if precipitation is stored in metres (max < 0.1), else 1 (already mm)."""
    return 1000.0 if np.nanmax(m[:, :, pcol]) < 0.1 else 1.0


def _style_panel(a, letter, ylabel, bottom_row):
    """Common per-panel styling: y-label, grid, above-left letter, date x-axis."""
    a.set_ylabel(ylabel)
    a.grid(alpha=0.3)
    a.text(-0.06, 1.05, f"{letter}.", transform=a.transAxes,
           fontweight="bold", fontsize=19, ha="right", va="bottom")
    a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    if bottom_row:
        a.set_xlabel("Valid Date")
        a.tick_params(axis="x", rotation=30)
    else:
        a.tick_params(labelbottom=False)   # dates only on the bottom row


def _colorbar(fig, norm):
    """Baseline-centered colorbar with a dashed marker at beta=1."""
    sm = cm.ScalarMappable(norm=norm, cmap=CMAP)
    sm.set_array([])
    cax = fig.add_axes([0.895, 0.27, 0.02, 0.46])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Feature Multiplier β")
    cb.ax.axhline(1.0, color="k", ls="--", lw=2)   # beta=1 == dashed baseline
    return cb


def _draw_lines(ax, dates, b, m, COL, scale, norm):
    """Draw the beta fan on each panel; return the baseline (beta=1) line handle."""
    base = None
    for j, col in enumerate(COL):
        for i, bb in enumerate(b):
            y = m[i, :, col] * scale[j]
            if abs(bb - 1.0) < 1e-9:
                base, = ax[j].plot(dates, y, "k--", lw=2.4, zorder=6)   # baseline
            else:
                ax[j].plot(dates, y, color=CMAP(norm(bb)), lw=2.1)
    return base


# --------------------------------------------------------------------------- #
# Figure 1 - Concept 99, global within-AR dial
# --------------------------------------------------------------------------- #
def plot_99_global(out="dial_99_reversion.png"):
    b, m, cols, dates = _load("dial_99_rollout.npz")
    ylab = ["Within-AR IVT (kg m$^{-1}$ s$^{-1}$)",
            "Within-AR Water Vapor (kg m$^{-2}$)",
            "Within-AR 10 m Wind (m s$^{-1}$)",
            "Within-AR Precipitation (mm)"]
    COL = [_col(cols, k) for k in ("ivt", "iwv", "wind", "precip")]
    scale = [1, 1, 1, _precip_scale(m, _col(cols, "precip"))]
    norm = TwoSlopeNorm(vmin=float(b.min()), vcenter=1.0, vmax=float(b.max()))

    fig, ax = plt.subplots(2, 2, figsize=(14, 11))
    ax = ax.ravel()
    base = _draw_lines(ax, dates, b, m, COL, scale, norm)
    for j in range(4):
        _style_panel(ax[j], "ABCD"[j], ylab[j], bottom_row=j in (2, 3))
    fig.subplots_adjust(left=0.08, right=0.87, top=0.88, bottom=0.15,
                        hspace=0.24, wspace=0.22)
    _colorbar(fig, norm)
    fig.legend([base], ["β = 1 (Baseline)"], loc="lower center",
               bbox_to_anchor=(0.5, -0.02), frameon=True)
    fig.suptitle("Concept 99's Handle on Global ARs",
                 fontsize=22, fontweight="bold", y=0.985)
    fig.savefig(f"{PATCH_DIR}/{out}", dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print("saved", out)


# --------------------------------------------------------------------------- #
# Figure 2 - Concept 1592, W. N. America regional dial (landfall window shaded)
# --------------------------------------------------------------------------- #
def plot_1592_regional(out="dial_1592_panels.png"):
    b, m, cols, dates = _load("dial_1592_rollout.npz")
    ylab = ["Peak IVT (kg m$^{-1}$ s$^{-1}$)",
            "Integrated Water Vapor (kg m$^{-2}$)",
            "Max 10 m Wind (m s$^{-1}$)",
            "6 h Precipitation (mm)"]
    COL = [_col(cols, k) for k in ("peak", "iwv", "wind", "precip")]
    scale = [1, 1, 1, _precip_scale(m, _col(cols, "precip"))]
    norm = TwoSlopeNorm(vmin=float(b.min()), vcenter=1.0, vmax=float(b.max()))

    fig, ax = plt.subplots(2, 2, figsize=(14, 11))
    ax = ax.ravel()
    base = _draw_lines(ax, dates, b, m, COL, scale, norm)
    for j in range(4):
        ax[j].axvspan(*LANDFALL, color="0.85", zorder=0)   # AR landfall window
        _style_panel(ax[j], "ABCD"[j], ylab[j], bottom_row=j in (2, 3))
    fig.subplots_adjust(left=0.08, right=0.87, top=0.88, bottom=0.15,
                        hspace=0.24, wspace=0.22)
    _colorbar(fig, norm)
    lf_patch = Patch(facecolor="0.85", edgecolor="none")
    fig.legend([base, lf_patch], ["β = 1 (Baseline)", "AR Landfall Window"],
               loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=True)
    fig.suptitle("Concept 1592 Dials a Western North America AR",
                 fontsize=23, fontweight="bold", y=0.985)
    fig.savefig(f"{PATCH_DIR}/{out}", dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    plot_99_global()
    plot_1592_regional()
