"""Combine meanivt_99_3153.png (split -> A, B) with ivt_dist_99_3153_big.png
(centered below -> C). Labels sit ABOVE the panels, outside the images."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = "/scratch/euh7ys/climate_xai/plots/"
OUT = P + "combined_ivt_99_3153.png"

maps = plt.imread(P + "meanivt_99_3153.png")
dist = plt.imread(P + "ivt_dist_99_3153_big.png")

h, w = maps.shape[:2]
A_img = maps[:, : w // 2]
B_img = maps[:, w // 2 :]

def trim(img, thresh=0.995):
    g = img[..., :3].min(2)
    rows = np.where((g < thresh).any(1))[0]; cols = np.where((g < thresh).any(0))[0]
    return img[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]

A_img, B_img, C_img = trim(A_img), trim(B_img), trim(dist)

PW = 7.0
ha = PW * A_img.shape[0] / A_img.shape[1]
hb = PW * B_img.shape[0] / B_img.shape[1]
hc = PW * C_img.shape[0] / C_img.shape[1]
top_h = max(ha, hb)
fig = plt.figure(figsize=(2 * PW + 0.4, top_h + hc + 1.6))
gs = fig.add_gridspec(2, 4, height_ratios=[top_h, hc], hspace=0.08, wspace=0.04,
                      left=0.01, right=0.99, top=0.94, bottom=0.01)

ax_a = fig.add_subplot(gs[0, 0:2])
ax_b = fig.add_subplot(gs[0, 2:4])
ax_c = fig.add_subplot(gs[1, 1:3])
for ax, img, lab in ((ax_a, A_img, "A."), (ax_b, B_img, "B."), (ax_c, C_img, "C.")):
    ax.imshow(img); ax.axis("off")
    ax.text(0.0, 1.02, lab, transform=ax.transAxes, fontsize=24, fontweight="bold",
            va="bottom", ha="left")

fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("WROTE:", OUT)
