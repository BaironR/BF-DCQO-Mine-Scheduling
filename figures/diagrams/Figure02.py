"""
Figure 2: qubit ordering (indexing) used in the manuscript.

Qubits are grouped by production period: period t contains first the M block
qubits and then the L slack qubits, that is, the indices kappa_t + 1, ...,
kappa_t + M + L, with kappa_t = (t-1)(M+L) [Eq. (26)].

Vector drawing made only with matplotlib (no AI-generated images).
Output: ../../outputs/Figure02.pdf (vector, the one used by the .tex file)
        ../../outputs/Figure02.png (preview)

How to use: from the FiguresMakers folder, with the virtual environment activated,
    python Figure02.py
"""

# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------
from pathlib import Path          # path handling

import numpy as np                # for the smooth brace curve
import matplotlib.pyplot as plt   # figure and axes
# FancyBboxPatch -> rounded qubit boxes; FancyArrowPatch -> bottom arrow
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ----------------------------------------------------------------------------
# Output directory: ../../outputs, relative to the location of this script
# ----------------------------------------------------------------------------
OUT = Path(__file__).resolve().parents[2] / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# Typography (fonttype 42 = real text inside the PDF)
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                     "mathtext.fontset": "dejavusans"})

# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------
FS = 1.0                                   # global font scale (increase if it looks too small)
NAVY = "#1f3f7a"                           # dark blue: "Period ..." titles
TXT = "#2b2b2b"                            # general text
BLOCK_FC, BLOCK_EC = "#dbe6f7", "#2f5fb3"  # fill and border of block qubits (blue)
SLACK_FC, SLACK_EC = "#dcefe1", "#3a9a5b"  # fill and border of slack qubits (green)
BRACE = "#6b3d8f"                          # color of the braces (purple)


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def brace(ax, x0, x1, y, depth=0.35, color=BRACE, lw=1.6):
    """Draw a horizontal brace pointing downward, from x0 to x1, at height y.
    depth is the total brace depth. Returns the (x, y) position of its tip,
    so the text can be placed just below it."""
    xm = 0.5 * (x0 + x1)                   # center of the brace (where the tip is)
    q = depth / 2                          # height of the flat segment
    t = np.linspace(0, 1, 60)              # parameter used to draw each segment

    def seg(xa, xb, ya, yb):
        # smooth curved segment from (xa, ya) to (xb, yb), with an S-like (cubic) shape
        s = 3 * t ** 2 - 2 * t ** 3
        return xa + (xb - xa) * t, ya + (yb - ya) * s

    # the brace is built with 6 segments: left curve, line, descent to the tip,
    # ascent from the tip, line, and right curve.
    for (xa, xb, ya, yb) in [(x0, x0 + 0.25, y, y - q), (x0 + 0.25, xm - 0.25, y - q, y - q),
                             (xm - 0.25, xm, y - q, y - depth), (xm, xm + 0.25, y - depth, y - q),
                             (xm + 0.25, x1 - 0.25, y - q, y - q), (x1 - 0.25, x1, y - q, y)]:
        xs, ys = seg(xa, xb, ya, yb)
        ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round")
    return xm, y - depth


def register(ax, x0, y0, n, fc, ec, labels, cell=0.62, gap=0.08):
    """Draw a row of n qubit boxes starting at (x0, y0).
    labels: list of n elements; if one element is None, an ellipsis is drawn
    at that position instead of a box. Returns the starting and ending x of the row."""
    xs = []
    for i in range(n):
        x = x0 + i * (cell + gap)          # position of box i
        lab = labels[i]
        if lab is None:
            # position with "..." (intermediate qubits not drawn)
            ax.text(x + cell / 2, y0 + cell / 2, r"$\cdots$", ha="center", va="center",
                    fontsize=13 * FS, color=TXT)
        else:
            # rounded qubit box
            ax.add_patch(FancyBboxPatch((x, y0), cell, cell, boxstyle="round,pad=0,rounding_size=0.08",
                                        fc=fc, ec=ec, lw=1.3))
        xs.append(x)
    return xs[0], xs[-1] + cell


# ----------------------------------------------------------------------------
# Figure and axis (in "drawing units": x between 0 and 24.6, y between -0.4 and 8.6)
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11.5, 4.6))   # size in inches
ax.set_xlim(0, 24.6)
ax.set_ylim(-0.4, 8.6)
ax.set_aspect("equal")                         # square boxes
ax.axis("off")                                 # no axes

cell, gap = 0.62, 0.08                         # box size and spacing (same as register)
y_reg = 5.0                                    # height at which qubit rows are drawn


def period_group(x_start, title, b1, b2, s1, s2):
    """Draw the qubit group of a period: M block qubits + L slack qubits.
    b1, b2: labels of the first and last block qubit; s1, s2: same for slack qubits.
    Returns the starting and ending x of the complete group."""
    # 6 positions per register: 3 boxes, "...", 2 boxes
    bl = [b1, "", "", None, "", b2]
    sl = [s1, "", "", None, "", s2]
    bx0, bx1 = register(ax, x_start, y_reg, len(bl), BLOCK_FC, BLOCK_EC, bl)      # blocks
    sx0, sx1 = register(ax, bx1 + 0.9, y_reg, len(sl), SLACK_FC, SLACK_EC, sl)    # slack, 0.9 further to the right
    # index of the first and last qubit above each register (aligned to the edges to avoid overlap)
    for (x, lab, ha, col) in [(bx0, b1, "left", BLOCK_EC), (bx1, b2, "right", BLOCK_EC),
                              (sx0, s1, "left", SLACK_EC), (sx1, s2, "right", SLACK_EC)]:
        ax.text(x, y_reg + cell + 0.15, lab, ha=ha, va="bottom", fontsize=12 * FS, color=col)
    # name of each register
    ax.text((bx0 + bx1) / 2, y_reg + cell + 0.95, "block variables", ha="center", va="bottom",
            fontsize=12 * FS, color=BLOCK_EC, fontweight="bold")
    ax.text((sx0 + sx1) / 2, y_reg + cell + 0.95, "slack variables", ha="center", va="bottom",
            fontsize=12 * FS, color=SLACK_EC, fontweight="bold")
    # period title, centered above the whole group
    ax.text((bx0 + sx1) / 2, y_reg + cell + 1.8, title, ha="center", va="bottom",
            fontsize=14 * FS, color=NAVY, fontweight="bold")
    # braces below each register and below the full group, with their text
    xm, yb = brace(ax, bx0, bx1, y_reg - 0.15)
    ax.text(xm, yb - 0.12, r"$M$ qubits", ha="center", va="top", fontsize=12 * FS, color=TXT)
    xm, yb = brace(ax, sx0, sx1, y_reg - 0.15)
    ax.text(xm, yb - 0.12, r"$L$ qubits", ha="center", va="top", fontsize=12 * FS, color=TXT)
    xm, yb = brace(ax, bx0, sx1, y_reg - 1.25, depth=0.4)
    ax.text(xm, yb - 0.12, r"$M+L$ qubits", ha="center", va="top", fontsize=12 * FS, color=TXT)
    return bx0, sx1


# ----------------------------------------------------------------------------
# Figure content
# ----------------------------------------------------------------------------
# period 1 group (on the left)
g1 = period_group(0.6, r"Period $t=1$", r"$q_1$", r"$q_M$", r"$q_{M+1}$", r"$q_{M+L}$")
# ellipsis between period 1 and period t
ax.text(11.35, y_reg + cell / 2, r"$\cdots$", ha="center", va="center", fontsize=22 * FS, color=TXT)
# group of a generic period t (on the right), with indices written using kappa_t
g2 = period_group(12.6, r"Period $t$", r"$q_{\kappa_t+1}$", r"$q_{\kappa_t+M}$",
                  r"$q_{\kappa_t+M+1}$", r"$q_{\kappa_t+M+L}$")
# ellipsis after period t (periods t+1, ..., T follow)
ax.text(g2[1] + 0.45, y_reg + cell / 2, r"$\cdots$", ha="left", va="center", fontsize=22 * FS, color=TXT)

# line with the definition of kappa_t and the variable -> qubit mapping rule
ax.text(12.0, 2.15,
        r"$\kappa_t=(t-1)(M+L)$" "      "
        r"$y_{j,t}\ (\mathrm{or}\ x_{j,t})\rightarrow q_{\kappa_t+j}$" "      "
        r"$s_{r,t}\rightarrow q_{\kappa_t+M+r}$",
        ha="center", va="center", fontsize=13 * FS, color=TXT)

# bottom arrow: qubit index increases from left to right
ax.add_patch(FancyArrowPatch((g1[0], 1.1), (g2[1] + 1.4, 1.1), arrowstyle="-|>", mutation_scale=18,
                             lw=1.8, color=TXT))
ax.text(12.0, 0.65, r"qubit index (grouped by period),   total  $N_q=T(M+L)$",
        ha="center", va="top", fontsize=12.5 * FS, color=TXT)


# ----------------------------------------------------------------------------
# Layout check: warns if two texts overlap or if any text is
# outside the figure area. It does not modify anything; it only prints.
# ----------------------------------------------------------------------------
def check_text_overlaps(fig, ax):
    fig.canvas.draw()                              # draw to know the real size of the texts
    r = fig.canvas.get_renderer()
    # bounding box (in pixels) of each non-empty text
    bbs = [(t.get_text(), t.get_window_extent(r)) for t in ax.texts if t.get_text().strip()]
    # pairs of texts whose boxes intersect
    bad = [(a[0], b[0]) for i, a in enumerate(bbs) for b in bbs[i + 1:] if a[1].overlaps(b[1])]
    # texts that extend outside the axis area
    fbb = ax.get_window_extent(r)
    out = [t for t, bb in bbs if bb.x0 < fbb.x0 or bb.x1 > fbb.x1 or bb.y0 < fbb.y0 or bb.y1 > fbb.y1]
    if bad or out:
        print("LAYOUT ISSUES:", *[f"  overlapping texts {a!r} / {b!r}" for a, b in bad],
              *[f"  outside the figure {t!r}" for t in out], sep="\n")
    else:
        print("layout check OK: no overlapping or clipped text")


check_text_overlaps(fig, ax)

# ----------------------------------------------------------------------------
# Save: vector PDF (for the paper) and preview PNG
# ----------------------------------------------------------------------------
fig.savefig(OUT / "Figure02.pdf", bbox_inches="tight", pad_inches=0.05)
fig.savefig(OUT / "Figure02.png", dpi=250, bbox_inches="tight", pad_inches=0.05)
print(f"saved {OUT / 'Figure02.pdf'}")
