"""
Figure 3: interaction graphs induced by the validity penalties (top)
and the precedence penalties (bottom) in the at and by formulations. Edges are drawn
exactly according to Eqs. (4), (6), (15), and (17) of the manuscript:

  validity   at : x_{j,t} x_{j,tau} for all t < tau        -> complete graph, T(T-1)/2 edges
             by : y_{j,t-1} y_{j,t}                        -> chain, T-1 edges
  precedence at : x_{j,t} x_{k,tau} for all tau <= t       -> T(T+1)/2 edges per relation
             by : y_{j,t} y_{k,t}                          -> T edges per relation

Vector drawing made only with matplotlib (no AI-generated images).
Output: ../../outputs/Figure03.pdf (vector, the one used by the .tex file)
        ../../outputs/Figure03.png (preview)

How to use: from the FiguresMakers folder, with the virtual environment activated,
    python Figure03.py
"""

# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------
from pathlib import Path          # path handling

import matplotlib.pyplot as plt   # figure and axes
# FancyBboxPatch -> panel frames; Circle -> nodes (variables);
# Arc -> arcs for "at" validity; FancyArrowPatch -> time axis
from matplotlib.patches import FancyBboxPatch, Circle, Arc, FancyArrowPatch

# ----------------------------------------------------------------------------
# Output directory: ../../outputs, relative to the location of this script
# ----------------------------------------------------------------------------
OUT = Path(__file__).resolve().parents[2] / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# Typography (fonttype 42 = real text inside the PDF)
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                     "mathtext.fontset": "dejavusans"})

# ----------------------------------------------------------------------------
# Parameters and style
# ----------------------------------------------------------------------------
FS = 1.0                 # global font scale (increase if it looks too small)
T = 5                    # number of periods drawn (the last one is labeled "T")
NAVY = "#1f3f7a"         # dark blue: titles
TXT = "#2b2b2b"          # general text
FRAME = "#9fb3d6"        # panel border
C_J = "#2f5fb3"          # color of successor block j variables (blue)
C_K = "#c0392b"          # color of predecessor block k variables (red)
EDGE_C = "#555555"       # color of the edges (quadratic couplings)
R_NODE = 0.30            # radius of each node

# ----------------------------------------------------------------------------
# Figure and axis (in "drawing units": W x H)
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 8.4))   # size in inches
W, H = 24.0, 16.8
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")                       # round circles, not ellipses
ax.axis("off")


# ----------------------------------------------------------------------------
# Helper drawing functions
# ----------------------------------------------------------------------------
def frame(x0, y0, w, h):
    """White frame with light-blue border and rounded corners (one panel)."""
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.3",
                                fc="white", ec=FRAME, lw=1.3, zorder=0))


def node(x, y, color, label, below=False):
    """Node (binary variable) at (x, y) with its label above or below (below=True)."""
    ax.add_patch(Circle((x, y), R_NODE, fc=color, ec="white", lw=1.5, zorder=5))
    dy = -0.48 if below else 0.48                       # label-node separation
    ax.text(x, y + dy, label, ha="center", va="top" if below else "bottom",
            fontsize=12 * FS, color=color, zorder=6)


def edge(p, q, lw=1.4, color=EDGE_C):
    """Straight edge between points p and q (a quadratic coupling)."""
    ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, zorder=3, solid_capstyle="round")


def arc_edge(xa, xb, y, color=EDGE_C, lw=1.4):
    """Arc above the node line connecting two nodes at the same height.
    Its height grows with the distance between nodes so they do not overlap."""
    xc, w = 0.5 * (xa + xb), abs(xb - xa)               # centro y ancho del arco
    h = 0.42 * w                                         # altura proporcional al ancho
    ax.add_patch(Arc((xc, y), w, 2 * h, theta1=0, theta2=180, color=color, lw=lw, zorder=3))


# ----------------------------------------------------------------------------
# Figure layout: 2 columns (at, by) x 2 rows (validity, precedence)
# ----------------------------------------------------------------------------
col_x = [3.6, 14.1]                       # left edge of the at / by columns
pw, ph = 9.7, 6.5                         # width and height of each panel
row_y = [8.75, 1.15]                      # lower edge of the validity / precedence rows
dx = 1.75                                 # horizontal spacing between periods

# column titles (top)
ax.text(col_x[0] + pw / 2, H - 0.35, r"$\mathit{at}$ formulation", ha="center", va="top",
        fontsize=16 * FS, fontweight="bold", color=NAVY)
ax.text(col_x[0] + pw / 2, H - 1.05, r"(extraction-period encoding, $x_{j,t}$)", ha="center",
        va="top", fontsize=12 * FS, color=TXT)
ax.text(col_x[1] + pw / 2, H - 0.35, r"$\mathit{by}$ formulation", ha="center", va="top",
        fontsize=16 * FS, fontweight="bold", color=NAVY)
ax.text(col_x[1] + pw / 2, H - 1.05, r"(cumulative encoding, $y_{j,t}$)", ha="center", va="top",
        fontsize=12 * FS, color=TXT)
# row titles (left)
ax.text(1.75, row_y[0] + ph / 2, "Validity\npenalty\n(one block $j$)", ha="center", va="center",
        fontsize=13 * FS, fontweight="bold", color=NAVY, linespacing=1.4)
ax.text(1.75, row_y[1] + ph / 2, "Precedence\npenalty\n(one relation\n$k\\in\\Lambda(j)$)",
        ha="center", va="center", fontsize=13 * FS, fontweight="bold", color=NAVY, linespacing=1.4)

# the four frames
for cx in col_x:
    for ry in row_y:
        frame(cx, ry, pw, ph)


def time_positions(cx):
    """x positions of the T periods, centered inside the panel starting at cx."""
    x0 = cx + (pw - (T - 1) * dx) / 2
    return [x0 + i * dx for i in range(T)]


def time_axis(cx, y):
    """Gray time-axis arrow below the nodes, with the label "t"."""
    xs = time_positions(cx)
    ax.add_patch(FancyArrowPatch((xs[0] - 0.6, y), (xs[-1] + 0.8, y), arrowstyle="-|>",
                                 mutation_scale=12, lw=1.0, color="#999999", zorder=1))
    ax.text(xs[-1] + 0.85, y - 0.15, "$t$", ha="left", va="top", fontsize=12 * FS, color="#777777")


# period labels: 1, 2, ..., T-1 and the last one "T"
tlabels = [f"{i + 1}" for i in range(T - 1)] + ["T"]

# ---------------------------------------------------------------- validity (top row)
for c, (cx, var) in enumerate(zip(col_x, ["x", "y"])):   # c = 0 -> at (x), c = 1 -> by (y)
    y = row_y[0] + 2.0                                     # height of the node row
    xs = time_positions(cx)
    time_axis(cx, y - 1.05)
    if c == 0:
        # at: all pairs of periods of the same block -> complete graph (arcs)
        for a in range(T):
            for b in range(a + 1, T):
                arc_edge(xs[a], xs[b], y)
        count = r"$T(T-1)/2$ couplings per block"
        note = "every pair of periods"
    else:
        # by: only consecutive periods -> chain (straight lines)
        for a in range(T - 1):
            edge((xs[a], y), (xs[a + 1], y), lw=2.0)
        count = r"$T-1$ couplings per block"
        note = "consecutive periods only"
    # x_{j,t} or y_{j,t} nodes
    for i, xx in enumerate(xs):
        node(xx, y, C_J, f"${var}_{{j,{tlabels[i]}}}$", below=True)
    ax.text(cx + pw / 2, row_y[0] + ph - 0.45, note, ha="center", va="top", fontsize=12.5 * FS,
            color=TXT, fontweight="bold")                  # description at the top of the panel
    ax.text(cx + pw / 2, row_y[0] + 0.35, count, ha="center", va="bottom", fontsize=12 * FS,
            color=TXT)                                     # number of couplings below

# ---------------------------------------------------------------- precedence (bottom row)
for c, (cx, var) in enumerate(zip(col_x, ["x", "y"])):
    yk = row_y[1] + 4.55        # height of predecessor k nodes (upper row)
    yj = row_y[1] + 2.25        # height of successor j nodes (lower row)
    xs = time_positions(cx)
    if c == 0:
        # at: x_{j,t} couples with x_{k,tau} for all tau <= t (Eq. 6)
        for t in range(T):
            for tau in range(t + 1):
                edge((xs[t], yj), (xs[tau], yk), lw=1.1)
        count = r"$T(T+1)/2$ couplings per relation"
        note = r"$x_{j,t}$ coupled to $x_{k,\tau}$ for all $\tau\leq t$"
    else:
        # by: y_{j,t} couples only with y_{k,t} of the same period (Eq. 17)
        for t in range(T):
            edge((xs[t], yj), (xs[t], yk), lw=2.0)
        count = r"$T$ couplings per relation"
        note = r"$y_{j,t}$ coupled to $y_{k,t}$ only"
    # k nodes (label above) and j nodes (label below)
    for i, xx in enumerate(xs):
        node(xx, yk, C_K, f"${var}_{{k,{tlabels[i]}}}$", below=False)
        node(xx, yj, C_J, f"${var}_{{j,{tlabels[i]}}}$", below=True)
    ax.text(cx + pw / 2, row_y[1] + ph - 0.35, note, ha="center", va="top", fontsize=12.5 * FS,
            color=TXT, fontweight="bold")
    ax.text(cx + pw / 2, row_y[1] + 0.35, count, ha="center", va="bottom", fontsize=12 * FS,
            color=TXT)

# ---------------------------------------------------------------- legend (bottom)
ly = 0.45                                                  # altura de la leyenda
items = [(C_J, r"block $j$ (successor)"), (C_K, r"block $k\in\Lambda(j)$ (predecessor)")]
for lx, (color, lab) in zip([4.2, 10.9], items):           # x position of each entry
    ax.add_patch(Circle((lx, ly), 0.2, fc=color, ec="white", lw=1, zorder=5))
    ax.text(lx + 0.4, ly, lab, ha="left", va="center", fontsize=12 * FS, color=TXT)
lx = 18.6                                                  # "quadratic coupling" entry
ax.plot([lx, lx + 0.8], [ly, ly], color=EDGE_C, lw=1.6)
ax.text(lx + 1.0, ly, "quadratic coupling", ha="left", va="center", fontsize=12 * FS, color=TXT)


# ----------------------------------------------------------------------------
# Layout check: warns if two texts overlap or if any text is
# outside the figure area. It does not modify anything; it only prints.
# ----------------------------------------------------------------------------
def check_text_overlaps(fig, ax):
    fig.canvas.draw()                              # draw to know the real size of the texts
    r = fig.canvas.get_renderer()
    bbs = [(t.get_text(), t.get_window_extent(r)) for t in ax.texts if t.get_text().strip()]
    bad = [(a[0], b[0]) for i, a in enumerate(bbs) for b in bbs[i + 1:] if a[1].overlaps(b[1])]
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
fig.savefig(OUT / "Figure03.pdf", bbox_inches="tight", pad_inches=0.05)
fig.savefig(OUT / "Figure03.png", dpi=250, bbox_inches="tight", pad_inches=0.05)
print(f"saved {OUT / 'Figure03.pdf'}")
