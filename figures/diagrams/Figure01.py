"""
Figure 1: simplified open-pit mine production scheduling problem.

Vector drawing made only with matplotlib (no AI-generated images).
Output: ../../outputs/Figure01.pdf (vector, the one used by the .tex file)
        ../../outputs/Figure01.png (preview)

Panels
  1  Block value changes over time (discounting v_{j,t} = v_j / (1+d)^(t-1))
  2  Precedence constraint (slope stability), 2D cone section
  3  Top view of the immediate predecessors (3x3 grid in the upper bench)
  4  Extraction capacity per period (n_t <= C)
  5  Spatial discretization and pit evolution over time

How to use: from the FiguresMakers folder, with the virtual environment activated,
    python Figure01.py
At the end, the script prints the result of an automatic layout check that warns
if any text or block extends outside its panel or if two texts overlap.
"""

# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------
from pathlib import Path          # OS-independent path handling

import numpy as np                # numerical arrays (block values)
import matplotlib.pyplot as plt   # figure and axes
# Geometric shapes drawn manually:
#   Polygon        -> faces of each cube
#   FancyBboxPatch -> rounded-corner boxes (frame of each panel)
#   Circle         -> blue circle with the panel number
#   Rectangle      -> simple rectangles (color bar, cells, legends)
#   FancyArrowPatch-> arrows
from matplotlib.patches import Polygon, FancyBboxPatch, Circle, Rectangle, FancyArrowPatch
# TwoSlopeNorm: normalization with a central value (0) different from the midpoint
# LinearSegmentedColormap: to define our own color scale
# to_rgb: converts a "#rrggbb" color into an (r, g, b) tuple between 0 and 1
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap, to_rgb

# ----------------------------------------------------------------------------
# Output directory: ../../outputs, relative to the location of this script
# ----------------------------------------------------------------------------
OUT = Path(__file__).resolve().parents[2] / "outputs"
OUT.mkdir(parents=True, exist_ok=True)          # create it if it does not exist

# Typography: DejaVu Sans for text and formulas; fonttype 42 keeps text
# as real text inside the PDF (selectable and accepted by Elsevier)
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "mathtext.fontset": "dejavusans"})

# ----------------------------------------------------------------------------
# Style: global colors and sizes (changing them here affects the whole figure)
# ----------------------------------------------------------------------------
NAVY = "#1f3f7a"     # dark blue: titles, numbered circles, and arrows
FRAME = "#9fb3d6"    # light blue: panel border
TXT = "#2b2b2b"      # nearly black gray: general text
EDGE = "#3a3a3a"     # dark gray: cube edges
P_COLORS = ["#2f5fb3", "#f08a24", "#3a9a5b"]      # colors of periods 1, 2, and 3

# Panel 1 color scale: dark blue (waste) -> light gray (v=0) -> red (ore)
# Each tuple is (position on the scale between 0 and 1, color), from 1.0 to 0.0 every 0.1
VALUE_CMAP = LinearSegmentedColormap.from_list(
    "value", [(1.0, "#de4b36"),    # high-grade ore (maximum value): red
              (0.9, "#e06958"),
              (0.8, "#e1887a"),
              (0.7, "#e3a69b"),
              (0.6, "#e4c5bd"),
              (0.5, "#e6e3df"),    # v = 0: neutral light gray
              (0.4, "#bec2cb"),
              (0.3, "#96a1b7"),
              (0.2, "#6f81a2"),
              (0.1, "#47608e"),
              (0.0, NAVY)][::-1])  # waste (most negative value): dark blue
                                   # [::-1]: matplotlib requires increasing positions

# Oblique projection to draw cubes in 2D: the depth axis (y)
# is drawn upward and to the right. KX and KY control the tilt.
KX, KY = 0.42, 0.30

# Global font scale (1.0 = base size). Increase it if it looks too small in the paper.
FS = 1.75     # general text
FS_T = 1.5    # panel titles


# ----------------------------------------------------------------------------
# Funciones auxiliares de dibujo
# ----------------------------------------------------------------------------
def shade(c, f):
    """Lighten (f > 1) or darken (f < 1) a color; used to give volume to faces."""
    c = np.array(to_rgb(c))                       # color como arreglo [r, g, b]
    if f < 1:
        return tuple(np.clip(c * f, 0, 1))        # oscurecer: multiplicar
    return tuple(c + (1 - c) * (f - 1))           # aclarar: mezclar con blanco


def proj(x, y, z, ox, oy, s):
    """Project a 3D point (x, y, z) onto the 2D plane of the figure.
    (ox, oy) is the drawing origin and s is the size of a cube."""
    return ox + s * (x + KX * y), oy + s * (z + KY * y)


def draw_voxels(ax, voxels, ox, oy, s, lw=0.35, zbase=5):
    """Draw a set of cubes (voxels) in oblique view.

    voxels: list of (i, j, k, color) or (i, j, k, color, alpha), where
            i = column (x axis), j = depth (y axis), k = height (z axis).
    The "painter's algorithm" is used: back and lower cubes are drawn first,
    then front and upper cubes, which cover the previous ones.
    Only visible faces are drawn: front (-y), top (+z), and right (+x).
    """
    occ = {(v[0], v[1], v[2]) for v in voxels}             # occupied positions
    # drawing order: deepest first (-j), then bottom to top (k), then left to right (i)
    order = sorted(voxels, key=lambda v: (-v[1], v[2], v[0]))
    for v in order:
        i, j, k, col = v[:4]
        a = v[4] if len(v) > 4 else 1.0                    # transparency (1 = opaque)
        P = lambda dx, dy, dz: proj(i + dx, j + dy, k + dz, ox, oy, s)  # projected vertex
        faces = []
        # each face is drawn only if there is no adjacent cube attached to it
        if (i, j - 1, k) not in occ:   # front face (base color)
            faces.append(([P(0, 0, 0), P(1, 0, 0), P(1, 0, 1), P(0, 0, 1)], 1.0))
        if (i, j, k + 1) not in occ:   # top face (lighter)
            faces.append(([P(0, 0, 1), P(1, 0, 1), P(1, 1, 1), P(0, 1, 1)], 1.22))
        if (i + 1, j, k) not in occ:   # right face (darker)
            faces.append(([P(1, 0, 0), P(1, 1, 0), P(1, 1, 1), P(1, 0, 1)], 0.74))
        for poly, f in faces:
            ax.add_patch(Polygon(poly, closed=True, fc=shade(col, f), ec=EDGE, lw=lw,
                                 alpha=a, zorder=zbase, joinstyle="round"))


# Panel registry for the final layout check.
# Each entry: (number, (x0, y0, width, height), index of the first text,
#                index of the first patch, index of the first line)
PANELS = []


def panel(ax, x0, y0, w, h, num, title, subtitle=None):
    """Draw the frame of a panel with its numbered circle, title, and lower subtitle.
    (x0, y0) is the bottom-left corner; w and h are width and height in drawing units."""
    # store how many elements existed before, to know which belong to this panel
    PANELS.append((num, (x0, y0, w, h), len(ax.texts), len(ax.patches) + 1, len(ax.lines)))
    # white frame with light-blue border and rounded corners
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.35",
                                fc="white", ec=FRAME, lw=1.3, zorder=1))
    # blue circle with the panel number (upper-left corner)
    cx, cy = x0 + 0.7, y0 + h - 0.7
    ax.add_patch(Circle((cx, cy), 0.42, fc=NAVY, ec="none", zorder=20))
    ax.text(cx, cy - 0.01, str(num), color="white", ha="center", va="center",
            fontsize=12 * FS_T, fontweight="bold", zorder=21)
    # title to the right of the circle
    ax.text(cx + 0.62, cy, title, color=NAVY, ha="left", va="center",
            fontsize=13.5 * FS_T, fontweight="bold", zorder=21)
    # optional subtitle, centered at the bottom of the panel
    if subtitle:
        ax.text(x0 + w / 2, y0 + 0.65, subtitle, color=TXT, ha="center", va="center",
                fontsize=10.5 * FS, zorder=21)


# ----------------------------------------------------------------------------
# Synthetic orebody for panel 1: values on an NX x NY x NZ grid
# (k = 0 is the bottom bench). It is illustrative, not real data.
# ----------------------------------------------------------------------------
NX, NY, NZ = 9, 5, 6                       # blocks in x, y, z
gx, gy, gz = np.meshgrid(np.arange(NX), np.arange(NY), np.arange(NZ), indexing="ij")
# (normalized) distance to the center of the orebody; the center is on the front face (y=0)
r2 = ((gx - 4.0) / 2.4) ** 2 + ((gy - 0.0) / 2.6) ** 2 + ((gz - 2.2) / 2.0) ** 2
VALUES = 1.7 * np.exp(-r2) - 0.3           # high-grade ore core (positive) surrounded by waste (negative)
VMAX = np.abs(VALUES).max()                # maximum absolute value, to scale the colors

# 0 always stays at the center of the color scale (light gray). vmin is the most
# negative, so edge blocks take the full blue half of the scale
NORM = TwoSlopeNorm(vmin=VALUES.min(), vcenter=0.0, vmax=VMAX)

# ----------------------------------------------------------------------------
# Figure: a single axis occupying the whole canvas, in "drawing units"
# (W x H). All positions below are given in these units.
# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 10.2))       # size in inches
ax = fig.add_axes([0, 0, 1, 1])            # axis covering the whole figure
W, H = 32.0, 20.4                          # width and height in drawing units
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")                     # 1 unit in x = 1 unit in y (undeformed cubes)
ax.axis("off")                             # no axes or ticks

# ============================================================================
# Panel 1: block value changes over time (discounting)
# ============================================================================
p1 = (0.3, 10.6, 13.6, 9.5)                # (x0, y0, width, height) of the panel
panel(ax, *p1, 1, "Block value changes over time",
      r"$v_{j,t}=v_j/(1+d)^{\,t-1}$")
d_ill = 0.45                               # exaggerated discount rate just to make the effect visible
s1 = 0.36                                  # size of each cube
# three models are drawn: period 1, period 2, and "period t" (here t = 4)
for n, (t, lab) in enumerate([(1, "Period 1"), (2, "Period 2"), (4, r"Period $t$")]):
    ox = p1[0] + 0.6 + n * 4.3             # horizontal position of each model
    oy = p1[1] + 3.75                      # vertical position (same for all three)
    fac = 1.0 / (1 + d_ill) ** (t - 1)     # discount factor of period t
    # color of each block according to its discounted value v_j * fac
    vox = [(i, j, k, VALUE_CMAP(NORM(VALUES[i, j, k] * fac)))
           for i in range(NX) for j in range(NY) for k in range(NZ)]
    draw_voxels(ax, vox, ox, oy, s1, lw=0.25)
    # "Period ..." label centered above each model
    ax.text(ox + s1 * (NX + KX * NY) / 2, p1[1] + p1[3] - 1.75, lab, ha="center",
            fontsize=11 * FS, color=TXT)
# time arrow below the models
ya = p1[1] + 3.35
ax.add_patch(FancyArrowPatch((p1[0] + 0.9, ya), (p1[0] + p1[2] - 0.9, ya), arrowstyle="-|>",
                             mutation_scale=16, lw=2, color=NAVY, zorder=10))
ax.add_patch(Circle((p1[0] + 0.9, ya), 0.12, fc=NAVY, ec="none", zorder=11))   # starting point
ax.text(p1[0] + 0.9, ya - 0.3, "Early periods", fontsize=10 * FS, ha="left", va="top", color=TXT)
ax.text(p1[0] + p1[2] - 0.9, ya - 0.3, "Late periods", fontsize=10 * FS, ha="right", va="top", color=TXT)
# color bar (manually drawn colorbar)
cb_x, cb_y, cb_w, cb_h = p1[0] + 3.0, p1[1] + 1.75, 6.0, 0.32   # position and size
grad = np.linspace(0, 1, 256).reshape(1, -1)                     # horizontal gradient 0 -> 1
ax.imshow(grad, extent=(cb_x, cb_x + cb_w, cb_y, cb_y + cb_h), cmap=VALUE_CMAP,
          aspect="auto", zorder=10)
ax.add_patch(Rectangle((cb_x, cb_y), cb_w, cb_h, fc="none", ec=EDGE, lw=0.6, zorder=11))  # border
ax.text(cb_x - 0.15, cb_y + cb_h / 2, "Low value", ha="right", va="center", fontsize=9.5 * FS, color=TXT)
ax.text(cb_x + cb_w + 0.15, cb_y + cb_h / 2, "High value", ha="left", va="center",
        fontsize=9.5 * FS, color=TXT)
ax.text(cb_x + cb_w / 2, cb_y + cb_h + 0.08, r"$v_{j,t}$", ha="center", va="bottom", fontsize=10 * FS)
# tick and label for the value 0 at the center of the bar
ax.plot([cb_x + cb_w / 2] * 2, [cb_y - 0.08, cb_y], color=EDGE, lw=0.8, zorder=11)
ax.text(cb_x + cb_w / 2, cb_y - 0.12, "0", ha="center", va="top", fontsize=9 * FS, color=TXT)

# ============================================================================
# Panel 2: restricción de precedencia (corte 2D del cono de talud)
# ============================================================================
p2 = (14.2, 10.6, 9.3, 9.5)
panel(ax, *p2, 2, "Precedence constraint",
      "Cone extracted before $j$")
ax.text(p2[0] + 0.35, p2[1] + p2[3] - 1.45, "Slope stability, 2D section", fontsize=10 * FS,
        color=TXT, ha="left", va="center")
s2 = 0.68                                  # size of each cube
ox2 = p2[0] + 2.75                         # origin of the cone drawing
oy2 = p2[1] + 1.55
vox2 = []
# the cone has 4 benches: 1, 3, 5, and 7 blocks (lev 0 = bench of the target block, at the bottom)
for lev, half in enumerate([0, 1, 2, 3]):
    for i in range(3 - half, 4 + half):    # bench columns, centered at i = 3
        if lev == 0:
            col = P_COLORS[0]                        # target block j (blue)
        elif lev == 1:
            col = "#6d6f73"                          # immediate predecessors Λ(j) (dark gray)
        else:
            col = "#c8c9cc"                          # rest of the cone (light gray)
        vox2.append((i, 0, lev, col))
draw_voxels(ax, vox2, ox2, oy2, s2, lw=0.6)
# white hatching over the immediate predecessors, to distinguish them also in black and white
for i in (2, 3, 4):
    x, y = proj(i, 0, 1, ox2, oy2, s2)
    ax.add_patch(Rectangle((x, y), s2, s2, fc="none", ec="white", lw=0, hatch="////",
                           alpha=0.45, zorder=8))
# "X" on the target block
xt, yt = proj(3.5, 0, 0.5, ox2, oy2, s2)
ax.text(xt, yt, "X", ha="center", va="center", fontsize=10 * FS, fontweight="bold",
        color="white", zorder=9)
# 45-degree dashed lines marking the cone boundaries
xa, ya_ = proj(3, 0, 0, ox2, oy2, s2)
xb, yb = proj(0, 0, 3 + 1, ox2, oy2, s2)
ax.plot([xa, xb - 0.15], [ya_, yb + 0.15], ls=(0, (4, 3)), color="#555", lw=1, zorder=9)   # border izquierdo
xa, ya_ = proj(4, 0, 0, ox2, oy2, s2)
xb, yb = proj(7, 0, 4, ox2, oy2, s2)
ax.plot([xa, xb + 0.15], [ya_, yb + 0.15], ls=(0, (4, 3)), color="#555", lw=1, zorder=9)   # border derecho
# bench names on the left
for lev, lab in [(0, "Bench 4"), (1, "Bench 3"), (2, "Bench 2"), (3, "Bench 1")]:
    _, yy = proj(0, 0, lev + 0.5, ox2, oy2, s2)
    ax.text(p2[0] + 0.35, yy, lab, fontsize=9.5 * FS, va="center", color=TXT)
# legend (upper-left corner of the panel)
lx, ly = p2[0] + 0.5, p2[1] + p2[3] - 2.25
for n, (c, lab) in enumerate([(P_COLORS[0], "Target $j$"), ("#6d6f73", r"Immediate Predecessor"),
                              ("#c8c9cc", "Allowable extraction cone")]):
    yy = ly - n * 0.62                     # vertical spacing between entries
    ax.add_patch(Rectangle((lx, yy - 0.2), 0.4, 0.4, fc=c, ec=EDGE, lw=0.5, zorder=10))  # color square
    ax.text(lx + 0.6, yy + 0.01, lab, fontsize=9.5 * FS, va="center", color=TXT, zorder=10)

# ============================================================================
# Panel 3: top view of the immediate predecessors
# ============================================================================
p3 = (23.8, 10.6, 7.9, 9.5)
panel(ax, *p3, 3, "Top view")
ax.text(p3[0] + 0.35, p3[1] + p3[3] - 1.45, "Immediate Predecessors", fontsize=10 * FS,
        color=TXT, ha="left", va="center")
cell, gap = 2.1, 0.15                      # size of each cell and spacing between them
gx0 = p3[0] + (p3[2] - 3 * cell - 2 * gap) / 2   # initial x to center the grid in the panel
gy0 = p3[1] + 0.65                         # initial y of the grid
# 3x3 grid of hatched blocks (the 9 predecessors in the upper bench)
for a in range(3):
    for b in range(3):
        x = gx0 + a * (cell + gap)
        y = gy0 + b * (cell + gap)
        ax.add_patch(Rectangle((x, y), cell, cell, fc="#d9dadc", ec=EDGE, lw=0.9,
                               hatch="///", zorder=5))
# central cell: position of block j (which is in the bench below)
xc = gx0 + (cell + gap)
yc = gy0 + (cell + gap)
ax.add_patch(Rectangle((xc + 0.35, yc + 0.35), cell - 0.7, cell - 0.7, fc=P_COLORS[0],
                       ec="white", lw=1.2, zorder=6, alpha=0.95))
ax.text(xc + cell / 2, yc + cell / 2, "X", color="white", fontsize=21 * FS, fontweight="bold",
        ha="center", va="center", zorder=7)

# ============================================================================
# Panel 4: extraction capacity per period
# ============================================================================
p4 = (0.3, 0.3, 11.4, 10.0)
panel(ax, *p4, 4, "Mining capacity constraint",
      r"At most $C$ blocks per period:  $n_t \leq C$")
C = 6                                      # capacity: maximum number of blocks per period
n_t = [5, 6, 4]                            # blocks extracted in each period shown
labels = ["Period 1", "Period 2", r"Period $t$"]
bw, bg = 0.92, 0.1                         # width of each cell and spacing between cells
for r, (n, lab) in enumerate(zip(n_t, labels)):
    y = p4[1] + p4[3] - 2.6 - r * 2.25     # height of row r (from top to bottom)
    ax.text(p4[0] + 0.5, y + 0.45, lab, fontsize=11 * FS, va="center", color=TXT)
    x0b = p4[0] + 3.6                      # x where the cells begin
    for c in range(C):
        x = x0b + c * (bw + bg)
        if c < n:
            # occupied cell: a cube in the period color
            vox = [(0, 0, 0, P_COLORS[min(r, 2)])]
            draw_voxels(ax, vox, x, y, bw * 0.86, lw=0.6)
        else:
            # empty cell: dashed square (unused capacity = slack)
            ax.add_patch(Rectangle((x, y), bw * 0.86, bw * 0.86, fc="none", ec="#9a9a9a",
                                   lw=0.9, ls=(0, (3, 2)), zorder=5))
    # horizontal bracket below the C cells
    xe = x0b + C * (bw + bg) - bg + 0.05
    ax.plot([x0b, xe], [y - 0.32, y - 0.32], color=TXT, lw=1)
    for xx in (x0b, xe):
        ax.plot([xx, xx], [y - 0.42, y - 0.22], color=TXT, lw=1)     # vertical end marks
    ax.text((x0b + xe) / 2, y - 0.82, f"$n_t={n}$  of  $C={C}$", ha="center", va="center",
            fontsize=9.5 * FS, color=TXT)
ax.text(p4[0] + p4[2] / 2, p4[1] + 1.35, "Dashed: unused capacity (slack)",
        ha="center", fontsize=9.5 * FS, color="#555")

# ============================================================================
# Panel 5: pit discretization and evolution over time
# ============================================================================
p5 = (12.0, 0.3, 19.7, 10.0)
panel(ax, *p5, 5, "Discretization and evolution over time")
NX5, NY5, NZ5 = 9, 5, 5                    # size of the block model in this panel
s5 = 0.37                                  # size of each cube
cx5 = 4                                    # central pit column


def pit_period(i, j, k):
    """Period in which block (i, j, k) is extracted; 0 if it is not extracted.
    k is measured from the bottom (the upper bench is k = NZ5-1).
    The pits of each period are nested 45-degree cones (Chebyshev distance),
    so each extracted set respects precedence."""
    depth = NZ5 - 1 - k                            # depth measured from the surface
    dist = max(abs(i - cx5), j)                    # cone centered on the front face (section view)
    for t, w in enumerate([2, 3, 4], start=1):     # half-width of the pit at the surface in each period
        if dist <= w - depth:
            return t
    return 0


# 5a: complete block model (spatial discretization)
ox5a, oy5a = p5[0] + 0.85, p5[1] + 3.6
vox = [(i, j, k, "#d4d5d8") for i in range(NX5) for j in range(NY5) for k in range(NZ5)]
draw_voxels(ax, vox, ox5a, oy5a, s5, lw=0.3)
# x, y, z axes at the corner of the model
axo = proj(0, 0, 0, ox5a - 0.35, oy5a - 0.35, s5)                 # origin of the axes
for (dx, dy, dz, lab) in [(2.6, 0, 0, "x"), (0, 3.2, 0, "y"), (0, 0, 2.6, "z")]:
    end = (axo[0] + s5 * (dx + KX * dy), axo[1] + s5 * (dz + KY * dy))   # tip of each axis
    ax.add_patch(FancyArrowPatch(axo, end, arrowstyle="-|>", mutation_scale=9, lw=1, color=TXT,
                                 zorder=12))
    ax.text(end[0] + 0.1, end[1] + 0.05, lab, fontsize=9.5 * FS, color=TXT, zorder=12)
ax.text(ox5a + s5 * (NX5 + KX * NY5) / 2, p5[1] + 1.65, "Block model\n(spatial)",
        ha="center", va="center", fontsize=10.5 * FS, color=TXT)

# vertical line separating the block model from the temporal evolution
xdiv = p5[0] + 6.15
ax.plot([xdiv, xdiv], [p5[1] + 1.0, p5[1] + p5[3] - 1.2], color=FRAME, lw=1)

# 5b: the pit at each period (section through the pit center)
for n, t in enumerate([1, 2, 3]):
    ox = p5[0] + 6.45 + n * 4.35           # horizontal position of each model
    oy = p5[1] + 3.6
    vox = []
    for i in range(NX5):
        for j in range(NY5):
            for k in range(NZ5):
                tp = pit_period(i, j, k)
                if 0 < tp < t:
                    continue                       # extracted in a previous period: do not draw it
                # colored if extracted in this period; gray if it remains in place
                col = P_COLORS[t - 1] if tp == t else "#d4d5d8"
                vox.append((i, j, k, col))
    draw_voxels(ax, vox, ox, oy, s5, lw=0.3)
    ax.text(ox + s5 * (NX5 + KX * NY5) / 2, p5[1] + p5[3] - 2.0, f"Period {t}", ha="center",
            fontsize=11 * FS, color=TXT)
    # small arrow between one period and the next
    if n < 2:
        xa = ox + s5 * (NX5 + KX * NY5) + 0.05
        ax.add_patch(FancyArrowPatch((xa, oy + 1.0), (xa + 0.45, oy + 1.0), arrowstyle="-|>",
                                     mutation_scale=12, lw=1.5, color=NAVY, zorder=12))
ax.text(p5[0] + 6.45 + 1.5 * 4.35 + 0.75, p5[1] + 1.65,
        "Colored: extracted in this period\n"
        "Earlier periods already removed",
        ha="center", va="center", fontsize=10.5 * FS, color=TXT)


# ----------------------------------------------------------------------------
# Layout check: every text, shape, and line must remain inside its panel
# (with a margin), and texts within the same panel must not overlap each other.
# It does not modify the figure: it only prints a warning if it finds problems.
# ----------------------------------------------------------------------------
def check_layout(fig, ax, margin=0.25):
    fig.canvas.draw()                              # draw the figure to know the real size of each text
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()                  # transform pixels -> drawing units
    problems = []
    # range of elements in each panel: from its first index to that of the next panel
    bounds = [p[2:] for p in PANELS] + [(len(ax.texts), len(ax.patches) + 1, len(ax.lines))]
    for n, (num, (x0, y0, w, h), it, ip, il) in enumerate(PANELS):
        jt, jp, jl = bounds[n + 1]
        inner = (x0 + margin, y0 + margin, x0 + w - margin, y0 + h - margin)   # allowed area
        boxes = []
        for kind, arts in (("text", ax.texts[it:jt]), ("patch", ax.patches[ip:jp - 1]),
                           ("line", ax.lines[il:jl])):
            for a in arts:
                bb = a.get_window_extent(renderer)                             # bounding box in pixels
                (bx0, by0), (bx1, by1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
                if bx0 < inner[0] or by0 < inner[1] or bx1 > inner[2] or by1 > inner[3]:
                    name = a.get_text() if kind == "text" else kind
                    problems.append(f"panel {num}: {name!r} fuera del panel "
                                    f"({bx0:.2f},{by0:.2f})-({bx1:.2f},{by1:.2f})")
                if kind == "text":
                    boxes.append((a.get_text(), bx0, by0, bx1, by1))
        # pairs of texts in the same panel whose boxes intersect
        for a in range(len(boxes)):
            for b in range(a + 1, len(boxes)):
                ta, ax0, ay0, ax1, ay1 = boxes[a]
                tb, bx0, by0, bx1, by1 = boxes[b]
                if ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1:
                    problems.append(f"panel {num}: textos montados {ta!r} / {tb!r}")
    if problems:
        print("LAYOUT ISSUES:")
        for p in problems:
            print("  -", p)
    else:
        print("layout check OK: everything is inside its panel, with no overlapping text")
    return problems


check_layout(fig, ax)

# ----------------------------------------------------------------------------
# Save: vector PDF (for the paper) and preview PNG
# ----------------------------------------------------------------------------
fig.savefig(OUT / "Figure01.pdf", bbox_inches="tight", pad_inches=0.05)
fig.savefig(OUT / "Figure01.png", dpi=200, bbox_inches="tight", pad_inches=0.05)
print(f"saved {OUT / 'Figure01.pdf'}")
