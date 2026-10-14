import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch
import numpy as np
from pathlib import Path

# Academic-style typography (LaTeX / MathText)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",  # Render the mathematical 'v' as in LaTeX
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# Output directory: ../../outputs, relative to this script
OUT = Path(__file__).resolve().parents[2] / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

S = 1.0          # block size
DX, DY = 0.32, 0.22  # isometric offset

def shade(c, f):
    c = np.array(matplotlib.colors.to_rgb(c))
    return tuple(np.clip(c * f, 0, 1)) if f < 1 else tuple(c + (1 - c) * (f - 1))

def cube(ax, x, y, color, alpha=1.0, z=1, edge="#2b2b2b", lw=0.8):
    front = [(x, y), (x + S, y), (x + S, y + S), (x, y + S)]
    top = [(x, y + S), (x + S, y + S), (x + S + DX, y + S + DY), (x + DX, y + S + DY)]
    side = [(x + S, y), (x + S + DX, y + DY), (x + S + DX, y + S + DY), (x + S, y + S)]
    for poly, f in ((side, 0.72), (top, 1.25), (front, 1.0)):
        ax.add_patch(Polygon(poly, closed=True, fc=shade(color, f), ec=edge,
                             lw=lw, alpha=alpha, zorder=z, joinstyle="round"))

P1, P2, NX = "#2f5fb3", "#f08a24", "#a7a9ac"

blocks = {1: (0, 0), 2: (1, 0), 3: (2, 0), 4: (3, 0), 5: (4, 0),
          6: (1, 1), 7: (2, 1), 8: (3, 1), 9: (2, 2)}
values = {1: 20, 2: -45, 3: 60, 4: -30, 5: 25, 6: 120, 7: 320, 8: 220, 9: 500}
sched = {3: P1, 4: P1, 5: P1, 2: P2, 7: P2, 8: P2, 1: NX, 6: NX, 9: NX}

def pos(col, row):
    return col * S, -row * S

fig, ax = plt.subplots(figsize=(8.0, 3.8), facecolor="white")

# Extend the left axis limit (-2.2) to accommodate bench labels
ax.set_xlim(-2.2, 12.0)
ax.set_ylim(-2.8, 1.8)
ax.set_aspect("equal")
ax.axis("off")

# Model blocks
for j in sorted(blocks, key=lambda k: (-blocks[k][1], blocks[k][0])):
    col, row = blocks[j]
    x, y = pos(col, row)
    cube(ax, x, y, sched[j], z=3 + (2 - row))
    txtc = "white" if sched[j] != NX else "#2b2b2b"
    
    # Block number (font size 14)
    ax.text(x + S / 2, y + 0.60, str(j), ha="center", va="center", fontsize=14,
            fontweight="bold", color=txtc, zorder=10)
    
    # v_j = block value (LaTeX mathematical 'v', approximately 9-point font)
    ax.text(x + S / 2, y + 0.26, f"$v_{{{j}}}={values[j]}$", ha="center", va="center",
            fontsize=8.5, color=txtc, zorder=10)

# Bench labels (Bench 1, Bench 2, Bench 3) to the left of the pit
for row, name in enumerate(["Bench 1", "Bench 2", "Bench 3"]):
    ax.text(-0.5, -row * S + 0.5, name, ha="right", va="center", fontsize=12,
             color="#333333", zorder=12)

# Legend in its original position (top)
card = FancyBboxPatch((7.2, -1.85), 4.4, 3.3, boxstyle="round,pad=0.02,rounding_size=0.1",
                      fc="white", ec="#cccccc", lw=0.8, zorder=13)
ax.add_patch(card)

ax.text(7.4, 1.15, "Optimal schedule", fontsize=10, fontweight="bold", color="#111111", zorder=14, va="center")
ax.text(7.4, 0.75, r"$M=9,\ T=2,\ C=3,\ d=0.08$", fontsize=8.5, color="#333333", zorder=14, va="center")

items = [(P1, "Period 1  (3, 4, 5)"), (P2, "Period 2  (2, 7, 8)"), (NX, "Not extracted  (1, 6, 9)")]
for i, (c, lab) in enumerate(items):
    yy = 0.18 - i * 0.58
    sx = 0.40
    ax.add_patch(Polygon([(7.45, yy - 0.18), (7.45 + sx, yy - 0.18), (7.45 + sx, yy + 0.18), (7.45, yy + 0.18)],
                         fc=c, ec="#2b2b2b", lw=0.7, zorder=14))
    ax.add_patch(Polygon([(7.45, yy + 0.18), (7.45 + sx, yy + 0.18), (7.45 + sx + 0.12, yy + 0.26), (7.45 + 0.12, yy + 0.26)],
                         fc=shade(c, 1.25), ec="#2b2b2b", lw=0.7, zorder=14))
    ax.add_patch(Polygon([(7.45 + sx, yy - 0.18), (7.45 + sx + 0.12, yy - 0.10), (7.45 + sx + 0.12, yy + 0.26), (7.45 + sx, yy + 0.18)],
                         fc=shade(c, 0.72), ec="#2b2b2b", lw=0.7, zorder=14))
    ax.text(8.05, yy, lab, fontsize=8.5, va="center", zorder=14, color="#222222")

ax.text(7.4, -1.55, r"Labels: block $j$ and value $v_j$", fontsize=7.5, color="#555555", zorder=14, va="center")

plt.show()
fig.savefig(OUT / "Figure05.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "Figure05.png", dpi=170, bbox_inches="tight", pad_inches=0.02)
