import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


# ============================================================
# Input files
# ============================================================

AT_JSON = Path(__file__).resolve().parent / "../../data/At_22Q_random_100_independent_R_s0.json"
BY_JSON = Path(__file__).resolve().parent / "../../data/By_22Q_random_100_independent_R_s0.json"

R_TARGET_PAIRED = 30

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"


# ============================================================
# Paper-style plotting configuration
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 6.5,
    "axes.labelsize": 6.5,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "legend.fontsize": 6.5,
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})

AT_COLOR = "#1f77b4"
BY_COLOR = "#ff7f0e"


# ============================================================
# Helpers
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_aggregate_probability_statistics(data):
    agg = data["aggregate_statistics"]

    R = np.asarray(agg["R_values"], dtype=int)

    median = np.asarray(
        agg["median_raw_optimal_probability_by_R"],
        dtype=float,
    )
    q1 = np.asarray(
        agg["q1_raw_optimal_probability_by_R"],
        dtype=float,
    )
    q3 = np.asarray(
        agg["q3_raw_optimal_probability_by_R"],
        dtype=float,
    )

    # Convert from probability [0,1] to percentage [%]
    median *= 100.0
    q1 *= 100.0
    q3 *= 100.0

    n_instances = int(agg["number_instances_total"])

    return R, median, q1, q3, n_instances


def load_instance_probabilities_at_R(data, R_target):
    out = {}

    for instance in data["instances"]:
        run = int(instance["run"])
        p = float(
            instance["R_results"][str(R_target)]["raw_optimal_probability"]
        )
        out[run] = p * 100.0  # percentage

    return out


# ============================================================
# Figure 8: Median + IQR vs R
# ============================================================

def plot_median_iqr_probability(at_data, by_data):
    R_at, median_at, q1_at, q3_at, n_at = load_aggregate_probability_statistics(at_data)
    R_by, median_by, q1_by, q3_by, n_by = load_aggregate_probability_statistics(by_data)

    if not np.array_equal(R_at, R_by):
        raise ValueError("At and By do not contain the same R values.")

    if n_at != n_by:
        raise ValueError("At and By do not contain the same number of instances.")

    R = R_at

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    ax.fill_between(
        R, q1_at, q3_at,
        color=AT_COLOR,
        alpha=0.18,
        linewidth=0,
    )

    ax.fill_between(
        R, q1_by, q3_by,
        color=BY_COLOR,
        alpha=0.18,
        linewidth=0,
    )

    ax.plot(
        R, median_at,
        color=AT_COLOR,
        marker="o",
        markersize=3.0,
        linewidth=1.2,
        label="At",
    )

    ax.plot(
        R, median_by,
        color=BY_COLOR,
        marker="s",
        markersize=3.0,
        linewidth=1.2,
        label="By",
    )

    ax.set_xlabel(r"Maximum BF-DCQO iterations, $R$")
    ax.set_ylabel(r"Median $P_{\mathrm{opt}}(R)$ (%)")

    ax.set_xlim(1, 30)
    ax.set_ylim(bottom=0)

    ax.set_xticks(np.arange(5, 31, 5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.grid(True, which="major", linewidth=0.35, alpha=0.25)

    ax.tick_params(which="major", length=4, width=0.8)
    ax.tick_params(which="minor", length=2, width=0.6)

    ax.legend(
        loc="upper left",
        frameon=False,
        handlelength=2.0,
        handletextpad=0.5,
    )

    fig.tight_layout()

    png_path = OUTPUT_DIR / "Figure08.png"

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    return {
        "png": png_path,
        "R": R,
        "median_at": median_at,
        "q1_at": q1_at,
        "q3_at": q3_at,
        "median_by": median_by,
        "q1_by": q1_by,
        "q3_by": q3_by,
        "n_instances": n_at,
    }


# ============================================================
# Figure 9: Paired comparison At vs By at R = 30
# ============================================================

def plot_paired_scatter(at_data, by_data, R_target):
    at = load_instance_probabilities_at_R(at_data, R_target)
    by = load_instance_probabilities_at_R(by_data, R_target)

    common_runs = sorted(set(at) & set(by))

    if not common_runs:
        raise RuntimeError("No paired instances were found.")

    x_at = np.asarray(
        [at[i] for i in common_runs],
        dtype=float
    )

    y_by = np.asarray(
        [by[i] for i in common_runs],
        dtype=float
    )

    # ========================================================
    # Paired statistics
    # ========================================================

    tol = 1e-15

    n_by_better = int(
        np.sum(y_by > x_at + tol)
    )

    n_at_better = int(
        np.sum(x_at > y_by + tol)
    )

    n_equal = (
        len(common_runs)
        - n_by_better
        - n_at_better
    )

    # ========================================================
    # Figure
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(3.5, 3.0)
    )

    # One point for each paired random instance
    ax.scatter(
        x_at,
        y_by,
        s=18,
        marker="o",
        facecolor=BY_COLOR,
        edgecolor="black",
        linewidth=0.35,
        alpha=0.72,
    )

    # ========================================================
    # Equality line: At = By
    # ========================================================

    max_val = max(
        float(np.max(x_at)),
        float(np.max(y_by))
    )

    upper = (
        max_val * 1.06
        if max_val > 0
        else 1.0
    )

    ax.plot(
        [0, upper],
        [0, upper],
        linestyle="--",
        linewidth=1.0,
        color="black",
        label=r"$P_{\mathrm{opt,By}}=P_{\mathrm{opt,At}}$",
    )

    # ========================================================
    # Axes
    # ========================================================

    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)

    ax.set_xlabel(
        rf"$P_{{\mathrm{{opt,At}}}}(R={R_target})$ (\%)"
    )

    ax.set_ylabel(
        rf"$P_{{\mathrm{{opt,By}}}}(R={R_target})$ (\%)"
    )

    ax.xaxis.set_minor_locator(
        AutoMinorLocator()
    )

    ax.yaxis.set_minor_locator(
        AutoMinorLocator()
    )

    # ========================================================
    # Grid and ticks
    # ========================================================

    ax.grid(
        True,
        which="major",
        linewidth=0.35,
        alpha=0.25
    )

    ax.tick_params(
        which="major",
        length=4,
        width=0.8
    )

    ax.tick_params(
        which="minor",
        length=2,
        width=0.6
    )

    # ========================================================
    # Legend
    # ========================================================
    # Moved slightly to the right and downward so it does not
    # overlap the y=5 region and occupies the empty central area.

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.22, 0.82),
        frameon=False,
        handlelength=2.0,
        handletextpad=0.5,
    )

    # ========================================================
    # Save
    # ========================================================

    fig.tight_layout()

    png_path = (
        OUTPUT_DIR
        / "Figure09.png"
    )

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ========================================================
    # Return statistics
    # ========================================================

    return {
        "png": png_path,
        "x_at": x_at,
        "y_by": y_by,
        "n_total": len(common_runs),
        "n_by_better": n_by_better,
        "n_at_better": n_at_better,
        "n_equal": n_equal,
    }


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    at_data = load_json(AT_JSON)
    by_data = load_json(BY_JSON)

    fig1 = plot_median_iqr_probability(at_data, by_data)
    fig2 = plot_paired_scatter(at_data, by_data, R_TARGET_PAIRED)

    print(f"Instances per R: {fig1['n_instances']}")
    print()

    print("Figure 8: median + IQR")
    print(
        f"  R = 30, At median = {fig1['median_at'][-1]:.6f}% "
        f"(IQR: {fig1['q1_at'][-1]:.6f}% - {fig1['q3_at'][-1]:.6f}%)"
    )
    print(
        f"  R = 30, By median = {fig1['median_by'][-1]:.6f}% "
        f"(IQR: {fig1['q1_by'][-1]:.6f}% - {fig1['q3_by'][-1]:.6f}%)"
    )
    print(f"  Saved: {fig1['png']}")
    print()

    print(f"Figure 9: paired comparison at R = {R_TARGET_PAIRED}")
    print(f"  By better than At: {fig2['n_by_better']}")
    print(f"  At better than By: {fig2['n_at_better']}")
    print(f"  Equal: {fig2['n_equal']}")
    print(f"  Saved: {fig2['png']}")


if __name__ == "__main__":
    main()
