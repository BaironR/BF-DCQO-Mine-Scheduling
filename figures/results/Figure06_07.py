from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"

AT_FILE = BASE_DIR / "../../data/At_results_independent_R.json"
BY_FILE = BASE_DIR / "../../data/By_results_independent_R.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
COLOR_AT = "#1f77b4"
COLOR_BY = "#ff7f0e"
COLOR_OPT = "#d62728"
HISTOGRAM_BINS = 80
HISTOGRAM_X_MIN = -5.0
HISTOGRAM_X_MAX = 40.0

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 6.5,
    "axes.labelsize": 6.5,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6.5,
    "mathtext.fontset": "cm",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.major.size": 4,
    "ytick.minor.size": 2,
    "axes.linewidth": 0.8,
})


def load_results(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path.resolve()}"
        )
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_results(data: dict, expected_formulation: str) -> None:
    required_keys = [
        "formulation",
        "R_values",
        "optimal_energy",
        "raw_terminal_optimal_probability_by_R",
        "runs",
    ]
    missing = [key for key in required_keys if key not in data]
    if missing:
        raise KeyError(
            f"Missing fields in {expected_formulation}: {missing}"
        )
    if data["formulation"] != expected_formulation:
        raise ValueError(
            f"Expected formulation {expected_formulation}, "
            f"but the JSON indicates {data['formulation']}."
        )


def format_axes(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)

    ax.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
    )
    ax.minorticks_on()
    ax.grid(
        axis="both",
        linestyle="-",
        alpha=0.15,
        linewidth=0.5,
    )


def save_and_close(fig, filename: str) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    path = OUTPUT_DIR / filename
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close(fig)
    print("Figure saved:", path)


def terminal_mean_energy(data: dict) -> np.ndarray:
    return np.asarray(
        [
            run["mean_energy_by_iteration"][-1]
            for run in data["runs"]
        ],
        dtype=float,
    )


def terminal_std_energy(data: dict) -> np.ndarray:
    return np.asarray(
        [
            run["std_energy_by_iteration"][-1]
            for run in data["runs"]
        ],
        dtype=float,
    )


def terminal_histogram_data(data: dict):
    frequencies = data["runs"][-1][
        "terminal_energy_frequencies"
    ]

    energies = np.asarray(
        [float(energy) for energy in frequencies.keys()],
        dtype=float,
    )

    counts = np.asarray(
        [int(count) for count in frequencies.values()],
        dtype=np.int64,
    )

    return energies, counts


def weighted_histogram(energies, counts, bins):
    hist, edges = np.histogram(
        energies,
        bins=bins,
        weights=counts,
    )
    hist = (
        hist.astype(float)
        / float(np.sum(counts))
    )
    return hist, edges


def common_histogram_bins(at_energy, at_count, by_energy, by_count, bins_count):
    all_energies = np.concatenate([at_energy, by_energy])
    all_weights = np.concatenate([at_count, by_count])
    return np.histogram_bin_edges(
        all_energies,
        bins=bins_count,
        range=(HISTOGRAM_X_MIN, HISTOGRAM_X_MAX),
        weights=all_weights,
    )


def plot_raw_optimal_probability(at: dict, by: dict, R: np.ndarray) -> None:
    p_at = (
        100.0
        * np.asarray(
            at["raw_terminal_optimal_probability_by_R"],
            dtype=float,
        )
    )

    p_by = (
        100.0
        * np.asarray(
            by["raw_terminal_optimal_probability_by_R"],
            dtype=float,
        )
    )

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    ax.plot(
        R,
        p_at,
        color=COLOR_AT,
        marker="o",
        markersize=2.6,
        markeredgewidth=0.4,
        linewidth=1.0,
        label="At",
    )

    ax.plot(
        R,
        p_by,
        color=COLOR_BY,
        marker="s",
        markersize=2.6,
        markeredgewidth=0.4,
        linewidth=1.0,
        label="By",
    )

    ax.set_xlabel(
        r"Maximum BF-DCQO iterations, $R$",
        labelpad=3,
    )

    ax.set_ylabel(
        r"$P_{\mathrm{opt}}(R)$ (%)",
        labelpad=3,
    )

    ax.set_xticks(
        np.arange(
            5,
            int(R[-1]) + 1,
            5,
        )
    )

    ax.set_xlim(
        1,
        int(R[-1]),
    )

    format_axes(ax)

    ax.legend(
        frameon=False,
        loc="upper left",
        handlelength=1.8,
        handletextpad=0.5,
        borderaxespad=0.8,
    )

    save_and_close(
        fig,
        "Figure06.png",
    )



def plot_terminal_energy_histogram(
    at: dict,
    by: dict,
    R: np.ndarray,
    optimal_energy: float,
) -> None:
    at_energy, at_count = terminal_histogram_data(at)
    by_energy, by_count = terminal_histogram_data(by)

    common_bins = common_histogram_bins(
        at_energy,
        at_count,
        by_energy,
        by_count,
        HISTOGRAM_BINS,
    )

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    ax.hist(
        at_energy,
        bins=common_bins,
        weights=at_count,
        density=True,
        alpha=0.55,
        color=COLOR_AT,
        edgecolor="#0f3b5a",
        linewidth=0.5,
        hatch="//",
        label="At",
    )

    ax.hist(
        by_energy,
        bins=common_bins,
        weights=by_count,
        density=True,
        alpha=0.45,
        color=COLOR_BY,
        edgecolor="#8c4300",
        linewidth=0.5,
        hatch="\\",
        label="By",
    )

    ax.axvline(
        optimal_energy,
        color=COLOR_OPT,
        linestyle="--",
        linewidth=1.2,
        label=r"$E_{\mathrm{opt}}$",
    )

    ax.text(
        x=optimal_energy - 3.0,
        y=1.08,
        s=rf"${optimal_energy:.2f}$",
        transform=ax.get_xaxis_transform(),
        color=COLOR_OPT,
        fontsize=7,
        fontweight="bold",
        verticalalignment="top",
    )

    ax.set_xlabel(
        "QUBO energy",
        labelpad=3,
    )

    ax.set_ylabel(
        "Probability density",
        labelpad=3,
    )

    ax.set_xlim(
        HISTOGRAM_X_MIN,
        HISTOGRAM_X_MAX,
    )

    ax.set_xticks(
        np.arange(
            HISTOGRAM_X_MIN,
            HISTOGRAM_X_MAX + 1,
            5,
        )
    )

    format_axes(ax)

    ax.legend(
        frameon=False,
        loc="upper right",
        handlelength=1.8,
        handletextpad=0.5,
        borderaxespad=0.8,
    )

    save_and_close(
        fig,
        "Figure07.png",
    )


def print_summary(at: dict, by: dict, R: np.ndarray) -> None:
    p_at = np.asarray(
        at["raw_terminal_optimal_probability_by_R"],
        dtype=float,
    )

    p_by = np.asarray(
        by["raw_terminal_optimal_probability_by_R"],
        dtype=float,
    )

    print("\n=== SUMMARY ===")

    print(
        f"At raw P_opt at R={R[-1]}: "
        f"{100.0 * p_at[-1]:.6f}%"
    )

    print(
        f"By raw P_opt at R={R[-1]}: "
        f"{100.0 * p_by[-1]:.6f}%"
    )

    if p_at[-1] > 0.0:
        print(
            f"By/At ratio at R={R[-1]}: "
            f"{p_by[-1] / p_at[-1]:.4f}"
        )


def main() -> None:
    at = load_results(AT_FILE)
    by = load_results(BY_FILE)

    validate_results(at, "At")
    validate_results(by, "By")

    R = np.asarray(
        at["R_values"],
        dtype=int,
    )

    R_by = np.asarray(
        by["R_values"],
        dtype=int,
    )

    if not np.array_equal(R, R_by):
        raise ValueError(
            "At and By do not have the same R values."
        )

    optimal_energy = float(
        at["optimal_energy"]
    )

    plot_raw_optimal_probability(
        at,
        by,
        R,
    )


    plot_terminal_energy_histogram(
        at,
        by,
        R,
        optimal_energy,
    )

    print_summary(
        at,
        by,
        R,
    )

    print(
        "\nAll figures were saved to:",
        OUTPUT_DIR.resolve(),
    )


if __name__ == "__main__":
    main()
