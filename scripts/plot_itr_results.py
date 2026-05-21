#!/usr/bin/env python3
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "itr_output" / "configured_results.csv"
FIGURES = ROOT / "figures"

COLORS = {
    "pitome": "#0072B2",
    "tome": "#D55E00",
    "none": "#E6A700",
}

LABELS = {
    "pitome": "PiToMe",
    "tome": "ToMe",
    "none": "No-compress",
}


def load_rows():
    with RESULTS.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["ratio"] = float(row["ratio"])
        row["Rsum"] = float(row["Rsum"])
    return rows


def plot_rsum(rows):
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
        }
    )

    labels = [f'{LABELS[r["method"]]}\nr={r["ratio"]:g}' for r in rows]
    values = [r["Rsum"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("white")
    ax.bar(labels, values, color=[COLORS[r["method"]] for r in rows])
    ax.set_ylabel("Rsum", fontsize=16, weight="bold")
    ax.set_ylim(550, 575)
    ax.grid(True, axis="y", color="#DDDDDD", linewidth=0.8, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", direction="in", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    for i, value in enumerate(values):
        ax.text(i, value + 0.35, f"{value:.2f}", ha="center", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "itr_rsum.png", dpi=220)
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    rows = load_rows()
    plot_rsum(rows)
    print(FIGURES / "itr_rsum.png")


if __name__ == "__main__":
    main()
