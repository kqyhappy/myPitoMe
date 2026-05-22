#!/usr/bin/env python3
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RESULT_CANDIDATES = [
    ROOT / "outputs" / "ic_output" / "eval-DEIT-T-224.csv",
    ROOT / "outputs" / "ic_outputs" / "eval-DEIT-T-224.csv",
]

COLORS = {
    "pitome": "#0072B2",
    "tome": "#D55E00",
    "none": "#E6A700",
}

MARKERS = {
    "pitome": "o",
    "tome": "s",
    "none": "*",
}

LABELS = {
    "pitome": "PiToMe",
    "tome": "ToMe",
    "none": "No-compress",
}


def find_results_path() -> Path:
    for path in RESULT_CANDIDATES:
        if path.is_file():
            return path
    candidates = "\n".join(f"  {path}" for path in RESULT_CANDIDATES)
    raise FileNotFoundError(f"Could not find image classification results:\n{candidates}")


def load_results() -> pd.DataFrame:
    data = pd.read_csv(find_results_path())
    data.columns = [column.strip() for column in data.columns]
    data["algo"] = data["algo"].str.strip()
    data["model"] = data["model"].str.strip()
    data["ratio"] = data["ratio"].astype(float)
    data["gflops"] = data["gflops"].astype(float)
    data["acc_1"] = data["acc_1"].astype(float)
    data = data[data["algo"].isin(["none", "pitome", "tome"])]
    return data.drop_duplicates(subset=["model", "algo", "ratio"], keep="last")


def plot_tradeoff(data: pd.DataFrame) -> Path:
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
        }
    )

    model = data["model"].iloc[0]
    none = data[data["algo"] == "none"].sort_values("ratio").iloc[-1]

    fig, ax = plt.subplots(figsize=(6.7, 4.4))
    fig.patch.set_facecolor("white")

    for algo in ["pitome", "tome"]:
        curve = data[data["algo"] == algo].sort_values("ratio", ascending=True)
        ax.plot(
            curve["gflops"],
            curve["acc_1"],
            marker=MARKERS[algo],
            linewidth=2.2,
            markersize=6.0,
            color=COLORS[algo],
            label=LABELS[algo],
        )
        for _, row in curve.iterrows():
            ax.annotate(
                f'{row["ratio"]:g}',
                (row["gflops"], row["acc_1"]),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=12,
                color=COLORS[algo],
            )

    ax.scatter(
        [none["gflops"]],
        [none["acc_1"]],
        marker=MARKERS["none"],
        s=210,
        color=COLORS["none"],
        edgecolor="#4D4D4D",
        linewidth=0.7,
        zorder=5,
        label=LABELS["none"],
    )
    ax.annotate(
        "1.0",
        (none["gflops"], none["acc_1"]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=12,
        color="#4D4D4D",
    )

    ax.set_title(model, fontsize=18, weight="bold")
    ax.set_xlabel("GFLOPs", fontsize=16, weight="bold")
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=16, weight="bold")
    ax.grid(True, color="#DDDDDD", linewidth=0.8, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(max(0, data["gflops"].min() * 0.84), data["gflops"].max() * 1.08)
    ax.set_ylim(max(0, data["acc_1"].min() - 4), min(100, data["acc_1"].max() + 3))
    ax.tick_params(axis="x", direction="in", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.legend(loc="lower right", frameon=False, fontsize=13)
    fig.tight_layout()

    FIGURES.mkdir(exist_ok=True)
    figure_path = FIGURES / "ic_flops_accuracy_tradeoff.png"
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def main() -> None:
    figure_path = plot_tradeoff(load_results())
    print(figure_path)


if __name__ == "__main__":
    main()
