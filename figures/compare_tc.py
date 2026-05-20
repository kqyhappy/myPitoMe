from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "tc_outputs"
FIGURE_PATH = Path(__file__).resolve().parent / "tc_flops_accuracy_tradeoff.png"

FILES = {
    "sst2": OUTPUT_DIR / "eval_tc_BERT-B_sst2.csv",
    "rotten": OUTPUT_DIR / "eval_tc_BERT-B_rotten.csv",
    "imdb": OUTPUT_DIR / "eval_tc_BERT-B_imdb.csv",
}

DISPLAY_NAMES = {
    "sst2": "SST-2",
    "rotten": "Rotten Tomatoes",
    "imdb": "IMDb",
}

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


def load_results() -> pd.DataFrame:
    frames = []
    for dataset, path in FILES.items():
        df = pd.read_csv(path)
        df["dataset"] = dataset
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data = data[data["algo"].isin(["none", "pitome", "tome"])]
    return data.drop_duplicates(subset=["dataset", "algo", "ratio"], keep="last")


def plot_tradeoff(data: pd.DataFrame) -> None:
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), sharey=False)
    fig.patch.set_facecolor("white")

    for ax, dataset in zip(axes, FILES):
        sub = data[data["dataset"] == dataset].copy()
        none = sub[sub["algo"] == "none"].sort_values("ratio").iloc[-1]

        for algo in ["pitome", "tome"]:
            curve = sub[sub["algo"] == algo].sort_values("ratio", ascending=True)
            ax.plot(
                curve["gflops"],
                curve["acc"],
                marker=MARKERS[algo],
                linewidth=2.2,
                markersize=5.8,
                color=COLORS[algo],
                label=LABELS[algo],
            )

        ax.scatter(
            [none["gflops"]],
            [none["acc"]],
            marker=MARKERS["none"],
            s=190,
            color=COLORS["none"],
            edgecolor="#4D4D4D",
            linewidth=0.7,
            zorder=5,
            label=LABELS["none"],
        )

        ax.set_title(DISPLAY_NAMES[dataset], fontsize=18, weight="bold")
        ax.set_xlabel("GFLOPs", fontsize=16, weight="bold")
        ax.grid(True, color="#DDDDDD", linewidth=0.8, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(max(0, sub["gflops"].min() * 0.85), sub["gflops"].max() * 1.08)
        ax.set_ylim(max(50, sub["acc"].min() - 3), min(100, sub["acc"].max() + 2))
        ax.tick_params(axis="x", direction="in", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)

    axes[0].set_ylabel("Accuracy (%)", fontsize=16, weight="bold")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    plot_tradeoff(load_results())
