from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REVIEW_ROOT = ROOT / "checkpoints" / "reviewer_revision"
FIG_ROOT = ROOT / "figures"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_sensitivity() -> None:
    summary = load_json(REVIEW_ROOT / "sensitivity_summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)

    colors = {"iemocap": "#1f77b4", "meld": "#d62728"}
    markers = {"iemocap": "o", "meld": "s"}

    for dataset in ["iemocap", "meld"]:
        x = [0.1, 0.3, 0.5, 0.7]
        y = [summary[dataset]["mask_prob"][str(value)]["test_weighted_f1"] * 100 for value in x]
        axes[0].plot(x, y, marker=markers[dataset], color=colors[dataset], linewidth=2.2, label=dataset.upper())

    axes[0].set_xlabel("Mask probability p")
    axes[0].set_ylabel("Weighted-F1 (%)")
    axes[0].set_title("Sensitivity to masking probability")
    axes[0].set_xticks([0.1, 0.3, 0.5, 0.7])
    style_axes(axes[0])
    axes[0].legend(frameon=False)

    for dataset in ["iemocap", "meld"]:
        x = [0.8, 0.9, 0.99]
        y = [summary[dataset]["beta"][str(value)]["test_weighted_f1"] * 100 for value in x]
        axes[1].plot(x, y, marker=markers[dataset], color=colors[dataset], linewidth=2.2, label=dataset.upper())

    axes[1].set_xlabel("Momentum decay β")
    axes[1].set_ylabel("Weighted-F1 (%)")
    axes[1].set_title("Sensitivity to EMA decay")
    axes[1].set_xticks([0.8, 0.9, 0.99])
    axes[1].set_xticklabels(["0.8", "0.9", "0.99"])
    style_axes(axes[1])

    for suffix in ["pdf", "png"]:
        fig.savefig(FIG_ROOT / f"reviewer_sensitivity.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def smooth(values: list[float], window: int = 5) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if len(array) < window:
        return array
    kernel = np.ones(window, dtype=np.float32) / window
    padded = np.pad(array, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot_gradient_conflict() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    colors = {"adamw": "#1f77b4", "magma": "#ff7f0e"}

    histories = {}
    for optimizer_name in ["adamw", "magma"]:
        path = REVIEW_ROOT / "gradient_conflict" / optimizer_name / "history.json"
        histories[optimizer_name] = load_json(path)

    for optimizer_name, history in histories.items():
        epochs = history["epoch"]
        ta = smooth(history["grad_cosine_text_audio"], window=5)
        av = smooth(history["grad_cosine_audio_visual"], window=5)
        axes[0].plot(epochs, ta, linewidth=2.2, color=colors[optimizer_name], label=optimizer_name.upper())
        axes[1].plot(epochs, av, linewidth=2.2, color=colors[optimizer_name], label=optimizer_name.upper())

    axes[0].axhline(0.0, color="#666666", linestyle=":", linewidth=1.0)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Gradient cosine")
    axes[0].set_title("Text-audio conflict trajectory")
    style_axes(axes[0])
    axes[0].legend(frameon=False)

    axes[1].axhline(0.0, color="#666666", linestyle=":", linewidth=1.0)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Gradient cosine")
    axes[1].set_title("Audio-visual alignment trajectory")
    style_axes(axes[1])

    for suffix in ["pdf", "png"]:
        fig.savefig(FIG_ROOT / f"reviewer_gradient_conflict.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    plot_sensitivity()
    plot_gradient_conflict()


if __name__ == "__main__":
    main()
