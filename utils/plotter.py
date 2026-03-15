from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.patheffects as pe
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

try:
    import scienceplots  # noqa: F401

    plt.style.use(["science", "grid", "no-latex"])
except Exception:
    sns.set_theme(style="whitegrid")

plt.rcParams["text.usetex"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300


def plot_histories(
    histories: dict[str, dict[str, list[float]]],
    output_path: str | Path,
    metric_key: str = "weighted_f1",
    metric_label: str = "Weighted-F1",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=300)

    for name, history in histories.items():
        epochs = history.get("epoch", [])
        axes[0].plot(epochs, history.get("train_loss", []), linewidth=2.2, label=f"{name} train")
        axes[0].plot(epochs, history.get("val_loss", []), linewidth=2.2, linestyle="--", label=f"{name} val")
        axes[1].plot(epochs, history.get(f"val_{metric_key}", []), linewidth=2.2, label=f"{name} val")
        axes[1].plot(epochs, history.get(f"test_{metric_key}", []), linewidth=2.2, linestyle="--", label=f"{name} test")

    axes[0].set_title("Convergence of Training and Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[1].set_title(f"Convergence of Validation and Test {metric_label}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(metric_label)
    for axis in axes:
        axis.legend(frameon=False)
        axis.margins(x=0.02)
    figure.tight_layout(pad=1.1)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: list[str],
    output_path: str | Path,
    title: str = "Confusion Matrix",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.4, 6.2), dpi=300)
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        square=True,
        linewidths=0.6,
        linecolor="white",
        ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")
    axis.tick_params(axis="x", rotation=35)
    axis.tick_params(axis="y", rotation=0)
    figure.tight_layout(pad=1.1)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def plot_layerwise_masking(
    layerwise_history: list[dict[str, dict[str, float]]],
    output_path: str | Path,
    metric_key: str = "masked_ratio",
    title: str = "Late-epoch layer-wise masking rate",
    trailing_epochs: int = 5,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not layerwise_history:
        return

    selected = layerwise_history[-min(len(layerwise_history), trailing_epochs) :]
    buckets = sorted({bucket for epoch_stats in selected for bucket in epoch_stats})
    if not buckets:
        return

    averages = []
    for bucket in buckets:
        values = [epoch_stats.get(bucket, {}).get(metric_key, 0.0) for epoch_stats in selected]
        averages.append(float(np.mean(values)))

    figure, axis = plt.subplots(figsize=(8.2, 4.6), dpi=300)
    bars = axis.bar(buckets, averages, color=["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2", "#937860"][: len(buckets)])
    axis.set_title(title)
    axis.set_ylabel(metric_key.replace("_", " ").title())
    axis.set_ylim(0.0, max(0.05, max(averages) * 1.2))
    axis.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, averages):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    figure.tight_layout(pad=1.0)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def write_architecture_mermaid(output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mermaid = """flowchart TD
    A[Text CLS Embeddings] --> B[Text Projection]
    C[openSMILE Audio] --> D[Audio Projection]
    E[DenseNet Visual] --> F[Visual Projection]
    B --> G[EV-Gate Distance Estimator]
    D --> G
    F --> G
    G --> H[Stable Path]
    G --> I[Conflict Path]
    H --> J[Fused Utterance State]
    I --> J
    J --> K[Speaker Embedding Residual]
    K --> L[BiGRU Context Encoder]
    L --> M[Emotion Classifier]
    M --> N[Cross-Entropy Loss]
    N --> O[MoMask Gradient Filter]
    O --> P[AdamW Parameter Update]
    """
    output_path.write_text(mermaid, encoding="utf-8")


def _panel(axis, x, y, w, h, title, fc="#f6f9ff", ec="#d6e3ff"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.0,
        edgecolor=ec,
        facecolor=fc,
        alpha=0.95,
        zorder=0,
    )
    patch.set_path_effects([pe.SimplePatchShadow(offset=(2, -2), alpha=0.10), pe.Normal()])
    axis.add_patch(patch)
    axis.text(x + 0.015, y + h - 0.035, title, ha="left", va="center", fontsize=11, fontweight="bold", color="#243b53")


def _box(
    axis,
    x,
    y,
    w,
    h,
    title,
    subtitle: str | None = None,
    fc="#e9f2ff",
    ec="#4c72b0",
    title_size=11,
    subtitle_size=9,
    align="center",
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.6,
        edgecolor=ec,
        facecolor=fc,
        zorder=3,
    )
    patch.set_path_effects([pe.SimplePatchShadow(offset=(2.4, -2.4), alpha=0.12), pe.Normal()])
    axis.add_patch(patch)
    tx = x + w / 2 if align == "center" else x + 0.02
    ha = "center" if align == "center" else "left"
    ty = y + h * (0.60 if subtitle else 0.50)
    axis.text(tx, ty, title, ha=ha, va="center", fontsize=title_size, fontweight="bold", color="#1f2933", zorder=4)
    if subtitle:
        axis.text(tx, y + h * 0.33, subtitle, ha=ha, va="center", fontsize=subtitle_size, color="#52606d", zorder=4)


def _arrow(axis, start, end, color="#385170", lw=1.7, style="solid", curve=0.0, zorder=2):
    line = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=lw,
        color=color,
        linestyle=style,
        connectionstyle=f"arc3,rad={curve}",
        shrinkA=6,
        shrinkB=6,
        zorder=zorder,
    )
    axis.add_patch(line)
    return line


def _draw_document_icon(axis, x, y, w, h, color="#5b8bd9"):
    axis.add_patch(Rectangle((x, y), w, h, linewidth=1.4, edgecolor=color, facecolor="white", zorder=5))
    axis.add_patch(
        Polygon(
            [[x + w * 0.72, y + h], [x + w, y + h], [x + w, y + h * 0.72]],
            closed=True,
            edgecolor=color,
            facecolor="#eef4ff",
            linewidth=1.1,
            zorder=6,
        )
    )
    for ratio in [0.72, 0.54, 0.36]:
        axis.plot([x + w * 0.16, x + w * 0.82], [y + h * ratio, y + h * ratio], color=color, linewidth=1.4, zorder=6)


def _draw_wave_icon(axis, x, y, w, h, color="#3ca6a6"):
    xs = np.linspace(x, x + w, 60)
    base = y + h * 0.5
    ys = base + 0.23 * h * np.sin(np.linspace(0, 3.6 * np.pi, xs.size)) * np.cos(np.linspace(0, 1.2 * np.pi, xs.size))
    axis.plot(xs, ys, color=color, linewidth=2.0, zorder=6)
    axis.plot([x, x + w], [base, base], color="#b6dedd", linewidth=0.9, zorder=5)


def _draw_image_icon(axis, x, y, w, h, color="#8a74d6"):
    axis.add_patch(Rectangle((x, y), w, h, linewidth=1.4, edgecolor=color, facecolor="white", zorder=5))
    axis.add_patch(
        Circle(
            (x + w * 0.78, y + h * 0.77),
            radius=min(w, h) * 0.08,
            facecolor="#efeaff",
            edgecolor=color,
            linewidth=1.0,
            zorder=6,
        )
    )
    axis.add_patch(
        Polygon(
            [[x + w * 0.12, y + h * 0.18], [x + w * 0.38, y + h * 0.56], [x + w * 0.56, y + h * 0.34], [x + w * 0.82, y + h * 0.68], [x + w * 0.88, y + h * 0.18]],
            closed=False,
            fill=False,
            edgecolor=color,
            linewidth=1.8,
            zorder=6,
        )
    )


def _draw_projection_icon(axis, x, y, w, h, color="#7b8794"):
    heights = [0.25, 0.45, 0.68, 0.52, 0.32]
    gap = w * 0.06
    width = w * 0.10
    start = x + w * 0.12
    for idx, height in enumerate(heights):
        bx = start + idx * (width + gap)
        axis.add_patch(
            Rectangle((bx, y + h * 0.16), width, h * height, facecolor="#e7edf5", edgecolor=color, linewidth=1.0, zorder=6)
        )


def _draw_gate_icon(axis, x, y, w, h):
    center = (x + w * 0.52, y + h * 0.60)
    axis.add_patch(Circle(center, radius=min(w, h) * 0.18, facecolor="#fff8ef", edgecolor="#e6863b", linewidth=1.7, zorder=6))
    axis.add_patch(Circle((x + w * 0.28, y + h * 0.74), radius=min(w, h) * 0.05, facecolor="#5b8bd9", edgecolor="none", zorder=6))
    axis.add_patch(Circle((x + w * 0.28, y + h * 0.46), radius=min(w, h) * 0.05, facecolor="#3ca6a6", edgecolor="none", zorder=6))
    axis.add_patch(Circle((x + w * 0.28, y + h * 0.18), radius=min(w, h) * 0.05, facecolor="#8a74d6", edgecolor="none", zorder=6))
    axis.plot([x + w * 0.34, x + w * 0.44], [y + h * 0.74, y + h * 0.66], color="#5b8bd9", linewidth=1.6, zorder=6)
    axis.plot([x + w * 0.34, x + w * 0.44], [y + h * 0.46, y + h * 0.57], color="#3ca6a6", linewidth=1.6, zorder=6)
    axis.plot([x + w * 0.34, x + w * 0.44], [y + h * 0.18, y + h * 0.49], color="#8a74d6", linewidth=1.6, zorder=6)
    axis.plot([center[0], x + w * 0.80], [center[1], y + h * 0.72], color="#65a765", linewidth=2.0, zorder=6)
    axis.plot([center[0], x + w * 0.80], [center[1], y + h * 0.36], color="#d48b42", linewidth=2.0, zorder=6)
    axis.text(x + w * 0.79, y + h * 0.76, "stable", ha="center", va="center", fontsize=8.4, color="#5f9d5f", zorder=7)
    axis.text(x + w * 0.80, y + h * 0.30, "conflict", ha="center", va="center", fontsize=8.4, color="#d48b42", zorder=7)


def _draw_sequence_icon(axis, x, y, w, h, color="#6278d2"):
    xs = np.linspace(x + w * 0.12, x + w * 0.88, 6)
    ys = y + h * 0.48 + np.array([0.07, -0.03, 0.05, -0.04, 0.03, -0.02]) * h
    axis.plot(xs, ys, color="#9fb0f2", linewidth=1.5, zorder=5)
    for px, py in zip(xs, ys):
        axis.add_patch(Circle((px, py), radius=min(w, h) * 0.07, facecolor="white", edgecolor=color, linewidth=1.4, zorder=6))


def _draw_probability_icon(axis, x, y, w, h):
    heights = [0.25, 0.42, 0.68, 0.38, 0.55]
    colors = ["#c6d4ff", "#ffd3df", "#ff90b0", "#ffd3df", "#c6d4ff"]
    gap = w * 0.05
    bar_width = w * 0.12
    start = x + w * 0.15
    for idx, (height, color) in enumerate(zip(heights, colors)):
        bx = start + idx * (bar_width + gap)
        axis.add_patch(
            Rectangle((bx, y + h * 0.18), bar_width, h * height, facecolor=color, edgecolor="#d25a78", linewidth=1.0, zorder=6)
        )


def _draw_loss_icon(axis, x, y, w, h):
    axis.add_patch(Circle((x + w * 0.35, y + h * 0.56), radius=min(w, h) * 0.17, facecolor="#fffdf5", edgecolor="#d6a13a", linewidth=1.4, zorder=6))
    axis.text(x + w * 0.35, y + h * 0.56, "y", ha="center", va="center", fontsize=10, color="#9c6f11", zorder=7, fontweight="bold")
    axis.plot([x + w * 0.52, x + w * 0.82], [y + h * 0.56, y + h * 0.56], color="#d6a13a", linewidth=1.6, zorder=6)
    axis.text(x + w * 0.86, y + h * 0.56, "L", ha="center", va="center", fontsize=10, color="#9c6f11", zorder=7, fontweight="bold")


def _draw_momask_icon(axis, x, y, w, h):
    cols, rows = 5, 3
    cell_w = w * 0.11
    cell_h = h * 0.12
    start_x = x + w * 0.14
    start_y = y + h * 0.26
    for row in range(rows):
        for col in range(cols):
            face = "#ede7ff"
            if (row, col) in {(0, 1), (1, 3), (2, 2)}:
                face = "#ffffff"
            axis.add_patch(
                Rectangle(
                    (start_x + col * cell_w * 1.25, start_y + row * cell_h * 1.25),
                    cell_w,
                    cell_h,
                    facecolor=face,
                    edgecolor="#8d72cc",
                    linewidth=1.0,
                    zorder=6,
                )
            )
    axis.plot([x + w * 0.12, x + w * 0.88], [y + h * 0.74, y + h * 0.74], color="#8d72cc", linewidth=1.5, zorder=6)
    axis.plot([x + w * 0.12, x + w * 0.33], [y + h * 0.74, y + h * 0.81], color="#8d72cc", linewidth=1.5, zorder=6)
    axis.plot([x + w * 0.56, x + w * 0.88], [y + h * 0.81, y + h * 0.67], color="#8d72cc", linewidth=1.5, zorder=6)
    axis.text(x + w * 0.50, y + h * 0.86, "mask", ha="center", va="center", fontsize=8.2, color="#7a5cc2", zorder=7)


def write_architecture_figure(output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.8, 7.8), dpi=300)
    fig.patch.set_facecolor("#fbfdff")
    ax.set_facecolor("#fbfdff")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _panel(ax, 0.03, 0.12, 0.30, 0.76, "Utterance encoders")
    _panel(ax, 0.36, 0.12, 0.26, 0.76, "Conflict-aware fusion")
    _panel(ax, 0.66, 0.12, 0.29, 0.76, "Context reasoning and optimization")

    _box(ax, 0.06, 0.68, 0.17, 0.13, "Text", "RoBERTa", fc="#eaf2ff", ec="#5b8bd9")
    _box(ax, 0.06, 0.47, 0.17, 0.13, "Audio", "openSMILE", fc="#e8fbff", ec="#3ca6a6")
    _box(ax, 0.06, 0.26, 0.17, 0.13, "Visual", "DenseNet", fc="#f2edff", ec="#8a74d6")
    _draw_document_icon(ax, 0.076, 0.710, 0.040, 0.055, color="#5b8bd9")
    _draw_wave_icon(ax, 0.074, 0.510, 0.050, 0.045, color="#3ca6a6")
    _draw_image_icon(ax, 0.075, 0.290, 0.046, 0.052, color="#8a74d6")

    _box(ax, 0.25, 0.68, 0.07, 0.13, "Proj.", "$d$", fc="#f4f7fb", ec="#c0ccd9", title_size=10, subtitle_size=9)
    _box(ax, 0.25, 0.47, 0.07, 0.13, "Proj.", "$d$", fc="#f4f7fb", ec="#c0ccd9", title_size=10, subtitle_size=9)
    _box(ax, 0.25, 0.26, 0.07, 0.13, "Proj.", "$d$", fc="#f4f7fb", ec="#c0ccd9", title_size=10, subtitle_size=9)
    _draw_projection_icon(ax, 0.258, 0.708, 0.055, 0.048)
    _draw_projection_icon(ax, 0.258, 0.498, 0.055, 0.048)
    _draw_projection_icon(ax, 0.258, 0.288, 0.055, 0.048)

    _box(ax, 0.41, 0.53, 0.16, 0.22, "EV-Gate", "expectancy-violation routing", fc="#fff1e5", ec="#e6863b", title_size=12)
    _draw_gate_icon(ax, 0.425, 0.555, 0.13, 0.14)
    _box(ax, 0.41, 0.18, 0.16, 0.10, "Speaker residual", None, fc="#eef8ef", ec="#59a56d", title_size=10)
    ax.text(0.49, 0.205, r"$h_t + e_{spk}$", ha="center", va="center", fontsize=9, color="#527f5a", zorder=6)

    _box(ax, 0.71, 0.61, 0.19, 0.12, "BiGRU", "dialogue encoder", fc="#eef2ff", ec="#6278d2")
    _draw_sequence_icon(ax, 0.735, 0.635, 0.14, 0.055)
    _box(ax, 0.71, 0.42, 0.19, 0.11, "MLP + Softmax", "emotion head", fc="#fff0f4", ec="#d25a78")
    _draw_probability_icon(ax, 0.735, 0.438, 0.14, 0.05)
    _box(ax, 0.71, 0.23, 0.08, 0.09, "Loss", None, fc="#fff6e7", ec="#d6a13a", title_size=10)
    _draw_loss_icon(ax, 0.718, 0.238, 0.064, 0.05)
    _box(ax, 0.82, 0.21, 0.11, 0.13, "MoMask", "masked gradient", fc="#f4efff", ec="#8d72cc", title_size=11, subtitle_size=8)
    _draw_momask_icon(ax, 0.831, 0.228, 0.088, 0.08)

    _arrow(ax, (0.23, 0.745), (0.25, 0.745), color="#4f6b8a")
    _arrow(ax, (0.23, 0.535), (0.25, 0.535), color="#4f6b8a")
    _arrow(ax, (0.23, 0.325), (0.25, 0.325), color="#4f6b8a")

    _arrow(ax, (0.32, 0.745), (0.41, 0.66), color="#4f6b8a")
    _arrow(ax, (0.32, 0.535), (0.41, 0.64), color="#4f6b8a")
    _arrow(ax, (0.32, 0.325), (0.41, 0.62), color="#4f6b8a")

    _arrow(ax, (0.49, 0.53), (0.44, 0.37), color="#5f9d5f", curve=0.06)
    _arrow(ax, (0.49, 0.53), (0.54, 0.37), color="#d48b42", curve=-0.06)
    _arrow(ax, (0.44, 0.28), (0.49, 0.205), color="#5f9d5f")
    _arrow(ax, (0.54, 0.28), (0.49, 0.205), color="#d48b42")

    _arrow(ax, (0.57, 0.205), (0.71, 0.67), color="#4f6b8a", curve=0.12)
    _arrow(ax, (0.57, 0.61), (0.71, 0.67), color="#4f6b8a")
    _arrow(ax, (0.81, 0.61), (0.81, 0.53), color="#4f6b8a")
    _arrow(ax, (0.81, 0.42), (0.75, 0.32), color="#b5556a")
    _arrow(ax, (0.79, 0.275), (0.82, 0.275), color="#8d72cc")
    _arrow(ax, (0.82, 0.21), (0.58, 0.79), color="#8d72cc", style="dashed", curve=-0.30, zorder=1)

    ax.text(0.50, 0.94, "Conflict-aware MERC with EV-Gate fusion and MoMask optimization", ha="center", va="center", fontsize=15, fontweight="bold", color="#102a43")
    ax.text(0.50, 0.905, "Solid arrows denote forward inference; the dashed arc denotes masked optimization feedback.", ha="center", va="center", fontsize=9.5, color="#52606d")
    ax.text(0.49, 0.41, "utterance fusion", ha="center", va="bottom", fontsize=8.5, color="#7b8794")
    ax.text(0.875, 0.36, "backpropagation", ha="center", va="bottom", fontsize=8.5, color="#7b8794")

    fig.tight_layout(pad=0.5)
    figure_format = output_path.suffix.lower().replace(".", "") or "pdf"
    fig.savefig(output_path, format=figure_format, bbox_inches="tight")
    plt.close(fig)
