#!/usr/bin/env python3
"""Visualize simple sequence encodings and embedding spaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

try:
    import seaborn as sns
except ImportError:
    sns = None


STANDARD_AA = list("ACDEFGHIKLMNPQRSTVWY")
LABEL_COLORS = {
    1: "#91BFFA",
    0: "#FFFEDF",
    -1: "#B7E4B2",
}
LABEL_NAMES = {
    1: "1: DBP",
    0: "0: Non-DBP",
    -1: "-1: ambiguity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create embedding visualization figures.")
    parser.add_argument(
        "--input",
        default="washed_data/human_protein_table_embedded.tsv",
        help="TSV with embedded_sequence and average_pooling_embedding columns.",
    )
    parser.add_argument(
        "--outdir",
        default="Results/03_Embedding_results",
        help="Output directory for PNG figures.",
    )
    parser.add_argument(
        "--tsne-per-class",
        type=int,
        default=1000,
        help="Maximum number of samples per class for t-SNE visualization.",
    )
    parser.add_argument("--tsne-random-state", type=int, default=42)
    return parser.parse_args()


def load_json_vector(value: object) -> list[float]:
    return json.loads(str(value))


def first_slice(values: list[float], width: int = 30, pad: bool = False) -> list[float]:
    sliced = values[: min(len(values), width)]
    if pad and len(sliced) < width:
        sliced = sliced + [np.nan] * (width - len(sliced))
    return sliced


def one_hot_matrix(sequence: str, width: int = 30) -> tuple[np.ndarray, list[str]]:
    residues = list(str(sequence).strip().upper()[:width])
    aa_index = {aa: i for i, aa in enumerate(STANDARD_AA)}
    matrix = np.zeros((len(residues), len(STANDARD_AA)), dtype=float)
    for row, residue in enumerate(residues):
        col = aa_index.get(residue)
        if col is not None:
            matrix[row, col] = 1.0
    return matrix, residues


def aa_label_cmap() -> ListedColormap:
    if sns is not None:
        return ListedColormap(sns.color_palette("mako", n_colors=len(STANDARD_AA)).as_hex())
    return ListedColormap(plt.cm.viridis(np.linspace(0.15, 0.9, len(STANDARD_AA))))


def draw_cell_grid(ax: plt.Axes, n_rows: int, n_cols: int, linewidth: float = 0.55) -> None:
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=linewidth)
    ax.tick_params(which="minor", length=0)


def plot_encoding_heatmaps(df: pd.DataFrame, outdir: Path) -> None:
    first = df.iloc[0]
    one_hot, residue_labels = one_hot_matrix(first["sequence"])
    esm_first = np.array(first_slice(load_json_vector(first["embedded_sequence"]), 30), dtype=float)
    avg_first = np.array(first_slice(load_json_vector(first["average_pooling_embedding"]), 20), dtype=float)

    fig = plt.figure(figsize=(9.4, 8.0))
    grid = fig.add_gridspec(1, 5, width_ratios=[0.55, 5.0, 0.25, 0.25, 0.13], wspace=0.58)
    ax_origin = fig.add_subplot(grid[0, 0])
    ax_onehot = fig.add_subplot(grid[0, 1])
    ax_esm = fig.add_subplot(grid[0, 2])
    ax_pooling = fig.add_subplot(grid[0, 3])
    cax = fig.add_subplot(grid[0, 4])

    aa_to_color_index = {aa: i for i, aa in enumerate(STANDARD_AA)}
    origin_values = np.array(
        [aa_to_color_index.get(residue, np.nan) for residue in residue_labels],
        dtype=float,
    ).reshape(-1, 1)
    origin_cmap = aa_label_cmap()
    origin_cmap.set_bad("#D9D9D9")
    ax_origin.imshow(
        np.ma.masked_invalid(origin_values),
        aspect="equal",
        interpolation="nearest",
        cmap=origin_cmap,
        vmin=0,
        vmax=len(STANDARD_AA) - 1,
    )
    for row, residue in enumerate(residue_labels):
        ax_origin.text(0, row, residue, ha="center", va="center", fontsize=8, color="white")
    ax_origin.set_title("Origin", fontsize=9)
    ax_origin.set_xticks([])
    ax_origin.set_yticks([])
    draw_cell_grid(ax_origin, len(residue_labels), 1)
    ax_origin.text(0, len(residue_labels) + 0.65, "\u22ee", ha="center", va="center", fontsize=14, clip_on=False)

    onehot_cmap = ListedColormap(["#D9D9D9", "#91BFFA"])
    ax_onehot.imshow(one_hot, aspect="equal", interpolation="nearest", cmap=onehot_cmap, vmin=0, vmax=1)
    ax_onehot.set_yticks(np.arange(len(residue_labels)))
    ax_onehot.set_yticklabels(residue_labels)
    ax_onehot.set_xticks(np.arange(len(STANDARD_AA)))
    ax_onehot.set_xticklabels(STANDARD_AA)
    ax_onehot.set_xlabel("One-hot amino acid code")
    ax_onehot.set_ylabel("AA sequence")
    ax_onehot.tick_params(axis="both", length=0, labelsize=8)
    draw_cell_grid(ax_onehot, len(residue_labels), len(STANDARD_AA), linewidth=0.25)
    ax_onehot.text(
        -1.15,
        len(residue_labels) + 0.65,
        "\u22ee",
        ha="center",
        va="center",
        fontsize=14,
        clip_on=False,
    )

    vmax = float(max(np.nanmax(np.abs(esm_first)), np.nanmax(np.abs(avg_first))))
    vmax = vmax if vmax > 0 else 1.0
    vector_cmap = plt.get_cmap("coolwarm").copy()
    vector_cmap.set_bad("#D9D9D9")

    im = ax_esm.imshow(
        esm_first.reshape(-1, 1),
        aspect="equal",
        interpolation="nearest",
        cmap=vector_cmap,
        vmin=-vmax,
        vmax=vmax,
    )
    ax_pooling.imshow(
        np.ma.masked_invalid(avg_first.reshape(-1, 1)),
        aspect="equal",
        interpolation="nearest",
        cmap=vector_cmap,
        vmin=-vmax,
        vmax=vmax,
    )

    for ax, label, length in [
        (ax_esm, "ESM-embedding\nfirst 30/320D", len(esm_first)),
        (ax_pooling, "Avg Pooling\n20D", len(avg_first)),
    ]:
        ax.set_title(label, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        draw_cell_grid(ax, length, 1)
        ax.tick_params(axis="both", length=0)
        ax.text(0, length + 0.65, "\u22ee", ha="center", va="center", fontsize=14, clip_on=False)

    fig.colorbar(im, cax=cax)

    fig.savefig(outdir / "01_encoding_heatmaps.png", dpi=400, transparent=True, bbox_inches="tight")
    plt.close(fig)


def pca_numpy(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    x_centered = x - x.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(x_centered, full_matrices=False)
    scores = x_centered @ vt[:2].T
    explained = singular_values**2 / np.sum(singular_values**2)
    return scores[:, :2], explained[:2] * 100


def parse_matrix(series: pd.Series) -> np.ndarray:
    return np.vstack([np.array(load_json_vector(value), dtype=float) for value in series])


def scatter_pca(ax: plt.Axes, scores: np.ndarray, explained: np.ndarray, labels: np.ndarray, title: str) -> None:
    for label in [1, 0, -1]:
        mask = labels == label
        ax.scatter(
            scores[mask, 0],
            scores[mask, 1],
            s=13,
            c=LABEL_COLORS[label],
            edgecolors="black",
            linewidths=0.25,
            alpha=0.82,
            label=LABEL_NAMES[label],
        )
    ax.set_title(title)
    ax.set_xlabel("PC1 ({:.1f}%)".format(explained[0]))
    ax.set_ylabel("PC2 ({:.1f}%)".format(explained[1]))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)


def get_embedding_matrices(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = df["is_DBP"].astype(int).to_numpy()
    esm_matrix = parse_matrix(df["embedded_sequence"])
    avg_matrix = parse_matrix(df["average_pooling_embedding"])
    return labels, esm_matrix, avg_matrix


def add_label_legend(ax: plt.Axes) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=LABEL_COLORS[label],
            markeredgecolor="black",
            markersize=6,
            label=LABEL_NAMES[label],
        )
        for label in [1, 0, -1]
    ]
    ax.legend(handles=handles, frameon=True, edgecolor="black", loc="best")


def plot_embedding_pca(df: pd.DataFrame, outdir: Path) -> None:
    labels, esm_matrix, avg_matrix = get_embedding_matrices(df)
    esm_scores, esm_explained = pca_numpy(esm_matrix)
    avg_scores, avg_explained = pca_numpy(avg_matrix)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5), sharex=False, sharey=False)
    scatter_pca(axes[0], esm_scores, esm_explained, labels, "ESM")
    scatter_pca(axes[1], avg_scores, avg_explained, labels, "Pooling")

    add_label_legend(axes[1])

    fig.tight_layout()
    fig.savefig(outdir / "02_embedding_pca.png", dpi=400, transparent=True, bbox_inches="tight")
    plt.close(fig)


def stratified_sample(df: pd.DataFrame, per_class: int, random_state: int) -> pd.DataFrame:
    parts = []
    for label, group in df.groupby("is_DBP", sort=True):
        n = min(len(group), per_class)
        parts.append(group.sample(n=n, random_state=random_state))
    return pd.concat(parts, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def tsne_numpy(x: np.ndarray, random_state: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    perplexity = min(30, max(5, (len(x) - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    return tsne.fit_transform(x)


def scatter_embedding_2d(ax: plt.Axes, scores: np.ndarray, labels: np.ndarray, title: str) -> None:
    for label in [1, 0, -1]:
        mask = labels == label
        ax.scatter(
            scores[mask, 0],
            scores[mask, 1],
            s=13,
            c=LABEL_COLORS[label],
            edgecolors="black",
            linewidths=0.25,
            alpha=0.82,
            label=LABEL_NAMES[label],
        )
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_embedding_tsne(df: pd.DataFrame, outdir: Path, per_class: int, random_state: int) -> None:
    sampled = stratified_sample(df, per_class=per_class, random_state=random_state)
    labels = sampled["is_DBP"].astype(int).to_numpy()
    esm_matrix = parse_matrix(sampled["embedded_sequence"])
    avg_matrix = parse_matrix(sampled["average_pooling_embedding"])

    esm_scores = tsne_numpy(esm_matrix, random_state=random_state)
    avg_scores = tsne_numpy(avg_matrix, random_state=random_state)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5), sharex=False, sharey=False)
    scatter_embedding_2d(axes[0], esm_scores, labels, "ESM")
    scatter_embedding_2d(axes[1], avg_scores, labels, "Pooling")

    add_label_legend(axes[1])
    fig.tight_layout()
    fig.savefig(outdir / "03_embedding_tsne.png", dpi=400, transparent=True, bbox_inches="tight")
    plt.close(fig)


def plot_embedding_projection_panel(
    df: pd.DataFrame,
    outdir: Path,
    per_class: int,
    random_state: int,
) -> None:
    labels, esm_matrix, avg_matrix = get_embedding_matrices(df)
    esm_pca, esm_explained = pca_numpy(esm_matrix)
    avg_pca, avg_explained = pca_numpy(avg_matrix)

    sampled = stratified_sample(df, per_class=per_class, random_state=random_state)
    sampled_labels, sampled_esm, sampled_avg = get_embedding_matrices(sampled)
    esm_tsne = tsne_numpy(sampled_esm, random_state=random_state)
    avg_tsne = tsne_numpy(sampled_avg, random_state=random_state)

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.7), sharex=False, sharey=False)
    scatter_pca(axes[0, 0], esm_pca, esm_explained, labels, "ESM PCA")
    scatter_pca(axes[0, 1], avg_pca, avg_explained, labels, "Pooling PCA")
    scatter_embedding_2d(axes[1, 0], esm_tsne, sampled_labels, "ESM t-SNE")
    scatter_embedding_2d(axes[1, 1], avg_tsne, sampled_labels, "Pooling t-SNE")
    add_label_legend(axes[0, 1])

    fig.tight_layout()
    fig.savefig(outdir / "02_embedding_projection_panel.png", dpi=400, transparent=True, bbox_inches="tight")
    plt.close(fig)


def plot_esm_vector_heatmap(df: pd.DataFrame, outdir: Path, n_rows: int = 20) -> None:
    vectors = parse_matrix(df["embedded_sequence"].head(n_rows))
    labels = df["protein_id"].astype(str).head(n_rows).tolist()
    vmax = float(np.nanpercentile(np.abs(vectors), 99))
    vmax = vmax if vmax > 0 else 1.0

    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    im = ax.imshow(
        vectors,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_xlabel("Hidden width")
    ax.set_ylabel("Protein")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xticks(np.linspace(0, vectors.shape[1] - 1, 6, dtype=int))
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", length=0)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(outdir / "03_esm_vector_heatmap.png", dpi=400, transparent=True, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, sep="\t")
    required = {"protein_id", "is_DBP", "sequence", "embedded_sequence", "average_pooling_embedding"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Missing required columns: {}".format(", ".join(sorted(missing))))

    plot_encoding_heatmaps(df, outdir)
    plot_embedding_projection_panel(df, outdir, args.tsne_per_class, args.tsne_random_state)
    plot_esm_vector_heatmap(df, outdir)

    print("Saved: {}".format(outdir / "01_encoding_heatmaps.png"))
    print("Saved: {}".format(outdir / "02_embedding_projection_panel.png"))
    print("Saved: {}".format(outdir / "03_esm_vector_heatmap.png"))


if __name__ == "__main__":
    main()
