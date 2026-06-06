#!/usr/bin/env python3
"""Clean and profile the human protein table."""

from __future__ import annotations

import argparse
import re
import shutil
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np
import pandas as pd


STANDARD_AA = list("ACDEFGHIKLMNPQRSTVWY")
ANNOTATION_COLUMNS = [
    "go_ids",
    "go_molecular_function",
    "go_biological_process",
    "go_cellular_component",
    "pfam_ids",
    "interpro_ids",
    "keywords",
    "function_cc",
]
ENTRY_FREQUENCY_COLUMNS = [
    "go_ids",
    "go_molecular_function",
    "go_biological_process",
    "go_cellular_component",
]
KEYWORD_COLUMN = "keywords"
EXCLUDED_KEYWORDS = {
    "3d-structure",
    "alternative splicing",
    "direct protein sequencing",
    "disease variant",
    "polymorphism",
    "proteomics identification",
    "reference proteome",
}

HISTOGRAM_YELLOW = "#FFFEDF"
KDE_BLUE = "#91BFFA"
PROFILE_CMAP = plt.get_cmap("YlGnBu")
LENGTH_UPPER_QUANTILE = 0.99
SHORT_ENTRY_LABELS = {
    "GO:0005634": "5634",
    "GO:0005829": "5829",
    "GO:0005886": "5886",
    "GO:0005737": "5737",
    "GO:0005654": "5654",
    "GO:0016020": "6020",
    "GO:0008270": "8270",
    "GO:0070062": "0062",
    "zinc ion binding": "Zn binding",
    "identical protein binding": "Prot bind",
    "ATP binding": "ATP",
    "RNA binding": "RNA",
    "DNA-binding transcription factor activity, RNA polymerase II-specific": "DB-TF PolII",
    "metal ion binding": "Metal",
    "RNA polymerase II cis-regulatory region sequence-specific DNA binding": "PolII DNA",
    "DNA binding": "DNA",
    "sequence-specific double-stranded DNA binding": "Seq dsDNA",
    "DNA-binding transcription factor activity": "DB-TF",
    "regulation of transcription by RNA polymerase II": "Reg PolII",
    "positive regulation of transcription by RNA polymerase II": "Pos PolII",
    "signal transduction": "Signaling",
    "negative regulation of transcription by RNA polymerase II": "Neg PolII",
    "positive regulation of DNA-templated transcription": "Pos DNA",
    "cell differentiation": "Differ.",
    "apoptotic process": "Apoptosis",
    "regulation of DNA-templated transcription": "Reg DNA",
    "negative regulation of DNA-templated transcription": "Neg DNA",
    "positive regulation of cell population proliferation": "Pos prolif",
    "negative regulation of apoptotic process": "Neg apop.",
    "G protein-coupled receptor signaling pathway": "GPCR sig",
    "intracellular signal transduction": "Intra sig",
    "nucleus": "Nucleus",
    "cytosol": "Cytosol",
    "plasma membrane": "Plasma",
    "cytoplasm": "Cytoplasm",
    "nucleoplasm": "Nucleoplasm",
    "membrane": "Membrane",
    "extracellular exosome": "Exosome",
    "extracellular region": "Extracell",
    "endoplasmic reticulum membrane": "ER membrane",
    "extracellular space": "Extracell. space",
    "endoplasmic reticulum": "ER",
    "Golgi apparatus": "Golgi",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean the raw human protein table and generate dataset profile plots."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="Data/human_protein_table.tsv",
        help="Path to the raw human protein TSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default="Results/01_Dataset_profile",
        help="Directory for profile tables and PNG figures.",
    )
    parser.add_argument(
        "--washed-dir",
        default="washed_data",
        help="Directory for the cleaned TSV file.",
    )
    return parser.parse_args()


def read_table(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input data file does not exist: {input_path}")

    data = pd.read_csv(input_path, sep="\t", dtype=str, keep_default_na=False)
    required = {"protein_id", "sequence", "sequence_length"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    return data


def clean_data(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    cleaned = data.copy()
    cleaned["sequence"] = cleaned["sequence"].fillna("").str.strip().str.upper()
    cleaned["computed_sequence_length"] = cleaned["sequence"].str.len()

    empty_mask = cleaned["sequence"].eq("") | cleaned["computed_sequence_length"].eq(0)
    without_empty = cleaned.loc[~empty_mask].copy()

    duplicate_mask = without_empty.duplicated(subset=["sequence"], keep="first")
    washed = without_empty.loc[~duplicate_mask].copy().reset_index(drop=True)

    counts = {
        "raw_entries": int(len(data)),
        "empty_entries": int(empty_mask.sum()),
        "duplicate_entries": int(duplicate_mask.sum()),
        "valid_entries": int(len(washed)),
    }
    return washed, counts


def save_washed_data(washed: pd.DataFrame, input_path: Path, washed_dir: Path) -> Path:
    washed_dir.mkdir(parents=True, exist_ok=True)
    output_path = washed_dir / f"{input_path.stem}_washed.tsv"
    washed.to_csv(output_path, sep="\t", index=False)
    return output_path


def kde_pdf(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        return np.zeros_like(grid)

    std = np.std(values, ddof=1)
    iqr = np.subtract(*np.percentile(values, [75, 25]))
    sigma = min(std, iqr / 1.349) if iqr > 0 else std
    bandwidth = 0.9 * sigma * (n ** (-1 / 5)) if sigma > 0 else 1.0
    bandwidth = max(float(bandwidth), 1e-6)

    scaled = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * scaled**2).sum(axis=1)
    return density / (n * bandwidth * np.sqrt(2 * np.pi))


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    fig.savefig(output_path, dpi=400, transparent=True, bbox_inches="tight")
    plt.close(fig)


def prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)


def gradient_colors(n_colors: int) -> list[str]:
    if n_colors <= 0:
        return []
    if n_colors == 1:
        return [to_hex(PROFILE_CMAP(0.18))]
    positions = np.linspace(0.18, 0.92, n_colors)
    return [to_hex(PROFILE_CMAP(position)) for position in positions]


def ellipsize_middle(text: str, max_chars: int = 46) -> str:
    text = str(text)
    if len(text) <= max_chars:
        return text
    keep = max_chars - 3
    left = int(np.ceil(keep * 0.58))
    right = keep - left
    return f"{text[:left].rstrip()}...{text[-right:].lstrip()}"


def ellipsize_labels(labels: pd.Series, max_chars: int = 46) -> list[str]:
    return [ellipsize_middle(str(label), max_chars=max_chars) for label in labels]


def short_entry_labels(labels: pd.Series, max_chars: int = 30) -> list[str]:
    return [
        ellipsize_middle(SHORT_ENTRY_LABELS.get(str(label), str(label)), max_chars=max_chars)
        for label in labels
    ]


def strip_go_reference(entry: str) -> str:
    return re.sub(r"\s*\[GO:\d+\]\s*$", "", entry, flags=re.IGNORECASE).strip()


def get_plotted_lengths(washed: pd.DataFrame) -> tuple[np.ndarray, float]:
    lengths = pd.to_numeric(washed["computed_sequence_length"], errors="coerce").to_numpy()
    lengths = lengths[np.isfinite(lengths)]
    positive_lengths = lengths[lengths > 0]
    upper_limit = np.quantile(positive_lengths, LENGTH_UPPER_QUANTILE)
    plotted_lengths = positive_lengths[positive_lengths <= upper_limit]
    return plotted_lengths, upper_limit


def draw_length_distribution(ax: plt.Axes, washed: pd.DataFrame) -> None:
    plotted_lengths, upper_limit = get_plotted_lengths(washed)
    ax.hist(
        plotted_lengths,
        bins=80,
        density=True,
        color=HISTOGRAM_YELLOW,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.95,
    )
    grid = np.linspace(plotted_lengths.min(), upper_limit, 600)
    ax.plot(grid, kde_pdf(plotted_lengths, grid), color=KDE_BLUE, linewidth=1.8)
    ax.set_xlim(plotted_lengths.min(), upper_limit)
    ax.set_xlabel("Sequence length (aa)")
    ax.set_ylabel("Density")
    style_axes(ax)


def plot_length_distribution(washed: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    draw_length_distribution(ax, washed)
    save_figure(fig, output_dir / "01_sequence_length_distribution.png")


def plot_amino_acid_counts(washed: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    residues = Counter("".join(washed["sequence"].astype(str)))
    special = sorted(residue for residue in residues if residue not in STANDARD_AA)
    order = STANDARD_AA + special
    labels = [aa if aa in STANDARD_AA else f"{aa}*" for aa in order]
    counts = [residues.get(aa, 0) for aa in order]

    aa_table = pd.DataFrame(
        {
            "amino_acid": order,
            "x_label": labels,
            "count": counts,
            "is_special": [aa not in STANDARD_AA for aa in order],
        }
    )
    aa_table = aa_table.sort_values("count", ascending=False).reset_index(drop=True)
    aa_table.to_csv(output_dir / "03_amino_acid_counts.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    draw_amino_acid_counts(ax, aa_table)
    save_figure(fig, output_dir / "02_amino_acid_counts.png")
    return aa_table


def draw_amino_acid_counts(ax: plt.Axes, aa_table: pd.DataFrame) -> None:
    ax.bar(
        aa_table["x_label"],
        aa_table["count"],
        color=gradient_colors(len(aa_table)),
        edgecolor="#4A4A4A",
        linewidth=0.25,
    )
    ax.set_xlabel("Amino acid")
    ax.set_ylabel("Count")
    annotate_special_amino_acids(ax, aa_table)
    style_axes(ax)


def annotate_special_amino_acids(ax: plt.Axes, aa_table: pd.DataFrame) -> None:
    total_count = aa_table["count"].sum()
    special_rows = aa_table[aa_table["is_special"]]
    if special_rows.empty or total_count <= 0:
        return
    y_max = aa_table["count"].max()
    for _, row in special_rows.iterrows():
        x_index = aa_table.index[aa_table["x_label"] == row["x_label"]][0]
        ratio = row["count"] / total_count * 100
        ax.text(
            x_index,
            y_max * 0.09,
            "0.003‰",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def get_annotation_coverage(washed: pd.DataFrame) -> pd.DataFrame:
    columns = [col for col in ANNOTATION_COLUMNS if col in washed.columns]
    coverage = []
    for col in columns:
        non_empty = washed[col].fillna("").astype(str).str.strip().ne("").sum()
        coverage.append(
            {
                "annotation": col,
                "non_empty_count": int(non_empty),
                "coverage": float(non_empty / len(washed)),
            }
        )

    coverage_table = pd.DataFrame(coverage)
    coverage_table = coverage_table.sort_values("coverage", ascending=False).reset_index(
        drop=True
    )
    return coverage_table


def draw_annotation_coverage(ax: plt.Axes, coverage_table: pd.DataFrame) -> None:
    x_positions = np.arange(len(coverage_table))
    colors = gradient_colors(len(coverage_table))
    ax.bar(x_positions, coverage_table["coverage"], color=colors)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        coverage_table["annotation"],
        rotation=35,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_xlabel("Annotation field")
    ax.set_ylabel("Coverage")
    ax.set_ylim(0, 1.05)
    style_axes(ax)


def plot_annotation_coverage(washed: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    coverage_table = get_annotation_coverage(washed)
    coverage_table.to_csv(output_dir / "05_annotation_coverage.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    draw_annotation_coverage(ax, coverage_table)
    save_figure(fig, output_dir / "04_annotation_coverage.png")
    return coverage_table


def split_annotation_entries(series: pd.Series, column: str) -> list[str]:
    values = series.fillna("").astype(str)
    entries: list[str] = []
    for value in values:
        if not value.strip():
            continue
        raw_entries = value.split(";")
        for entry in raw_entries:
            clean_entry = " ".join(entry.strip().split())
            if not clean_entry:
                continue
            if column != "go_ids":
                clean_entry = strip_go_reference(clean_entry)
            if clean_entry:
                entries.append(clean_entry)
    return entries


def annotation_entry_frequency_tables(
    washed: pd.DataFrame, output_dir: Path, top_n: int = 25
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    combined_tables: list[pd.DataFrame] = []
    for col in [col for col in ENTRY_FREQUENCY_COLUMNS if col in washed.columns]:
        counts = Counter(split_annotation_entries(washed[col], col))
        table = pd.DataFrame(counts.most_common(top_n), columns=["entry", "count"])
        table = table.sort_values("count", ascending=False).reset_index(drop=True)
        tables[col] = table
        compact = table.copy()
        compact.insert(0, "rank", compact.index + 1)
        compact.insert(0, "annotation", col)
        combined_tables.append(compact)
    if combined_tables:
        pd.concat(combined_tables, ignore_index=True).to_csv(
            output_dir / "07_annotation_entry_frequencies_top.csv", index=False
        )
    return tables


def keyword_frequency_table(
    washed: pd.DataFrame, output_dir: Path, top_n: int = 20
) -> pd.DataFrame:
    if KEYWORD_COLUMN not in washed.columns:
        return pd.DataFrame(columns=["keyword", "count"])

    keywords = [
        keyword
        for keyword in split_annotation_entries(washed[KEYWORD_COLUMN], KEYWORD_COLUMN)
        if keyword.casefold() not in EXCLUDED_KEYWORDS
    ]
    counts = Counter(keywords)
    table = pd.DataFrame(counts.most_common(top_n), columns=["keyword", "count"])
    table = table.sort_values("count", ascending=False).reset_index(drop=True)
    table.to_csv(output_dir / "09_keyword_frequencies_top20.csv", index=False)
    return table


def plot_annotation_entry_frequencies(
    tables: dict[str, pd.DataFrame], output_dir: Path, top_n: int = 8
) -> None:
    for column, table in tables.items():
        top = table.head(top_n).sort_values("count", ascending=False)
        fig_height = max(4.2, 0.42 * len(top) + 1.2)
        fig, ax = plt.subplots(figsize=(8.2, fig_height))
        ax.barh(
            short_entry_labels(top["entry"], max_chars=34),
            top["count"],
            color=gradient_colors(len(top)),
        )
        ax.invert_yaxis()
        ax.set_xlabel("Count")
        ax.set_ylabel(column)
        ax.tick_params(axis="y", labelsize=8)
        style_axes(ax)
        fig.tight_layout()
        figure_numbers = {
            "go_molecular_function": "10",
            "go_cellular_component": "11",
            "go_biological_process": "12",
            "go_ids": "13",
        }
        prefix = figure_numbers.get(column, "10")
        save_figure(fig, output_dir / f"{prefix}_annotation_frequency_{column}.png")


def draw_annotation_entry_frequency(
    ax: plt.Axes, table: pd.DataFrame, y_label: str, top_n: int = 8
) -> None:
    top = table.head(top_n).sort_values("count", ascending=False)
    ax.barh(
        short_entry_labels(top["entry"], max_chars=9),
        top["count"],
        color=gradient_colors(len(top)),
    )
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_ylabel(y_label)
    ax.yaxis.set_label_coords(-0.22, 0.5)
    ax.tick_params(axis="y", labelsize=7, pad=1)
    style_axes(ax)


def plot_keyword_frequencies(keyword_table: pd.DataFrame, output_dir: Path) -> None:
    if keyword_table.empty:
        return

    top = keyword_table.sort_values("count", ascending=False)
    fig_height = max(5.0, 0.36 * len(top) + 1.2)
    fig, ax = plt.subplots(figsize=(7.2, fig_height))
    draw_keyword_frequencies(ax, top, max_chars=52)
    save_figure(fig, output_dir / "08_keyword_frequencies_top20.png")


def draw_keyword_frequencies(
    ax: plt.Axes, keyword_table: pd.DataFrame, max_chars: int = 52
) -> None:
    top = keyword_table.sort_values("count", ascending=False)
    ax.barh(
        ellipsize_labels(top["keyword"], max_chars=max_chars),
        top["count"],
        color=gradient_colors(len(top)),
    )
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_ylabel("Keyword")
    ax.tick_params(axis="y", labelsize=8)
    style_axes(ax)


def plot_dataset_profile_panel(
    washed: pd.DataFrame,
    coverage_table: pd.DataFrame,
    keyword_table: pd.DataFrame,
    output_dir: Path,
) -> None:
    if keyword_table.empty:
        return

    fig = plt.figure(figsize=(12.8, 8.4))
    ax_length = fig.add_axes([0.08, 0.61, 0.39, 0.30])
    ax_coverage = fig.add_axes([0.08, 0.17, 0.39, 0.30])
    ax_keywords = fig.add_axes([0.63, 0.07, 0.34, 0.84])

    draw_length_distribution(ax_length, washed)
    draw_annotation_coverage(ax_coverage, coverage_table)
    draw_keyword_frequencies(ax_keywords, keyword_table, max_chars=44)

    save_figure(fig, output_dir / "14_dataset_profile_panel.png")


def plot_frequency_panel(
    aa_table: pd.DataFrame, entry_tables: dict[str, pd.DataFrame], output_dir: Path
) -> None:
    fig = plt.figure(figsize=(10.8, 11.0))
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.45, 1.0, 1.0],
        hspace=0.58,
        wspace=0.50,
    )
    ax_aa = fig.add_subplot(grid[0, :])
    draw_amino_acid_counts(ax_aa, aa_table)

    layout = [
        ("go_molecular_function", "GO MF", grid[1, 0]),
        ("go_cellular_component", "GO CC", grid[1, 1]),
        ("go_biological_process", "GO BP", grid[2, 0]),
        ("go_ids", "GO ID", grid[2, 1]),
    ]
    for column, y_label, cell in layout:
        ax = fig.add_subplot(cell)
        if column in entry_tables:
            draw_annotation_entry_frequency(ax, entry_tables[column], y_label=y_label)
        else:
            ax.axis("off")

    save_figure(fig, output_dir / "15_profile_frequency_panel.png")


def save_summary_tables(washed: pd.DataFrame, counts: dict[str, int], output_dir: Path) -> None:
    pd.DataFrame(counts.items(), columns=["metric", "value"]).to_csv(
        output_dir / "00_cleaning_summary.csv", index=False
    )

    lengths = pd.to_numeric(washed["computed_sequence_length"], errors="coerce")
    summary = pd.DataFrame(
        {
            "metric": [
                "protein_count",
                "min_length",
                "q1_length",
                "median_length",
                "mean_length",
                "q3_length",
                "max_length",
                f"length_p{int(LENGTH_UPPER_QUANTILE * 100)}",
            ],
            "value": [
                len(washed),
                lengths.min(),
                lengths.quantile(0.25),
                lengths.median(),
                lengths.mean(),
                lengths.quantile(0.75),
                lengths.max(),
                lengths.quantile(LENGTH_UPPER_QUANTILE),
            ],
        }
    )
    summary.to_csv(output_dir / "06_dataset_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    washed_dir = Path(args.washed_dir)
    prepare_output_dir(output_dir)

    raw_data = read_table(input_path)
    washed, counts = clean_data(raw_data)
    washed_path = save_washed_data(washed, input_path, washed_dir)

    save_summary_tables(washed, counts, output_dir)
    plot_length_distribution(washed, output_dir)
    aa_table = plot_amino_acid_counts(washed, output_dir)
    coverage_table = plot_annotation_coverage(washed, output_dir)
    entry_tables = annotation_entry_frequency_tables(washed, output_dir)
    plot_annotation_entry_frequencies(entry_tables, output_dir)
    keyword_table = keyword_frequency_table(washed, output_dir)
    plot_keyword_frequencies(keyword_table, output_dir)
    plot_dataset_profile_panel(washed, coverage_table, keyword_table, output_dir)
    plot_frequency_panel(aa_table, entry_tables, output_dir)

    print(f"Input data: {input_path}")
    print(f"Washed data: {washed_path}")
    print(f"Results directory: {output_dir}")
    print(f"Duplicate entries: {counts['duplicate_entries']}")
    print(f"Empty entries: {counts['empty_entries']}")
    print(f"Valid entries: {counts['valid_entries']}")


if __name__ == "__main__":
    main()
