#!/usr/bin/env python3
"""Analyze false positives and false negatives for the ESM-2 + SVM classifier."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import ssl
import urllib.request
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 9
TOP_N = 15


def mako_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "mako_like",
        ["#0B0405", "#241124", "#3B255A", "#3E4C8A", "#2F7797", "#2FA3A0", "#7FD0BA", "#DEF5E5"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ESM-2 + SVM misclassification annotation analysis.")
    parser.add_argument(
        "--embedded",
        default="washed_data/human_protein_table_embedded.tsv",
        help="TSV containing protein_id, is_DBP, sequence, and embedded_sequence.",
    )
    parser.add_argument(
        "--washed",
        default="washed_data/human_protein_table_washed.tsv",
        help="Washed annotation TSV containing GO and domain columns.",
    )
    parser.add_argument(
        "--outdir",
        default="Results/05_Misclassification",
        help="Output directory.",
    )
    return parser.parse_args()


def prepare_output_dir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for child in outdir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def load_vector(value: object) -> np.ndarray:
    return np.asarray(json.loads(str(value)), dtype=float)


def vector_matrix(series: pd.Series) -> np.ndarray:
    return np.vstack([load_vector(value) for value in series])


def make_balanced_binary_dataset(df: pd.DataFrame) -> pd.DataFrame:
    binary = df[df["is_DBP"].isin([0, 1])].copy()
    positives = binary[binary["is_DBP"] == 1]
    negatives = binary[binary["is_DBP"] == 0]
    sampled_negatives = negatives.sample(n=len(positives), random_state=RANDOM_STATE, replace=False)
    balanced = pd.concat([positives, sampled_negatives], axis=0)
    return balanced.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)


def make_svm(random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def clean_go_function(entry: str) -> str:
    entry = re.sub(r"\s*\[GO:\d+\]\s*$", "", entry.strip())
    return " ".join(entry.split())


def split_semicolon_entries(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [" ".join(item.strip().split()) for item in str(value).split(";") if item.strip()]


def molecular_function_counter(rows: pd.DataFrame) -> Counter:
    counter: Counter = Counter()
    for value in rows["go_molecular_function"].fillna(""):
        for entry in split_semicolon_entries(value):
            cleaned = clean_go_function(entry)
            if cleaned:
                counter[cleaned] += 1
    return counter


def domain_counter(rows: pd.DataFrame) -> Counter:
    counter: Counter = Counter()
    for field in ["pfam_ids", "interpro_ids"]:
        for value in rows[field].fillna(""):
            for entry in split_semicolon_entries(value):
                if entry:
                    counter[entry] += 1
    return counter


def fetch_domain_name(accession: str) -> str:
    database = "pfam" if accession.startswith("PF") else "interpro"
    url = f"https://www.ebi.ac.uk/interpro/api/entry/{database}/{accession}/"
    contexts = [None, ssl._create_unverified_context()]
    for context in contexts:
        try:
            kwargs = {"timeout": 20}
            if context is not None:
                kwargs["context"] = context
            with urllib.request.urlopen(url, **kwargs) as response:
                payload = json.load(response)
            metadata = payload.get("metadata", {})
            name = metadata.get("name", "")
            if isinstance(name, dict):
                return name.get("name") or name.get("short") or accession
            if isinstance(name, str) and name:
                return name
            hierarchy = metadata.get("hierarchy") or {}
            return hierarchy.get("name") or accession
        except Exception:
            continue
    return accession


def build_domain_name_mapping(accessions: list[str], outdir: Path) -> dict[str, str]:
    accessions = sorted(set(accessions))
    mapping_path = outdir / "domain_name_mapping.csv"
    if mapping_path.exists():
        mapping = pd.read_csv(mapping_path)
        cached = dict(zip(mapping["accession"], mapping["domain_name"]))
    else:
        cached = {}

    rows = []
    result = {}
    for accession in accessions:
        name = cached.get(accession) or fetch_domain_name(accession)
        result[accession] = name
        rows.append({"accession": accession, "domain_name": name})
        print(f"{accession}: {name}")
    pd.DataFrame(rows).to_csv(mapping_path, index=False)
    return result


def named_domain_counter(id_counts: Counter, domain_names: dict[str, str], top_n: int = TOP_N) -> Counter:
    named_counts: Counter = Counter()
    for accession, count in id_counts.most_common(top_n):
        named_counts[domain_names.get(accession, accession)] += count
    return named_counts


def counter_to_frame(counter: Counter, top_n: int = TOP_N) -> pd.DataFrame:
    return pd.DataFrame(counter.most_common(top_n), columns=["term", "count"])


def shorten_label(label: str, max_len: int = 34) -> str:
    if len(label) <= max_len:
        return label
    keep = max_len - 3
    left = keep // 2
    right = keep - left
    return f"{label[:left]}...{label[-right:]}"


def plot_barh(ax: plt.Axes, data: pd.DataFrame, title: str) -> None:
    if data.empty:
        ax.text(0.5, 0.5, "No annotation", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    plot_data = data.sort_values("count", ascending=True)
    colors = mako_cmap()(np.linspace(0.25, 0.9, len(plot_data)))
    ax.barh(np.arange(len(plot_data)), plot_data["count"], color=colors, edgecolor="#333333", linewidth=0.5)
    ax.set_yticks(np.arange(len(plot_data)))
    ax.set_yticklabels([shorten_label(term) for term in plot_data["term"]], fontsize=8)
    ax.set_xlabel("Count")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_frequency_outputs(misclassified: pd.DataFrame, outdir: Path) -> None:
    false_positive = misclassified[misclassified["error_type"] == "false_positive"]
    false_negative = misclassified[misclassified["error_type"] == "false_negative"]
    fp_domain_ids = domain_counter(false_positive)
    fn_domain_ids = domain_counter(false_negative)
    top_domain_ids = [term for term, _ in fp_domain_ids.most_common(TOP_N)]
    top_domain_ids += [term for term, _ in fn_domain_ids.most_common(TOP_N)]
    domain_names = build_domain_name_mapping(top_domain_ids, outdir)

    tables = {
        "false_positive_molecular_function": counter_to_frame(molecular_function_counter(false_positive)),
        "false_positive_domain": counter_to_frame(named_domain_counter(fp_domain_ids, domain_names)),
        "false_negative_molecular_function": counter_to_frame(molecular_function_counter(false_negative)),
        "false_negative_domain": counter_to_frame(named_domain_counter(fn_domain_ids, domain_names)),
    }
    for name, table in tables.items():
        table.to_csv(outdir / f"{name}_top{TOP_N}.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2))
    plot_barh(axes[0, 0], tables["false_positive_molecular_function"], "False positive: molecular function")
    plot_barh(axes[0, 1], tables["false_positive_domain"], "False positive: Pfam/InterPro domain")
    plot_barh(axes[1, 0], tables["false_negative_molecular_function"], "False negative: molecular function")
    plot_barh(axes[1, 1], tables["false_negative_domain"], "False negative: Pfam/InterPro domain")
    fig.tight_layout()
    fig.savefig(outdir / "01_svm_esm2_misclassification_annotation_frequency.png", dpi=400, transparent=True, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    prepare_output_dir(outdir)

    embedded = pd.read_csv(args.embedded, sep="\t")
    annotations = pd.read_csv(
        args.washed,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "protein_id",
            "protein_name",
            "sequence_length",
            "go_molecular_function",
            "pfam_ids",
            "interpro_ids",
            "keywords",
        ],
    )

    balanced = make_balanced_binary_dataset(embedded)
    train_valid, test = train_test_split(
        balanced,
        test_size=TEST_SIZE,
        stratify=balanced["is_DBP"],
        random_state=RANDOM_STATE,
    )
    x_train_valid = vector_matrix(train_valid["embedded_sequence"])
    y_train_valid = train_valid["is_DBP"].astype(int).to_numpy()
    x_test = vector_matrix(test["embedded_sequence"])
    y_test = test["is_DBP"].astype(int).to_numpy()

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    score_matrix = []
    pred_matrix = []
    for fold_idx, (train_idx, _) in enumerate(cv.split(x_train_valid, y_train_valid), start=1):
        model = make_svm(RANDOM_STATE + fold_idx)
        model.fit(x_train_valid[train_idx], y_train_valid[train_idx])
        scores = model.decision_function(x_test)
        preds = (scores >= 0).astype(int)
        score_matrix.append(scores)
        pred_matrix.append(preds)
        print(f"ESM-2 + SVM fold {fold_idx} finished.")

    mean_score = np.mean(np.vstack(score_matrix), axis=0)
    vote_pred = (np.mean(np.vstack(pred_matrix), axis=0) >= 0.5).astype(int)
    prediction = test[["protein_id", "is_DBP", "sequence"]].copy()
    prediction["svm_esm2_mean_score"] = mean_score
    prediction["predicted_label"] = vote_pred
    prediction["error_type"] = np.where(
        (prediction["is_DBP"] == 0) & (prediction["predicted_label"] == 1),
        "false_positive",
        np.where(
            (prediction["is_DBP"] == 1) & (prediction["predicted_label"] == 0),
            "false_negative",
            "correct",
        ),
    )

    merged = prediction.merge(annotations, on="protein_id", how="left")
    misclassified = merged[merged["error_type"].isin(["false_positive", "false_negative"])].copy()
    merged.to_csv(outdir / "00_svm_esm2_test_predictions.csv", index=False)
    misclassified.to_csv(outdir / "00_svm_esm2_misclassified_entries.csv", index=False)

    summary = (
        prediction["error_type"]
        .value_counts()
        .rename_axis("error_type")
        .reset_index(name="count")
    )
    summary.to_csv(outdir / "00_svm_esm2_misclassification_summary.csv", index=False)
    save_frequency_outputs(misclassified, outdir)

    print(summary.to_string(index=False))
    print(f"Saved: {outdir}")


if __name__ == "__main__":
    main()
