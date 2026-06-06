#!/usr/bin/env python3
"""Assign three-class DNA-binding protein labels with a rule-based logic tree."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABEL_MAP = {
    1: "high-confidence DBP",
    0: "high-confidence non-DBP",
    -1: "ambiguous or uncertain",
}

NUCLEAR_LOCATION_TERMS = [
    "nucleus",
    "nuclear",
    "nucleoplasm",
    "nucleolus",
    "chromatin",
    "chromosome",
]
STRONG_DBP_GO_IDS = {
    "GO:0003677": "DNA binding",
    "GO:0003690": "double-stranded DNA binding",
    "GO:0003697": "single-stranded DNA binding",
    "GO:0003700": "DNA-binding transcription factor activity",
    "GO:0000976": "transcription cis-regulatory region binding",
    "GO:0000977": "RNA polymerase II regulatory region sequence-specific DNA binding",
    "GO:0000978": "RNA polymerase II cis-regulatory region sequence-specific DNA binding",
    "GO:0000981": "DNA-binding transcription factor activity, RNA polymerase II-specific",
    "GO:0003684": "damaged DNA binding",
    "GO:0003691": "telomeric DNA binding",
    "GO:0043565": "sequence-specific DNA binding",
}
STRONG_DBP_MF_TERMS = [
    "dna binding",
    "double stranded dna binding",
    "single stranded dna binding",
    "sequence specific dna binding",
    "sequence specific double stranded dna binding",
    "transcription cis regulatory region binding",
    "promoter specific chromatin binding",
    "damaged dna binding",
    "methylated dna binding",
    "unmethylated cpg binding",
    "telomeric dna binding",
    "four way junction dna binding",
    "dna replication origin binding",
    "dna binding transcription factor activity",
    "rna polymerase ii cis regulatory region sequence specific dna binding",
]
STRONG_DBP_DOMAIN_TERMS = [
    "homeobox",
    "homeodomain",
    "helix turn helix",
    "winged helix",
    "basic helix loop helix",
    "bhlh",
    "basic leucine zipper",
    "bzip",
    "forkhead",
    "hmg box",
    "high mobility group box",
    "myb dna binding domain",
    "ets domain",
    "p53 dna binding domain",
    "nuclear receptor dna binding domain",
    "mads box",
    "t box",
    "paired box",
    "sry related hmg box",
    "arid",
    "at hook",
    "sap domain",
    "dna binding domain",
]
ZINC_FINGER_TERMS = [
    "zinc finger",
    "c2h2",
    "c2h2 type zinc finger",
]
ZINC_FINGER_CONTEXT_TERMS = [
    "dna",
    "transcription factor",
    "sequence specific",
    "promoter",
    "enhancer",
    "rna polymerase ii",
]
AMBIGUOUS_TERMS = [
    "zinc finger",
    "c2h2",
    "regulation of transcription",
    "positive regulation of transcription",
    "negative regulation of transcription",
    "regulation of gene expression",
    "transcription coregulator",
    "transcription coactivator",
    "transcription corepressor",
    "transcription regulator",
    "chromatin organization",
    "chromatin remodeling",
    "chromatin",
    "chromosome",
    "nucleosome",
    "histone",
    "polycomb",
    "swi snf",
    "nucleus",
    "nuclear",
    "nucleoplasm",
    "nucleolus",
    "rna binding",
    "mrna binding",
    "rrna binding",
    "trna binding",
    "nucleic acid binding",
    "polynucleotide binding",
]

PIE_COLORS = {
    1: "#91BFFA",
    0: "#FFFEDF",
    -1: "#D9D9D9",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create 1/0/-1 DBP labels.")
    parser.add_argument(
        "input",
        nargs="?",
        default="washed_data/human_protein_table_washed.tsv",
        help="Washed protein annotation TSV.",
    )
    parser.add_argument(
        "--output",
        default="washed_data/human_protein_table_labeled.tsv",
        help="Output TSV with protein_id, is_DBP, sequence.",
    )
    parser.add_argument(
        "--results-dir",
        default="Results/02_Label_construction",
        help="Directory for minimal label summaries and figures.",
    )
    return parser.parse_args()


def prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def norm_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).lower().replace("_", " ").replace("-", " ")


def split_entries(value: object) -> list[str]:
    return [" ".join(entry.strip().split()) for entry in str(value).split(";") if entry.strip()]


def contains_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def has_any_go_id(go_ids: object, dictionary: dict[str, str]) -> list[str]:
    present = set(split_entries(go_ids))
    return sorted(present.intersection(dictionary))


def row_text(row: pd.Series, fields: list[str]) -> str:
    return " ".join(norm_text(row.get(field, "")) for field in fields)


def has_cellular_localization(row: pd.Series) -> bool:
    go_cc = norm_text(row.get("go_cellular_component", ""))
    return bool(go_cc.strip())


def cellular_localization_state(row: pd.Series) -> tuple[str, str]:
    go_cc = norm_text(row.get("go_cellular_component", ""))
    if not go_cc.strip():
        return "unknown", "unknown go_cellular_component"
    if contains_any(go_cc, NUCLEAR_LOCATION_TERMS):
        return "nuclear", "nuclear signal in go_cellular_component"
    return "non_nuclear", "go_cellular_component present but no nuclear signal"


def go_evidence(row: pd.Series) -> tuple[bool, str]:
    matched_go = has_any_go_id(row.get("go_ids", ""), STRONG_DBP_GO_IDS)
    if matched_go:
        return True, "strong DBP GO ID: " + ",".join(matched_go)

    go_mf = norm_text(row.get("go_molecular_function", ""))
    if contains_any(go_mf, STRONG_DBP_MF_TERMS):
        return True, "strong GO molecular-function DBP term"
    return False, ""


def domain_evidence(row: pd.Series) -> tuple[bool, str]:
    domain_text = row_text(
        row,
        ["protein_name", "pfam_ids", "interpro_ids", "keywords", "function_cc"],
    )
    if contains_any(domain_text, STRONG_DBP_DOMAIN_TERMS):
        return True, "known DNA-binding domain term"

    has_zinc_finger = contains_any(domain_text, ZINC_FINGER_TERMS)
    if has_zinc_finger:
        context_text = row_text(
            row,
            [
                "protein_name",
                "go_molecular_function",
                "go_biological_process",
                "keywords",
                "function_cc",
            ],
        )
        if contains_any(context_text, ZINC_FINGER_CONTEXT_TERMS):
            return True, "zinc finger with DNA/transcription context"
    return False, ""


def ambiguity_evidence(row: pd.Series) -> tuple[bool, str]:
    all_text = row_text(
        row,
        [
            "protein_name",
            "go_molecular_function",
            "go_biological_process",
            "go_cellular_component",
            "pfam_ids",
            "interpro_ids",
            "keywords",
            "function_cc",
        ],
    )
    if contains_any(all_text, AMBIGUOUS_TERMS):
        return True, "ambiguous nuclear/transcription/chromatin/RNA/nucleic-acid signal"
    return False, ""


def judge_is_dbp(row: pd.Series) -> tuple[int, str]:
    localization, localization_reason = cellular_localization_state(row)
    if localization == "non_nuclear":
        return 0, localization_reason

    is_go_dbp, go_reason = go_evidence(row)
    if is_go_dbp:
        return 1, f"{localization_reason}; {go_reason}"

    is_domain_dbp, domain_reason = domain_evidence(row)
    if is_domain_dbp:
        return 1, f"{localization_reason}; {domain_reason}"

    is_ambiguous, ambiguity_reason = ambiguity_evidence(row)
    if is_ambiguous:
        return -1, f"{localization_reason}; {ambiguity_reason}"

    return 0, f"{localization_reason}; no GO/domain DBP or ambiguity evidence"


def build_dictionary_table() -> pd.DataFrame:
    rows = []
    rows.append(["localization_gate", "go_cellular_component", "empty", "<empty>", "unknown localization; continue to GO/domain/ambiguity rules"])
    for term in NUCLEAR_LOCATION_TERMS:
        rows.append(["localization_gate", "go_cellular_component", "contains", term, "nuclear localization; continue to GO/domain/ambiguity rules"])
    rows.append(["localization_gate", "go_cellular_component", "non-empty without nuclear term", "<other localization>", "non-nuclear localization; label 0"])
    for go_id, description in STRONG_DBP_GO_IDS.items():
        rows.append(["go_positive", "go_ids", "exact", go_id, description])
    for term in STRONG_DBP_MF_TERMS:
        rows.append(["go_positive", "go_molecular_function", "contains", term, "strong DBP molecular function"])
    for term in STRONG_DBP_DOMAIN_TERMS:
        rows.append(["domain_positive", "protein_name/pfam_ids/interpro_ids/keywords/function_cc", "contains", term, "strong DNA-binding domain"])
    for term in ZINC_FINGER_TERMS:
        rows.append(["zinc_finger", "protein_name/pfam_ids/interpro_ids/keywords/function_cc", "contains", term, "requires context"])
    for term in ZINC_FINGER_CONTEXT_TERMS:
        rows.append(["zinc_finger_context", "protein_name/go_mf/go_bp/keywords/function_cc", "contains", term, "zinc finger positive context"])
    for term in AMBIGUOUS_TERMS:
        rows.append(["ambiguous", "all annotation text", "contains", term, "ambiguous if GO/domain positive evidence is absent"])
    return pd.DataFrame(rows, columns=["rule_group", "source_field", "match_mode", "term_or_id", "description"])


def save_pie_chart(label_counts: pd.Series, output_dir: Path) -> None:
    order = [1, 0, -1]
    values = [int(label_counts.get(label, 0)) for label in order]
    labels = [
        f"1: DBP\n{values[0]}",
        f"0: Non-DBP\n{values[1]}",
        f"-1: ambiguity\n{values[2]}",
    ]
    colors = [PIE_COLORS[label] for label in order]

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    wedges, texts = ax.pie(
        values,
        labels=None,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"edgecolor": "white", "linewidth": 1.0},
    )
    for wedge, label in zip(wedges, labels):
        theta = (wedge.theta1 + wedge.theta2) / 2
        radius = 0.58
        x = radius * np.cos(np.deg2rad(theta))
        y = radius * np.sin(np.deg2rad(theta))
        ax.text(x, y, label, ha="center", va="center", fontsize=9)
    ax.set_aspect("equal")
    fig.savefig(output_dir / "01_label_distribution_pie.png", dpi=400, transparent=True, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    results_dir = Path(args.results_dir)
    prepare_output_dir(results_dir)

    data = pd.read_csv(input_path, sep="\t", dtype=str, keep_default_na=False)
    required = {"protein_id", "sequence"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    judged = data.apply(judge_is_dbp, axis=1, result_type="expand")
    data["is_DBP"] = judged[0].astype(int)
    data["label_evidence"] = judged[1]

    labeled = data[["protein_id", "is_DBP", "sequence"]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(output_path, sep="\t", index=False)

    label_counts = data["is_DBP"].value_counts().sort_index()
    summary = (
        label_counts.rename_axis("is_DBP")
        .reset_index(name="count")
        .assign(description=lambda x: x["is_DBP"].map(LABEL_MAP))
    )
    summary.to_csv(results_dir / "00_label_summary.csv", index=False)
    build_dictionary_table().to_csv(results_dir / "02_label_dictionary.csv", index=False)
    save_pie_chart(label_counts, results_dir)

    for label in [1, 0, -1]:
        print(f"is_DBP={label} entries: {int(label_counts.get(label, 0))}")
    print(f"Labeled data: {output_path}")
    print(f"Dictionary table: {results_dir / '02_label_dictionary.csv'}")


if __name__ == "__main__":
    main()
