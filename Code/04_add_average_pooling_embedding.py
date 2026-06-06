#!/usr/bin/env python3
"""Add a simple average-pooling sequence embedding to an embedded TSV.

This control embedding does not use a protein language model. Each residue is
represented by a 20-dimensional one-hot vector over standard amino acids, then
the vectors are averaged across the sequence. The resulting vector is therefore
the amino-acid composition:

    average_pooling_embedding = mean(one_hot(residue_i))

The column is stored as a JSON-encoded list of 20 floats in this order:

    A C D E F G H I K L M N P Q R S T V W Y
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


STANDARD_AA = list("ACDEFGHIKLMNPQRSTVWY")
OUTPUT_COLUMN = "average_pooling_embedding"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append a simple one-hot average-pooling embedding column."
    )
    parser.add_argument(
        "--input",
        default="washed_data/human_protein_table_embedded.tsv",
        help="Input TSV containing at least protein_id, is_DBP, sequence, embedded_sequence.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output TSV. Defaults to overwriting the input file.",
    )
    return parser.parse_args()


def average_pooling_embedding(sequence: object) -> list[float]:
    seq = str(sequence).strip().upper()
    counts = {aa: 0 for aa in STANDARD_AA}
    valid_count = 0

    for residue in seq:
        if residue in counts:
            counts[residue] += 1
            valid_count += 1

    if valid_count == 0:
        return [0.0 for _ in STANDARD_AA]

    return [counts[aa] / valid_count for aa in STANDARD_AA]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path

    if not input_path.exists():
        raise FileNotFoundError(f"Input TSV does not exist: {input_path}")

    df = pd.read_csv(input_path, sep="\t")
    if "sequence" not in df.columns:
        raise ValueError("Input TSV must contain a 'sequence' column.")

    df[OUTPUT_COLUMN] = [
        json.dumps(average_pooling_embedding(seq), separators=(",", ":"))
        for seq in df["sequence"]
    ]

    df.to_csv(output_path, sep="\t", index=False)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Added column: {OUTPUT_COLUMN}")
    print(f"Embedding dimension: {len(STANDARD_AA)}")


if __name__ == "__main__":
    main()
