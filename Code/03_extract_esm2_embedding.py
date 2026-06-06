#!/usr/bin/env python3
"""Extract ESM-2 protein embeddings and append them to the labeled TSV.

Server usage example:

    cd /gpfs1/share/ljf_bioinfo2026

    source /gpfs1/share/26bioinfolab_grp7/dbp_prediction/config/cluster-env.sh
    export HF_HOME=/gpfs1/share/lfl2026/huggingface_cache
    export HF_HUB_CACHE=${HF_HOME}/hub

    /gpfs1/share/26bioinfolab_grp7/dbp_prediction/.venv/bin/python \
        extract_esm2_embedding.py

Environment notes from the server:

    - System `python` was Python 2.7.18 and cannot run this workflow.
    - System `python3` was Python 3.6.8 but lacked pandas/torch/transformers.
    - The usable course environment was:
        /gpfs1/share/26bioinfolab_grp7/dbp_prediction/.venv/bin/python
    - The course environment config was:
        /gpfs1/share/26bioinfolab_grp7/dbp_prediction/config/cluster-env.sh
    - The target data directory was:
        /gpfs1/share/ljf_bioinfo2026

Output:

    Given `human_protein_table_labeled.tsv`, this writes
    `human_protein_table_embedded.tsv` in the same directory, adding one column:
    `embedded_sequence`, a JSON-encoded list of floats.
"""

from __future__ import print_function

import argparse
import glob
import json
import os


MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
MAX_AA_PER_CHUNK = 1000
BATCH_SIZE = 8


def parse_args():
    parser = argparse.ArgumentParser(description="Extract ESM-2 embeddings for labeled proteins.")
    parser.add_argument(
        "--input",
        default=None,
        help="Input labeled TSV. Defaults to the first *_labeled.tsv in the current directory.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output embedded TSV. Defaults to replacing _labeled with _embedded.",
    )
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-aa-per-chunk", type=int, default=MAX_AA_PER_CHUNK)
    return parser.parse_args()


def find_input_file(input_arg):
    if input_arg:
        return input_arg
    files = sorted(glob.glob("*_labeled.tsv"))
    if not files:
        raise IOError("No *_labeled.tsv file found in current directory.")
    return files[0]


def make_output_path(input_path, output_arg):
    if output_arg:
        return output_arg
    base, _ = os.path.splitext(input_path)
    if base.endswith("_labeled"):
        base = base[: -len("_labeled")]
    return base + "_embedded.tsv"


def sequence_chunks(seq, size):
    seq = str(seq).strip().upper()
    if not seq:
        return [""]
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def embed_batch(seqs, tokenizer, model, device, torch_module):
    with torch_module.no_grad():
        encoded = tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1022,
            return_special_tokens_mask=True,
        )
        encoded = dict((k, v.to(device)) for k, v in encoded.items())

        # `special_tokens_mask` is needed for pooling but is not accepted by EsmModel.forward().
        special_tokens_mask = encoded.pop("special_tokens_mask")

        out = model(**encoded)
        hidden = out.last_hidden_state

        mask = encoded["attention_mask"].bool()
        special = special_tokens_mask.bool()
        residue_mask = mask & (~special)

        residue_mask_f = residue_mask.unsqueeze(-1).float()
        summed = (hidden * residue_mask_f).sum(dim=1)
        counts = residue_mask_f.sum(dim=1).clamp(min=1.0)

        emb = summed / counts
        return emb.cpu()


def embed_sequence(seq, tokenizer, model, device, batch_size, max_aa_per_chunk, torch_module):
    chunks = sequence_chunks(seq, max_aa_per_chunk)
    all_embs = []
    all_weights = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        emb = embed_batch(batch, tokenizer, model, device, torch_module)
        all_embs.append(emb)
        all_weights.extend([len(x) for x in batch])

    embs = torch_module.cat(all_embs, dim=0)
    weights = torch_module.FloatTensor(all_weights).unsqueeze(1)
    protein_emb = (embs * weights).sum(dim=0) / weights.sum()
    return protein_emb.tolist()


def main():
    args = parse_args()
    try:
        import pandas as pd
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing embedding dependency: {}. Run this script with the course "
            "server environment, e.g. /gpfs1/share/26bioinfolab_grp7/"
            "dbp_prediction/.venv/bin/python.".format(exc)
        )

    input_path = find_input_file(args.input)
    output_path = make_output_path(input_path, args.output)

    print("Input:  {}".format(input_path))
    print("Output: {}".format(output_path))

    df = pd.read_csv(input_path, sep="\t")
    required = set(["protein_id", "is_DBP", "sequence"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Missing columns: {}".format(missing))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    model = model.to(device)
    model.eval()

    embedded = []
    total = len(df)
    for idx, row in df.iterrows():
        emb = embed_sequence(
            row["sequence"],
            tokenizer,
            model,
            device,
            args.batch_size,
            args.max_aa_per_chunk,
            torch,
        )
        embedded.append(json.dumps(emb))

        if (idx + 1) % 100 == 0:
            print("Embedded {}/{}".format(idx + 1, total))

    df["embedded_sequence"] = embedded
    df.to_csv(output_path, sep="\t", index=False)

    print("Done.")
    print("Saved: {}".format(output_path))


if __name__ == "__main__":
    main()
