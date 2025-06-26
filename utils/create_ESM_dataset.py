#!/usr/bin/env python
"""
Create a parquet dataset that combines metadata with ESM-3 embeddings
for each MHC allele.

Usage:  python create_dataset.py
"""

import os
import pathlib
import re

import numpy as np
import pandas as pd
from tqdm import tqdm                          # progress bars
from typing import Dict
from processing_functions import create_progressive_k_fold_cross_validation, create_k_fold_leave_one_out_stratified_cv, normalize_netmhcpan_allele_to_pmgen

# ---------------------------------------------------------------------
# 1. CONFIGURATION – adjust if your paths change
# ---------------------------------------------------------------------
dataset_name = "PMGen_sequences" # "NetMHCpan_dataset"
mhc_class = 2
CSV_PATH   = pathlib.Path(f"../data/NetMHCpan_dataset/combined_data_{mhc_class}.csv")
NPZ_PATH   = pathlib.Path(
    f"../data/ESM/esmc_600m/{dataset_name}/mhc{mhc_class}_encodings.npz"
)
OUT_PARQUET = pathlib.Path(
    f"../data/Custom_dataset/{dataset_name}/mhc_{mhc_class}/mhc{mhc_class}_with_esm_embeddings.parquet"
)

# If you want to *save* each array as its own .npy rather than store the
# full tensor inside parquet, set this to a directory; otherwise None.
EMB_OUT_DIR = pathlib.Path(
    f"../data/Custom_dataset/{dataset_name}/mhc_{mhc_class}/mhc{mhc_class}_encodings"
)   # or `None` to embed directly

AUGMENTATION = "down_sampling"  # "GNUSS", None
K = 10  # Number of folds for cross-validation
# ---------------------------------------------------------------------


def load_mhc_embeddings(npz_path: pathlib.Path) -> Dict[str, np.ndarray]:
    """
    Returns {allele_name: 187×1152 tensor}.
    """
    emb_dict = {}
    with np.load(npz_path) as npz_file:
        for k in npz_file.files:
            emb = npz_file[k]
            # if emb.shape != (36, 1152):
            #     print(f"[WARN] Skipping {k}: shape {emb.shape} != (36,1152)")
            #     continue
            emb_dict[k] = emb
    print(f"Loaded embeddings for {len(emb_dict):,} alleles.")
    return emb_dict


def normalise_netmhcpan_allele(a: str) -> str:
    """
    Make the allele string format identical to the keys inside the NPZ.
    Tweak this if your formats differ!
    """
    # remove * and : from the allele name
    # and convert to upper case
    # e.g. "HLA-A*01:01" -> "HLA-A0101"
    # e.g. "HLA-A*01:01:01" -> "HLA-A010101"
    # remove spaces
    # e.g. "HLA-A 01:01" -> "HLA-A0101"
    # TODO fix eg. HLA-DRA/HLA-DRB1_0101 > DRB10101 eg. HLA-DRA/HLA-DRB1_0401 > DRB10401
    # Format heterodimer allele strings, e.g. "HLA-DRA/HLA-DRB1_0101" -> "DRB10101"
    a = a.strip()
    if "HLA-DRA/" in a:
        # For heterodimers, take the second chain and format as e.g. DRB10101
        _, second = a.split("/", 1)
        a = second.replace("HLA-", "")

    elif "mice-" in a:
        _, second = a.split("/", 1)
        a = second.replace("mice-", "")
    else:
        a = a.replace("/HLA", "")
    # remove "*" and spaces
    a = a.replace("*", "").replace(" ", "")

    return a


# def normalise_allele2(a: str) -> tuple[str, str]:
#     """
#     Make the allele string format identical to the keys inside the NPZ.
#     Tweak this if your formats differ!
#     """
#     # remove * and : from the allele name
#     # and convert to upper case
#     # e.g. "HLA-A*01:01" -> "HLA-A0101"
#     # e.g. "HLA-A*01:01:01" -> "HLA-A010101"
#     # remove spaces
#     # e.g. "HLA-A 01:01" -> "HLA-A0101"
#     a = a.strip()
#     if "HLA-DRA/" in a:
#         # eg. "HLA-DRA/HLA-DRB1_0401" -> a: "HLA-DRA1_0401" b: "HLA-DRB1_0401"
#         # For heterodimers, take the second chain and format as e.g. DRB10101
#         _, second = a.split("/", 1)
#         b = second.replace("HLA-", "")
#         a = b.replace("HLA-DRB1_", "HLA-DRA1_")
#         return a, b
#
#     # TODO fix later
#     # elif "mice-" in a:
#     #     _, second = a.split("/", 1)
#     #     a = second.replace("mice-", "")
#     #     return a, a
#
#     else:
#         a = a.replace("/HLA", "")
#     # remove "*" and spaces
#     a = a.replace("*", "").replace(" ", "")
#     return a, None


def normalise_allele_NetMHCPan_toPMGen(a: str) -> tuple[str, str]:
    a = a.strip()
    # remove : * _ from the allele name
    # e.g. "HLA-A*01:01" -> "HLA-A0101"

    #
    if ":" in a:
        # remove ":" from the allele name
        # e.g. "HLA-A*01:01" -> "HLA-A0101"
        a = a.replace(":", "")
    if "HLA-DRA/" in a:
        _, second = a.split("/", 1)
        b = second.replace("HLA-", "").replace("*", "").replace(" ", "")
        a1 = b.replace("HLA-DRB1_", "HLA-DRA1_")
        return a1, b
    if "_" in a:
        name, digits = a.split("_", 1)
        name = name.upper()
        if not name.startswith("HLA-"):
            name = f"HLA-{name}"
        return f"{name}{digits}", None
    if "BoLa-" in a:
        # eg. "BoLa-1*04:01" -> "BOLA-10401"
        prefix, rest = a.split("-", 1)
        prefix = prefix.replace("BoLa", "BOLA").upper()
        digits = rest.replace("*", "").replace(":", "")
        return f"{prefix}{digits}", None
    if "*" in a:
        prefix, rest = a.split("*", 1)
        prefix = prefix.upper()
        digits = rest.replace(":", "")
        return f"{prefix}{digits}", None

    return a, None



def attach_embeddings(
    df: pd.DataFrame,
    emb_dict: Dict[str, np.ndarray],
    out_dir: pathlib.Path = None,
) -> pd.DataFrame:
    """
    Add either:
      • a column "mhc_embedding" holding the full tensor  (object dtype)
      • or a column "mhc_embedding_path" pointing to a .npy on disk.

    Rows with no matching embedding are dropped (could also be imputed).
    """
    # Pre-compute paths if we are writing .npy files
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    paths, embeds = [], []
    # Extract unique alleles
    unique_alleles = df["allele"].unique()
    print(f"Processing {len(unique_alleles)} unique alleles...")
    allele_to_key_map = {}
    allele_to_emb_map = {}

    # Process each unique allele once
    for allele in tqdm(unique_alleles, desc="Creating allele mapping"):
        key = normalise_netmhcpan_allele(allele)
        emb = emb_dict.get(key)

        if emb is None:
            if mhc_class == 2:
                key1, key2 = normalize_netmhcpan_allele_to_pmgen(allele)
                print(key1, key2)
                emb1 = emb_dict.get(key1)
                emb2 = emb_dict.get(key2)
                if emb1 is not None and emb2 is not None:
                    try:
                        emb = np.concatenate([emb1, emb2], axis=0)
                    except ValueError as e:
                        print(f"Error concatenating embeddings for {allele}: {e}")
                        emb = None
                elif emb1 is not None:
                    emb = emb1
                elif emb2 is not None:
                    emb = emb2
                else:
                    print(f"No embedding found for {allele} (keys tried: {key1}, {key2})")
                    emb = None
                # # Try to find the first chain
                # prefix_matches1 = [k for k in emb_dict if k.startswith(key1)] if key1 else []
                # if key2 is not None:
                #     # Try to find the second chain
                #     prefix_matches2 = [k for k in emb_dict if k.startswith(key2)]
                #     if prefix_matches1 and prefix_matches2:
                #         key = max(prefix_matches1, key=len)
                #         print(f"{allele} -> {key}")
                #         emb = emb_dict.get(key)
                #         if emb is None:
                #             key = max(prefix_matches2, key=len)
                #             print(f"{allele} -> {key}")
                #             emb = emb_dict.get(key)
                #     else:
                #         key = max(prefix_matches1, key=len) if prefix_matches1 else None
                #         print(f"{allele} -> {key}")
                #         emb = emb_dict.get(key)
            else:
                key1, _ = normalize_netmhcpan_allele_to_pmgen(allele)
                print(key1)
                # # find exact prefix matches for longer names
                # prefix_matches = [k for k in emb_dict if k.startswith(key1)]
                # if prefix_matches:
                #     key = max(prefix_matches, key=len)
                print(f"{allele} -> {key1}")
                emb = emb_dict.get(key1)

        allele_to_key_map[allele] = key
        allele_to_emb_map[allele] = emb

    # Now use the maps for all rows
    paths, embeds = [], []
    for allele in tqdm(df["allele"], desc="Attaching embeddings"):
        key = allele_to_key_map.get(allele)
        emb = allele_to_emb_map.get(allele)

        if emb is None:
            paths.append(None)
            embeds.append(None)
            continue

        if out_dir is None:
            embeds.append(emb)
            paths.append(None)
        else:
            fname = key.replace("*", "").replace(" ", "").replace("/HLA", "") + ".npy"
            fpath = out_dir / fname
            if not fpath.exists():                    # avoid double-saving
                np.save(fpath, emb)
            paths.append(str(fpath))
            embeds.append(None)

    if out_dir is None:
        df = df.assign(mhc_embedding=embeds)
    else:
        df = df.assign(mhc_embedding_path=paths)

    # Drop rows where we could not find an embedding
    n_before = len(df)
    # print unique alleles of missing embeddings
    if "mhc_embedding" in df.columns:
        missing_alleles = df.loc[df["mhc_embedding"].isna(), "allele"].unique()
    else:
        missing_alleles = df.loc[df["mhc_embedding_path"].isna(), "allele"].unique()
    if len(missing_alleles) > 0:
        print(f"Missing embeddings for {len(missing_alleles):,} alleles:")
        print(", ".join(sorted(missing_alleles)))
    else:
        print("✓ All embeddings found.")
    df = df.dropna(
        subset=["mhc_embedding"] if out_dir is None else ["mhc_embedding_path"]
    )
    print(f"Dropped {n_before - len(df):,} rows with no MHC-{mhc_class} embedding.")
    return df


def create_test_set(df: pd.DataFrame, bench_df= pd.DataFrame, samples_per_label: int=10000) -> dict[str, pd.DataFrame]:
    """
    Create two test sets:
      - test1: equally sample samples_per_label from each assigned_label
      - test2: select all entries of the allele with the lowest count in train
    """
    datasets = {}
    train_df_ = df.copy()

    # test1: sample equally from each label
    test1 = (
        train_df_
        .groupby('assigned_label', group_keys=False)
        .sample(n=samples_per_label, random_state=42)
        .reset_index(drop=True)
    )
    train_mask = ~train_df_.index.isin(test1.index)
    train_updated = train_df_.loc[train_mask].reset_index(drop=True)

    # remove alleles that are in the benchmark dataset from the train set
    if not bench_df.empty:
        # ensure the allele column is of type string
        bench_df['allele'] = bench_df['allele'].astype(str)
        # filter out alleles that are in the benchmark dataset
        train_updated = train_updated[
            ~train_updated['allele'].isin(bench_df['allele'])
        ].reset_index(drop=True)

        # drop the rows with nan assigned_label, allele and long_mer from the benchmark dataset
        bench_df = bench_df.dropna(subset=['assigned_label', 'allele', 'long_mer'])
        # ensure the assigned_label is of type int
        bench_df['assigned_label'] = bench_df['assigned_label'].astype(int)
        # ensure the long_mer is of type string
        bench_df['long_mer'] = bench_df['long_mer'].astype(str)
        # ensure the allele is of type string
        bench_df['allele'] = bench_df['allele'].astype(str)
        # ensure the mhc_class is of type int
        bench_df['mhc_class'] = bench_df['mhc_class'].astype(int)
        datasets["benchmark_dataset"] = bench_df


    # test2: remove lowest-frequency allele from train to form test2
    allele_counts = train_updated['allele'].value_counts()
    n_test2_samples = 0
    i = 1
    while n_test2_samples < 1000:
        i += 1
        lowest_alleles = allele_counts.nsmallest(n=i).index.tolist()
        test2 = train_updated[train_updated['allele'].isin(lowest_alleles)].copy()
        n_test2_samples = test2.shape[0]

    train_updated = (
        train_updated
        [train_updated['allele'].isin(lowest_alleles) == False]
        .reset_index(drop=True)
    )

    datasets['train'] = train_updated
    datasets['test1'] = test1
    datasets['test2'] = test2

    return datasets


def main() -> None:
    print("→ Loading cleaned NetMHCpan CSV")
    df = pd.read_csv(
        CSV_PATH,
        usecols=["long_mer", "assigned_label", "allele", "mhc_class"],
    )

    # filter out
    df = df[df["mhc_class"] == mhc_class]

    print(len(df), "rows in the dataset after filtering for MHC class", mhc_class)

    # drop mhc_class column
    df = df.drop(columns=["mhc_class"])

    print(f"→ Loading MHC class {mhc_class} embeddings")
    emb_dict = load_mhc_embeddings(NPZ_PATH)

    # save keys to a text file
    emb_keys_path = NPZ_PATH.parent / f"mhc{mhc_class}_emb_keys.txt"
    with open(emb_keys_path, "w") as f:
        for key in emb_dict.keys():
            f.write(f"{key}\n")

    print("→ Merging")
    df = attach_embeddings(df, emb_dict, EMB_OUT_DIR)

    print(len(df), "rows in the dataset after filtering for Mer")

    # print("→ Dropping duplicates and nones")
    df = df.drop_duplicates(subset=["long_mer", "allele"])
    df = df.dropna(subset=["long_mer"])

    print(len(df), "rows in the dataset after dropping duplicates and Nones")

    # move the label column to the end
    label_col = "assigned_label"
    cols = [col for col in df.columns if col != label_col] + [label_col]
    df = df[cols]

    print(len(df), "rows in the dataset after dropping duplicates and Nones")

    print(f"→ Writing parquet to {OUT_PARQUET}")
    df.to_parquet(OUT_PARQUET, engine="pyarrow", index=False, compression="zstd")

    print(f"→ Dataset shape: {df.shape}")


    # load benchmark datasets
    # bench_df1 = pd.read_csv(
    #     "../data/Custom_dataset/benchmark_Conbot.csv",
    #     usecols=["long_mer", "binding_label", "allele", "mhc_class"],
    #     # rename columns to match the main dataset
    #     dtype={"binding_label": "int8", "allele": "string", "mhc_class": "int8"},
    #     names=["long_mer", "assigned_label", "allele", "mhc_class"],
    #     # convert binding_label to assigned_label
    #     converters={"assigned_label": lambda x: 1 if x == "1" else 0},
    # )
    bench_df1 = pd.read_csv(
        "../data/Custom_dataset/benchmark_Conbot.csv")
    # rename columns to match the main dataset
    bench_df1 = bench_df1.rename(columns={
        "binding_label": "assigned_label",
    })
    # ensure the assigned_label is of type int
    bench_df1['assigned_label'] = bench_df1['assigned_label'].astype(int)
    # ensure the allele is of type string
    bench_df1['allele'] = bench_df1['allele'].astype(str)
    # convert II to 2 # and I to 1
    bench_df1['mhc_class'] = bench_df1['mhc_class'].replace({"II": 2, "I": 1})

    print(bench_df1.columns)
    # TODO process later
    bench_df2 = pd.read_csv(
        "../data/Custom_dataset/benchmark_ConvNeXT.csv",
        usecols=["allele"],

    )

    # combine benchmark datasets
    bench_df = pd.concat([bench_df1, bench_df2], ignore_index=True)

    print("→ Create and save test sets")
    datasets = create_test_set(df, bench_df)
    for name, subset in datasets.items():
        print(f"{name.capitalize()} set shape: {subset.shape}")
        if name != "train":
            subset.to_parquet(OUT_PARQUET.parent / f"{name}.parquet", index=False, engine="pyarrow", compression="zstd")

    ###
    # TODO remove later
    print("→ Loading existing dataset from parquet")
    datasets = {}
    datasets['train'] = pd.read_parquet(OUT_PARQUET, engine="pyarrow")
    ###

    # Drop NaNs before converting to int
    print("→ number of rows in train set before normalization:", len(datasets['train']))
    datasets['train'] = datasets['train'].dropna(subset=['assigned_label', 'allele'])
    print("→ number of rows in train set after normalization:", len(datasets['train']))
    datasets['train']['assigned_label'] = datasets['train']['assigned_label'].astype(int)

    print("→ Creating cross-validation folds")
    folds = create_k_fold_leave_one_out_stratified_cv(
        datasets['train'],
        target_col="assigned_label",
        k=K,
        id_col="allele",
        augmentation=AUGMENTATION,
    )

    print("→ Saving folds to CSV")
    # Ensure the output directory exists /folds
    (OUT_PARQUET.parent / "folds").mkdir(parents=True, exist_ok=True)
    held_out_ids_path = OUT_PARQUET.parent / "folds" / "held_out_ids.txt"
    if held_out_ids_path.exists():
       held_out_ids_path.unlink()

    for fold_id, (train_df, val_df, validation_ids) in enumerate(folds, start=1):
       train_path = OUT_PARQUET.parent / f"folds/fold_{fold_id}_train.parquet"
       val_path   = OUT_PARQUET.parent / f"folds/fold_{fold_id}_val.parquet"
       train_df.to_parquet(train_path, index=False, engine="pyarrow", compression="zstd")
       val_df.to_parquet(val_path, index=False, engine="pyarrow", compression="zstd")
       print(f"Saved fold {fold_id} train to {train_path}")
       print(f"Saved fold {fold_id} val to {val_path}")
       if isinstance(validation_ids, list):
           ids_str = ", ".join(map(str, validation_ids))
       else:
           ids_str = str(validation_ids)
       with open(held_out_ids_path, "a") as f:
           f.write(f"Fold {fold_id}: {ids_str}\n")

    print("✓ Done")


if __name__ == "__main__":
    main()
