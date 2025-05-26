#!/usr/bin/env python
"""
requires python3.10+
esm_embed.py
============

Generate protein embeddings with ESM-C or any other EvolutionaryScale/Meta
protein language model.

Examples
--------
# 1. Local ESM-C-300M on GPU 0, output NPZ
python esm_embed.py --input proteins.fa --model esmc_300m \
                    --device cuda:0 --outfile embeddings.npz

# 2. Remote ESM-3-large (98 B) via Forge API
export ESM_API_TOKEN="hf_xxxxxxxxxxxxxxxxxx"
python esm_embed.py --input proteins.fa --model esm3-98b-2024-08 \
                    --remote --outfile embeds.parquet
"""
from __future__ import annotations
import argparse, os, sys, json, time, itertools, pathlib, warnings
from typing import List, Tuple, Dict, Iterable
from esm.sdk.api import ESMProtein, LogitsConfig

import torch
import numpy as np
import pandas as pd
import tqdm
import csv

###############################################################################
# ---------------------------  I/O utilities  ---------------------------------
###############################################################################
def read_dat(path: str) -> List[Tuple[str, str]]:
    """Read tab-separated file: first col is id, second is sequence."""
    seqs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if "\t" in line:
                line = line.split("\t", 1)
            elif " " in line:
                line = line.split(" ")
            else:
                line = line.split(maxsplit=1)
            if not line or len(line) < 2:
                print(line)
                continue

            if len(line) == 2:
                seqs.append((line[0], line[1]))

    return seqs


def read_csv(path: str = [0, 1]) -> List[Tuple[str, ...]]:
    seqs: List[Tuple[str, ...]] = []
    selected_cols = ["simple_allele", "sequence", "mhc_types"]
    file = pd.read_csv(path, sep=",", usecols=selected_cols)
    file = file[file["mhc_types"] == 1]
    print(file.columns)
    # convert simple allele, remove * and : to mtach with netmhcpan
    file[selected_cols[0]] = file[selected_cols[0]].str.replace("*", "")
    file[selected_cols[0]] = file[selected_cols[0]].str.replace(":", "")
    # return as list of tuples
    for index, row in file.iterrows():
        seqs.append((row[selected_cols[0]], row[selected_cols[1]]))
    return seqs

###############################################################################
# -------------  Local model loader (ESM-C, ESM-2, ESM3-open)  ----------------
###############################################################################
def load_local_model(model_name: str, device: str):
    """
    Return (model, to_tensor_fn) for **new** ESM-C / ESM-3
    or (model, batch_converter_fn) for **legacy** ESM-2.
    """
    try:                                              # new-style ESM-C
        if model_name.startswith("esmc"):
            from esm.models.esmc import ESMC
            from esm.sdk.api import ESMProtein, LogitsConfig
            model = ESMC.from_pretrained(model_name).to(device)
            def embed_one(seq: str):
                protein = ESMProtein(sequence=seq)
                t = model.encode(protein)
                out = model.logits(t, LogitsConfig(sequence=True,
                                                   return_embeddings=True))
                return out.embeddings.mean(0).cpu().numpy()
            return embed_one
        # new-style ESM-3 open weight
        if model_name.startswith("esm3"):
            from esm.models.esm3 import ESM3
            from esm.sdk.api import ESMProtein, LogitsConfig
            model = ESM3.from_pretrained(model_name).to(device)
            def embed_one(seq: str):
                protein = ESMProtein(sequence=seq)
                t = model.encode(protein)
                out = model.logits(t, LogitsConfig(sequence=True,
                                                   return_embeddings=True))
                return out.embeddings.mean(0).cpu().numpy()
            return embed_one
    except ImportError:
        pass  # will try legacy route below

    # ---------- legacy facebookresearch/esm (ESM-1/2) ----------
    try:
        from esm import pretrained
        create = getattr(pretrained, model_name,
                         pretrained.load_model_and_alphabet)
        model, alphabet = create(model_name) if callable(create) else create()
        model.eval().to(device)
        converter = alphabet.get_batch_converter()
        def embed_one(seq: str):
            _, _, toks = converter([("x", seq)])
            with torch.inference_mode():
                rep = model(toks.to(device),
                            repr_layers=[model.num_layers])["representations"]
            return rep[model.num_layers][0, 1:len(seq)+1].mean(0).cpu().numpy()
        return embed_one
    except Exception as e:
        raise RuntimeError(f"Don’t know how to load {model_name}: {e}")

# --------------------------------------------------------------------- #
# Remote (Forge / AWS) wrapper
# --------------------------------------------------------------------- #
def load_remote_client(model_name: str, token: str):
    import esm
    client = esm.sdk.client(model_name, token=token)
    from esm.sdk.api import ESMProtein, LogitsConfig
    def embed_one(seq: str):
        protein = ESMProtein(sequence=seq)
        t = client.encode(protein)
        out = client.logits(t, LogitsConfig(sequence=True,
                                            return_embeddings=True))
        return out.embeddings.mean(0)
    return embed_one

###############################################################################
# --------------------  Embedding extraction pipeline  ------------------------
###############################################################################
# @torch.inference_mode()
# def embed_local(
#     model,
#     batch_converter,
#     sequences: List[Tuple[str, str]],
#     batch_size: int = 8,
#     device: str = "cpu",
#     pooling: str = "mean",
# ) -> Dict[str, np.ndarray]:
#     """
#     Run local model and return {id: embedding(ndarray)} dict.
#     Pooling: "mean" (default) or "cls".
#     """
#     embeds = {}
#     for i in range(0, len(sequences), batch_size):
#         sub = sequences[i : i + batch_size]
#         labels, strs, toks = batch_converter(sub)
#         toks = toks.to(device)
#         reps = model(toks, repr_layers=[model.num_layers])["representations"][
#             model.num_layers
#         ]  # (B, L, D)
#         for label, tok, rep in zip(labels, toks, reps):
#             if pooling == "mean":
#                 mask = tok != model.alphabet.padding_idx
#                 embed = rep[mask].mean(0)
#             else:  # CLS – first token after padding_tok
#                 embed = rep[0]
#             embeds[label] = embed.cpu().numpy()
#     return embeds


def embed_remote(
    client,
    sequences: List[Tuple[str, str]],
    batch_size: int = 16,
    pooling: str = "mean",
) -> Dict[str, np.ndarray]:
    """
    Use cloud client; supports .embed() (ESM-C) and .generate_embeddings() (ESM-3).
    """
    embeds = {}
    for i in range(0, len(sequences), batch_size):
        sub = sequences[i : i + batch_size]
        ids, seqs = zip(*sub)
        if hasattr(client, "embed"):
            out = client.embed(seqs, pooling)
        else:
            out = client.generate_embeddings(seqs, pooling)
        embeds.update({idx: emb for idx, emb in zip(ids, out)})
    return embeds


###############################################################################
# --------------------------  Main CLI handler  ------------------------------
###############################################################################
def main(**local_args):
    parser = argparse.ArgumentParser(description="ESM embedding generator")

    if local_args:
        args = parser.parse_args([])
        for k, v in local_args.items():
            setattr(args, k, v)
    else:
        parser.add_argument("--input", required=True, help="dat file")
        parser.add_argument("--model", default="esmc_300m", help="Model name/id")
        parser.add_argument("--outfile", required=True, help="Output .npz or .parquet")
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
        parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
        parser.add_argument("--remote", action="store_true", help="Use cloud API")
        parser.add_argument("--api_token", default=os.getenv("ESM_API_TOKEN"), help="Forge/NIM token")
        args = parser.parse_args()

    seqs = None
    if args.input.endswith(".csv"):
        seqs = read_csv(args.input)
    elif args.input.endswith(".dat"):
        seqs = read_dat(args.input)
    if not seqs:
        sys.exit("No sequences found!")

    if args.remote:
        if args.api_token is None:
            sys.exit("Provide --api_token or set ESM_API_TOKEN")
        client = load_remote_client(args.model, args.api_token)
        embeddings = embed_remote(client, seqs, args.batch_size, args.pooling)
        print("embeddings with", args.model)
        sequences = {seq_id: seq for seq_id, seq in seqs}
    else:
        embed_one = load_local_model(args.model, args.device)
        print("embeddings with", args.model)
        # print len of sequences
        print(f"Number of sequences: {len(seqs)}")
        # print first 5 sequences
        print("First 5 sequences:", seqs[:5])
        embeddings = {}
        for seq_id, seq in tqdm.tqdm(seqs, desc="Embedding sequences"):
            embeddings[seq_id] = embed_one(seq)
        sequences = {seq_id: seq for seq_id, seq in seqs}


    # ------------------  save ------------------
    out_path = pathlib.Path(args.outfile)
    if out_path.suffix == ".npz":
        np.savez_compressed(out_path, **embeddings)
        # create a .csv file and save the sequences
        with open(out_path.with_suffix(".csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "sequence"])
            for seq_id, seq in sequences.items():
                writer.writerow([seq_id, seq])
    elif out_path.suffix == ".parquet":
        df = pd.DataFrame(
            [(k, v.astype(np.float32)) for k, v in embeddings.items()],
            columns=["id", "embedding"],
        )
        df.to_parquet(out_path, index=False)
    else:
        sys.exit("outfile must end with .npz or .parquet")

    print(f"[✓] Saved embeddings for {len(embeddings)} sequences to {out_path}")


if __name__ == "__main__":
    # example run
    model = "esmc_600m"  # "esm3-98b-2024-08" "esm3_sm_open_v1", "esm3-open"
    # dat_path = "data/NetMHCpan_dataset/NetMHCpan_train/MHC_pseudo.dat"
    dat_path = "data/NetMHCpan_dataset/NetMHCIIpan_train/pseudosequence.2016.all.X.dat"
    # dat_path = "data/HLA_alleles/pseudoseqs/PMGen_pseudoseq.csv"
    out_path = "data/ESM/mhc2_encodings.npz"
    remote = False
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size = 1
    pooling = "mean"
    # run the script
    main(input=dat_path, model=model, outfile=out_path, remote=remote,
         device=device, batch_size=batch_size, pooling=pooling)
