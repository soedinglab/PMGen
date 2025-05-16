# load Conbotnet subset data
import os
import random
import sys
import pandas as pd
import numpy as np
from utils.processing_functions import create_k_fold_leave_one_out_stratified_cross_validation, create_progressive_k_fold_cross_validation


def load_data(file_path, sep=","):
    """
    Load data from a file and return a DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")

    df = pd.read_csv(file_path, sep=sep)
    return df


def select_columns(df, columns):
    """
    Select specific columns from a DataFrame.
    """
    print("DF columns:", df.columns)
    print("Selected columns:", columns)
    return df[columns]


# def get_netmhcpan_allele(allele, netmhcpan_dataset=None, allele_cache={}):
#     """
#     Get the allele from the sequence in NetMHCpan dataset.
#     Returns the allele with the highest sequence similarity.
#
#     Parameters:
#     -----------
#     allele : str or list
#         Allele(s) to format and match in NetMHCpan format
#     netmhcpan_dataset : pd.DataFrame, optional
#         Pre-loaded NetMHCpan dataset to avoid repeated file reading
#     allele_cache : dict, optional
#         Cache of previously processed alleles
#
#     Returns:
#     --------
#     str or dict
#         Matching allele(s) in NetMHCpan format
#     """
#     # Check if we're processing a single allele or multiple alleles
#     if isinstance(allele, list) or isinstance(allele, np.ndarray):
#         # Process unique alleles and create a mapping
#         unique_alleles = set(allele)
#         allele_map = {}
#         for a in unique_alleles:
#             allele_map[a] = get_netmhcpan_allele(a, netmhcpan_dataset, allele_cache)
#         return allele_map
#
#     # Check if this allele is already in cache
#     if allele in allele_cache:
#         return allele_cache[allele]
#
#     # Load dataset if not provided
#     if netmhcpan_dataset is None:
#         netmhcpan_dataset = pd.read_csv("data/HLA_alleles/pseudoseqs/PMGen_pseudoseq.csv")
#
#     # Format allele name to NetMHCpan format
#     formatted_allele = format_allele(allele)
#
#     # First, try exact match (case-insensitive)
#     exact_matches = netmhcpan_dataset[netmhcpan_dataset['allele'].str.lower() == formatted_allele.lower()]
#
#     if not exact_matches.empty:
#         result = exact_matches.iloc[0]['allele']
#     else:
#         # If no exact match, try partial match
#         partial_matches = netmhcpan_dataset[netmhcpan_dataset['allele'].str.contains(formatted_allele, case=False)]
#
#         if not partial_matches.empty:
#             result = partial_matches.iloc[0]['allele']
#         else:
#             # If no match at all, report and use formatted allele
#             print(f"No match found for allele: {allele} (formatted as {formatted_allele})")
#             result = formatted_allele
#
#     # Cache the result
#     allele_cache[allele] = result
#     return result
#
# def format_allele(allele):
#     """Helper function to format allele names to NetMHCpan format"""
#     # Format DRB alleles (like DRB11402 to DRB1*14:02)
#     if allele.startswith('DRB') and not '-' in allele:
#         locus = allele[:4]
#         allele_num = allele[4:]
#         if len(allele_num) >= 4:
#             return f"{locus}*{allele_num[:2]}:{allele_num[2:]}"
#
#     # Format MHC-II heterodimers (like HLA-DQA10501-DQB10301 to HLA-DQA1*05:01-DQB1*03:01)
#     elif '-' in allele and not '*' in allele:
#         parts = allele.split('-')
#         if len(parts) == 2 and '-' in parts[1]:  # Handle cases with another hyphen
#             prefix, alpha_beta = parts
#             alpha, beta = alpha_beta.split('-')
#
#             # Format alpha and beta chains
#             alpha = format_chain(alpha)
#             beta = format_chain(beta)
#
#             return f"{prefix}-{alpha}-{beta}"
#         elif len(parts) == 3:  # Format like HLA-DQA10501-DQB10301
#             prefix, alpha, beta = parts
#
#             # Format alpha and beta chains
#             alpha = format_chain(alpha)
#             beta = format_chain(beta)
#
#             return f"{prefix}-{alpha}-{beta}"
#
#     # Default case - return allele as is
#     return allele
#
# def format_chain(chain):
#     """Format an MHC chain like DQA10501 to DQA1*05:01"""
#     if len(chain) >= 7:  # e.g., DQA10501
#         locus = chain[:4]
#         num = chain[4:]
#         return f"{locus}*{num[:2]}:{num[2:]}"
#     return chain

from tqdm import tqdm
# Optional: for parquet streaming
import pyarrow as pa
import pyarrow.parquet as pq


def process_chunk_df(df_chunk, binding_labels, mhc_sequences):
    """
    Process a DataFrame chunk: assign binding_label and mhc_sequence, map labels to ints.
    """
    # Standardize column names
    if 'allotype' not in df_chunk.columns and 'MHC' in df_chunk.columns:
        df_chunk = df_chunk.rename(columns={'MHC': 'allotype'})
    if 'peptide' not in df_chunk.columns and 'long_mer' in df_chunk.columns:
        df_chunk = df_chunk.rename(columns={'long_mer': 'peptide'})

    # Assign binding labels via map on MultiIndex for vectorized lookup
    mi = pd.MultiIndex.from_frame(df_chunk[['allotype', 'peptide']])
    df_chunk['binding_label'] = mi.map(binding_labels)

    # Assign mhc_sequence via map
    df_chunk['mhc_sequence'] = df_chunk['allotype'].map(mhc_sequences)

    # Map string labels to integers
    unique_labels = pd.unique(df_chunk['binding_label'].dropna())
    label_mapping = {lbl: i for i, lbl in enumerate(sorted(unique_labels))}
    df_chunk['binding_label'] = df_chunk['binding_label'].map(label_mapping)

    return df_chunk


def add_binding_label_streaming(input_path, map_csv, output_dir=None, is_netmhcpan=False,
                                csv_chunksize=1_000_000):
    """
    Stream-process large files (CSV or Parquet) on disk, without loading fully into memory.

    Parameters:
    -----------
    input_path : str
        Path to a file or directory of files (.csv or .parquet) to process.
    csv_path : str
        Path to small CSV file for binding_label and mhc_sequence mappings.
    output_dir : str, optional
        Directory to write processed files; if None, original files will be overwritten.
    is_netmhcpan : bool
        Whether the CSV mapping file uses 'assigned_label' instead of 'binding_label'.
    csv_chunksize : int
        Number of rows per chunk when reading mapping CSV (if large), default 1e6.
    """
    print(f"Starting streaming addition of binding labels for: {input_path}")
    df_map = map_csv

    # Ensure column consistency
    df_map = df_map.rename(columns={
        'long_mer': 'peptide'
    })
    print(f"Loaded csv file with shape: {df_map.shape}")
    # Build dicts for mapping
    binding_labels = dict(zip(zip(df_map['allele'], df_map['peptide']), df_map['binding_label']))
    mhc_sequences = dict(zip(df_map['allele'], df_map['mhc_sequence'])) if 'mhc_sequence' in df_map.columns else {}

    def get_output_path(in_path):
        fname = os.path.basename(in_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            return os.path.join(output_dir, fname)
        return in_path

    def stream_file(file_path):
        fname = os.path.basename(file_path)
        print(f"\n--- Processing file: {fname} ---")
        out_path = get_output_path(file_path)
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.csv':
            first_write = True
            for i, chunk in enumerate(tqdm(pd.read_csv(file_path, chunksize=csv_chunksize), desc=f"Chunks ({fname})")):
                print(f"Processing chunk {i + 1}")
                processed = process_chunk_df(chunk, binding_labels, mhc_sequences)
                processed.to_csv(out_path, mode='w' if first_write else 'a', index=False, header=first_write)
                first_write = False
            print(f"Finished writing CSV to: {out_path}")

        elif ext in ['.parquet', '.pq']:
            pf = pq.ParquetFile(file_path)
            writer = None
            print(f"Parquet row groups: {pf.num_row_groups}")
            for rg in tqdm(range(pf.num_row_groups), desc=f"RowGroups ({fname})"):
                print(f"Reading row group {rg}")
                partial = pf.read_row_group(rg).to_pandas()
                processed = process_chunk_df(partial, binding_labels, mhc_sequences)
                table = pa.Table.from_pandas(processed)
                if writer is None:
                    writer = pq.ParquetWriter(get_output_path(file_path), table.schema)
                writer.write_table(table)
            if writer:
                writer.close()
            print(f"Finished writing Parquet to: {out_path}")

        else:
            print(f"Skipped unsupported file type: {fname}")

    # Handle directory or single file
    if os.path.isdir(input_path):
        files = [f for f in os.listdir(input_path) if os.path.splitext(f)[1].lower() in ['.csv', '.parquet', '.pq']]
        print(f"Found {len(files)} files to process in directory.")
        for fname in files:
            file_path = os.path.join(input_path, fname)
            stream_file(file_path)
    else:
        stream_file(input_path)


def main(dataset_name="Conbotnet", mhc_type="mhc2", subset_prop=1.0, n_folds=5, process_fold_n=None, chunk_n=0):
    """
    Prepare datasets for Pep2Vec training and evaluation.

    Parameters:
    -----------
    dataset_name : str
        Name of the dataset directory (e.g., "Conbotnet", "ConvNeXT-MHC", "NetMHCpan_dataset")
    mhc_type : str
        Type of MHC ("mhc1" or "mhc2")
    subset_prop : float
        Proportion of data to use (1.0 = all data)
    n_folds : int
        Number of cross-validation folds
    """
    # Setup paths
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")
    dataset_dir = os.path.join(data_dir, dataset_name)
    folds_dir = os.path.join(dataset_dir, "folds")
    output_dir = os.path.join(data_dir, "Pep2Vec", dataset_name)

    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(folds_dir, exist_ok=True)

    # Define paths and columns based on dataset
    is_netmhcpan = dataset_name.lower() == "netmhcpan_dataset"

    if not is_netmhcpan:
        # Other datasets have separate train/test files
        paths = {
            'train': os.path.join(dataset_dir, "train.csv"),
            'test': os.path.join(dataset_dir, "test_all.csv")
        }
        columns = ["allele", "long_mer", "binding_label", "mhc_sequence"]
        rename_map = {"allele": "allotype", "long_mer": "peptide"}  # required for pep2vec

    # Column configuration
    # Rename label column for NetMHCpan
    if is_netmhcpan:
        # NetMHCpan has a single cleaned_data.csv file
        # cleaned_data_path = os.path.join(dataset_dir, f"combined_data_{mhc_type[-1]}.csv")
        cleaned_data_path = os.path.join(dataset_dir, f"chunks_I/subset_balanced_300k.csv")
        print(cleaned_data_path)
        rename_map = {"allele": "allotype", "Peptide": "peptide"}  # required for pep2vec
        rename_map["assigned_label"] = "binding_label"
        columns = ["allotype", "peptide", "binding_label"]

    # Load and process datasets
    datasets = {}

    if is_netmhcpan:
        # Load NetMHCpan dataset from single file
        print(f"Loading NetMHCpan dataset from {cleaned_data_path}")
        # df = load_data(cleaned_data_path)
        usecols = ['allele', 'peptide', 'assigned_label', 'mhc_sequence', 'mhc_class']
        rng = random.Random(42)
        df = pd.read_csv(cleaned_data_path, usecols=usecols)
        if subset_prop < 1.0:
            df = df.sample(frac=subset_prop, random_state=42)
        df_map = df.rename(columns={'assigned_label': 'binding_label', 'peptide': 'long_mer'})
        print(f"Full dataset shape: {df.shape}")

        # rename columns
        if "binding_label" not in df.columns and "assigned_label" in df.columns:
            df.rename(columns=rename_map, inplace=True)

        # Check the dtype
        print("mhc_class dtype:", df["mhc_class"].dtype)

        # And get counts of each class to see their distribution
        print(df["mhc_class"].value_counts(dropna=False))

        # Filter by MHC class based on mhc_type parameter
        mhc_class = 2 if mhc_type.lower() == "mhc2" else 1
        df = df[df["mhc_class"] == mhc_class].copy()
        print(f"Dataset after filtering for MHC class {mhc_class}: {df.shape}")

        subset = df[["allotype", "peptide", "binding_label"]]
        print(subset.info())
        print(subset.head(10))
        print(subset.isna().sum())
        total = len(df)
        uniques = df[["allotype", "peptide"]].dropna().drop_duplicates().shape[0]
        print(f"Total rows: {total:,}; unique (allotype, peptide): {uniques:,}")
        print(df.filter(like="label").columns)
        print(df["binding_label"].notna().sum())

        # TODO fix
        # Process dataframes
        # select + clean
        train_df = (
            df[["allotype", "peptide", "binding_label"]]
            .dropna()
            .drop_duplicates(subset=["allotype", "peptide"])
        )

        print(f"Train dataset shape after processing: {train_df.shape}")
        print(train_df["binding_label"].value_counts())

        datasets['train'] = train_df
        datasets['test'] = None
    else:
        # Regular dataset processing with separate files
        for name, path in paths.items():
            df = pd.read_csv(path)
            df_map = df.copy()

            # Process dataframe
            df = (select_columns(df, columns)
                  .rename(columns=rename_map)
                  .drop_duplicates(subset=["allotype", "peptide"])
                  .dropna())

            print(f"{name.capitalize()} dataset shape after processing: {df.shape}")
            datasets[name] = df

    # # change the allele names to the ones in the NetMHCpan dataset
    # # Load the NetMHCpan dataset once
    # netmhcpan_dataset = pd.read_csv("data/HLA_alleles/pseudoseqs/PMGen_pseudoseq.csv")
    # # Create a shared cache for alleles
    # allele_cache = {}
    #
    # # Apply to train and test datasets with shared resources
    # datasets['train']['allotype'] = datasets['train']['allotype'].apply(
    #     lambda x: get_netmhcpan_allele(x, netmhcpan_dataset, allele_cache))
    # datasets['test']['allotype'] = datasets['test']['allotype'].apply(
    #     lambda x: get_netmhcpan_allele(x, netmhcpan_dataset, allele_cache))

    # define test1 and test2 datasets
    # test1: balanced sampling with equal representation from each binding_label
    train_df_ = datasets['train']

    # Determine sample size (minimum of 1000 or smallest class size)
    samples_per_label = min(1000, min(train_df_['binding_label'].value_counts()))
    print(f"Creating balanced test1 with {samples_per_label} samples per label")

    # Sample equally from each label
    test1 = (train_df_
             .groupby('binding_label', group_keys=False)  # no re‑index shuffle
             .sample(n=samples_per_label, random_state=42)  # vectorised sample
             .reset_index(drop=True))

    train_mask = ~train_df_.index.isin(test1.index)
    train_updated = train_df_.loc[train_mask]

    datasets['train'] = train_updated

    # test2: select allele with lowest sample count
    allele_counts = datasets['train']['allotype'].value_counts()
    lowest_allele = allele_counts.idxmin()
    test2 = datasets['train'][datasets['train']['allotype'] == lowest_allele].copy()
    datasets['train'] = datasets['train'][datasets['train']['allotype'] != lowest_allele].reset_index(drop=True)

    if process_fold_n == 0 or process_fold_n == None:
        # Create k-fold cross-validation splits
        k = n_folds
        folds = create_progressive_k_fold_cross_validation(
            datasets['train'], k=k, target_col="binding_label",
            id_col="allotype"
        )

        held_out_ids_path = os.path.join(folds_dir, "held_out_ids.txt")
        if os.path.exists(held_out_ids_path):
            os.remove(held_out_ids_path)

        # Save folds
        for i, (train_set, val_set, held_out_id) in enumerate(folds):
            # only keep the allele and peptide columns
            train_set = train_set[["allotype", "peptide"]]
            val_set = val_set[["allotype", "peptide"]]
            # drop nan and duplicates
            train_set = train_set.dropna().drop_duplicates(subset=["allotype", "peptide"])
            val_set = val_set.dropna().drop_duplicates(subset=["allotype", "peptide"])
            # TODO find a better solution later - pep2vec can't handle long peptides longer than ~25 might work upto 29
            train_set = train_set[train_set["peptide"].str.len() <= 25]
            val_set = val_set[val_set["peptide"].str.len() <= 25]

            # reset the index
            train_set.reset_index(drop=True, inplace=True)
            val_set.reset_index(drop=True, inplace=True)
            with open(os.path.join(os.path.dirname(__file__), "data", dataset_name, "folds", f"train_set_fold_{i}.csv"),
                      mode="w", encoding="utf-8") as train_file:
                train_set.to_csv(train_file, sep=",", index=True, header=True)
            with open(os.path.join(os.path.dirname(__file__), "data", dataset_name, "folds", f"val_set_fold_{i}.csv"),
                      mode="w", encoding="utf-8") as val_file:
                val_set.to_csv(val_file, sep=",", index=True, header=True)
            with open(held_out_ids_path, "a") as f:
                f.write(f"Fold {i}: {held_out_id}\n")

        # Save the test set
        # if dataset has a test set, save it as well
        if datasets.get('test') is not None:
            datasets['test'].to_csv(os.path.join(folds_dir, "test_original.csv"), index=False, header=True)
        # Save the test1 and test2 sets
        test1.to_csv(os.path.join(folds_dir, "test1_stratified.csv"), index=False, header=True)
        test2.to_csv(os.path.join(folds_dir, "test2_single_unique_allele.csv"), index=False, header=True)
        print("Test dataset saved.")

    # load the fold csv files for specified fold
    train_path = os.path.join(folds_dir, f"train_set_fold_{process_fold_n}.csv")
    val_path = os.path.join(folds_dir, f"val_set_fold_{process_fold_n}.csv")
    if os.path.exists(train_path) and os.path.exists(val_path):
        datasets['train'] = pd.read_csv(train_path, index_col=0)
        datasets['val'] = pd.read_csv(val_path, index_col=0)
    else:
        raise FileNotFoundError(f"Fold files not found for fold {process_fold_n}")

    ################### Pep2Vec ###################
    # automatically get the number of cores
    num_cores = max(os.cpu_count() - 2, 1)
    # TODO only process one fold? with process_fold_n
    if process_fold_n == None:
        for i in range(n_folds):
            for split in ['train', 'val']:
                csv_in = os.path.join(folds_dir, f"{split}_set_fold_{i}.csv")
                out_pq = os.path.join(output_dir, f"pep2vec_output_{split}_fold_{i}.parquet")
                if os.path.exists(csv_in):
                    os.system(
                        f"./Pep2Vec/pep2vec.bin --num_threads {num_cores} --dataset {csv_in} --output_location {out_pq} --mhctype {mhc_type}")
    else:
        for split in ['train', 'val']:
            csv_in = os.path.join(folds_dir, f"{split}_set_fold_{process_fold_n}.csv")
            out_pq = os.path.join(output_dir, f"pep2vec_output_{split}_fold_{process_fold_n}.parquet")
            if os.path.exists(csv_in):
                os.system(
                    f"./Pep2Vec/pep2vec.bin --num_threads {num_cores} --dataset {csv_in} --output_location {out_pq} --mhctype {mhc_type}")

    # # test set
    test_file = os.path.join(folds_dir, "test_original.csv")
    if os.path.exists(test_file):
        output_file = os.path.join(output_dir, f"pep2vec_output_test.parquet")
        os.system(
            f"./Pep2Vec/pep2vec.bin --num_threads {num_cores} --dataset {test_file} --output_location {output_file} --mhctype {mhc_type}")
    # Process test1 and test2 datasets
    for test_name in ["test1_stratified", "test2_single_unique_allele"]:
        test_file = os.path.join(folds_dir, f"{test_name}.csv")
        if os.path.exists(test_file):
            output_file = os.path.join(output_dir, f"pep2vec_output_{test_name}.parquet")
            os.system(
                f"./Pep2Vec/pep2vec.bin --num_threads {num_cores} --dataset {test_file} --output_location {output_file} --mhctype {mhc_type}")
    ###############################################

    return df_map


if __name__ == "__main__":
    if len(sys.argv) > 1:
        fold_n = sys.argv[1]
    else:
        fold_n = None

    print(f"Running for Fold {fold_n}")

    # main("Conbotnet", "mhc2", 1, 5, fold_n)
    # df_map = None
    # if not df_map:
    #     df1 = pd.read_csv(os.path.join("data", "Conbotnet", "train.csv"),)
    #     df2 = pd.read_csv(os.path.join("data", "Conbotnet", "test_all.csv"),)
    #     df_map = pd.concat([df1, df2])
    # add_binding_label_streaming(
    #     input_path=os.path.join("data", "Pep2Vec", "Conbotnet"),
    #     map_csv=df_map,
    #     output_dir=os.path.join("data", "Pep2Vec", "Conbotnet_new_subset"),
    #     is_netmhcpan=False
    # )

    df_map = main("ConvNeXT-MHC", "mhc1", 0.01, 5, fold_n)
    # df_map = None
    # if not df_map:
    #     df1 = pd.read_csv(os.path.join("data", "ConvNeXT-MHC", "train.csv"),)
    #     df2 = pd.read_csv(os.path.join("data", "ConvNeXT-MHC", "test_all.csv"),)
    #     df_map = pd.concat([df1, df2])
    add_binding_label_streaming(
        input_path=os.path.join("data", "Pep2Vec", "ConvNeXT-MHC"),
        map_csv=df_map,
        output_dir=os.path.join("data", "Pep2Vec", "ConvNeXT-MHC_new_subset")
    )
    # main("NetMHCpan_dataset", "mhc2", 0.01, 5)
    # add_binding_label(
    #     df_path=os.path.join("data", "Pep2Vec", "NetMHCIIpan_dataset"),
    #     train_path=os.path.join("data", "NetMHCIIpan_dataset", "cleaned_data.csv"),
    #     output_dir=os.path.join("data", "Pep2Vec", "NetMHCIpan_dataset_new_subset")
    # )
    # df_map = main("NetMHCpan_dataset", "mhc1", 1, 5, fold_n)
    # add_binding_label_streaming(
    #     input_path=os.path.join("data", "Pep2Vec", "NetMHCpan_dataset"),
    #     map_csv=df_map,
    #     output_dir=os.path.join("data", "Pep2Vec", "NetMHCIpan_dataset_new_subset"),
    #     is_netmhcpan=True
    # )
