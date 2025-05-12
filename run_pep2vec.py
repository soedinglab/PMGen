# load Conbotnet subset data
import os
import sys
import pandas as pd
import numpy as np
from utils.processing_functions import create_k_fold_leave_one_out_stratified_cross_validation
from sklearn.model_selection import train_test_split

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

def add_binding_label(df_path, train_path, output_dir=None):
    """
    Add binding labels and MHC sequences to the DataFrame based on allotype and peptide pairs.

    Parameters:
    -----------
    df_path : DataFrame, str, or directory path
        DataFrame, path to a file, or directory containing multiple files to add binding labels to
    train_path : str
        Path to training data CSV file with binding labels
    output_dir : str, optional
        Directory to save labeled files, if None will overwrite original files

    Returns:
    --------
    DataFrame or dict of DataFrames with binding labels and MHC sequences added
    """
    # Load train and test binding label mappings
    train_df = pd.read_csv(train_path)

    # rename columns if needed
    if "allele" not in train_df.columns:
        train_df.rename(columns={"allotype": "allele"}, inplace=True)
    if "long_mer" not in train_df.columns:
        train_df.rename(columns={"peptide": "long_mer"}, inplace=True)
    if "binding_label" not in train_df.columns:
        if "assigned_label" in train_df.columns:
            train_df.rename(columns={"assigned_label": "binding_label"}, inplace=True)
        else:
            raise ValueError("No binding label column found in the training data.")

    # Create binding label mappings
    binding_labels = {
        (row["allele"], row["long_mer"]): row["binding_label"]
        for _, row in train_df.iterrows()
    }

    # Create MHC sequence mappings
    if "mhc_sequence" in train_df.columns:
        mhc_sequences = {
            row["allele"]: row["mhc_sequence"]
            for _, row in train_df.iterrows()
        }
    else:
        mhc_sequences = {}

    # Combine MHC sequence mappings (test data takes precedence if duplicates)
    mhc_sequences = {**mhc_sequences}

    # Check if df_path is a directory
    if isinstance(df_path, str) and os.path.isdir(df_path):
        # Process all files in the directory
        results = {}
        for filename in os.listdir(df_path):
            file_path = os.path.join(df_path, filename)
            if os.path.isfile(file_path) and (file_path.endswith('.csv') or file_path.endswith('.parquet')):
                print(f"Processing file: {filename}")
                results[filename] = process_single_file(
                    file_path, binding_labels, mhc_sequences, output_dir
                )
            # remove the file after processing
            if output_dir and os.path.exists(file_path):
                os.remove(file_path)
        return results
    else:
        # Process a single file or DataFrame
        return process_single_file(df_path, binding_labels, mhc_sequences, output_dir)


def process_single_file(df_or_path, binding_labels, mhc_sequences, output_dir=None):
    """Helper function to process a single file or DataFrame"""
    # Process the provided dataframe or file
    if isinstance(df_or_path, str):
        if df_or_path.endswith('.parquet'):
            df = pd.read_parquet(df_or_path)
        else:
            df = pd.read_csv(df_or_path)
        file_path = df_or_path
    else:
        df = df_or_path
        file_path = None

    # Determine if this is a test file based on filename
    is_test = file_path and "test" in os.path.basename(file_path).lower()
    binding_labels = binding_labels if is_test else binding_labels

    # Check if columns use 'allotype'/'peptide' naming convention (parquet files)
    # or 'allele'/'long_mer' naming convention (CSV files)
    has_allotype_col = "allotype" in df.columns
    has_peptide_col = "peptide" in df.columns

    # Assign labels based on mapping with appropriate column names
    if has_allotype_col and has_peptide_col:
        # Parquet file naming convention
        df["binding_label"] = df.apply(
            lambda row: binding_labels.get((row["allotype"], row["peptide"])),
            axis=1
        )
        # Add MHC sequence
        df["mhc_sequence"] = df["allotype"].map(mhc_sequences)
    else:
        # CSV file naming convention
        df["binding_label"] = df.apply(
            lambda row: binding_labels.get((row["allele"], row["long_mer"])),
            axis=1
        )
        # Add MHC sequence
        df["mhc_sequence"] = df["allele"].map(mhc_sequences)

    # Map string labels to integers
    label_mapping = {
        label: i for i, label in enumerate(sorted(df["binding_label"].dropna().unique()))
    }
    df["binding_label"] = df["binding_label"].map(label_mapping)

    # Save the file if a path was provided and output_dir is specified
    if file_path and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(file_path))
        if file_path.endswith('.parquet'):
            df.to_parquet(output_path)
        else:
            df.to_csv(output_path, index=False)

    return df


def main(dataset_name="Conbotnet", mhc_type="mhc2", subset_prop=1.0, n_folds=5):
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

    if is_netmhcpan:
        # NetMHCpan has a single cleaned_data.csv file
        cleaned_data_path = os.path.join(dataset_dir, "cleaned_data.csv")
        columns = ["long_mer","convoluted_label", "assigned_label", "MHC_class", "allele"]
    else:
        # Other datasets have separate train/test files
        paths = {
            'train': os.path.join(dataset_dir, "train.csv"),
            'test': os.path.join(dataset_dir, "test_all.csv")
        }
        columns = ["allele", "long_mer", "binding_label", "mhc_sequence"]

    # Column configuration
    rename_map = {"allele": "allotype", "long_mer": "peptide"}  # required for pep2vec

    # Rename label column for NetMHCpan
    if is_netmhcpan:
        rename_map["assigned_label"] = "binding_label"
        # TODO if mhc_sequence is required it must be further processed
        columns = ["allotype", "peptide", "binding_label"]

    # Load and process datasets
    datasets = {}

    if is_netmhcpan:
        # Load NetMHCpan dataset from single file
        print(f"Loading NetMHCpan dataset from {cleaned_data_path}")
        df = load_data(cleaned_data_path)
        print(f"Full dataset shape: {df.shape}")

        # Filter by MHC class based on mhc_type parameter
        mhc_class = 2 if mhc_type.lower() == "mhc2" else 1
        df = df[df["mhc_class"] == mhc_class].copy()
        print(f"Dataset after filtering for MHC class {mhc_class}: {df.shape}")

        # rename columns
        df.rename(columns=rename_map, inplace=True)

        # Process dataframes
        train_df = (select_columns(df, columns)
                    .rename(columns=rename_map)
                    .drop_duplicates(subset=["allotype", "peptide"])
                    .dropna())

        print(f"Train dataset shape after processing: {train_df.shape}")

        datasets['train'] = train_df
        datasets['test'] = None
    else:
        # Regular dataset processing with separate files
        for name, path in paths.items():
            df = load_data(path)
            print(f"{name.capitalize()} dataset shape: {df.shape}")

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
    # test1: stratified 10% sampling from train
    train_df_ = datasets['train']
    train_updated, test1 = train_test_split(
        train_df_,
        test_size=0.1,
        stratify=train_df_['binding_label'],
        random_state=42
    )
    datasets['train'] = train_updated.reset_index(drop=True)

    # test2: select allele with lowest sample count
    allele_counts = datasets['train']['allotype'].value_counts()
    lowest_allele = allele_counts.idxmin()
    test2 = datasets['train'][datasets['train']['allotype'] == lowest_allele].copy()
    datasets['train'] = datasets['train'][datasets['train']['allotype'] != lowest_allele].reset_index(drop=True)

    # Create k-fold cross-validation splits
    k = n_folds
    folds = create_k_fold_leave_one_out_stratified_cross_validation(
        datasets['train'], k=k, target_col="binding_label",
        id_col="allotype", train_size=0.8, subset_prop=subset_prop
    )

    # Clear previous held_out_ids file if it exists
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
        with open(os.path.join(os.path.dirname(__file__), "data", dataset_name, "folds", f"train_set_fold_{i}.csv"), mode="w", encoding="utf-8") as train_file:
            train_set.to_csv(train_file, sep=",", index=True, header=True)
        with open(os.path.join(os.path.dirname(__file__), "data", dataset_name, "folds", f"val_set_fold_{i}.csv"), mode="w", encoding="utf-8") as val_file:
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

    ################### Pep2Vec ###################
    # TODO pep2vec fails for some alleles, need to check the error
    # split each file to subsets for each unique allele and run pep2vec
    # create tmp directory
    # os.makedirs(os.path.join(output_dir, "tmp"), exist_ok=True)
    # unique_alleles = datasets['train']['allotype'].unique()
    # for allele in unique_alleles:
    #     allele_df = datasets['train'][datasets['train']['allotype'] == allele]
    #     allele_df.to_csv(os.path.join(output_dir, "tmp", f"pep2vec_input_{allele}.csv"), index=False, header=True)
    # # run pep2vec for each allele
    # failed_alleles_file = os.path.join(output_dir, "failed_alleles.txt")
    # for allele in unique_alleles:
    #     try:
    #         input_file = os.path.join(output_dir,"tmp", f"pep2vec_input_{allele}.csv")
    #         output_file = os.path.join(output_dir, f"pep2vec_output_{allele}.parquet")
    #         exit_code = os.system(
    #             f"./Pep2Vec/pep2vec.bin --num_processes 20 --num_threads 10 --dataset {input_file} --output_location {output_file} --mhctype {mhc_type}")
    #         if exit_code != 0:
    #             with open(failed_alleles_file, "a") as f:
    #                 f.write(f"{allele}\n")
    #             print(f"Failed to process allele: {allele}")
    #     except Exception as e:
    #         with open(failed_alleles_file, "a") as f:
    #             f.write(f"{allele} (Error: {str(e)})\n")
    #         print(f"Error processing allele {allele}: {str(e)}")

    # automatically get the number of cores
    num_cores = os.cpu_count()-2
    for i in range(k):
        train_file = os.path.join(os.path.dirname(__file__), "data", dataset_name, "folds", f"train_set_fold_{i}.csv")
        output_file = os.path.join(output_dir, f"pep2vec_output_fold_{i}.parquet")
        os.system(f"./Pep2Vec/pep2vec.bin --num_threads {num_cores}  --dataset {train_file} --output_location {output_file} --mhctype {mhc_type}")
        validation_file = os.path.join(os.path.dirname(__file__), "data", dataset_name, "folds", f"val_set_fold_{i}.csv")
        output_file = os.path.join(output_dir, f"pep2vec_output_val_fold_{i}.parquet")
        os.system(f"./Pep2Vec/pep2vec.bin --num_threads {num_cores}  --dataset {validation_file} --output_location {output_file} --mhctype {mhc_type}")

    # # test set
    # test_file = os.path.join(folds_dir, "test_original.csv")
    # if os.path.exists(test_file):
    #     output_file = os.path.join(output_dir, f"pep2vec_output_test.parquet")
    #     os.system(
    #         f"./Pep2Vec/pep2vec.bin --num_threads {num_cores} --dataset {test_file} --output_location {output_file} --mhctype {mhc_type}")
    # # Process test1 and test2 datasets
    # for test_name in ["test1_stratified", "test2_single_unique_allele"]:
    #     test_file = os.path.join(folds_dir, f"{test_name}.csv")
    #     if os.path.exists(test_file):
    #         output_file = os.path.join(output_dir, f"pep2vec_output_{test_name}.parquet")
    #         os.system(
    #             f"./Pep2Vec/pep2vec.bin --num_threads {num_cores} --dataset {test_file} --output_location {output_file} --mhctype {mhc_type}")
    ###############################################


if __name__ == "__main__":
    # main("Conbotnet", "mhc2", 0.001)
    # add_binding_label(
    #     df_path=os.path.join("data", "Pep2Vec", "Conbotnet"),
    #     train_path=os.path.join("data", "Conbotnet", "train.csv"),
    #     test_path=os.path.join("data", "Conbotnet", "test_all.csv"),
    #     output_dir=os.path.join("data", "Pep2Vec", "Conbotnet_new_subset")
    # )
    # main("ConvNeXT-MHC", "mhc1", 0.001, 5)
    # add_binding_label(
    #     df_path=os.path.join("data", "Pep2Vec", "ConvNeXT-MHC"),
    #     train_path=os.path.join("data", "ConvNeXT-MHC", "train.csv"),
    #     output_dir=os.path.join("data", "Pep2Vec", "ConvNeXT-MHC_new_subset")
    # )
    # main("NetMHCpan_dataset", "mhc2", 0.01, 5)
    # add_binding_label(
    #     df_path=os.path.join("data", "Pep2Vec", "NetMHCIIpan_dataset"),
    #     train_path=os.path.join("data", "NetMHCIIpan_dataset", "cleaned_data.csv"),
    #     output_dir=os.path.join("data", "Pep2Vec", "NetMHCIpan_dataset_new_subset")
    # )
    main("NetMHCpan_dataset", "mhc1", 0.01, 5)
    add_binding_label(
        df_path=os.path.join("data", "Pep2Vec", "NetMHCIpan_dataset"),
        train_path=os.path.join("data", "NetMHCIIpan_dataset", "cleaned_data.csv"),
        output_dir=os.path.join("data", "Pep2Vec", "NetMHCIpan_dataset_new_subset")
    )

