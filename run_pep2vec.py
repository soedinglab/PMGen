# load Conbotnet subset data
import os
import sys
import pandas as pd
import numpy as np
from utils.processing_functions import create_k_fold_leave_one_out_stratified_cross_validation
from run_pMHC_DL import train_and_evaluate_scqvae

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

def main(dataset_name="Conbotnet", mhc_type="mhc2"):
    # Conbotnet dataset paths
    # ConBotNet only contains mhc class II

    # Setup paths
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")
    conbotnet_dir = os.path.join(data_dir, dataset_name)
    folds_dir = os.path.join(conbotnet_dir, "folds")
    output_dir = os.path.join(data_dir, "Pep2Vec", dataset_name)

    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(folds_dir, exist_ok=True)

    # Define datasets
    paths = {
        'train': os.path.join(conbotnet_dir, "train.csv"),
        'test': os.path.join(conbotnet_dir, "test_all.csv")
    }

    # Column configuration
    columns = ["allele", "long_mer", "binding_label"]
    rename_map = {"allele": "allotype", "long_mer": "peptide"}

    # Load and process datasets
    datasets = {}
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

    # Create k-fold cross-validation splits
    k = 5
    folds = create_k_fold_leave_one_out_stratified_cross_validation(
        datasets['train'], k=k, target_col="binding_label",
        id_col="allotype", train_size=0.8, subset_prop=0.1
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
    datasets['test'].to_csv(os.path.join(folds_dir, "test_set.csv"), index=False, header=True)
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

    for i in range(k):
        train_file = os.path.join(os.path.dirname(__file__), "data", dataset_name, "folds", f"train_set_fold_{i}.csv")
        output_file = os.path.join(output_dir, f"pep2vec_output_fold_{i}.parquet")
        os.system(f"./Pep2Vec/pep2vec.bin --num_processes 20 --num_threads 10  --dataset {train_file} --output_location {output_file} --mhctype {mhc_type}")
        validation_file = os.path.join(os.path.dirname(__file__), "data", dataset_name, "folds", f"val_set_fold_{i}.csv")
        output_file = os.path.join(output_dir, f"pep2vec_output_val_fold_{i}.parquet")
        os.system(f"./Pep2Vec/pep2vec.bin --num_processes 20 --num_threads 10  --dataset {validation_file} --output_location {output_file} --mhctype {mhc_type}")
    # test set
    test_file = os.path.join(os.path.dirname(__file__), "data", dataset_name, "folds", "test_set.csv")
    output_file = os.path.join(output_dir, f"pep2vec_output_test_fold_{i}.parquet")
    os.system(f"./Pep2Vec/pep2vec.bin --num_processes 20 --num_threads 10  --dataset {test_file} --output_location {output_file} --mhctype {mhc_type}")
    ###############################################


if __name__ == "__main__":
    main("Conbotnet", "mhc2")
    main("ConvNeXT-MHC", "mhc1")
