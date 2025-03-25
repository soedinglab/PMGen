import json
import sys

from run_utils import run_and_parse_netmhcpan
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import shutil
from concurrent.futures import ProcessPoolExecutor


"""
Guide to run the script:
1. Run the script with the argument process_data to process the data
    - arg1: process_data arg2: None
2. Run the script with the argument run_netmhcpan to run the NetMHCpan for the processed data
    - arg1: run_netmhcpan arg2: chunk_number (0-9)
3. Run the script with the argument combine_results to combine the results from the NetMHCpan runs
    - arg1: combine_results arg2: None

"""


def load_data(path):
    data = pd.read_csv(path, delim_whitespace=True, header=None)
    return data


def process_data(mhcII_path = "data/NetMHCpan_dataset/NetMHCIIpan_train/",
    tmp_path = "data/NetMHCpan_dataset/tmp/"):
    """
    Load and process MHCII data for NetMHCpan
    This function works for only HLA inputs
    Args:
        mhcII_path:
        tmp_path:

    Returns:

    """

    if not os.path.isdir(tmp_path):
        os.mkdir(tmp_path)

    el_data = pd.DataFrame(columns=["peptide", "assigned_label", "allele", "core"])
    ba_data = pd.DataFrame(columns=["peptide", "assigned_label", "allele", "core"])

    # Check directory exists
    if not os.path.exists(mhcII_path):
        print(f"Directory not found: {mhcII_path}")
        return

    train_files = [f for f in os.listdir(mhcII_path) if "train" in f]
    print(f"Found {len(train_files)} train files")

    for file in tqdm(train_files):
        if "BA" in file or "ba" in file:
            data = load_data(mhcII_path + file)
            print(f"Loaded BA data from {file}: {ba_data.shape}")

            # Rename columns if necessary
            if data.shape[1] >= 4:
                data.columns = ["peptide", "assigned_label", "allele", "core"] + list(range(data.shape[1] - 4))
                # add ba data to the dataframe
                ba_data = pd.concat([ba_data, data], ignore_index=True)
            else:
                print(f"Warning: File {file} has insufficient columns: {data.shape[1]}")

        elif "EL" in file or "el" in file:
            data = load_data(mhcII_path + file)
            print(f"Loaded EL data from {file}: {data.shape}")

            # Rename columns if necessary
            if data.shape[1] >= 4:  # Ensure data has enough columns
                data.columns = ["peptide", "assigned_label", "allele", "core"] + list(range(data.shape[1] - 4))
                # add el data to the dataframe
                el_data = pd.concat([el_data, data], ignore_index=True)
            else:
                print(f"Warning: File {file} has insufficient columns: {data.shape[1]}")
        else:
            # skip the file and print a warning
            print(f"Skipping file: {file}")

    print(f"After loading data: {el_data.shape}")
    if el_data.empty:
        print("No data was loaded. Check file paths and content.")
        return

    # get the cell lines rows, drop the rows that has HLA- in their allele column
    el_data = el_data[el_data["allele"].str.contains("HLA-") == False]
    print(f"After removing HLA- rows: {el_data.shape}")

    ## drop rows with the same allele and peptide (1)
    el_data = el_data.drop_duplicates(subset=["allele", "peptide"], keep="first")
    print(f"After removing duplicates: {el_data.shape}")

    ## get allele names of cell lines
    allelelist = mhcII_path + "allelelist.txt"  # change to allelelist for MHCI

    # Check if allele list file exists
    if not os.path.exists(allelelist):
        print(f"Allele list file not found: {allelelist}")
        return

    allele_map = pd.read_csv(allelelist, delim_whitespace=True, header=None)
    allele_map.columns = ["key", "allele_list"]
    print(f"Loaded allele map with {len(allele_map)} entries")

    # convert the second column to list if , is present, else return a one element list
    allele_map["allele_list"] = allele_map["allele_list"].apply(lambda x: x.split(",") if "," in x else [x])

    # Create a dictionary for more efficient lookup
    allele_dict = dict(zip(allele_map["key"], allele_map["allele_list"]))

    # update the el_data['allele'] dataframe with the allele names
    el_data["allele"] = el_data["allele"].apply(lambda x: allele_dict.get(x, []))
    print(f"After mapping alleles: {el_data.shape}")

    # Check if any allele mappings are empty
    empty_alleles = el_data[el_data["allele"].apply(lambda x: len(x) == 0)].shape[0]
    print(f"Rows with empty allele mappings: {empty_alleles}")

    # add the identifiers
    # el_data['cell_line_id'] = el_data.index
    # print the length of the el_data
    # print(f"Length of el_data: {len(el_data)}")

    # decouple rows with multiple alleles, if the row['allele'] has multiple alleles, then create a new row for each allele
    # el_data = el_data.explode("allele")
    # print(f"After exploding alleles: {el_data.shape}")

    # reset index for unique identifier
    # el_data = el_data.reset_index(drop=True)

    # First add DRA/ before the allele names for alleles with DRB
    el_data["allele"] = el_data["allele"].apply(lambda x: ["HLA-DRA/HLA-" + a if "DRB" in a else a for a in x])

    # all DRB alleles should have HLA-DRA/ in front of them, assert this
    assert el_data[el_data["allele"].apply(lambda x: any("DRB" in a and "HLA-DRA/HLA-" not in a for a in x))].empty

    # if H-2 is present in the allele name, add a mice-DRA/mice- in front of it
    el_data["allele"] = el_data["allele"].apply(lambda x: ["mice-DRA/mice-" + a if "H-2" in a else a for a in x])

    # split double alleles with /
    # if / not in allele name, replace the second - with /
    el_data["allele"] = el_data["allele"].apply(
        lambda x: [a[:a.find("-", a.find("-") + 1)] + "/HLA-" + a[a.find("-", a.find("-") + 1) + 1:] if a.count(
            "-") >= 2 and "/" not in a else a for a in x])

    # First add DRA/ before the allele names for alleles with DRB
    el_data["allele"] = el_data["allele"].apply(lambda x: "HLA-DRA/HLA-" + x if "DRB" in x else x)

    # all DRB alleles should have HLA-DRA/ in front of them, assert this
    assert el_data[el_data["allele"].apply(lambda x: "DRB" in x and "HLA-DRA/HLA-" not in x)].empty

    # if H-2 is present in the allele name, add a mice-DRA/mice- in front of it
    el_data["allele"] = el_data["allele"].apply(lambda x: "mice-DRA/mice-" + x if "H-2" in x else x)

    # split double alleles with /
    # if / not in allele name, replace the second - with /
    el_data["allele"] = el_data["allele"].apply(
        lambda x: x[:x.find("-", x.find("-") + 1)] + "/HLA-" + x[x.find("-", x.find("-") + 1) + 1:] if x.count("-") >= 2 and "/" not in x else x)

    # split to two variables, one with labels = 0 and one with labels = 1
    el_data_0 = el_data[el_data["assigned_label"] == 0]
    el_data_1 = el_data[el_data["assigned_label"] == 1]

    # drop the labels column from el_data_1
    # el_data_1 = el_data_1.drop(columns=["label"])
    print(f"Data with label 0: {el_data_0.shape}")

    # Print some sample data for debugging
    # print(el_data_1.head())

    # drop the rows with the same allele and peptide (2)
    # el_data_1 = el_data_1.drop_duplicates(subset=["allele", "peptide"], keep="first")
    # print(f"After removing duplicates: {el_data_1.shape}")

    # Verify that "/" is present in all allele names
    invalid_alleles = el_data_0[el_data_0["allele"].apply(lambda x: "/" not in x)]
    if not invalid_alleles.empty:
        print("Alleles without '/' separator:")
        print(invalid_alleles)

    # # Get unique alleles and run NetMHCpan for each
    # unique_alleles = el_data_1["allele"].unique()
    # print(f"Processing {len(unique_alleles)} unique alleles")

    # # get unique cell line ids
    # unique_cell_lines = el_data_1["cell_line_id"].unique()
    #
    # # assert that the cell_line_id is unique
    # print(f"Processing {len(unique_cell_lines)} unique cell line ids")

    return el_data_0, el_data_1, ba_data


def run_netmhcpan_(el_data,  true_label ,tmp_path, results_dir, chunk_number):
    # chunk_dataframe = pd.DataFrame(columns=["MHC", "Peptide", "Of", "Core", "Core_Rel", "Inverted", "Identity", "Score_EL", "%Rank_EL", "Exp_Bind", "Score_BA", "%Rank_BA", "Affinity(nM)", "long_mer", "cell_line_id"])
    chunk_df_path = os.path.join(results_dir, f"el{true_label}_chunk_{chunk_number}.csv")
    dropped_rows = pd.DataFrame()
    dropped_rows_path = os.path.join(results_dir, f"el{true_label}_dropped_rows_{chunk_number}.csv")
    for idx, cell_row in tqdm(el_data.iterrows(), total=el_data.shape[0]):
        number_of_alleles = len(eval(cell_row["allele"]))
        result_data = pd.DataFrame(columns=["MHC", "Peptide", "Of", "Core", "Core_Rel", "Inverted", "Identity", "Score_EL", "%Rank_EL", "Exp_Bind", "Score_BA", "%Rank_BA", "Affinity(nM)", "long_mer", "cell_line_id"])
        peptide = cell_row["peptide"]
        for allele in eval(cell_row["allele"]):
            # get the unique id for peptide
            unique_id = f"{peptide}_{allele}"

            # define path
            peptide_fasta_path = os.path.join(tmp_path, f"fasta/el{true_label}_peptide_{unique_id}.fasta")
            peptide_fasta_dir = os.path.dirname(peptide_fasta_path)

            # Ensure the directory exists
            if not os.path.exists(peptide_fasta_dir):
                os.makedirs(peptide_fasta_dir)

            # Write single peptide to fasta file
            with open(peptide_fasta_path, "w") as f:
                f.write(f">peptide\n{peptide}\n")

            # Output directory for this specific cell_line_id
            output_dir = os.path.join(tmp_path, f"output/")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            try:
                # Run NetMHCpan for this specific peptide
                result_ = run_and_parse_netmhcpan(
                    peptide_fasta_file=peptide_fasta_path,
                    mhc_type=2,
                    output_dir=output_dir,
                    mhc_allele=allele,
                    save_csv=False
                )

                # select the top 1 result from netmhcpan output
                if result_ is not None and not result_.empty:
                    result_ = result_.iloc[[0]]
                    result_['long_mer'] = peptide
                    result_['cell_line_id'] = str(chunk_number) + "_" + str(idx)
                    result_['convoluted_label'] = true_label
                    result_data = pd.concat([result_data, result_])

            except Exception as e:
                print(f"Error processing peptide {peptide} with allele {allele}: {str(e)}")
            # Clean up temporary files
            if os.path.exists(peptide_fasta_path):
                os.remove(peptide_fasta_path)

            # Get the actual directory path
            peptide_fasta_dir = os.path.dirname(peptide_fasta_path)
            try:
                shutil.rmtree(peptide_fasta_dir)
            except PermissionError:
                print(f"Warning: Could not remove directory {peptide_fasta_dir} due to permission error")

            try:
                shutil.rmtree(output_dir)
            except PermissionError:
                print(f"Warning: Could not remove directory {output_dir} due to permission error")

        # save the results to disk
        if not result_data.empty:
            # assert len(result_data) == number_of_alleles
            if len(result_data) != number_of_alleles:
                print(f"Warning: Expected {number_of_alleles} results but got {len(result_data)}")
            result_data['assigned_label'] = result_data['Score_BA'].apply(lambda x: 1 if eval(x) > 0.426 else 0)
            if true_label == 1:
                # if at least one label is not 1, select the highest score_BA and set the labels to 1, then save the allele in a list
                # Create a mask for cell lines with at least one positive label
                positive_cell_lines = result_data.groupby('cell_line_id')['assigned_label'].max() == 1
                positive_cell_lines = positive_cell_lines[positive_cell_lines].index.tolist()
                # Filter rows
                mask = result_data['cell_line_id'].isin(positive_cell_lines)
                dropped_df = result_data[~mask].copy()
                result_data = result_data[mask].copy()
                # Save dropped rows to separate file if there are any
                if not dropped_df.empty:
                    dropped_rows = pd.concat([dropped_rows, dropped_df])

            # save the results to the chunk_dataframe directly to the disk using append method
            result_data.to_csv(chunk_df_path, mode="a", header=not os.path.exists(chunk_df_path), index=False)
            dropped_rows.to_csv(dropped_rows_path, index=False)



# def run_netmhcpan(el_data_1, unique_alleles, tmp_path, results_dir):
#     for allele in unique_alleles:
#         # Filter data for the current allele
#         allele_data = el_data_1[el_data_1["allele"] == allele]
#
#         # Create allele-specific filename base
#         allele_filename = allele.replace("/", "_").replace("*", "").replace(":", "")
#
#         # Get peptides
#         peptides = allele_data['peptide'].values
#         print(f"Total peptides for {allele}: {len(peptides)}")
#
#         # Remove duplicate peptides
#         unique_peptides = np.unique(peptides)
#         print(f"Unique peptides for {allele}: {len(unique_peptides)}")
#
#         # Process each peptide individually
#         for peptide_idx, peptide in enumerate(unique_peptides):
#             # Check if this peptide was already processed
#             with open(os.path.join(results_dir, "processed_peptides.txt"), "r") if os.path.exists(
#                     os.path.join(results_dir, "processed_peptides.txt")) else open(os.devnull, "r") as f:
#                 processed = any(f"{allele}\t{peptide}\n" == line for line in f)
#                 if processed:
#                     continue
#
#             # get the cell_line_id
#             cell_line_id = allele_data[allele_data["peptide"] == peptide]["cell_line_id"].iloc[0]
#
#             # Use a timestamp to ensure unique filenames across processes
#             timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
#             unique_id = f"{peptide_idx}_{timestamp}"
#
#             # Create fasta file with single peptide
#             peptide_fasta_path = os.path.join(tmp_path, f"results/el0_peptide_{allele_filename}_{unique_id}.fasta")
#
#             # Write single peptide to fasta file
#             with open(peptide_fasta_path, "w") as f:
#                 f.write(f">peptide\n{peptide}\n")
#
#             # Output directory for this specific peptide
#             output_dir = os.path.join(tmp_path, f"results/el0_output_{allele_filename}_{unique_id}")
#
#             try:
#                 # Run NetMHCpan for this specific peptide
#                 allele_result = run_and_parse_netmhcpan(
#                     peptide_fasta_file=peptide_fasta_path,
#                     mhc_type=2,
#                     output_dir=output_dir,
#                     mhc_allele=allele
#                 )
#
#                 # Save results to disk immediately
#                 if allele_result is not None:
#                     # take the top 1 result
#                     allele_result = allele_result.head(1)  # taking the first row, because the output is sorted by BF score and we take the best score
#                     result_path = os.path.join(results_dir, f"{allele_filename}.parquet")
#                     # add long_mer
#                     allele_result['long_mer'] = peptide
#                     allele_result['cell_line_id'] = cell_line_id
#                     new_table = pa.Table.from_pandas(allele_result)
#                     if os.path.exists(result_path):
#                         existing_table = pq.read_table(result_path)
#                         combined_table = pa.concat_tables([existing_table, new_table])
#                         pq.write_table(combined_table, result_path)
#                     else:
#                         pq.write_table(new_table, result_path)
#
#                     with open(os.path.join(results_dir, "processed_peptides.txt"), "a") as processed_file:
#                         processed_file.write(f"{allele}\t{peptide}\n")
#
#             except Exception as e:
#                 print(f"Error processing peptide {peptide} with allele {allele}: {str(e)}")
#             finally:
#                 # Clean up temporary files immediately
#                 try:
#                     if os.path.exists(peptide_fasta_path):
#                         os.remove(peptide_fasta_path)
#                     if os.path.exists(output_dir) and os.path.isdir(output_dir):
#                         import shutil
#                         shutil.rmtree(output_dir)
#                 except Exception as e:
#                     print(f"Failed to remove temporary files: {str(e)}")


# def combine_results_(results_dir, final_output_path):
#
#     for cell_line_id:
#
#         pass
#     # combine all csv files
#     all_csv_files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith('.csv')]
#     dropped_rows = pd.DataFrame()
#     if all_csv_files:
#         print(f"Combining {len(all_csv_files)} csv files into final output")
#         df = None
#         for csv_file in all_csv_files:
#             try:
#                 df = pd.read_csv(csv_file)
#             except Exception as e:
#                 print(f"Error reading {csv_file}: {str(e)}")
#                 continue
#         if df:
#             # Standardize column names
#             if 'Peptide' in df.columns and 'peptide' not in df.columns:
#                 df['peptide'] = df['Peptide']
#             elif 'Peptide' in df.columns and 'peptide' in df.columns:
#                 df['peptide'] = df['peptide'].fillna(df['Peptide'])
#             if 'MHC' in df.columns and 'allele' not in df.columns:
#                 df['allele'] = df['MHC']
#             elif 'MHC' in df.columns and 'allele' in df.columns:
#                 df['allele'] = df['allele'].fillna(df['MHC'])
#             # Calculate label from Score_EL if label is empty or doesn't exist
#             if 'label' not in df.columns:
#                 if 'Score_BA' in df.columns:
#                     df['label'] = df['Score_BA'].apply(lambda x: 1 if eval(x) > 0.426 else 0)
#
#             # if at least one label is not 1, select the highest score_BA and set the labels to 1, then save the allele in a list
#             if 'cell_line_id' in df.columns and 'label' in df.columns:
#                 # Create a mask for cell lines with at least one positive label
#                 positive_cell_lines = df.groupby('cell_line_id')['label'].max() == 1
#                 positive_cell_lines = positive_cell_lines[positive_cell_lines].index.tolist()
#                 # Filter rows
#                 mask = df['cell_line_id'].isin(positive_cell_lines)
#                 dropped_df = df[~mask].copy()
#                 df = df[mask].copy()
#                 # Save dropped rows to separate file if there are any
#                 if not dropped_df.empty:
#                     dropped_rows = pd.concat([dropped_rows, dropped_df])
#
#     # print the length of the dropped rows
#     print(f"Length of dropped rows: {len(dropped_rows)}")

# def combine_results(results_dir, final_output_path):
#         # Combine all parquet files at the end
#         try:
#             all_parquet_files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith('.parquet')]
#             if all_parquet_files:
#                 print(f"Combining {len(all_parquet_files)} parquet files into final output")
#                 # Read all parquet files as pandas DataFrames
#                 dfs = []
#                 for parquet_file in all_parquet_files:
#                     try:
#                         df = pq.read_table(parquet_file).to_pandas()
#                         dfs.append(df)
#                     except Exception as e:
#                         print(f"Error reading {parquet_file}: {str(e)}")
#                         continue
#                 if dfs:
#                     # Concatenate DataFrames (pandas handles schema differences automatically)
#                     df = pd.concat(dfs, ignore_index=True)
#                     # Standardize column names
#                     if 'Peptide' in df.columns and 'peptide' not in df.columns:
#                         df['peptide'] = df['Peptide']
#                     elif 'Peptide' in df.columns and 'peptide' in df.columns:
#                         df['peptide'] = df['peptide'].fillna(df['Peptide'])
#                     if 'MHC' in df.columns and 'allele' not in df.columns:
#                         df['allele'] = df['MHC']
#                     elif 'MHC' in df.columns and 'allele' in df.columns:
#                         df['allele'] = df['allele'].fillna(df['MHC'])
#                     # Calculate label from Score_EL if label is empty or doesn't exist
#                     if 'label' not in df.columns:
#                         if 'Score_BA' in df.columns:
#                             df['label'] = df['Score_BA'].apply(lambda x: 1 if eval(x) > 0.4 else 0)  # Threshold can be adjusted
#                         else:
#                             df['label'] = None
#                     # Handle cell_line_id filtering
#                     if 'cell_line_id' in df.columns and 'label' in df.columns:
#                         # Create a mask for cell lines with at least one positive label
#                         positive_cell_lines = df.groupby('cell_line_id')['label'].max() == 1
#                         positive_cell_lines = positive_cell_lines[positive_cell_lines].index.tolist()
#                         # Filter rows
#                         mask = df['cell_line_id'].isin(positive_cell_lines)
#                         dropped_df = df[~mask].copy()
#                         df = df[mask].copy()
#                         # Save dropped rows to separate file if there are any
#                         if not dropped_df.empty:
#                             dropped_output_path = final_output_path.replace('.parquet', '_dropped.parquet')
#                             pq.write_table(pa.Table.from_pandas(dropped_df), dropped_output_path)
#                             print(
#                                 f"Dropped {dropped_df.shape[0]} rows from {len(set(dropped_df['cell_line_id']))} cell lines with no positive labels")
#                             print(f"Dropped rows saved to: {dropped_output_path}")
#                             print(f"Kept {df.shape[0]} rows from {len(set(df['cell_line_id']))} cell lines with at least one positive label")
#                     # Keep only required columns if they exist
#                     keep_cols = ["allele", "peptide", "label"]
#                     existing_cols = [col for col in keep_cols if col in df.columns]
#                     if existing_cols:
#                         df = df[existing_cols]
#                     # Save the combined DataFrame to parquet
#                     pq.write_table(pa.Table.from_pandas(df), final_output_path)
#                     print(f"Combined results shape: {df.shape}")
#                     print(f"Final output saved to: {final_output_path}")
#                 else:
#                     df = pd.DataFrame(columns=["allele", "peptide", "label"])
#                     print("No valid DataFrames were created from parquet files")
#             else:
#                 df = pd.DataFrame(columns=["allele", "peptide", "label"])
#                 print("No parquet files were generated")
#         except Exception as e:
#             print(f"Error combining parquet files: {str(e)}")
#             import traceback
#             traceback.print_exc()
#             df = pd.DataFrame(columns=["allele", "peptide", "label"])
#         return df


# def parallelize_netmhcpan(el_data, tmp_path, results_dir):
#     run_netmhcpan_(el_data, 1, tmp_path, results_dir, 0)
#
#     # with ProcessPoolExecutor() as executor:
#     #     tasks = []
#     #     for allele in unique_cell_line:
#     #         subset_data = el_data_1[el_data_1["allele"] == allele]
#     #         tasks.append(executor.submit(run_netmhcpan_, subset_data, [allele], tmp_path, results_dir))
#     #     for task in tasks:
#     #         task.result()

def run_(arg1, arg2):
    mhcII_path = "data/NetMHCpan_dataset/NetMHCIIpan_train/"
    tmp_path = "data/NetMHCpan_dataset/tmp/"
    results_dir = "data/NetMHCpan_dataset/results/"
    final_output_path = "data/NetMHCpan_dataset/combined_results.parquet"

    # make tmp directory
    if not os.path.isdir(tmp_path):
        os.mkdir(tmp_path)

    # make results directory
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    if arg1 == "process_data":
        ########################
        # Load data
        el_data_0, el_data_1, ba_data = process_data(
            mhcII_path=mhcII_path,
            tmp_path=tmp_path)

        # save the variables
        el_data_0.to_csv("data/NetMHCpan_dataset/tmp/el_data_0.csv", index=False)
        el_data_1.to_csv("data/NetMHCpan_dataset/tmp/el_data_1.csv", index=False)
        ba_data.to_csv("data/NetMHCpan_dataset/tmp/ba_data.csv", index=False)
        params = {
            # "unique_cell_lines": unique_cell_lines.tolist() if hasattr(unique_cell_lines, "tolist") else unique_cell_lines,
            "tmp_path": tmp_path,
            "results_dir": results_dir
        }
        with open("data/NetMHCpan_dataset/tmp/params.json", "w") as f:
            json.dump(params, f)

        # split el_data to 10 chunks and save separately
        chunk_size = len(el_data_1) // 256
        for i in range(256):
            el_data_1_chunk = el_data_1.iloc[i * chunk_size: (i + 1) * chunk_size]
            el_data_1_chunk.to_csv(f"data/NetMHCpan_dataset/tmp/el_data_1_chunk_{i}.csv", index=False)
        ########################

    if arg1 == "run_netmhcpan":
        # load the variables and parameters from JSON files
        with open("data/NetMHCpan_dataset/tmp/params.json", "r") as f:
            params = json.load(f)
        # unique_alleles = np.array(params["unique_alleles"])
        tmp_path = params["tmp_path"]
        results_dir = params["results_dir"]

        # load the variables
        el_data_1 = pd.read_csv(
            f"data/NetMHCpan_dataset/tmp/el_data_1_chunk_{arg2}.csv")
        el_data_0 = pd.read_csv("data/NetMHCpan_dataset/tmp/el_data_0.csv")

        # print the len of el_data_1
        print(f"Length of el_data_1: {len(el_data_1)}")

        # subset 1000 random samples from the data for testing
        # el_data_1 = el_data_1.sample(n=1000, random_state=42)
        # unique_cell_lines = el_data_1["cell_line_id"].unique()

        # Run NetMHCpan in parallel
        # parallelize_netmhcpan(el_data_1, tmp_path, results_dir)

        # run the netmhcpan for the el_data_1
        run_netmhcpan_(el_data_1, 1, tmp_path, results_dir, arg2)

        # run the netmhcpan for the el_data_0
        # run_netmhcpan_(el_data_0, 0, tmp_path, results_dir, arg2)

    # if arg1 == "combine_results":
    #     # Combine results
    #     df = combine_results_(results_dir, final_output_path)
    #
    #     el_data_0 = pd.read_csv("data/NetMHCpan_dataset/tmp/el_data_0.csv")
    #     ba_data = pd.read_csv("data/NetMHCpan_dataset/tmp/ba_data.csv")
    #
    #     # concatenate the dataframes df and el_data_0 and ba_data
    #     df = pd.concat([df, el_data_0], ignore_index=True)
    #     df = pd.concat([df, ba_data], ignore_index=True)
    #
    #     # save the final output
    #     df.to_csv("data/NetMHCpan_dataset/NetMHCIIpan_all.csv", index=False)

def main():
    if len(sys.argv) < 1:
        print("Usage: python script1.py <process_data/run_netmhcpan/combine_results> <chunk_number> (only required when running run_netmhcpan)")
        sys.exit(1)

    if sys.argv[1] not in ["process_data", "run_netmhcpan", "combine_results"]:
        print("Invalid argument. Please specify one of the following: process_data, run_netmhcpan, combine_results")

    if sys.argv[1] == "run_netmhcpan":
        run_(sys.argv[1], sys.argv[2])

    else:
        run_(sys.argv[1], None)

if __name__ == "__main__":
    main()