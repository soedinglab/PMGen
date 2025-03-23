import json
import sys

from run_utils import run_and_parse_netmhcpan
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
import multiprocessing
from functools import partial
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

    el_data = pd.DataFrame(columns=["peptide", "label", "allele", "core"])
    ba_data = pd.DataFrame(columns=["peptide", "label", "allele", "core"])

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
                data.columns = ["peptide", "label", "allele", "core"] + list(range(data.shape[1] - 4))
                # add ba data to the dataframe
                ba_data = pd.concat([ba_data, data], ignore_index=True)
            else:
                print(f"Warning: File {file} has insufficient columns: {data.shape[1]}")

        elif "EL" in file or "el" in file:
            data = load_data(mhcII_path + file)
            print(f"Loaded EL data from {file}: {data.shape}")

            # Rename columns if necessary
            if data.shape[1] >= 4:  # Ensure data has enough columns
                data.columns = ["peptide", "label", "allele", "core"] + list(range(data.shape[1] - 4))
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
    el_data['cell_line_id'] = el_data.index

    # decouple rows with multiple alleles, if the row['allele'] has multiple alleles, then create a new row for each allele
    el_data = el_data.explode("allele")
    print(f"After exploding alleles: {el_data.shape}")

    # reset index for unique identifier
    el_data = el_data.reset_index(drop=True)

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
    el_data_0 = el_data[el_data["label"] == 0]
    el_data_1 = el_data[el_data["label"] == 1]

    # drop the labels column from el_data_1
    # el_data_1 = el_data_1.drop(columns=["label"])
    print(f"Data with label 0: {el_data_0.shape}")

    # Print some sample data for debugging
    print(el_data_1.head())

    # drop the rows with the same allele and peptide (2)
    el_data_1 = el_data_1.drop_duplicates(subset=["allele", "peptide"], keep="first")
    print(f"After removing duplicates: {el_data_1.shape}")

    # Verify that "/" is present in all allele names
    invalid_alleles = el_data_0[el_data_0["allele"].apply(lambda x: "/" not in x)]
    if not invalid_alleles.empty:
        print("Alleles without '/' separator:")
        print(invalid_alleles)

    # Get unique alleles and run NetMHCpan for each
    unique_alleles = el_data_0["allele"].unique()
    print(f"Processing {len(unique_alleles)} unique alleles")

    return el_data_0, el_data_1, unique_alleles, ba_data


def run_netmhcpan(el_data_1, unique_alleles, tmp_path, results_dir):
    for allele in unique_alleles:
        # Filter data for the current allele
        allele_data = el_data_1[el_data_1["allele"] == allele]

        # Create allele-specific filename base
        allele_filename = allele.replace("/", "_").replace("*", "").replace(":", "")

        # Get peptides
        peptides = allele_data['peptide'].values
        print(f"Total peptides for {allele}: {len(peptides)}")

        # Remove duplicate peptides
        unique_peptides = np.unique(peptides)
        print(f"Unique peptides for {allele}: {len(unique_peptides)}")

        # Process each peptide individually
        for peptide_idx, peptide in enumerate(unique_peptides):
            # Check if this peptide was already processed
            with open(os.path.join(results_dir, "processed_peptides.txt"), "r") if os.path.exists(
                    os.path.join(results_dir, "processed_peptides.txt")) else open(os.devnull, "r") as f:
                processed = any(f"{allele}\t{peptide}\n" == line for line in f)
                if processed:
                    continue

            # get the cell_line_id
            cell_line_id = allele_data[allele_data["peptide"] == peptide]["cell_line_id"].iloc[0]

            # Use a timestamp to ensure unique filenames across processes
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            unique_id = f"{peptide_idx}_{timestamp}"

            # Create fasta file with single peptide
            peptide_fasta_path = os.path.join(tmp_path, f"results/el0_peptide_{allele_filename}_{unique_id}.fasta")

            # Write single peptide to fasta file
            with open(peptide_fasta_path, "w") as f:
                f.write(f">peptide\n{peptide}\n")

            # Output directory for this specific peptide
            output_dir = os.path.join(tmp_path, f"results/el0_output_{allele_filename}_{unique_id}")

            try:
                # Run NetMHCpan for this specific peptide
                allele_result = run_and_parse_netmhcpan(
                    peptide_fasta_file=peptide_fasta_path,
                    mhc_type=2,
                    output_dir=output_dir,
                    mhc_allele=allele
                )

                # Save results to disk immediately
                if allele_result is not None:
                    # take the top 1 result
                    allele_result = allele_result.head(1)  # taking the first row, because the output is sorted by BF score and we take the best score
                    result_path = os.path.join(results_dir, f"{allele_filename}.parquet")
                    new_table = pa.Table.from_pandas(allele_result)
                    # add long_mer
                    new_table['long_mer'] = peptide
                    new_table['cell_line_id'] = cell_line_id
                    if os.path.exists(result_path):
                        existing_table = pq.read_table(result_path)
                        combined_table = pa.concat_tables([existing_table, new_table])
                        pq.write_table(combined_table, result_path)
                    else:
                        pq.write_table(new_table, result_path)

                    with open(os.path.join(results_dir, "processed_peptides.txt"), "a") as processed_file:
                        processed_file.write(f"{allele}\t{peptide}\n")

            except Exception as e:
                print(f"Error processing peptide {peptide} with allele {allele}: {str(e)}")
            finally:
                # Clean up temporary files immediately
                try:
                    if os.path.exists(peptide_fasta_path):
                        os.remove(peptide_fasta_path)
                    if os.path.exists(output_dir) and os.path.isdir(output_dir):
                        import shutil
                        shutil.rmtree(output_dir)
                except Exception as e:
                    print(f"Failed to remove temporary files: {str(e)}")


def combine_results(results_dir, final_output_path):
    # Combine all parquet files at the end
    try:
        all_parquet_files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith('.parquet')]
        if all_parquet_files:
            print(f"Combining {len(all_parquet_files)} parquet files into final output")
            # Read and combine all parquet files
            tables = []
            for parquet_file in all_parquet_files:
                table = pq.read_table(parquet_file)
                tables.append(table)

            if tables:
                combined_table = pa.concat_tables(tables)
                pq.write_table(combined_table, final_output_path)
                df = combined_table.to_pandas()

                # Standardize column names
                if 'Peptide' in df.columns and 'peptide' not in df.columns:
                    df['peptide'] = df['Peptide']
                elif 'Peptide' in df.columns and 'peptide' in df.columns:
                    df['peptide'] = df['peptide'].fillna(df['Peptide'])

                if 'MHC' in df.columns and 'allele' not in df.columns:
                    df['allele'] = df['MHC']
                elif 'MHC' in df.columns and 'allele' in df.columns:
                    df['allele'] = df['allele'].fillna(df['MHC'])

                # Calculate label from Score_EL if label is empty or doesn't exist
                if 'label' not in df.columns:
                    if 'Score_EL' in df.columns:
                        df['label'] = df['Score_EL'].apply(lambda x: 1 if x > 0.426 else 0)  # Threshold can be adjusted
                    else:
                        df['label'] = None

                # for each unique df["cell_line_id"], if at least one row has label == 1 continue, else drop them and save them in a seperate file
                if 'cell_line_id' in df.columns and 'label' in df.columns:
                    # Create a mask for cell lines with at least one positive label
                    positive_cell_lines = df.groupby('cell_line_id')['label'].max() == 1
                    positive_cell_lines = positive_cell_lines[positive_cell_lines].index.tolist()

                    # Filter rows
                    mask = df['cell_line_id'].isin(positive_cell_lines)
                    dropped_df = df[~mask].copy()
                    df = df[mask].copy()

                    # Save dropped rows to separate file if there are any
                    if not dropped_df.empty:
                        dropped_output_path = final_output_path.replace('.parquet', '_dropped.parquet')
                        pq.write_table(pa.Table.from_pandas(dropped_df), dropped_output_path)
                        print(
                            f"Dropped {dropped_df.shape[0]} rows from {len(set(dropped_df['cell_line_id']))} cell lines with no positive labels")
                        print(f"Dropped rows saved to: {dropped_output_path}")

                # Keep only required columns
                keep_cols = ["allele", "peptide", "label"]
                existing_cols = [col for col in keep_cols if col in df.columns]
                df = df[existing_cols]

                print(f"Combined results shape: {df.shape}")
                print(f"Final output saved to: {final_output_path}")
            else:
                df = pd.DataFrame(columns=["allele", "peptide", "label"])
                print("No results were found in parquet files")
        else:
            df = pd.DataFrame(columns=["allele", "peptide", "label"])
            print("No parquet files were generated")
    except Exception as e:
        print(f"Error combining parquet files: {str(e)}")
        df = pd.DataFrame(columns=["allele", "peptide", "label"])

    return df


def parallelize_netmhcpan(el_data_1, unique_alleles, tmp_path, results_dir):
    with ProcessPoolExecutor() as executor:
        tasks = []
        for allele in unique_alleles:
            subset_data = el_data_1[el_data_1["allele"] == allele]
            tasks.append(executor.submit(run_netmhcpan, subset_data, [allele], tmp_path, results_dir))
        for task in tasks:
            task.result()

def run_(arg1, arg2):
    mhcII_path = "data/NetMHCpan_dataset/NetMHCIIpan_train/"
    tmp_path = "data/NetMHCpan_dataset/tmp/"
    results_dir = "data/NetMHCpan_dataset/tmp/results/"
    final_output_path = "data/NetMHCpan_dataset/tmp/combined_results.parquet"

    # make results directory
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    if arg1 == "process_data":
        ########################
        # Load data
        el_data_0, el_data_1, unique_alleles, ba_data = process_data(
            mhcII_path=mhcII_path,
            tmp_path=tmp_path)

        # save the variables
        el_data_0.to_csv("data/NetMHCpan_dataset/tmp/el_data_0.csv", index=False)
        el_data_1.to_csv("data/NetMHCpan_dataset/tmp/el_data_1.csv", index=False)
        ba_data.to_csv("data/NetMHCpan_dataset/tmp/ba_data.csv", index=False)
        params = {
            "unique_alleles": unique_alleles.tolist() if hasattr(unique_alleles, "tolist") else unique_alleles,
            "tmp_path": tmp_path,
            "results_dir": results_dir
        }
        with open("data/NetMHCpan_dataset/tmp/params.json", "w") as f:
            json.dump(params, f)

        # split el_data to 10 chunks and save seperately
        chunk_size = len(el_data_1) // 10
        for i in range(10):
            el_data_1_chunk = el_data_1.iloc[i * chunk_size: (i + 1) * chunk_size]
            el_data_1_chunk.to_csv(f"data/NetMHCpan_dataset/tmp/el_data_1_chunk_{i}.csv", index=False)
        ########################

    if arg1 == "run_netmhcpan":
        # load the variables and parameters from JSON files
        with open("data/NetMHCpan_dataset/tmp/params.json", "r") as f:
            params = json.load(f)
        unique_alleles = np.array(params["unique_alleles"])
        tmp_path = params["tmp_path"]
        results_dir = params["results_dir"]

        # load the variables

        el_data_1 = pd.read_csv(
            f"data/NetMHCpan_dataset/tmp/el_data_1_chunk_{arg2}.csv")

        # print the len of el_data_1
        print(f"Length of el_data_1: {len(el_data_1)}")

        # subset 1000 random samples from the data for testing
        # el_data_1 = el_data_1.sample(n=100, random_state=42)

        # Run NetMHCpan in parallel
        parallelize_netmhcpan(el_data_1, unique_alleles, tmp_path, results_dir)

    if arg1 == "combine_results":
        # Combine results
        df = combine_results(results_dir, final_output_path)

        el_data_0 = pd.read_csv("data/NetMHCpan_dataset/tmp/el_data_0.csv")
        ba_data = pd.read_csv("data/NetMHCpan_dataset/tmp/ba_data.csv")

        # concatenate the dataframes df and el_data_0 and ba_data
        df = pd.concat([df, el_data_0], ignore_index=True)
        df = pd.concat([df, ba_data], ignore_index=True)

        # save the final output
        df.to_csv("data/NetMHCpan_dataset/tmp/NetMHCIIpan_all.csv", index=False)

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