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

def main():
    # Conbotnet dataset paths
    # ConBotNet only contains mhc class II
    train_dataset_path = os.path.join(os.path.dirname(__file__), "data", "Conbotnet", "train.csv")
    test_dataset_path = os.path.join(os.path.dirname(__file__), "data", "Conbotnet", "test_all.csv")
    output_dir = os.path.join(os.path.dirname(__file__), "data", "Pep2Vec", "Conbotnet")
    os.makedirs(output_dir, exist_ok=True)

    # Load datasets
    train_df = load_data(train_dataset_path)
    test_df = load_data(test_dataset_path)
    print("Train dataset shape:", train_df.shape)
    print("Test dataset shape:", test_df.shape)
    # Select relevant columns
    train_df = select_columns(train_df, ["allele", "long_mer", "binding_label"])
    test_df = select_columns(test_df, ["allele", "long_mer", "binding_label"])
    print("Train dataset shape after column selection:", train_df.shape)
    print("Test dataset shape after column selection:", test_df.shape)

    # Rename columns
    train_df = train_df.rename(columns={"allele": "allotype", "long_mer": "peptide", "binding_label": "binding_label"})
    test_df = test_df.rename(columns={"allele": "allotype", "long_mer": "peptide", "binding_label": "binding_label"})

    # Remove duplicates
    train_df = train_df.drop_duplicates(subset=["allotype", "peptide"])
    test_df = test_df.drop_duplicates(subset=["allotype", "peptide"])

    # Remove rows with NaN values
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    print("Train dataset shape after removing duplicates and NaN values:", train_df.shape)
    print("Test dataset shape after removing duplicates and NaN values:", test_df.shape)

    # Create k-fold cross-validation splits
    k = 5
    folds = create_k_fold_leave_one_out_stratified_cross_validation(
        train_df, k=k, target_col="binding_label", id_col="allotype", train_size=0.8, subset_prop=0.1
    )
    # Unpack the folds to get train and validation sets
    train_dfs = [fold[0] for fold in folds]
    val_dfs = [fold[1] for fold in folds]
    held_out_id = [fold[2] for fold in folds]

    print("Number of training sets:", len(train_dfs))
    print("Number of validation sets:", len(val_dfs))
    # TODO Fix later, why header is not saved after the first fold
    # mkdir folds
    os.makedirs(os.path.join(os.path.dirname(__file__), "data", "Conbotnet", "folds"), exist_ok=True)
    if os.path.exists(os.path.join(os.path.dirname(__file__), "data", "Conbotnet", "folds", "held_out_ids.txt")):
        os.remove(os.path.join(os.path.dirname(__file__), "data", "Conbotnet", "folds", "held_out_ids.txt"))

    for i, (train_set, val_set, held_out_id) in enumerate(zip(train_dfs, val_dfs, held_out_id)):
        train_set.to_csv(os.path.join(os.path.dirname(__file__), "data", "Conbotnet", "folds", f"train_set_fold_{i}.csv"), index=True, header=True)
        val_set.to_csv(os.path.join(os.path.dirname(__file__), "data", "Conbotnet", "folds", f"val_set_fold_{i}.csv"), index=True, header=True)
        with open(os.path.join(os.path.dirname(__file__), "data", "Conbotnet", "folds", f"held_out_ids.txt"), "a") as f:
            f.write(f"Fold {i}: {held_out_id}\n")

    # Save the test set
    test_df.to_csv(os.path.join(os.path.dirname(__file__), "data", "Conbotnet","folds", "test_set.csv"), index=True, header=True)
    print("Test dataset saved.")

    ################### Pep2Vec ###################
    # # prepare data for pep2vec
    # for i in range(k):
    #     train_set = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "Conbotnet", "folds", f"train_set_fold_{i}.csv"))
    #     val_set = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "Conbotnet", "folds", f"val_set_fold_{i}.csv"))
    #     train_set = train_set.drop(columns=["binding_label"])
    #     val_set = val_set.drop(columns=["binding_label"])
    #
    #     # save the train and validation sets
    #     train_set.to_csv(os.path.join(os.path.dirname(__file__), "data", "Conbotnet", "folds", f"train_set_fold_{i}.csv"), index=True, header=True)
    #     val_set.to_csv(os.path.join(os.path.dirname(__file__), "data", "Conbotnet", "folds", f"val_set_fold_{i}.csv"), index=True, header=True)

    # TODO write for loop for all folds
    # run Pep2Vec for fold 0 training set
    os.system(f"./Pep2Vec/pep2vec.bin --dataset {os.path.join(os.path.dirname(__file__), 'data', 'Conbotnet', 'folds', 'train_set_fold_0.csv')} "
              f"--output_location {os.path.join(output_dir, 'pep2vec_output_fold_0.parquet')} "
              f"--mhctype mhc2")
    ###############################################

    # run SCQ_Autoencoder
    # Load the parquet file generated by Pep2Vec
    parquet_path = os.path.join(output_dir, f'pep2vec_output_fold_0.parquet')
    print(f"Loading Pep2Vec embeddings from: {parquet_path}")
    pep2vec_data = pd.read_parquet(parquet_path)
    print(f"Pep2Vec data shape: {pep2vec_data.shape}")

    train_and_evaluate_scqvae(parquet_path)

if __name__ == "__main__":
    main()