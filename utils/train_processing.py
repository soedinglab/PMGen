import pandas as pd
import os


def process_dataframe(df):
    # Step 1: Sort IDs with 2 or 5 parts
    filtered_df = df[df['targetid'].apply(lambda x: len(x.split('_')) in [2, 4, 5])]

    # Step 2: Extract first and second parts of 5-part IDs and first part of 2-part IDs
    filtered_df['Modified_ID'] = filtered_df['targetid'].apply(
        lambda x: '_'.join(x.split('_')[:2]) if len(x.split('_')) >= 4 else x.split('_')[0])

    # Step 3: Remove duplicate rows based on the modified IDs
    final_df = filtered_df.drop_duplicates(subset='Modified_ID')

    return final_df


def prepare_dataframe(df, output_dir = "prepared_dataframe.csv"):
    SEQ = []
    ID = []
    for index, row in df.iterrows():
        LEN = len(row["targetid"].split("_"))
        if LEN==2:
            seq_chain_1 = row["target_chainseq"].split("/")[0]
            seq_chain_2 = 0
            id_1 = row["Modified_ID"]
            id_2 = 0
        elif LEN==5:
            seq_chain_1 = row["target_chainseq"].split("/")[0]
            seq_chain_2 = row["target_chainseq"].split("/")[1]
            id_1 = "DRA"
            id_2 = row["Modified_ID"]
        elif LEN==4:
            seq_chain_1 = row["target_chainseq"].split("/")[0]
            seq_chain_2 = row["target_chainseq"].split("/")[1]
            id_1 = row["targetid"].split("-")[1]
            id_2 = row["targetid"].split("-")[2]
        if seq_chain_2 != 0:
            SEQ.append(seq_chain_2)
            ID.append(id_2)
        SEQ.append(seq_chain_1)
        ID.append(id_1)
        print(SEQ)
        print(ID)

    DF = pd.DataFrame({"ID":ID, "SEQ" : SEQ})
    DF.to_csv(f"{output_dir}", sep="\t", index=False)
    return DF

def prepare_preprocess_dataframe(df, output_dir):
    df = process_dataframe(df)
    df_2 = prepare_dataframe(df, output_dir)
    return df, df_2

df1 = pd.read_csv(
    "/scratch/users/a.hajialiasgarynaj01/Master_thesis/Pipline_test_PANDORA_AFoptimization/AFopt/datasets_alphafold_finetune/pmhc_finetune/combo_1and2_valid.tsv",
    sep="\t")
df2 = pd.read_csv(
    "/scratch/users/a.hajialiasgarynaj01/Master_thesis/Pipline_test_PANDORA_AFoptimization/AFopt/datasets_alphafold_finetune/pmhc_finetune/combo_1and2_train.tsv",
    sep="\t")
df = pd.concat([df1, df2], ignore_index=True)
df, df_2 = prepare_preprocess_dataframe(df, output_dir="prepared_dataframe.csv")
df.to_csv(
    "/scratch/users/a.hajialiasgarynaj01/Master_thesis/Pipline_test_PANDORA_AFoptimization/processing/reference_seqs_chopped.csv",
    sep="\t", index=False)

