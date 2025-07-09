import pandas as pd
import os
import re
import numpy as np
import subprocess
from Bio.PDB import PDBParser, PPBuilder, PDBIO
from Bio import pairwise2, PDB, SeqIO
from Bio.Align import substitution_matrices
from scipy.spatial.distance import cdist
import shutil
import random
import sys
import Levenshtein
from typing import List
import warnings
# Suppress the specific warning
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from user_setting import netmhcipan_path, netmhciipan_path, pmgen_abs_dir

def process_dataframe(df):
    # Step 1: Sort IDs with 2 or 5 parts
    filtered_df = df[df['targetid'].apply(lambda x: len(x.split('_')) in [2, 4, 5])]

    # Step 2: Extract first and second parts of 5-part IDs and first part of 2-part IDs
    filtered_df['Modified_ID'] = filtered_df['targetid'].apply(
        lambda x: '_'.join(x.split('_')[:2]) if len(x.split('_')) >= 4 else x.split('_')[0])

    # Step 3: Remove duplicate rows based on the modified IDs
    final_df = filtered_df.drop_duplicates(subset='Modified_ID')

    return final_df


def prepare_dataframe(df, output_dir="prepared_dataframe.csv"):
    SEQ = []
    ID = []
    for index, row in df.iterrows():
        LEN = len(row["targetid"].split("_"))
        if LEN == 2:
            seq_chain_1 = row["target_chainseq"].split("/")[0]
            seq_chain_2 = 0
            id_1 = row["Modified_ID"]
            id_2 = 0
        elif LEN == 5:
            seq_chain_1 = row["target_chainseq"].split("/")[0]
            seq_chain_2 = row["target_chainseq"].split("/")[1]
            id_1 = "DRA"
            id_2 = row["Modified_ID"]
        elif LEN == 4:
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

    DF = pd.DataFrame({"ID": ID, "SEQ": SEQ})
    DF.to_csv(f"{output_dir}", sep="\t", index=False)
    return DF


def prepare_preprocess_dataframe(df, output_dir):
    df = process_dataframe(df)
    df_2 = prepare_dataframe(df, output_dir)
    return df, df_2


# Function to modify the sequence IDs
def modify_sequence_id(sequence_id, MHC_2=True):
    # Remove ":" and "*" from the sequence ID
    sequence_id = sequence_id.split(" ")[1]
    modified_id = "HLA-" + sequence_id.replace("*", "")
    if MHC_2:
        modified_id = modified_id.replace(":", "")
    return modified_id


def BA_raw_processing(input_df, Allele_type, num_queries,
                      Allele_col=2, BA_col=1, peptide_col=0, pos_portion=0.5, random_state=42):
    """
    Gets BA raw data (like from netMHCpan) and preprocess that
    :param input_df: BA input dataframe
    :param Allele_type: list for MHC-I values HLA-(A,B,C,E,F,G) for MHC-II values HLA-(DQ,DR,DA,DO)
    :param: num_queries: int, number of queries you want to get
    :param Allele_col: the column index corresponds to allele
    :param BA_col: correspond column BA
    :param peptide_col: set peptide column
    :param: pos_portion: between 0 and one, the portion of binder (positive) queries among total
    :return: preprocessed DF
    """
    df = input_df.iloc[:, [peptide_col, BA_col, Allele_col]]
    df.columns = ["peptide", "BA", "Allele"]
    exclude_values = ['H-2', 'Mamu', "BoLA", "Patr", "SLA", "Gogo"]
    df = df[~df['Allele'].str.contains('|'.join(exclude_values))]
    df = df[df["Allele"].str.contains("|", Allele_type)]
    print(len(df))
    Label = [0 if v < 0.42 else 1 for v in df["BA"].tolist()]
    
    df["Label"] = Label
    print("num label 1: ",len(df[df["Label"]==1]))
    print("num label 0: ",len(df[df["Label"]==0]))
    if pos_portion != None:
        pos = int(pos_portion * num_queries)
        neg = num_queries - pos
        df_pos = df[df["Label"] == 1].sample(n=pos, random_state=random_state, replace=False)
        df_neg = df[df["Label"] == 0].sample(n=neg, random_state=random_state, replace=False)
        df = pd.concat([df_pos, df_neg], ignore_index=True)
    return df




def HLA_annotation(df, MHC_type):
    hla_list = list(df["Allele"])
    peptide_list = list(df["peptide"])
    BA_list = list(df["BA"])
    label_list = list(df["Label"])
    if MHC_type == 1:
        source = "data/processed_mhc_1.fasta"
        fasta = SeqIO.parse(source, "fasta")
        mhc_1_dict = {"name": [], "sequence": [], "id": [], "BA":[], "peptide":[], "Label":[]}
        not_found=[]

        for hla,pep,ba, label, rrrr in zip(hla_list,peptide_list,BA_list, label_list,range(len(hla_list))):
            print(f"index:{rrrr} allele: {hla}")
            probs = {}

            # Initialize the iterator only once before the loop
            fasta = SeqIO.parse(source, "fasta")

            for record in fasta:
                if str(record.description) == hla:
                    probs = {"name": str(record.description), "sequence": str(record.seq)}
                    mhc_1_dict["name"].append(probs["name"])
                    mhc_1_dict["sequence"].append(probs["sequence"])
                    mhc_1_dict["id"].append(hla)
                    mhc_1_dict["BA"].append(ba)
                    mhc_1_dict["peptide"].append(pep)
                    mhc_1_dict["Label"].append(label)
                    break

            if len(probs) != 2 and not len(probs) > 2:
                # Use the iterator initialized before the loop
                fasta = SeqIO.parse(source, "fasta")
                for record in fasta:
                    if hla in str(record.description):
                        probs = {"name": str(record.description), "sequence": str(record.seq)}
                        mhc_1_dict["name"].append(probs["name"])
                        mhc_1_dict["sequence"].append(probs["sequence"])
                        mhc_1_dict["id"].append(hla)
                        mhc_1_dict["BA"].append(ba)
                        mhc_1_dict["peptide"].append(pep)
                        mhc_1_dict["Label"].append(label)
                        break
            if len(probs) == 0:
                not_found.append(hla)

    if MHC_type == 2:
        source = "data/processed_mhc_2.fasta"
        fasta = SeqIO.parse(source, "fasta")
        mhc_1_dict = {"name": [], "sequence": [], "id": [], "BA":[], "peptide":[], "Label":[]}
        not_found = []

        for hla,pep,ba,label, rrrr in zip(hla_list,peptide_list,BA_list,label_list, range(len(hla_list))):
            # Check if "DRB" is in hla
            print(f"index:{rrrr} allele: {hla}")
            if "DRB" in hla:
                hla_1 = "HLA-" + hla.replace("_", "")
                probs_1 = {}
                probs_2 = {"name": "DRA01",
                           "sequence": "HVIIQAEFYLNPDQSGEFMFDFDGDEIFHVDMAKKETVWRLEEFGRFASFEAQGALANIAVDKANLEIMTKRSN"}

                # Search for exact matches first
                fasta = SeqIO.parse(source, "fasta")
                for record in fasta:
                    if str(record.description) == hla_1 and len(probs_1) != 2:
                        probs_1 = {"name": f"{str(record.description)}/{probs_2['name']}",
                                   "sequence": f"{str(record.seq)}/{probs_2['sequence']}"}
                        mhc_1_dict["name"].append(f"{str(record.description)}/{probs_2['name']}")
                        mhc_1_dict["sequence"].append(f"{str(record.seq)}/{probs_2['sequence']}")
                        mhc_1_dict["id"].append(hla)
                        mhc_1_dict["BA"].append(ba)
                        mhc_1_dict["peptide"].append(pep)
                        mhc_1_dict["Label"].append(label)
                        print(record.description)
                        break

                # If no exact match found, search for similar descriptions
                if len(probs_1) < 2:
                    fasta = SeqIO.parse(source, "fasta")
                    for record in fasta:
                        if hla_1 in str(record.description):
                            probs_1 = {"name": f"{str(record.description)}/{probs_2['name']}",
                                       "sequence": f"{str(record.seq)}/{probs_2['sequence']}"}
                            mhc_1_dict["name"].append(f"{str(record.description)}/{probs_2['name']}")
                            mhc_1_dict["sequence"].append(f"{str(record.seq)}/{probs_2['sequence']}")
                            mhc_1_dict["id"].append(hla)
                            mhc_1_dict["BA"].append(ba)
                            mhc_1_dict["peptide"].append(pep)
                            mhc_1_dict["Label"].append(label)
                            break

            else:
                hla_1 = "HLA-" + hla.split("-")[1]
                hla_2 = "HLA-" + hla.split("-")[2]
                probs_1, probs_2 = {}, {}

                # Search for exact matches first
                fasta = SeqIO.parse(source, "fasta")
                for record in fasta:
                    if str(record.description) == hla_1 and len(probs_1) < 2:
                        probs_1 = {"name": str(record.description), "sequence": str(record.seq)}

                    if str(record.description) == hla_2 and len(probs_2) < 2:
                        probs_2 = {"name": str(record.description), "sequence": str(record.seq)}

                    if len(probs_2) == 2 and len(probs_1) == 2:
                        mhc_1_dict["name"].append(f"{probs_1['name']}/{probs_2['name']}")
                        mhc_1_dict["sequence"].append(f"{probs_1['sequence']}/{probs_2['sequence']}")
                        mhc_1_dict["id"].append(hla)
                        mhc_1_dict["BA"].append(ba)
                        mhc_1_dict["peptide"].append(pep)
                        mhc_1_dict["Label"].append(label)
                        break

                # If no exact match found, search for similar descriptions
                if (len(probs_1) != 2 or len(probs_2) != 2) and not (len(probs_1) > 2 or len(probs_2) > 2):
                    fasta = SeqIO.parse(source, "fasta")
                    for record in fasta:
                        if len(probs_1) != 2 and hla_1 in str(record.description):
                            probs_1 = {"name": str(record.description), "sequence": str(record.seq)}

                        if len(probs_2) != 2 and hla_2 in str(record.description):
                            probs_2 = {"name": str(record.description), "sequence": str(record.seq)}

                        if len(probs_2) == 2 and len(probs_1) == 2:
                            mhc_1_dict["name"].append(f"{probs_1['name']}/{probs_2['name']}")
                            mhc_1_dict["sequence"].append(f"{probs_1['sequence']}/{probs_2['sequence']}")
                            mhc_1_dict["id"].append(hla)
                            mhc_1_dict["BA"].append(ba)
                            mhc_1_dict["peptide"].append(pep)
                            mhc_1_dict["Label"].append(label)
                            break

                if len(probs_1) == 0:
                    print('###NOT added')
                    not_found.append(hla)

    return mhc_1_dict, pd.DataFrame({"notfound_hlas": not_found})


def merge_mhc_data(mhc_dict, your_dataframe, id_column="Allele"):
    """
    Merge MHC data from mhc_1_dict into your_dataframe based on the specified ID column.
    Parameters:
    - mhc_1_dict (dict): Dictionary containing MHC data with keys 'name', 'sequence', and 'id'.
    - your_dataframe (pd.DataFrame): Pandas DataFrame containing the original data.
    - id_column (str): Name of the column in your_dataframe to use as the ID for merging.
    Returns:
    - pd.DataFrame: Merged DataFrame containing the original data and additional MHC columns.
    """
    # Create a DataFrame from mhc_1_dict
    mhc_df = pd.DataFrame(mhc_dict)
    # Assuming "id" in mhc_1_dict corresponds to id_column in your_dataframe
    subsetted_dataframe = your_dataframe[your_dataframe[id_column].isin(mhc_df['id'])]

    # Merge the subsetted DataFrame with mhc_df on the id_column
    merged_dataframe = pd.merge(subsetted_dataframe, mhc_df, how='left', left_on=id_column, right_on='id')
    # Drop the redundant "id_y" column (which is the same as "id_x")
    return merged_dataframe
    
    
   
def filter_fasta_by_length(input_fasta, output_fasta, length_threshold):
    """
    Filter a FASTA file based on sequence length.

    Parameters:
    - input_fasta (str): Path to the input FASTA file.
    - output_fasta (str): Path to the output FASTA file after filtering.
    - length_threshold (int): Minimum length threshold for sequences to be included.

    Returns:
    None
    """
    with open(output_fasta, 'w') as output_handle:
        for record in SeqIO.parse(input_fasta, 'fasta'):
            if len(record.seq) >= length_threshold:
                SeqIO.write(record, output_handle, 'fasta')
                
    
    
    
    
def get_aminoacid_sequences(pdb_file, exclude_residues=None):
    """
    Get the amino acid sequences from a PDB file, excluding specified residues.

    Parameters:
    - pdb_file (str): Path to the PDB file.
    - exclude_residues (list or None): List of residue names to exclude from the sequences. If None, no exclusions.

    Returns:
    - tuple: (Concatenated amino acid sequence from all chains, Dictionary mapping chain identifiers to their amino acid sequences.)
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    # Initialize an empty string to store the concatenated amino acid sequence
    concatenated_sequence = ''
    # Initialize a dictionary to store sequences for each chain
    chain_sequences = {}
    # Iterate through the structure to extract the sequences
    for model in structure:
        for chain in model:
            chain_id = chain.id
            chain_sequence = ''
            for residue in chain:

                # Check if the residue is an amino acid and not in the exclusion list
                if PDB.is_aa(residue) and (exclude_residues is None or residue.get_resname() not in exclude_residues):
                    try:
                        chain_sequence += PDB.Polypeptide.three_to_one(residue.get_resname())
                    except KeyError:
                        chain_sequence += 'C'
            # Update the concatenated sequence
            concatenated_sequence += chain_sequence
            # Update the dictionary with chain sequences
            chain_sequences[chain_id] = chain_sequence
            # Add separator between chains
            concatenated_sequence += '/'
    # Remove the trailing separator
    concatenated_sequence = concatenated_sequence.rstrip('/')
    return concatenated_sequence, chain_sequences
    
    
    
def process_string(input_string):
    """
    Processes a given string by extracting numbers and removing both numbers and spaces.
    Parameters:
    - input_string (str): The input string to be processed.
    Returns:
    - numbers_variable (list): A list containing the extracted numbers as integers.
    - final_result (str): The input string without both numbers and spaces.
    """
    # Find all numbers in the string using regular expression
    numbers = re.findall(r'\d+', input_string)
    # Extracted numbers as a new variable
    numbers_variable = list(map(int, numbers))
    # Remove numbers from the original string
    string_without_numbers = re.sub(r'\d+', '', input_string)
    # Remove spaces from the string without numbers
    final_result = string_without_numbers.replace(' ', '')

    return numbers_variable, final_result

def map_query_template(target_sequence, template_sequence, tar_dist=0, temp_dist=0):
    blosum62 = substitution_matrices.load("BLOSUM62")
    #alignments = pairwise2.align.localds(target_sequence, template_sequence, MatrixInfo.blosum62, -11, -1)
    alignments = pairwise2.align.localds(target_sequence, template_sequence, blosum62, -11, -1)
    alignment = alignments[0]
    list_alignment = list((pairwise2.format_alignment(*alignment)).split("\n"))
    start_target,_ = process_string(list_alignment[0])
    if len(start_target)==0: start_target=0
    else: start_target=int(start_target[0]) - 1
    start_template,_ = process_string(list_alignment[2])
    if len(start_template)==0: start_template=0
    else: start_template=int(start_template[0]) - 1
    signs = str(list_alignment[1]).replace(" ","")
    L = []
    for i,s in enumerate(signs):
        if s=="|":
            L.append(f"{i+start_target+tar_dist}:{i+start_template+temp_dist}")
    final = ""
    for l in L:
        if len(final)==0:
            final = l
        else:
            final = final + ";" + l

    return final

def prepare_alignment_file_without_peptide(template_id, mhc_seq, mhc_type, output_path, template_path,
                           template_csv_path="../data/templates/all_templates.csv"):
    """
    this function generates the alignment file needed as the input for AFfine
    template id: template id in *.pdb format
    mhc_seq: string mhc sequence, for mhc_2 each chain seperated by "/"
    mhc_type: int 1 or 2
    output_path: str where the alignment file to be saved
    template_path: str template fild pdb path
    template_csv_path(optional): str template csv database
    """
    template_csv = pd.read_csv(template_csv_path, sep="\t", header=0)
    template_df = template_csv[template_csv["pdb_file_name"]==template_id]

    if mhc_type==1:
        template_seq = str(template_df["concatenated_sequence"].tolist()[0]).split("/")[0]
        full_template_seq = str(template_df["concatenated_sequence"].tolist()[0]).replace("/","")
        df = pd.DataFrame({"template_pdbfile" : [template_path],
            "target_to_template_alignstring" : [map_query_template(target_sequence=mhc_seq, template_sequence=template_seq)],
            "identities":[len(map_query_template(target_sequence=mhc_seq, template_sequence=template_seq).split(";"))],
            "target_len" : [len(mhc_seq)],
            "template_len" : [len(full_template_seq)]})

    if mhc_type==2:
        template_seq_1 = str(template_df["concatenated_sequence"].tolist()[0]).split("/")[0]
        template_seq_2 = str(template_df["concatenated_sequence"].tolist()[0]).split("/")[1]
        full_template_seq = str(template_df["concatenated_sequence"].tolist()[0]).replace("/", "")
        mhc_seq_1 = mhc_seq.split("/")[0]
        mhc_seq_2 = mhc_seq.split("/")[1]
        align_seq_1 = map_query_template(target_sequence=mhc_seq_1, template_sequence=template_seq_1)
        align_seq_2 = map_query_template(target_sequence=mhc_seq_2, template_sequence=template_seq_2,
                                   temp_dist=len(template_seq_1), tar_dist=len(mhc_seq_1))
        align_seq = align_seq_1 + ";" + align_seq_2
        align_seq = align_seq.rstrip(";")
        df  = pd.DataFrame({"template_pdbfile":[template_path],
                            "target_to_template_alignstring":[align_seq],
                            "identities":[len(align_seq.split(";"))],
                            "target_len":[len(mhc_seq.replace("/",""))],
                            "template_len":[len(full_template_seq)]})
    if output_path:
        df.to_csv(output_path,sep="\t", index=False)
    return df

def prepare_alignment_file_with_peptide(pdb_file, target_seq, mhc_type,
                                        output_path, peptide=True):
    """
    This function generates the alignment files needed for peptide-MHC alphafold predictions
    :param pdb_file: str, path to template pdb files
    :param target_seq: str, target sequence, each chain and peptide separated by "/"
    :param mhc_type: int, 1 or 2
    :param output_path: str, the output file and directory path
    :peptide output_path: bool, if there is peptide in mhc, default is Flase
    :return: p-mhc dataframe and output csv
    """
    concat_seqs , _ = get_aminoacid_sequences(pdb_file)

    if mhc_type==1:
        #template part
        template_mhc_seq = concat_seqs.split("/")[0]
        full_template_seq = concat_seqs.replace("/","")
        peptide_seq_align = ""
        if peptide:
            tmpl_peptide_seq = concat_seqs.split("/")[2]
            peptide_seq_align = map_query_template(target_sequence=tmpl_peptide_seq, template_sequence=tmpl_peptide_seq,
                                                   tar_dist=(len(target_seq.replace("/",""))-len(tmpl_peptide_seq)),
                                                   temp_dist=len(concat_seqs.replace("/",""))-len(tmpl_peptide_seq))
        #Alignemnt part
        mhc_seq_align = map_query_template(target_sequence=target_seq.split("/")[0],
                                           template_sequence=template_mhc_seq)
        alignment = mhc_seq_align + ";" + peptide_seq_align
        alignment = alignment.rstrip(";")
        df = pd.DataFrame({"template_pdbfile" : [pdb_file],
            "target_to_template_alignstring" : [alignment],
            "identities":[len(alignment.split(";"))],
            "target_len" : [len(target_seq.replace("/",""))],
            "template_len" : [len(concat_seqs.replace("/",""))]})

    if mhc_type==2:
        template_mhc_seq_1 = concat_seqs.split("/")[0]
        template_mhc_seq_2 = concat_seqs.split("/")[1]
        peptide_seq_align = ""
        if peptide:
            tmpl_peptide_seq = concat_seqs.split("/")[2]
            peptide_seq_align = map_query_template(tmpl_peptide_seq,tmpl_peptide_seq,
                                                   len(target_seq.replace("/",""))-len(tmpl_peptide_seq),
                                                   len(template_mhc_seq_1)+len(template_mhc_seq_2))
        mhc_seq_align_1 = map_query_template(target_seq.split("/")[0], template_mhc_seq_1)
        mhc_seq_align_2 = map_query_template(target_seq.split("/")[1], template_mhc_seq_2,
                                             len(target_seq.split("/")[0]),
                                             len(template_mhc_seq_1))
        alignment = mhc_seq_align_1 + ";" + mhc_seq_align_2 + ";" + peptide_seq_align
        alignment = alignment.rstrip(";")
        df = pd.DataFrame({"template_pdbfile" : [pdb_file],
            "target_to_template_alignstring" : [alignment],
            "identities":[len(alignment.split(";"))],
            "target_len" : [len(target_seq.replace("/",""))],
            "template_len" : [len(concat_seqs.replace("/",""))]})
    if output_path:
        df.to_csv(output_path, sep="\t", index=False)
    return df
    
    
    


def netmhccsv_to_pandora_csv(netmhcpan_df,plus_index=0):
    mhcpan_ids = netmhcpan_df.iloc[:,0]
    mhc_type = netmhcpan_df.iloc[:,6]
    sequences = netmhcpan_df["sequence"]
    peptides = netmhcpan_df["peptide"]
    #for mhcpan data
    new_ids = []
    #for pandora data
    PEPTIDES = []
    CHOPPED_IDS = []
    M_chain = []
    N_chain = []
    MHC_class = []
    for id,type,seq,pep,index in zip(mhcpan_ids,mhc_type,sequences,peptides, range(len(mhcpan_ids))):
        if type==1:
            if len(id)>10:
                id = id[:10]
            CHOPPED_IDS.append(id)
            M_chain.append(seq)
            N_chain.append("None")
            MHC_class.append("I")

        if type==2:
            if "DRB" in id.split("/")[0]:
                id = "HLA-" + id.split("/")[1] + "/" + id.split("/")[0]
                CHOPPED_IDS.append(id.replace("/", ";"))
                M_chain.append(seq.split("/")[1])
                N_chain.append(seq.split("/")[0])
                MHC_class.append("II")
            else:
                CHOPPED_IDS.append(id.replace("/",";"))
                M_chain.append(seq.split("/")[0])
                N_chain.append(seq.split("/")[1])
                MHC_class.append("II")

        PEPTIDES.append(pep)
        id = re.sub(r"[^a-zA-Z0-9]", "_", id) + f"_{index + plus_index}"
        new_ids.append(id)

    netmhcpan_df["new_ids"] = new_ids
    df = pd.DataFrame({"new_ids":new_ids,
                       "peptide":PEPTIDES,
                       "IDs":CHOPPED_IDS,
                       "M_chain":M_chain,
                       "N_chain":N_chain,
                       "MHC_class":MHC_class})
    return df, netmhcpan_df


def correct_netmhcpan(netmhcpan_df):
    SEQ = []
    NAME = []
    for seq,name,type in zip(netmhcpan_df.sequence, netmhcpan_df.name, netmhcpan_df.type):
        if type==2 and "DRB" in name.split("/")[0]:
            sequence = seq.split("/")[1] + "/" + seq.split("/")[0]
            N = name.split("/")[1] + "/" + name.split("/")[0]

        else:
            sequence = seq
            N = name
        SEQ.append(sequence)
        NAME.append(N)
    netmhcpan_df["sequence"] = SEQ
    netmhcpan_df["name"] = NAME
    return netmhcpan_df




def split_and_generate_pdb(source_pdb, output_pdb, cut_num):
    """
    This function is used for mhc_2 and peptide-mhc_1 structures. Gets the output structure
    of AFfine in one chain and backs the two chain structure
    :param source_pdb:
    :param output_pdb:
    :param cut_num:
    :param second_cut_num:
    :return:
    """
    # Load the source PDB file
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('source_structure', source_pdb)

    # Create a new structure for both chains
    new_structure = PDB.Structure.Structure('combined_structure')

    # Create two separate models for the chains
    model_a = PDB.Model.Model('ID')


    # Create new chains for model_a and model_b
    chain_a = PDB.Chain.Chain('A')
    chain_b = PDB.Chain.Chain('P')

    # Iterate through the structure and split the chain
    for model in structure:
        for chain in model:
            # Extract the first 100 amino acids as chain_A
            for residue in chain:
                if residue.id[1] <= cut_num:
                    chain_a.add(residue.copy())

            # Extract the next 80 amino acids (101-180) as chain_B
            for residue in chain:
                if cut_num < residue.id[1] <= (len(chain) + 200):
                    chain_b.add(residue.copy())

    # Add the chains to the models
    model_a.add(chain_a)
    model_a.add(chain_b)

    # Add the models to the new structure
    new_structure.add(model_a)
    # Save the combined structure to the output PDB file
    with open(output_pdb, 'w') as out_pdb_file:
        # Write the structure without MODEL and ENDMDL lines
        pdb_io = PDB.PDBIO()
        pdb_io.set_structure(new_structure)
        pdb_io.save(out_pdb_file, write_end=False)








def split_and_generate_pdb_peptide_mhc2(source_pdb, output_pdb, cut_nums):
    """
    This function is used for mhc_2 and peptide-mhc_1 structures. Gets the output structure
    of AFfine in one chain and backs the two chain structure
    :param source_pdb:
    :param output_pdb:
    :param cut_nums: list of two cut nums
    :return:
    """
    # Load the source PDB file
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('source_structure', source_pdb)

    # Create a new structure for both chains
    new_structure = PDB.Structure.Structure('combined_structure')

    # Create two separate models for the chains
    model_a = PDB.Model.Model('ID')


    # Create new chains for model_a and model_b
    chain_a = PDB.Chain.Chain('A')
    chain_b = PDB.Chain.Chain('B')
    chain_p = PDB.Chain.Chain("P")

    # Iterate through the structure and split the chain
    for model in structure:
        for chain in model:
            # Extract the first 100 amino acids as chain_A
            for residue in chain:
                if residue.id[1] <= cut_nums[0]:
                    chain_a.add(residue.copy())

            # Extract the next 80 amino acids (101-180) as chain_B
            for residue in chain:
                if cut_nums[0] < residue.id[1] <= (200 + cut_nums[0] + cut_nums[1]):
                    chain_b.add(residue.copy())

            for residue in chain:
                if (200 + cut_nums[0] + cut_nums[1]) < residue.id[1] <= (400 + len(chain)):
                    chain_p.add(residue.copy())

    # Add the chains to the models
    model_a.add(chain_a)
    model_a.add(chain_b)
    model_a.add(chain_p)

    # Add the models to the new structure
    new_structure.add(model_a)
    # Save the combined structure to the output PDB file
    with open(output_pdb, 'w') as out_pdb_file:
        # Write the structure without MODEL and ENDMDL lines
        pdb_io = PDB.PDBIO()
        pdb_io.set_structure(new_structure)
        pdb_io.save(out_pdb_file, write_end=False)


def split_and_renumber_pdb(input_pdb, output_dir, n=100, mhc_type=None):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", input_pdb)

    # Get the first model
    model = structure[0]

    # Process residues
    chains = []
    current_chain = []
    prev_res_id = None

    for chain in model:
        for res in chain.get_residues():
            if res.get_id()[0] != ' ':  # Skip heteroatoms
                continue
            res_id = res.get_id()[1]
            if prev_res_id is not None and res_id - prev_res_id >= n:
                chains.append(current_chain)
                current_chain = []
            current_chain.append(res.copy())  # Copy residue to avoid modifying original structure
            prev_res_id = res_id

    if current_chain:
        chains.append(current_chain)

    # Renumber and assign new chains
    chain_ids = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if mhc_type:
        if mhc_type==1: chain_ids='AP'
        elif mhc_type==2: chain_ids='ABP'
    else:
        if len(chains) == 2: chain_ids='AP'
        elif len(chains) == 3: chain_ids='ABP'

    new_structure = PDB.Structure.Structure("new_protein")
    new_model = PDB.Model.Model(0)
    new_structure.add(new_model)
    for chain_idx, chain_residues in enumerate(chains):
        new_chain = PDB.Chain.Chain(chain_ids[chain_idx % len(chain_ids)])
        new_residue_index = 1  # Reset numbering for each new chain
        for res in chain_residues:
            res.id = (' ', new_residue_index, ' ')
            new_chain.add(res)
            new_residue_index += 1
        new_model.add(new_chain)
    # Save new PDB
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_pdb = os.path.join(output_dir, f'multichain_{os.path.basename(input_pdb)}')
    io = PDB.PDBIO()
    io.set_structure(new_structure)
    io.save(output_pdb)
    return output_pdb


'''def get_coords_from_res(residue, atom='CB'):
    if atom in residue:
        return residue[atom].coord

    else:
        if residue.get_resname() != 'GLY':
            print(f'not {atom} , {residue.get_resname()} used, instead used another.')
        if 'CA' in residue:
            return residue['CA'].coord
        if 'C' in residue:
            return residue['C'].coord
        if 'O' in residue:
            return residue['O'].coord
        if 'N' in residue:
            return residue['N'].coord
    print('no atom for residue found, skiping it with unreasonable coords')
    return np.array([1e+9, 1e+9, 1e+9])'''


def get_coords_from_res(residue):
    """Compute the average coordinates of side-chain atoms.
    If side-chain atoms are missing, fall back to backbone atoms."""
    # Define backbone atoms to exclude
    backbone_atoms = {'CA', 'N', 'C', 'O'}
    # Extract side-chain atom coordinates
    side_chain_coords = [
        atom.coord for atom in residue.get_atoms() if atom.get_name() not in backbone_atoms
    ]
    if side_chain_coords:
        return np.mean(side_chain_coords, axis=0)  # Average side-chain coordinates
    # Fallback mechanism if no side-chain atoms exist (e.g., Glycine)
    if residue.get_resname() != 'GLY':
        print(f"No side-chain atoms for {residue.get_resname()}, using backbone.")
    # Try accessing backbone atoms in a priority order
    for atom_name in ['CA', 'C', 'O', 'N']:
        try:
            return residue[atom_name].coord
        except KeyError:
            continue  # If atom is missing, try the next one
    return np.array([1e+9, 1e+9, 1e+9])



def get_distance_matrices(input_pdb, target_chain, atom='CB'):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", input_pdb)
    target_residues = structure[0][target_chain].get_residues()
    target_coordinates = [get_coords_from_res(residue) for residue in target_residues]
    target_coordinates = np.array(target_coordinates)  # (N, 3)

    other_chains = [chain for chain in structure.get_chains() if chain.id != target_chain]
    final_dict = {}
    for chain in other_chains:
        chain_residues = list(chain.get_residues())  # Convert generator to list
        if chain_residues:  # Ensure the list is not empty
            chain_id = chain_residues[0].get_parent().id
            coordinates = np.array([get_coords_from_res(residue) for residue in chain_residues])
            dist_matrix = cdist(target_coordinates, coordinates)
            #dist_masked = np.where(dist_matrix <= thr, 1., 0.)
            final_dict[chain_id] = dist_matrix  # Dict of coordinates (N_peptide, N_chain)
    return final_dict # {'A':matrix, or 'A' and 'B' dist matrices}

def get_hotspots(distance_matrix_dict, thr=6.0):
    final_dict = {}
    for key, dist_matrix in distance_matrix_dict.items():
        dist_masked = np.where(dist_matrix <= thr, 1., 0.)
        positions = np.argwhere(dist_masked == 1.)
        final_dict[key] = positions
    return final_dict

def extract_hotspot_sequence(hotspot_dict, input_pdb):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", input_pdb)
    for chain, pairs in hotspot_dict.items():
        target_residues = structure[0][chain].get_residues()


def correct_residue_indexes(input_pdb, output_pdb):
    parser = PDBParser()
    structure = parser.get_structure('input', input_pdb)

    # Initialize the new residue index
    new_residue_index = 1

    # Iterate over the structure and correct the residue indexes
    for model in structure:
        for chain in model:
            for residue in chain:
                residue.id = (' ', new_residue_index, ' ')
                new_residue_index += 1

    # Write the corrected structure to the output file
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)




def remove_rows_from_file(file_path, df):
    # Check if the file exists
    if not os.path.isfile(file_path):
        print(f"File {file_path} does not exist.")
        return [[]]

    # Read the IDs from the file
    with open(file_path, 'r') as f:
        ids_to_remove = [line.strip() for line in f]

    # Check if the file is empty
    if not ids_to_remove:
        print(f"File {file_path} is empty.")
        return [[]]

    # Create a boolean mask to identify rows to keep
    mask = ~df['new_ids'].isin(ids_to_remove)

    # Remove rows where the mask is False
    new_df = df[mask].reset_index(drop=True)

    return [new_df]

def rename_files(directory, num_template, copy_if_less_template=False):
    # Iterate through files in the specified directory
    i = 0
    for filename in os.listdir(directory):
        if filename.endswith(".pdb") and "BL" in filename:
            # Get the full path of the file
            file_path = os.path.join(directory, filename)

            # Create a new name based on your requirements
            new_filename = f"mod{i}.pdb"
            i = i + 1

            # Rename the file
            new_file_path = os.path.join(directory, new_filename)
            shutil.move(file_path, new_file_path)
    modified_files = os.listdir(directory)
    modified_files = [os.path.join(directory, i) for i in modified_files if '.pdb' in i and 'mod' in i]
    print(f'Only {len(modified_files)} PANDORA templates were generated')
    if copy_if_less_template: # if less templates are created, copies randomly
        print(f'copy_if_less_template==True, So, Duplicate Randomly')

        if len(modified_files) < num_template and len(modified_files) != 0:
            J = num_template - len(modified_files)
            for jj in range(J):
                source_file = os.path.join(directory, random.choice(modified_files))
                dest_file = os.path.join(directory, f'mod{num_template - 1 - jj}.pdb')
                shutil.copy(source_file, dest_file)
    return modified_files




def parse_netmhcpan_file(file_path):
    # Read the entire file
    with open(file_path, 'r') as f:
        content = f.read()
    # Split into sections using dashed lines (flexible length)
    sections = re.split(r'-{50,}', content)
    tables = []
    itis_header = False
    for section in sections:
        lines = section.strip().split('\n')
        if not lines:
            continue
        # Look for header starting with "Pos"
        data = []
        for i, line in enumerate(lines):
            if itis_header: # if previous one was header, no add data until reach end of table
                if re.match(r'^\s*\d+\s+', line):  # Collect data rows (lines starting with a number)
                    data.append(re.split(r'\s+', line.strip())[:len(columns)])
            if line.strip().startswith("Pos"):
                header_line = line
                itis_header = True
                # Extract column names from header (split on 2+ spaces) and remove "BinderLevel" in the end which is empty usually
                columns = re.split(r'\s+', header_line.strip())
                columns = columns[:-1] if columns[-1] == 'BindLevel' else columns
                break
        if itis_header and len(data) != 0: # if header found and data is loaded, write the table
            df = pd.DataFrame(data=data, columns=columns)
            try:# most likely works for mhc1
                df = df.astype({'Score_EL':'float', 'Aff(nM)':'float'})
                df = df.sort_values(["Aff(nM)", "Score_EL"], ascending=[True, False])
            except:# for mhc2
                df = df.astype({'Score_EL':'float', 'Affinity(nM)':'float'})
                df = df.sort_values(["Affinity(nM)", "Score_EL"], ascending=[True, False])
            itis_header = False # after data, again refresh the header and search for next table
            tables.append(df)
    return pd.concat(tables)
    

def find_similar_strings(a: str, file_path: str, num_matches=100):
    # Read the file and store each line as a vocab
    assert a.split('-')[0] in ['HLA', 'DLA', 'SLA', 'Mamu', 'Patr', 'Bola', 'BOLA', 'BoLA', 'MICA', 'MICB', 'mice'], (f''
                                f'Input allele not found, Please provide full alleles, e.g "HLA-D*A", "SLA-*"'
                                f'\n Found {a} ')
    allele = a.split('-')[0]
    not_allele = [i for i in ['HLA', 'DLA', 'SLA', 'Mamu', 'Patr', 'BoLA', 'BOLA', 'MICA', 'MICB', 'mice'] if i!= allele]
    complex_alleles = ['DQA', 'DQB', 'DPA', 'DPB'] # need to be splited further e.g. HLA-DQA01-DQB01
    patterns = [r'D.B', r'D.A', r'-A\d{1,2}', r'-B\d{1,2}', r'-C\d{1,2}', r'-\d{1,2}', r'DPA\d{1,2}', r'DPB\d{1,2}',
                r'DQA\d{1,2}', r'DQB\d{1,2}', r'DRB\d{1,2}']
    pattern = []
    for p in patterns:
        match = re.search(p, a)
        if match: pattern.append(match.group(0))
    vocabs = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                v = line.strip()
                if any(x in v for x in complex_alleles):
                    splitted = v.split('-')
                    if len(splitted) == 3: #[HLA, DPA01, DPB01]
                        vocabs.append('HLA' + '-' + splitted[1])
                        vocabs.append('HLA' + '-' + splitted[2])
                else: vocabs.append(v)
    similarity_scores = [(vocab.split(' ')[0], Levenshtein.ratio(a, vocab.split(' ')[0])) for vocab in vocabs]
    if len(similarity_scores) == 0:
        return None
    sorted_vocabs = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    smaller_final_vocab = sorted_vocabs
    if pattern:
        for p in pattern:
            much_smaller_final_vocab = [i for i in smaller_final_vocab if p in i[0]]
            if len(much_smaller_final_vocab)!= 0:
                smaller_final_vocab = much_smaller_final_vocab
    final_vocab1 = []
    for voc in smaller_final_vocab:
        if not any(no_al in voc[0] for no_al in not_allele):
            final_vocab1.append(voc)
    if len(final_vocab1) != 0: smaller_final_vocab = final_vocab1
    final_vocab2 = [i for i in smaller_final_vocab if allele in i[0]]  # based on allele ('SLA' 'DLA' 'HLA' are very similar in Levenshtein)
    if len(final_vocab2) != 0:
        return [i[0] for i in final_vocab2[0:num_matches]]
    elif len(smaller_final_vocab) != 0:
        return [i[0] for i in smaller_final_vocab[0:num_matches]]
    elif len(sorted_vocabs) != 0:
        return [i[0] for i in sorted_vocabs[0:num_matches]]
    else:
        return None

def match_inputseq_to_netmhcpan_allele(sequence, mhc_type, mhc_allele=None,
                                       netmhcpan_data_path=None,
                                       pseudoseq_path=None):
    '''
        finds the match in netMHCpan Alleles. Works only for one allele (MHC-2 A and B) should be separately
        ran by this code mhc_allele overwrites sequence
    '''
    if not netmhcpan_data_path: netmhcpan_data_path = os.path.join(pmgen_abs_dir, 'data/HLA_alleles/netmhcpan_alleles')
    if not pseudoseq_path: pseudoseq_path= os.path.join(pmgen_abs_dir, 'data/HLA_alleles/pseudoseqs/PMGen_pseudoseq.csv')
    if not mhc_allele: assert isinstance(sequence, str)
    assert mhc_type in [1,2]
    df = pd.read_csv(pseudoseq_path)
    df = df[df['mhc_types']==mhc_type]
    netmhcpan_data_path = os.path.join(netmhcpan_data_path, f'mhc{mhc_type}')
    if not mhc_allele: # find with alignment
        sequences = df.sequence.tolist()
        simple_alleles = df.simple_allele.tolist()
        # run alignment
        scores = [
            pairwise2.align.globalxx(sequence, seq.replace('-', ''), score_only=True)
            for seq in sequences ]
        sorted_data = sorted(zip(simple_alleles, sequences, scores),
                             key=lambda x: x[2],  # Sort by score (third element)
                             reverse=True)
        alleles = [i[0] for i in sorted_data]
        # for each found allele in our db, search in netmhcpan db:
    else:
        alleles = [mhc_allele]
    matched_allele = None
    for a in alleles:
        matched_allele = find_similar_strings(a, netmhcpan_data_path)
        if matched_allele: break
    return matched_allele[0]



def run_netmhcpan(peptide_fasta, allele_list, output, mhc_type,
                  netmhcipan_path=netmhcipan_path, netmhciipan_path=netmhciipan_path):
    assert mhc_type in [1, 2]

    if mhc_type == 1:
        cmd = [str(netmhcipan_path), '-f', str(peptide_fasta),
               '-BA', '-a', str(allele_list[0])]

    elif mhc_type == 2:
        final_allele = ""
        for allele in allele_list:
            if 'H-2' in allele or 'DRB' in allele:
                final_allele = allele
            else:
                if 'DQA' in allele or 'DPA' in allele:
                    final_allele += allele
                if 'DQB' in allele or 'DPB' in allele:
                    final_allele += f'-{allele.replace("HLA-", "")}'
        cmd = [str(netmhciipan_path), '-f', str(peptide_fasta),
               '-BA', '-u', '-s', '-length', '9,10,11,12,13,14,15,16,17,18',
               '-inptype', '0', '-a', str(final_allele)]
    # Open the output file and redirect stdout to it
    with open(output, 'w') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, check=True)


def fetch_polypeptide_sequences(pdb_path):
    """
    Fetches the polypeptide sequences from a PDB file.
    Args:
        pdb_path (str): Path to the PDB file.
    Returns:
        dict: A dictionary where keys are chain IDs and values are polypeptide sequences.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_path)
    ppb = PPBuilder()
    sequences = {}
    for model in structure:
        for chain in model:
            chain_id = chain.id
            polypeptides = ppb.build_peptides(chain)
            if polypeptides:
                sequence = ''.join([str(pp.get_sequence()) for pp in polypeptides])  # Convert Seq to str
                sequences[chain_id] = sequence
    return sequences


def extract_scores_from_proteinmpnn_fasta(fasta_file: str) -> List[float]:
    """
    Reads a FASTA file and extracts score values from headers.

    Args:
        fasta_file (str): Path to the FASTA file

    Returns:
        List[float]: List of score values extracted from headers
    """
    scores = []
    try:
        with open(fasta_file, 'r') as f:
            for line in f:
                # Check if line is a header (starts with '>')
                if line.startswith('>'):
                    # Extract score using regex
                    # Looking for 'score=' followed by a number (including decimals)
                    score_match = re.search(r'score=([\d.]+)', line)
                    if score_match:
                        # Convert the matched score string to float
                        score = float(score_match.group(1))
                        scores.append(score)
                    else:
                        print(f"Warning: No score found in header: {line.strip()}")
        return scores
    except FileNotFoundError:
        print(f"Error: File '{fasta_file}' not found")
        return []
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return []


def split_pdb_chains(input_pdb, output_dir):
    """
    Splits a PDB file into separate files for each chain.
    Args:
    input_pdb (str): Path to the input PDB file.
    output_dir (str): Directory to save the output PDB files.
    Returns:
    list: List of paths to the created chain PDB files.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Set up the PDB parser
    parser = PDB.PDBParser(QUIET=True)
    # Parse the PDB file
    structure = parser.get_structure("protein", input_pdb)
    # Get the base name of the input file (without extension)
    base_name = os.path.splitext(os.path.basename(input_pdb))[0]
    output_files = []
    # Iterate through each chain in the structure
    for chain in structure.get_chains():
        chain_id = chain.id
        # Create a new structure with only this chain
        new_structure = PDB.Structure.Structure(chain_id)
        new_model = PDB.Model.Model(0)
        new_structure.add(new_model)
        new_model.add(chain)
        # Set the output file name
        output_file = os.path.join(output_dir, f"{base_name}_chain_{chain_id}.pdb")
        # Save the chain as a new PDB file
        io = PDB.PDBIO()
        io.set_structure(new_structure)
        io.save(output_file)
        output_files.append(output_file)
        print(f"Saved chain {chain_id} to {output_file}")
    return output_files


def align_and_find_anchors_mhc(peptide1, peptide2, mhc_type):
    """
    Aligns two peptide sequences and identifies anchor positions for mhcs.
    Parameters:
        peptide1 (str): The first peptide sequence (query peptide).
        peptide2 (str): The second peptide sequence (predicted core).
        mhc_type (int): The mhc type as 1 or 2.
    Returns:
        tuple: (aligned_pept1, aligned_pept2, predicted_anchors)
            - aligned_pept1 (str): Aligned sequence for peptide1.
            - aligned_pept2 (str): Aligned sequence for peptide2.
            - predicted_anchors (list): List of two integers representing the anchor positions in peptide1.
    """
    # Perform global alignment
    assert mhc_type in [1,2]
    alignment = pairwise2.align.globalxx(peptide1, peptide2)
    # If multiple alignments exist, prefer one with no gap at the second position
    if len(alignment) > 1:
        flag = False
        for prediction in alignment:
            if prediction[1][1] != '-' and prediction[0][1] != '-':
                pept1 = prediction[0]
                pept2 = prediction[1]
                flag = True
                break
        if not flag:
            pept1 = alignment[0][0]
            pept2 = alignment[0][1]
    else:
        pept1 = alignment[0][0]
        pept2 = alignment[0][1]
    # Remove gaps if they appear in the same position in both sequences (except at start)
    to_remove = []
    for i, (aa1, aa2) in enumerate(zip(pept1, pept2)):
        if aa1 == aa2 == '-' and i != 0:
            to_remove.append(i)
    for x in reversed(to_remove):
        pept1 = pept1[:x] + pept1[x + 1:]
        pept2 = pept2[:x] + pept2[x + 1:]
    # pept1 and pept2 are aligned: e.g AGHKMILEP and -GH-MI---
    if mhc_type == 2:
        if len(peptide1) >= 11:
            predicted_anchors = [3, 6, 8, 11]
        else:
            predicted_anchors = [2, 5, 7, 10]
        for i, p1, p2 in zip(range(len(pept1)), pept1, pept2):
            if p1 == p2 != '-':
                predicted_anchors = [i+1, i+3, i+5, i+8]
                break
    if mhc_type == 1:
        # Initialize predicted anchors (default: position 2 and length of peptide1)
        predicted_anchors = [2, len(peptide1)]
        # Find the first anchor
        p1 = 0
        p2 = 0
        for i in range(len(pept2)):
            if i == 1 and pept2[i] != '-' and pept1[i] != '-':
                predicted_anchors[0] = p1 + 1
                break
            elif i == 0 and pept2[i] != '-' and pept1[i] != '-':
                predicted_anchors[0] = p1 + 1
                break
            elif i > 1 and pept2[i] != '-':
                predicted_anchors[0] = p1 + 1
                break
            if pept1[i] != '-':
                p1 += 1
            if pept2[i] != '-':
                p2 += 1
        # Find the second anchor
        for i in range(len(pept2)):
            if pept2[::-1][i] != '-':
                predicted_anchors[1] = len([j for j in pept1[:len(pept1) - i] if j != '-'])
                break
    return predicted_anchors, pept1, pept2


def read_and_extract_core_plddt_from_df_with_anchor(df, output_folder, path_to_af='alphafold', multiple_anchors=False):
    cols = list(df.columns)
    for col in cols: assert col in ['peptide', 'mhc_seq', 'anchors', 'mhc_type', 'id']
    BEST_LDDTs = []
    BEST_STRUCTURES = {}
    for num, row in df.iterrows():
        anchors = row['anchors']
        peptide = row['peptide']
        mhc_seq = row['mhc_seq']
        id = str(row['id'])
        mhc_type = row['mhc_type']
        # extract first and last anchors to define the core
        if int(mhc_type) == 2: assert len(anchors.split(';')) == 4, f'anchors for mhc2 incorrect, found {anchors}'
        if int(mhc_type) == 1: assert len(anchors.split(';')) == 2, f'anchors for mhc1 incorrect, found {anchors}'
        anchor1 = int(anchors.split(';')[0]) - 1
        anchor2 = int(anchors.split(';')[-1]) - 1
        mhc_len = len(mhc_seq.replace('/', ''))
        # check if the numbers are correct
        combined_seq = mhc_seq.replace('/', '') + peptide
        after = mhc_len + anchor2
        if anchor1 > 0:
            before = anchor1 - 1 + mhc_len # start from one aa before first anchor
            assert combined_seq[before:after] == peptide[anchor1-1:anchor2], (f'peptide lddt incorrectly chosen: core supposed'
                                                                            f'to be {peptide[anchor1-1:anchor2]}, found: {combined_seq[before:after]}'
                                                                            f'\n before:{before}, after:{before}')
        else:
            before = anchor1 + mhc_len # if first anchor is already the first amino acid
            assert combined_seq[before:after] == peptide[anchor1:anchor2], (f'peptide lddt incorrectly chosen: core supposed'
                                                                            f'to be {peptide[anchor1:anchor2]}, found: {combined_seq[before:after]}'
                                                                            f'\n before:{before}, after:{before}')

        # find the folder that alphafold outputs are in
        alphafold_folder = os.path.join(output_folder, path_to_af, id)
        arrays = os.listdir(alphafold_folder)
        arrays = [i for i in arrays if id in i and '_plddt.npy' in i]
        assert len(arrays) > 0, f'No array files found in {alphafold_folder}'
        best_plddt_incore = 0
        tmp_df = {}
        for arr in arrays: #inside each anchor, if we have multiple models, choose the best
            arr_path = os.path.join(alphafold_folder, arr)
            plddt = np.load(arr_path)
            core_plddt = np.mean(plddt[before:after])
            if core_plddt > best_plddt_incore:
                best_plddt_incore = core_plddt
                assert '_plddt.npy' in arr, f'_plddt.npy not in {arr}'
                BEST_STRUCTURES[id] = arr.replace('_plddt.npy', '.pdb')
            tmp_df[arr] = [core_plddt]
        pd.DataFrame(tmp_df).to_csv(os.path.join(alphafold_folder, f'{id}_core_lddt.csv'))
        # get the final best plddt among models
        BEST_LDDTs.append(best_plddt_incore)
    # add  core plddts to df and find the best out of different anchors
    df['core_plddt'] = BEST_LDDTs
    if multiple_anchors:
        df['unique_ids'] = ['_'.join(i.split('_')[:-1]) for i in df['id'].tolist()] # 6OKJ_0 --> 6OKJ
    else:
        df['unique_ids'] = df['id'].tolist()
    best_structures = os.path.join(output_folder, 'best_structures')
    os.makedirs(best_structures, exist_ok=True)
    BEST_DF = []
    for unq in pd.unique(df['unique_ids']).tolist():
        subset_df = df[df['unique_ids']==unq]
        best_plddt_row = subset_df[subset_df['core_plddt'] == subset_df['core_plddt'].max()].head(1) # first row with best core plddt
        assert len(best_plddt_row) == 1, f'Len should be 1, found: {len(best_plddt_row)}\n{best_plddt_row}'
        id = str(pd.DataFrame(best_plddt_row)['id'].tolist()[0])
        BEST_DF.append(best_plddt_row)
        best_structure_input = os.path.join(output_folder, path_to_af, id, BEST_STRUCTURES[id])
        assert os.path.exists(best_structure_input), f'does not exists: {best_structure_input}'
        shutil.copy(best_structure_input, os.path.join(best_structures, f'{id}_PMGen.pdb'))
    final_df = pd.concat(BEST_DF)
    final_df.to_csv(os.path.join(output_folder, 'final_df.tsv'), sep='\t', index=False)
    print(f'## All best structures saved in {best_structures} ##')
    print(f'## final_df is saved in {os.path.join(output_folder, "final_df.tsv")} ##')
    return final_df


def alignment_to_string(alignment):
    """
    Converts an alignment into a string format where ';' separates pairs
    and ':' denotes the index mapping (query:target).
    :param alignment: A tuple or list of two aligned sequences (query, target)
    :return: A formatted string showing index mappings
    """
    query, target = alignment
    mapping = []
    q_idx, t_idx = 0, 0  # Track original indices
    for q_res, t_res in zip(query, target):
        if q_res != "-" and t_res != "-":  # Both residues align
            mapping.append(f"{q_idx}:{t_idx}")
        if q_res != "-":  # Move query index if not a gap
            q_idx += 1
        if t_res != "-":  # Move target index if not a gap
            t_idx += 1
    return ";".join(mapping), q_idx, t_idx

def alignment_to_df(no_modelling_output_dict, output_dir):
    '''runs in case of no-modelling, when we have no pandora output'''
    template_pdbfile = []
    target_to_template_alignstring = []
    identities = []
    target_len = []
    template_len = []
    for i in range(len(no_modelling_output_dict['template_id'])):
        tp = no_modelling_output_dict['template_path'][i]
        aln, q_idx, t_idx = alignment_to_string([no_modelling_output_dict['aln_target'][i], no_modelling_output_dict['aln_template'][i]])
        template_pdbfile.append(tp)
        target_to_template_alignstring.append(aln)
        identities.append(len(aln.split(';')))
        target_len.append(q_idx)
        template_len.append(t_idx)
    df = pd.DataFrame({
        "template_pdbfile": template_pdbfile,
        "target_to_template_alignstring": target_to_template_alignstring,
        "identities": identities,
        "target_len": target_len,
        "template_len": template_len})
    df.to_csv(output_dir, sep='\t', index=False)








