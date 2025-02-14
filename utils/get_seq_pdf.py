
import pandas as pd
import os
import re
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from processing_functions import BA_raw_processing, HLA_annotation, merge_mhc_data, filter_fasta_by_length, get_aminoacid_sequences

# Directory containing PDB files
pdb_directory = '/home/mpg08/a.hajialiasgarynaj01/PANDORA_databases/default/PDBs/pMHCII'

# Columns for the DataFrame
columns = ["pdb_file_name", "concatenated_sequence", "chain_names"]

# Initialize an empty DataFrame
result_df = pd.DataFrame(columns=columns)

# Iterate through each PDB file in the directory
for pdb_file in os.listdir(pdb_directory):
    print(pdb_file)
    if pdb_file.endswith('.pdb'):
        pdb_path = os.path.join(pdb_directory, pdb_file)
        
        # Get amino acid sequences
        concatenated_seq, chain_seqs = get_aminoacid_sequences(pdb_path, exclude_residues=["HOH", "H2O"])
        
        # Append the results to the DataFrame
        result_df = result_df.append({
            "pdb_file_name": pdb_file,
            "concatenated_sequence": concatenated_seq,
            "chain_names": "/".join(chain_seqs.keys())
        }, ignore_index=True)

# Display the resulting DataFrame
result_df.to_csv("../data/templates/templates_2.csv", sep="\t", index=False)
print(result_df)

