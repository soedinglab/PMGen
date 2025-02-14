import pandas as pd
import os
import re
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.PDB import PDBParser
from Bio import PDB
from Bio.SeqUtils import seq3
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo
from processing_functions import BA_raw_processing, HLA_annotation, merge_mhc_data, filter_fasta_by_length, get_aminoacid_sequences, process_string, map_query_template, prepare_alignment_file_with_peptide, prepare_alignment_file_without_peptide


tar_seq = "EEHVI/STYYYYY/PEVIPMFSALSEGAT"



df = prepare_alignment_file_with_peptide(pdb_file="/scratch/users/a.hajialiasgarynaj01/Master_thesis/Pipline_test_PANDORA_AFoptimization/data/templates/pdb_mhcII/1SJE.pdb", target_seq=tar_seq, mhc_type=1,output_path="../data/templates/all_templates.csv", peptide=True)
print(df)

