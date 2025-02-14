import pandas as pd
import os
import re
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from processing_functions import BA_raw_processing, HLA_annotation, merge_mhc_data, filter_fasta_by_length, get_aminoacid_sequences
import sys

df = pd.read_csv(f'{sys.argv[1]}', sep='\t', header=0)
output_df = sys.argv[2]

netmhc_pan_dataframe = BA_raw_processing(input_df=df, Allele_type=["A","B","C"], num_queries=360000,
                      Allele_col=2, BA_col=1, peptide_col=0, pos_portion=None,  random_state=None)

mhc_2_dict, NF = HLA_annotation(df=netmhc_pan_dataframe, MHC_type=1)

DF = pd.DataFrame(mhc_2_dict)

DF.to_csv(output_df, sep='\t', index=False)
