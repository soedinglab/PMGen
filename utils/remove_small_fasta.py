from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import pairwise2
import re

input_file = "data/mhc_2.fasta"
output_file = "data/processed_mhc_2.fasta"
min_length = 70

with open(output_file, "w") as out_file:
    for record in SeqIO.parse(input_file, "fasta"):
        if len(record.seq) >= min_length:
            SeqIO.write(record, out_file, "fasta")
