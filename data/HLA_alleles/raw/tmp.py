import os
from Bio import SeqIO

# Define the directory containing FASTA files
fasta_dir = "./"

# Loop through all files in the directory
for filename in os.listdir(fasta_dir):
    if filename.endswith(".fasta") or filename.endswith(".fa"):  # Ensure it's a FASTA file
        filepath = os.path.join(fasta_dir, filename)
        
        with open(filepath, "r") as file:
            for record in SeqIO.parse(file, "fasta"):
                print(f"{filename}: {record.id}")  # Print only the first header
                break  # Stop after the first record

