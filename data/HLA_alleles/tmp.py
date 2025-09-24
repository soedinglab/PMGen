import os
from Bio import SeqIO

# Define the directory containing FASTA files
fasta_dir = "reference_allele_fasta"

# Loop through all files in the directory
for filename in os.listdir(fasta_dir):
#if filename.endswith(".fasta") or filename.endswith(".fa"):  # Ensure it's a FASTA file
    filepath = os.path.join(fasta_dir, filename)
    
    modified_records = []
    
    for record in SeqIO.parse(filepath, "fasta"):
        new_header = f"{record.id}|{record.description}".replace(" ", "|")
        record.id = new_header
        record.description = ""
        modified_records.append(record)
    
    # Write back the modified records to the same file (or a new file if needed)
    with open(filepath, "w") as output_handle:
        SeqIO.write(modified_records, output_handle, "fasta")
    print(filepath)

print("Headers updated successfully!")

