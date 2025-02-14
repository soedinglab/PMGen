from Bio import SeqIO

# Input and output file paths
input_fasta_file = "data/mhc2.fasta"
output_fasta_file = "data/mhc_2.fasta"

# Function to modify the sequence IDs
def modify_sequence_id(sequence_id,MHC_2=True):
    # Remove ":" and "*" from the sequence ID
    sequence_id = sequence_id.split(" ")[1]
    modified_id = "HLA-"+sequence_id.replace("*", "")
    if MHC_2:
    	modified_id = modified_id.replace(":","") 
    return modified_id

# Iterate through the sequences, modify the IDs, and save to a new file
with open(output_fasta_file, "w") as output_handle:
    for record in SeqIO.parse(input_fasta_file, "fasta"):
        # Modify the sequence ID
        modified_id = modify_sequence_id(record.description)
        record.id = modified_id
        record.description = ""
        # Write the modified record to the output file
        SeqIO.write(record, output_handle, "fasta")
