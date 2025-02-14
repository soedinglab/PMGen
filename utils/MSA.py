from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import pairwise2
import re

def extract_between_patterns(input_string, start_pattern, end_pattern):
    # Use re.search to find the start and end patterns
    start_match = re.search(start_pattern, input_string)
    end_match = re.search(end_pattern, input_string)

    # Check if both patterns were found
    if start_match and end_match:
        # Extract the substring between the patterns
        start_index = start_match.end()
        end_index = end_match.start()
        result = input_string[start_index:end_index]
        return start_pattern + result + end_pattern
    else:
        # If one or both patterns are not found, return an empty string
        return ""



def chop_seq_fasta(fasta_file, start_pattern, end_pattern, output_file, cut_limit=90):
    # Read sequences from the FASTA file
    fasta_sequences = list(SeqIO.parse(fasta_file, "fasta"))

    # Perform alignment and process the sequences
    processed_sequences = []
    seen_sequences = []
    for record in fasta_sequences:
        sequence = str(record.seq)
        result = extract_between_patterns(sequence, start_pattern, end_pattern)
        if len(result) <= (len(start_pattern+end_pattern)+cut_limit):
            continue
        elif result not in seen_sequences:
            seen_sequences.append(result)
            processed_seq = SeqRecord(Seq(result), id=record.id, description=f"{record.description} / len={len(result)}")
            processed_sequences.append(processed_seq)
    # Save the processed sequences to a new FASTA file
    SeqIO.write(processed_sequences, output_file, "fasta")

cuts = [90, 90, 90, 90, 85, 70, 10, 10, 10,10,10,10,10,10,10, 10, 10, 10]
types = ["A", "B", "C", "E", "F", "G", "DMA", "DMB", "DOA","DOB","DPA1","DPB1","DQA1","DQA2","DQB1","DQB2","DRA","DRB",]
S = ["GSHSMR", "GSHSMR", "SHSMRY", "GSHSL", "GSHSL", "SHSMRY", "VPEAPT", "GGFVAH", "KADHMGSY","PEDFVIQ", "HVSTYA","YVY","HVAS","EDIVADH", "RDSPE", "AEARDFPK","HVIIQA", "FLE"]
E = ["RRYLENG", "RRYLENG", "LENGK", "HLEPPKTHV", "LENGKET", "GKEML", "QQIGPKLD", "SLTNRTRPP", "ERSNRSR","YRLGAPFTV", "LIQRSN","AVTLQ","NS","AATNEVP", "RVEP", "VEPTVTIS", "TKRSN", "FTVQ"]
for cut, type, s, e in zip(cuts, types, S, E):
    # Example usage:
    fasta_file = f"/scratch/users/a.hajialiasgarynaj01/Master_thesis/Pipline_test_PANDORA_AFoptimization/processing/data/IMGT_processed/{type}_prot.fasta"  # replace with your input FASTA file
    start_pattern = s  # replace with your single sequence
    end_pattern = e
    output_file = f"data/processed_{type}.fasta"  # replace with your desired output FASTA file

    chop_seq_fasta(fasta_file, start_pattern, end_pattern, output_file, cut)
    
















