import re
def extract_scores_from_fasta(fasta_path):
    """
    Extracts headers and their corresponding scores from a FASTA file.

    Args:
        fasta_path (str): Path to the FASTA file.

    Returns:
        dict: A dictionary with headers as keys and scores as values (as floats).
    """
    score_dict = {}

    with open(fasta_path, 'r') as file:
        lines = file.readlines()
    
    for i in range(0, len(lines), 2):  # Iterate over header-sequence pairs
        header = lines[i].strip()
        match = re.search(r'score=([\d.]+)', header)
        
        if match:
            score = float(match.group(1))
            score_dict[header] = score

    return score_dict
    
    
a = extract_scores_from_fasta('multichain_5NPZ_model_1_model_2_ptm.fa')
for header, score in a.items():
    print(f"{header} -> Score: {score}")

