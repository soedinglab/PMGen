from Bio import SeqIO, AlignIO
import os
import subprocess
import pandas as pd
import shutil
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def find_longest_sequence(fasta_file):
    longest_seq = ""
    longest_id = ""
    max_length = 0

    for record in fasta_file:
        if len(record.seq) > max_length:
            max_length = len(record.seq)
            longest_seq = str(record.seq)
            longest_id = f"{record.description}"

    return longest_id, longest_seq


def filter_fasta_by_header(fasta_file, search_string):
    """
    Reads a FASTA file and returns a list of SeqIO objects for sequences
    whose headers contain a specified search string.
    """
    filtered_sequences = []

    for record in fasta_file:
        full_header = f"{record.description}"
        if search_string.lower() in full_header.lower():
            filtered_sequences.append(record)

    return filtered_sequences

def filter_longest_seq_to_df(folder = '../data/HLA_alleles/raw',
                             output='../data/HLA_alleles/longest_allels.csv'):
    DICT = {}
    files = os.listdir(folder)
    for file in files:
        path = os.path.join(folder, file)
        fasta_file = SeqIO.parse(path, "fasta")
        if 'DRB345' in file:
            for dr in ['DRB3', 'DRB4', 'DRB5']:
                fasta_file = SeqIO.parse(path, "fasta")
                filtered_sequences = filter_fasta_by_header(fasta_file, dr)
                hla, seq = find_longest_sequence(filtered_sequences)
                DICT[dr] = [seq, len(seq), hla]
        else:
            hla, seq = find_longest_sequence(fasta_file)
            DICT[file.split('_prot')[0]] = [seq, len(seq), hla]
    df = pd.DataFrame(DICT, index=['Sequence', 'Len', 'Allele'])
    df.transpose().to_csv(output, index=True)
    return df.transpose()



def df_to_fasta(df, seq_col, output, header_col=None):
    df = pd.read_csv(df, index_col=0)
    for i, row in df.iterrows():
        with open(os.path.join(output, row.name), 'w') as f:
            seq = row[seq_col]
            header = row[header_col] if header_col else row.name
            f.write(f'>{header}\n{seq}')
        f.close()


def filter_mafft_alignment(query_fasta, reference_fasta, aligned_output, filtered_output, coverage_threshold=50):
    """
    Runs MAFFT to align sequences, filters out sequences with alignment coverage below the given threshold,
    removes alignment gaps, and saves the filtered sequences to a new FASTA file.

    Parameters:
        query_fasta (str): Path to the input FASTA file containing multiple sequences.
        reference_fasta (str): Path to the reference FASTA file.
        aligned_output (str): Path to save the MAFFT-aligned output.
        filtered_output (str): Path to save the filtered sequences without gaps.
        coverage_threshold (int): Minimum coverage percentage required to keep a sequence.
    """
    # Run MAFFT alignment
    mafft_cmd = f"mafft --auto --reorder --keeplength --addfragments {query_fasta} {reference_fasta} > {aligned_output}"
    subprocess.run(mafft_cmd, shell=True, check=True)

    # Read the aligned sequences
    alignment = AlignIO.read(aligned_output, "fasta")
    num_records_init = len(alignment)

    # Load reference sequence from the original file (to ensure correct identification)
    ref_seq_record = next(SeqIO.parse(reference_fasta, "fasta"))
    ref_seq_id = ref_seq_record.id
    ref_seq_length = len(ref_seq_record.seq)  # Original reference sequence length

    # Compute coverage correctly
    def compute_coverage(query_seq):
        aligned_length = sum(1 for res in query_seq if res != "-")
        return (aligned_length / ref_seq_length) * 100

    # Filter sequences based on coverage and remove gaps
    filtered_sequences = []
    for record in alignment:
        if record.id == ref_seq_id:  # Always keep reference sequence
            filtered_sequences.append(record)
            continue

        coverage = compute_coverage(str(record.seq))
        if coverage >= coverage_threshold:
            new_seq = SeqRecord(
                seq=Seq(str(record.seq).replace("-", "").replace("X", "")),  # Remove gaps & X
                id= f'{record.id}',
            description = record.description
            )
            filtered_sequences.append(new_seq)

    num_records_final = len(filtered_sequences)

    # Save filtered sequences without gaps
    SeqIO.write(filtered_sequences, filtered_output, "fasta")

    print(f"Initial sequences: {num_records_init}")
    print(f"Filtered sequences: {num_records_final}")
    print(f"Filtered sequences (without gaps) saved to {filtered_output}")

    return num_records_init, num_records_final


def save_filtered_fasta_by_header(input_fasta, search_string, output_fasta):
    """
    Filters sequences from a FASTA file based on whether the header contains a given string.

    Parameters:
        input_fasta (str): Path to the input FASTA file.
        search_string (str): The string to search for in sequence headers.
        output_fasta (str): Path to save the filtered sequences.
    """
    # Read and filter sequences
    filtered_sequences = [
        record for record in SeqIO.parse(input_fasta, "fasta")
        if search_string in f'{record.id} {record.description}'
    ]
    SeqIO.write(filtered_sequences, output_fasta, "fasta")

    print(f"Filtered sequences saved to {output_fasta}")


def count_fasta_records(fasta_path):
    return sum(1 for _ in SeqIO.parse(fasta_path, "fasta"))


def run_mmseqs(input_path, output_path, seq_iden=0.9, seq_coverage=0.95, tmp='tmp'):
    cmd = f"mmseqs easy-cluster {input_path} {output_path} {tmp} --min-seq-id {seq_iden} -c {seq_coverage}"
    subprocess.run(cmd, shell=True, check=True)

def mafft_aln_and_process_mmseq(path_reference, path_raw, path_processed, mafft_coverage_threshold=80,
                          mmseq_seq_iden=0.9, mmseq_seq_coverage=0.95):
    '''
    A custom function, not generalizable. does Mafft aln between raw and processed data and then
    saves them with more than 80% sequence identity. Then does mmseq clustering for them.
    '''

    path_reference = path_reference.rstrip('/')
    path_raw = path_raw.rstrip('/')
    path_processed = path_processed.rstrip('/')
    os.makedirs(path_processed + '/mmseq_clust', exist_ok=True)
    SPECIES = ['homo', 'SLA', 'Patr', 'Mamu', 'BOLA', 'DLA']
    sums = {f'homo_1':0, f'homo_2':0,
            f'SLA_1':0, f'SLA_2':0,
            f'Patr_1':0, f'Patr_2':0,
            f'Mamu_1':0, f'Mamu_2':0,
            f'BOLA_1':0, f'BOLA_2':0,
            f'DLA_1':0, f'DLA_2':0}
    counts = {}
    for species in SPECIES:
        folders = os.listdir(path_reference)
        folders = [i for i in folders if f'-{species}' in i]

        for ref in folders:
            '''if species=='homo' and ref in ['DRB3-homo', 'DRB4-homo', 'DRB5-homo']:
                save_filtered_fasta_by_header(input_fasta=f"{path_raw}/DRB345-{species}_prot.fasta",
                                              search_string=ref,
                                              output_fasta=f"{path_raw}/{ref}_prot.fasta")'''
            # RUN MAFFT
            num_records_init, num_records_final = filter_mafft_alignment(
                                    query_fasta=f"{path_raw}/{ref}_prot.fasta",
                                    reference_fasta=f"{path_reference}/{ref}",
                                    aligned_output=f"{path_processed}/{ref}_aln.fa",
                                    filtered_output=f"{path_processed}/{ref}_seqs_80.fa",
                                    coverage_threshold=mafft_coverage_threshold  # Change this value as needed
                                )
            # RUN MMSEQ2
            os.makedirs(f"{path_processed}/mmseq_clust", exist_ok=True)
            run_mmseqs(input_path = f"{path_processed}/{ref}_seqs_80.fa",
                       output_path = f"{path_processed}/mmseq_clust/{ref}_clust",
                       seq_iden=mmseq_seq_iden, seq_coverage=mmseq_seq_coverage, tmp='tmp')
            #os.remove(f"{path_processed}/{ref}_aln.fa")
            mhc_type = 2 if 'D' in ref[0] else 1
            sums[f'{species}_{mhc_type}'] = sums[f'{species}_{mhc_type}'] + num_records_final
            cluster_rep = count_fasta_records(f"{path_processed}/mmseq_clust/{ref}_clust_rep_seq.fasta")
            counts[ref] = [num_records_init, num_records_final, num_records_init - num_records_final, mhc_type, cluster_rep, species]
    counts = pd.DataFrame(counts)
    counts.index = ['n_before_processing', 'n_after_processing', 'n_diff', 'mhc_type', 'cluster_rep', 'species']
    counts = counts.transpose()
    mhc1 = pd.DataFrame([[counts[counts['mhc_type'] == 1]['n_before_processing'].sum(),
                          counts[counts['mhc_type'] == 1]['n_after_processing'].sum(),
                          counts[counts['mhc_type'] == 1]['n_diff'].sum(),
                          1,
                          counts[counts['mhc_type'] == 1]['cluster_rep'].sum(),
                          'all']],
                        columns=counts.columns)
    mhc2 = pd.DataFrame([[counts[counts['mhc_type'] == 2]['n_before_processing'].sum(),
                          counts[counts['mhc_type'] == 2]['n_after_processing'].sum(),
                          counts[counts['mhc_type'] == 2]['n_diff'].sum(),
                          2,
                          counts[counts['mhc_type'] == 2]['cluster_rep'].sum(),
                          'all']],
                        columns=counts.columns)
    counts = pd.concat([counts, mhc1, mhc2], ignore_index=False)
    counts.to_csv(f'{path_processed}/number_changes2.csv', index=True)

def mhc2_cluster_combination_seqs(path_to_cluster_representatives, out_dir, iden_for_representative='_rep'):
    files = [os.path.join(path_to_cluster_representatives, i)
             for i in os.listdir(path_to_cluster_representatives)
             if iden_for_representative in i and 'fasta' in i and
             'D' in i[0]]
    out_dir = out_dir.rstrip('/')
    os.makedirs(out_dir, exist_ok=True)
    DICT = {'DMA':'DMB', 'DOA':'DOB', 'DPA1':'DPB1', 'DQA1':'DQB1', 'DRA':'DRB1-DRB3-DRB4-DRB5'}
    SPECIES = ['homo', 'SLA', 'Patr', 'Mamu', 'BOLA', 'DLA']
    total_len = 0
    for species in SPECIES:
        if species=='homo':
            DICT = {'DMA': 'DMB', 'DOA': 'DOB', 'DPA1': 'DPB1', 'DQA1': 'DQB1', 'DRA': 'DRB1-DRB3-DRB4-DRB5'}
        else:
            DICT = {'DMA': 'DMB', 'DOA': 'DOB', 'DPA': 'DPB', 'DQA': 'DQB', 'DRA': 'DRB'}
        for key, value in DICT.items():
            alpha = [i for i in files if key in i and species in i]
            assert len(alpha)<=1, f"len incorrect alpha: {alpha}, {species}, {len(alpha)}, {key}, {value}"
            if len(alpha) == 0: continue # no file found for that species
            alpha = alpha[0]
            for val in value.split('-'):
                alpha_rec = SeqIO.parse(alpha, "fasta")
                beta = [i for i in files if val in i and species in i]
                assert len(beta) == 1, f"len incorrect beta: {beta}, {species}, {len(beta)}, {key}, {val}"
                beta = beta[0]
                with open(f"{out_dir}/{key}_{val}-{species}.fasta", "w") as f:
                    for rec_a in alpha_rec:
                        beta_rec = SeqIO.parse(beta, "fasta")
                        for rec_b in beta_rec:
                            f.write(f'>{rec_a.description};;{rec_b.description}\n'
                                    f'{rec_a.seq}/{rec_b.seq}\n')
                            total_len += 1
                f.close()
        print('Total Combinations:', total_len)
    df = {}
    for file in [i for i in os.listdir(out_dir) if '.fasta' in i and not os.path.isdir(os.path.join(out_dir, i))]:
        records = count_fasta_records(os.path.join(out_dir, file))
        df[file] = [records]
    pd.DataFrame(df).transpose().to_csv(out_dir + '/stats_combination.csv')



def create_parsefold_input_from_representatives(fasta_dir, peptides=[], mhc_type=2, iden='_rep', num=0, no_iden=None):
    assert mhc_type in [2,1], f'mhc type incorrect {mhc_type}'
    if len(peptides) == 0:
        if mhc_type == 1:
            peptides = ['MRMATPLLM'] #'NLVPMVATVAA', cytomegalovirus pp65 bound to A*02:01 https://www.nature.com/articles/srep35326
                                                    # CLIP https://pubmed.ncbi.nlm.nih.gov/16678175/
        elif mhc_type == 2:
            peptides = ['MRMATPLLMQALPM'] #CLIP peptide human https://www.sciencedirect.com/science/article/pii/S0022283602014377?via%3Dihub
    DF = []
    list_of_dir = [i for i in os.listdir(fasta_dir) if '.fasta' in i or '.fa' in i]
    list_of_dir = [i for i in list_of_dir if not os.path.isdir(os.path.join(fasta_dir, i)) and iden in i]
    if no_iden:
        list_of_dir = [i for i in list_of_dir if no_iden not in i]
    print(list_of_dir)
    for input_fasta in [os.path.join(fasta_dir, i) for i in list_of_dir]:
        if '_rep' in input_fasta and '.fasta' in input_fasta:
            data = SeqIO.parse(input_fasta, "fasta")
            for record in data:
                for peptide in peptides:
                    df = pd.DataFrame({
                        'peptide':[peptide],
                        'mhc_seq':[str(record.seq)],
                        'mhc_type':[mhc_type],
                        'anchors':[None],
                        'id':[f'{record.id}_{num}']
                    })
                    num = num + 1
                    DF.append(df)
    DF = pd.concat(DF)
    return DF






#df = filter_longest_seq_to_df()
#df_to_fasta('../data/HLA_alleles/longest_allels.csv',
#            'proceesed_seq',
#            '../data/HLA_alleles')

path_reference = '../data/HLA_alleles/reference_allele_fasta'
path_raw = '../data/HLA_alleles/raw'
path_processed = '../data/HLA_alleles/processed'

mafft_aln_and_process_mmseq(path_reference=path_reference,
                            path_raw=path_raw,
                            path_processed=path_processed,
                            mafft_coverage_threshold=80,
                            mmseq_seq_iden=0.95,
                            mmseq_seq_coverage=0.95)
mhc2_cluster_combination_seqs(path_to_cluster_representatives='../data/HLA_alleles/processed/mmseq_clust',
                              out_dir='../data/HLA_alleles/processed/mmseq_clust/mhc2_rep_combinations',
                              iden_for_representative='_rep')
df1 = create_parsefold_input_from_representatives(fasta_dir='../data/HLA_alleles/processed/mmseq_clust', peptides=[], mhc_type=1, no_iden='D')
df2 = create_parsefold_input_from_representatives(fasta_dir='../data/HLA_alleles/processed/mmseq_clust/mhc2_rep_combinations', peptides=[], mhc_type=2,
                                                  num = len(df1), iden='_D')
DF = pd.concat([df1, df2])
DF.to_csv('CLIP_example.tsv', sep='\t', index=False)