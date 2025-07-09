# Examples of how to run the pipline
### Design Peptides and MHC-Pseudosequences using Alphafold-Initial Guess Approach
`
python run_PMGen.py --run single --mode wrapper --output_dir tmppandora/ --benchmark --max_cores 8 --initial_guess --df data/example/smaller_wrapper_exmaple.tsv  --multiple_anchors --top_k 1 --peptide_design --num_sequences_peptide 50 --num_sequences_mhc 50 --only_pseudo_sequence_design --binder_pred --return_all_outputs
`