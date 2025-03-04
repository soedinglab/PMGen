import argparse
import shutil

import pandas as pd
from run_utils import run_parsefold_wrapper, run_parsefold_modeling, run_proteinmpnn
from Bio import SeqIO
import warnings
import os
from Bio import BiopythonDeprecationWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

def remove_files_in_directory(directory):
    """Remove all files inside the given directory (excluding directories and symlinks)."""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):  # Only remove files, not directories or links
            os.remove(file_path)


def main():
    parser = argparse.ArgumentParser(description="Run ParseFold wrapper or modeling.")

    # Default settings
    parser.add_argument('--run', choices=['parallel', 'single'], required=False, default='parallel',
                        help='Run mode: parallel or single (one by one). parallel only works if --mode wrapper')
    parser.add_argument('--mode', choices=['wrapper', 'modeling'], required=False, default='wrapper',
                        help='Recommended. Select wrapper or modeling mode')

    # Required arguments for modeling mode
    parser.add_argument('--peptide', type=str, help='Peptide sequence')
    parser.add_argument('--mhc_seq', type=str, help='MHC Sequence. for MHC-I provide one chain sequence.'
                                                    'for MHC-II provide chain_A/chain_B separated by "/", respectively', default=None)
    parser.add_argument('--mhc_fasta', type=str, help='a Fasta file with single a header and sequence For MHC-I'
                                                    'only provide one chain. For MHC-II provide >Alpha and >Beta Sequences.'
                                                    'If you want to run for multiple structures, use --df and --mode wrapper', default=None)
    parser.add_argument('--mhc_allele', type=str, help='MHC allele (optional). We recommend to use --mhc_seq instead.')
    parser.add_argument('--mhc_type', type=int, help='MHC type', choices=[1,2])
    parser.add_argument('--id', type=str, help='Identifier')
    parser.add_argument('--anchors', type=str, default=None, help='Anchor positions as a list, e.g [2,9]. (optional)')
    parser.add_argument('--predict_anchor', action='store_true', help='Recommended. Enable anchor prediction')

    # General arguments
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--num_templates', type=int, default=4, help='Number of engineered templates')
    parser.add_argument('--num_recycles', type=int, default=3, help='Number of recycles')
    parser.add_argument('--models', nargs='+', default=['model_2_ptm'], help='List of models to use. if fine'
                                                                             'tuned, please use _ft at the end of your model.')
    parser.add_argument('--alphafold_param_folder', type=str, default='AFfine/af_params/params_original/',
                        help='Path to AlphaFold parameters')
    parser.add_argument('--fine_tuned_model_path', type=str,
                        default='AFfine/af_params/params_finetune/params/model_ft_mhc_20640.pkl',
                        help='Path to fine-tuned model')
    parser.add_argument('--benchmark', action='store_true', help='Enable benchmarking')
    parser.add_argument('--best_n_templates', type=int, default=1, help='Best N templates')
    parser.add_argument('--n_homology_models', type=int, default=1, help='Number of homology models')
    parser.add_argument('--max_ram', type=int, default=3, help='Maximum RAM GB per job (only for parallel mode)')
    parser.add_argument('--max_cores', type=int, default=4, help='Maximum number of CPU cores (only for parallel mode)')

    # Wrapper mode argument
    parser.add_argument('--df', type=str, help='Recommended. Path to input TSV file (required for wrapper mode)'
                                               'check the github documentation for more information.')

    # ProteinMPNN Arguments
    parser.add_argument('--peptide_design', action='store_true', help='Enables peptide design.')
    parser.add_argument('--only_pseudo_sequence_design', action='store_true', help='Enables MHC pseudo-sequence design.')
    parser.add_argument('--mhc_design', action='store_true', help='Enables whole mhc design. we recommend to use only_pseudo_sequence_design mode.')
    parser.add_argument('--num_sequences_peptide', type=int, default=10, help='Number of peptide sequences to be generated. Works only with --peptide_design')
    parser.add_argument('--num_sequences_mhc', type=int, default=10, help='Number of mhc sequences to be generated. Works only with --only_pseudo_sequence_design or --mhc_design')
    parser.add_argument('--sampling_temp', type=float, default=0.1, help='ProteinMPNN sampling temperature.')
    parser.add_argument('--batch_size', type=int, default=1, help='ProteinMPNN batch size.')
    parser.add_argument('--hot_spot_thr', type=float, default=6.0, help='Distance threshold to peptide, to define hot-spots on mhc.')

    # Setting to Run only a part:
    parser.add_argument('--no_alphafold', action='store_false', help='does not run alphafold.')

    args = parser.parse_args()

    # Validation for modeling mode
    if args.mode == 'modeling':
        if not args.mhc_seq and not args.mhc_allele:
            raise ValueError("Either --mhc_seq or --mhc_allele must be provided for modeling mode.")
        if not args.id:
            args.id = args.peptide  # Use peptide as ID if not provided

    # Run wrapper mode
    if args.mode == 'wrapper':
        if not args.df:
            raise ValueError("--df is required for wrapper mode")
        df = pd.read_csv(args.df, sep='\t')
        # outputs for proteinmpnn later
        tmp_pdb_dict = {}
        for i, row in df.iterrows():
            out_alphafold = os.path.join(args.output_dir, 'alphafold', row['id'])
            tmp_pdb_dict[row['id']] = out_alphafold

        runner = run_parsefold_wrapper(df=df, output_dir=args.output_dir, num_templates=args.num_templates,
                                       num_recycles=args.num_recycles, models=args.models,
                                       alphafold_param_folder=args.alphafold_param_folder,
                                       fine_tuned_model_path=args.fine_tuned_model_path,
                                       benchmark=args.benchmark, best_n_templates=args.best_n_templates,
                                       n_homology_models=args.n_homology_models)
        if args.run == 'parallel':
            runner.run_wrapper_parallel(max_ram=args.max_ram, max_cores=args.max_cores, run_alphafold=args.no_alphafold)
        else:
            runner.run_wrapper(run_alphafold=args.no_alphafold)
        # check outputs if they exist:
        output_pdbs_dict = {}
        for key, value in tmp_pdb_dict.items():
            # {'id':[out1, out2], 'id2':[out1, out2], ...}
            output_pdbs_dict[key] = [os.path.join(value, i) for i in os.listdir(value) if i.endswith('.pdb') and 'model_' in i and not i.endswith('.npy')]

    # Run modeling mode
    elif args.mode == 'modeling':
        sequences = []
        if not args.mhc_seq:
            for record in SeqIO.parse(args.mhc_fasta, "fasta"):
                sequences.append(str(record.seq))
            if args.mhc_type == 1:
                sequence = sequences[0]
            elif args.mhc_type == 2: sequence = sequences[0] + "/" + sequences[1]
        else:
            sequence = args.mhc_seq
        runner = run_parsefold_modeling(peptide=args.peptide, mhc_seq=sequence, mhc_type=args.mhc_type,
                                        id=args.id, output_dir=args.output_dir, anchors=args.anchors,
                                        mhc_allele=args.mhc_allele, predict_anchor=args.predict_anchor,
                                        num_templates=args.num_templates, num_recycles=args.num_recycles,
                                        models=args.models, alphafold_param_folder=args.alphafold_param_folder,
                                        fine_tuned_model_path=args.fine_tuned_model_path,
                                        benchmark=args.benchmark, best_n_templates=args.best_n_templates,
                                        n_homology_models=args.n_homology_models)
        runner.run_parsefold(run_alphafold=args.no_alphafold)
        output_pdbs_dict = runner.output_pdbs_dict # {'id':[output1, output2, ...]}

    print("Alphafold Runs completed.")
    # get the pdb outputs for listing them and protmpnn


    if args.peptide_design or args.only_pseudo_sequence_design or args.mhc_design:
        print("### Start ProteinMPNN runs ###")
        print('files:\n', output_pdbs_dict)
        for id_m, path_list in output_pdbs_dict.items():
            directory = os.path.join(args.output_dir, 'protienmpnn', id_m)
            os.makedirs(directory, exist_ok=True)
            for path in path_list:
                #outdir/proteinmpnn/id/model_i/
                model_dir = os.path.join(directory, path.split('/')[-1].strip('.pdb')) # for each alphafold model, one path inside id created
                os.makedirs(model_dir, exist_ok=True)
                shutil.copy(path, os.path.join(model_dir, path.split('/')[-1])) #af/model --> outdir/proteinmpnn/id/model_i/model.pdb
                parsefold_pdb = os.path.join(model_dir, path.split('/')[-1])
                print('#########',parsefold_pdb)
                runner_mpnn = run_proteinmpnn(parsefold_pdb=parsefold_pdb, output_dir=model_dir,
                                         num_sequences_peptide=args.num_sequences_peptide,
                                         num_sequences_mhc=args.num_sequences_mhc,
                                         peptide_chain='P', mhc_design=args.mhc_design, peptide_design=args.peptide_design,
                                         only_pseudo_sequence_design=args.only_pseudo_sequence_design,
                                         anchor_pred=True,
                                         sampling_temp=args.sampling_temp, batch_size=args.batch_size,
                                         hot_spot_thr=args.hot_spot_thr)
                runner_mpnn.run()


if __name__ == "__main__":
    main()
