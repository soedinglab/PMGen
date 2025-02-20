import argparse
import pandas as pd
from run_utils import run_parsefold_wrapper, run_parsefold_modeling
from Bio import SeqIO
import warnings
from Bio import BiopythonDeprecationWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)



def main():
    parser = argparse.ArgumentParser(description="Run ParseFold wrapper or modeling.")

    # Default settings
    parser.add_argument('--run', choices=['parallel', 'single'], required=False, default='single',
                        help='Run mode: parallel or single (one by one). parallel only works if --mode wrapper')
    parser.add_argument('--mode', choices=['wrapper', 'modeling'], required=False, default='modeling',
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
    parser.add_argument('--models', nargs='+', default=['model_2_ptm'], help='List of models to use')
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
        df = pd.read_csv(args.df, sep='\t').iloc[:2, :]
        runner = run_parsefold_wrapper(df=df, output_dir=args.output_dir, num_templates=args.num_templates,
                                       num_recycles=args.num_recycles, models=args.models,
                                       alphafold_param_folder=args.alphafold_param_folder,
                                       fine_tuned_model_path=args.fine_tuned_model_path,
                                       benchmark=args.benchmark, best_n_templates=args.best_n_templates,
                                       n_homology_models=args.n_homology_models)
        if args.run == 'parallel':
            runner.run_wrapper_parallel(max_ram=args.max_ram, max_cores=args.max_cores)
        else:
            runner.run_wrapper()

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
        runner.run_parsefold()

    print("Run completed.")


if __name__ == "__main__":
    main()
