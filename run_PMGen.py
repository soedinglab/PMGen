import argparse
import pandas as pd
from run_utils import (run_PMGen_wrapper, run_PMGen_modeling, protein_mpnn_wrapper, bioemu_assertions,
                       MultipleAnchors, get_best_structres, retrieve_anchors_and_fixed_positions, assert_iterative_mode,
                       collect_generated_binders, create_new_input_and_fixed_positions, create_fixed_positions_if_given,
                       mutation_screen)
import shutil
from Bio import SeqIO
import warnings
import subprocess
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
    allowed_mpnn_models = [i.replace('.pt','') for i in os.listdir('ProteinMPNN/vanilla_model_weights/')]
    parser = argparse.ArgumentParser(description="Run PMGen wrapper or modeling.")

    # Default settings
    parser.add_argument('--run', choices=['parallel', 'single'], required=False, default='parallel',
                        help='Run mode: parallel or single (one by one). parallel only works if --mode wrapper')
    parser.add_argument('--mode', choices=['wrapper', 'modeling'], required=False, default='wrapper',
                        help=' wrapper mode Recommended. wrapper for multiple sample prediction, modeling for single sample prediction')

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
    parser.add_argument('--best_n_templates', type=int, default=4, help='Best N templates')
    parser.add_argument('--n_homology_models', type=int, default=1, help='Number of homology models')
    parser.add_argument('--max_ram', type=int, default=3, help='Maximum RAM GB per job (only for parallel mode)')
    parser.add_argument('--max_cores', type=int, default=4, help='Maximum number of CPU cores (only for parallel mode)')
    parser.add_argument('--dirty_mode', action='store_true')
    parser.add_argument('--initial_guess', action='store_true', help='Activates Faster and more accurate AF initial Guess mode instead of Homology modelling mode')
    parser.add_argument('--verbose', action='store_true')

    # Wrapper mode argument
    parser.add_argument('--df', type=str, help='Recommended. Path to input TSV file (required for wrapper mode)'
                                               'check the github documentation for more information.')
    parser.add_argument('--multiple_anchors', action='store_true', help='If enabled, not the best anchor, but all predicted anchors will '
                                                                      'be used for separate predictions and the best ones will be reported. default False.')
    parser.add_argument('--top_k', type=int, default=3, help='if --multiple_anchors is True, number of top anchors to predict the structures for.')
    parser.add_argument('--best_structures', action='store_true', help='If multiple anchors or multiple models are being used'
                                                                       'Activating this flag will find the best structures from their core'
                                                                       'predicted lddt and saves them in a --output_dir/best_structures'
                                                                       'directory.')

    # ProteinMPNN Arguments
    parser.add_argument('--peptide_design', action='store_true', help='Enables peptide design.')
    parser.add_argument('--only_pseudo_sequence_design', action='store_true', help='Enables MHC pseudo-sequence design.')
    parser.add_argument('--mhc_design', action='store_true', help='Enables whole mhc design. we recommend to use only_pseudo_sequence_design mode.')
    parser.add_argument('--num_sequences_peptide', type=int, default=10, help='Number of peptide sequences to be generated. Works only with --peptide_design')
    parser.add_argument('--num_sequences_mhc', type=int, default=10, help='Number of mhc sequences to be generated. Works only with --only_pseudo_sequence_design or --mhc_design')
    parser.add_argument('--sampling_temp', type=float, default=1.5, help='ProteinMPNN sampling temperature.')
    parser.add_argument('--batch_size', type=int, default=1, help='ProteinMPNN batch size.')
    parser.add_argument('--hot_spot_thr', type=float, default=6.0, help='Distance threshold to peptide, to define hot-spots on mhc.')
    parser.add_argument('--binder_pred', action='store_true', help='Enables binder prediction from ProteinMPNN generated peptides. It then extracts and reports the best binders.')
    parser.add_argument("--fix_anchors", action='store_true', help='If set, does not design anchor positions in peptide generation')
    parser.add_argument("--peptide_random_fix_fraction", type=float, default=0., help="Disables design for a random fraction of amino acids in peptide")
    parser.add_argument('--fixed_positions_given', action='store_true', help="Optional, If enabled, it uses the fixed positions given by user in --df."
                                                                             "Fixed positions should be provided as a list for each row in --df, and the columnname should be"
                                                                             "'fixed_positions'. It will automaticallly enables --fix_anchors as well, but uses fixed positions"
                                                                             "given to it only.")
    parser.add_argument("--proteinmpnn_model_name", type=str, default="v_48_020_soft_ft", help=f"ProteinMPNN model name. Allowed values: {allowed_mpnn_models}")

    # BioEmu Argumetns
    parser.add_argument('--run_bioemu', action='store_true', help='Enables bioemu pMHC sampling.')
    parser.add_argument('--bioemu_num_samples', type=int, default=100, help='Sampling rounds in bioemu. You might get lower number of structures'
                                                                           'if --filter_samples is active')
    parser.add_argument('--bioemu_batch_size_100', type=int, default=10, help='Batch size you use for a sequence of length 100. The batch size '
                                                                              'will be calculated from this, assuming that the memory requirement to compute '
                                                                              'each sample scales quadratically with the sequence length.')
    parser.add_argument('--bioemu_filter_samples', action='store_true', help='Filter out unphysical samples with e.g. long bond distances or steric clashes.')
    parser.add_argument('--bioemu_run_on_iter', type=int, default=None, help='Optional, only works when iterative_peptide_gen > 0. Runs bioemu on the structure taken'
                                                                             'from the iteration number given by user. If not set, runs on the 0 iteration by default.')

    # Mutation screen Arguments
    parser.add_argument('--mutation_screen', action='store_true', help='samples <n_mutations> point mutations over each structure.'
                                                                       'e.g, if n_mutations==1, and peptide length is 9, samples 9 mutations '
                                                                       'one over each position using structure aware peptide selection pipeline.')
    parser.add_argument('--n_mutations', type=int, default=1, help='number of mutations in a single run on a single peptide in mutation_screen.')


    # Setting to Run only a part:
    parser.add_argument('--no_alphafold', action='store_false', help='does not run alphafold.')
    parser.add_argument('--only_protein_mpnn', action='store_true', help='Skips PANDORA and AF modeling, and runs ProteinMPNN for already available predictions.')
    parser.add_argument('--no_pandora', action='store_false', help='does not run pandora')
    parser.add_argument('--protein_mpnn_dryrun', action='store_true', help='Overwrites all proteinMPNN flags and just dry run. hotspots are saved.')
    parser.add_argument('--return_all_outputs', action='store_true', help='If active, returns all alphafold outputs')
    parser.add_argument('--no_protein_mpnn', action='store_true', help='If enabled, skips ProteinMPNN. Useful when want to run bioemu.')
    parser.add_argument('--only_collect_generated_binders', action='store_true', help='Use it only if you have a previous run with generated binders and you do not want'
                                                                                      'to run proteinmpnn again, and instead you want to predict and collect binders using BA predictions.'
                                                                                      'Overwrites all proteinmpnn flags.')
    parser.add_argument('--only_mutation_screen', action='store_true')

    # Setting to run iterative peptide generation
    parser.add_argument('--iterative_peptide_gen', type=int, default=0, help='If used, the iterative peptide generation is performed, defines the number of iterations.')

    args = parser.parse_args()
    assert(args.proteinmpnn_model_name) in allowed_mpnn_models, f"Allowed models: {allowed_mpnn_models}"
    bioemu_assertions(args)
    for iteration in range(args.iterative_peptide_gen + 1):
        if iteration == 0:
            fixed_positions_path = create_fixed_positions_if_given(args)
            if args.fixed_positions_given:
                args.fix_anchors = True
        else: fixed_positions_path = None
        # if we have entered the iterative generation mode
        if args.iterative_peptide_gen > 0:
            print(f"***** Entering Iterative Peptide Generation Mode: Iter {iteration} *****")
            assert_iterative_mode(args)
            if f"iter_{iteration-1}" in args.output_dir.split("/") or f"iter_{iteration-1}/" in args.output_dir.split("/"):
                args.output_dir = "/".join(args.output_dir.split("/")[:-1]) # handle outputdir/iter_0/iter_1 issue
            args.output_dir = os.path.join(args.output_dir, f"iter_{iteration}")
            os.makedirs(args.output_dir, exist_ok=True)
            input_path_iter = os.path.join(args.output_dir, f"input_df_{iteration}.tsv")
            if iteration == 0:
                shutil.copy(args.df, input_path_iter)
            elif iteration > 0:
                if collected_generated_binders_path: # to replace inputs of new iter with outputs of last iter
                    if args.fix_anchors:
                        prev_outpath = os.path.join("/".join(args.output_dir.split("/")[:-1]), f"iter_{iteration-1}")
                        fixed_positions_path = os.path.join(prev_outpath, "fixed_positions.tsv")
                    else: fixed_positions_path = None
                    print(f"#################### {iteration}: collected_generated_binders_path: {collected_generated_binders_path} \n"
                          f"fixed_positions_path: {fixed_positions_path}")
                    create_new_input_and_fixed_positions(args=args,
                                                         best_generated_peptides_path=collected_generated_binders_path,
                                                         iter=iteration,
                                                         fixed_positions_path=fixed_positions_path)
                else:
                    raise ValueError(f"collected_generated_binders_path does not exist, so iteration {iteration} can not happen:"
                                     f"\n Debugging message: args.fix_anchors: {args.fix_anchors}")
            args.df = input_path_iter

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
            AMINO_ACIDS = set('ARNDCEQGHILKMFPSTWYV/')
            try:
                df = pd.read_csv(args.df, sep='\t')
                print(df)
                _ = df['mhc_seq']
            except:
                df = pd.read_csv(args.df)
                print(df)
                _ = df['mhc_seq']


            df['mhc_seq'] = [''.join([aa.upper() for aa in seq if aa.upper() in AMINO_ACIDS]) for seq in df['mhc_seq'].tolist()]  # remove gaps from df:
            if args.multiple_anchors:
                L1 = len(df)
                ma = MultipleAnchors(args, args.dirty_mode)
                df = ma.process()
                L2 = len(df)
                print(f'New DataFrame has {L2} rows, {L2 - L1} difference with original df')


            # outputs for proteinmpnn later
            tmp_pdb_dict = {}
            for i, row in df.iterrows():
                out_alphafold = os.path.join(args.output_dir, 'alphafold', row['id'])
                tmp_pdb_dict[row['id']] = out_alphafold

            runner = run_PMGen_wrapper(df=df, output_dir=args.output_dir, num_templates=args.num_templates,
                                           num_recycles=args.num_recycles, models=args.models,
                                           alphafold_param_folder=args.alphafold_param_folder,
                                           fine_tuned_model_path=args.fine_tuned_model_path,
                                           benchmark=args.benchmark, best_n_templates=args.best_n_templates,
                                           n_homology_models=args.n_homology_models, pandora_force_run=args.no_pandora,
                                            no_modelling=args.initial_guess, return_all_outputs=args.return_all_outputs)
            if args.run == 'parallel' and not args.only_protein_mpnn and not args.only_mutation_screen:
                runner.run_wrapper_parallel(max_ram=args.max_ram, max_cores=args.max_cores, run_alphafold=args.no_alphafold)
            elif args.run == 'single' and not args.only_protein_mpnn and not args.only_mutation_screen:
                runner.run_wrapper(run_alphafold=args.no_alphafold)
            else:
                print('--Warning!-- Only ProteinMPNN mode, Alphafold and PANDORA are skipped.')
            # check outputs if they exist:
            output_pdbs_dict = {}
            for key, value in tmp_pdb_dict.items():
                # {'id':[out1, out2], 'id2':[out1, out2], ...}
                output_pdbs_dict[key] = [os.path.join(value, i) for i in os.listdir(value) if i.endswith('.pdb') and
                                         'model_' in i and not i.endswith('.npy')]
            if args.best_structures: # get best structure out of multiple models and multiple predicted anchors:
                _ = get_best_structres(args.output_dir, df, args.multiple_anchors)
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
            runner = run_PMGen_modeling(peptide=args.peptide, mhc_seq=sequence, mhc_type=args.mhc_type,
                                            id=args.id, output_dir=args.output_dir, anchors=args.anchors,
                                            mhc_allele=args.mhc_allele, predict_anchor=args.predict_anchor,
                                            num_templates=args.num_templates, num_recycles=args.num_recycles,
                                            models=args.models, alphafold_param_folder=args.alphafold_param_folder,
                                            fine_tuned_model_path=args.fine_tuned_model_path,
                                            benchmark=args.benchmark, best_n_templates=args.best_n_templates,
                                            n_homology_models=args.n_homology_models,
                                            pandora_force_run=args.no_pandora,
                                            return_all_outputs=args.return_all_outputs)
            if not args.only_protein_mpnn and not args.only_mutation_screen:
                runner.run_PMGen(run_alphafold=args.no_alphafold)
            else:
                print('Only ProteinMPNN mode, Alphafold and PANDORA are skipped.')
            output_pdbs_dict = {}
            out_alphafold = os.path.join(args.output_dir, 'alphafold', args.id)
            output_pdbs_dict[args.id] = [os.path.join(out_alphafold, i) for i in os.listdir(out_alphafold) if
                                         i.endswith('.pdb') and 'model_' in i and not i.endswith('.npy')]
            # {'id':[output1, output2, ...]}

        if not args.only_protein_mpnn and not args.only_mutation_screen:
            print("Alphafold Runs completed.")
        else:
            print('Alphafold Runs Skipped!')
        # get the pdb outputs for listing them and protmpnn

        if args.fix_anchors:
            anchor_and_peptide = retrieve_anchors_and_fixed_positions(args,
                                  peptide_random_fix_fraction=args.peptide_random_fix_fraction,
                                  fixed_positions_path=fixed_positions_path)
        else:
            anchor_and_peptide = None
        if not args.no_protein_mpnn:
            if args.peptide_design or args.only_pseudo_sequence_design or args.mhc_design or args.protein_mpnn_dryrun or args.only_collect_generated_binders:
                if args.protein_mpnn_dryrun:
                    args.peptide_design = False
                    args.only_pseudo_sequence_design = False
                    args.mhc_design = False
                    print("Running ProteinMPNN Dry-Run")
                if args.only_collect_generated_binders:
                    args.peptide_design = False
                    args.only_pseudo_sequence_design = False
                    args.mhc_design = False
                    args.binder_pred = True
                    print("Predicting Only Binders")
                print("### Start ProteinMPNN runs ###")
                print('files:\n', output_pdbs_dict)
                protein_mpnn_wrapper(output_pdbs_dict, args, args.max_cores, anchor_and_peptide=anchor_and_peptide, mode=args.run)
                if (args.peptide_design and args.binder_pred and args.mode == "wrapper") or args.only_collect_generated_binders:
                    print("Collecting the best binders")
                    collected_generated_binders_path = collect_generated_binders(args, df, iteration)

    if args.mutation_screen:
        print('**Mutation Screen Initiating**')
        assert os.path.join(args.output_dir, 'alphafold'), f'alphafold directory does not exist!'
        mt = mutation_screen(args, df)
        mt.run_mutation_screen()


    if args.run_bioemu:
        print('**BioEmu runs initiating**')

        output_dir = args.output_dir
        bioemu_input_df_path = args.df

        if args.iterative_peptide_gen > 0: #iterative mode --> which iteration to run bioemu on? only one can be used.
            bioemu_run_on_iter = 0
            if args.bioemu_run_on_iter:
                assert args.bioemu_run_on_iter <= args.iterative_peptide_gen, f'Please make sure --iterative_peptide_gen is less or equal to  --iterative_peptide_gen'
                bioemu_run_on_iter = args.bioemu_run_on_iter
            output_dir = os.path.join("/".join(args.output_dir.split("/")[:-1]), f"iter_{bioemu_run_on_iter}") # outputdir/
            bioemu_input_df_path = os.path.join(output_dir, f'input_df_{bioemu_run_on_iter}.tsv')

        bioemu_output_dir = os.path.join(output_dir, 'bioemu')
        cache_embeds_dir = os.path.join(output_dir, 'alphafold')
        os.makedirs(bioemu_output_dir, exist_ok=True)
        assert os.path.isdir(cache_embeds_dir), f'{cache_embeds_dir} not found'
        assert os.path.isdir(bioemu_output_dir), f'{bioemu_output_dir} not found'
        assert os.path.isfile(bioemu_input_df_path), f'{bioemu_input_df_path} not found'
        print(f'Running on cache_embeds_dir: {cache_embeds_dir}')

        bioemu_df = pd.read_csv(bioemu_input_df_path, sep='\t')
        for i, row in bioemu_df.iterrows():
            sequence = str(row['mhc_seq'].replace('/','') + row['peptide'])
            id = str(row['id'])
            bioemu_output_dir_id = os.path.join(bioemu_output_dir, id)
            cmd = ['./run_bioemu.sh', '--sequence', sequence, '--id', id, '--output_dir', bioemu_output_dir_id,
                   '--cache_embeds_dir', cache_embeds_dir, '--bioemu_num_samples', str(args.bioemu_num_samples),
                   '--bioemu_batch_size_100', str(args.bioemu_batch_size_100)]
            cmd = cmd + ['--bioemu_filter_samples'] if args.bioemu_filter_samples else cmd
            print(f"🌀 Starting iteration {i}: {cmd}")
            try:
                result = subprocess.run(cmd, check=True)
                print(f"✅ Bioemurun-> Iteration {i}, id {id}, outpath {bioemu_output_dir_id} completed successfully\n")
            except subprocess.CalledProcessError as e:
                print(f"❌ Bioemurun-> Iteration {i}, id {id}, outpath {bioemu_output_dir_id} failed with error code {e.returncode}")
                # Optional: stop if a command fails


if __name__ == "__main__":
    main()
