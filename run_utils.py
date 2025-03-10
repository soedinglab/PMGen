import psutil
import numpy as np
import concurrent.futures
import sys
import contextlib
sys.path.append("PANDORA")
import os
import json
import shutil
from PANDORA import Target
from PANDORA import Pandora
from PANDORA import Database
import glob
from utils import processing_functions
import pandas as pd
import subprocess
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



class run_PMGen_modeling():
    def __init__(self, peptide, mhc_seq, mhc_type, id, output_dir='output',
                  anchors=None, mhc_allele=None, predict_anchor=True,
                 num_templates=4, num_recycles=3, models=['model_2_ptm'],
                 alphafold_param_folder = 'AFfine/af_params/params_original/',
                 fine_tuned_model_path='AFfine/af_params/params_finetune/params/model_ft_mhc_20640.pkl',
                 benchmark=False, n_homology_models=1, best_n_templates=1,
                 pandora_force_run=True):
        """
        Initializes the PMGen modeling pipeline.

        Args:
            peptide (str): Peptide sequence.
            mhc_seq (str): MHC sequence(s) (single for MHC-I, two chains for MHC-II).
            mhc_type (int): MHC class (1 or 2).
            id (str): Unique identifier for the run.
            output_dir (str): Directory for output files. Default is 'output'.
            anchors (list or tuple, optional): Anchor positions for MHC binding.
            mhc_allele (str, optional): Specific MHC allele name.
            predict_anchor (bool): Whether to predict anchor residues. Default is True.
            num_templates (int): Number of templates used in modeling.
            num_recycles (int): Number of AlphaFold recycling iterations.
            models (list): List of AlphaFold models to use.
            alphafold_param_folder (str): Path to AlphaFold parameter files.
            fine_tuned_model_path (str): Path to fine-tuned AlphaFold model.
            benchmark (bool): Use different allele compared to the actual allele. make sure the id shouldbe pdb id.
            n_homology_models (int): number of initial peptide homology models to generate by modeller, default=1.
            best_n_templates (int): number of found templates used for homology modeling via modeler, default=1.
            pandora_force_run (bool): Weather to force run pandora or not, default=True.
        """
        super().__init__()
        self.peptide = peptide
        self.mhc_seq = mhc_seq
        self.mhc_type = mhc_type
        self.output_dir = output_dir
        self.anchors = anchors
        self.mhc_allele = mhc_allele
        self.predict_anchor = predict_anchor
        self.id = id
        self.num_templates = num_templates
        self.num_recycles = num_recycles
        self.models = models
        self.alphafold_param_folder = alphafold_param_folder
        self.fine_tuned_model_path = fine_tuned_model_path
        self.benchmark = benchmark
        self.n_homology_models = n_homology_models
        self.best_n_templates = best_n_templates
        self.pandora_force_run = pandora_force_run
        self.input_assertion()
        if len(self.models) > 1:
            print(f'\n #### Warning! You are running for multiple models {self.models}'
                  f'Please make sure your model names are correct.'
                  f'For fine-tuned models please use "_ft" as a identifier in model param '
                  f'.pkl file ####\n')
        # input derived args
        self.mhc_type_greek='I' if self.mhc_type==1 else 'II'
        self.m_chain, self.n_chain = (self.mhc_seq + '/').split('/')[0], (self.mhc_seq + '/').split('/')[1]
        self.pandora_output = os.path.join(self.output_dir, 'pandora')
        self.db = Database.load() # load pandora db
        self.alignment_output = os.path.join(self.output_dir, 'alignment')
        self.alphafold_out = os.path.join(self.output_dir, 'alphafold')
        self.alphafold_input_file = os.path.join(self.alphafold_out, self.id, f'alphafold_input_file.tsv')
        # vars defined later
        self.template_id = None

    def run_PMGen(self, test_mode=False, run_alphafold=True):
        """
        Runs the full PMGen pipeline, including Pandora alignment and AlphaFold prediction.
        Args:
            test_mode (bool): If True, runs in test mode without full execution.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        self.template_id = self.run_pandora(self.pandora_force_run)
        #pandora_template_path = os.path.join(self.pandora_output, self.id, self.template_id)
        #aln_output_file = os.path.join(self.alignment_output, self.id + 'no_pep.tsv')
        ## create aln files for non-pep
        #df_aln_nopep = self.alignment_without_peptide(self.template_id, aln_output_file, pandora_template_path)
        ## create aln files for with-pep
        # list of pandora generated template paths
        pdb_files = processing_functions.rename_files(os.path.join(self.pandora_output, self.id), self.num_templates)
        mhc_pep_seq = self.mhc_seq + '/' + self.peptide
        aln_output_file = os.path.join(self.alignment_output, self.id + '_with_pep.tsv')
        _ = self.alignment_with_peptide(pdb_files, mhc_pep_seq, output_path=aln_output_file)
        ## Prepare Alphafold Fine Input files
        os.makedirs(self.alphafold_out + f'/{self.id}', exist_ok=True)
        self.alphafold_preparation(template_aln_file=aln_output_file, mhc_pep_seq=mhc_pep_seq, output=self.alphafold_input_file)
        self.output_pdbs_dict = {}
        if run_alphafold:
            print('## To run Alphafold Please Make Sure GPU is Available and can be found ##')
            self.run_alphafold(input_file=self.alphafold_input_file, output_prefix=self.alphafold_out + '/')
            # get the paths for proteinmpnn
            self.output_pdbs_dict[self.id] = [os.path.join(self.alphafold_out, i) for i in os.listdir(self.alphafold_out) if i.endswith('.pdb') and 'model_' in i and not i.endswith('.npy')]

    def run_pandora(self, force_run=True):
        """
        Runs the Pandora module to generate template structures for MHC modeling.

        Returns:
            str: Template ID used in the modeling process.
        """
        os.makedirs(self.pandora_output, exist_ok=True)
        log_file = f'{self.pandora_output}/{self.id}/pandora.log'
        mhc_allele = [] if not self.mhc_allele else [self.mhc_allele]
        anchor = [] if self.anchors is None else self.anchors
        # Redirect stdout and stderr to the log file
        os.makedirs(self.pandora_output + '/' + self.id, exist_ok=True)
        shoud_I_run = 'Yes'
        if not force_run:
            # check if mod*.pdb and template file exists
            try:
                files = [file for file in glob.glob(os.path.join(self.pandora_output, self.id, '????.pdb')) if
                         "mod" not in file.split("/")[-1]]
                if files:
                    template_id = files[0].split("/")[-1]
                models = [file for file in glob.glob(os.path.join(self.pandora_output, self.id, '????.pdb')) if
                         "mod" in file.split("/")[-1]]
                if len(models) >= self.num_templates:
                    shoud_I_run = 'No'
                print(f'Mode force_run == {force_run}, and PANDORA has already finished for {self.id}, no need to run it!')
            except:
                shoud_I_run = 'Yes'

        if shoud_I_run == 'Yes':
            with open(log_file, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                try:
                    print(f"Starting {self.id} initialization...")
                    target = Target(id=self.id, peptide=self.peptide, allele_type=mhc_allele,
                                    MHC_class=self.mhc_type_greek, M_chain_seq=self.m_chain,
                                    N_chain_seq=self.n_chain, output_dir=self.pandora_output,
                                    use_netmhcpan=self.predict_anchor, anchors=anchor)
                    case = Pandora.Pandora(target, self.db)
                    case.model(n_loop_models=self.num_templates, benchmark=self.benchmark,
                               n_homology_models=self.n_homology_models,
                               best_n_templates=self.best_n_templates)
                    print("Pandora modeling completed successfully.")
                except Exception as e:
                    print(f"❌ An error occurred during template engineering {self.id}: {str(e)}", file=sys.stderr)
                    raise
        print("✔Pandora run completed. Check log file for details:", log_file)
        # get template id used in pandora
        files = [file for file in glob.glob(os.path.join(self.pandora_output, self.id, '????.pdb')) if
                 "mod" not in file.split("/")[-1]]
        if files:
            template_id = files[0].split("/")[-1]
            print(f"✔ {self.id} log: Template ID used for homology modeling: {template_id}")
            return template_id
        else:
            print(f"❌ {self.id} log: No template ID found.")
            return None

    def alignment_without_peptide(self, template_id, output_path, template_path,
                                       template_csv_path="data/all_templates.csv"):
        """
        Generates an alignment file for MHC without the peptide.
        Args:
            template_id (str): PDB ID used as a template.
            output_path (str): File path to save the alignment.
            template_path (str): Path to the template structure.
            template_csv_path (str): Path to the CSV file containing template sequences.
        Returns:
            pd.DataFrame: Alignment data.
        """
        os.makedirs(self.alignment_output, exist_ok=True)
        df = processing_functions.prepare_alignment_file_without_peptide(template_id=template_id,
                                                                    mhc_seq=self.mhc_seq,
                                                                    mhc_type=self.mhc_type,
                                                                    output_path=output_path,
                                                                    template_path=template_path,
                                                                    template_csv_path=template_csv_path)
        return df

    def alignment_with_peptide(self, pdb_files, mhc_pep_seq, output_path):
        """
        Generates an alignment file for MHC with the peptide.

        Args:
            pdb_files (list): List of PDB template file paths.
            mhc_pep_seq (str): Full MHC-peptide sequence.
            output_path (str): File path to save the alignment.

        Returns:
            pd.DataFrame: Alignment data.
        """
        os.makedirs(self.alignment_output, exist_ok=True)
        DF = []
        for pdb_file in pdb_files:
            df = processing_functions.prepare_alignment_file_with_peptide(pdb_file=pdb_file,
                                                                     target_seq=mhc_pep_seq,
                                                                     mhc_type=self.mhc_type,
                                                                     output_path=None,
                                                                     peptide=True)
            DF.append(df)
        DF = pd.concat(DF, ignore_index=True)
        DF.to_csv(output_path, sep='\t', index=False)
        return DF

    def alphafold_preparation(self, template_aln_file, mhc_pep_seq, output):
        """
        Prepares input files for AlphaFold.

        Args:
            template_aln_file (str): Path to the template alignment file.
            mhc_pep_seq (str): Full MHC-peptide sequence.
            output (str): Path to save the prepared input file.
        """
        df = pd.DataFrame({"target_chainseq": [mhc_pep_seq],
                           "templates_alignfile": [template_aln_file],
                           "targetid": [self.id]})
        df.to_csv(output, sep='\t', index=False)

    def run_alphafold(self, input_file, output_prefix):
        """
        Runs AlphaFold with the specified input and parameters.

        Args:
            input_file (str): Path to the input file.
            output_prefix (str): Prefix for output files.
        """
        model_params_files = ''
        model_names = ''
        for model in self.models:
            i = 'classic' if '_ft' not in model else self.fine_tuned_model_path
            model_params_files += f'{i} '
            model_names += f'{model} '

        model_names = model_names.rstrip()
        model_params_files = model_params_files.rstrip()
        command = [
            "python", "AFfine/run_prediction.py",
            "--targets", f"{input_file}",
            "--data_dir", f"{self.alphafold_param_folder}",
            "--outfile_prefix", f"{output_prefix}",
            "--model_names", *model_names.split(),
            "--model_params_files", *model_params_files.split(),
            "--ignore_identities",
            "--num_recycles", f"{self.num_recycles}"
        ]

        print(command)
        try:
            # Run the command with unbuffered output
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

            print("### Running AFfine/run_prediction.py ###", flush=True)

            # Print stdout and stderr in real-time
            while True:
                output = process.stdout.readline()
                if output:
                    print(output, end="", flush=True)
                elif process.poll() is not None:
                    break  # Process has finished

            # Capture remaining stderr output
            for err_line in process.stderr:
                print(err_line, end="", flush=True)

            exit_code = process.wait()
            if exit_code == 0:
                print("\n✔ Alphafold executed successfully!", flush=True)
            else:
                self.error_handling()
                print("\n❌ Command failed with exit code:", exit_code, flush=True)


        except Exception as e:
            self.error_handling()
            print("\n❌ Error running command:", str(e), flush=True)

    def error_handling(self, errotype='affine'):
        if errotype == 'affine':
            print('### ERROR MESSAGE ###:'
                  'Alphafold Run Failed, It is common, please follow the debugging steps below: \n'
                  '1- check if you have ptxas: "which ptxas" if not do:\n'
                  'conda install -c nvidia cuda-nvcc\n'
                  'or\n'
                  'export PATH=/usr/local/cuda/bin:$PATH\n'
                  'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH')

    def input_assertion(self):
        assert isinstance(self.peptide, str), f"peptide must be a string, found: {self.peptide}"
        assert self.mhc_type in [1, 2], f"mhc_type must be an integer value of 1 or 2, found: {self.peptide}"
        assert isinstance(self.mhc_seq, str), f"mhc_seq must be a string, found: {self.mhc_seq}"
        assert isinstance(self.output_dir, str), f"output_dir must be a string, found: {self.output_dir}"
        if self.anchors:
            assert isinstance(self.anchors, (tuple, list)), (f"anchors must be a tuple or list, found: {self.anchors}"
                                                             f"alternatively use predict_anchor==True")
            self.anchors = list(self.anchors)
        else:
            assert self.predict_anchor==True, f'If anchors arg is empty, please set predict_anchor=True'
        if self.mhc_allele is not None:
            assert isinstance(self.mhc_allele, str), (f'mhc_allele should be string, found: {self.mhc_allele}'
                                                      f'\n example for MHC-I: HLA-B40:02'
                                                      f'\n example for MHC-II: HLA-DRA01;HLA-DRB11')
        assert isinstance(self.predict_anchor, bool), f'predict_anchor must be a bool, found:{self.predict_anchor}'
        if self.mhc_type==2:
            assert len(self.mhc_seq.split('/')) == 2, (f'mhc_seq for mhc_type==2 should be two '
                                             f'sequences separated by "/", found: {self.mhc_seq}'
                                             f'\n please follow the order "M-chain/Nchain"')
        elif self.mhc_type==1:
            assert len(self.mhc_seq.split('/')) == 1, (f'mhc_seq for mhc_type==1 should be one '
                                             f'sequence without "/" as seperator: {self.mhc_seq}')
        assert isinstance(self.id, str), f'id must be a string, found: {self.id}'
        assert isinstance(self.num_templates, int), f'num_templates must be an integer, found {self.num_templates}'
        assert isinstance(self.num_recycles, int), f'num_recycles must be an integer, found {self.num_recycles}'
        assert isinstance(self.models, list), f'models must be a list, found {self.models}'
        assert isinstance(self.fine_tuned_model_path, str), f'fine_tuned_model_path must be a string, found {self.fine_tuned_model_path}'
        assert isinstance(self.alphafold_param_folder, str), f'alphafold_param_folder must be a string, found {self.alphafold_param_folder}'


class run_PMGen_wrapper():
    def __init__(self, df, output_dir, num_templates=4, num_recycles=3, models=['model_2_ptm'],
                 alphafold_param_folder='AFfine/af_params/params_original/',
                 fine_tuned_model_path='AFfine/af_params/params_finetune/params/model_ft_mhc_20640.pkl',
                 max_ram_per_job=3, num_cpu=1, benchmark=False, best_n_templates=1, n_homology_models=1,
                 pandora_force_run=True):
        """
        Initializes the run_PMGen_wrapper class.
        :param df: pandas DataFrame containing input data. Required columns:
            - 'peptide' (str): Peptide sequence.
            - 'mhc_seq' (str): MHC sequence (one chain for MHC-I, two for MHC-II).
            - 'mhc_type' (int): Type of MHC (1 for MHC-I, 2 for MHC-II).
            - 'anchors' (str or NaN): Two numbers (MHC-I) or four numbers (MHC-II) separated by ";". If not provided, anchors will be predicted.
            - 'id' (str): Unique identifier for each row.
        :param output_dir: str, path to the output directory. This directory will be created if it does not exist.
        :param num_templates: int, number of structural templates to use (default=4).
        :param num_recycles: int, number of recycles in AlphaFold inference (default=3).
        :param models: list of str, names of AlphaFold models to use (default=['model_2_ptm']).
        :param alphafold_param_folder: str, path to the folder containing original AlphaFold model parameters.
            - Must be an existing directory.
        :param fine_tuned_model_path: str, path to the fine-tuned AlphaFold model parameters.
            - Must be an existing file.
        :param max_ram_per_job: int, maximum RAM (in GB) per parallel process (default=3).
        :param num_cpu: int, number of CPU cores to use (default=1).
        :param benchmark: bool, to do becnhmarking.
        :param best_n_templates: int, how many models to used for homology modeling after sequence aln search, default=1.
        :param n_homology_models: int, number of initial models to be done with modeller homology modelling, default=1.
        The function `input_assertion()` checks if all inputs are correctly formatted and whether required files and directories exist.
        Raises:
            - AssertionError if any input is invalid.
        """
        self.df = df
        self.output_dir = output_dir
        self.num_templates = num_templates
        self.num_recycles = num_recycles
        self.models = models
        self.alphafold_param_folder = alphafold_param_folder
        self.fine_tuned_model_path = fine_tuned_model_path
        self.max_ram_per_job = max_ram_per_job
        self.num_cpu = num_cpu
        self.benchmark = benchmark
        self.best_n_templates = best_n_templates
        self.n_homology_models = n_homology_models
        self.pandora_force_run = pandora_force_run
        self.input_assertion()

    def run_wrapper(self, run_alphafold=True):
        INPUT_DF = []
        for step, row in self.df.iterrows():
            anchors = [int(r) for r in row['anchors'].split(';')] if isinstance(row['anchors'], str) and ';' in row['anchors'] else None
            try:
                mhc_allele = row['mhc_allele'] if row['mhc_allele'] else None
            except:
                mhc_allele = None
            predict_anchor = False if anchors else True
            runner = run_PMGen_modeling(peptide=row['peptide'], mhc_seq=row['mhc_seq'],
                                           mhc_type=row['mhc_type'], id=f"{row['id']}", output_dir=self.output_dir,
                                            anchors=anchors, mhc_allele=mhc_allele, predict_anchor=predict_anchor,
                                            num_templates=self.num_templates, num_recycles=self.num_recycles,
                                            models=self.models, alphafold_param_folder=self.alphafold_param_folder,
                                            fine_tuned_model_path=self.fine_tuned_model_path, benchmark=self.benchmark,
                                            n_homology_models=self.n_homology_models, best_n_templates=self.best_n_templates,
                                            pandora_force_run=self.pandora_force_run)
            runner.run_PMGen(run_alphafold=False)
            input_df = pd.read_csv(runner.alphafold_input_file, sep='\t', header=0)
            input_df['targetid'] = [str(row['id']) + '/' + str(row['id'])] # id/id
            INPUT_DF.append(input_df)
        if run_alphafold:
            alphafold_out = self.output_dir + '/alphafold'
            pd.concat(INPUT_DF).to_csv(f'{alphafold_out}/alphafold_input_file.tsv', sep='\t', index=False)
            runner.run_alphafold(input_file=f'{alphafold_out}/alphafold_input_file.tsv', output_prefix=alphafold_out + '/')


    def get_available_memory(self):
        """ Returns available system memory in GB """
        memory = psutil.virtual_memory()
        return memory.available / (1024 ** 3)  # Convert bytes to GB


    def process_row(self, row):
        """ Process each row to generate input data for Alphafold """
        anchors = [int(r) for r in row['anchors'].split(';')] if isinstance(row['anchors'], str) and ';' in row[
            'anchors'] else None
        try:
            mhc_allele = row['mhc_allele'] if row['mhc_allele'] else None
        except:
            mhc_allele = None
        predict_anchor = False if anchors else True
        runner = run_PMGen_modeling(peptide=row['peptide'], mhc_seq=row['mhc_seq'],
                                        mhc_type=row['mhc_type'], id=f"{row['id']}", output_dir=self.output_dir,
                                        anchors=anchors, mhc_allele=mhc_allele, predict_anchor=predict_anchor,
                                        num_templates=self.num_templates, num_recycles=self.num_recycles,
                                        models=self.models, alphafold_param_folder=self.alphafold_param_folder,
                                        fine_tuned_model_path=self.fine_tuned_model_path, benchmark=self.benchmark,
                                        n_homology_models=self.n_homology_models, best_n_templates=self.best_n_templates,
                                        pandora_force_run=self.pandora_force_run)
        runner.run_PMGen(run_alphafold=False)
        input_df = pd.read_csv(runner.alphafold_input_file, sep='\t', header=0)
        input_df['targetid'] = [str(row['id']) + '/' + str(row['id'])]  # id/id
        return input_df

    def run_wrapper_parallel(self, max_ram=3, max_cores=4, run_alphafold=True):
        """
        Processes rows of input data in parallel, utilizing available system memory and CPU cores.
        It ensures that the system memory does not exceed the specified `max_ram` per job,
        divides the work among multiple processes, and then runs Alphafold on the final input file.
        Args:
            max_ram (float, optional): Maximum amount of system memory (in GB) allocated per parallel job. Default is 3 GB.
            max_cores (int, optional): Maximum number of CPU cores to use for parallel processing. Default is 4.
        """
        # List to hold the processed dataframes
        INPUT_DF = []
        # Monitor system memory to ensure it doesn't exceed max_ram
        available_memory = self.get_available_memory()
        print(f"Available memory: {available_memory} GB")
        # Calculate maximum number of jobs based on available memory and max_ram (in GB)
        max_jobs = max_cores
        if available_memory < max_ram:
            max_jobs = int(np.floor(available_memory / max_ram))  # Limit jobs based on available memory
        print(f"Max concurrent jobs based on available memory: {max_jobs}")
        # Use ProcessPoolExecutor for parallelism
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_jobs) as executor:
            # Submit tasks for each row
            futures = {executor.submit(self.process_row, row): row for _, row in self.df.iterrows()}
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    INPUT_DF.append(result)
                except Exception as e:
                    print(f"Error processing row: {e}")
        # Combine all the dataframes into one and save to file
        alphafold_out = self.output_dir + '/alphafold'
        pd.concat(INPUT_DF).to_csv(f'{alphafold_out}/alphafold_input_file.tsv', sep='\t', index=False)
        # Now, run alphafold on the final input file
        # Initialize a final runner for Alphafold
        # Run alphafold model, initialize a runner
        row = self.df.iloc[0, :]
        anchors = [int(r) for r in row['anchors'].split(';')] if isinstance(row['anchors'], str) and ';' in row[
            'anchors'] else None
        try:
            mhc_allele = row['mhc_allele'] if row['mhc_allele'] else None
        except:
            mhc_allele = None
        predict_anchor = False if anchors else True
        runner = run_PMGen_modeling(peptide=row['peptide'], mhc_seq=row['mhc_seq'],
                                            mhc_type=row['mhc_type'], id=f"{row['id']}", output_dir=self.output_dir,
                                            anchors=anchors, mhc_allele=mhc_allele, predict_anchor=predict_anchor,
                                            num_templates=self.num_templates, num_recycles=self.num_recycles,
                                            models=self.models, alphafold_param_folder=self.alphafold_param_folder,
                                            fine_tuned_model_path=self.fine_tuned_model_path)
        if run_alphafold:
            runner.run_alphafold(input_file=f'{alphafold_out}/alphafold_input_file.tsv', output_prefix=alphafold_out + '/')


    def input_assertion(self):
        """
        Validates input arguments to ensure correct data types and formats.
        """
        assert isinstance(self.df, pd.DataFrame), f"df must be a pandas DataFrame, found: {type(self.df)}"
        assert isinstance(self.output_dir,
                          str), f"output_dir must be a string (directory path), found: {type(self.output_dir)}"
        assert isinstance(self.num_templates,
                          int), f"num_templates must be an integer, found: {type(self.num_templates)}"
        assert isinstance(self.num_recycles, int), f"num_recycles must be an integer, found: {type(self.num_recycles)}"
        assert isinstance(self.models, list) and all(isinstance(m, str) for m in self.models), (
            f"models must be a list of strings, found: {self.models}"
        )
        required_columns = {'peptide', 'mhc_seq', 'mhc_type', 'anchors', 'id'}
        missing_columns = required_columns - set(self.df.columns)
        assert not missing_columns, f"df is missing required columns: {missing_columns}"
        assert self.df['peptide'].apply(lambda x: isinstance(x, str)).all(), "All peptide values must be strings."
        assert self.df['mhc_seq'].apply(lambda x: isinstance(x, str)).all(), "All mhc_seq values must be strings."
        assert self.df['mhc_type'].apply(
            lambda x: x in [1, 2]).all(), "MHC type must be either 1 (MHC-I) or 2 (MHC-II)."
        def valid_anchor_format(anchor, mhc_type):
            if pd.isna(anchor):
                return True  # Allow missing anchors (to be predicted)
            parts = anchor.split(";")
            return (mhc_type == 1 and len(parts) == 2) or (mhc_type == 2 and len(parts) == 4)
        assert self.df.apply(lambda row: valid_anchor_format(row['anchors'], row['mhc_type']), axis=1).all(), (
            "Anchors must be two numbers separated by ';' for MHC-I and four for MHC-II."
        )
        assert isinstance(self.alphafold_param_folder,
                          str), f"alphafold_param_folder must be a string, found: {type(self.alphafold_param_folder)}"
        assert os.path.isdir(
            self.alphafold_param_folder), f"alphafold_param_folder does not exist or is not a directory: {self.alphafold_param_folder}"
        assert isinstance(self.fine_tuned_model_path,
                          str), f"fine_tuned_model_path must be a string, found: {type(self.fine_tuned_model_path)}"
        assert os.path.isfile(
            self.fine_tuned_model_path), f"fine_tuned_model_path does not exist or is not a file: {self.fine_tuned_model_path}"






class run_proteinmpnn():
    def __init__(self, PMGen_pdb, output_dir,
                 num_sequences_peptide=10, num_sequences_mhc=3,
                peptide_chain='P', mhc_design=True, peptide_design=True,
                 only_pseudo_sequence_design=True, anchor_pred=True,
                 sampling_temp=0.05, batch_size=1, hot_spot_thr=6.0,
                 save_hotspots=True):
        '''
        Args:
            PMGen_pdb: (str) Single Chain pdb path generated by PMGen AFfine.
            output_dir: (str) Output path to save files. in outpath, a proteinmpnn folder is created.
            num_sequences_peptide: (int) peptide sequences to be generated.
            num_sequences_mhc: (int) mhc sequences to be generated, for both only_pseudo_sequence_design and mhc_design setting.
            peptide_chain: (str) the chain assigned to peptide in mulichain generated structure in this class.
            mhc_design: (bool), weather design mhc or not.
            peptide_design: (bool) weather design peptide or not.
            only_pseudo_sequence_design: (bool) weather design mhc pseudoseq or not.
            anchor_pred: (bool) weather be used to calculate anchor scores or not.
            sampling_temp: (float) proteinMPNN samling temprature.
            batch_size: (int) proteinMPNN batch size.
            hot_spot_thr: (float) threshold in distance angstrom to peptide amino acids.
            save_hotspots: (bool) weather save hotspots or not.
        '''
        self.pdb = PMGen_pdb
        self.output_dir = output_dir
        self.peptide_chain = peptide_chain
        self.num_sequences_peptide = num_sequences_peptide
        self.num_sequences_mhc = num_sequences_mhc
        self.peptide_design = peptide_design
        self.mhc_design = mhc_design
        self.sampling_temp = sampling_temp
        self.batch_size = batch_size
        self.only_pseudo_sequence_design = only_pseudo_sequence_design
        self.anchor_pred = anchor_pred
        self.hot_spot_thr = hot_spot_thr
        self.save_hotspots = save_hotspots
        self.input_assertion()
        os.makedirs(self.output_dir, exist_ok=True)
        self.multichain_pdb = processing_functions.split_and_renumber_pdb(self.pdb,
                                                                          os.path.join(self.output_dir, 'multichain_pdb'),
                                                                          n=180) #multichain pdb,
        # get distance matrices vs peptide as self.chain_dict_dist
        self.chain_dict_dist = processing_functions.get_distance_matrices(input_pdb=self.multichain_pdb,
                                                                          target_chain=self.peptide_chain, atom='CB')
        self.hot_spots = processing_functions.get_hotspots(self.chain_dict_dist, thr=self.hot_spot_thr)#{mhc_chain:[n,2]--> [[1,2], [2, 10]...]}

    def run(self, thr=5., atom='CB'):
        if self.mhc_design: # redesign the whole MHC
            self.__mhc_design()
        if self.peptide_design:
            self.__peptide_design()
        if self.only_pseudo_sequence_design:
            self.__only_pseudo_sequence_design()
        if self.anchor_pred:
            pass
        if self.save_hotspots:
            output_dir = os.path.join(self.output_dir, 'hotspots.npz')
            np.savez(output_dir, **self.hot_spots)

    def __mhc_design(self):
        output_dir = os.path.join(self.output_dir, 'mhc_design')
        print(f'***** Full MHC Sequence Generation Mode Start\n saving at{output_dir}')
        os.makedirs(output_dir, exist_ok=True)
        path_for_parsed_chains = os.path.join(output_dir, "parsed_pdbs.jsonl")
        path_for_assigned_chains = os.path.join(output_dir, "assigned_pdbs.jsonl")
        chains_to_design = ' '.join(list(self.chain_dict_dist.keys()))
        # Run parse_multiple_chains.py
        subprocess.run([
            "python", "ProteinMPNN/helper_scripts/parse_multiple_chains.py",
            "--input_path", os.path.dirname(self.multichain_pdb),
            "--output_path", path_for_parsed_chains
        ], check=True)
        # Run assign_fixed_chains.py --> design chains of mhc and fix peptide chain 'P'
        subprocess.run([
            "python", "ProteinMPNN/helper_scripts/assign_fixed_chains.py",
            "--input_path", path_for_parsed_chains,
            "--output_path", path_for_assigned_chains,
            "--chain_list", chains_to_design
        ], check=True)
        # Run protein_mpnn_run.py
        subprocess.run([
            "python", "-W", "ignore", "ProteinMPNN/protein_mpnn_run.py",
            "--jsonl_path", path_for_parsed_chains,
            "--chain_id_jsonl", path_for_assigned_chains,
            "--out_folder", output_dir,
            "--num_seq_per_target", f'{self.num_sequences_mhc}',
            "--sampling_temp", f'{self.sampling_temp}',
            "--seed", "37",
            "--batch_size", f'{self.batch_size}',
            "--save_probs", "1",
            "--save_score", "1"
        ], check=True)
        print('Full MHC Sequence Generation Mode Done! *****\n')


    def __peptide_design(self):
        output_dir = os.path.join(self.output_dir, 'peptide_design')
        print(f'***** Peptide Generation Mode Start\n saving at{output_dir}')
        os.makedirs(output_dir, exist_ok=True)
        path_for_parsed_chains = os.path.join(output_dir, "parsed_pdbs.jsonl")
        path_for_assigned_chains = os.path.join(output_dir, "assigned_pdbs.jsonl")
        chains_to_design = 'P'
        # Run parse_multiple_chains.py
        subprocess.run([
            "python", "ProteinMPNN/helper_scripts/parse_multiple_chains.py",
            "--input_path", os.path.dirname(self.multichain_pdb),
            "--output_path", path_for_parsed_chains
        ], check=True)
        # Run assign_fixed_chains.py --> design chains of mhc and fix peptide chain 'P'
        subprocess.run([
            "python", "ProteinMPNN/helper_scripts/assign_fixed_chains.py",
            "--input_path", path_for_parsed_chains,
            "--output_path", path_for_assigned_chains,
            "--chain_list", chains_to_design
        ], check=True)
        # Run protein_mpnn_run.py
        subprocess.run([
            "python", "-W", "ignore", "ProteinMPNN/protein_mpnn_run.py",
            "--jsonl_path", path_for_parsed_chains,
            "--chain_id_jsonl", path_for_assigned_chains,
            "--out_folder", output_dir,
            "--num_seq_per_target", f'{self.num_sequences_peptide}',
            "--sampling_temp", f'{self.sampling_temp}',
            "--seed", "37",
            "--batch_size", f'{self.batch_size}',
            "--save_probs", "1",
            "--save_score", "1"
        ], check=True)
        print('Full MHC Sequence Generation Mode Done! *****\n')

    def __only_pseudo_sequence_design(self):
        output_dir = os.path.join(self.output_dir, 'only_pseudo_sequence_design')
        print(f'***** MHC Pseudo-Sequence Generation Mode Start\n saving at{output_dir}')
        os.makedirs(output_dir, exist_ok=True)
        path_for_parsed_chains = os.path.join(output_dir, "parsed_pdbs.jsonl")
        path_for_assigned_chains = os.path.join(output_dir, "assigned_pdbs.jsonl")
        path_for_fixed_positions = os.path.join(output_dir,"fixed_pdbs.jsonl")
        chains_to_design = ' '.join(list(self.chain_dict_dist.keys()))
        design_only_positions = ""

        for key, value in self.hot_spots.items():
            unique_val = np.unique(value[:, 1])
            design_only_positions += " ".join([str(i + 1) for i in unique_val]) + ", "
        self.design_only_positions = design_only_positions.strip(", ")
        # Run parse_multiple_chains.py
        subprocess.run([
            "python", "ProteinMPNN/helper_scripts/parse_multiple_chains.py",
            "--input_path", os.path.dirname(self.multichain_pdb),
            "--output_path", path_for_parsed_chains
        ], check=True)
        # Run assign_fixed_chains.py --> design chains of mhc and fix peptide chain 'P'
        subprocess.run([
            "python", "ProteinMPNN/helper_scripts/assign_fixed_chains.py",
            "--input_path", path_for_parsed_chains,
            "--output_path", path_for_assigned_chains,
            "--chain_list", chains_to_design
        ], check=True)
        # Run make_fixed_positions_dict.py --> specify mhc pseudo sequences at mhc chains
        subprocess.run([
            "python", "ProteinMPNN/helper_scripts/make_fixed_positions_dict.py",
            "--input_path", path_for_parsed_chains,
            "--output_path", path_for_fixed_positions,
            "--chain_list", chains_to_design,
            "--position_list", design_only_positions,
            "--specify_non_fixed"
        ])
        # Run protein_mpnn_run.py
        subprocess.run([
            "python", "-W", "ignore", "ProteinMPNN/protein_mpnn_run.py",
            "--jsonl_path", path_for_parsed_chains,
            "--chain_id_jsonl", path_for_assigned_chains,
            "--fixed_positions_jsonl", path_for_fixed_positions,
            "--out_folder", output_dir,
            "--num_seq_per_target", f'{self.num_sequences_mhc}',
            "--sampling_temp", f'{self.sampling_temp}',
            "--seed", "37",
            "--batch_size", f'{self.batch_size}',
            "--save_probs", "1",
            "--save_score", "1"
        ], check=True)
        print('MHC Pseudo Sequence Generation Mode Done! *****\n')


    def input_assertion(self):
        assert isinstance(self.pdb, str), f'PMGen_pdb should be a string, found {self.pdb}'
        assert isinstance(self.output_dir, str), f'output_dir should be a string, found {self.output_dir}'
        assert isinstance(self.peptide_chain, str), f'peptide_chain should be a string, found {self.peptide_chain}'
        assert isinstance(self.peptide_design, bool), f'peptide_design should be boolean, found {self.peptide_design}'
        assert isinstance(self.mhc_design, bool), f'peptide_design should be boolean, found {self.mhc_design}'
        assert isinstance(self.only_pseudo_sequence_design, bool), f'peptide_design should be boolean, found {self.only_pseudo_sequence_design}'
        assert isinstance(self.anchor_pred, bool), f'anchor_pred should be boolean, found {self.anchor_pred}'
        assert isinstance(self.num_sequences_peptide, int), f'num_sequences_peptide should be int, found {self.num_sequences_peptide}'
        assert isinstance(self.num_sequences_mhc, int), f'num_sequences_mhc should be int, found {self.num_sequences_mhc}'
        assert isinstance(self.sampling_temp, float), f'sampling_temp should be float, found {self.sampling_temp}'
        assert isinstance(self.batch_size, int), f'batch_size should be int, found {self.batch_size}'
        assert isinstance(self.save_hotspots, bool), f'save_hotspots should be bool, found {self.save_hotspots}'


def run_single_proteinmpnn(path, directory, args):
    """Function to be executed in parallel for each path in path_list"""
    model_dir = os.path.join(directory, path.split('/')[-1].strip('.pdb'))  # Create model-specific directory
    os.makedirs(model_dir, exist_ok=True)

    # Copy PDB file to the new directory
    shutil.copy(path, os.path.join(model_dir, path.split('/')[-1]))
    PMGen_pdb = os.path.join(model_dir, path.split('/')[-1])
    print('#########', PMGen_pdb)

    # Run ProteinMPNN
    runner_mpnn = run_proteinmpnn(
        PMGen_pdb=PMGen_pdb, output_dir=model_dir,
        num_sequences_peptide=args.num_sequences_peptide,
        num_sequences_mhc=args.num_sequences_mhc,
        peptide_chain='P', mhc_design=args.mhc_design, peptide_design=args.peptide_design,
        only_pseudo_sequence_design=args.only_pseudo_sequence_design,
        anchor_pred=True,
        sampling_temp=args.sampling_temp, batch_size=args.batch_size,
        hot_spot_thr=args.hot_spot_thr
    )
    runner_mpnn.run() #


def protein_mpnn_wrapper(output_pdbs_dict, args, max_jobs, mode='parallel'):
    """Main function that runs in either 'parallel' or 'single' mode."""
    if mode == 'parallel':
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_jobs) as executor:
            futures = []
            for id_m, path_list in output_pdbs_dict.items():
                directory = os.path.join(args.output_dir, 'protienmpnn', id_m)
                os.makedirs(directory, exist_ok=True)  # Ensure main directory exists

                for path in path_list:
                    futures.append(executor.submit(run_single_proteinmpnn, path, directory, args))

            # Wait for all processes to finish
            for future in futures:
                future.result()  # This will re-raise any exceptions if they occur

    elif mode == 'single':
        for id_m, path_list in output_pdbs_dict.items():
            directory = os.path.join(args.output_dir, 'protienmpnn', id_m)
            os.makedirs(directory, exist_ok=True)  # Ensure main directory exists

            for path in path_list:
                run_single_proteinmpnn(path, directory, args)  # Run sequentially

    else:
        raise ValueError("Invalid mode! Choose 'parallel' or 'single'.")


