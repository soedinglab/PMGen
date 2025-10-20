import psutil
import numpy as np
import concurrent.futures
import sys
from multiprocessing import Pool, cpu_count
import contextlib
import ast
import fnmatch
from numpy.core.defchararray import endswith
from itertools import product
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
                 benchmark=False, n_homology_models=1, best_n_templates=4,
                 pandora_force_run=True, no_modelling=False,
                 return_all_outputs=False):
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
            best_n_templates (int): number of found templates used for homology modeling via modeler, default=4.
            pandora_force_run (bool): Weather to force run pandora or not, default=True.
            no_modelling (bool): If active, no modeller homology modeling happens and only PANDORA is used for template search and alignment.
            return_all_outputs (bool): If active, all alphafold outputs are saved.
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
        self.no_modelling = no_modelling
        self.return_all_outputs = return_all_outputs
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
        self.no_modelling_output_dict = None

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
        if not self.no_modelling:
            _ = self.alignment_with_peptide(pdb_files, mhc_pep_seq, output_path=aln_output_file)
        else:
            os.makedirs(self.alignment_output, exist_ok=True)
            processing_functions.alignment_to_df(self.no_modelling_output_dict, output_dir=aln_output_file)
        ## Prepare Alphafold Fine Input files
        os.makedirs(self.alphafold_out + f'/{self.id}', exist_ok=True)
        self.alphafold_preparation(template_aln_file=aln_output_file, mhc_pep_seq=mhc_pep_seq, output=self.alphafold_input_file)
        self.output_pdbs_dict = {}
        if run_alphafold:
            print('## To run Alphafold Please Make Sure GPU is Available and can be found ##')
            self.run_alphafold(input_file=self.alphafold_input_file, output_prefix=self.alphafold_out + f'/{self.id}/')
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
                    case = Pandora.Pandora(target, self.db, no_modelling=self.no_modelling)
                    case.model(n_loop_models=self.num_templates, benchmark=self.benchmark,
                               n_homology_models=self.n_homology_models,
                               best_n_templates=self.best_n_templates)
                    print("Pandora modeling completed successfully.")
                except Exception as e:
                    print(f"❌ An error occurred during template engineering {self.id}: {str(e)}", file=sys.stderr)
                    raise
        print("✔Pandora run completed. Check log file for details:", log_file)
        if self.no_modelling:
            if shoud_I_run == 'Yes':
                self.no_modelling_output_dict = case.no_modelling_output_dict

            else:
                with open(os.path.join(self.pandora_output, self.id, 'no_modelling_output_dict.json'), 'r') as f:
                    self.no_modelling_output_dict = json.load(f)
            self.no_modelling_assertion()

        # get template id used in pandora
        if not self.no_modelling:
            files = [file for file in glob.glob(os.path.join(self.pandora_output, self.id, '????.pdb')) if
                     "mod" not in file.split("/")[-1]]
            if files:
                template_id = files[0].split("/")[-1]
                print(f"✔ {self.id} log: Template ID used for homology modeling: {template_id}")
                return template_id
            else:
                print(f"❌ {self.id} log: No template ID found.")
                return None
        else: print(f"No modelling mode is done, files generated in: {self.pandora_output + '/' + self.id}")

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
        template_pdb_dict_path = os.path.join(self.pandora_output, self.id, "no_modelling_output_dict.json")
        df = pd.DataFrame({"target_chainseq": [mhc_pep_seq],
                           "templates_alignfile": [template_aln_file],
                           "targetid": [self.id],
                           "template_pdb_dict": [template_pdb_dict_path]})
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
        if not self.no_modelling:
            command += ['--no_initial_guess']
        else:
            print(' -- Alphafold Initial Guess Mode, No homology models will be used --')
        if self.return_all_outputs:
            command += ['--return_all_outputs']

        print('AFfine Command: \n',command)
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

            # Capture remaining stderr outputHi, y
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

    def no_modelling_assertion(self):

        required_keys = [
            'template_anchors', 'target_anchors', 'template_id', 'aln_M',
            'aln_B_N', 'aln_P', 'aln_template', 'aln_target', 'template_path'
        ]
        missing_keys = [key for key in required_keys if key not in self.no_modelling_output_dict]
        assert not missing_keys, f"Missing keys in no_modelling_output_dict: {missing_keys}"
        empty_keys = [key for key in required_keys if not self.no_modelling_output_dict[key]]
        assert not empty_keys, f"Empty values found for keys: {empty_keys}"

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
                 pandora_force_run=True, no_modelling=False, return_all_outputs=False):
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
        :param no_modelling (bool): If active, no modeller homology modeling happens and only PANDORA is used for template search and alignment.
        :param pandora_force_run (bool): If active, PANDORA will be forced to run even if files already exist.
        :param return_all_outputs (bool): If active, all alphafold outputs are saved.
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
        self.no_modelling = no_modelling
        self.return_all_outputs = return_all_outputs
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
                                            pandora_force_run=self.pandora_force_run, no_modelling=self.no_modelling,
                                            return_all_outputs=self.return_all_outputs)
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
                                        pandora_force_run=self.pandora_force_run, no_modelling=self.no_modelling,
                                        return_all_outputs=self.return_all_outputs)
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
                                            fine_tuned_model_path=self.fine_tuned_model_path, no_modelling=self.no_modelling,
                                            return_all_outputs=self.return_all_outputs)
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
                 sampling_temp=5, batch_size=1, hot_spot_thr=6.0,
                 save_hotspots=True, binder_pred=False, fix_anchors=False,
                 anchor_and_peptide=None, return_match_allele=False):
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
            binder_pred: (bool) if True, predicts binding probability of generated peptides, Default: False.
            fix_anchors: (bool) if True, fixes anchor positions in peptide generative mode, Default: False.
            anchor_and_peptide: (tuple, list) Defines anchor positions on the peptide and peptide seq as [anchors, peptide].
                                Only used if fix_anchors==True, Default: False
            return_match_allele: (bool) Returns a list of one or two elementsself.matched_alleles if binder_pred is True,
                                Depending on MHC type.
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
        self.binder_pred = binder_pred
        self.fix_anchors = fix_anchors
        self.anchor_and_peptide = anchor_and_peptide
        self.return_match_allele = return_match_allele
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
        if self.binder_pred:
            self.__binder_pred()

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
        final_run_command = [
            "python", "-W", "ignore", "ProteinMPNN/protein_mpnn_run.py",
            "--jsonl_path", path_for_parsed_chains,
            "--chain_id_jsonl", path_for_assigned_chains,
            "--out_folder", output_dir,
            "--num_seq_per_target", f'{self.num_sequences_peptide}',
            "--sampling_temp", f'{self.sampling_temp}',
            "--seed", "37",
            "--batch_size", f'{self.batch_size}',
            "--save_probs", "1",
            "--save_score", "1",
            "--omit_AAs", "X",
        ]
        if self.fix_anchors:# to fix anchors, fixed_pdbs file and design_only_positions should be generated
            # we have anchors, we need to define designable positions which are non-anchor positions
            pep_len = len(self.anchor_and_peptide[1])
            anchors = [int(i) for i in self.anchor_and_peptide[0]]
            design_only_positions = np.arange(1, pep_len+1, dtype=int)
            design_only_positions = design_only_positions[~np.isin(design_only_positions, anchors)] # anchors removed and non anchors remained
            path_for_fixed_positions = os.path.join(output_dir, "fixed_pdbs.jsonl")
            design_only_positions = " ".join(map(str, design_only_positions)) # e.g "1 2 3 4 6 7 8 10"
            subprocess.run([
                "python", "ProteinMPNN/helper_scripts/make_fixed_positions_dict.py",
                "--input_path", path_for_parsed_chains,
                "--output_path", path_for_fixed_positions,
                "--chain_list", chains_to_design,
                "--position_list", design_only_positions,
                "--specify_non_fixed"
            ])
            final_run_command += ["--fixed_positions_jsonl", path_for_fixed_positions]
        # Run protein_mpnn_run.py
        subprocess.run(final_run_command, check=True)
        print('Full MHC Sequence Generation Mode Done! *****\n')

    def __binder_pred(self):
        output_dir = os.path.join(self.output_dir, 'peptide_design')
        peptide_fasta_file = [i for i in os.listdir(output_dir+'/'+'seqs') if i.endswith('.fa')][0]
        peptide_fasta_file = os.path.join(output_dir+'/'+'seqs', peptide_fasta_file)
        mhc_type = 2 if len(self.chain_dict_dist.keys()) == 2 else 1
        mhc_seq_dict = processing_functions.fetch_polypeptide_sequences(self.multichain_pdb)
        mhc_seq_list = [mhc_seq_dict['A'], mhc_seq_dict['B']] if mhc_type==2 else [mhc_seq_dict['A']]
        out_dir = os.path.join(output_dir, 'binder_pred')
        if self.return_match_allele:
            df, self.matched_allele = run_and_parse_netmhcpan(peptide_fasta_file, mhc_type, out_dir, mhc_seq_list=mhc_seq_list,
                                         mhc_allele=None, dirty_mode=False, return_match_allele=self.return_match_allele)
        else:
            df = run_and_parse_netmhcpan(peptide_fasta_file, mhc_type, out_dir,
                                                         mhc_seq_list=mhc_seq_list,
                                                         mhc_allele=None, dirty_mode=False,
                                                         return_match_allele=self.return_match_allele)

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
        assert isinstance(self.fix_anchors, bool), f'fix_anchors must be bool, found {self.fix_anchors}'
        if self.fix_anchors:
            assert isinstance(self.anchor_and_peptide[0], (tuple, list)), f'anchors in anchor_and_peptide must be [tuple, list], found {self.anchor_and_peptide[0]}'
            assert isinstance(self.anchor_and_peptide[1], str), f'peptide in anchor_and_peptide must be a str, found {self.anchor_and_peptide[1]}'


def run_single_proteinmpnn(path, directory, args, anchor_and_peptide=None):
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
        hot_spot_thr=args.hot_spot_thr,
        binder_pred=args.binder_pred,
        fix_anchors=args.fix_anchors,
        anchor_and_peptide=anchor_and_peptide
    )
    runner_mpnn.run() #


def protein_mpnn_wrapper(output_pdbs_dict, args, max_jobs, anchor_and_peptide=None, mode='parallel'):
    """Main function that runs in either 'parallel' or 'single' mode."""
    if mode == 'parallel':
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_jobs) as executor:
            futures = []
            for id_m, path_list in output_pdbs_dict.items():
                directory = os.path.join(args.output_dir, 'protienmpnn', id_m)
                os.makedirs(directory, exist_ok=True)  # Ensure main directory exists
                if anchor_and_peptide: aap = anchor_and_peptide[id_m] # [[1,2,4,5], 'ASGSGS'] --> [anchors, peptide]
                else: aap = None
                for path in path_list:
                    futures.append(executor.submit(run_single_proteinmpnn, path, directory, args, aap))

            # Wait for all processes to finish
            for future in futures:
                future.result()  # This will re-raise any exceptions if they occur

    elif mode == 'single':
        for id_m, path_list in output_pdbs_dict.items():
            directory = os.path.join(args.output_dir, 'protienmpnn', id_m)
            os.makedirs(directory, exist_ok=True)  # Ensure main directory exists
            if anchor_and_peptide: aap = anchor_and_peptide[id_m]  # [[1,2,4,5], 'ASGSGS'] --> [anchors, peptide]
            else: aap = None

            for path in path_list:
                run_single_proteinmpnn(path, directory, args, aap)  # Run sequentially

    else:
        raise ValueError("Invalid mode! Choose 'parallel' or 'single'.")


def run_and_parse_netmhcpan(peptide_fasta_file, mhc_type, output_dir, mhc_seq_list=[], mhc_allele=None,
                            dirty_mode=False, verbose=False, outfilename='netmhcpan_out', return_match_allele=False,
                            match_with_netmhcpan=True):
    assert mhc_type in [1,2]
    if not mhc_allele and len(mhc_seq_list) == 0:
        raise ValueError(f'at least one of mhc_seq_list or mhc_allele should be provided')
    os.makedirs(output_dir, exist_ok=True)
    outfile = os.path.join(output_dir, f'{outfilename}.txt')
    outfile_csv = os.path.join(output_dir, f'{outfilename}.csv')

    if mhc_type == 1: # only one sequence in mhc_seq_list
        if not mhc_allele:
            assert len(mhc_seq_list) == 1, (f'for mhc1, only one sequence should be inside mhc_seq_list, '
                                            f'found {len(mhc_seq_list)}: {mhc_seq_list}')
        else: mhc_seq_list = ['', '']
    elif mhc_type == 2:
        if not mhc_allele:
            assert len(mhc_seq_list) == 2, (f'for mhc2, two sequences should be inside mhc_seq_list, '
                                            f'with the Alpha/Beta order '
                                            f'found {len(mhc_seq_list)}: {mhc_seq_list}')
        else:
            assert len(mhc_allele.split('/'))==2, (f'mhc_allele for mhc class 2, should contant both alpha and beta alleles seperated by "/"\n example: DRA/DRB*01. found {mhc_allele}')
            mhc_seq_list = ['', '']

    matched_allele = []
    for i in range(2):
        allele = mhc_allele.split('/')[i] if mhc_type==2 and mhc_allele else mhc_allele
        a = allele
        if match_with_netmhcpan:
            a = processing_functions.match_inputseq_to_netmhcpan_allele(mhc_seq_list[i], mhc_type, allele)
        matched_allele.append(a)
        if mhc_type == 1: break
    if verbose: print("Matched Alleles", matched_allele)
    processing_functions.run_netmhcpan(peptide_fasta_file, matched_allele, outfile, mhc_type)
    df = processing_functions.parse_netmhcpan_file(outfile)
    df.to_csv(outfile_csv, index=False)
    if not dirty_mode:
        os.remove(outfile)

    if not return_match_allele: return df
    else: return df, matched_allele




class MultipleAnchors:
    def __init__(self, args, dirty_mode=False):
        """
        Initialize the MultipleAnchors class

        Args:
            args: input arguments for run_PMGen.py
            dirty_mode: boolean flag for file handling
        """
        assert args.mode == 'wrapper', f'multiple anchors option only works with wrapper mode'
        self.args = args
        self.dirty_mode = dirty_mode
        self.tmp = os.path.join(args.output_dir, 'tmp')
        os.makedirs(self.tmp, exist_ok=True)
        assert args.top_k >= 1, f'top_k should be at least 1, found {args.top_k}'

    def _process_row(self, row):
        """Process a single row and return results"""
        peptide_fasta_file = os.path.join(self.tmp, f'{str(row.id)}.fasta')
        with open(peptide_fasta_file, 'w') as f:
            f.write(f'>{str(row.id)}\n{str(row.peptide)}')
        mhc_type = int(row.mhc_type)
        assert mhc_type in [1, 2], f'mhc_type in dataframe should be either 1 or 2, found {mhc_type}'
        mhc_seq_list = str(row.mhc_seq).split('/')
        if mhc_type == 2:
            assert len(mhc_seq_list) == 2, (f'mhc_seq for mhc_type==2, should be "Alpha/Beta" separated by "/", '
                                            f'found: \n {str(row.mhc_seq)}')
        elif mhc_type == 1:
            assert len(mhc_seq_list) == 1, (f'mhc_seq for mhc_type==1, should be string with no "/", '
                                            f'found: \n {str(row.mhc_seq)}')
        netmhc_df = run_and_parse_netmhcpan(peptide_fasta_file, mhc_type, self.tmp, mhc_seq_list, verbose=False, outfilename=str(row.id))
        seen_cores = []
        results = {'anchors': [], 'mhc_seqs': [], 'ids': [], 'peptides': [], 'mhc_types': []}
        counter = 0
        for j, net_row in netmhc_df.iterrows():
            peptide2 = str(net_row['Core'])
            peptide1 = str(row.peptide)
            predicted_anchors, pept1, pept2 = processing_functions.align_and_find_anchors_mhc(peptide1, peptide2,
                                                                                              mhc_type)
            if not predicted_anchors in seen_cores:
                seen_cores.append(predicted_anchors)
                results['anchors'].append(";".join([str(pp) for pp in predicted_anchors]))
                results['mhc_seqs'].append(str(row['mhc_seq']))
                results['ids'].append(str(row['id']) + '_' + str(counter))
                results['peptides'].append(str(row['peptide']))
                results['mhc_types'].append(int(row['mhc_type']))
                counter += 1
            if counter == self.args.top_k: break
        return results

    def process(self):
        """
        Process the dataframe in parallel using multiprocessing and return results

        Returns:
            DataFrame with processed results
        """
        df = pd.read_csv(self.args.df, sep='\t')
        print(f" Starting Multiple Anchor Mode on {self.args.max_cores} cores. Make Sure NetMHCpan is installed")
        # Determine number of processes
        num_processes = min(cpu_count(), int(self.args.max_cores))
        # Create multiprocessing pool
        with Pool(processes=num_processes) as pool:
            # Process rows in parallel
            results = pool.map(self._process_row, [row for _, row in df.iterrows()])
        # Combine results
        all_anchors = []
        all_mhc_seqs = []
        all_ids = []
        all_peptides = []
        all_mhc_types = []

        for result in results:
            all_anchors.extend(result['anchors'])
            all_mhc_seqs.extend(result['mhc_seqs'])
            all_ids.extend(result['ids'])
            all_peptides.extend(result['peptides'])
            all_mhc_types.extend(result['mhc_types'])

        # Create final DataFrame
        DF = pd.DataFrame({
            'peptide': all_peptides,
            'mhc_seq': all_mhc_seqs,
            'anchors': all_anchors,
            'mhc_type': all_mhc_types,
            'id': all_ids
        })
        output_file = os.path.join(self.args.output_dir, 'Multiple_Anchors_input.tsv')
        DF.to_csv(output_file, sep='\t', index=False)
        if not self.dirty_mode:
            txt_files = glob.glob(os.path.join(self.tmp, "*"))
            for file in txt_files:
                try:
                    os.remove(file)
                except:
                    pass
        return DF



def get_best_structres(output_dir, df, multiple_anchors):
    final_df = processing_functions.read_and_extract_core_plddt_from_df_with_anchor(df = df,
                                                                         output_folder = output_dir,
                                                                         path_to_af = 'alphafold',
                                                                         multiple_anchors = multiple_anchors)
    return final_df


def open_json(path):
    with open(path, 'rb') as f:
        a = json.load(f)
    return a


def fixed_anchor_pos(peptide, peptide_random_fix_fraction, anchors):
    pep_positions = np.arange(1, len(peptide) + 1, dtype=int)
    non_anchors = pep_positions[~np.isin(pep_positions, np.array(anchors))]
    sizen = int(len(non_anchors) * peptide_random_fix_fraction)
    fixed_non_anchors = list([int(i) for i in np.random.choice(non_anchors, size=sizen, replace=False)])
    return fixed_non_anchors


def load_final_dict_from_df(df_path):
    """
    Load the 'final' dictionary from a saved DataFrame TSV file.
    Args:
        df_path (str): Path to the 'fixed_positions.tsv' file.
    Returns:
        dict: A dictionary where keys are IDs and values are [anchor list, peptide string].
    """
    df = pd.read_csv(df_path, sep='\t')

    final = {}
    for _, row in df.iterrows():
        id_ = row['id']
        anchor = ast.literal_eval(row['anchor']) if isinstance(row['anchor'], str) else row['anchor']
        peptide = row['peptide']
        final[id_] = [anchor, peptide]
    return final


def retrieve_anchors_and_fixed_positions(args, save_anchors=True, peptide_random_fix_fraction=0.0, fixed_positions_path=None):
    '''
    Three conditions exist for retrieving anchors:
    1. User has given anchor:
       -> Retrieved from args.anchors (usually passed directly or from an Excel file)
    2. Anchor predicted in template engineering mode:
       -> Retrieved from 'args.output_dir/pandora/<id>/anchors.json'
    3. Anchor predicted in initial guess model:
       -> Retrieved from 'args.output_dir/pandora/<id>/no_modelling_output_dict.json'

    params:
    fixed_positions_path: if fixed positions already exist, provide the path
    should be a file
    Returns:
        If modeling mode: list of anchors
        If wrapper mode: dict {<id>: anchors}
    '''
    # Helper function to load anchors from a JSON file
    def load_anchors_from_file(file_path, key):
        assert os.path.exists(file_path), f"File not found: {file_path}"
        data = open_json(file_path)
        return data[key][0] if key == 'target_anchors' else data[key]  # [0] only for 'target_anchors'
    final = None
    if not fixed_positions_path:
        # MODE 1: Modeling mode
        if args.mode == 'modeling':
            if args.anchors:
                # Case 1: User has provided anchors manually
                fixed_non_anchors = fixed_anchor_pos(args.peptide, peptide_random_fix_fraction, args.anchors)
                final = {args.id: [list(args.anchors) + fixed_non_anchors, args.peptide]}
            else:
                # Case 2 or 3: Anchors should be read from model output
                file_name = 'no_modelling_output_dict.json' #if args.initial_guess else 'anchors.json'
                key = 'target_anchors' #if args.initial_guess else 'anchors'
                file_path = os.path.join(args.output_dir, 'pandora', args.id, file_name)
                anchors = load_anchors_from_file(file_path, key)
                # Ensure that returned anchors are in list format
                assert isinstance(anchors, list), "Anchors must be a list."
                fixed_non_anchors = fixed_anchor_pos(args.peptide, peptide_random_fix_fraction, anchors)
                final = {args.id: [anchors + fixed_non_anchors, args.peptide]}
        # MODE 2: Wrapper mode — for processing multiple inputs in batch
        elif args.mode == 'wrapper':
            # Determine which file to read:
            # Case A: args.multiple_anchors enabled → read from 'Multiple_Anchors_input.tsv'
            # Case B: otherwise → read from args.df (input file provided by user)
            file = (
                os.path.join(args.output_dir, 'Multiple_Anchors_input.tsv')
                if args.multiple_anchors else args.df
            )
            assert isinstance(file, str), "Input file path must be a string."
            # Read TSV or CSV file
            df = pd.read_csv(file, sep='\t' if file.endswith('.tsv') else ',')
            # Ensure required columns exist
            assert 'id' in df.columns, f"'id' column is required in input file \n {df}"
            assert 'anchors' in df.columns, f"'anchors' column is required in input file \n {df}"
            final = {}
            # Iterate over each entry (row) in the file
            for _, row in df.iterrows():
                id_ = row['id']
                anchor = row['anchors']
                peptide = row['peptide']
                # If anchor is not available in file, retrieve from modeling outputs
                if pd.isna(anchor) or anchor in ('None', ''):
                    file_name = 'no_modelling_output_dict.json' #if args.initial_guess else 'anchors.json'
                    key = 'target_anchors' #if args.initial_guess else 'anchors'
                    file_path = os.path.join(args.output_dir, 'pandora', id_, file_name)
                    anchor = load_anchors_from_file(file_path, key)
                # Store anchor in dictionary
                if not isinstance(anchor, list): anchor = list(anchor)
                fixed_non_anchors = fixed_anchor_pos(peptide, peptide_random_fix_fraction, anchor)
                final[id_] = [anchor + fixed_non_anchors, peptide]  # {id1: anchor1, id2: anchor2, ...}
        if not isinstance(final, dict): raise ValueError('No anchor could be retrieved: unrecognized mode.')
        if save_anchors:
            rows = [{"id": id_, "anchor": value[0], "peptide": value[1]} for id_, value in final.items()]
            df = pd.DataFrame(rows)
            df.to_csv(os.path.join(args.output_dir, 'fixed_positions.tsv'), sep='\t', index=False)
    else:
        final = load_final_dict_from_df(fixed_positions_path) #if user provides fixed_position_path it will read final. used in iterative mode.
    return final


def create_fixed_positions_if_given(args):
    if args.fixed_positions_given:
        if args.iterative_peptide_gen > 0:
            outpath = os.path.join(args.output_dir, 'iter_0')
        else:
            outpath = os.path.join(args.output_dir)
        os.makedirs(outpath, exist_ok=True)
        try:
            df = pd.read_csv(args.df, sep='\t')
            anchor = df['fixed_positions'].tolist()
        except:
            df = pd.read_csv(args.df)
            anchor = df['fixed_positions'].tolist()
        id = df.id.tolist()
        peptide = df.peptide.tolist()
        final = pd.DataFrame({'id':id, 'anchor':anchor, 'peptide':peptide})
        o = os.path.join(str(outpath), 'fixed_positions.tsv')
        final.to_csv(o, sep='\t', index=False)
        return o
    else: return None

def assert_iterative_mode(args):
    assert isinstance(args.iterative_peptide_gen, int), f"Flag:iterative_peptide_gen must be an int, found {args.iterative_peptide_gen}"
    if args.iterative_peptide_gen > 0:
        assert args.mode == "wrapper", f"When using iterative_peptide_gen, '--mode wrapper' must be set, found {args.mode}"
        assert args.peptide_design == True, f"When using iterative_peptide_gen, '--peptide_design' flag must be used."
        assert args.df, f"When using iterative_peptide_gen, --df must be given"
        assert args.df.endswith('.tsv'), (f"When using iterative_peptide_gen, please make sure your "
                                          f"--df file endswith '.tsv' and is a tab separated file, found {args.df}")
        assert args.binder_pred == True, f"When using iterative_peptide_gen, --binder_pred flag must be used."
        assert len(args.models) == 1, (f"When using iterative_peptide_gen, only one model must be used, found {args.models}"
                                      f"and len(): {len(args.models)}")
        if not args.fix_anchors: print("Warning, You have chosen iterative mode, "
                                       "and it is better to also set 'fix_anchors flag'"
                                       "for better results.")


def swap_columns(df, col1, col2):
    """
    Swap the positions of two columns in a DataFrame without changing their content.
    Args:
        df (pd.DataFrame): The original DataFrame.
        col1 (str): Name of the first column.
        col2 (str): Name of the second column.
    Returns:
        pd.DataFrame: New DataFrame with col1 and col2 swapped.
    """
    cols = list(df.columns)
    idx1, idx2 = cols.index(col1), cols.index(col2)
    cols[idx1], cols[idx2] = cols[idx2], cols[idx1]
    return df[cols]


def collect_generated_binders(args, df, iter, debugging=True):
    protmpnn_path = os.path.join(args.output_dir, "protienmpnn")
    id_paths = os.listdir(protmpnn_path)
    DF = []
    for id in id_paths:
        idp = os.path.join(protmpnn_path, id)
        if os.path.isdir(idp):
            model_folders = os.listdir(idp)
            if len(model_folders) > 0:
                try:
                    if len(model_folders) > 1:
                        model_folder = next(
                            i for i in model_folders
                            if "model_" in i and "peptide_design" in os.listdir(os.path.join(idp, i))
                        )
                    else:
                        model_folder = model_folders[0]
                    netmhcpan_out = os.path.join(idp, model_folder, "peptide_design", "binder_pred", "netmhcpan_out.csv")
                    if not os.path.exists(netmhcpan_out):
                        continue
                    data = pd.read_csv(netmhcpan_out)
                    # Handling missing expected columns safely
                    if "Identity" not in data.columns or "Peptide" not in data.columns:
                        continue
                    id_original_peptide = data["Identity"].tolist()[0]
                    original_filtered = data[data["Identity"]==id_original_peptide]
                    best_original_data = original_filtered.loc[original_filtered["%Rank_EL"].idxmin()]
                    best_original_peptide = max(
                        original_filtered[original_filtered["Identity"] == id_original_peptide]["Peptide"],
                        key=len
                    )

                    data_filtered = data[data["Identity"] != id_original_peptide]
                    if data_filtered.empty:
                        continue
                    best_iden_data = data_filtered.loc[data_filtered["%Rank_EL"].idxmin()]
                    best_iden = best_iden_data["Identity"]
                    best_peptide = max(
                        data_filtered[data_filtered["Identity"] == best_iden]["Peptide"],
                        key=len
                    )
                    assert len(best_original_peptide) == len(best_peptide), (f"original peptide: {best_original_peptide}, \n"
                                                                             f"generated peptide: {best_peptide}")
                    # Flexible affinity column name
                    aff_col = "Affinity(nM)" if "Affinity(nM)" in best_iden_data else "Aff(nM)"
                    best_peptide_dict = pd.DataFrame({
                        "id": [id],
                        f"generated_peptide_{iter}": [best_peptide],
                        f"generated_affinity_{iter}": [best_iden_data[aff_col]],
                        f"original_affinity_{iter}": [best_original_data[aff_col]]
                    })
                    DF.append(best_peptide_dict)
                except (IndexError, StopIteration, FileNotFoundError, KeyError, ValueError):
                    continue  # skip malformed entries safely
    if not DF:
        if debugging:
            raise ValueError(f"No valid binder prediction outputs found.\n"
                         f"Debugginh message:\n"
                         f"id_paths: {id_paths}, \n"
                         f"model_folders: {model_folders}\n")
        else:
            raise ValueError(f"No valid binder prediction outputs found.")
    DF = pd.concat(DF, ignore_index=True)
    final = pd.merge(df, DF, on=['id'], how='inner')
    p = os.path.join(args.output_dir, f"best_generated_peptides_{iter}.tsv")
    final.to_csv(p, sep="\t", index=False)
    print(f"Best binders saved in {p}")
    return p


def create_new_input_and_fixed_positions(args, best_generated_peptides_path, iter, fixed_positions_path=None):
    bgp_df = pd.read_csv(best_generated_peptides_path, sep='\t', header=0)
    bgp_df = swap_columns(bgp_df, col1=f"generated_peptide_{iter-1}", col2="peptide")
    bgp_df = bgp_df.drop(["peptide", f"original_affinity_{iter-1}", f"generated_affinity_{iter-1}"], axis=1)
    bgp_df = bgp_df.rename(columns={f"generated_peptide_{iter-1}": "peptide"})
    outpath = os.path.join(args.output_dir, f"input_df_{iter}.tsv")
    bgp_df.to_csv(outpath, sep='\t', index=False) # new input generated
    # now fixed positions should be kept in new input as well
    if fixed_positions_path: # if user has used fix_positions it should be saved in iter, with new peptides
        fp_df = pd.read_csv(fixed_positions_path, sep="\t", header=0)
        fp_df = fp_df.drop("peptide", axis=1)
        fp_df = fp_df.rename(columns={"anchor": "fixed_pos"})
        fp_df = pd.merge(fp_df, bgp_df, on=["id"])
        print(fp_df)
        fp_df = fp_df[["id","peptide","fixed_pos"]]
        fp_df = fp_df.rename(columns={"fixed_pos": "anchor"})
        outpath = outpath.replace(f"input_df_{iter}.tsv", "fixed_positions.tsv")
        fp_df.to_csv(outpath, sep="\t", index=False)




def bioemu_assertions(args):
    if args.run_bioemu:
        args.return_all_outputs = True  # make sure AF2 outputs are returned
        assert args.mode == 'wrapper', f'Bioemu is availabe only in wrapper mode: use --mode wrapper'
        if args.iterative_peptide_gen > 0: #iterative mode --> which iteration to run bioemu on? only one can be used.
            if args.bioemu_run_on_iter:
                assert args.bioemu_run_on_iter <= args.iterative_peptide_gen, f'Please make sure --iterative_peptide_gen is less or equal to  --iterative_peptide_gen'


from itertools import product


def generate_mutants(peptide, positions, fasta_out, id, one_indexed=True):
    """
    Generate all possible amino acid mutants at specified positions in a peptide
    and save them to a FASTA file.
    Args:
        peptide (str): Original peptide sequence.
        positions (list[int]): Positions to mutate (1-indexed by default).
        fasta_out (str): Path to output FASTA file.
        one_indexed (bool): If True, positions are 1-indexed; if False, 0-indexed.
    """
    # Standard amino acid alphabet
    AAs = list("ACDEFGHIKLMNPQRSTVWY")
    # Convert positions to 0-indexed if needed
    if one_indexed:
        positions = [p - 1 for p in positions]
    # Generate possible substitutions (excluding the original residue)
    mutation_space = [
        [aa for aa in AAs if aa != peptide[pos]] for pos in positions]
    # Generate all combinations of mutations
    mutation_combinations = list(product(*mutation_space))
    with open(fasta_out, "w") as f:
        for i, combo in enumerate(mutation_combinations, 1):
            # Create a mutable list of the peptide
            mutant = list(peptide)
            # Apply mutations
            for pos, new_aa in zip(positions, combo):
                mutant[pos] = new_aa
            mutant_seq = "".join(mutant)
            mutation_description = "_".join(
                f"{peptide[pos]}{pos + 1}{mut}" for pos, mut in zip(positions, combo)
            )
            # Write FASTA entry
            f.write(f">{id}_{i}_{mutation_description}\n{mutant_seq}\n")
    print(f"✅ Generated {len(mutation_combinations)} mutants and saved to {fasta_out}")


def filter_peptides_from_netmhcpan_csv_output(df):
    """
    1. Removes rows where 'Identity' starts with 'multichain'.
    2. For each remaining Identity, keeps the row with the longest peptide.
    3. Sorts the resulting rows by '%Rank_EL' in ascending order (lower = better).
    """
    # Step 1: Remove 'multichain' rows
    df = df[~df['Identity'].astype(str).str.startswith('multichain')].copy()
    # Step 2: Compute peptide lengths
    df['Peptide_length'] = df['Peptide'].str.len()
    # Step 3: For each Identity, keep the row with the longest peptide
    df = df.loc[df.groupby('Identity')['Peptide_length'].idxmax()]
    # Step 4: Sort by '%Rank_EL' (ascending)
    df = df.sort_values(by='%Rank_EL', ascending=True)
    # Step 5: Drop helper column and reset index
    df = df.drop(columns='Peptide_length').reset_index(drop=True)
    df.drop_duplicates(subset=['Peptide'], inplace=True)
    df.drop_duplicates(subset=['Identity'], inplace=True)
    return df

class mutation_screen():
    def __init__(self, args, df, **kwargs):
        self.args = args
        self.af_path = os.path.join(args.output_dir, 'alphafold')
        self.ids = [i for i in os.listdir(self.af_path) if os.path.isdir(os.path.join(self.af_path, i))]
        self.ms_path = os.path.join(self.args.output_dir, 'mutation_screen')
        self.mt_num = args.n_mutations
        self.df = df
        os.makedirs(self.ms_path, exist_ok=True)

    def find_pdb_matches(self, file_list, id):
        pattern = f"{id}*.pdb"
        matched = [f for f in file_list if fnmatch.fnmatch(f, pattern)]
        return bool(matched), matched

    def linear_positional_combinations(self, pep_len, n):
        if n > pep_len:
            raise ValueError("n cannot be larger than the peptide length")
        return [list(range(i, i + n)) for i in range(pep_len - n + 1)]

    def process_single_id(self, id):
        """
        Process one id. This function can be called in parallel.
        Returns the combined dataframe for that id.
        """
        all_dfs = []
        id_dir = os.path.join(self.af_path, id)
        if os.path.exists(id_dir):
            files = os.listdir(id_dir)
            mhc_type = int(self.df[self.df['id']==id]['mhc_type'].tolist()[0])
            mhc_seq = str(self.df[self.df['id']==id]['mhc_seq'].tolist()[0])
            boolval, matched = self.find_pdb_matches(files, id)
            if boolval:
                pdb_path = os.path.join(id_dir, matched[0])
                id_path = os.path.join(self.ms_path, id)
                os.makedirs(id_path, exist_ok=True)
                peptide = str(self.df[self.df['id']==id]['peptide'].tolist()[0])
                pep_len = len(peptide)
                lin_pep_combs = self.linear_positional_combinations(pep_len, self.mt_num)
                COMB_DF = []

                for comb in lin_pep_combs:
                    comb = [i+1 for i in comb]  # 1-indexed
                    fixed = [i for i in range(1, pep_len+1) if i not in comb]
                    comb_path = os.path.join(id_path, '_'.join(map(str, comb)))

                    runner = run_proteinmpnn(
                        PMGen_pdb=pdb_path,
                        output_dir=comb_path,
                        num_sequences_peptide=self.args.num_sequences_peptide,
                        peptide_chain='P',
                        mhc_design=False,
                        peptide_design=True,
                        only_pseudo_sequence_design=False,
                        anchor_pred=False,
                        sampling_temp=self.args.sampling_temp,
                        batch_size=self.args.batch_size,
                        binder_pred=True,
                        fix_anchors=True,
                        anchor_and_peptide=(fixed, peptide),
                        return_match_allele=True
                    )
                    runner.run()

                    df_sampled = pd.read_csv(os.path.join(comb_path, 'peptide_design', 'binder_pred', 'netmhcpan_out.csv'))
                    df_sampled = filter_peptides_from_netmhcpan_csv_output(df_sampled).iloc[:3, :]
                    aff_col = "Affinity(nM)" if "Affinity(nM)" in df_sampled.columns else "Aff(nM)"
                    dataframe_sampled = pd.DataFrame({
                        'peptide': df_sampled.Peptide.tolist(),
                        'mhc_seq': [mhc_seq] * len(df_sampled),
                        'mhc_type': [mhc_type] * len(df_sampled),
                        'anchors': [None] * len(df_sampled),
                        'id': [f"{id}_sampled_{i}_{'_'.join(map(str, comb))}" for i in range(len(df_sampled))],
                        'BA': df_sampled[aff_col].tolist(),
                        'rank_el': df_sampled["%Rank_EL"].tolist(),
                        'mut_id': ['_'.join(map(str, comb))] * len(df_sampled)
                    })

                    if self.args.benchmark:
                        peptide_fasta_file = os.path.join(comb_path, 'peptide_design', 'binder_pred', 'mutants_bench.fa')
                        generate_mutants(peptide, comb, peptide_fasta_file, id, one_indexed=True)
                        df_mut = run_and_parse_netmhcpan(
                            peptide_fasta_file,
                            mhc_type=mhc_type,
                            output_dir=os.path.join(comb_path, 'peptide_design', 'binder_pred'),
                            mhc_seq_list=[],
                            mhc_allele='/'.join(runner.matched_allele),
                            dirty_mode=False, verbose=False,
                            outfilename='netmhcpan_out_mutant_benchmark',
                            return_match_allele=False,
                            match_with_netmhcpan=False
                        )
                        df_mut = filter_peptides_from_netmhcpan_csv_output(df_mut)
                        dataframe_mut = pd.DataFrame({
                            'peptide': df_mut.Peptide.tolist(),
                            'mhc_seq': [mhc_seq] * len(df_mut.Peptide),
                            'mhc_type': [mhc_type] * len(df_mut.Peptide),
                            'anchors': [None] * len(df_mut.Peptide),
                            'id': df_mut.Identity.tolist(),
                            'BA': df_mut[aff_col].tolist(),
                            'rank_el': df_mut["%Rank_EL"].tolist(),
                            'mut_id': ['_'.join(map(str, comb))] * len(df_mut.Peptide)
                        })
                        final_dataframe = pd.concat([dataframe_sampled, dataframe_mut])
                    else:
                        final_dataframe = dataframe_sampled.copy()

                    final_dataframe.to_csv(os.path.join(comb_path, 'comb_output.csv'), index=False)
                    COMB_DF.append(final_dataframe)

                COMB_DF = pd.concat(COMB_DF)
                COMB_DF.to_csv(os.path.join(id_path, f'id_out_{self.mt_num}.csv'), index=False)
                all_dfs.append(COMB_DF)

        if all_dfs:
            return pd.concat(all_dfs)
        else:
            return pd.DataFrame()

    def run_mutation_screen(self):
        if self.args.run == 'parallel':
            with Pool(processes=self.args.max_cores) as pool:
                results = pool.map(self.process_single_id, self.df.id.tolist())
            ALL_DF = pd.concat([r for r in results if not r.empty])
        else:
            # sequential mode
            ALL_DF = []
            for id in self.df.id.tolist():
                df_id = self.process_single_id(id)
                if not df_id.empty:
                    ALL_DF.append(df_id)
            ALL_DF = pd.concat(ALL_DF)

        ALL_DF.to_csv(os.path.join(self.args.output_dir, f'mutation_selection_{self.mt_num}.csv'), index=False)




