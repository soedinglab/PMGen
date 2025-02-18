import pandas as pd
import sys
sys.path.append("PANDORA")
import os
from PANDORA import Target
from PANDORA import Pandora
from PANDORA import Database
import glob
from utils import processing_functions
import pandas as pd
import subprocess

import warnings
# Suppress the specific warning
#warnings.filterwarnings("ignore")


class run_parsefold_modeling():
    def __init__(self, peptide, mhc_seq, mhc_type, id, output_dir='output',
                  anchors=None, mhc_allele=None, predict_anchor=True,
                 num_templates=4, num_recycles=3, models=['model_2_ptm'],
                 alphafold_param_folder = 'AFfine/af_params/params_original/',
                 fine_tuned_model_path='AFfine/af_params/params_finetune/params/model_ft_mhc_20640.pkl'
                 ):
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
        # vars defined later
        self.template_id = None

    def run_parsefold(self, test_mode=False):
        os.makedirs(self.output_dir, exist_ok=True)
        self.template_id = self.run_pandora()
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
        alphafold_input_file = os.path.join(self.alphafold_out, 'alphafold_input_file')
        os.makedirs(self.alphafold_out, exist_ok=True)
        self.alphafold_preparation(template_aln_file=aln_output_file, mhc_pep_seq=mhc_pep_seq, output=alphafold_input_file)
        self.run_alphafold(input_file=alphafold_input_file, output_prefix=self.alphafold_out + f'/{self.id}_')

    def run_pandora(self):
        os.makedirs(self.pandora_output, exist_ok=True)
        mhc_allele = [] if not self.mhc_allele else [self.mhc_allele]
        anchor = [] if self.anchors is None else self.anchors
        # run pandora
        target = Target(id=self.id, peptide=self.peptide, allele_type=mhc_allele,
                        MHC_class=self.mhc_type_greek, M_chain_seq=self.m_chain,
                        N_chain_seq=self.n_chain, output_dir=self.pandora_output,
                        use_netmhcpan=self.predict_anchor, anchors=anchor)
        # save ind self.pandora_output/self.id
        case = Pandora.Pandora(target, self.db)
        case.model(n_loop_models=self.num_templates)
        # get template id used in pandora
        files = [file for file in glob.glob(os.path.join(self.pandora_output, self.id, '????.pdb')) if "mod" not in file.split("/")[-1]]
        template_id = files[0].split("/")[-1]
        return template_id

    def alignment_without_peptide(self, template_id, output_path, template_path,
                                       template_csv_path="data/all_templates.csv"):
        '''
        used for empty mhc pocket prediction
        :param template_id: str, PDB id used as template for homology modeling.
        :param output_path: str, save output path.
        :param template_path: str, full template path.
        :param template_csv_path: str, path to csv file containing all templates and their sequences.
        :return: alignment dataframe, saved alignment files at given path.
        '''
        os.makedirs(self.alignment_output, exist_ok=True)
        df = processing_functions.prepare_alignment_file_without_peptide(template_id=template_id,
                                                                    mhc_seq=self.mhc_seq,
                                                                    mhc_type=self.mhc_type,
                                                                    output_path=output_path,
                                                                    template_path=template_path,
                                                                    template_csv_path=template_csv_path)
        return df

    def alignment_with_peptide(self, pdb_files, mhc_pep_seq, output_path):
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
        df = pd.DataFrame({"target_chainseq": [mhc_pep_seq],
                           "templates_alignfile": [template_aln_file],
                           "targetid": [self.id]})
        df.to_csv(output, sep='\t', index=False)

    def run_alphafold(self, input_file, output_prefix):
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
            "--model_names", f"{model_names}",
            "--model_params_files", f"{model_params_files}",
            "--ignore_identities"
        ]
        print(command)
        try:
            # Run the command and stream output line by line
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
            print("### Running AFopt/run_prediction.py ###")
            # Print stdout and stderr in real-time
            for line in process.stdout:
                print(line, end="")  # Print each line immediately
            for err_line in process.stderr:
                print(err_line, end="")  # Print stderr lines immediately
            # Wait for the process to complete
            exit_code = process.wait()
            if exit_code == 0:
                print("\n✔ Alphafold executed successfully!")
            else:
                print("\n❌ Command failed with exit code:", exit_code)
        except Exception as e:
            print("\n❌ Error running command:", str(e))

    def input_assertion(self):
        assert isinstance(self.peptide, str), f"peptide must be a string, found: {self.peptide}"
        assert self.mhc_type in [1, 2], f"mhc_seq must be an integer value of 1 or 2, found: {self.peptide}"
        assert isinstance(self.mhc_seq, str), f"mhc_seq must be a string, found: {self.mhc_seq}"
        assert isinstance(self.output_dir, str), f"output_dir must be a string, found: {self.output_dir}"
        if self.anchors:
            assert isinstance(self.anchors, (tuple, list)), (f"anchors must be a tuple or list, found: {self.anchors}"
                                                             f"alternatively use predict_anchor==True")
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


