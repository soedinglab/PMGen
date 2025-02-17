import pandas
import os
from PANDORA import Target
from PANDORA import Pandora
from PANDORA import Database

class run_parsefold_modeling():
    def __init__(self, peptide, mhc_seq, mhc_type, output_dir,
                 id=None, anchors=None, mhc_allele=None, predict_anchor=True,
                 num_templates=4):
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
        self.input_assertion()
        # input derived args
        self.mhc_type_greek='I' if self.mhc_type==1 else 'II'
        self.m_chain, self.n_chain = (self.mhc_seq + '/').split('/')[0], (self.mhc_seq + '/').split('/')[1]
        self.pandora_output = os.path.join(self.output_dir, 'pandora_outs')
        self.db = Database.load() # load pandora db



    def run_pandora(self, num_templates=4):
        mhc_allele = [] if not self.mhc_allele else mhc_allele=[self.mhc_allele]
        anchor = [] if self.anchors is None else self.anchors
        # run pandora
        target = Target(id=self.id, peptide=self.peptide, allele_type=mhc_allele,
                        MHC_class=self.mhc_type_greek, M_chain_seq=self.m_chain,
                        N_chain_seq=self.n_chain, output_dir=self.pandora_output,
                        use_netmhcpan=self.predict_anchor, anchors=anchor)
        case = Pandora.Pandora(target, self.db)
        case.model(n_loop_models=self.num_templates)


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

