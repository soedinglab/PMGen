from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
import PANDORA
from PANDORA import Contacts
from PANDORA import Modelling_functions
from PANDORA import Anchors
from abc import ABC, abstractmethod
import os
import re

class PMHC(ABC):

    def __init__(self, id, peptide = '', allele_type = [], MHC_class = 'I',
                 M_chain_seq = '', B2M_seq='', N_chain_seq = '', anchors = [],
                 helix=False, sheet=False):
        ''' pMHC class. Acts as a parent class to Template and Target

        Args:
            id: (string) PDB identifier
            allele_type: (list) list of MHC alleles (or allele)
            peptide: (string) peptide sequence
            MHC_class: (string) either 'I' or 'II' denoting MHC class I and MHC class II respectively
            M_chain_seq: (string) M chain sequence for the Alpha chain
            N_chain_seq: (string) N chain sequence for the Beta chain
            anchors:  (list) list of integers specifying which residue(s) of the peptide should be fixed as an anchor
                        during the modelling. MHC class I typically has 2 anchors, while MHC class II typically has 4.
        '''
        super().__init__()
        self.id = id
        self.MHC_class = MHC_class
        self.peptide = peptide
        self.M_chain_seq = M_chain_seq
        self.B2M_seq = B2M_seq
        self.N_chain_seq = N_chain_seq
        self.anchors = anchors
        self.helix = helix
        self.sheet = sheet

        if type(allele_type) == list:
            self.allele_type = allele_type
        elif type(allele_type) == str:
            self.allele_type = [allele_type]
        else:
            raise Exception('The provided allele_type should be a string or a list of strings')


        @abstractmethod
        def info(self):
            pass

        @abstractmethod
        def calc_contacts(self):
            pass

        @abstractmethod
        def calc_anchor_contacts(self):
            pass


class Template(PMHC):

    def __init__(self, id, peptide='',  allele_type=[], MHC_class='I',
                 M_chain_seq='', B2M_seq='', N_chain_seq='', anchors=[], G_domain_span=False,
                 helix=False, sheet=False, pdb_path=False, pdb=False,
                 remove_biopython_object=False, reverse= False):
        ''' Template structure class. This class holds all information of a template structure that is used for
            homology modelling. This class needs a id, allele and the path to a pdb file to work. (sequence info of
            the chains and peptide can be fetched from the pdb)

        Args:
            id: (string) PDB identifier
            allele_type: (list) list of MHC alleles (or allele)
            peptide: (string) peptide sequence
            MHC_class:  (string) either 'I' or 'II' denoting MHC class I and MHC class II respectively
            M_chain_seq: (string) M chain sequence for the Alpha chain
            N_chain_seq: (string) N chain sequence for the Beta chain
            anchors: (list) list of integers specifying which residue(s) of the peptide should be fixed as an anchor
                        during the modelling. MHC class I typically has 2 anchors, while MHC class II typically has 4.
            G_domain_span (list): span of the G domain(s) over the sequence. The format should be [(1, 90),(1, 86)]
            pdb_path: (string) path to pdb file. The user should provide this argument if they want to use
                a template outside the normal templates folder.
            pdb: (Bio.PDB) Biopython PBD object
        '''
        super().__init__(id, peptide=peptide, allele_type=allele_type,
                         MHC_class=MHC_class, M_chain_seq=M_chain_seq,
                         B2M_seq=B2M_seq, N_chain_seq=N_chain_seq, anchors=anchors,
                         helix=helix, sheet=sheet)
        self.id = id
        self.pdb = pdb
        self.contacts = False

        if not G_domain_span:
            if self.MHC_class == 'I':
                self.G_domain_span=PANDORA.MHCI_G_domain
            elif self.MHC_class=='II':
                self.G_domain_span=PANDORA.MHCII_G_domain

        if type(self.allele_type) == str:
            self.allele_type = [self.allele_type]

        self.check_allele_name()
        self.reverse = reverse

        if not os.path.isfile(self.get_pdb_path()) and not pdb:
            print(f'ERROR:file {self.get_pdb_path()} not found')
            raise Exception(f'Provide a PDB structure to the Template object first.')

        if not pdb: # If the path to a pdb file or a Bio.PDB object is given, parse the pdb
            self.parse_pdb()

        if anchors == []:
            self.calc_anchors()

        #Remove self.pdb as it's not useful anymore and takes a lot of memory
        if remove_biopython_object:
            self.pdb = None


    def get_pdb_path(self):
        if not self.reverse:
           return os.path.join(PANDORA.PANDORA_data, 'PDBs', f'pMHC{self.MHC_class}', f'{self.id}.pdb')
        else:
           return os.path.join(PANDORA.PANDORA_data, 'PDBs', f'pMHC{self.MHC_class}_reversed', f'{self.id}_reverse.pdb')
        

    def parse_pdb(self, custom_map={"MSE":"M"}):
        '''Loads pdb from path, updates self.pdb field and self.chain_seq/self.peptide if they were empty

        Args:
            custom_map (dict): custom map of 3-letter to 1-letter residues translation,
                                used by Bio.SeqUtiles.seq1 to decide how to assign
                                non-canonical residues. Defaults to {"MSE":"M"}.
                '''

        if self.get_pdb_path() and not self.pdb: #if there is a path to a pdb provided and there is not already a self.pdb...
            parser = PDBParser(QUIET=True)  # Create a parser object, used to read pdb files
            self.pdb = parser.get_structure('MHC', self.get_pdb_path()) #Create Bio.PDB object

        # If the chains or peptide are not given by the user, fetch them from the pdb
        # Get the chain sequences
        chain_seqs = [seq1(''.join([res.resname for res in chain]),
                           custom_map=custom_map) for chain in self.pdb.get_chains()]
        # Update chain and peptide fields if emtpy
        if self.MHC_class == 'I':
            if self.M_chain_seq == '':
                self.M_chain_seq = chain_seqs[0]
            if self.B2M_seq == '':
                self.B2M_seq = chain_seqs[1]
            if not self.peptide:
                self.peptide = chain_seqs[-1]
        if self.MHC_class == 'II':
            if self.M_chain_seq == '' and self.N_chain_seq == '':
                self.M_chain_seq = chain_seqs[0]
                self.N_chain_seq = chain_seqs[1]
            if not self.peptide:
                self.peptide = chain_seqs[-1]

    def info(self):
        """ Print the basic info of this structure

        """
        print('This is a %s structure.' %(type(self).__name__))
        print('ID: %s' %self.id)
        print('Type: MHC class %s' %self.MHC_class)
        print('Alleles: %s' % self.allele_type)
        if self.M_chain_seq != '':
            print('Alpha chain length: %s' %len(self.M_chain_seq))
        if self.N_chain_seq != '' and self.MHC_class == 'II':
            print('Beta chain length: %s' %len(self.N_chain_seq))
        print('Peptide length: %s' %len(self.peptide))
        if self.M_chain_seq != '':
            print('Alpha chain: %s' % self.M_chain_seq)
        if self.N_chain_seq != '':
            print('Beta chain: %s' % self.N_chain_seq)

        print('Peptide: %s' % self.peptide)
        print('Anchors: %s' %self.anchors)

        if self.sheet:
            print('Beta-sheet: %s' % self.sheet)
        if self.helix:
            print('Alpha-helix: %s' % self.helix)
        if self.get_pdb_path():
            print('Path to PDB file: %s' %self.get_pdb_path())
        if not self.pdb:
            print('Biopython PDB structure: no PDB loaded')
        else:
            print('Biopython PDB structure:')
            for (k, v) in self.pdb.header.items():
                print('\t'+k + ':', v)

    def calc_contacts(self):
        if self.pdb:
            self.contacts = Contacts.Contacts(self.pdb)
        else:
            raise Exception('Provide a PDB structure to the Template object first')

    def calc_anchors(self):
        if self.MHC_class == 'I':
            self.anchors = Anchors.pMHCI_anchors(self.pdb)
        if self.MHC_class == 'II':
            self.anchors = Anchors.pMHCII_anchors(self.pdb)

    def calc_anchor_contacts(self):
        if self.pdb and self.anchors:
            self.anchor_contacts = Contacts.Contacts(self.pdb, anchors=self.anchors).anchor_contacts
        else:
            raise Exception('Provide an initial model (.ini PDB) and anchor positions to the Target object first')

    def check_allele_name(self):
        """
        Checks the spell of the allele name and tried to correct it.
        Prints out warning if the allele name does not seem correct.
        """
        #if self.MHC_class == 'I':
        for i, allele in enumerate(self.allele_type):
            regexp = re.search(r'([A-Z]{1}[a-z]{3}|[A-Z]{3})[-][A-Z0-9]{0,4}[*][0-9]{2,3}[:][0-9]{2,3}',allele)
            if regexp is not None:
                #if the allele name is valid
                pass
            else:
                regexp = re.search(r'([A-Z]{1}[a-z]{3}|[A-Z]{3})[-][A-Z0-9]{0,4}[*][0-9]{4,6}',allele)
                if regexp is not None:
                    print('WARNING: Allele name missing ":"')
                    print(regexp.group(0))
                    print('PANDORA will try to correct the allele name.')
                    if len(allele.split('*')[-1]) <=5:
                        new_allele = allele[:-2] + ':' + allele[-2:]
                    else:
                        new_allele = allele[:-3] + ':' + allele[-3:]

                    print('New attempted allele name: ' + new_allele)
                    print('Is this allele name correct?')
                    self.allele_type[i] = new_allele
                else:
                    print('WARNING: Allele name syntax not recognized for allele', allele)
                    print('The allele will not be changed')


class Target(PMHC):

    def __init__(self, id, peptide, allele_type=[], MHC_class = 'I',
                 M_chain_seq = '', N_chain_seq = '', 
                 B2M_seq='', anchors = [],
                 helix=False, sheet=False, templates = False,
                 use_netmhcpan = False, use_templ_seq=False, output_dir=False,
                 rm_netmhcpan_output=True, reverse=False):
        ''' Target structure class. This class needs an ID (preferably a PDB ID), allele and pepide information.

        Args:
            id: (string) PDB identifier
            peptide: (string) peptide sequence
            allele_type: (list) list of MHC alleles (or allele)
            MHC_class: (string) either 'I' or 'II' denoting MHC class I and MHC class II respectively
            M_chain_seq: (string) M chain sequence for the Alpha chain
            N_chain_seq: (string) N chain sequence for the Beta chain
            anchors: (list) list of integers specifying which residue(s) of the peptide should be fixed as an anchor
                        during the modelling. MHC class I typically has 2 anchors, while MHC class II typically has 4.
            templates: Template object. The user can specify that PANDORA uses a certain structure as template.
            use_netmhcpan (bool): If True, uses local installation of NetMHCPan to predict the anchors when
                                  anchor positions are not provided. Defaults to False.
            use_templ_seq (bool): If True, if no MHC chain sequences could be retrieved starting from the allele name,
                                   it will use the best template MHC sequences for the modelling.
            output_dir: (string) Path to output directory. Defaults to current working directory.
            rm_netmhcpan_output: (bool) If True, removes the netmhcpan infile and outfile after having used them for netmhcpan.
        ''' 
        
        anchors.sort()
        super().__init__(id, peptide=peptide, allele_type=allele_type, 
                         MHC_class=MHC_class, M_chain_seq=M_chain_seq, 
                         N_chain_seq=N_chain_seq, B2M_seq=B2M_seq, 
                         anchors=anchors, helix=helix, sheet=sheet)
        self.templates = templates
        self.initial_model = False
        self.contacts = False
        self.anchor_contacts = False
        self.reverse = reverse

        # Changes all special characters in the case id to '-'
        self.id = re.sub('[^a-zA-Z0-9 \n\.]', '_', id)

        if output_dir == False:
            self.output_dir = os.getcwd()
        else:
            self.output_dir = output_dir

        self.output_dir = f"{self.output_dir}/{self.id}"

        # Output directory is created
        self.make_output_dir()

        # If the user does provide sequence info, make sure both the M and N chain are provided
        # if MHC_class == 'II' and M_chain_seq != '' and N_chain_seq == '':
        #     raise Exception('Provide both the M and N chain sequences for MHC class II targets or none at all')
        # if MHC_class == 'II' and N_chain_seq != '' and M_chain_seq == '':
        #     raise Exception('Provide both the M and N chain sequences for MHC class II targets or none at all')

        if M_chain_seq != '' and allele_type == []:
            pass #retrieve allele_type from blast db
        if MHC_class == 'II' and N_chain_seq != '' and allele_type == []:
             pass #retrieve allele_type from blast db

        self.fill_allele_seq_info(use_templ_seq=use_templ_seq)

        # If anchors are not provided, predict them from the peptide length
        if MHC_class =='I' and anchors == []:
            #Use Canonical anchors
            if use_netmhcpan == False:
                print('WARNING: no anchor positions provided. Pandora will assign them to canonical anchor position.')
                print('If you want PANDORA to use NetMHCpan to predict the anchors set use_netmhcpan as True')
                anchor_1 = 2
                anchor_2 = len(peptide)
                anchors = [anchor_1, anchor_2]
                self.anchors = anchors
            #Use NetMHCpan to predict the anchors
            else:
                print('WARNING: no anchor positions provided. Pandora will predict them using NetMHCpan')
                # predict the anchors
                try:
                    self.anchors = Modelling_functions.predict_anchors_netMHCpan(self.peptide, self.allele_type, self.output_dir, rm_netmhcpan_output=rm_netmhcpan_output)
                    print('Predicted anchors: %s' %self.anchors)
                except Exception as e:
                    print('Error: Something went wrong when predicting the anchors using netMHCpan')
                    raise Exception(e)

        if MHC_class =='II' and anchors == []:
            print('WARNING: no anchor positions provided. Pandora will predict them using netMHCIIpan.')
            # predict the anchors
            try:
                self.anchors = Modelling_functions.predict_anchors_netMHCIIpan(self.peptide, self.allele_type, self.output_dir, rm_netmhcpan_output=rm_netmhcpan_output)
            except Exception as e:
                print('Error: Something went wrong when predicting the anchors using netMHCIIpan')
                raise Exception(e)

        print('###############################################')
        self.info()

    def info(self):
        """ Print the basic info of this structure

        """
        print('This is a %s structure.' % (type(self).__name__))
        print('ID: %s' % self.id)
        print('Type: MHC class %s' % self.MHC_class)
        print('Alleles: %s' % self.allele_type)
        if self.M_chain_seq != '':
            print('Alpha chain length: %s' %len(self.M_chain_seq))
        if self.N_chain_seq != '' and self.MHC_class == 'II':
            print('Beta chain length: %s' %len(self.N_chain_seq))
        print('Peptide length: %s' %len(self.peptide))
        if self.M_chain_seq != '':
            print('Alpha chain: %s' % self.M_chain_seq)
        if self.N_chain_seq != '':
            print('Beta chain: %s' % self.N_chain_seq)
        print('Peptide: %s' % self.peptide)
        print('Anchors: %s' % self.anchors)
        if self.sheet:
            print('Beta-sheet: %s' % self.sheet)
        if self.helix:
            print('Alpha-helix: %s' % self.helix)
        if self.templates:
            print('Using template %s for homology modelling' %self.templates)
        if self.initial_model:
            print('An initial model has been provided.')

    def calc_contacts(self):
        if self.initial_model:
            self.contacts = Contacts.Contacts(self.initial_model)
        else:
            raise Exception('Provide a PDB structure to the Template object first')

    def calc_anchor_contacts(self):
        if self.initial_model and self.anchors:
            self.anchor_contacts = Contacts.Contacts(self.initial_model, anchors=self.anchors).anchor_contacts
        else:
            raise Exception('Provide an initial model (.ini PDB) and anchor positions to the Target object first')

    def check_allele_name(self):
        """
        Checks the spell of the allele name and tried to correct it.
        Prints out warning if the allele name does not seem correct.
        """
        #if self.MHC_class == 'I':
        for i, allele in enumerate(self.allele_type):
            regexp = re.search(r'([A-Z]{1}[a-z]{3}|[A-Z]{3})[-][A-Z0-9]{0,4}[*][0-9]{2,3}[:][0-9]{2,3}',allele)
            if regexp is not None:
                #if the allele name is valid
                pass
            else:
                regexp = re.search(r'([A-Z]{1}[a-z]{3}|[A-Z]{3})[-][A-Z0-9]{0,4}[*][0-9]{4,6}',allele)
                if regexp is not None:
                    print('\nWARNING: Allele name missing ":"')
                    print(regexp.group(0))
                    print('PANDORA will try to correct the allele name.')
                    if len(allele.split('*')[-1]) <=5:
                        new_allele = allele[:-2] + ':' + allele[-2:]
                    else:
                        new_allele = allele[:-3] + ':' + allele[-3:]

                    print('New attempted allele name: ' + new_allele)
                    print('Is this your allele?')
                    self.allele_type[i] = new_allele
                else:
                    print('\nWARNING: Allele name syntax not recognized')

    def retrieve_MHC_refseq(self, input_file = None, chain='M', permissive=False):
        """
        Retrieves MHC reference sequence from fasta file.

        Args:
            input_file (str, optional): Path to the input reference fasta file. Defaults to None.
            chain (str): ID of the chain to be retrieved (M or N). Defaults to 'M'.
            permissive (bool): If True, if no chain of the exact allele can be found, 
                it will try to retrieve a chain from the same allele subgroup. Defaults to False.

        Returns:
            None.

        """

        # Define correct fasta file
        if input_file == None:
            if self.allele_type[0].startswith('HLA'):
                input_file = PANDORA.PANDORA_data+ '/mhcseqs/HLA_cleaned.fasta'
            else:
                input_file = PANDORA.PANDORA_data+ '/mhcseqs/MHC_cleaned.fasta'

        # Parse Fasta file
        fasta_sequences = SeqIO.parse(input_file,'fasta')
        ref_sequences = {seq.id : str(seq.seq) for seq in fasta_sequences}

        if chain == 'M':
            alleles = [x for x in self.allele_type if any(y in x for y in PANDORA.alpha_genes)]
        elif chain == 'N':
            alleles = [x for x in self.allele_type if any(y in x for y in PANDORA.beta_genes)]
        # Return the right sequences
        seq_flag = False
        #for seq in fasta_sequences:
        for seq in ref_sequences:
            if any(allele in seq for allele in alleles):
                if chain == 'M':
                    self.M_chain_seq = ref_sequences[seq]
                    seq_flag = True
                elif chain == 'N':
                    self.N_chain_seq = ref_sequences[seq]
                    seq_flag = True
                break
            else:
                pass

        if seq_flag == False and permissive:
            #fasta_sequences = SeqIO.parse(input_file,'fasta')
            available_alleles = [seq for seq in ref_sequences]
            #print('input', input_file)
            #print('DQA1: ', [x for x in available_alleles if 'HLA-DQA1*05' in x])
            corrected_alleles = Modelling_functions.allele_name_adapter(self.MHC_class,
                                                                        alleles,
                                                                        available_alleles)
            #print('CORRECTED ALLELES: ', corrected_alleles)
            for seq in ref_sequences:
                if any(allele  in seq for allele in corrected_alleles):
                    if chain == 'M':
                        self.M_chain_seq = ref_sequences[seq]
                        seq_flag = True
                    elif chain == 'N':
                        self.N_chain_seq = ref_sequences[seq]
                        seq_flag = True
                    break
                else:
                    pass


        if seq_flag == True:
            print('Chain %s MHC sequence correctly retrieved' %chain)
        else:
            print('\nWARNING: No MHC seq could be retrieved with the given MHC allele name')
            print('Your MHC allele might be missing from the IPDMHC/IMGTHLA database.')
            raise Exception('No MHC chain corresponding to allele name found')
            #print("The model will be generated using the best template's MHC sequence.")
            #print("To be sure you use the right sequence, please double check your MHC allele name")
            #print("or provide the MHC sequence as target.M_chain_seq (and target.N_chain_seq for MHCII beta chain)")

    def fill_allele_seq_info(self, use_templ_seq=False):
        """Fills in MHC-II alpha chain name if missing and it tries to retireve the
        sequence according to the allele name of vice versa

        Args:
            use_templ_seq (bool, optional): If true, it uses the template MHC sequence 
                for each chain a sequence could not be found. This function is
                mainly for benchmarking reason. Defaults to False.

        Raises:
            Exception: _description_
            Exception: _description_
            Exception: _description_
        """
        if self.allele_type:
            # Check allele name
            self.check_allele_name()


        #Check if there are allele name for each MHC chain
        M_allele_flag = False
        N_allele_flag = False
        if any(x in y for x in PANDORA.alpha_genes for y in self.allele_type):
            M_allele_flag = True
        if self.MHC_class == 'II':
            if any(x in y for x in PANDORA.beta_genes for y in self.allele_type):
                N_allele_flag = True
                # In the case of HLA-DRB, add HLA-DRA alpha chain
                if any('HLA-DRB' in y for y in self.allele_type) and M_allele_flag == False:
                    self.allele_type.extend(['HLA-DRA*01:01', 'HLA-DRA*01:02'])
                    M_allele_flag = True
                    print('\nWARNING: chain Beta allele name found only.')
                    print('PANDORA will assume chain alpha is HLA-DRA*01')

        #Check if there are allele names for each MHC chain
        if self.M_chain_seq =='' and M_allele_flag:
            print('\nNo MHC alpha chain sequence was provided. Trying to retrieve it from reference sequences...')
            try:
                self.retrieve_MHC_refseq(chain='M')
            except:
                print('\nWARNING: Something went wrong while retrieving the reference sequence.')
                print('Please provide a M_chain_seq for your target.')
                print('###################')
                print('You can find all the reference MHC sequences used in PANDORA')
                print(f' and use them for your target in {PANDORA.PANDORA_data}/mhcseqs')
                print('###################')
                print('All the HLA sequences are downloaded from:')
                print('https://www.ebi.ac.uk/ipd/imgt/hla/')
                print('Non-human MHC sequences are downloaded from:')
                print('https://www.ebi.ac.uk/ipd/mhc/')
                if not use_templ_seq:
                    raise Exception('No MHC chain available')
                else:
                    print('###################')
                    print('PANDORA will try to model case %s by using the best template M chain sequence' %self.id)
        if self.M_chain_seq =='' and not M_allele_flag:
            print('\nWARNING: Missing M chain (Alpha chain) sequence and allele name.')
            if not use_templ_seq:
                raise Exception('No MHC chain available')
            else:
                print('PANDORA will try to model case %s by using the best template M chain sequence' %self.id)
                print('We strongly advice to provide either allele name or chain sequence for chain M')

        if self.M_chain_seq !='' and not M_allele_flag:
            print('\nNo MHC alpha chain allele was provided. Trying to retrieve it from reference sequences...')
            #Blast against reference database
            try:
                blast_results = Modelling_functions.blast_mhc_seq(self.M_chain_seq,
                                                                  chain='M',
                                                                  blastdb=PANDORA.PANDORA_data + '/BLAST_databases/refseq_blast_db/refseq_blast_db')
                #Take only the allele names with the highest id score
                top_id = blast_results[0][1]
                self.allele_type.extend([x[0] for x in blast_results if x[1] == top_id])
            except:
                print('\nWARNING: something went wrong when trying to retrieve chain M allele')
                print('with blast. Is blastp properly installed as working as "/bin/bash blastp"?')

        if self.MHC_class == 'II' and self.N_chain_seq =='' and N_allele_flag:
            print('\nNo MHC sequence was provided. Trying to retrieve it from reference sequences...')
            try:
                self.retrieve_MHC_refseq(chain='N')
            except:
                print('Something went wrong while retrieving the reference sequence.')
                print('Please provide a N_chain_seq for your target.')
                print('###################')
                print('You can find all the reference MHC sequences used in PANDORA')
                print(f' and use them for your target in {PANDORA.PANDORA_data}/mhcseqs')
                print('###################')
                print('All the HLA sequences are downloaded from:')
                print('https://www.ebi.ac.uk/ipd/imgt/hla/')
                print('Non-human MHC sequences are downloaded from:')
                print('https://www.ebi.ac.uk/ipd/mhc/')
                if not use_templ_seq:
                    raise Exception('No MHC chain available')
                else:
                    print('###################')
                    print('PANDORA will try to model case %s by using the best template N chain sequence' %self.id)
        elif self.MHC_class == 'II' and self.N_chain_seq =='' and not N_allele_flag:
                print('\nWARNING: Missing N chain (Beta chain) sequence and allele name.')
                print('PANDORA will try to model case %s by using the best template N chain sequence' %self.id)
                print('We strongly advice to provide either allele name or chain sequence for chain N')

        if self.MHC_class == 'II' and self.N_chain_seq !='' and not N_allele_flag:
            print('No MHC alpha chain allele was provided. Trying to retrieve it from reference sequences...')
            #Blast against reference database
            try:
                blast_results = Modelling_functions.blast_mhc_seq(self.N_chain_seq,
                                                                  chain='N',
                                                                  blastdb=PANDORA.PANDORA_data + '/BLAST_databases/refseq_blast_db/refseq_blast_db')
                #Take only the allele names with the highest id score
                top_id = blast_results[0][1]
                self.allele_type.extend([x[0] for x in blast_results if x[1] == top_id])
            except:
                print('\nWARNING: something went wrong when trying to retrieve chain M allele')
                print('with blast. Is blastp properly installed as working as "/bin/bash blastp"?')

    def make_output_dir(self):
        ''' Create an output directory and move the template pdb there
            Uses self.output_dir (str): Path to output directory. Defaults to os.getcwd().
        Args:
            None
        '''

        # create an output directory
        try:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                if not os.path.exists(self.output_dir):
                    raise Exception('A problem occurred while creating output directory')
        except:
            raise Exception('A problem occurred while creating output directory')
