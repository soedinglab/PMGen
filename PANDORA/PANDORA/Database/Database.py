import pickle
import os
import subprocess
import json
from joblib import Parallel, delayed
import argparse

import PANDORA
from PANDORA import Template
from PANDORA import Database_functions


class Database:

    def __init__(self):
        self.MHCI_data = {}
        self.MHCII_data = {}
        self.ref_MHCI_sequences = {}
        self.__IDs_list_MHCI = []
        self.__IDs_list_MHCII = []
        self.reverse = False
        
    def __reverse(self):
        for temp in self.MHCII_data:
            peptide = self.MHCII_data[temp].peptide
            self.MHCII_data[temp].peptide = peptide[::-1]
            self.MHCII_data[temp].anchors = [len(peptide) - anchor + 1 for anchor in self.MHCII_data[temp].anchors][::-1]
            self.MHCII_data[temp].reverse = not self.MHCII_data[temp].reverse
    
    def set_reverse(self, reverse):
        if reverse:
            if not self.reverse:
                self.__reverse()
        else:
            if self.reverse:
                self.__reverse()
        self.reverse = reverse

    def download_data(self, data_dir = PANDORA.PANDORA_data, download = True):
        """download_data(self, data_dir = PANDORA.PANDORA_data, download = True)
        Download all MHC structures and get a two lists that contains all MHCI and MHCII IDs respectively"""

        if download:
            print('Downloading structures ...')
            Database_functions.download_unzip_imgt_structures(data_dir = data_dir, del_inn_files=True, del_kabat_files=True)
        self.__IDs_list_MHCI = Database_functions.download_ids_imgt('MH1', data_dir = data_dir, out_tsv='all_MHI_IDs.tsv')
        self.__IDs_list_MHCII = Database_functions.download_ids_imgt('MH2', data_dir = data_dir, out_tsv='all_MHII_IDs.tsv')


    def clean_MHCI_file(self, pdb_id, data_dir, remove_biopython_object):
        """ Clean all MHCI structures"""
        try: 
            templ = Database_functions.parse_pMHCI_pdb(pdb_id,
                                                    indir = data_dir + '/PDBs/IMGT_retrieved/IMGT3DFlatFiles',
                                                    outdir = data_dir + '/PDBs/pMHCI',
                                                    bad_dir = data_dir + '/PDBs/Bad/pMHCI',
                                                    remove_biopython_object=remove_biopython_object)
            if templ != None:
                #self.MHCI_data[pdb_id] = templ
                return (pdb_id, templ)

        except Exception as e:
            print('something went wrong parsing %s:' %pdb_id)
            print(e)

    def clean_MHCII_file(self, pdb_id, data_dir, remove_biopython_object):
        """ Clean all MHCII structures. Returns a list of bad PDBs"""
        try: 
            templ = Database_functions.parse_pMHCII_pdb(pdb_id,
                                                   indir = data_dir + '/PDBs/IMGT_retrieved/IMGT3DFlatFiles',
                                                   outdir = data_dir + '/PDBs/pMHCII',
                                                   bad_dir = data_dir + '/PDBs/Bad/pMHCII',
                                                   remove_biopython_object=remove_biopython_object)
            if templ != None:
                #self.MHCI_data[pdb_id] = templ
                return (pdb_id, templ)

        except Exception as e:
            print('something went wrong parsing %s' %pdb_id)
            print(e)

    def update_ref_sequences(self):
        """Downloads and parse HLA and other MHC sequences to compile reference fastas.
        Returns a dictionary that can be used to select the desired reference sequence"""
        self.ref_MHCI_sequences = Database_functions.generate_mhcseq_database()

    def construct_database(self, save=PANDORA.PANDORA_data + '/PANDORA_database.pkl', data_dir = PANDORA.PANDORA_data,
                           MHCI=True, MHCII=True, download=True,
                           update_ref_sequences=True, 
                           remove_biopython_objects = True,
                           n_jobs = 1):
        '''construct_database(self, save=PANDORA.PANDORA_data + '/PANDORA_database.pkl', data_dir = PANDORA.PANDORA_data, MHCI=True, MHCII=True, download=True, update_ref_sequences=True, remove_biopython_objects = True, n_jobs = 1)
        Construct the database. Download, clean and add all structures

        Args:
            save (str/bool): Filename of database pkl object. If False, does not save the database pkl. 
                If a path is provided, saved the database .pkl to that path. Defaults to PANDORA.PANDORA_data + '/default/PANDORA_database.pkl'.
            data_dir (str): Path of data directory. Defaults to PANDORA.PANDORA_data.
            MHCI (bool): Parse data for MHCI. Defaults to True.
            MHCII (bool): Parse data for MHCII. Defaults to True.
            download (bool): If True, download the structures data from IMGT. Defaults to True.
            update_ref_sequences (bool): If True, downloads and parse reference sequence strcutres. Defaults to True
            remove_biopython_objects (bool): If True, removes the biopython pdb 
                objects from the template objects to make the database considerably lighter.
                Switch to False only if the biopython objects are necessary. Defaults to True.
            n_jobs (int): number of parallel processes to use. Set to -1 to use all the available cores.
                Defaults to 1.
            
        Returns: Database object

        '''
        #Generate the necessary folders
        create_db_folders()

        # Download the data
        self.download_data(download = download, data_dir = data_dir)

        # Construct the MHCI database
        if MHCI:
            # Parse all MHCI files
            templates = Parallel(n_jobs = n_jobs)(delayed(self.clean_MHCI_file)(id, data_dir, remove_biopython_objects) for id in self.__IDs_list_MHCI)
            templates = [x for x in templates if x != None]
            self.MHCI_data = {key: value for (key, value) in templates}

        # Construct the MHCII database
        if MHCII:
            # Parse all MHCII files
            templates = Parallel(n_jobs = n_jobs)(delayed(self.clean_MHCII_file)(id, data_dir, remove_biopython_objects) for id in self.__IDs_list_MHCII)
            templates = [x for x in templates if x != None]
            self.MHCII_data = {key: value for key, value in templates}

        #Download and parse HLA and MHC sequences reference data
        if update_ref_sequences:
            self.update_ref_sequences()

        self.construct_both_blast_db()

        if save:
            self.save(save)

        print('Database correctly generated')

    def add_structure(self, id, allele_type, peptide = '', 
                      MHC_class = 'I', chain_seq = [], anchors = [], 
                      pdb_path = False, pdb = False, remove_biopython_object=True):
        ''' Add a single structure to the database

        Args:
            id: (str) PDB identifier
            allele_type: (lst) list of MHC alleles (or allele)
            peptide: (str) peptide sequence
            MHC_class: (str) either 'I' or 'II' denoting MHC class I and MHC class II respectively
            chain_seq: (lst) list of chain sequence(s) for the M and N (Alpha and Beta) chain respectively
            anchors: (lst) list of integers specifying which residue(s) of the peptide should be fixed as an anchor
                        during the modelling. MHC class I typically has 2 anchors, while MHC class II typically has 4.
            pdb_path: (str) path to pdb file
            pdb: (Bio.PDB) Biopython PBD object

        '''

        if not id:
            if not pdb_path or not pdb:
                raise ValueError('Structure id or path of .pdb files was not given. Enter value for id and pdb_path')
        # Add to MHCI data
        if MHC_class == 'I':
            self.MHCI_data[id] = Template(id, allele_type, peptide, 
                                               MHC_class, chain_seq, anchors, 
                                               pdb_path, pdb, remove_biopython_object)
        # Add to MHCII data
        if MHC_class == 'II':
            self.MHCII_data[id] = Template(id, allele_type, peptide, 
                                                MHC_class, chain_seq, anchors, 
                                                pdb_path, pdb, remove_biopython_object)


    def write_db_into_fasta(self, outfile):
        """
        Writes structure db into a fasta file (to be later used to build a blast database)

        Args:
            outfile (str): output file path.

        Returns:
            None.

        """

        sequences = []
        for template in self.MHCI_data.values():
            #Get Header and sequence
            header, seq = Database_functions.get_sequence_for_fasta(
                                        template, MHC_class='I', chain='M')
            #Keep only the G-domain
            seq = seq[PANDORA.MHCI_G_domain[0][0]:PANDORA.MHCI_G_domain[0][1]]
            #Append to the list
            sequences.append((header, seq))

        for template in self.MHCII_data.values():
            #Get Header and sequence
            header, seq = Database_functions.get_sequence_for_fasta(
                                        template, MHC_class='II', chain='M')
            #Keep only the G-domain
            seq = seq[PANDORA.MHCII_G_domain[0][0]:PANDORA.MHCII_G_domain[0][1]]
            #Append to the list
            sequences.append((header, seq))

            #Get Header and sequence
            header, seq = Database_functions.get_sequence_for_fasta(
                                        template, MHC_class='II', chain='N')
            #Keep only the G-domain
            seq = seq[PANDORA.MHCII_G_domain[1][0]:PANDORA.MHCII_G_domain[1][1]]
            #Append to the list
            sequences.append((header, seq))

        self.all_sequences = sequences

        with open(outfile, 'w') as outfasta:
            for sequence in sequences:
                header = sequence[0]
                seq = sequence[1]
                outfasta.write('> %s\n' %header)
                outfasta.write('\n'.join(seq[j:j+60] for j in range(0, len(seq), 60)) + '\n')



    def construct_blast_db(self, infile, outpath, db_name):
        """
        Construc blast database for seq based template selection

        Args:
            outpath (str, optional): Data dir folder. Defaults to PANDORA.PANDORA_data.
            db_name (str, optional): Name of the db folder and fasta file. Defaults to 'MHC_blast_db'.
        Returns:
            None.

        """
        # if not os.path.isdir(outpath):
        #     subprocess.check_call('mkdir %s' %outpath, shell=True)

        # out_fasta = outpath+'/'+db_name+'.fasta'
        # self.write_db_into_fasta(outfile=out_fasta)

        subprocess.check_call((' ').join(['makeblastdb','-dbtype','prot',
                                          '-in', infile,'-out',
                                          outpath + '/' + db_name]), shell=True)

    def construct_both_blast_db(self, data_dir=PANDORA.PANDORA_data):

        #Define db name and path
        db_name = 'templates_blast_db'
        outpath = data_dir + '/BLAST_databases/' + db_name
        out_fasta = outpath + '/'+ db_name +'.fasta'

        #Create db directory
        if not os.path.isdir(outpath):
            subprocess.check_call('mkdir %s' %outpath, shell=True)

        #Create .fasta for the db
        self.write_db_into_fasta(outfile=out_fasta)

        #Construct blast database for blast-based sequence-based template selection
        self.construct_blast_db(infile = out_fasta,
                                outpath=outpath,
                                db_name=db_name)

        #Define db name and path
        db_name = 'refseq_blast_db'
        outpath = data_dir + '/BLAST_databases/' + db_name
        out_fasta = outpath + '/' + db_name + '.fasta'

        #Create db directory
        if not os.path.isdir(outpath):
            subprocess.check_call('mkdir %s' %outpath, shell=True)

        #Create .fasta for the db
        command='cat %s/mhcseqs/HLA_cleaned.fasta %s/mhcseqs/MHC_cleaned.fasta > %s' %(data_dir,
                                                                              data_dir,
                                                                              out_fasta)
        subprocess.check_call(command, shell=True)

        #Construct blast database for retriving mhc allele
        self.construct_blast_db(infile=out_fasta,
                                outpath=outpath,
                                db_name=db_name)



    def remove_structure(self, id =''):
        ''' Removes a structure (by id) from the database

        Args:
            id: (str) PDB ID

        '''

        # Remove structure from database
        self.MHCI_data.pop(id, None)
        self.MHCII_data.pop(id, None)

    def save(self, fn = PANDORA.PANDORA_data + '/PANDORA_database.pkl'):
        """Save the database as a pickle file

        :param fn: (str) pathname of file
        """
        with open(fn, "wb") as pkl_file:
            pickle.dump(self, pkl_file)

def load(file_name = PANDORA.PANDORA_data + '/PANDORA_database.pkl'):
    """Loads a pre-generated database


    Args:
        file_name (str): Dabase file name/path. 
            Defaults to PANDORA.PANDORA_data + '/PANDORA_database.pkl'.

    Returns:
        Database.Database: Database object.

    Example:
        >>> db = Database.load('MyDatabase.pkl')

    """
    try:
        with open(file_name, 'rb') as inpkl:
            db = pickle.load(inpkl)
            db.reverse = False
            for temp in db.MHCII_data:
                db.MHCII_data[temp].reverse = False 
        return db
    except FileNotFoundError:
        raise Exception('Database file not found. Are you sure you have it? If not, run Database.construct_database()')


def create_db_folders(db_path=None):
    """Generates the database folders AND the config.json file if absent

    Args:
        db_path (str, optional): Path to the database to generate. If None,
                    it will look for a path provided in the config.json file.
                    Otherwise it will write or overrite the config.json file with 
                    the provided path. Defaults to None.

    Raises:
        Exception: _description_
    """
    config_file = f"{PANDORA.PANDORA_path}/config.json"
    if db_path != None:
        data = {'data_folder_name' : db_path}
        json_object = json.dumps(data)
        with open(f"{PANDORA.PANDORA_path}/config.json", "w") as outfile:
            outfile.write(json_object)
    elif os.path.exists(config_file):
        with open(config_file) as f:
            data = json.load(f)
            db_path = data['data_folder_name']
    else:
        raise Exception('No db_path provided or config.json file found')

    parent_db_path = ('/').join(db_path.split('/')[:-1])
    dirs = [parent_db_path,
            db_path,
            f'{db_path}/mhcseqs', 
            f'{db_path}/BLAST_databases',
            f'{db_path}/PDBs',
            f'{db_path}/PDBs/pMHCI', 
            f'{db_path}/PDBs/pMHCII',
            f'{db_path}/PDBs/Bad', 
            f'{db_path}/PDBs/Bad/pMHCI',
            f'{db_path}/PDBs/Bad/pMHCII', 
            f'{db_path}/PDBs/IMGT_retrieved',
            ]

    for D in dirs:
        if not os.path.isdir(os.path.expanduser(D)):
            try:
                subprocess.check_call(f'mkdir {D}', shell=True)
            except Exception as e:
                print(f'Could not make directory: {D} \n Reason: {e}')
        else:
            print(f'WARNING: folder {D} already exists!')

def fetch_database(db_out_path, db_url='https://surfdrive.surf.nl/files/index.php/s/D8f0n4ulfeZzsmJ/download'):
    """Downloads the pre-generated database.

    Args:
        db_out_path (str): Path to the database to be downloaded,  
            should be pointing at a "PANDORA_databases" folder.
        db_url (str, optional): URL  database. 
            Defaults to 'https://surfdrive.surf.nl/files/index.php/s/D8f0n4ulfeZzsmJ/download'.

    Raises:
        Exception: If the PANDORA_database.pkl file is not found in the destination folder,
            it raises an exception.
    """    

    try:
        parent_db_path = ('/').join(db_out_path.split('/')[:-1])

        print('Downloading pre-built database ...')
        os.popen(f'wget {db_url} -O {parent_db_path}/default.tar.gz').read()
        print('Copying the database')
        os.popen(f'tar -xzvf {parent_db_path}/default.tar.gz -C {parent_db_path}').read()
        os.popen(f'rm {parent_db_path}/default.tar.gz').read()
        print('Checking...')
        if not os.path.exists(f'{db_out_path}/PANDORA_database.pkl'):
            print('Database correctly retrieved')
        else:
            print('ERROR: Something is missing from the retrieved database.')
            print('Please check the path you provided. Use Database.create_db_folders to generate the necessary folders.')
            raise Exception('Missing PANDORA_database.pkl')

    except Exception as e:
        print(f'WARNING: received error while installing database: {e}')
        print('To be able to use PANDORA you will have to generate a new database. Please follow the instructions in the README.')

def install_database(db_path='~/PANDORA_databases/default'):
    """Wrapper to create the database folders and fetch the zenodo database.

    Args:
        db_path (str, optional): Path where to download the database. 
            Defaults to '~/PANDORA_databases/default'.
    """    
    create_db_folders(db_path)
    fetch_database(db_out_path=db_path)
