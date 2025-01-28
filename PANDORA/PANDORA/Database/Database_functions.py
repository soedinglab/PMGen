import os
import subprocess
import urllib.request
import urllib.parse
from copy import deepcopy
from Bio.PDB import PDBParser
from Bio.PDB import PDBIO
from Bio.PDB import parse_pdb_header
from Bio import SeqIO
import gzip
import shutil
import PANDORA
from PANDORA import Contacts
from PANDORA import Template
from Bio.PDB import NeighborSearch
from Bio.SeqUtils import seq1
from Bio.PDB import Chain
from string import ascii_uppercase


def fresh_parse_dirs():
    ''' Removes all parsed files + log files for a fresh start. Used for development.

    '''

    dirs_to_clean = ['/PDBs/pMHCI', '/PDBs/pMHCII', '/PDBs/Bad/pMHCI', '/PDBs/Bad/pMHCII']
    files = [PANDORA.PANDORA_data + '/PDBs/Bad/log_MHCI.csv',
             PANDORA.PANDORA_data + '/PDBs/Bad/log_MHCII.csv']

    for d in dirs_to_clean:
        files = files + [PANDORA.PANDORA_data + d + '/' + i for i in os.listdir(PANDORA.PANDORA_data + d)]

    cnt = 0
    for f in files:
        if os.path.exists(f):
            os.remove(f)
            cnt+=1
    print('Removed %s files' %cnt)



def download_unzip_imgt_structures(data_dir = PANDORA.PANDORA_data,
                                    del_inn_files = True, del_kabat_files = True):
    ''' download_unzip_imgt_structures(data_dir = PANDORA.PANDORA_data, del_inn_files = True, del_kabat_files = True)
    Downloads the complete structural dataset

    Args:
        data_dir: (string) path of data directory
        del_inn_files: (bool) if True (default) deletes all inn files
        del_kabat_files: (bool) if True (default) deletes all kabat files

    '''

    # Changing working directory
    os.chdir(data_dir + '/PDBs/IMGT_retrieved/')
    # Downloading IMGT dataset
    os.system('wget https://www.imgt.org/download/3Dstructure-DB/IMGT3DFlatFiles.tgz')
    # Uncompressing
    os.system('gunzip IMGT3DFlatFiles.tgz')
    os.system('tar -xvf IMGT3DFlatFiles.tar')

    try:
        os.system(f'rm {data_dir}/PDBs/IMGT_retrieved/IMGT3DFlatFiles.tgz')
    except:
        pass
    os.system(f'rm {data_dir}/PDBs/IMGT_retrieved/IMGT3DFlatFiles.tar')
    # Removing non-PDB files
    if del_inn_files:
        os.system(f'rm {data_dir}/PDBs/IMGT_retrieved/IMGT3DFlatFiles/*.inn.gz')
    if del_kabat_files:
        os.system(f'rm {data_dir}/PDBs/IMGT_retrieved/IMGT3DFlatFiles/*.prot.gz')
    os.chdir(PANDORA.PANDORA_path)


def download_ids_imgt(ReceptorType, data_dir = PANDORA.PANDORA_data, out_tsv = False):
    ''' download_ids_imgt(ReceptorType, data_dir = PANDORA.PANDORA_data, out_tsv = False)
    Queries IMGT with the ReceptorType for PDBs.

    Args:
        ReceptorType: (string) Receptor query for IMGT: 'MH1' or 'MH2'
        data_dir: (bool/string) if not False, produces a tsv file names as out_tsv
        out_tsv: (bool/string) if not False, produces a tsv file names as out_tsv

    Returns: (list) Returns the list of IDs provided by IMGT.

    '''
    
    params = urllib.parse.urlencode({ 'ReceptorType' : ReceptorType,
             'type-entry': 'PDB'})

    url = "https://www.imgt.org/3Dstructure-DB/cgi/3Dquery.cgi?%s" % params

    with urllib.request.urlopen(url) as response:
        text = response.read().decode('utf-8')
        text = text.splitlines()

    IDs_list = []
    IDs_list = [x for x in text if 'href' in x and 'pdbcode' in x]
    IDs_list = [x.split('"') for x in IDs_list]
    IDs_list = [x[3][-4:] for x in IDs_list]

    if out_tsv:
        outfile = open(f'{data_dir}/{out_tsv}', 'w')
        outfile.write(ReceptorType + ' IMGT IDs\n')
        for ID in IDs_list:
            outfile.write(ID + '\n')
        outfile.close()

    return IDs_list


def get_chainid_alleles_MHCI(pdbf):
    '''    Takes as input an IMGT preprocessed PDB file of p:MHC I.
           Returns a dictionary containing alleles and relative identity scores for each
           G-domain in the given pdb from the REMARK.

    Args:
        pdbf: (string) path to IMGT pdb file

    Returns: (dict) MHCI alleles

    '''

    # test: multiple chains 3GJG, multiple alleles 1AO7
    ### Parsing file and extracting remarks
    with open(pdbf) as infile:
        remarks = []
        for line in infile:
            if line.startswith('REMARK 410'):
                row = [x for x in line.rstrip().split(' ') if x != '']
                del row[:2]
                remarks.append(row)
    remarks = [x for x in remarks if x != []]

    ### Dividing each remark section into a chains dictionary
    chains = {}
    flag = False
    for row in remarks:
        if row[0] == 'Chain' and row[1] == 'ID' and len(row) == 4:
            chainID = row[2][-1]
            chains[chainID] = []
            chains[chainID].append(row)
            flag = True
        elif flag == True:
            chains[chainID].append(row)

    ### Extracting MHC I Alpha chains
    mhc_a = {}  # MHC I Alpha
    for chain in chains:
        try:
            if chains[chain][1][3] == 'I-ALPHA':
                mhc_a[chain] = chains[chain]
        except:
            pass

    ### Extracting alleles info
    mhc_a_alleles = {}
    for chain in mhc_a:
        G_dom_alleles = {'G-ALPHA1': [], 'G-ALPHA2': []}
        key = False
        for row in mhc_a[chain]:
            if row[0] == 'G-DOMAIN':
                try:
                    if row[3] == 'description' and row[4] == 'G-ALPHA1':
                        key = 'G-ALPHA1'
                    elif row[3] == 'description' and row[4] == 'G-ALPHA2':
                        key = 'G-ALPHA2'
                    elif key:
                        if row[2] == 'gene' and row[3] == 'and' and row[4] == 'allele':
                            G_dom_alleles[key] += row[5:]
                        else:
                            key = False
                except IndexError:
                    pass
        mhc_a_alleles[chain] = deepcopy(G_dom_alleles)

    mhc_a_alleles_percs = {}
    for chain in mhc_a_alleles:
        mhc_a_alleles_percs[chain] = {}
        for key in mhc_a_alleles[chain]:
            mhc_a_alleles_percs[chain][key] = {}
            ### Allele info are always given with four elements: Gender, Spieces, Allele, Percentage
            for block in range(int(len(mhc_a_alleles[chain][key]) / 4)):
                allele = mhc_a_alleles[chain][key][2 + (4 * block)]
                perc = float(
                    mhc_a_alleles[chain][key][3 + (4 * block)].replace('(', '').replace('%)', '').replace(',', ''))
                mhc_a_alleles_percs[chain][key][allele] = perc

    return mhc_a_alleles_percs


def get_chainid_alleles_MHCII(pdbf):
    ''' Takes as input an IMGT preprocessed PDB file of p:MHC II.
        Returns a dictionary containing alleles andrelative identity scores for each
        G-domain in the given pdb from the REMARK.

    Args:
        pdbf: (string) path to IMGT pdb file

    Returns: (dict) MHCI alleles

    '''
    # test: multiple chains 3GJG, multiple alleles 1AO7
    ### Parsing file and extracting remarks
    with open(pdbf) as infile:
        remarks = []
        for line in infile:
            if line.startswith('REMARK 410'):
                row = [x for x in line.rstrip().split(' ') if x != '']
                del row[:2]
                remarks.append(row)
    remarks = [x for x in remarks if x != []]

    ### Dividing each remark section into a chains dictionary
    chains = {}
    flag = False
    for row in remarks:
        if row[0] == 'Chain' and row[1] == 'ID' and len(row) == 4:
            chainID = row[2][-1]
            chains[chainID] = []
            chains[chainID].append(row)
            flag = True
        elif flag == True:
            chains[chainID].append(row)

    ### Extracting MHC II Alpha and Beta chains
    mhc_a = {}  # MHC II Alpha
    mhc_b = {}  # MHC II Beta
    for chain in chains:
        try:
            if chains[chain][1][3].endswith('II-ALPHA'):
                mhc_a[chain] = chains[chain]
            elif chains[chain][1][3].endswith('II-BETA') or chains[chain][1][3].endswith('II-BETA-PEPTIDE'):
                mhc_b[chain] = chains[chain]
            elif any(x == ['G-DOMAIN', 'IMGT', 'domain', 'description', 'G-BETA'] for x in chains[chain]):#chains[chain][1][3].endswith('1C-LIKE') and any('G-BETA' in x for x in chains[chain]):
                mhc_b[chain] = chains[chain]
            
        except:
            pass

    ### Extracting alleles info
    mhc_a_alleles = {x: [] for x in mhc_a}
    mhc_b_alleles = {x: [] for x in mhc_b}

    for chain in mhc_a:
        key = False
        for row in mhc_a[chain]:
            if row[0] == 'G-DOMAIN':
                try:
                    if row[3] == 'description' and row[4] == 'G-ALPHA':
                        key = 'G-ALPHA'
                    elif key:
                        if row[2] == 'gene' and row[3] == 'and' and row[4] == 'allele':
                            mhc_a_alleles[chain] += row[5:]
                        else:
                            key = False
                except IndexError:
                    pass

    for chain in mhc_b:
        key = False
        for row in mhc_b[chain]:
            if row[0] == 'G-DOMAIN':
                try:
                    if row[3] == 'description' and row[4] == 'G-BETA':
                        key = 'G-BETA'
                    elif key:
                        if row[2] == 'gene' and row[3] == 'and' and row[4] == 'allele':
                            mhc_b_alleles[chain] += row[5:]
                        else:
                            key = False
                except IndexError:
                    pass

    return {'Alpha': mhc_a_alleles, 'Beta': mhc_b_alleles}


def format_alleles_MHCI(alleles, MHC_chains):
    ''' Format alleles to a list

    Args:
        alleles: (dict): output from get_chainid_alleles_MHCI()

    Returns: (lst): list of allele types

    '''
    try:
        alpha = [[k for k, v in i.items()] for i in [alleles['A'][i] for i in [i for i in alleles['A'].keys()]]]
    except KeyError:
        try:
            c = MHC_chains[0]
            alpha = [[k for k, v in i.items()] for i in [alleles[c][i] for i in [i for i in alleles[c].keys()]]]
        except KeyError:
            c = [i for i in alleles.keys()][0]
            alpha = [[k for k, v in i.items()] for i in [alleles[c][i] for i in [i for i in alleles[c].keys()]]]

    a_allele = sum(alpha, [])
    return a_allele


def format_alleles_MHCII(alleles, MHC_chains):
    ''' Format alleles to a list

    Args:
        alleles: (dict): output from get_chainid_alleles_MHCI()

    Returns: (lst, lst): tuple of lists of allele types for the alpha and beta chain respectively

    '''

    try:
        alpha = sum([alleles['Alpha'][i] for i in [i for i in alleles['Alpha'].keys()]], [])
    except KeyError:
        try:
            alpha = sum([alleles[MHC_chains[0]][i] for i in [i for i in alleles[MHC_chains[0]].keys()]], [])
        except KeyError:
            alpha = sum([alleles['A'][i] for i in [i for i in alleles['A'].keys()]], [])

    try:
        beta = sum([alleles['Beta'][i] for i in [i for i in alleles['Beta'].keys()]], [])
    except KeyError:
        try:
            beta = sum([alleles[MHC_chains[1]][i] for i in [i for i in alleles[MHC_chains[1]].keys()]], [])
        except KeyError:
            beta = sum([alleles['B'][i] for i in [i for i in alleles['B'].keys()]], [])
    a_allele = list(set([alpha[i - 1] for i in range(3, int(len(alpha)), 4)]))
    b_allele = list(set([beta[i - 1] for i in range(3, int(len(beta)), 4)]))
    return a_allele, b_allele


def get_resolution(pdbf):
    ''' Returns the resolution in Angstrom from the given pdb

    Args:
        pdbf (str): path to the pdb file

    Returns:
        resolution (float): resolution of the model, in Angstrom

    '''
    header = parse_pdb_header(pdbf)
    resolution = header['resolution']
    return resolution


def change_modified_res(pdb_file, change_SEP = True, change_F2F = True, change_CSO = True, change_CIR = True):
    ''' Change modified residues into regular residues

    Args:
        pdb_file: (str): Path to pdb file

    Returns: (bool/str): If nothing was changed, return False, else, resturn a message with the changes.

    '''

    with open(pdb_file) as f:
        infile = []
        for line in f:
            infile.append(line)

    res_changed = []

    with open(pdb_file, 'w') as f:
        for line in infile:
            l = [x for x in line.split(' ') if x != '']
            if line.startswith('ATOM') or line.startswith('HETATM'):

                # Change SEO into SER
                if ('SEP' in l[3] or 'SEP' in l[2]) and l[2] not in ['P', 'O1P', 'O2P', 'O3P', 'HA', 'HB2','HB3']:
                    if change_SEP:
                        f.write(line.replace('HETATM', 'ATOM  ').replace('SEP', 'SER'))
                        res_changed.append('SEP -> SER')
                    else:
                        f.write(line)
                elif ('SEP' in l[3] or 'SEP' in l[2]) and l[2] in ['P', 'O1P', 'O2P', 'O3P', 'HA', 'HB2','HB3']:
                    if change_SEP:
                        pass
                    else:
                        f.write(line)

                # Change CIR into ARG
                elif ('CIR' in l[3] or 'CIR' in l[2]) and l[2] not in ['F1', 'F2']:
                    if change_CIR:
                        if l[2] == 'O7':
                            f.write(line.replace('CIR', 'ARG').replace('O7', 'N2').replace('O', 'N').replace('HETATM', 'ATOM  '))
                        elif l[2] == 'N2':
                            f.write(line.replace('HETATM', 'ATOM  ').replace('CIR', 'ARG').replace('N2', 'N '))
                        elif l[2] == 'C2':
                            f.write(line.replace('HETATM', 'ATOM  ').replace('CIR', 'ARG').replace('C2', 'CA'))
                        elif l[2] == 'C1':
                            f.write(line.replace('HETATM', 'ATOM  ').replace('CIR', 'ARG').replace('C1','C '))
                        else:
                            f.write(line.replace('HETATM', 'ATOM  ').replace('CIR', 'ARG'))
                        res_changed.append('CIR -> ARG')
                    else:
                        f.write(line)
                elif ('CIR' in l[3] or 'CIR' in l[2]) and l[2] in ['F1', 'F2']:
                    if change_CIR:
                        pass
                    else:
                        f.write(line)

                # Change F2F into PHE
                elif ('F2F' in l[3] or 'F2F' in l[2]) and l[2] not in ['F1', 'F2']:
                    if change_F2F:
                        f.write(line.replace('HETATM', 'ATOM  ').replace('F2F', 'PHE'))
                        res_changed.append('F2F -> PHE')
                    else:
                        f.write(line)
                elif ('F2F' in l[3] or 'F2F' in l[2]) and l[2] in ['F1', 'F2']:
                    if change_F2F:
                        pass
                    else:
                        f.write(line)

                # Change CSO into CYS
                elif ('CSO' in l[3] or 'CSO' in l[2]) and l[2] not in ['OD']:
                    if change_CSO:
                        f.write(line.replace('HETATM', 'ATOM  ').replace('CSO', 'CYS'))
                        res_changed.append('CSO -> CYS')
                    else:
                        f.write(line)
                elif ('CSO' in l[3] or 'CSO' in l[2]) and l[2] in ['OD']:
                    if change_F2F:
                        pass
                    else:
                        f.write(line)

                # Keep all the other lines unchanged
                else:
                    f.write(line)
            else:
                f.write(line)

    if res_changed != []:
        return 'Changed PTM residues into normal residues: ' + '; '.join(list(set(res_changed)))
    else:
        return False



def replace_chain_names(chains, pdb, replacement_chains=['M', 'N', 'P']):
    ''' Replace chain names by another chain name in a bio.pdb object
    Be advised: some text editors might not read the characted in intermediate_chains.
    This will not affect this function.

    Args:
        chains: (list) chains to replace
        pdb: bio.pdb object
        replacement_chains: (list) replacement names (in order of chains to replace)

    Returns: bio.pdb object with changed chain names

    '''
    # First give them a greek letter, so names that are already present are taken care of.
    intermediate_chains = ['𝜶','𝜷','𝜸','𝜹','𝜺','𝜻','𝜼','𝜽','𝜾','𝜿','𝝀','𝝁','𝝂','𝝃','𝝄','𝝇','𝝈','𝝊','𝝋','𝝌','𝝎']
    for i in chains:
        for chain in pdb.get_chains():
            if chain.id == i:
                chain.id = intermediate_chains[chains.index(i)]

    # for i in chains:
    for chain in pdb.get_chains():
        if chain.id == intermediate_chains[intermediate_chains.index(chain.id)]:
            chain.id = replacement_chains[intermediate_chains.index(chain.id)]

    return pdb


def remember_IMGT_numbering(pdb_obj, forget=False):
    ''' Puts the IMGT numbering in the pdb residue id (' ', 1045, 'A') --> (' ', 1045, '1045A')

    Args:
        pdb: (Bio.PDB): Bio.PDB object
        forget: (bool): if True, the IMGT numbering will be cleared. This is needed for writing a pdb file

    Returns:  (Bio.PDB): Bio.PDB object with IMGT numbering in the residiue id.

    '''
    if forget:
        for res in pdb_obj.get_residues():
            res.id = (' ', res.id[1], ' ')

        return pdb_obj

    for res in pdb_obj.get_residues():
        res.id = (' ', res.id[1], str(res.id[1]) + str(res.id[2]).replace(' ', ''))

    return pdb_obj


def renumber(pdb):
    ''' Renumbers the pdb. Each chain starts at 1

    Args:
        pdb: Bio.PDb object

    Returns: Bio.PDb object with renumbered residues

    '''

    for chain in pdb.get_chains():
        nr = 1
        for res in chain:
            res.id = ('X', nr, res.id[2])
            nr += 1
    for chain in pdb.get_chains():
        for res in chain:
            res.id = (' ', res.id[1], ' ')

    return pdb


def write_pdb(pdb, out_path, pdb_id,
    IMGT_link='https://www.imgt.org/3Dstructure-DB/cgi/details.cgi?pdbcode=XXXX&Part=File'):
    ''' Write bio.pdb object to file, can use the header of the original pdb (bio.pdb cant remember file headers)

    Args:
        pdb: bio.pdb object
        out_path: (string) output path of pdb file

    Returns:

    '''

    def line_prepender(filename, line):
        ''' Add a line in front of a file

        Args:
            filename: (string) filepath
            line: (string) line to prepend

        '''

        with open(filename, 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write(line.rstrip('\r\n') + '\n' + content)

    # If the original pdb file path is given, use that header and paste it before the ATOM lines
    #if get_header_from:
    #    header = get_head_and_remarks(get_header_from)  # get the header

    # Write pdb
    io = PDBIO()
    io.set_structure(pdb)
    io.save(out_path)

    # Add header with link to the IMGT structure
    IMGT_link = IMGT_link.replace('XXXX', pdb_id)
    header = 'REMARK 410 ' + IMGT_link
    line_prepender(out_path, header)


def unzip_pdb(ID, indir, outdir):
    ''' Unzips a pdb, move it to another directory and return the filepath

    Args:
        ID: (string) pdb id
        indir:  (string) location of pdb.gz files
        outdir:  (string) output location

    Returns: (string) path to unzipped pdb file

    '''

    ## unzip pdb and move to outdir
    try:
        with gzip.open('%s/IMGT-%s.pdb.gz' % (indir, ID), 'rb') as f_in:
            # Check if the file is empty
            if f_in.seek(0, whence=2) == 0:
                raise Exception('File is empty')

        with gzip.open('%s/IMGT-%s.pdb.gz' % (indir, ID), 'rb') as f_in:
            with open('%s/%s.pdb' % (outdir, ID), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    except FileNotFoundError:
        print('ERROR TYPE #1: File not found. %s' % ID)
    return '%s/%s.pdb' % (outdir, ID)


def find_peptide_chain(pdb, min_len=7, max_len=25):
    ''' Find the pdb chain that is most likely the peptide based on its size

    Args:
        pdb: (Bio.PDB): Bio.PDB object
        min_len: (int): minimal peptide length to consider
        max_len: (int): maximal peptide length to consider

    Returns: (str): Most likely chain that is the peptide

    '''

    # Find most likely peptide chain: first chain to be 7 < len(chain) < 25
    pept_chain = []
    for chain in pdb.get_chains():
        if len(chain) >= min_len and len(chain) <= max_len and chain.id != ' ':
            # print(chain.id)# Is this chain between 7 and 25?
            heteroatoms = False
            for res in chain:
                # print(res.id)
                if res.id[0] != " " and res.id[0] != 'W':  # Check if a res in this chain is a heteroatom
                    print('\tHeteroatoms in peptide chain')
                    heteroatoms = True
                    break
            if heteroatoms == False:  # If all residues are oke, add this to the list of peptide chains
                pept_chain.append(chain.id)

    pept_chain = pept_chain[0]  # Take the first pept chain. If there are multiple, they are probably duplicates

    return pept_chain


def remove_irregular_chains(pdb, chains_to_keep):
    ''' Removes all chains that are not specified to be kept

    Args:
        pdb: Bio.PDB object
        chains_to_keep: list of strings: ['A', 'C', 'B']

    Returns: Bio.PDB object

    '''
    for _ in range(len([c for c in pdb.get_chains()])):
        for i in pdb.get_chains():
            for model in pdb:
                for chain in model:
                    if chain.id not in chains_to_keep or chain.id in [' ','','  ','_'] + list(range(1,20)):
                        model.detach_child(chain.id)

    return pdb


def remove_duplicated_chains(pdb):
    ''' In very rare cases, PDBParser duplicates the same chain multiple times. If that happens, this function removes
        all duplicates

    Args:
        pdb: Bio.PDB object

    Returns: Bio.PDB object

    '''
    # If there are multipel models, remove them so the pdb contains only one: pdb[0]
    if len([model.id for model in pdb]) > 1:
        # Remove all extra models that are not pdb[0]
        for _ in range(len([i.id for i in pdb])):
            for model in pdb:
                if model.id > 0:
                    pdb.detach_child(model.id)

    # Find all unique chains
    chains_to_keep = sorted(list(set([i.id for i in pdb.get_chains()])))
    # Check if there are duplicates. If true, remove them
    if len(chains_to_keep) != len([i for i in pdb.get_chains()]):

        # A list of placeholder names
        intermediate_chains = ['𝜶','𝜷','𝜸','𝜹','𝜺','𝜻','𝜼','𝜽','𝜾','𝜿','𝝀','𝝁','𝝂','𝝃','𝝄','𝝇','𝝈','𝝊','𝝋','𝝌','𝝎']

        # Function to find a key by its value in a dict
        def getkey(dic, val):
           for key, value in dic.items():
              if val == value:
                 return key

        # Find out when a duplicated chain occurs
        chain_id_seen = {}
        cnt = 0
        for chain in pdb.get_chains():
            cnt += 1
            if chain.id not in chain_id_seen:
                chain_id_seen[cnt] = chain.id

        # Rename all unique (non duplicated) chains to a greek placeholder
        cnt = 0
        for chain in pdb.get_chains():
            cnt += 1
            if chain.id in chains_to_keep and cnt == getkey(chain_id_seen,chain.id):
                chain.id = intermediate_chains[cnt-1]
        # Remove all chains that are not greek placeholders aka the duplicates
        for _ in range(len([c for c in pdb.get_chains()])):
            for model in pdb:
                for chain in model:
                    if chain.id not in intermediate_chains:
                        model.detach_child(chain.id)

        # change the chain names back to the original names
        for chain in pdb.get_chains():
            chain.id = chains_to_keep[intermediate_chains.index(chain.id)]

    return pdb


def find_chains_MHCI(pdb, pept_chain, all_MHC_chains):
    ''' Find the MHCI chains

    Args:
        pdb (Bio.PDB.PDBParser): Bio.PDB object of a peptide-MHCI structure
        pept_chain (str): chain ID of the peptide
        all_MHC_chains (list): list of all MHC alpha chains in the pdb file

    Returns: list of chains

    '''

    # Find contacts between peptide chain and other chains
    cont = Contacts.Contacts(pdb).chain_contacts
    c = [i for i in cont if i[1] == pept_chain or i[5] == pept_chain]
    # Make a list of all chains that contact the peptide chain
    chain_cont = [i for i in sum([[i[1],i[5]] for i in c], []) if i != pept_chain]
    # Make sure only MHC chains are present in the list
    chain_cont = [i for i in chain_cont if i in all_MHC_chains]
    # Make sure the chain is longer than 120 residues. This prevents selecting e.g. two peptides in the binding groove
    chain_cont = [i for i in chain_cont if i in [c.id for c in pdb.get_chains() if len(c) > 120]]

    # Find the chain having the most contacts with the peptide. This should also filter out TCR chains
    if len(set(chain_cont)) >= 1:
        bound_MHC_chains = sorted([ss for ss in set(chain_cont)], key=chain_cont.count, reverse=True)[0]
        bound_MHC_chains = [bound_MHC_chains, pept_chain]

        # Find B2M chain bound to the first alpha chain in bound_MHC_chains
        alpha_chain= bound_MHC_chains[0]
        b2m_c = [i for i in cont if (
            i[1] == alpha_chain or i[5] == alpha_chain) and
            i[1] != pept_chain and i[1] != ' ' and i[5]!= pept_chain and i[5]!=' ']
        chain_cont = [i for i in sum([[i[1],i[5]] for i in b2m_c], []) if i != alpha_chain]
        chain_cont = [i for i in chain_cont if i in [c.id for c in pdb.get_chains() if 90 < len(c) < 120]]
        b2m_chain = sorted([ss for ss in set(chain_cont)], key=chain_cont.count, reverse=True)[0]
        # Add B2M chain ID between alpha chain id and peptide chain id
        bound_MHC_chains.insert(1,b2m_chain)
    else:
        print('Found >1 MHC I chains')
        raise Exception

    return bound_MHC_chains


def find_chains_MHCII(pdb, pept_chain):
    ''' Find the MHCI chains

    Args:
        pdb: Bio.PDB object

    Returns: list of chains

    '''
    # todo this works for imgt, need to check if M and N are selected
    # Find contacts between peptide chain and other chains
    cont = Contacts.Contacts(pdb).chain_contacts
    c = [i for i in cont if i[1] == pept_chain or i[5] == pept_chain]
    # Make a list of all chains that contact the peptide chain
    chain_cont = [i for i in sum([[i[1],i[5]] for i in c], []) if i != pept_chain]
    # Make sure the chain is longer than 120 residues. This prevents selecting e.g. two peptides in the binding groove
    chain_cont = [i for i in chain_cont if i in [c.id for c in pdb.get_chains() if len(c) > 120]]

    # Find the two chains that have the most contacts with the peptide. This should also filter out TCR chains
    if len(set(chain_cont)) >= 2:
        MHC_chains = sorted([ss for ss in set(chain_cont)], key=chain_cont.count, reverse=True)[:2]
        MHC_chains = sorted(MHC_chains)
        MHC_chains = MHC_chains + [pept_chain]
    else:
        print('Found >2 MHC I chains')
        raise Exception

    return MHC_chains


def seqs_from_pdb(pdb_file, MHC_chains):
    ''' Use SeqIO to get the amino acid sequences from a PDB file.

    Args:
        pdb_file: (string) path to PDB file
        MHC_chains: (list) names of the M (, N) and P chains

    Returns: (list) AA sequence for the M (, N) and P chain respectively

    '''

    chain_seqs = {record.id: record.seq for record in SeqIO.parse(pdb_file, 'pdb-seqres')}
    seqs = []
    for c in MHC_chains:
        chain = [i for i in chain_seqs.keys()][0].split(':')[0] + ':' + c
        seqs.append(str(chain_seqs[chain]))

    return seqs


def check_missing_pept_residues(pdb, chain='P'):
    ''' Checks for missing residues in a chain by calculating N_{res_i-1}-Ca_{res_i} distance. This distance never
        exceeds ~2.5, so if the N_{res_i-1}-Ca_{res_i} distance > 3 --> residues are missing.

    Args:
        pdb: Bio.PDB object.
        chain: (string) name of the chain that is tested for missing residues.

    Returns: bool. True if residues are missing from the chain

    '''
    # Calculate the distances between every N of residue x and CA of residue x-1.
    N_CA_dist = []
    # Cutoff threshold. If this distance is exceeded, there is a residue gap
    threshold = 3
    # Check for missing chain residues
    prev_atom = [i for i in pdb[0][chain].get_atoms()][0]
    for res in pdb[0][chain]:
        for atom in res:
            if atom.id == 'N':
                N_CA_dist.append(atom - prev_atom)

            if atom.id == 'CA':
                prev_atom = atom

    # If the distance is greater than 3, there is a distance gap because there is a residue missing.
    return any([i > threshold for i in N_CA_dist])


def check_pMHC(pdb, mhc_class):
    ''' Tests parsed pMHC structures: chain numbering, naming and length

    Args:
        pdb: Bio.PDB object
        mhc_class (str): MHC class of the complex

    Returns: Bool

    '''
    requirements = [False, False, False, True]
    chains = [i.id for i in pdb.get_chains()]
    chain_len = {i.id:len(i) for i in pdb.get_chains()}

    # 1. Check chain names and the number of chains
    if mhc_class=='I':
        if 'M' == chains[0] and 'B'==chains[1] and 'P' == chains[-1] and not 'N' in chains:
            requirements[0] = True
    elif mhc_class=='II':
        if 'M' == chains[0] and 'N' == chains[1] and 'P' == chains[-1]:
            requirements[0] = True

    # 2. Check M,N chain length
    if mhc_class=='I':
        if chain_len['M'] > 120 and 90 < chain_len['B'] < 120:
            requirements[1] = True
    elif mhc_class=='II':
        if chain_len['M'] > 120 and chain_len['N'] > 120:
            requirements[1] = True

    # 3. Check peptide length
    if chain_len['P'] > 6 and chain_len['P'] < 26:
        requirements[2] = True

    # 4. Check numbering. Does every chain start at 1 and end at its chain length?
    for c in pdb.get_chains():
        res = [r.id for r in c]
        if not res[0][1] == 1:
            requirements[3] = False
        if not res[-1][1] == len(c):
            requirements[3] = False

    # If all tests are TRUE, the p:MHC structure is successfully passed
    if all(requirements):
        return True
    else:
        return False


def check_non_canonical_res(chain):
    ''' Check if there are non-canonical residues in the chain of a Bio.PDB chain object

    Args:
        chain: Bio.PDB chain object

    Returns: bool, true if there are non canonical residues in the chain

    '''
    letters = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU", "MET", "ASN", "PRO", "GLN", "ARG",
               "SER", "THR", "VAL", "TRP", "TYR"]

    return any([r.resname not in letters for r in chain if r.resname])


def check_hetatoms_in_binding_groove(pdb, MHC_chains, mhc_class='I'):
    ''' Checks if there are heteroatoms in the binding groove between MHC and the peptide. Heteroatoms near the
        peptide on the outside of the binding groove are allowed. This takes ~ 0.008 seconds.

    Args:
        pdb: Bio.PDB object
        MHC_chains (list): list of MHC chains for the M and P chains respectively
        mhc_class (str): MHC class

    Returns: bool, true if there are problematic heteroatoms in the binding groove

    '''

    letters = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU", "MET", "ASN", "PRO", "GLN", "ARG",
               "SER", "THR", "VAL", "TRP", "TYR", 'HOH', "MSE"]

    # Define the chain names and the center MHC residue from which distances are calculated.
    if mhc_class=='I':
        MHC = MHC_chains[0]
        MHC_res = [8]
        search_dist = 18
    elif mhc_class=='II':
        MHC = MHC_chains[1]
        MHC_res = [12, 29]
        search_dist = 18

    P = MHC_chains[-1]

    # Take the CA of a central residue (res 8) in MHCI
    MHC_ch_atoms = [a for a in pdb[0][MHC].get_atoms() if a.id == 'CA' and a.get_parent().id[1] in MHC_res]
    # Take all CA atoms of the peptide chain
    pept_atoms = [a for a in pdb[0][P].get_atoms() if a.id == 'CA']

    potential_junk = []
    for chain in pdb.get_chains():
        for res in chain:
            # If the chain has non-amino acids -> add the atoms to the list.
            if res.resname not in letters:
                for atom in res:
                    potential_junk.append(atom)

    # calculate distances
    atom_dist = NeighborSearch(atom_list=MHC_ch_atoms + potential_junk + pept_atoms).search_all(search_dist)

    # format the distances
    cont = []
    for pair in atom_dist:
        cont.append((pair[1].get_parent().resname, pair[1].get_parent().get_parent().id,
                    pair[1].get_parent().id[1], pair[1].get_id(),pair[0].get_parent().resname,
                    pair[0].get_parent().get_parent().id,pair[0].get_parent().id[1],
                    pair[0].get_id(),pair[0] - pair[1]))
    # remove intra-chain contacts
    cont = [i for i in cont if i[1] != i[5]]
    cont = cont + [(i[4], i[5], i[6], i[7], i[0], i[1], i[2], i[3], i[8]) for i in cont if i[1] == P or i[5] == P]
    cont = list(dict.fromkeys(cont))

    # Find the min distance between the peptide and the central MHC residue
    pept_MHC_dist = min((x[-1], x) for x in [i for i in cont if i[1] == P and i[5] == MHC])

    # remove junk-junk distances
    cont = [i for i in cont if i[5] in [MHC, P]]

    # Find all unique pieces of junk
    junk = list(set([(i[0],i[1],i[2]) for i in cont if i[1] not in [MHC, P]]))

    log_message = False

    for piece in junk:
        # For every piece of junk find the min junk-MHC and min junk-peptide distances.
        try:
            # Get the distances for this specific piece of junk
            piece_of_junk_cont = []
            for i in cont:
                if (i[0], i[1], i[2]) == piece:
                    piece_of_junk_cont.append(i)
            # Calculate min junk-MHC and min junk-peptide distances.
            junk_MHC_dist = min((x[-1], x) for x in [i for i in piece_of_junk_cont if i[5] == MHC])
            # max distance between junk and pept
            junk_pept_dist = min((x[-1], x) for x in [i for i in piece_of_junk_cont if i[5] == P])

            # Find out of the piece of junk is inside the binding groove
            if junk_pept_dist[0] < 6:
                log_message = 'There are heteroatoms within 6 Angstrom of the peptide chain'
                # Check if the distance between junk and peptide is smaller than the distance between pept and MHC.
                # If the junk is in between the peptide and MHC, this will be true, but also if the junk is on the outside
                if junk_pept_dist[0] < pept_MHC_dist[0]:
                    # Check if the distance between MHC and junk is smaller than the distance between pept and MHC.
                    # If the junk is in between the peptide and MHC, this will be true
                    if junk_MHC_dist[0] < pept_MHC_dist[0]:
                        return True, log_message

        except:
            pass

    return False, log_message


def log(ID, error, logfile, verbose=True):
    ''' Keeps track of what goes wrong while parsing

    Args:
        ID: (string) PDB ID
        error: (string) error to append to log file
        logfile: (string) path to logfile
        verbose: (Bool) print error?
    '''

    # Create log file
    if not os.path.exists(logfile):
        with open(logfile, 'w') as f:
            f.write('ID,error\n')

    if verbose:
        print('\t' + error)
    with open(logfile, 'a') as f:
        f.write('%s,%s\n' % (ID, error))


def find_merged_pept_chains(pdb_file):
    ''' Checks if the peptide is merged to another chain (have the same chain ID). If True, return a dict telling
        wich chain the peptide is merged to and which residues belong to the peptide. It takes this info from the header

    Args:
        pdb_file: (str): Path to pdb file

    Returns: (dict): {chain_ID:(1:n)}

    '''


    # Get the remarks from the pdb file
    with open(pdb_file) as infile:
        remarks = []
        for line in infile:
            if line.startswith('REMARK 410'):
                row = [x for x in line.rstrip().split(' ') if x != '']
                del row[:2]
                remarks.append(row)
    remarks = [x for x in remarks if x != []]

    ### Dividing each remark section into a chains dictionary
    chains = {}
    flag = False
    for row in remarks:
        if row[0] == 'Chain' and row[1] == 'ID' and len(row) == 4:
            chainID = row[2][-1]
            chains[chainID] = []
            chains[chainID].append(row)
            flag = True
        elif flag:
            chains[chainID].append(row)

    pept_chains = {}
    try:
        for i in chains:
            for line in chains[i]:
                if 'PEPTIDE(' in ''.join(line) or '[PEPTIDE' in ''.join(line) :
                    # add the resnr of the peptideto the dict, with the chain as key
                    pept_chains[i] = tuple([int(i) for i in ''.join(line).split('(')[1].split(')')[0].split('-')])
    except:
        pass

    if pept_chains == {}:
        return False
    return pept_chains


def un_merge_pept_chain(pdb, pdb_file):
    ''' Cut a peptide from a MHC chain if they are merged into the same chain

    Args:
        pdb: (Bio.PDB): Bio.PDB pdb object
        pdb_file: (str): Path to PDB file

    Returns: (Bio.PDB): Bio.PDB pdb object with the peptide as a separate chain

    '''

    pept_chain = find_merged_pept_chains(pdb_file)
    # First check if there are no peptide between 7 and 25 in the pdb
    if not any(c > 7 and c < 26 for c in [len(c) for c in pdb.get_chains()]):
        if pept_chain: #If it found a merged peptide in the pdb header..
            # print('Found merged peptide chain')
            for cn in pept_chain:
                pept_res = [res for res in pdb[0][cn] if res.id[1] in range(pept_chain[cn][0], pept_chain[cn][1] + 1)]
                pdb = extract_residues(pept_res, cn, pdb)

            # Renumber the pdb, because some residues got removed, the numbering is now wrong
            pdb = renumber(pdb)

            return pdb, 'Successfully cut merged peptide from MHC chain'
    return pdb, False


def extract_residues(res_lst, chain_id, pdb):
    ''' Extract residues from a pdb and put them in a new chain in the same pdb

    Args:
        res_lst: (lst): List of Bio.PDB <residue> objects (pdb.get_residues())
        chain_id: (str): name of chain were res_lst originates from
        pdb: (Bio.PDB): Bio.PDB object

    Returns: (Bio.PDB): Bio.PDB object with the residues from res_lst in a new chain

    '''

    # Sometimes there is a gap between the peptide and the end of the peptide-MHC adapter. By calculating the
    # distance between residues, such a gap can be detected. If there is a gap, then take all residues up to
    # that gap. (All residues, incl those after the gap, will be deleted from the original chain later)

    # Calculate the distances between every N of residue x and CA of residue x-1.
    # See 'check_missing_pept_residues()'
    N_CA_dist = []
    # Check for missing chain residues
    prev_atom = [a for a in res_lst[0].get_atoms()][0]
    for res in res_lst:
        for atom in res:
            if atom.id == 'N':
                N_CA_dist.append(atom - prev_atom)
            if atom.id == 'CA':
                prev_atom = atom
    # Take all the residues until the gap, if there is a gap
    pept_res_keep = res_lst
    if any(i > 3 for i in N_CA_dist):
        pept_res_keep = res_lst[:next(x for x, val in enumerate(N_CA_dist) if val > 3)]

    # Make a new empty chain using a letter that is not a chain in the pdb yet.
    new_chain_name = [x for x in ascii_uppercase if x not in [i.id for i in pdb.get_chains()]][0]
    new_chain = Chain.Chain(new_chain_name)
    pdb[0].add(new_chain)

    # Add residues to the new chain
    for res in pept_res_keep:
        pdb[0][new_chain_name].add(res)

    # Remove residues from the old chain
    for res in res_lst:
        pdb[0][chain_id].detach_child(res.id)

    # Also add the parent to the new residues (othewise the Contacts class will complain)
    for res in pdb[0][new_chain_name]:
        res.parent = pdb[0][new_chain_name]

    return pdb


def ensure_order(pdb, MHC_chains):
    ''' Checks if the peptide chain is the last chain, in the pdb, if not reoder them so the MHC chains come before the
        peptide chain.

    Args:
        pdb: (Bio.PDB): Bio.PDB object containing the Alpha, (Beta for MHCII) and Peptide chain
        MHC_chains: (lst): List of the names of the chains in this order: M, N (only for MHCII) and P

    Returns: (Bio.PDB): Bio.PBD object with the MHC chains first and the peptide chain last

    '''
    # Check if the peptide chain is the last chain, If not --> reorder
    if [i.id for i in pdb.get_chains()][-1] != MHC_chains[-1]:
        # Store the peptide chain
        p_chain = pdb[0][MHC_chains[-1]]
        # Remove the peptide chain from the pdb
        pdb[0].detach_child(MHC_chains[-1])
        # Add chain back to the pdb
        pdb[0].add(p_chain)
    return pdb


def check_DM_chaperone(alleles):
    ''' Check if the DM chaperone is involved in this structure by checking the allele type.
        MHC class II molecules require HLA-DM (H2‑DM in mice) to facilitate the exchange of the CLIP fragment. The
        involvement of this protein can influence the structure of MHCII

    Args:
        alleles: (lst): list of allele types

    Returns: (bool): True if there are HLA-DM (human) or H2-DM (mouse)

    '''
    if any('HLA-DM' in i for i in alleles) or any('H2-DM' in i for i in alleles):
        return True
    return False


def find_pept_secondary_structure(pdb_file, pdb, pept_chain):
    ''' Using the annotation in the IMGT PDB file, find secondary structures in the peptide

    Args:
        pdb_file: (str): Path to the pdb file
        pdb: (Bio.PDB): Bio.PDB object
        pept_chain: (str): Name of the peptide chain

    Returns: (dct, str/bool): dictionairy with [sheet] or [helix] denoting the location of the structure in MODELLER
                format, False if there are no B-sheets or A-helices, or a string with their location

    '''

    snd_struc = {}

    log_message = []
    with open(pdb_file) as f:
        helix = []
        sheet = []
        for line in f:
            if line.startswith('HELIX'):
                helix.append([x for x in line[:-2].split(' ') if x != ''])
            if line.startswith('SHEET'):
                sheet.append([x for x in line[:-2].split(' ') if x != ''])

    # Find Alpha helices
    for i in helix:
        if i[4][0] == pept_chain and i[7][0] == pept_chain:
            snd_struc['helix'] = [i[5], i[8]]
            log_message.append(
                'Found an alpha-helix in the peptide between residue %s (%s) and %s (%s)' % (i[5], i[3], i[8], i[6]))

    # Find Beta sheest
    sheets = []
    for i in sheet:
        if any(x == pept_chain for x in i[3:]):
            line = i[3:]
            sheets.append(list(set([line[y + 1] for y in [x for x in range(len(line)) if line[x] == pept_chain]])))
    if sheets != []:
        # Find the beginning and end of the B-sheet
        sheet_start, sheet_stop = min([int(i) for i in sum(sheets,[])]), max([int(i) for i in sum(sheets,[])])
        # Find out how long the sheet is
        len_sheet = sheet_stop-sheet_start+1

        hairpin = False
        if len_sheet >= 4:
            start_CA = [res for res in pdb[0][pept_chain] if res.id[1] == sheet_start][0]['CA']
            end_CA = [res for res in pdb[0][pept_chain] if res.id[1] == sheet_stop][0]['CA']
            if start_CA-end_CA < 6:
                hairpin = True

        if hairpin:
            # format for modeller
            snd_struc['sheet']=["N:%s:P" %sheet_start, "O:%s:P" %sheet_stop, int(-(len_sheet - 2)/2)]
            log_message.append('Found a beta-sheet hairpin in the peptide chain. Starts at %s; %s for %s h-bonds' %(snd_struc['sheet'][0], snd_struc['sheet'][1], (len_sheet - 2)/2))
        else:
            log_message.append('Warning: Found a beta-strand in peptide. This is most likely a crystal artifact or it is causeb by a bound TCR')

    if log_message == []:
        return False, False
    else:
        return snd_struc, '; '.join(list(set(log_message)))


def hardcode_cut_peptide(pdb_id, pdb):
    ''' Some structures were only cleanable by manually cutting the peptide from another chain.

    Args:
        pdb_id: (str): ID of PDB file
        pdb: (Bio.PDB): Bio.PDB object

    Returns:  (Bio.PDB): Bio.PDB object with the previously merged peptide in a separate chain

    '''

    if pdb_id in ['6DFS', '4MAY', '3PL6', '4GRL', '1IAO']:

        if pdb_id == '6DFS':
            chain, res_nr = 'D', [1,14]

        if pdb_id == '4MAY':
            chain, res_nr = 'D', [1,12]

        if pdb_id == '3PL6':
            chain, res_nr = 'D', [1,14]

        if pdb_id == '4GRL':
            chain, res_nr = 'D', [1,14]

        if pdb_id == '1IAO':
            chain, res_nr = 'B', [1,14]


        pept_res = [res for res in pdb[0][chain] if res.id[1] in range(res_nr[0], res_nr[1] + 1)]
        pdb = extract_residues(pept_res, chain, pdb)
        pdb = renumber(pdb)

    return pdb


def check_clip_peptide(pdb_file):
    ''' Checks if the bound peptide is the clip peptide. If it is, this structure is not suitable for modelling.

    Args:
        pdb_file: (str): path to pdb file

    Returns: (bool)

    '''

    with open(pdb_file) as f:
        for line in f:
            if 'clip' in line or 'CLIP' in line:
                return True
    return False


def parse_pMHCI_pdb(pdb_id,
                     indir = PANDORA.PANDORA_data + '/PDBs/IMGT_retrieved/IMGT3DFlatFiles',
                     outdir = PANDORA.PANDORA_data + '/PDBs/pMHCI',
                     bad_dir = PANDORA.PANDORA_data + '/PDBs/Bad/pMHCI',
                     keep_bad=False, custom_map={"MSE":"M"},
                     remove_biopython_object=True):
    ''' Clean one MHCI pdb file downloaded from IMGT

        Args:
        pdb_id (str): id of pdb file
        indir (str): path of the input dir (where the .gz files are)
        outdir (str): path of the output dir (where the unzipped .pdb files go)
        bad_dir (str): path of the output dir (where the unsuitable .pdb files go)
        keep_bad (str): Keep the unsuitable pdb files in the bad_dir. If False, 
        it will remove the files instead of moving them. Defaults to False.
        custom_map (dict): custom map for 3-to-1 letter aa name translation.

    Returns: Template object

    '''
    logfile = os.path.dirname(bad_dir) + '/log_MHCI.csv'

    # Check if file exists
    if os.path.isfile('%s/IMGT-%s.pdb.gz' % (indir, pdb_id)):
        print('Parsing %s' % pdb_id)
        if pdb_id == '3RGV':
            log(pdb_id, 'Skipped, non canonical template related to issue #47', logfile)
            raise Exception
        # try:
        # Unzip file (also check if the file is not empty) and save the path of this file
        pdb_file = unzip_pdb(pdb_id, indir, outdir)

        try:
            log_message = change_modified_res(pdb_file, change_CIR=False)
            if log_message:
                log(pdb_id, 'Warning, ' + log_message, logfile)

            pdb = PDBParser(QUIET=True).get_structure('MHCI', pdb_file)
            # Remove waters and duplicated chains, then renumber
            pdb = remove_duplicated_chains(pdb)
            # pdb = remember_IMGT_numbering(pdb)
            pdb = renumber(pdb)

            # Get allele per each chain
            try:
                alleles = get_chainid_alleles_MHCI(pdb_file)
            except:
                log(pdb_id, 'Failed, Could not find allele type', logfile)
                raise Exception('NoAlleleType')

            # Check if the peptide is merged to the MHC, cut it loose and put it in a new chain
            try:            
                pdb, log_message = un_merge_pept_chain(pdb, pdb_file)
                if log_message:
                    log(pdb_id, 'Warning, ' + log_message, logfile)
            except:
                log(pdb_id, 'Failed, Could not cut peptide from MHC chain', logfile)
                raise Exception('NotDetachedPeptide')

            chain_lens = '; '.join([i.id + ':' + str(len(i)) for i in pdb.get_chains() if i.id != ' '])
            try:                # Find the peptide chain
                pept_chain = find_peptide_chain(pdb)
            except:

                log(pdb_id, 'Failed, Could not find a suitable peptide chain with a length between 7 and 25. Found: ' + chain_lens, logfile)
                raise Exception('PeptideLengthError')

            if check_non_canonical_res(pdb[0][pept_chain]):
                log(pdb_id, 'Failed, Non canonical residues in the peptide chain', logfile)
                raise Exception('PeptNonCanonRes')

            if check_missing_pept_residues(pdb, chain=pept_chain):
                log(pdb_id, 'Failed, Peptide chain is missing residues', logfile)
                raise Exception('PeptMissingResidues')

            try:                 # Find out which chains are the Alpha and Peptide chain
                MHC_chains = find_chains_MHCI(pdb, pept_chain, list(alleles.keys()))
            except:
                log(pdb_id, 'Failed, Could not locate Alpha chain. Found: ' + chain_lens, logfile)
                raise Exception('AlphaChainError')
            
            helix, sheet = False, False
            try:
                snd_struc, log_message = find_pept_secondary_structure(pdb_file, pdb, pept_chain)

                if log_message:
                    log(pdb_id, 'Warning, ' + log_message, logfile)
                    if 'helix' in snd_struc:
                        helix = snd_struc['helix']
                    if 'sheet' in snd_struc:
                        sheet = snd_struc['sheet']

            except:
                log(pdb_id, 'Failed, Error in finding secondary structures in the peptide', logfile)
                raise Exception('SecondaryStructError')
            
            # Get allele per each chain
            try:
                a_allele = format_alleles_MHCI(alleles, MHC_chains)
            except:
                log(pdb_id, 'Failed, Could not format alleles', logfile)
                raise Exception('AlleleFormatError')
            
            hetatm_in_groove, log_message = check_hetatoms_in_binding_groove(pdb, MHC_chains, mhc_class='I')
            if hetatm_in_groove:
                log(pdb_id, 'Failed, Heteroatoms in binding groove between the peptide and MHC', logfile)
                raise Exception('HETAMInGroove')
            if log_message:
                log(pdb_id, 'Warning, ' + log_message, logfile)
            
            try:                 # Reformat chains
                pdb = remove_irregular_chains(pdb, MHC_chains)  # Remove all other chains from the PBD that we dont need
                pdb = ensure_order(pdb, MHC_chains)
                pdb = replace_chain_names(MHC_chains, pdb, ['M', 'B', 'P'])  # Rename chains to M,B,P # Renumber from 1
            except:
                log(pdb_id, 'Failed, Could not reformat structure', logfile)
                raise Exception('ReformatError')
            
            try:  # get the chain sequences from the pdb file
                # seqs = seqs_from_pdb(pdb_file, MHC_chains)
                #seqs = [seq1(''.join([res.resname for res in chain])) for chain in pdb.get_chains()]
                seqs = {chain.id : seq1(''.join([res.resname for res in chain]), custom_map=custom_map) for chain in pdb.get_chains()}
            except:
                log(pdb_id, 'Failed, Could not fetch chain sequences from pdb file', logfile)
                raise Exception('SeqFetchingError')

            if not check_pMHC(pdb, mhc_class='I'):
                log(pdb_id, 'Failed, Structure did not pass the test.', logfile)
                raise Exception('StructTestError')

            # Create MHC_structure object
            try:
                templ = Template(pdb_id, allele_type=a_allele, M_chain_seq=seqs['M'],
                                    B2M_seq=seqs['B'], peptide=seqs['P'], pdb=pdb,
                                    sheet=sheet, helix=helix,
                                    remove_biopython_object=remove_biopython_object)
            except:
                log(pdb_id, 'Failed, Template object could not be created', logfile)
                raise Exception('TemplObjError')

            #Check if the anchor calculation went ok
            if templ.anchors[1] - templ.anchors[0] < 6:
                log(pdb_id, 'Failed, the anchors seem to be too close to each other. This might depend on GitHub issue #146', logfile)
                raise Exception('AnchorProximityError')
            
            # clear_pdb = remember_IMGT_numbering(deepcopy(pdb), forget=True)
            # Finally, write the cleaned pdb to the output dir. Keep the header of the original file.
            write_pdb(pdb=pdb, out_path='%s/%s.pdb' % (outdir, pdb_id),
                        pdb_id=pdb_id)#, pdb_file)
            
            return templ

        except Exception as err:  # If something goes wrong, append the ID to the bad_ids list
            print('Moved to bad IDs')
            print(("Exception: {0}".format(err)))
            if keep_bad == True:
                log(pdb_id, 'Moved to bad IDs dir', logfile)
                os.system('mv %s/%s.pdb %s/%s.pdb' % (outdir, pdb_id, bad_dir, pdb_id))
            else:
                log(pdb_id, 'Removed', logfile)
                os.system('rm %s/%s.pdb' % (outdir, pdb_id))




def parse_pMHCII_pdb(pdb_id,
                      indir=PANDORA.PANDORA_data + '/PDBs/IMGT_retrieved/IMGT3DFlatFiles',
                      outdir = PANDORA.PANDORA_data + '/PDBs/pMHCII',
                      bad_dir = PANDORA.PANDORA_data + '/PDBs/Bad/pMHCII',
                      keep_bad=False, custom_map={"MSE":"M"},
                      remove_biopython_object=True):
    ''' Clean one MHCII pdb file downloaded from IMGT

    Args:
        pdb_id (str): id of pdb file
        indir (str): path of the input dir (where the .gz files are)
        outdir (str): path of the output dir (where the unzipped .pdb files go)
        bad_dir (str): path of the output dir (where the unsuitable .pdb files go)
        keep_bad (str): Keep the unsuitable pdb files in the bad_dir. If False, 
            it will remove the files instead of moving them. Defaults to False.
        custom_map (dict): custom map for 3-to-1 letter aa name translation.

    Returns: Template object

    '''
    # set paths for in and out directories
    logfile = os.path.dirname(bad_dir) + '/log_MHCII.csv'

    if os.path.isfile('%s/IMGT-%s.pdb.gz' % (indir, pdb_id)):
        print('Parsing %s' % pdb_id)
        try:
            # Unzip file (also check if the file is not empty) and save the path of this file
            pdb_file = unzip_pdb(pdb_id, indir, outdir)

            log_message = change_modified_res(pdb_file, change_CIR=False)
            if log_message:
                log(pdb_id, 'Warning, ' + log_message, logfile)

            # Check if the CLIP peptide is involved
            if check_clip_peptide(pdb_file):
                log(pdb_id, 'Failed, MHC structure is bound to CLIP peptide', logfile)
                raise Exception

            # Load pdb file as an Bio.PDB object, check for duplicate chains and renumber the pdb
            pdb = PDBParser(QUIET=True).get_structure('MHCII', pdb_file)
            pdb = remove_duplicated_chains(pdb)
            # pdb = remember_IMGT_numbering(pdb)
            pdb = renumber(pdb)

            # Get allele per each chain
            try:
                alleles = get_chainid_alleles_MHCII(pdb_file)
            except:
                log(pdb_id, 'Failed, Could not find allele type', logfile)
                raise Exception

            try:            #Check if the peptide is merged to the MHC, cut it loose and put it in a new chain
                pdb = hardcode_cut_peptide(pdb_id, pdb) #For 4 structures, manually cut them loose
                pdb, log_message = un_merge_pept_chain(pdb, pdb_file)
                if log_message:
                    log(pdb_id, 'Warning, ' + log_message, logfile)
            except:
                log(pdb_id, 'Failed, Could not cut peptide from MHC chain', logfile)
                raise Exception

            chain_lens = '; '.join([i.id + ':' + str(len(i)) for i in pdb.get_chains() if i.id != ' '])
            try:                # Find the peptide chain
                pept_chain = find_peptide_chain(pdb)
            except:
                log(pdb_id, 'Failed, Could not find a suitable peptide chain with a length between 7 and 25. Found: ' + chain_lens, logfile)
                raise Exception

            if check_non_canonical_res(pdb[0][pept_chain]):
                log(pdb_id, 'Failed, Non canonical residues in the peptide chain', logfile)
                raise Exception

            if check_missing_pept_residues(pdb, chain=pept_chain):
                log(pdb_id, 'Failed, Peptide chain is missing residues', logfile)
                raise Exception

            try:                 # Find out which chains are the Alpha and Peptide chain
                MHC_chains = find_chains_MHCII(pdb, pept_chain)
            except:
                log(pdb_id, 'Failed, Could not locate Alpha/Beta chain. Found: ' + chain_lens, logfile)
                raise Exception

            try:
                helix, sheet = False, False
                snd_struc, log_message = find_pept_secondary_structure(pdb_file, pdb, pept_chain)
                if log_message:
                    log(pdb_id, 'Warning, ' + log_message, logfile)
                    if 'helix' in snd_struc:
                        helix = snd_struc['helix']
                    if 'sheet' in snd_struc:
                        sheet = snd_struc['sheet']
            except:
                log(pdb_id, 'Failed, Error in finding secondary structures in the peptide', logfile)
                raise Exception

            # Get allele per each chain
            try:
                a_allele, b_allele = format_alleles_MHCII(alleles, MHC_chains)
            except:
                log(pdb_id, 'Failed, Could not format alleles', logfile)
                raise Exception

            # Check if the DM chaperone is involved
            if check_DM_chaperone(a_allele + b_allele):
                log(pdb_id, 'Failed, HLA-DM or H2-DM chaperone protein is involved.', logfile)
                raise Exception

            hetatm_in_groove, log_message = check_hetatoms_in_binding_groove(pdb, MHC_chains, mhc_class='II')
            if hetatm_in_groove:
                log(pdb_id, 'Failed, Heteroatoms in binding groove between the peptide and MHC', logfile)
                raise Exception
            if log_message:
                log(pdb_id, 'Warning, ' + log_message, logfile)

            try:                 # Reformat chains
                pdb = remove_irregular_chains(pdb, MHC_chains)  # Remove all other chains from the PBD that we dont need
                pdb = ensure_order(pdb, MHC_chains)
                pdb = replace_chain_names(MHC_chains, pdb, ['M', 'N', 'P'])   # Rename chains to M,N,P
            except:
                log(pdb_id, 'Failed, Could not reformat structure', logfile)
                raise Exception

            try:  # get the chain sequences from the pdb file
                seqs = {chain.id : seq1(''.join([res.resname for res in chain]), custom_map=custom_map) for chain in pdb.get_chains()}
            except:
                log(pdb_id, 'Failed, Could not fetch chain sequences from pdb file', logfile)
                raise Exception

            if not check_pMHC(pdb, mhc_class='II'): #test if the pdb is parsed correctly
                log(pdb_id, 'Failed, Structure did not pass the test.', logfile)
                raise Exception

            # Create MHC_structure object
            templ = Template(pdb_id, allele_type=a_allele + b_allele,
                                  M_chain_seq=seqs['M'], N_chain_seq=seqs['N'],
                                  peptide=seqs['P'], MHC_class='II', pdb=pdb,
                                  helix=helix, sheet=sheet,
                                  remove_biopython_object=remove_biopython_object)

            # clear_pdb = remember_IMGT_numbering(deepcopy(pdb), forget=True)
            # Finally, write the cleaned pdb to the output dir. Keep the header of the original file.
            write_pdb(pdb=pdb, out_path='%s/%s.pdb' % (outdir, pdb_id),
                        pdb_id=pdb_id)#, pdb_file)

            return templ

        except Exception as err:  # If something goes wrong, append the ID to the bad_ids list
            print('Moved to bad IDs')
            print(("Exception: {0}".format(err)))
            if keep_bad == True:
                log(pdb_id, 'Moved to bad IDs dir', logfile)
                os.system('mv %s/%s.pdb %s/%s.pdb' % (outdir, pdb_id, bad_dir, pdb_id))
            else:
                log(pdb_id, 'Removed', logfile)
                os.system('rm %s/%s.pdb' % (outdir, pdb_id))

def get_sequence_for_fasta(template, MHC_class, chain):
    # alpha_chains = ['HLA-A', 'HLA-B', 'HLA-C', 'HLA-E', 'HLA-F', 'HLA-G',
    #                 'HLA-DQA', 'HLA-DRA', 'HLA-DPA',
    #                 'H2-EA', 'MH2-AA']
    # beta_chains = ['HLA-DQB', 'HLA-DRB', 'HLA-DPB',
    #                'H2-EB', 'MH2-AB', 'H2-AB']

    if chain == 'M':
        alleles = [x for x in template.allele_type if any(y in x for y in PANDORA.alpha_genes)]
        header = template.id+'_alpha' +'; '+ (',').join(alleles)

        seq = template.M_chain_seq
        #Clip sequences to keep only the G-domain
        if MHC_class =='I':
            seq = seq[:PANDORA.MHCI_G_domain[0][1]]
        elif MHC_class =='II':
            seq = seq[:PANDORA.MHCII_G_domain[0][1]]

    elif chain == 'N':
        alleles = [x for x in template.allele_type if any(y in x for y in PANDORA.beta_genes)]
        header = template.id+'_beta' +'; '+ (',').join(alleles)

        seq = template.N_chain_seq
        #Clip sequences to keep only the G-domain
        seq = seq[:PANDORA.MHCII_G_domain[1][1]]

    return header, seq


def generate_mhcseq_database(data_dir = PANDORA.PANDORA_data + '/mhcseqs',
                             HLA_out = 'HLA_cleaned.fasta',
                             nonHLA_out = 'MHC_cleaned.fasta'):
    """generate_mhcseq_database(data_dir=PANDORA.PANDORA_data, HLA_out='HLA_cleaned.fasta', nonHLA_out='MHC_cleaned.fasta')
    Downloads and parse HLA and other MHC sequences to compile reference fastas

    Args:
        data_dir (str, optional): Data directory. Defaults to PANDORA.PANDORA_data.
        HLA_out (str, optional): Output file for HLA sequences. Defaults to 'HLA_cleaned.fasta'.
        nonHLA_out (str, optional): Output file for non human MHCs. Defaults to 'MHC_cleaned.fasta'.

    Returns:
        None.

    """

    #HLAs: https://raw.githubusercontent.com/ANHIG/IMGTHLA/Latest/hla_prot.fasta
    #MHCs: https://raw.githubusercontent.com/ANHIG/IPDMHC/Latest/MHC_prot.fasta


    # Changing working directory
    #start_dir = os.getcwd()
    #os.chdir(data_dir)

    # Download and parse sequences
    # Human sequences
    ref_MHCI_sequences = generate_hla_database(data_dir, HLA_out = HLA_out)
    # Non-human sequences
    ref_MHCI_sequences.update(generate_nonhla_database(data_dir, nonHLA_out = nonHLA_out))

    # Change back working directory
    #os.chdir(start_dir)
    return ref_MHCI_sequences


def generate_hla_database(data_dir, HLA_out = 'HLA_cleaned.fasta'):
    """
    Downloads and parse HLA sequences

    Args:
        HLA_out (str, optional): Output file for HLA sequences. Defaults to 'HLA_cleaned.fasta'.

    Returns:
        None.

    """
    ###
    # Human alleles
    ###
    # Rename pre-existing raw file
    try:
        os.system('mv %s/HLA_raw.fasta %s/OLD_HLA_raw.fasta' %(data_dir, data_dir))
    except:
        pass

    # Download Human data
    url = 'https://raw.githubusercontent.com/ANHIG/IMGTHLA/Latest/hla_prot.fasta'
    command = (' ').join(['wget', url, '-O', f'{data_dir}/HLA_raw.fasta'])
    proc = subprocess.Popen(command,  executable='/bin/bash',
                                 shell=True, stdout=subprocess.PIPE)
    print(proc.stdout.read())

    HLAs = {}
    to_write = {}
    #Parse the fasta files
    for seq_record in SeqIO.parse(f'{data_dir}/HLA_raw.fasta', "fasta"):
        allele_fullname = seq_record.description.split(' ')[1]
        #allele_significant = allele_fullname[:8]
        #Take only up to the allele identifyer, ignore the silent mutations
        allele_significant = ':'.join(allele_fullname.split(':')[:3])
        #if allele_significant[-1] == ':':
        #    allele_significant = allele_significant[:-1]
        #If the gene name is A, B, C, E, F, G
        if allele_fullname.endswith('N') or allele_fullname.endswith('Q'):
            pass
        #Take only sequences which legth is consistent with MHCI or MHCII chains
        elif ((allele_fullname.split('*')[0][:2] in ['A', 'B', 'C',
                                                 'E', 'F', 'G'] and
              175 < int(seq_record.description.split(' ')[2]) < 380) or
              (allele_fullname.split('*')[0][:2] in ['DP', 'DQ', 'DR'] and
               170 < int(seq_record.description.split(' ')[2]) < 270)):
            try:
                HLAs[allele_significant].append(seq_record)
            except KeyError:
                HLAs[allele_significant] = [seq_record]
        else:
            pass

    #Sort HLA sequences by length. Keep the longest
    for allele in HLAs:
        #If there is only one sequence for the allele
        if len(HLAs[allele]) == 1:
            to_write['HLA-'+allele] = str(HLAs[allele][0].seq)
        else:
            #print(HLAs[allele])
            putatives = sorted(HLAs[allele], key=len, reverse=True)
            #print(putatives, allele)
            #No further filtering criteria are used and the first sequence is taken as reference.
            to_write['HLA-'+allele] = str(putatives[0].seq)

    #Write output fasta file
    with open(data_dir + '/' + HLA_out, 'w') as outfile:
        for allele in to_write:
            outfile.write('>'+allele+'\n')
            for i in range(len(to_write[allele])):
                outfile.write(to_write[allele][i])
                if (i + 1) % 60 == 0: #add line break each time pgcd equal 0
                    outfile.write('\n')
                elif i == len(to_write[allele])-1:
                    outfile.write('\n')

    # Remove pre-existing raw file
    try:
        os.system(f'rm {data_dir}/OLD_HLA_raw.fasta')
    except:
        pass

    return to_write

def generate_nonhla_database(data_dir, nonHLA_out = 'MHC_cleaned.fasta'):
    """
    Downloads and parse non human MHC sequences

    Args:
        nonHLA_out (str, optional): Output file for non human MHCs. Defaults to 'MHC_cleaned.fasta'.

    Returns:
        None.

    """
    ###
    # Other animals alleles
    ###
    # Rename pre-existing raw file
    try:
        os.system(f'mv {data_dir}/MHC_raw.fasta {data_dir}/OLD_MHC_raw.fasta')
    except:
        pass

    # Download other animlas data
    #os.system('wget https://raw.githubusercontent.com/ANHIG/IPDMHC/Latest/MHC_prot.fasta')
    url = 'https://raw.githubusercontent.com/ANHIG/IPDMHC/Latest/MHC_prot.fasta'
    command = (' ').join(['wget', url, '-O', f'{data_dir}/MHC_raw.fasta'])
    proc = subprocess.Popen(command,  executable='/bin/bash',
                                 shell=True, stdout=subprocess.PIPE)
    print(proc.stdout.read())

    MHCs = {}
    to_write = {}
    #Parse the fasta file
    fasta = f'{data_dir}/MHC_raw.fasta'
    for seq_record in SeqIO.parse(fasta, "fasta"):
        allele_fullname = seq_record.description.split(' ')[1]
        #allele_significant = allele_fullname[:8]
        #If the allele name ends with ':', trim it away
        #if allele_significant[-1] == ':':
        #    allele_significant = allele_significant[:-1]
        #If the gene name is Spieces name (Xxxx-A*0 or SLA-A*0)
        #regexp = re.search(r'([A-Z]{1}[a-z]{3}|[A-Z]{3})[-][A-Z0-9]{1:2}[*][0-9]{2:3}[:][0-9]{2:3}',allele_fullname.split('-')[0])
        #if regexp is not None:
            #print(regexp.group(0))
        if allele_fullname.endswith('N') or allele_fullname.endswith('Q'):
            pass
        elif int(seq_record.description.split(' ')[2]) < 350 or int(seq_record.description.split(' ')[2]) > 380:
            pass
        else:
            try:
                MHCs[allele_fullname].append(seq_record)
            except KeyError:
                MHCs[allele_fullname] = [seq_record]

    #Sort MHC sequences by length. Keep the longest
    for allele in MHCs:
        #If there is only one sequence for the allele
        if len(MHCs[allele]) == 1:
            to_write[allele] = str(MHCs[allele][0].seq)
        else:
            #print(HLAs[allele])
            putatives = sorted(MHCs[allele], key=len, reverse=True)
            #print(putatives, allele)
            #No further filtering criteria are used and the first sequence is taken as reference.
            to_write[allele] = str(putatives[0].seq)

    #Write output fasta file
    with open(data_dir + '/' + nonHLA_out, 'w') as outfile:
        for allele in to_write:
            outfile.write('>'+allele+'\n')
            for i in range(len(to_write[allele])):
                outfile.write(to_write[allele][i])
                if (i + 1) % 60 == 0: #add line break each time pgcd equal 0
                    outfile.write('\n')
                elif i == len(to_write[allele])-1:
                    outfile.write('\n')
            #outfile.write(to_write[allele]+'\n')

    # Remove pre-existing raw file
    try:
        os.system(f'rm {data_dir}/OLD_MHC_raw.fasta')
    except:
        pass

    return to_write
