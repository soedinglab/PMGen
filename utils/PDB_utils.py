from Bio.PDB import PDBParser, PDBIO, Chain
import os.path
import numpy as np
import pandas as pd
import shutil
import re
import os
import sys

# Add the directory containing processing_functions.py to the Python path
current_directory = os.path.dirname(os.path.abspath(__file__))
processing_directory = os.path.join(current_directory, '..', 'processing')
sys.path.append(processing_directory)
import processing_functions

def one_chainer(structure, output): # removes all chains and makes a chain called chain C

    # Parse the structure
    parser = PDBParser()
    structure = parser.get_structure('my_structure', structure)
    
    # Create a PDBIO object
    io = PDBIO()
    
    # Create a new chain
    new_chain = Chain.Chain('C')  # or whatever your desired ID is
    
    # Add all residues from existing chains to the new chain
    for model in structure:
        for chain in model:
            for residue in chain:
                new_chain.add(residue)
    
    # Remove the existing chains
    diction = list(model.child_dict.keys())
    for model in structure:
        for child in diction : model.detach_child(f'{child}')
    
    # Add the new chain to the model
    model.add(new_chain)
    
    # Save the structure
    io.set_structure(structure)
    io.save(output)
    
def copy_pdbs(tsv, pdb_dest, pae_dest, plddt_dest):
    """
    This function is used for copying pdb, pae and plddt files of the final predictions to 
    output file of each sample
    :param tsv: 
    :param pdb_dest: 
    :param pae_dest: 
    :param plddt_dest: 
    :return: 
    """
    df = pd.read_csv(tsv, sep="\t", header=0)
    source_pdb = df.close_pdb_file[0]
    source_pae = df.PAE_file[0]
    source_plddt = df.PLDDT_file[0]
    shutil.copy(source_pdb, pdb_dest)
    shutil.copy(source_pae, pae_dest)
    shutil.copy(source_plddt, plddt_dest)


def pdb_multichainer(seq, source_pdb_path, out_pdb_path, type):
    """
    Generates a pdb multichain from the given pdb file.
    :param seq: 
    :param source_pdb_path: 
    :param out_pdb_path: 
    :param type: 
    :return: 
    """
    if type==1:
        # chain A for seq and B for peptide are generated
        processing_functions.split_and_generate_pdb(
            source_pdb=source_pdb_path, output_pdb=out_pdb_path, cut_num=len(seq))  # differ mhc_1
        # residue indeces are corrected
        processing_functions.correct_residue_indexes(out_pdb_path, out_pdb_path)
    if type==2:
        # chain A for seq1 and B for seq2 and P for peptide are generated
        num_1, num_2 = len(seq.split("/")[0]), len(seq.split("/")[1])
        processing_functions.split_and_generate_pdb_peptide_mhc2(
            source_pdb=source_pdb_path,
            output_pdb=out_pdb_path,
            cut_nums=[num_1, num_2])  # differ mhc_2
        # residue indeces are corrected
        processing_functions.correct_residue_indexes(out_pdb_path, out_pdb_path)
    
def MHC_cleaver_pandora_templates(input, output):

    # Parse the structure
    parser = PDBParser()
    structure = parser.get_structure('my_structure', input)   
    L=[]
    for model in structure:
        for chain in model:
            L.append(chain.id)

    for model in structure:
        for chain in model:
            if chain.id=="B": cutoff_index=0 # Chain B in MHC I is removed
            if chain.id=="M" and "B" in L: cutoff_index=185 # chain M in MHC I is cutted to len 185
            if chain.id=="M" and "N" in L: cutoff_index=95 # chain M in MHC II is cutted to len 95
            if chain.id=="N": cutoff_index=95  # chain N in MHC II is cutted to len 95
            if chain.id=="P": cutoff_index=1000 # no cutting for chain P in both MHCs
            for residue in list(chain):
                if int(residue.id[1]) > cutoff_index:
                    chain.detach_child(residue.id)
    # Save the modified structure
    io = PDBIO()
    io.set_structure(structure)
    io.save(output)



def one_chainer_pandora_templates(structure, output): # removes all chains and makes a chain called chain C

    # Parse the structure
    parser = PDBParser()
    structure = parser.get_structure('my_structure', structure)
    
    # Create a PDBIO object
    io = PDBIO()
    
    # Create a new chain
    new_chain = Chain.Chain('C')  # or whatever your desired ID is
    
    # Add all residues from existing chains to the new chain
    LEN = []
    for model in structure:
        a = 0
        for chain in model:
            LEN.append(a)
            a += len(chain)
    for model in structure:
        for chain, l in zip(model, LEN):
            for residue in chain:
                residue.id = (' ',int(residue.id[1])+l,' ')
                new_chain.add(residue)
    
    # Remove the existing chains

    diction = list(model.child_dict.keys())
    for model in structure:
        for child in diction : model.detach_child(f'{child}')
    
    # Add the new chain to the model
    model.add(new_chain)
    
    # Save the structure
    io.set_structure(structure)
    io.save(output)
