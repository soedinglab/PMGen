from pyexpat import model
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from Bio import pairwise2
import os
from Bio.PDB import PDBIO
import PANDORA
import traceback
from Bio.PDB import *
import sys


class Model:

    def __init__(self, target, model_path='', output_dir=False, pdb=False, molpdf=0, dope=0):
        ''' Initiate model object
        Args:
            target: Target object
            model_path: (string) path to hypothetical model
            output_dir: (string) output directory
            pdb:  Bio.PDB object of the hypothetical model
            molpdf: (float) molpdf score
            dope:  (float) DOPE score
        '''

        self.target = target
        self.model_path = model_path
        self.molpdf = molpdf
        self.dope = dope

        # Define the output directory
        if output_dir == False:
            self.output_dir = f"{os.getcwd()}/{self.target.id}"
        else:
            self.output_dir = output_dir
        


        # Check if the user gave either the path to the model pdb or the pdb itself.
        if self.model_path == '' and not pdb:
            raise Exception('Provide the path to a model structure or a Bio.PDB object')
        # If there is a model path and no pdb, parse the pdb structure from that path.
        if not pdb:
            self.pdb = PDBParser(QUIET=True).get_structure(self.target.id, self.model_path)


    def calc_LRMSD(self, reference_pdb, atoms = ['C', 'CA', 'N', 'O'], ligand_zone="whole"):
        """Calculate the L-RMSD between the decoy and reference structure (ground truth).
            This function requires the pdb2sql module for L-RMSD calculation.

        Args:
            reference_pdb: Bio.PDB object or path to pdb file
            atoms (list, optional): The list of atoms of the ligand selected to calculte the LRMSD . Defaults to ['C', 'CA', 'N', 'O'].
            ligand_zone (str, optional): The region of the Ligand selected to calculate the LRMSD. Defaults to "whole".

        Raises:
            Exception: PDB2SQL LRMSD claulation failed
        Returns: (float) L-RMSD calculated by PDB2SQL
        """        

        #from pdb2sql import pdb2sql, superpose, StructureSimilarity
        from pdb2sql import StructureSimilarity

        # load target pdb
        if isinstance(reference_pdb, str):  # if its a string, it should be the path of the pdb, then load pdb first
            ref = PDBParser(QUIET=True).get_structure(self.target.id, reference_pdb)
        else:
            ref = reference_pdb

        # Define file names as variables
        model_name = self.model_path.split('/')[-1].split('.')[1]
        #decoy_path = '%s/%s_decoy.pdb' % (self.output_dir, model_name)
        #ref_path = '%s/%s_ref.pdb' % (self.output_dir, model_name)

        # Define zones to align
        #M_lzone = list(range(4,73))
        #N_lzone = list(range(10,80))

        ## pdb2sql needs 1 big chain and 1 ligand chain with correct numbering, for MHCII, this means merging the chains.       
        #if os.path.exists(decoy_path)==False and ligand_zone.lowe()=="whole":
        if  ligand_zone.lower()=="whole":
            decoy_path, ref_path = homogenize_pdbs(self.pdb, ref, atoms, self.output_dir, model_name)
        ## calculates core LRMSD    
        elif ligand_zone.lower() == "core":
            decoy_path, ref_path = homogenize_pdbs(self.pdb, ref, atoms, self.output_dir, model_name, anchors = self.target.anchors)
        elif ligand_zone.lower() == "flanking":
            decoy_path, ref_path = homogenize_pdbs(self.pdb, ref, atoms, self.output_dir,model_name, anchors = self.target.anchors, flanking=True)

       
        start_dir = os.getcwd()
        os.chdir(self.output_dir)
        # Produce lzone file for the l-rmsd calculation
        #lzone = get_Gdomain_lzone(ref_path, self.output_dir, self.target.MHC_class)
        #TODO: check if it's MHC I or II and adapt for chain M and N
        # Get decoy structure to superpose
        #decoy_db = pdb2sql(decoy_path)
        #decoy_lzone = np.asarray(decoy_db.get('x,y,z', resSeq=M_lzone))

        # Get ref structure to superpose
        #ref_db = pdb2sql(ref_path)
        #ref_lzone = np.asarray(ref_db.get('x,y,z', resSeq=M_lzone))

        # Align the G domains
        #superpose.superpose_selection()

        try:
            # Calculate l-rmsd between decoy and reference with pdb2sql
            sim = StructureSimilarity(decoy_path, ref_path)
            #self.lrmsd = sim.compute_lrmsd_fast(method='svd', name=atoms, lzone = lzone)
            if ligand_zone.lower()=="whole":
                self.lrmsd = sim.compute_lrmsd_pdb2sql(exportpath=None, method='svd', name = atoms)
            ## calculates LRMSD for peptide core region
            elif ligand_zone.lower() == "core":
                self.core_lrmsd = sim.compute_lrmsd_pdb2sql(exportpath=None, method='svd', name=atoms)
            ## calculates RMSD for peptide flanking region
            elif ligand_zone.lower() == "flanking":
                self.flanking_lrmsd = sim.compute_lrmsd_pdb2sql(exportpath=None, method='svd', name=atoms)
        except:
            print('An error occurred while calculating the rmsd for target %s, model %s for %s' %(ref_path, decoy_path, ligand_zone))
            #print('An error occurred while calculating the rmsd for target %s, model %s' %(self.target.id, self.model_path))
            traceback.print_exc()
            raise Exception('Please check your model and ref info for model %s' %self.model_path)

        # remove intermediate files
        os.system('rm %s %s' %(decoy_path, ref_path))
        os.chdir(start_dir)

def merge_chains(pdb):
    ''' Merges two chains of MHCII to one chain. pdb2sql can only calculate L-rmsd with one chain.
    Args:
        pdb: Bio.PDB object
    Returns: Bio.PDB object with its M and N chain merged as M chain
    '''
    # Merge chains
    if 'N' in [chain.id for chain in pdb.get_chains()]:

        for j in pdb[0]['N'].get_residues():
            j.id = (j.id[0], j.id[1], 'M')
            pdb[0]['M'].add(j)

        for i in pdb.get_chains():
            for model in pdb:
                for chain in model:
                    if chain.id in ['N']:
                        model.detach_child(chain.id)
    return pdb


def renumber(pdb_ref, pdb_decoy, custom_map={"MSE":"M"}):
    ''' aligns two pdb's and renumber them accordingly.
    Args:
        pdb_ref:   Bio.PDB object
        pdb_decoy: Bio.PDB object
        
    Returns: Bio.PDB objects with renumbered residues
    '''
    ref_sequences = [[chain.id, seq1(''.join([res.resname for res in chain]), custom_map=custom_map)] for chain in pdb_ref.get_chains()]
                    #[[chain.id, ('').join([seq1(res.resname) for res in chain])]
                    #  for chain in pdb_ref.get_chains()]
    ref_sequences.sort()
    decoy_sequences = [[chain.id, seq1(''.join([res.resname for res in chain]), custom_map=custom_map)] for chain in pdb_decoy.get_chains()]
                        #[[chain.id, ('').join([seq1(res.resname) for res in chain])]
                      #for chain in pdb_decoy.get_chains()]
                        
    decoy_sequences.sort()
    
    assert(len(ref_sequences) == len(decoy_sequences))


    for ind in range(len(ref_sequences)):
        pair = pairwise2.align.globalxx(ref_sequences[ind][1], decoy_sequences[ind][1])[0]
        ref_sequences[ind][1]   = pair.seqA
        decoy_sequences[ind][1] = pair.seqB
        
    ref_sequences = [[seq[0],[i+1 for i,res in enumerate(seq[1]) if res != '-']] for seq in ref_sequences]
    decoy_sequences = [[seq[0],[i+1 for i,res in enumerate(seq[1]) if res != '-']] for seq in decoy_sequences]



    def assign(pdb, pdb_sequences):
        ''' Renumbers the pdb using aligned sequences. 
        Args:
            pdb_ref:   Bio.PDB object
            pdb_decoy: Bio.PDB object
            
        Returns: Bio.PDB objects with renumbered residues
        '''
        
        
        for chain in pdb.get_chains():
            for seq in pdb_sequences:
                if chain.id == seq[0]:
                    for ind, res in enumerate(chain):
                        res.id = ('X', seq[1][ind], res.id[2])
        for chain in pdb.get_chains():
            for res in chain:
                res.id = (' ', res.id[1], ' ')
        return pdb

    pdb_ref = assign(pdb_ref, ref_sequences)
    pdb_decoy = assign(pdb_decoy, decoy_sequences)

    return pdb_ref, pdb_decoy , ref_sequences, decoy_sequences

def homogenize_pdbs(decoy, ref, atoms, output_dir,  target_id = 'MHC' , anchors =False, flanking=False):
    ''' Make sure that the decoy and reference structure have the same structure sequences.
    Args:
        decoy: Bio.PDB object of the decoy structure
        ref: Bio.PDB object of the reference structure
        output_dir: (string) directory that is used to write intermediate files
    Returns: (tuple) Bio.PDB objects with the same structure sequence
    '''
    ref_p_len=len(ref[0]['P'])
    decoy_p_len=len(decoy[0]['P'])
    

    # If you give the anchors, the core L-RMSD will be calculated.
    # The peptide residues before and after the first and last anchor residue will be discarded.
    if anchors and not flanking:
        for x in range(len(decoy[0]['P'])):
            for i in decoy[0]['P']:
                if i.id[1] < anchors[0] or i.id[1] > anchors[-1]:
                    decoy[0]['P'].detach_child(i.id)
            for i in ref[0]['P']:
                if i.id[1] < anchors[0] or i.id[1] > anchors[-1]:
                    ref[0]['P'].detach_child(i.id)

    # If you give the anchors AND flanking = True, the flanking L-RMSD will be calculated. Only if the peptide is
    # also longer than the binding core. The peptide binding core will be discarded

    elif anchors and flanking and decoy_p_len > 9:
        for x in range(ref_p_len):
            for i in decoy[0]['P']:
                if i.id[1] >= anchors[0] and i.id[1] <= anchors[-1]:
                    decoy[0]['P'].detach_child(i.id)
            for i in ref[0]['P']:
                if i.id[1] >= anchors[0] and i.id[1] <= anchors[-1]:
                    ref[0]['P'].detach_child(i.id)


    # remove c-like domain and keep only g domain
    decoy = remove_C_like_domain(decoy)
    ref = remove_C_like_domain(ref)

    # merge chains of the decoy
    decoy = merge_chains(decoy)
    # merge chains of the reference
    ref = merge_chains(ref)

    #Renumber and trim unaligned residues
    ############### Added
    ref, decoy , ref_sequences, decoy_sequences= renumber(ref, decoy)

    ref ,decoy = trim_indels(ref, decoy, ref_sequences, decoy_sequences)

    # trim unaligned atoms
    #ref, decoy , ref_sequences, decoy_sequences= renumber(ref, decoy)
    ref, decoy = remove_mismatched_atoms_from_pdb( ref, decoy,  atoms)


    # Write pdbs
    decoy_path = '%s/%s_decoy.pdb' % (output_dir, target_id)
    io = PDBIO()
    io.set_structure(decoy)
    io.save(decoy_path, select=NotDisordered())

    ref_path = '%s/%s_ref.pdb' % (output_dir, target_id)
    io = PDBIO()
    io.set_structure(ref)
    io.save(ref_path, select=NotDisordered())

    return decoy_path, ref_path

def get_Gdomain_lzone(ref_pdb, output_dir, MHC_class):
    """ Produce a lzone file for pdb2sql.
    Args:
        ref_pdb (str): path to the pdb file to use for the lzone
        output_dir (str): output directory
        MHC_class (str): Class of the MHC
    Raises:
        Exception: In case there are unexpected chain names it raises an exception
    Returns:
        outfile (str): Path to the output file
    """

    ref_name = ref_pdb.split('/')[-1].split('.')[0]
    outfile = '%s/%s.lzone' %(output_dir, ref_name)
    if MHC_class == 'I':
        with open(outfile, 'w') as output:
            P = PDBParser(QUIET=1)
            structure = P.get_structure('r', ref_pdb)
            for chain in structure.get_chains():
                if chain.id == 'M':
                    for x in range(2,173):
                        output.write('zone %s%i-%s%i\n' %(chain.id, x, chain.id, x))
                    #output.write('zone %s2-%s172\n' %(chain.id, chain.id))
                    #output.write('zone %s2-%s172:%s2-%s172\n' %(chain.id, chain.id, chain.id, chain.id))
                elif chain.id == 'P':
                    pass
                    #output.write('fit\n')
                    #for residue in chain:
                    #    if residue.id[2] == ' ':
                    #        output.write('rzone %s%s-%s%s\n' %(chain.id, str(residue.id[1]), chain.id, str(residue.id[1])))
                else:
                    raise Exception('Unrecognized chain ID, different from M or P. Please check your file')
            #output.write('fit\n')

    elif MHC_class == 'II':
        #Chain M from 4 to 72; Chain N from 10 to 80
        with open(outfile, 'w') as output:
            P = PDBParser(QUIET=1)
            structure = P.get_structure('r', ref_pdb)
            for chain in structure.get_chains():
                if chain.id == 'M':
                    output.write('zone %s4-%s72:%s4-%s72\n' %(chain.id, chain.id, chain.id, chain.id))
                elif chain.id == 'N':
                    output.write('zone %s10-%s80:%s10-%s80\n' %(chain.id, chain.id, chain.id, chain.id))
                elif chain.id == 'P':
                    pass
                    #output.write('fit\n')
                    #for residue in chain:
                    #    if residue.id[2] == ' ':
                    #        output.write('rzone %s%s-%s%s\n' %(chain.id, str(residue.id[1]), chain.id, str(residue.id[1])))
                else:
                    raise Exception('Unrecognized chain ID, different from M, N or P. Please check your file')
            #output.write('fit\n')
    return outfile

def remove_C_like_domain(pdb, need_to_be_removed=None):
    """ Removes the C-like domain from a MHC struture and keeps only the G domain
    Args:
        pdb: (Bio.PDB): Bio.PDB object with chains names M (N for MHCII) and P
        need_to_be_removed (list, optional):list of atoms to remove from M chain. Defaults to None.
    Returns: (Bio.PDB): Bio.PDB object without the C-like domain
    """

    # If MHCII, remove the C-like domain from the M-chain (res 80 and higher) and the N-chain (res 90 and higher)
    if 'N' in [chain.id for chain in pdb.get_chains()]:

        residue_ids_to_remove_N = [res.id for res in pdb[0]['N'] if res.id[1] > 90]
        #  Remove them
        for id in residue_ids_to_remove_N:
            pdb[0]['N'].detach_child(id)

        residue_ids_to_remove_M = [res.id for res in pdb[0]['M'] if res.id[1] > 80]
        #  Remove them
        for id in residue_ids_to_remove_M:
            pdb[0]['M'].detach_child(id)

    # If MHCI, remove the C-like domain, which is from residue 180+
    if 'N' not in [chain.id for chain in pdb.get_chains()]:
        for chain in pdb.get_chains():
            if chain.id == 'M':
                if need_to_be_removed ==None:
                    need_to_be_removed = [res.id for res in chain if res.id[1] > 180]
                    _ = [chain.detach_child(x) for x in need_to_be_removed]

                else:
                    #  Remove the list of given atom names 
                    for id in need_to_be_removed:
                        #if (chain.__contains__(id)):
                            chain.detach_child((' ', id, ' '))

                
    return pdb

#ValueError: Invalid column name lzone. Possible names are
#['rowID', 'serial', 'name', 'altLoc', 'resName', 'chainID', 'resSeq',
# 'iCode', 'x', 'y', 'z', 'occ', 'temp', 'element', 'model']


def trim_indels(pdb_ref, pdb_decoy, ref_sequences, decoy_sequences):
    ''' Trim indels for both reference and decoy PDBs

    Args:
        pdb_ref (Bio.PDB): object with chains names M (N for MHCII) and P
        pdb_decoy (Bio.PDB): object with chains names M (N for MHCII) and P
        ref_sequences : List of residue numbers in reference PDB 
        decoy_sequences : List of residue numbers in decoy PDB 
        

    Returns: Bio.PDB objects (reference and decoy) with matched residues
    '''

    ref_decoy_inconsistency=list(set(ref_sequences[0][1]) - set(decoy_sequences[0][1]))
    decoy_ref_inconsistency=list(set(decoy_sequences[0][1]) - set(ref_sequences[0][1]))
   
    need_to_be_removed_ref=list()

    ref_M_length=ref_sequences[0][-1][-1]

    for i  in  ref_decoy_inconsistency:
        
        need_to_be_removed_ref.append(i)
        if i>1 and ((i-1) in ref_sequences[0][1]):
            need_to_be_removed_ref.append(i-1)
        if i<ref_M_length and  ((i+1) in ref_sequences[0][1]):
            need_to_be_removed_ref.append(i+1)
 
    need_to_be_removed_ref=list(set(need_to_be_removed_ref))

    
    decoy_M_length = decoy_sequences[0][-1][-1]
    need_to_be_removed_decoy=list()
    for i  in  decoy_ref_inconsistency:
  
        need_to_be_removed_decoy.append(i)
        if i>1 and ((i-1) in decoy_sequences[0][1]): 
            need_to_be_removed_decoy.append(i-1)
        if i<decoy_M_length and ((i+1) in decoy_sequences[0][1]):
            need_to_be_removed_decoy.append(i+1)
            
    need_to_be_removed_decoy=list(set(need_to_be_removed_decoy))

    new_remove_decoy=list(set(need_to_be_removed_ref) & set(decoy_sequences[0][1]))
    need_to_be_removed_decoy = list(set(need_to_be_removed_decoy + new_remove_decoy))

    new_remove_ref=list(set(need_to_be_removed_decoy) & set(ref_sequences[0][1]))
    need_to_be_removed_ref = list(set(need_to_be_removed_ref + new_remove_ref))
    
    try:
        pdb_ref = remove_C_like_domain(pdb_ref, need_to_be_removed_ref)
    except:
        print('These residue IDs not in ref: %s' %need_to_be_removed_ref)
        print(ref_sequences[0][1])
    try:
        pdb_decoy = remove_C_like_domain(pdb_decoy, need_to_be_removed_decoy)
    except:
        print(decoy_sequences[0][1])
        print('The residue IDs not in decoy: %s' %need_to_be_removed_decoy)
    return pdb_ref, pdb_decoy


def get_residue_atoms(decoy_res, ref_res):
    ''' Retrieves all atom names of a given residue for both reference and decoy

    Args:
        ref_res (Bio.PDB.Residue): Residue object from reference PDB
        decoy_res (Bio.PDB.Residue): Residue object from decoy PDB         

    Returns: Bio.PDB.Atom objects (reference and decoy) of the given residues
    '''
    
    decoy_atoms= list()
    for atom in decoy_res.get_atoms():
        decoy_atoms.append(atom.id)

    ref_atoms = list()
    for atom in ref_res.get_atoms():
        ref_atoms.append(atom.id)

    return decoy_atoms, ref_atoms


def remove_atoms_from_res( ref_chain,  decoy_chain, i):
    '''
    Mismatched residues in Reference is removed both from decoy and reference
    '''
    ref_chain.detach_child((' ', i+1, ' '))
    decoy_chain.detach_child((' ', i+1, ' '))
    print("Mismatch residue in ref vs decoy in resi %s ->  removed in ref & decoy" %(i+1))   

def remove_mismatched_atoms_from_res(diff_atoms, dif_res, atoms):
    '''
    Mismatched atoms in the given residue object is removed
    '''
    diff_removed = list()

    for d in diff_atoms:
        if d in atoms:
            dif_res.detach_child(d)
            diff_removed.append(d)
    return diff_removed

def remove_mismatched_atoms_from_pdb(ref, decoy, atoms):
    """ Mismatched atoms in the given PDB object is removed
    Args:
        ref (Bio.PDB): object with chains names M (N for MHCII) and P
        decoy (Bio.PDB): object with chains names M (N for MHCII) and P
        atoms (_type_): _description_

    Returns:
        _type_: _description_
    """  
    for c in decoy.get_chains():

       chain_id = c.id
       print("chain %s:" %chain_id)
       decoy_m_len=len(decoy[0][chain_id])

       for i in range(decoy_m_len):
           try:
           
                ref_res = ref[0][chain_id][(' ', i+1, ' ')]
                decoy_res = decoy[0][chain_id][(' ', i+1, ' ')]

                # if ref_res.get_resname() != decoy_res.get_resname(): # Eliminate mismatched residues
                if ref_res.get_resname() == 'MSE': # Eliminate mismatched residues
                    remove_atoms_from_res( ref[0][chain_id],  decoy[0][chain_id], i)

                else: # residues matched

                   decoy_atoms, ref_atoms = get_residue_atoms(decoy_res, ref_res)  

                   if(set(decoy_atoms) != set(ref_atoms)) :
                       diff_decoy = set(decoy_atoms) - set(ref_atoms)
                       if diff_decoy:
                           decoy_diff_occured = remove_mismatched_atoms_from_res(diff_decoy, decoy_res, atoms)
       
                       diff_ref = set(ref_atoms) - set(decoy_atoms)
                       if diff_ref:
                           ref_diff_occured = remove_mismatched_atoms_from_res(diff_ref, ref_res, atoms)

                       if decoy_diff_occured and len(decoy_diff_occured) >0:
                           print("Mismatched atoms of Decoy vs ref in residue %d is: %s" %(i+1,decoy_diff_occured))
                       if ref_diff_occured and len(ref_diff_occured) >0:
                           print("Mismatched atoms of ref vs decoy in residue %d is: %s" %(i+1,ref_diff_occured))           
           except:
               continue
    return ref, decoy


class NotDisordered(Select):  # Inherit methods from Select class
    '''
    Keep one Alternative location for the given atom
    '''
    def accept_atom(self, atom):
        keepAltID = 'A'
        if (not atom.is_disordered()) or atom.get_altloc() == keepAltID:
            atom.set_altloc(" ")  # Eliminate alt location ID before output.
            return True
        else:  # Alt location was not one to be output.
            return False
        

