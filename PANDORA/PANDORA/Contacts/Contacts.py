from Bio.PDB import PDBParser
from Bio.PDB import NeighborSearch

class Contacts:
    def __init__(self, PDB, output_file = False, anchors = False, pept_contacts = False, M_only = list(range(1,182)),
                 N_only = list(range(1,92)), cutoff = 5):
        ''' Calculate atom contacts between different chains of a protein or between peptide anchor residues and the
            rest of the MHC structure.

        :param PDB: (String) or (Bio.PDB) Accepts either the path to the pdb file or a Bio.PDB object
        :param output_file: (String) Path to output file. If anchors are provided, only write the anchors output
        :param anchors: (list) list of anchor positions. If this is provided, the atom contacts between the anchor
                        residues of the peptide and the rest of the structure are calculated
        :param cutoff: Cutoff that considers two atoms contacting. Is 5 Angstrom by default
        '''
        self.pdb_path = ''
        self.cutoff = cutoff
        self.anchor_contacts = []
        self.anchors = anchors
        self.output_file = output_file

        # check input type
        if isinstance(PDB, str): #if its a string, it should be the path of the pdb, then load pdb first
            self.pdb_path = PDB
             # Create a parser object, used to read pdb files
            self.PDB = PDBParser(QUIET=True).get_structure('MHC', PDB)
        else: #Else, its already a bio.pdb object
            self.PDB = PDB

        # Actually find the atom contacts between chains
        if not pept_contacts:
            self.chain_contacts = self.find_chain_contacts(self.PDB, self.cutoff)
        if pept_contacts:
            self.chain_contacts = self.find_pept_chain_contacts(self.PDB, M_only=M_only, N_only=N_only, cutoff=self.cutoff)

        # If the user supplied anchors, calculate the anchor contacts
        if anchors: # Calculate peptide anchor residue - structure contacts
            if isinstance(anchors, list) or isinstance(anchors, tuple):
                # First find which chain is the peptide chain by looking for the shortest chain
                chain_len = [len(i) for i in self.PDB.get_chains()]
                pept_chain = [i.id for i in self.PDB.get_chains()][chain_len.index(min(chain_len))]
                # Only keep the contacts of the peptide backbone and CA with the MHC
                self.anchor_contacts = [i for i in self.chain_contacts if (i[1] == pept_chain and i[2] in self.anchors or
                                        i[5] == pept_chain and i[6] in self.anchors) and i[3] in ['N', 'CA', 'C', 'O']]
            else:
                raise Exception('ERROR: target anchors are not a list nor a tuple')


        # Write output file if the user wants to
        if output_file:
            # Write output file if a filename is provided
            with open(output_file, 'w') as f:
                if anchors: # If anchors are provided, only write the anchor contact file
                    for i in self.anchor_contacts:
                        f.write('\t'.join('%s' %x for x in i) + '\n')
                else: # Else write all contacts. If you want both, this one already contains the anchor contacts
                      # ofcourse.
                    for i in self.chain_contacts:
                        f.write('\t'.join('%s' %x for x in i) + '\n')


    def find_chain_contacts(self, PDB, cutoff=5):
        ''' Calculate atom distances in a pdb object.

        :param pdb: Bio.PDB object
        :param cutoff: (int) Distance cutoff that determines atom contact, standard is 5 Angstrom
        :param all: (Bool) Calculate all atom contacts or just between chains
        :param specific_chain: (String) Only consider contacts between one specific chain and the rest.
        :param anchors: (list) list of anchor positions. If such a list is given, only the contacts between the atoms of
                                the peptide anchor residues and the rest of the structure are calculated.
        :return: list of tuples of: Res 1, Chain 1, Resnr 1, Atom 1, Res 2, Chainm 2, Resnr 2, Atom 2, Distance 1_2
        '''
        # Calculate distances
        atoms = [i for i in PDB.get_atoms()]
        atom_dist = NeighborSearch(atom_list=atoms).search_all(cutoff)
        out = []
        # Append (Res 1,Chain 1,Resnr 1,Atom 1,Res 2,Chain 2,Resnr 2,Atom 2,Distance 1_2) to list for each contact
        for pair in atom_dist:
            out.append((pair[1].get_parent().resname,
                        pair[1].get_parent().get_parent().id,
                        pair[1].get_parent().id[1],
                        pair[1].get_id(),
                        pair[0].get_parent().resname,
                        pair[0].get_parent().get_parent().id,
                        pair[0].get_parent().id[1],
                        pair[0].get_id(),
                        pair[0] - pair[1]))

        # If one would want all atom contacts (also within the same protein chain), the next line can be commented out
        out = [i for i in out if i[1] != i[5]] # filter out contacts between the same chain

        return out

    def find_pept_chain_contacts(self, PDB, M_only = list(range(1,182)), N_only = list(range(1,92)), cutoff=5):
        ''' Calculate atom distances in a pdb object.

        :param pdb: Bio.PDB object
        :param cutoff: (int) Distance cutoff that determines atom contact, standard is 5 Angstrom
        :param all: (Bool) Calculate all atom contacts or just between chains
        :param specific_chain: (String) Only consider contacts between one specific chain and the rest.
        :param anchors: (list) list of anchor positions. If such a list is given, only the contacts between the atoms of
                                the peptide anchor residues and the rest of the structure are calculated.
        :return: list of tuples of: Res 1, Chain 1, Resnr 1, Atom 1, Res 2, Chainm 2, Resnr 2, Atom 2, Distance 1_2
        '''
        # Calculate distances

        # Select the atoms in the peptide and the G-domain. This speeds up anchor calculation with 40/60%
        atoms = []
        atoms = atoms + [i for i in PDB[0]['P'].get_atoms()]
        if 'N' in [i.id for i in PDB.get_chains()]:  # Check if its MHCI
            for i in PDB[0]['M'].get_residues():  # Take the G-domain of the M chain
                if i.id[1] in M_only:
                    atoms = atoms + [i for i in i.get_atoms()]
            for i in PDB[0]['N'].get_residues():  # Take the G-domain of the N chain
                if i.id[1] in N_only:
                    atoms = atoms + [i for i in i.get_atoms()]
        elif 'N' not in [i.id for i in PDB.get_chains()]:
            for i in PDB[0]['M'].get_residues():  # Take the G-domain of the M chain
                if i.id[1] in M_only:
                    atoms = atoms + [i for i in i.get_atoms()]

        atom_dist = NeighborSearch(atom_list=atoms).search_all(cutoff)
        out = []
        # Append (Res 1,Chain 1,Resnr 1,Atom 1,Res 2,Chain 2,Resnr 2,Atom 2,Distance 1_2) to list for each contact
        for pair in atom_dist:
            out.append((pair[1].get_parent().resname,
                        pair[1].get_parent().get_parent().id,
                        pair[1].get_parent().id[1],
                        pair[1].get_id(),
                        pair[0].get_parent().resname,
                        pair[0].get_parent().get_parent().id,
                        pair[0].get_parent().id[1],
                        pair[0].get_id(),
                        pair[0] - pair[1]))

        # If one would want all atom contacts (also within the same protein chain), the next line can be commented out
        out = [i for i in out if i[1] != i[5]] # filter out contacts between the same chain

        return out

    def show(self):
        """Print the contacts. If anchor contacts have been calculated, only show them"""
        if self.anchor_contacts:
            for i in self.anchor_contacts:
                print(i)
        else:
            for i in self.chain_contacts:
                print(i)
