#!/usr/bin/env python
import modeller as M
from modeller import automodel as MA       # Load the automodel class
env = M.environ()

class MyLoop(MA.loopmodel):
    def special_patches(self, aln):
        # Rename both chains and renumber the residues in each
        self.rename_segments(segment_ids=['M', 'N', 'P'], renumber_residues=[1, 1, 1])
        
    ### Skipping randomness in Initial modelling. Uncomment this funtion only if you specifically mean it. 
    def build_ini_loop(self, atmsel):
        pass
    ###
    
    def select_loop_atoms(self):
        # Two residue ranges (both will be refined simultaneously)
        return M.selection(self.residue_range('1:P', '4:P'), self.residue_range('12:P', '15:P'))
        # note: the residue numbers and chain IDs refer to the built model, not to the template(s).

    ### SPECIAL_RESTRAINTS_BREAK ###  DO NOT DELETE THIS COMMENT
    
    def special_restraints(self, aln):
        rsr = self.restraints
        atoms = self.atoms
        modelling_stdev= float('0.1')  #STDEV MARKER   DO NOT DELETE THIS COMMENT
        
        # need to add more distance restraints later :)
        contact_file = open('contacts_134p.list', 'r')
        for contact_data_line in contact_file:
            contact_data = (contact_data_line.replace(' ', '')).split('\t')
            rsr.add(M.forms.gaussian(group=M.physical.xy_distance,feature=M.features.distance(atoms['%s:%s:%s' %(contact_data[3], contact_data[2], contact_data[1])], atoms['%s:%s:%s' %(contact_data[7], contact_data[6], contact_data[5])]),mean=float(contact_data[8]), stdev=modelling_stdev))

        contact_file.close()

        # Central Alpha helix
        # ALPHA-HELIX-MARKER  ### DO NOT DELETE THIS COMMENT

        # Central Beta hairpin
        # BETA-SHEET-MARKER  ### DO NOT DELETE THIS COMMENT