# 23-Aug-2018 14:36

import modeller as M
import modeller.automodel as MA
from MyLoop import MyLoop
import sys

M.log.verbose()                                # request verbose output
env = M.environ()                              # create a new MODELLER environment to build this model in

# directories for input atom files
env.io.atom_files_directory = ['./']

# Read in HETATM records from template PDBs
env.io.hetatm = True

a = MyLoop(env, alnfile= '6O4Z.ali',
knowns = ("6O51"), sequence = "6O4Z",
              loop_assess_methods = MA.assess.DOPE)
a.make(exit_stage=2)
