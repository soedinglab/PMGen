import os
import pandas as pd
import shutil
from Bio import PDB


def remove_chain_b(input_pdb, output_pdb):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", input_pdb)
    io = PDB.PDBIO()

    for model in structure:
        chains_to_remove = [chain for chain in model if chain.id == 'B']
        for chain in chains_to_remove:
            model.detach_child(chain.id)

    io.set_structure(structure)
    io.save(output_pdb)
    print(f"Processed PDB saved as: {output_pdb}")


# Example usage:
# remove_chain_b("input.pdb", "output.pdb")


folders = [i for i in os.listdir('./') if os.path.isdir(i)]
for folder in folders:
    file = os.path.join(folder, [i for i in os.listdir(folder) if 'mod0.pdb' in i][0])
    remove_chain_b(file, f'../pandora1/{folder}_pandora.pdb')
    #shutil.copy(file, f'../pandora1/{folder}_pandora.pdb')