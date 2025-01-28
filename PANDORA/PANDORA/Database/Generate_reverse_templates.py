# NumPy version: 1.26.4
# Biopython version: 1.84
# openmm + pdbfixer version: 8.1.1
# Python version: 3.12.2 | packaged by conda-forge | (main, Feb 16 2024, 20:50:58) [GCC 12.3.0]

import numpy as np
from simtk.openmm.app import PDBFile, Modeller, ForceField, Simulation
from pdbfixer import PDBFixer
from simtk import openmm
from openmm import LangevinIntegrator, Platform
from openmm.unit import kelvin, picosecond, femtoseconds
from Bio.PDB import PDBParser, PDBIO
import os
from typing import List, Dict, Tuple

def read_pdb(filename: str) -> list[tuple[str, np.ndarray, str, str, str, int]]:
    """
    Reads a PDB file, removes hydrogen atoms, the COO- terminal, 
    and replaces selenomethionine residues with methionine.

    Args:
        filename (str): Path to the PDB file.

    Returns:
        list: A list of tuples, each containing the PDB line, coordinates (numpy array),
              atom name, residue name, chain ID, and residue sequence number.
    """
    atoms = []
    with open(filename, 'r') as file:
        for line in file:
            # Replace selenomethionine with methionine
            line = line.replace('SE   MSE ', ' SD  MSE ').replace('          SE', '           S').replace('  MSE ', '  MET ')
            # Remove hydrogen and OXT atoms
            if line.startswith("ATOM") and line[13:16] != 'OXT' and line[77:79] != 'H ':
                atom_serial = int(line[6:11].strip())
                atom_name = line[12:16].strip()
                alt_loc = line[16].strip()
                res_name = line[17:20].strip()
                chain_id = line[21].strip()
                res_seq = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                atoms.append((line, np.array([x, y, z]), atom_name, res_name, chain_id, res_seq))
    return atoms

def calculate_rotation_matrix(direction: np.ndarray) -> np.ndarray:
    """
    Calculates a rotation matrix to align the input direction with the x-axis.

    Args:
        direction (np.ndarray): The vector direction to align.

    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    x_axis = np.array([1, 0, 0])
    axis = np.cross(direction, x_axis)
    angle = np.arccos(np.dot(direction, x_axis) / np.linalg.norm(direction))
    
    axis = axis / np.linalg.norm(axis)
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    
    x, y, z = axis
    matrix = np.array([
        [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, z*z*C + c]
    ])
    return matrix

def write_pdb(atoms: list[tuple[str, np.ndarray, str, str, str, int]], filename: str) -> None:
    """
    Writes the atoms data into a PDB file.

    Args:
        atoms (list): A list of tuples containing atom data.
        filename (str): Output PDB file name.
    """
    with open(filename, 'w') as file:
        for line, _, _, _, _, _ in atoms:
            file.write(line)

def reverse_residue_order(input_pdb: str, chain_id: str, output_pdb: str) -> None:
    """
    Reverses the order of residues in a specific chain of a PDB file.

    Args:
        input_pdb (str): Input PDB file.
        chain_id (str): The chain ID to reverse.
        output_pdb (str): Output PDB file.
    """
    with open(input_pdb, 'r') as file:
        lines = file.readlines()

    # Filter lines corresponding to the specified chain in ATOM records
    atom_lines = [line for line in lines if line.startswith("ATOM") and line[21] == chain_id]

    # Collect atoms by residue
    residue_dict = {}
    for line in atom_lines:
        residue_number = int(line[22:26].strip())
        if residue_number not in residue_dict:
            residue_dict[residue_number] = []
        residue_dict[residue_number].append(line)

    # Sort residues and reverse their order
    sorted_residues = sorted(residue_dict.items())
    sorted_residues.reverse()

    # Generate new residue numbers
    new_residue_order = {}
    new_number = 1
    for old_number, _ in sorted_residues:
        new_residue_order[old_number] = new_number
        new_number += 1

    # Replace residue numbers and maintain the order of atoms within each residue
    updated_lines = []
    for old_number, atoms in sorted_residues:
        for atom in atoms:
            new_residue_number = new_residue_order[old_number]
            new_line = atom[:22] + str(new_residue_number).rjust(4) + atom[26:]
            updated_lines.append(new_line)

    # Collect other lines and replace the original atom lines
    output_lines = [line for line in lines if not (line.startswith("ATOM") and line[21] == chain_id)]

    # Add the new atom lines in the reversed order
    output_lines += updated_lines

    # Write the new PDB file
    with open(output_pdb, 'w') as file:
        file.writelines(output_lines)

def reverse_backbone(atoms: list[tuple[str, np.ndarray, str, str, str, int]]) -> list[tuple[str, np.ndarray, str, str, str, int]]:
    """
    Mirrors the backbone carbonyl (C=O) group with the nitrogen (N) atom across a midpoint between the alpha carbons.

    Args:
        atoms (list): List of tuples containing atom data.

    Returns:
        list: A list of processed atoms with reversed backbone.
    """
    ca_indices = [i for i, atom in enumerate(atoms) if atom[2] == 'CA' and atom[4] == 'P']
    processed_atoms = list(atoms)  # Start with a copy of the original list

    for i in range(len(ca_indices) - 1):
        start_idx = ca_indices[i]
        end_idx = ca_indices[i + 1]
        # Only consider the segment between these two CA indices
        segment = atoms[start_idx+1:end_idx]  # Exclude CA atoms themselves

        ca1 = atoms[start_idx][1]
        ca2 = atoms[end_idx][1]
        midpoint = (ca1 + ca2) / 2
        direction = ca2 - ca1
        rotation_matrix = calculate_rotation_matrix(direction)

        # Rotate, flip x-coordinate, and rotate back only for N, C, O atoms (backbone atoms other than CA)
        for idx, (line, coord, atom_name, res_name, chain_id, res_seq) in enumerate(segment):
            if atom_name in ['N', 'C', 'O'] and chain_id == 'P':  # Process only these backbone atoms in chain P
                translated = coord - midpoint
                rotated = np.dot(rotation_matrix, translated)
                rotated[0] = -rotated[0]  # Flip x-coordinate
                unrotated = np.dot(rotation_matrix.T, rotated) + midpoint
                formatted_line = line[:30] + f"{unrotated[0]:8.3f}{unrotated[1]:8.3f}{unrotated[2]:8.3f}" + line[54:]
                processed_atoms[start_idx+1+idx] = (formatted_line, unrotated, atom_name, res_name, chain_id, res_seq)

    return processed_atoms

def reverse_peptide(pdb_filename: str, output_filename: str, chain_id: str = 'P') -> None:
    """
    Reverse the sequence of a peptide in a PDB file by reassigning atoms to the correct residues.
    This function also flips the termini, removes the terminal carbon at the peptide C-end,
    and the nitrogen at the N-end.

    Parameters:
    - pdb_filename (str): The input PDB file containing the peptide structure.
    - output_filename (str): The output PDB file to write the reversed peptide structure.
    - chain_id (str): The chain identifier of the peptide to be reversed. Default is 'P'.
    """
    
    # Read the PDB file lines
    with open(pdb_filename, 'r') as file:
        pdb_lines = file.readlines()
    
    # Filter out lines that do not belong to the specified chain
    non_chain_lines = [line for line in pdb_lines if line[21] != chain_id]
    chain_lines = [line for line in pdb_lines if line[21] == chain_id]
    
    # Create a dictionary to hold atom information for each residue
    residue_atoms: Dict[int, Dict[str, str]] = {}
    
    for line in chain_lines:
        residue_number = int(line[22:26])
        atom_name = line[12:16].strip()
        if residue_number not in residue_atoms:
            residue_atoms[residue_number] = {}
        residue_atoms[residue_number][atom_name] = line
    
    # Remove terminal oxygen atom at the first residue (if it exists)
    residue_atoms[1].pop('O', None)
    
    # Modify the N-terminal nitrogen atom at the first residue
    residue_atoms[1]['N'] = residue_atoms[1]['C'][:13] + 'N' + residue_atoms[1]['C'][14:77] + 'N  \n'
    residue_atoms[1].pop('C', None)
    
    num_residues = len(residue_atoms)
    
    # Update residue atom positions to match reversed sequence
    for i in range(1, num_residues):
        next_residue_coords = residue_atoms[i + 1]['CA'][17:26]
        residue_atoms[i]['N'] = residue_atoms[i]['N'][:17] + next_residue_coords + residue_atoms[i]['N'][26:]
    
    # Modify the C-terminal carbon atom at the last residue
    residue_atoms[num_residues]['C'] = residue_atoms[num_residues]['N'][:13] + '7' + residue_atoms[num_residues]['N'][14:77] + 'C  \n'
    residue_atoms[num_residues].pop('N', None)
    
    # Update atom positions for all residues except the first
    for i in range(2, num_residues + 1):
        prev_residue_coords = residue_atoms[i - 1]['CA'][17:26]
        residue_atoms[i]['C'] = residue_atoms[i]['C'][:17] + prev_residue_coords + residue_atoms[i]['C'][26:]
        residue_atoms[i]['O'] = residue_atoms[i]['O'][:17] + prev_residue_coords + residue_atoms[i]['O'][26:]
    
    # Final adjustment for the last residue's carbon atom
    residue_atoms[num_residues]['C'] = residue_atoms[num_residues]['C'][:13] + 'C' + residue_atoms[num_residues]['C'][14:77] + 'C  \n'
    
    # Collect and sort the modified chain lines
    sorted_chain_lines: List[str] = []
    for residue_number in sorted(residue_atoms.keys()):
        sorted_chain_lines.extend(residue_atoms[residue_number].values())
    
    # Combine the non-chain and modified chain lines
    final_lines = non_chain_lines + sorted(sorted_chain_lines, key=lambda line: int(line[22:26]))
    
    # Write the reversed peptide to the output PDB file
    with open(output_filename, 'w') as file:
        file.writelines(final_lines)
        
def fix_proline_sidechains(pdb_filename: str = 'tmp.pdb', chain_id: str = 'C') -> None:
    """
    Ensure that proline residues in a PDB file remain in the trans-configuration after flipping.
    This function recalculates the positions of the proline side chains by removing the existing
    CB, CD, and CG atoms and then adding them back with PDBFixer.

    Parameters:
    - pdb_filename (str): The PDB file to process. Default is 'tmp.pdb'.
    - chain_id (str): The chain identifier for the proline residues. Default is 'C'.
    """
    
    # Read the PDB file lines
    with open(pdb_filename, 'r') as file:
        pdb_lines = file.readlines()
    
    # Define the residues to exclude (proline side chains)
    excluded_residues = {f'CB  PRO {chain_id}', f'CD  PRO {chain_id}', f'CG  PRO {chain_id}'}
    
    # Filter out the excluded residues from the PDB lines
    updated_lines = [line for line in pdb_lines if line[13:22] not in excluded_residues]
    
    # Write the updated PDB lines back to the file
    with open(pdb_filename, 'w') as file:
        file.writelines(updated_lines)
    
    # Add missing atoms using PDBFixer (including proline side chain atoms)
    add_missing_atoms(pdb_filename, pdb_filename)

def add_missing_atoms(input_pdb_filename: str, output_pdb_filename: str) -> None:
    """
    Add missing atoms in a PDB structure, typically for termini and proline side chains.

    Parameters:
    - input_pdb_filename (str): The input PDB file containing the structure.
    - output_pdb_filename (str): The output PDB file with the added missing atoms.
    """
    
    # Initialize PDBFixer with the input PDB file
    fixer = PDBFixer(filename=input_pdb_filename)
    
    # Identify and add missing residues and atoms
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    
    # Write the updated structure to the output PDB file
    with open(output_pdb_filename, 'w') as file:
        PDBFile.writeFile(fixer.topology, fixer.positions, file)

def kabsch_algorithm(source_points: np.ndarray, target_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the optimal rotation matrix for superimposing two sets of points using the Kabsch algorithm.

    Parameters:
    - source_points (np.ndarray): The source coordinates to be rotated (N x 3).
    - target_points (np.ndarray): The target coordinates to align to (N x 3).

    Returns:
    - Tuple[np.ndarray, np.ndarray]: A tuple containing the rotation matrix (U) and the translation vector.
    """
    
    # Center the source and target points
    source_center = np.mean(source_points, axis=0)
    target_center = np.mean(target_points, axis=0)
    source_centered = source_points - source_center
    target_centered = target_points - target_center
    
    # Compute the covariance matrix
    covariance_matrix = np.dot(np.transpose(source_centered), target_centered)
    
    # Perform Singular Value Decomposition (SVD)
    V, S, W = np.linalg.svd(covariance_matrix)
    
    # Check for reflection and adjust if necessary
    if (np.linalg.det(V) * np.linalg.det(W)) < 0.0:
        W[-1, :] *= -1
    
    # Calculate the rotation matrix (U)
    rotation_matrix = np.dot(V, W)
    
    # Calculate the translation vector
    translation_vector = target_center - np.dot(source_center, rotation_matrix)
    
    return rotation_matrix, translation_vector

def apply_chirality_transformation(structure) -> None:
    """
    Adjusts the chirality of amino acids in the given structure to achieve the natural L-form.
    The transformation is applied by mirroring the alpha carbons (CAs) across the plane formed
    by the nitrogen (N), carbon (C), and beta carbon (CB) atoms for each amino acid. Glycines 
    are excluded as they are non-chiral. Proline residues are adjusted to the common trans isomer.

    Args:
        structure (Structure): A Bio.PDB structure object representing the protein.

    Returns:
        None
    """
    # Define the atoms involved in the transformation
    target_atoms = ['N', 'C', 'CB']

    # Reference matrix for the template where the CA should be
    reference_matrix = np.array([
        [-1.210, -0.777, 0.115], 
        [-0.071, 1.405, 0.123],
        [1.283, -0.654, 0.114]
    ])
    
    for model in structure:
        for chain in model:
            # Focus on chain 'C' only
            if chain.id == 'C':
                for residue in chain:
                    # Skip glycine residues as they are non-chiral
                    if residue.get_resname() != 'GLY':
                        # Get coordinates of target atoms
                        coords = np.array([residue[atom_name].coord for atom_name in target_atoms])

                        # Perform Kabsch alignment
                        U, translation_vector = kabsch_algorithm(coords, reference_matrix)

                        # Compute new coordinates after transformation
                        transformed_coords = np.dot(coords - np.mean(coords, axis=0), U) + np.mean(reference_matrix, axis=0)
                        transformed_coords = [[-0.003, 0.026, 0.452]]
                        transformed_coords = np.dot(transformed_coords - np.mean(reference_matrix, axis=0), U.T) + np.mean(coords, axis=0)

                        # Update the CA atom's coordinates in the residue
                        for atom in residue:
                            if atom.get_name() == 'CA':
                                atom.set_coord(transformed_coords[0])
                                
def reverse_peptide_chirality(pdb_input: str = "m_7zak4.pdb", output_pdb: str = 'modified_structure.pdb') -> None:
    """
    Reverses the chirality of amino acids in a PDB file, then saves the modified structure
    to a new PDB file. The transformation adjusts the chirality of all amino acids to the 
    natural L-form, excluding glycines and adjusting proline residues to the trans isomer.

    Args:
        pdb_input (str): Path to the input PDB file.
        output_pdb (str): Path to the output PDB file where the modified structure will be saved.

    Returns:
        None
    """
    # Parse the input PDB file to get the structure
    pdb_parser = PDBParser(QUIET=True)
    structure = pdb_parser.get_structure('ID', pdb_input)
    
    # Apply chirality transformation to the structure
    apply_chirality_transformation(structure)
    
    # Save the modified structure to the output PDB file
    pdb_io = PDBIO()
    pdb_io.set_structure(structure)
    pdb_io.save(output_pdb)
    
def fix_and_minimize_structure(input_pdb: str, output_pdb: str) -> None:
    """
    Fixes and minimizes a PDB structure using OpenMM and PDBFixer. The function reintroduces
    missing atoms and hydrogens, then runs a short molecular dynamics simulation to minimize
    the structure while freezing the backbone atoms of chain C.

    Args:
        input_pdb (str): Path to the input PDB file.
        output_pdb (str): Path to the output PDB file where the minimized structure will be saved.

    Returns:
        None
    """
    # Load the PDB file using PDBFixer
    fixer = PDBFixer(filename=input_pdb)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.4)

    # Create a Modeller based on the fixed topology and positions
    modeller = Modeller(fixer.topology, fixer.positions)

    # Load force field parameters
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    modeller.addHydrogens(forcefield)  # Add hydrogens after force field assignment

    # Create an OpenMM System
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=openmm.app.CutoffNonPeriodic)

    # Freeze the backbone atoms of chain C
    for residue in modeller.topology.residues():
        if residue.chain.id != 'C':  # Check if it's chain C
            for atom in residue.atoms():
                if atom.name not in ['H']:  # Check if it's a backbone atom
                    system.setParticleMass(atom.index, 0)  # Set mass to zero to "freeze"

    # Perform energy minimization
    integrator = LangevinIntegrator(300 * kelvin, 1 / picosecond, 2 * femtoseconds)
    platform = Platform.getPlatformByName('CPU')
    simulation = Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(maxIterations=10, tolerance=0.0001)

    # Report the minimized structure
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    minimized_positions = state.getPositions()
    PDBFile.writeFile(modeller.topology, minimized_positions, open(output_pdb, 'w'))
    
    energy = state.getPotentialEnergy()
    print(f"Minimized energy: {energy}")

    # Remove hydrogen atoms from the output PDB file
    with open(output_pdb, 'r') as file:
        lines = ''.join([line for line in file if line.startswith('ATOM') and line[77:79] != 'H '])
    
    with open(output_pdb, 'w') as file:
        file.write(lines)

def change_chain_id(filename: str, old_chain_id: str = 'A', new_chain_id: str = 'M') -> None:
    """
    Changes the chain ID of a specified chain in a PDB file. The function reads the PDB file,
    identifies lines corresponding to the specified old chain ID, and replaces it with a new chain ID.

    Args:
        filename (str): The path to the PDB file where the chain ID needs to be changed.
        old_chain_id (str): The chain ID to be replaced (default is 'A').
        new_chain_id (str): The new chain ID to replace the old one (default is 'M').

    Returns:
        None
    """
    # Read the contents of the PDB file
    lines = open(filename).read().splitlines()

    # Modify the chain ID for relevant lines
    for i, line in enumerate(lines):
        if (line.startswith('ATOM ') or line.startswith('TER ')) and line[21] == old_chain_id:
            lines[i] = line[:21] + new_chain_id + line[22:]

    # Write the modified content back to the PDB file
    with open(filename, 'w') as file:
        file.write('\n'.join(lines))
        
def reverse_pdb_structure(input_pdb: str = "1AQD.pdb", output_pdb: str = "rev.pdb") -> None:
    """
    Processes a PDB file by applying a series of transformations including reversing backbone,
    reversing peptide, fixing missing atoms and residues, adjusting chirality, and minimizing the structure.
    The function also changes chain IDs in the final PDB file and removes temporary files.

    Args:
        input_pdb (str): Path to the input PDB file to be processed.
        output_pdb (str): Path to the output PDB file where the final processed structure will be saved.

    Returns:
        None
    """
    # Read atoms from the input PDB file
    atoms = read_pdb(input_pdb)

    # Process the backbone atoms
    processed_atoms = reverse_backbone(atoms)
    
    # Write the processed atoms to a temporary PDB file
    temp_pdb = "tmp.pdb"
    write_pdb(processed_atoms, temp_pdb)
    
    # Reverse residue order and peptide sequence
    reverse_residue_order(temp_pdb, 'P', temp_pdb)
    reverse_peptide(temp_pdb, temp_pdb)
    
    # Add missing atoms and adjust chirality
    add_missing_atoms(temp_pdb, temp_pdb)
    reverse_peptide_chirality(temp_pdb, temp_pdb)
    
    # Fix proline residues
    fix_proline_sidechains(temp_pdb)
    
    # Perform minimization and save the final structure
    fix_and_minimize_structure(temp_pdb, output_pdb)
    
    # Change chain IDs in the output PDB file
    change_chain_id(output_pdb, 'A', 'M')
    change_chain_id(output_pdb, 'B', 'N')
    change_chain_id(output_pdb, 'C', 'P')
    
    # Remove temporary files
    if os.path.exists(temp_pdb):
        os.remove(temp_pdb)

if __name__ == "__main__":
	# Example usage: reverse_pdb_structure('1J8H.pdb')
	pass
