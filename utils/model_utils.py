"""
Protein Mutation Structure Predictor using SE(3)-Equivariant GNN
TensorFlow/Keras Implementation

BATCHED VERSION - All layers handle [B, ...] dimensions

Predicts mutant protein 3D structure given:
- Wild-type protein structure
- Mutation information
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
from Bio.PDB import PDBParser, Superimposer
from typing import List, Tuple, Optional, Dict
import pandas as pd
from dataclasses import dataclass
import functools
import os
import datetime


tf.random.set_seed(42)
mixed_precision = False
if mixed_precision:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

# ============================================================================
# Script Arguments
# ============================================================================
DATA_PREP=False
INPUT_PATH = '/home/amir/amir/ParseFold/PMGen/outputs/good_structures_mutscreen_structure_pred/mutation_structures/structures'
REFERENCE_PATH = '/home/amir/amir/ParseFold/pdbs/test/PMGen'
OUTPUT_DIR = '/home/amir/amir/ParseFold/PMGen/outputs/good_structures_mutscreen_structure_pred/arrays'

os.makedirs(OUTPUT_DIR, exist_ok=True)
# ============================================================================
# CONSTANTS
# ============================================================================

AMINO_ACIDS = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
]
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
NUM_AA = len(AMINO_ACIDS)

ATOM_TYPES = ['N', 'CA', 'C', 'O'] #'CB'
ATOM_TO_IDX = {a: i for i, a in enumerate(ATOM_TYPES)}
NUM_ATOM_TYPES = len(ATOM_TYPES)

# Physicochemical properties:
# [0: KD-hydrophobicity, 1: charge, 2: sulfur, 3: molecular_weight,
#  4: grantham_polarity, 5: miyazawa_hydrophobicity, 6: bulkiness,
#  7-29: BLOSUM62_vector (A, R, N, D, C, Q, E, G, H, I, L, K, M, F, P, S, T, W, Y, V, B, Z, X)]
AA_PROPERTIES = {
    'ALA': [1.8, 0, 0, 89, 8.1, 5.33, 11.5, 4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, -2, -1, 0],
    'ARG': [-4.5, 1, 0, 174, 10.5, 4.18, 14.28, -1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, -1, 0, -1],
    'ASN': [-3.5, 0, 0, 132, 11.6, 3.71, 12.82, -2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 3, 0, -1],
    'ASP': [-3.5, -1, 0, 133, 13.0, 3.59, 11.68, -2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 4, 1, -1],
    'CYS': [2.5, 0, 1, 121, 5.5, 7.93, 13.46, 0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2],
    'GLN': [-3.5, 0, 0, 146, 10.5, 3.87, 14.45, -1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0, 3, -1],
    'GLU': [-3.5, -1, 0, 147, 12.3, 3.65, 13.57, -1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1],
    'GLY': [-0.4, 0, 0, 75, 9.0, 4.48, 3.4, 0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, -1, -2, -1],
    'HIS': [-3.2, 0.5, 0, 155, 10.4, 5.1, 13.69, -2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0, 0, -1],
    'ILE': [4.5, 0, 0, 131, 5.2, 8.83, 21.4, -1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, -3, -3, -1],
    'LEU': [3.8, 0, 0, 131, 4.9, 8.47, 21.4, -1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, -4, -3, -1],
    'LYS': [-3.9, 1, 0, 146, 11.3, 2.95, 15.71, -1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0, 1, -1],
    'MET': [1.9, 0, 1, 149, 5.7, 8.95, 16.25, -1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, -3, -1, -1],
    'PHE': [2.8, 0, 0, 165, 5.2, 9.03, 19.8, -2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, -3, -3, -1],
    'PRO': [-1.6, 0, 0, 115, 8.0, 3.87, 17.43, -1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, -2, -1, -2],
    'SER': [-0.8, 0, 0, 105, 9.2, 4.09, 9.47, 1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0, 0, 0],
    'THR': [-0.7, 0, 0, 119, 8.6, 4.49, 15.77, 0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, -1, -1, 0],
    'TRP': [-0.9, 0, 0, 204, 5.4, 7.66, 21.67, -3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, -4, -3, -2],
    'TYR': [-1.3, 0, 0, 181, 6.2, 5.89, 18.03, -2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, -3, -2, -1],
    'VAL': [4.2, 0, 0, 117, 5.9, 7.63, 21.57, 0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, -3, -2, -1]
}

# Atom indices
ATOM_N = 0
ATOM_CA = 1
ATOM_C = 2
ATOM_O = 3
ATOM_CB = 4

VDW_RADII = {
    'N': 1.55,
    'CA': 1.70,
    'C': 1.70,
    'O': 1.52,
    #'CB': 1.70,
}

LDDT_THRESHOLDS = [0.5, 1.0, 2.0, 4.0]


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ProteinStructure:
    """Protein structure container"""
    coords: tf.Tensor  # [N_residues, N_atoms, 3] or [B, N_residues, N_atoms, 3]
    sequence: List[str]  # List of amino acid codes
    atom_mask: tf.Tensor  # [N_residues, N_atoms] or [B, N_residues, N_atoms]

    @property
    def n_residues(self) -> int:
        return len(self.sequence)

    @property
    def ca_coords(self) -> tf.Tensor:
        return self.coords[..., ATOM_TO_IDX['CA'], :]


@dataclass
class Mutation:
    """Single point mutation"""
    position: int  # 0-indexed
    wild_type: str  # 3-letter code
    mutant: str  # 3-letter code

    def __repr__(self):
        return f"{self.wild_type}{self.position + 1}{self.mutant}"


# ============================================================================
# PREP FUNCTIONS
# ============================================================================

def parse_and_align_pdb(input_pdb_path, reference_pdb_path):
    parser = PDBParser(QUIET=True)
    input_structure = parser.get_structure('input', input_pdb_path)
    ref_structure = parser.get_structure('reference', reference_pdb_path)
    input_residues = [res for res in input_structure.get_residues() if res.get_id()[0] == ' ']
    ref_residues = [res for res in ref_structure.get_residues() if res.get_id()[0] == ' ']
    input_ca_atoms = [res['CA'] for res in input_residues]
    ref_ca_atoms = [res['CA'] for res in ref_residues]
    super_imposer = Superimposer()
    super_imposer.set_atoms(ref_ca_atoms, input_ca_atoms)
    super_imposer.apply(input_structure.get_atoms())
    n_res = len(input_residues)
    coords = np.zeros((n_res, 4, 3), dtype=np.float32)
    aa_indices = np.zeros(n_res, dtype=np.int64)
    backbone_atoms = ['N', 'CA', 'C', 'O']
    for i, res in enumerate(input_residues):
        res_name = res.get_resname()
        aa_indices[i] = AA_TO_IDX.get(res_name, -1)
        for j, atom_name in enumerate(backbone_atoms):
            if atom_name in res:
                coords[i, j] = res[atom_name].get_coord()
    return coords, aa_indices #(N,4,3), (N,)

def parse_and_align_pdb_all(input_dir, reference_dir, output_dir, PAD_TOKEN=-2):
    os.makedirs(output_dir, exist_ok=True)
    input_strs = [f for f in os.listdir(input_dir) if f.endswith('.pdb')]
    print(f"Found {len(input_strs)} PDB files in {input_dir}")
    COORDS = []
    SEQS = []
    PDBS = []
    FULL_NAME = []
    RES_MASK = []
    max_len = 0
    failed = []
    for input_str in input_strs:
        inp_path = os.path.join(input_dir, input_str)
        pdb = input_str.split('_')[0]
        ref_path = os.path.join(reference_dir, f'{pdb}_PMGen.pdb')
        try:
            coords, aa_indices = parse_and_align_pdb(inp_path, ref_path)
            if coords.shape[0] > max_len:
                max_len = coords.shape[0]
            COORDS.append(coords)
            SEQS.append(aa_indices)
            PDBS.append(pdb)
            FULL_NAME.append(input_str)
            RES_MASK.append(coords.shape[0])
        except Exception as e:
            failed.append(input_str)
            print(f"Warning: Failed to process {input_str}: {e}")
    print(f"Successfully processed {len(COORDS)}/{len(input_strs)} structures")
    print(f"Max sequence length: {max_len}")
    if failed:
        print(f"Failed structures: {failed}")
    print("Padding arrays...")
    COORDS = [np.pad(c, ((0, max_len - c.shape[0]), (0, 0), (0, 0)), constant_values=0.) for c in COORDS]
    SEQS = [np.pad(s, (0, max_len - s.shape[0]), constant_values=PAD_TOKEN) for s in SEQS]
    COORDS = np.array(COORDS)
    SEQS = np.array(SEQS)
    print(f"Final COORDS shape: {COORDS.shape}")
    print(f"Final SEQS shape: {SEQS.shape}")
    DF = pd.DataFrame({'full_name': FULL_NAME, 'pdb': PDBS, 'length': RES_MASK})
    DF['index'] = range(len(DF))
    print("Saving outputs...")
    np.save(os.path.join(output_dir, 'coords.npy'), COORDS)
    np.save(os.path.join(output_dir, 'seqs.npy'), SEQS)
    DF.to_csv(os.path.join(output_dir, 'df.csv'), index=False)
    pair_df = DF.merge(DF, on='pdb', suffixes=('_1', '_2'))
    pair_df = pair_df[pair_df['index_1'] != pair_df['index_2']]
    pair_df.to_csv(os.path.join(output_dir, 'paired.csv'), index=False)
    print(f"Saved coords.npy, seqs.npy, and df.csv to {output_dir}")

# ============================================================================
# UTILITY FUNCTIONS (BATCHED)
# ============================================================================

def recompute_grad_preserve_dtype(fn):
    """Wrapper around tf.recompute_grad that preserves input dtypes."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # Capture dtypes from first call
        input_dtypes = [arg.dtype if hasattr(arg, 'dtype') else None for arg in args]

        def dtype_preserved_fn(*inner_args, **inner_kwargs):
            # Cast inputs back to expected dtypes
            casted_args = []
            for arg, dtype in zip(inner_args, input_dtypes):
                if dtype is not None and hasattr(arg, 'dtype') and arg.dtype != dtype:
                    casted_args.append(tf.cast(arg, dtype))
                else:
                    casted_args.append(arg)
            return fn(*casted_args, **inner_kwargs)

        return tf.recompute_grad(dtype_preserved_fn)(*args, **kwargs)

    return wrapper

def safe_norm(x: tf.Tensor, axis: int = -1, keepdims: bool = False,
              eps: float = 1e-8) -> tf.Tensor:
    """Compute norm with numerical stability"""
    return tf.sqrt(tf.reduce_sum(x ** 2, axis=axis, keepdims=keepdims) + eps)


def safe_normalize(x: tf.Tensor, axis: int = -1, eps: float = 1e-8) -> tf.Tensor:
    """Normalize vectors with numerical stability"""
    norm = safe_norm(x, axis=axis, keepdims=True, eps=eps)
    return x / norm


def build_radius_graph(coords: tf.Tensor, mask: tf.Tensor,
                       cutoff: float = 10.0) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Build radius graph from coordinates (BATCHED)

    Args:
        coords: [B, N, 3] atom coordinates
        mask: [B, N] valid atom mask
        cutoff: distance cutoff in Angstroms

    Returns:
        adj: [B, N, N] adjacency matrix
        dist: [B, N, N] distance matrix
    """
    # Compute pairwise distances [B, N, N, 3]
    diff = tf.expand_dims(coords, 2) - tf.expand_dims(coords, 1)
    dist = safe_norm(diff, axis=-1)  # [B, N, N]

    # Create adjacency based on cutoff
    adj = tf.cast(dist < cutoff, tf.float32)

    # Remove self-loops
    n_atoms = tf.shape(coords)[1]
    batch_size = tf.shape(coords)[0]
    adj = adj * (1.0 - tf.eye(n_atoms, batch_shape=[batch_size]))

    # Apply mask (only edges between valid atoms)
    mask_2d = tf.expand_dims(mask, 2) * tf.expand_dims(mask, 1)  # [B, N, N]
    adj = adj * mask_2d

    return adj, dist


def rbf_encode(distances: tf.Tensor, num_rbf: int = 32,
               d_min: float = 0.0, d_max: float = 20.0) -> tf.Tensor:
    """Radial basis function encoding of distances (supports any shape)"""
    centers = tf.linspace(d_min, d_max, num_rbf)
    gamma = (d_max - d_min) / num_rbf
    diff = tf.expand_dims(distances, -1) - centers
    return tf.exp(-(diff ** 2) / (2 * gamma ** 2))


def build_batched_node_adjacency(coords, mask, threshold):
    """
    Builds a (B, N, N) adjacency matrix based on distance and padding.
    Args:
        coords: Tensor of shape (B, N, 3)
        mask: Tensor of shape (B, N). 1.0 for NORMAL_TOKEN, 0.0 for PAD_TOKEN.
        threshold: Float, distance cutoff.
    Returns:
        adj: Tensor of shape (B, N, N) with 1.0 for neighbors and 0.0 otherwise.
    """
    # 1. Compute pairwise squared distances using broadcasting
    # [B, N, 1, 3] - [B, 1, N, 3] -> [B, N, N, 3]
    diff = tf.expand_dims(coords, 2) - tf.expand_dims(coords, 1)
    dist_sq = tf.reduce_sum(tf.square(diff), axis=-1)
    dist = tf.sqrt(dist_sq + 1e-8)
    # 2. Apply the threshold
    adj = tf.cast(dist <= threshold, tf.float32)
    # 3. Set diagonal to 0 (remove self-loops)
    # tf.linalg.set_diag needs a vector of shape (B, N) to put on the diagonal
    batch_size = tf.shape(coords)[0]
    n_res = tf.shape(coords)[1]
    adj = tf.linalg.set_diag(adj, tf.zeros([batch_size, n_res], dtype=tf.float32))
    # 4. Handle padding
    # A connection (i, j) is only valid if both node i and node j are not padded.
    # [B, N, 1] * [B, 1, N] -> [B, N, N]
    valid_connections_mask = tf.expand_dims(mask, -1) * tf.expand_dims(mask, -2)
    # Apply the mask to the adjacency matrix
    adj = adj * valid_connections_mask
    return adj #(B,N,N)


def build_batched_edge_matrices(node_adj_matrix: tf.Tensor, coords: tf.Tensor):
    """
    Builds incidence matrix, edge coadjacency matrix, edge distances, and edge mask.

    Args:
        node_adj_matrix: Tensor of shape (B, N, N) - the node adjacency matrix
        coords: Tensor of shape (B, N, 3) - node coordinates

    Returns:
        incidence: Tensor of shape (B, E, N) - edge-node incidence matrix
        edge_coadj: Tensor of shape (B, E, E) - edge coadjacency matrix
        edge_dist: Tensor of shape (B, E, 1) - Euclidean distance for each edge
        edge_mask: Tensor of shape (B, E) - 1.0 for real edges, 0.0 for padding
    """
    # 1. Extract upper triangle to get undirected edges
    upper_tri = tf.linalg.band_part(node_adj_matrix, 0, -1)
    upper_tri = upper_tri - tf.linalg.band_part(upper_tri, 0, 0)

    # 2. Get edge indices: [batch_idx, node_u, node_v]
    edge_indices = tf.where(upper_tri > 0.5)
    batch_ids = edge_indices[:, 0]
    node_u = edge_indices[:, 1]
    node_v = edge_indices[:, 2]

    # 3. Compute local edge IDs within each batch
    ones = tf.ones_like(batch_ids, dtype=tf.int32)
    global_edge_counts = tf.cumsum(ones)

    edges_per_batch = tf.reduce_sum(tf.cast(upper_tri > 0.5, tf.int32), axis=(1, 2))
    batch_starts = tf.gather(
        tf.concat([[0], tf.cumsum(edges_per_batch)], axis=0),
        tf.cast(batch_ids, tf.int32)
    )
    local_edge_ids = tf.cast(global_edge_counts - batch_starts - 1, tf.int32)

    # 4. Determine output shape
    max_edges = tf.reduce_max(edges_per_batch)
    batch_size = tf.shape(node_adj_matrix)[0]
    num_nodes = tf.shape(node_adj_matrix)[1]

    # Cast indices to int32
    batch_ids_int = tf.cast(batch_ids, tf.int32)
    node_u_int = tf.cast(node_u, tf.int32)
    node_v_int = tf.cast(node_v, tf.int32)

    # Common scatter indices for edge-level features (B, E)
    edge_scatter_indices = tf.stack([batch_ids_int, local_edge_ids], axis=1)

    # 5. Build incidence matrix (B, E, N)
    indices_u = tf.stack([batch_ids_int, local_edge_ids, node_u_int], axis=1)
    indices_v = tf.stack([batch_ids_int, local_edge_ids, node_v_int], axis=1)

    all_indices = tf.concat([indices_u, indices_v], axis=0)
    all_updates = tf.ones(tf.shape(all_indices)[0], dtype=tf.float32)

    incidence = tf.scatter_nd(
        all_indices,
        all_updates,
        shape=(batch_size, max_edges, num_nodes)
    )

    # 6. Compute coadjacency from incidence: C = I @ I^T
    edge_coadj = tf.matmul(incidence, incidence, transpose_b=True)
    edge_coadj = tf.linalg.set_diag(
        edge_coadj,
        tf.zeros([batch_size, max_edges], dtype=tf.float32)
    )
    edge_coadj = tf.cast(edge_coadj > 0, tf.float32)

    # 7. Compute edge distances (B, E, 1)
    coord_indices_u = tf.stack([batch_ids_int, node_u_int], axis=1)
    coord_indices_v = tf.stack([batch_ids_int, node_v_int], axis=1)

    coords_u = tf.gather_nd(coords, coord_indices_u)
    coords_v = tf.gather_nd(coords, coord_indices_v)

    distances = tf.sqrt(tf.reduce_sum(tf.square(coords_u - coords_v), axis=-1))

    edge_dist = tf.scatter_nd(
        edge_scatter_indices,
        distances,
        shape=(batch_size, max_edges)
    )
    edge_dist = tf.expand_dims(edge_dist, -1)

    # 8. Build edge mask (B, E)
    num_edges = tf.shape(edge_indices)[0]
    edge_mask = tf.scatter_nd(
        edge_scatter_indices,
        tf.ones(num_edges, dtype=tf.float32),
        shape=(batch_size, max_edges)
    )

    return incidence, edge_coadj, edge_dist, edge_mask


# ============================================================================
# CUSTOM KERAS LAYERS (BATCHED)
# ============================================================================
@tf.keras.utils.register_keras_serializable(package='custom_layers', name='RBFLayer')
class RBFLayer(layers.Layer):
    """Radial Basis Function encoding layer"""

    def __init__(self, num_rbf: int = 32, d_min: float = 0.0,
                 d_max: float = 20.0, **kwargs):
        super().__init__(**kwargs)
        self.num_rbf = num_rbf
        self.d_min = d_min
        self.d_max = d_max

    def build(self, input_shape):
        self.centers = tf.linspace(self.d_min, self.d_max, self.num_rbf)
        self.gamma = (self.d_max - self.d_min) / self.num_rbf

    def call(self, distances):
        """distances: any shape [...], returns [..., num_rbf]"""
        diff = tf.expand_dims(distances, -1) - tf.cast(self.centers, self.compute_dtype)
        return tf.exp(-(diff ** 2) / (2 * self.gamma ** 2))

@tf.keras.utils.register_keras_serializable(package='custom_layers', name='AttEGNNLayer')
class AttEGNNLayer(layers.Layer):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 64,
                 name='AttEGNN', heads: int = 4,
                 return_attention_map: bool = False, edge_att=False, **kwargs):
        super(AttEGNNLayer, self).__init__(name=name, **kwargs)  # BUG FIX: Must call parent __init__ for Keras layers to work properly
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self._name = name  # BUG FIX: Use _name to avoid conflict with parent class 'name' property
        self.heads = heads
        self.return_attention_map = return_attention_map
        self.edge_att = edge_att

    def build(self, input_shape):
        # general params
        self.scale = 1 / tf.sqrt(tf.cast(self.hidden_dim, dtype=self.compute_dtype))
        # Define Node layers --> self att
        self.node_ln1 = layers.LayerNormalization(name = f'{self._name}_node_ln1')
        self.W_node_key = self.add_weight(shape=(self.heads, 1, self.node_dim, self.hidden_dim), initializer='random_normal', trainable=True, name=f'{self._name}_W_node_key', dtype=self.compute_dtype)
        self.W_node_query = self.add_weight(shape=(self.heads, 1, self.node_dim, self.hidden_dim), initializer='random_normal', trainable=True, name=f'{self._name}_W_node_query', dtype=self.compute_dtype)
        self.W_node_value = self.add_weight(shape=(self.heads, 1, self.node_dim, self.node_dim), initializer='random_normal', trainable=True, name=f'{self._name}_W_node_value', dtype=self.compute_dtype)
        self.node_bias = self.add_weight(shape=(self.heads, 1, self.node_dim, 1), initializer='zeros', trainable=True, name=f'{self._name}_node_bias', dtype=self.compute_dtype)
        self.node_gate = self.add_weight(shape=(self.heads, 1, self.node_dim, self.node_dim), initializer='random_uniform', trainable=True, name=f'{self._name}_node_gate', dtype=self.compute_dtype)
        self.node_out = self.add_weight(shape=(int(self.heads * self.node_dim), self.node_dim), initializer='random_normal', trainable=True, name=f'{self._name}_node_out', dtype=self.compute_dtype)
        self.node_ln2 = layers.LayerNormalization(name = f'{self._name}_node_ln2')
        # Define Edge Layers --> self att
        if self.edge_att:
            self.edge_ln1 = layers.LayerNormalization(name = f'{self._name}_edge_ln1')
            self.W_edge_key = self.add_weight(shape=(self.heads, 1, self.edge_dim, self.hidden_dim), initializer='random_normal', trainable=True, name=f'{self._name}_W_edge_key', dtype=self.compute_dtype)
            self.W_edge_query = self.add_weight(shape=(self.heads, 1, self.edge_dim, self.hidden_dim), initializer='random_normal', trainable=True, name=f'{self._name}_W_edge_query', dtype=self.compute_dtype)
            self.W_edge_value = self.add_weight(shape=(self.heads, 1, self.edge_dim, self.edge_dim), initializer='random_normal', trainable=True, name=f'{self._name}_W_edge_value', dtype=self.compute_dtype)
            self.edge_bias = self.add_weight(shape=(self.heads, 1, self.edge_dim, 1), initializer='zeros', trainable=True, name=f'{self._name}_edge_bias', dtype=self.compute_dtype)
            self.edge_gate = self.add_weight(shape=(self.heads, 1, self.edge_dim, self.edge_dim), initializer='random_uniform', trainable=True, name=f'{self._name}_edge_gate', dtype=self.compute_dtype)
            self.edge_out = self.add_weight(shape=(int(self.heads * self.edge_dim), self.edge_dim),initializer='random_normal', trainable=True, name=f'{self._name}_edge_out', dtype=self.compute_dtype)
        else:
            self.edge_out1 = self.add_weight(shape=(self.edge_dim, self.hidden_dim),initializer='random_normal', trainable=True, name=f'{self._name}_edge_out1', dtype=self.compute_dtype)
            self.edge_out2 = self.add_weight(shape=(self.hidden_dim, self.edge_dim),initializer='random_normal', trainable=True, name=f'{self._name}_edge_out2', dtype=self.compute_dtype)
        self.edge_ln2 = layers.LayerNormalization(name = f'{self._name}_edge_ln2')
        # update edge || node --> edge
        self.update_w_edge1 = self.add_weight(shape=(self.edge_dim + self.node_dim, self.hidden_dim), initializer='random_normal', trainable=True, name=f'{self._name}_update_w_edge1')
        self.update_w_edge2 = self.add_weight(shape=(self.hidden_dim, self.edge_dim),initializer='random_normal', trainable=True, name=f'{self._name}_update_w_edge2')
        self.edge_ln3 = layers.LayerNormalization(name = f'{self._name}_edge_ln3')
        # update node || edge --> node
        self.update_w_node1 = self.add_weight(shape=(self.edge_dim + self.node_dim, self.hidden_dim), initializer='random_normal', trainable=True, name=f'{self._name}_update_w_node1')
        self.update_w_node2 = self.add_weight(shape=(self.hidden_dim, self.node_dim),initializer='random_normal', trainable=True, name=f'{self._name}_update_w_node2')
        self.node_ln3 = layers.LayerNormalization(name = f'{self._name}_node_ln3')
        # Coordinate net
        # Coordinate net
        self.coord_mlp1 = self.add_weight(shape=(2 * self.node_dim + 1, self.hidden_dim),
                                          initializer='random_normal', trainable=True,
                                          name=f'{self._name}_coord_mlp1', dtype=self.compute_dtype)
        self.coord_mlp2 = self.add_weight(shape=(self.hidden_dim, 1),
                                          initializer='random_normal', trainable=True,
                                          name=f'{self._name}_coord_mlp2', dtype=self.compute_dtype)

    def _node_attention(self, node_features, node_adj_matrix):
        node_Q = tf.matmul(node_features, self.W_node_query)  # (B,N,d_n) . (H, 1, d_n, d_h) -> (H,B,N,d_h)
        node_K = tf.matmul(node_features, self.W_node_key)  # (H,B,N,d_h)
        node_V = tf.matmul(node_features, self.W_node_value)  # (B,N,d_n) . (H,1,d_n, d_n) -> (H,B,N,d_n)
        node_G = tf.nn.sigmoid(tf.matmul(node_features, self.node_gate))  # (H,B,N,d_n)
        node_B = tf.matmul(node_features, self.node_bias)  # (H,B,N,1)
        # node attention
        node_QK = tf.matmul(node_Q, node_K, transpose_b=True) * self.scale  # (H,B,N,N)
        # mask out non connected nodes
        node_mask_out = -1e4 * (1. - tf.expand_dims(node_adj_matrix, axis=0))  # (1,B,N,N) all with 0. value become -1e9
        node_att = tf.nn.softmax(node_QK + node_mask_out + node_B)  # (H,B,N,N) + (1,B,N,N) + (H,B,N,1)
        node_att *= tf.expand_dims(node_adj_matrix, axis=0)  # (mask out padded ones or the ones with no neighbour)
        node_out = tf.matmul(node_att,
                             node_V)  # (H,B,N,N) * (H,B,N,d_n) -> (H,B,N,d_n) !! it is d_n so we can do resnet
        node_out += tf.expand_dims(node_features, 0)  # (1,B,N,d_n) + (H,B,N,d_n)
        node_out = self.node_ln1(node_out)
        # gating
        node_out *= node_G
        node_out = tf.transpose(node_out, perm=[1, 2, 3, 0])
        b, h, n, o = tf.shape(node_out)[0], tf.shape(node_out)[-1], tf.shape(node_out)[1], tf.shape(node_out)[2]
        node_out = tf.reshape(node_out, [b, n, h * o])  # (B, N, H * d_n)
        node_out = tf.nn.relu(tf.matmul(node_out, self.node_out))  # (B, N, d_n)
        return node_out, node_att

    def _edge_attention(self, edge_features, edge_coadj):
        if not self.edge_att:
            # convolutional operation H_1 = A*E*W
            AE = tf.matmul(edge_coadj, edge_features) #(B,E,E) * (B,E,d_e) -> (B,E,d_e)
            AEW = tf.matmul(AE, self.edge_out1) #(B,E,d_e) * (d_e, d_h) -> (B,E,d_h)
            edge_att = tf.matmul(AEW, self.edge_out2) #(B,E,d_h) * (d_h, d_e) -> (B,E,d_e)
            edge_out = tf.nn.relu(edge_att)
        else:
            edge_Q = tf.matmul(edge_features, self.W_edge_query) #(B,E,d_e) . (H,1,d_e,d_h) -> (H,B,E,d_h)
            edge_K = tf.matmul(edge_features, self.W_edge_key)
            edge_V = tf.matmul(edge_features, self.W_edge_value) # (B,E,d_e) . (H,1,d_e, d_e) -> (H,B,E,d_e)
            edge_G = tf.nn.sigmoid(tf.matmul(edge_features, self.edge_gate)) # (H,B,E,d_e)
            edge_B = tf.matmul(edge_features, self.edge_bias) # (H,B,E,1)
            # edge attention
            edge_QK = tf.matmul(edge_Q, edge_K, transpose_b=True) * self.scale
            #mask out non connected edges
            edge_mask_out = -1e4 * (1. - tf.expand_dims(edge_coadj, axis=0)) #(1,B,E,E)
            edge_att = tf.nn.softmax(edge_QK + edge_mask_out + edge_B) #(H,B,E,E) + (1,B,E,E) + (H,B,E,1)
            edge_att *= tf.expand_dims(edge_coadj, axis=0)
            edge_out = tf.matmul(edge_att, edge_V) # (H,B,E,E) * (H,B,E,d_e) -> (H,B,E,d_e)
            edge_out += tf.expand_dims(edge_features, 0) # (H,B,E,d_e) + (H,B,E,d_e)
            edge_out = self.edge_ln1(edge_out)
            # gating
            edge_out *= edge_G # (H,B,E,d_e) * (H,B,E,d_e)
            edge_out = tf.transpose(edge_out, perm=[1, 2, 3, 0])
            b, h, n, o = tf.shape(edge_out)[0], tf.shape(edge_out)[-1], tf.shape(edge_out)[1], tf.shape(edge_out)[2]
            edge_out = tf.reshape(edge_out, [b, n, h * o]) #(B,E,d_e * H)
            edge_out = tf.nn.relu(tf.matmul(edge_out, self.edge_out)) # (B, E, d_e)
        return edge_out, edge_att

    def _coord_update(self, node_update, coords_flat, node_adj_matrix, atom_mask_flat_zero):
        """
        Sparse coordinate update - O(E) memory instead of O(N²)
        Only computes for actual edges in adjacency matrix.
        """
        batch_size = tf.shape(coords_flat)[0]
        n_nodes = tf.shape(coords_flat)[1]
        # Get edge indices from adjacency: (E_total, 3) with [batch, src, dst]
        edge_indices = tf.where(node_adj_matrix > 0)
        batch_idx = tf.cast(edge_indices[:, 0], tf.int32)
        src_idx = tf.cast(edge_indices[:, 1], tf.int32)
        dst_idx = tf.cast(edge_indices[:, 2], tf.int32)
        # Flat indices for gathering: batch * N + node_idx
        flat_src = batch_idx * n_nodes + src_idx
        flat_dst = batch_idx * n_nodes + dst_idx
        # Gather node features for edge endpoints: (E_total, d_n)
        node_flat = tf.reshape(node_update, [-1, self.node_dim])
        node_i = tf.gather(node_flat, flat_src)
        node_j = tf.gather(node_flat, flat_dst)
        # Gather coordinates: (E_total, 3)
        coords_reshape = tf.reshape(coords_flat, [-1, 3])
        coord_i = tf.gather(coords_reshape, flat_src)
        coord_j = tf.gather(coords_reshape, flat_dst)
        # Relative position and distance: (E_total, 3), (E_total, 1)
        rel_pos = coord_i - coord_j
        dist = tf.sqrt(tf.reduce_sum(rel_pos ** 2, axis=-1, keepdims=True) + 1e-8)
        # Edge features and weights: (E_total, 2*d_n+1) -> (E_total, 1)
        coord_input = tf.concat([node_i, node_j, dist], axis=-1)
        coord_weights = tf.nn.silu(tf.matmul(coord_input, self.coord_mlp1))
        coord_weights = tf.matmul(coord_weights, self.coord_mlp2)
        # Weighted direction: (E_total, 3)
        rel_pos_norm = rel_pos / (dist + 1.0)
        weighted_dir = coord_weights * rel_pos_norm
        # Aggregate back to nodes using segment sum
        x_agg_flat = tf.math.unsorted_segment_sum(weighted_dir,flat_src,num_segments=batch_size * n_nodes)
        x_agg = tf.reshape(x_agg_flat, [batch_size, n_nodes, 3])
        # Apply mask and update
        x_agg = x_agg * tf.expand_dims(atom_mask_flat_zero, -1)
        x_new = coords_flat + x_agg
        return x_new

    def call(self, inputs, training=None):
        '''
        Inputs:
        node_features, #(B, N, d_n)
                edge_features, #(B, E, d_e)
                coords_flat, #(B, N, 3)
                atom_mask_flat_zero, #(B, N) pads are 0 else are 1
                edge_mask, #(B,E)  pads are 0 else are 1
                node_adj_matrix, #(B,N,N)
                edge_coadj, #(B,E,E)
                incidence) # (B,E,N)
        '''
        (node_features,             #(B, N, d_n)
        edge_features,              #(B, E, d_e)
        coords_flat,                #(B, N, 3)
        atom_mask_flat_zero,        #(B, N) pads are 0 else are 1
        edge_mask,                  #(B,E)  pads are 0 else are 1
        node_adj_matrix,            #(B,N,N)
        edge_coadj,                 #(B,E,E)
        incidence) = inputs         #(B,E,N)
        # =========== Pass self attention message for nodes ==============
        if training:
            node_out, node_att = recompute_grad_preserve_dtype(self._node_attention)(node_features, node_adj_matrix)
        else:
            node_out, node_att = self._node_attention(node_features, node_adj_matrix)
        node_out += node_features
        node_out = self.node_ln2(node_out)
        node_out *= tf.expand_dims(atom_mask_flat_zero, axis=-1) #mask out pads # (B, N, d_n)

        # =========== Pass self attention message for edges ==============
        if training:
            edge_out, edge_att = recompute_grad_preserve_dtype(self._edge_attention)(edge_features, edge_coadj)
        else:
            edge_out, edge_att = self._edge_attention(edge_features, edge_coadj)
        edge_out += edge_features
        edge_out = self.edge_ln2(edge_out)
        edge_out *= tf.expand_dims(edge_mask, axis=-1) # (B, E, d_e)

        # =========== Pass message node -> edge =============
        node_to_edge = tf.matmul(incidence, node_out) # (B,E,N) @ (B,N,d_n) -> (B,E,d_n)  # BUG FIX: Simplified matmul - directly multiply incidence with node features
        # update edge
        edge_concat = tf.concat([edge_out, node_to_edge], axis=-1) #(B,E,d_n+d_e)
        edge_update = tf.nn.relu(tf.matmul(edge_concat, self.update_w_edge1)) #(B,E,d_n+d_e) * (d_n+d_e, d_h) -> (B,E,d_h)
        edge_update = tf.nn.relu(tf.matmul(edge_update, self.update_w_edge2)) #(B,E,d_h) * (d_h, d_e) -> (B,E,d_e)
        edge_update = self.edge_ln3(edge_update + edge_features)
        edge_update *= tf.expand_dims(edge_mask, axis=-1) # (B, E, d_e)

        # =========== Pass message edge -> node =============
        edge_to_node = tf.matmul(incidence, edge_update, transpose_a=True) # (B,N,E) @ (B,E,d_e) -> (B,N,d_e)  # BUG FIX: Simplified - transpose incidence to (B,N,E) then multiply with edge features
        # update node
        node_concat = tf.concat([node_out, edge_to_node], axis=-1) #(B,N,d_n+d_e)
        node_update = tf.nn.relu(tf.matmul(node_concat, self.update_w_node1)) #(B,N,d_n+d_e) * (d_n+d_e, d_h) -> (B,N,d_h)
        node_update = tf.nn.relu(tf.matmul(node_update, self.update_w_node2)) #(B,N,d_n)
        node_update = self.node_ln3(node_update + node_features)
        node_update *= tf.expand_dims(atom_mask_flat_zero, axis=-1) #(B,N,d_n)

        # =========== Update Coordinates =============
        if training:
            x_new = recompute_grad_preserve_dtype(self._coord_update)(node_update, coords_flat, node_adj_matrix, atom_mask_flat_zero)
        else:
            x_new = self._coord_update(node_update, coords_flat, node_adj_matrix, atom_mask_flat_zero)

        if self.return_attention_map:
            return node_update, edge_update, x_new, node_att, edge_att
        else:
            return node_update, edge_update, x_new

    def get_config(self):
        config = super().get_config()
        config.update({
            'node_dim': self.node_dim,
            'edge_dim': self.edge_dim,
            'hidden_dim': self.hidden_dim,
            'name': self.name_l,
            'heads': self.heads,
            'return_attention_map': self.return_attention_map
        })
        return config

@tf.keras.utils.register_keras_serializable(package='custom_layers', name='EGNNLayer')
class EGNNLayer(layers.Layer):
    """
    E(n) Equivariant Graph Neural Network Layer (BATCHED)

    Uses dense adjacency matrix for batched operations.

    - Node features (h): [B, N, node_dim] invariant scalars
    - Coordinates (x): [B, N, 3] equivariant vectors
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 128,
                 update_coords: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.update_coords = update_coords

    def build(self, input_shape):
        # Message network: 2*node_dim + edge_dim + 1 (distance) -> hidden_dim
        self.message_net = keras.Sequential([
            layers.Dense(self.hidden_dim, activation='silu'),
            layers.Dense(self.hidden_dim, activation='silu')
        ], name='message_net')

        # Node update network
        self.node_net = keras.Sequential([
            layers.Dense(self.hidden_dim, activation='silu'),
            layers.Dense(self.node_dim)
        ], name='node_net')

        # Coordinate update network (outputs scalar)
        if self.update_coords:
            small_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
            self.coord_net = keras.Sequential([
                layers.Dense(self.hidden_dim, activation='silu'),
                layers.Dense(1, use_bias=False, kernel_initializer=small_init)
            ], name='coord_net')

        # Layer normalization
        self.layer_norm = layers.LayerNormalization()

    def call(self, inputs, training=None):
        """
        Args:
            inputs: tuple of (h, x, adj, edge_attr, atom_mask)
                h: [B, N, node_dim] node features
                x: [B, N, 3] coordinates
                adj: [B, N, N] adjacency matrix
                edge_attr: [B, N, N, edge_dim] edge features
                atom_mask: [B, N] valid atom mask
        """
        h, x, adj, edge_attr, atom_mask = inputs
        batch_size = tf.shape(h)[0]
        n_nodes = tf.shape(h)[1]

        # Compute relative positions and distances
        # [B, N, N, 3]
        rel_pos = tf.expand_dims(x, 2) - tf.expand_dims(x, 1)
        dist_sq = tf.reduce_sum(rel_pos ** 2, axis=-1, keepdims=True)  # [B, N, N, 1]

        # Build pairwise node features [B, N, N, 2*node_dim]
        h_i = tf.expand_dims(h, 2)  # [B, N, 1, node_dim]
        h_j = tf.expand_dims(h, 1)  # [B, 1, N, node_dim]
        h_i = tf.tile(h_i, [1, 1, n_nodes, 1])  # [B, N, N, node_dim]
        h_j = tf.tile(h_j, [1, n_nodes, 1, 1])  # [B, N, N, node_dim]

        # Build message input [B, N, N, 2*node_dim + edge_dim + 1]
        msg_input = tf.concat([h_i, h_j, edge_attr, dist_sq], axis=-1)

        # Compute messages [B, N, N, hidden_dim]
        messages = self.message_net(msg_input)

        # Mask messages with adjacency
        adj_expanded = tf.expand_dims(adj, -1)  # [B, N, N, 1]
        messages = messages * adj_expanded

        # Aggregate messages to nodes [B, N, hidden_dim]
        h_agg = tf.reduce_sum(messages, axis=2)

        # Update node features (with residual connection)
        h_update = self.node_net(tf.concat([h, h_agg], axis=-1))
        h_new = self.layer_norm(h + h_update)

        # Update coordinates (equivariant)
        if self.update_coords:
            coord_weights = self.coord_net(messages)  # [B, N, N, 1]

            # Normalize by distance to prevent instability
            dist = tf.sqrt(dist_sq + 1e-8)
            coord_weights = coord_weights / (dist + 1.0)

            # Mask with adjacency
            coord_weights = coord_weights * adj_expanded

            # Weighted relative positions [B, N, N, 3]
            coord_updates = coord_weights * rel_pos

            # Aggregate coordinate updates [B, N, 3]
            x_agg = tf.reduce_sum(coord_updates, axis=2)

            # Apply atom mask
            x_agg = x_agg * tf.expand_dims(atom_mask, -1)
            x_new = x + x_agg
        else:
            x_new = x

        return h_new, x_new

@tf.keras.utils.register_keras_serializable(package='custom_layers', name='ProteinFeatureEncoder')
class ProteinFeatureEncoder(layers.Layer):
    """Encodes protein structure and mutation info into features (BATCHED)"""

    def __init__(self, node_dim: int = 128, edge_dim: int = 64, cut_off= 7.5, PAD_TOKEN=-2., **kwargs):
        super().__init__(**kwargs)
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.cut_off = cut_off
        self.PAD_TOKEN = PAD_TOKEN

    def build(self, input_shape):
        # Amino acid embedding
        self.aa_embedding = layers.Embedding(NUM_AA, 32)

        # Atom type embedding
        self.atom_embedding = layers.Embedding(NUM_ATOM_TYPES, 16)

        # Mutation encoder
        self.mutation_encoder = keras.Sequential([
            layers.Dense(64, activation='silu'),
            layers.Dense(32)
        ], name='mutation_encoder')

        # Positional encoding
        self.pos_embedding = layers.Embedding(10, 16) #(max number of residues is around 100, peptide + mhc_pseudoseq)

        # Combined node encoder
        self.node_encoder = keras.Sequential([
            layers.Dense(self.node_dim, activation='silu'),
            layers.Dense(self.node_dim)
        ], name='feature_node_encoder')

        # Edge encoder
        self.rbf_layer = RBFLayer(num_rbf=32)
        self.edge_encoder = keras.Sequential([
            layers.Dense(self.edge_dim, activation='silu'),
            layers.Dense(self.edge_dim)
        ], name='feature_edge_encoder')

        # AA properties table
        aa_props_list = [AA_PROPERTIES[AMINO_ACIDS[i]] for i in range(NUM_AA)] #30 properties
        self.aa_props_table = tf.constant(aa_props_list, dtype=tf.float32)

    def call(self, inputs, training=None):
        """
        Args:
            inputs: dict with keys (all BATCHED):
                - coords: [B, N_res, N_atoms, 3]
                - wt_indices: [B, N_res] wild-type AA indices
                - mut_indices: [B, N_res] mutant AA indices
                - mutation_mask: [B, N_res] 1.0 for mutated positions
                - atom_mask: [B, N_res, N_atoms]
                - res_indices: [B, N_res * N_atoms]

        Returns:
            node_features: [B, N_res * N_atoms, node_dim]
            edge_features: [B, N_res * N_atoms, N_res * N_atoms, edge_dim]
            coords_flat: [B, N_res * N_atoms, 3]
            atom_mask_flat: [B, N_res * N_atoms]
        """
        compute_dtype = self.compute_dtype
        coords = inputs['coords']
        wt_indices = inputs['wt_indices']
        mut_indices = inputs['mut_indices']
        mutation_mask = inputs['mutation_mask']
        atom_mask = inputs['atom_mask']
        res_indices = inputs['res_indices']

        batch_size = tf.shape(coords)[0]
        n_res = tf.shape(coords)[1]
        n_atoms = NUM_ATOM_TYPES

        # Expand residue-level features to atom level
        # [B, N_res] -> [B, N_res * N_atoms]
        wt_idx_atoms = tf.repeat(wt_indices, n_atoms, axis=1)
        mut_idx_atoms = tf.repeat(mut_indices, n_atoms, axis=1)
        mutation_mask_atoms = tf.repeat(mutation_mask, n_atoms, axis=1)

        # Embeddings [B, N_res * N_atoms, dim]
        wt_embed = self.aa_embedding(wt_idx_atoms)
        mut_embed = self.aa_embedding(mut_idx_atoms)

        # Atom type embedding [B, N_res * N_atoms, 16]
        atom_types_single = tf.range(n_atoms)  # [N_atoms]
        atom_types_res = tf.tile(atom_types_single, [n_res])  # [N_res * N_atoms]
        atom_types = tf.tile(
            tf.expand_dims(atom_types_res, 0),
            [batch_size, 1]
        )  # [B, N_res * N_atoms]
        atom_embed = self.atom_embedding(atom_types)

        # Positional encoding [B, N_res * N_atoms, 16]
        pos_embed = self.pos_embedding(res_indices)

        # AA properties [B, N_res * N_atoms, 4]
        aa_props = tf.gather(self.aa_props_table, mut_idx_atoms)
        aa_props_mean = tf.reduce_mean(aa_props, axis=1, keepdims=True)
        aa_props_std = tf.math.reduce_std(aa_props, axis=1, keepdims=True) + 1e-6
        aa_props = (aa_props - aa_props_mean) / aa_props_std

        # Mutation features
        mutation_input = tf.concat([
            tf.expand_dims(mutation_mask_atoms, -1),
            wt_embed,
            mut_embed
        ], axis=-1)
        mutation_features = self.mutation_encoder(mutation_input)

        # Combine all node features
        node_input = tf.concat([
            tf.cast(mut_embed, compute_dtype),
            tf.cast(atom_embed, compute_dtype),
            tf.cast(aa_props, compute_dtype),
            tf.cast(mutation_features, compute_dtype),
            tf.cast(pos_embed, compute_dtype)
        ], axis=-1)
        node_features = self.node_encoder(node_input) #(B, N*atom, D)

        # Flatten coordinates and mask [B, N_res * N_atoms, 3]
        coords_flat = tf.reshape(coords, [batch_size, -1, 3])
        atom_mask_flat = tf.reshape(atom_mask, [batch_size, -1])
        #atom_mask_flat_zero = tf.where(atom_mask_flat == self.PAD_TOKEN, 0., 1) #(B,  N_res * N_atoms)
        atom_mask_flat_zero = tf.expand_dims(atom_mask_flat, -1) #(B, N_res * N_atoms, 1)
        node_features = node_features * tf.cast(atom_mask_flat_zero, compute_dtype) #(B, N_res*N_atoms, D) * (B, N_res * N_atoms, 1)
        # ================================================================
        # Compute edge features using RBF and edge encoder
        # ================================================================
        # Compute pairwise distances [B, N_res * N_atoms, N_res * N_atoms] -> (B,E,1)
        # coords_flat: [B, N_res * N_atoms, 3]
        node_adj_matrix = build_batched_node_adjacency(coords_flat, atom_mask_flat, self.cut_off) #[B, N_res * N_atoms, N_res * N_atoms]
        incidence, edge_coadj, edge_dist, edge_mask = build_batched_edge_matrices(node_adj_matrix, coords_flat) #(B,E,N), (B,E,E) , (B,E,1), (B,E) 1 for real and 0 for padded edges
        cov_edges_vs_non_cov = tf.where(edge_dist < 2., 1., 0.) #(B,E,1)
        rbf_features = self.rbf_layer(tf.squeeze(edge_dist, axis=-1)) #(B,E,32)
        edge_features = tf.concat([tf.cast(cov_edges_vs_non_cov, compute_dtype), rbf_features], axis=-1) #(B,E,33)
        edge_features = self.edge_encoder(edge_features) #(B,E,edge_dim)
        edge_features = edge_features * tf.expand_dims(tf.cast(edge_mask, compute_dtype), -1) # mask out
        # ================================================================

        return (node_features, #(B, N, d_n)
                edge_features, #(B, E, d_e)
                coords_flat, #(B, N, 3)
                tf.squeeze(atom_mask_flat_zero, axis=-1), #(B, N) pads are 0 else are 1
                edge_mask, #(B,E)  pads are 0 else are 1
                node_adj_matrix, #(B,N,N)
                edge_coadj, #(B,E,E)
                incidence) # (B,E,N)

# ============================================================================
# MAIN MODEL (BATCHED)
# ============================================================================

class MutationStructurePredictorAtt(Model):
    def __init__(self,
                 node_dim: int = 128, edge_dim: int = 64, hidden_dim: int = 128, heads: int = 4,
                 num_layers: int = 8, cutoff: float = 10.0, PAD_TOKEN=-2.,  **kwargs):
        super().__init__(**kwargs)
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.cutoff = cutoff
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.PAD_TOKEN = PAD_TOKEN
        # ==== Define Model ====
        self.encoder = ProteinFeatureEncoder(node_dim, edge_dim, self.cutoff, PAD_TOKEN=self.PAD_TOKEN)
        self.egnn_layers = [
            AttEGNNLayer(self.node_dim, self.edge_dim, self.hidden_dim,
                     f'AttEGNN_{i}', self.heads,
                      True)
            for i in range(num_layers)
        ]

    def call(self, inputs, training=None):
        coords = inputs['coords']
        atom_mask = inputs['atom_mask']
        batch_size = tf.shape(coords)[0]
        n_res = tf.shape(coords)[1]
        n_atoms = NUM_ATOM_TYPES

        # ================= encode features ===================
        (node_features,  # (B, N, d_n)
         edge_features,  # (B, E, d_e)
         coords_flat,  # (B, N, 3)
         atom_mask_flat_zero,  # (B, N) pads are 0 else are 1
         edge_mask,  # (B,E)  pads are 0 else are 1
         node_adj_matrix,  # (B,N,N)
         edge_coadj,  # (B,E,E)
         incidence  # (B,E,N)
         ) = self.encoder(inputs, training=training)
        #tf.print(inputs['wt_indices'])
        # ================= EGNN Layers ===================
        for egnn_layer in self.egnn_layers:
            node_features, edge_features, coords_flat, node_att, edge_att = egnn_layer((node_features,  # (B, N, d_n)
                                                     edge_features,  # (B, E, d_e)
                                                     coords_flat,  # (B, N, 3)
                                                     atom_mask_flat_zero,  # (B, N) pads are 0 else are 1
                                                     edge_mask,  # (B,E)  pads are 0 else are 1
                                                     node_adj_matrix,  # (B,N,N)
                                                     edge_coadj,  # (B,E,E)
                                                     incidence  # (B,E,N)
                                                     ), training=training)
            
        # Reshape to [B, N_res, N_atoms, 3]
        predicted_coords = tf.reshape(coords_flat, [batch_size, n_res, n_atoms, 3])
        return predicted_coords, node_att, edge_att


    def predict_structure(self, structure: ProteinStructure,
                          mutations: List[Mutation]) -> tf.Tensor:
        """Convenience method for inference (single sample)"""
        inputs = self._prepare_inputs(structure, mutations)
        # Add batch dimension
        inputs = {k: tf.expand_dims(v, 0) for k, v in inputs.items()}
        result = self(inputs, training=False)
        return result[0]  # Remove batch dimension

    def _prepare_inputs(self, structure: ProteinStructure,
                        mutations: List[Mutation]) -> Dict[str, tf.Tensor]:
        """Prepare model inputs from structure and mutations (single sample)"""
        n_res = structure.n_residues
        wt_indices = tf.constant(
            [AA_TO_IDX[aa] for aa in structure.sequence],
            dtype=tf.int32
        )
        mut_indices_np = np.array([AA_TO_IDX[aa] for aa in structure.sequence])
        mutation_mask_np = np.zeros(n_res, dtype=np.float32)
        for mut in mutations:
            mut_indices_np[mut.position] = AA_TO_IDX[mut.mutant]
            mutation_mask_np[mut.position] = 1.0
        return {
            'coords': structure.coords,
            'wt_indices': wt_indices,
            'mut_indices': tf.constant(mut_indices_np, dtype=tf.int32),
            'mutation_mask': tf.constant(mutation_mask_np, dtype=tf.float32),
            'atom_mask': structure.atom_mask
        }


class MutationStructurePredictor(Model):
    """
    SE(3)-Equivariant model for predicting mutant protein structures (BATCHED)
    """

    def __init__(self, node_dim: int = 128, edge_dim: int = 64,
                 num_layers: int = 8, cutoff: float = 10.0, K=3, **kwargs):
        super().__init__(**kwargs)
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.cutoff = cutoff
        self.K = K # after K layers, update adj matrix

        # Feature encoder
        self.encoder = ProteinFeatureEncoder(node_dim, edge_dim)

        # EGNN layers
        self.egnn_layers = [
            EGNNLayer(node_dim, edge_dim, update_coords=True, name=f'egnn_{i}')
            for i in range(num_layers)
        ]

        # RBF layers for edge feature updates (for layers after the first)
        self.rbf_layers = [
            RBFLayer(num_rbf=32, name=f'rbf_{i}')
            for i in range(num_layers - 1)
        ]

        # Edge encoders for updating edge features after coordinate updates
        self.edge_encoders = [
            keras.Sequential([
                layers.Dense(edge_dim, activation='silu'),
                layers.Dense(edge_dim)
            ], name=f'edge_encoder_{i}')
            for i in range(num_layers - 1)
        ]

        # Final coordinate head
        small_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        self.coord_head = keras.Sequential([
            layers.Dense(64, activation='silu'),
            layers.Dense(3, kernel_initializer=small_init, bias_initializer='zeros')
        ], name='coord_head')

    def call(self, inputs, training=None):
        """
        Args:
            inputs: dict containing (all BATCHED):
                - coords: [B, N_res, N_atoms, 3]
                - wt_indices: [B, N_res]
                - mut_indices: [B, N_res]
                - mutation_mask: [B, N_res]
                - atom_mask: [B, N_res, N_atoms]

        Returns:
            predicted_coords: [B, N_res, N_atoms, 3]
        """
        coords = inputs['coords']
        atom_mask = inputs['atom_mask']

        batch_size = tf.shape(coords)[0]
        n_res = tf.shape(coords)[1]
        n_atoms = NUM_ATOM_TYPES

        # Encode features
        # h: [B, N, node_dim]
        # edge_attr: [B, N, N, edge_dim]
        # x: [B, N, 3]
        # mask: [B, N]
        h, edge_attr, x, mask = self.encoder(inputs, training=training)

        # Build initial adjacency [B, N, N]
        adj, dist = build_radius_graph(x, mask, self.cutoff)

        # Message passing
        for i, egnn_layer in enumerate(self.egnn_layers):
            h, x = egnn_layer(
                (h, x, adj, edge_attr, mask),
                training=training
            )

            # Update graph and edge features with new coordinates
            adj , dist = build_radius_graph(x, mask, self.cutoff)

            # Update edge features for next layer
            if i < len(self.egnn_layers) - 1:
                rbf_features = self.rbf_layers[i](dist)  # [B, N, N, 32]
                edge_attr = self.edge_encoders[i](rbf_features)  # [B, N, N, edge_dim]

        # Final coordinate refinement
        coord_delta = self.coord_head(h)
        coord_delta = coord_delta * tf.expand_dims(mask, -1)
        x_final = x + coord_delta

        # Reshape to [B, N_res, N_atoms, 3]
        predicted_coords = tf.reshape(x_final, [batch_size, n_res, n_atoms, 3])

        return predicted_coords

    def predict_structure(self, structure: ProteinStructure,
                          mutations: List[Mutation]) -> tf.Tensor:
        """Convenience method for inference (single sample)"""
        inputs = self._prepare_inputs(structure, mutations)
        # Add batch dimension
        inputs = {k: tf.expand_dims(v, 0) for k, v in inputs.items()}
        result = self(inputs, training=False)
        return result[0]  # Remove batch dimension

    def _prepare_inputs(self, structure: ProteinStructure,
                        mutations: List[Mutation]) -> Dict[str, tf.Tensor]:
        """Prepare model inputs from structure and mutations (single sample)"""
        n_res = structure.n_residues

        wt_indices = tf.constant(
            [AA_TO_IDX[aa] for aa in structure.sequence],
            dtype=tf.int32
        )

        mut_indices_np = np.array([AA_TO_IDX[aa] for aa in structure.sequence])
        mutation_mask_np = np.zeros(n_res, dtype=np.float32)

        for mut in mutations:
            mut_indices_np[mut.position] = AA_TO_IDX[mut.mutant]
            mutation_mask_np[mut.position] = 1.0

        return {
            'coords': structure.coords,
            'wt_indices': wt_indices,
            'mut_indices': tf.constant(mut_indices_np, dtype=tf.int32),
            'mutation_mask': tf.constant(mutation_mask_np, dtype=tf.float32),
            'atom_mask': structure.atom_mask
        }


# ============================================================================
# ALIGNMENT (BATCHED)
# ============================================================================

def align_structures(wt_coords: tf.Tensor, mut_coords: tf.Tensor,
                     atom_mask: tf.Tensor = None) -> tf.Tensor:
    """
    Align mutant structure to wild-type using Kabsch algorithm (BATCHED)

    Args:
        wt_coords: [B, N_res, N_atoms, 3] or [N_res, N_atoms, 3] - reference
        mut_coords: [B, N_res, N_atoms, 3] or [N_res, N_atoms, 3] - to align
        atom_mask: [B, N_res, N_atoms] or [N_res, N_atoms] - valid atom mask

    Returns:
        aligned_mut_coords: same shape as mut_coords - aligned to wt_coords
    """
    # Handle non-batched input
    input_was_unbatched = len(wt_coords.shape) == 3
    if input_was_unbatched:
        wt_coords = tf.expand_dims(wt_coords, 0)
        mut_coords = tf.expand_dims(mut_coords, 0)
        if atom_mask is not None:
            atom_mask = tf.expand_dims(atom_mask, 0)

    batch_size = tf.shape(wt_coords)[0]
    n_res = tf.shape(wt_coords)[1]
    n_atoms = tf.shape(wt_coords)[2]

    # Use CA atoms for alignment
    wt_ca = wt_coords[:, :, ATOM_CA, :]  # [B, N_res, 3]
    mut_ca = mut_coords[:, :, ATOM_CA, :]

    if atom_mask is not None:
        mask = atom_mask[:, :, ATOM_CA]  # [B, N_res]
    else:
        mask = tf.ones([batch_size, n_res], dtype=wt_ca.dtype)

    mask_sum = tf.reduce_sum(mask, axis=1, keepdims=True) + 1e-8  # [B, 1]
    mask_col = tf.expand_dims(mask, -1)  # [B, N_res, 1]

    # Compute centroids [B, 3]
    wt_center = tf.reduce_sum(wt_ca * mask_col, axis=1) / mask_sum
    mut_center = tf.reduce_sum(mut_ca * mask_col, axis=1) / mask_sum

    # Center the structures [B, N_res, 3]
    wt_centered = (wt_ca - tf.expand_dims(wt_center, 1)) * mask_col
    mut_centered = (mut_ca - tf.expand_dims(mut_center, 1)) * mask_col

    # Compute optimal rotation (Kabsch algorithm)
    # H = mut^T @ wt -> [B, 3, 3]
    H = tf.matmul(mut_centered, wt_centered, transpose_a=True)

    # SVD
    s, U, V = tf.linalg.svd(H)

    # Handle reflection case
    det = tf.linalg.det(tf.matmul(V, U, transpose_b=True))  # [B]
    sign_matrix = tf.stack([
        tf.ones([batch_size]),
        tf.ones([batch_size]),
        det
    ], axis=1)  # [B, 3]
    sign_matrix = tf.linalg.diag(sign_matrix)  # [B, 3, 3]

    # Optimal rotation matrix [B, 3, 3]
    R = tf.matmul(tf.matmul(V, sign_matrix), U, transpose_b=True)

    # Apply transformation to ALL atoms
    # Flatten [B, N_res * N_atoms, 3]
    mut_flat = tf.reshape(mut_coords, [batch_size, -1, 3])

    # Center, rotate, translate
    mut_centered_flat = mut_flat - tf.expand_dims(mut_center, 1)
    aligned_flat = tf.matmul(mut_centered_flat, R, transpose_b=True)
    aligned_flat = aligned_flat + tf.expand_dims(wt_center, 1)

    # Reshape back [B, N_res, N_atoms, 3]
    aligned_mut_coords = tf.reshape(aligned_flat, [batch_size, n_res, n_atoms, 3])

    # Remove batch dimension if input was unbatched
    if input_was_unbatched:
        aligned_mut_coords = aligned_mut_coords[0]

    return aligned_mut_coords


# ============================================================================
# LOSSES (BATCHED)
# ============================================================================

class FAPELoss(keras.layers.Layer):
    """Frame Aligned Point Error from AlphaFold2 (BATCHED)"""

    def __init__(self, clamp_distance: float = 10.0, epsilon: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.clamp_distance = clamp_distance
        self.epsilon = epsilon

    def build_frames(self, coords: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Build local coordinate frames from backbone atoms (BATCHED)

        Args:
            coords: [B, N_res, N_atoms, 3]

        Returns:
            rotations: [B, N_res, 3, 3]
            translations: [B, N_res, 3]
        """
        N = coords[:, :, ATOM_N, :]   # [B, N_res, 3]
        CA = coords[:, :, ATOM_CA, :]
        C = coords[:, :, ATOM_C, :]

        v1 = C - CA
        v2 = N - CA

        e1 = safe_normalize(v1, axis=-1)

        dot = tf.reduce_sum(v2 * e1, axis=-1, keepdims=True)
        v2_orth = v2 - dot * e1
        e2 = safe_normalize(v2_orth, axis=-1)

        e3 = tf.linalg.cross(e1, e2)

        rotations = tf.stack([e1, e2, e3], axis=-1)  # [B, N_res, 3, 3]
        translations = CA

        return rotations, translations

    def call(self, pred_coords: tf.Tensor, true_coords: tf.Tensor,
             atom_mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Compute FAPE loss (BATCHED)

        Args:
            pred_coords: [B, N_res, N_atoms, 3]
            true_coords: [B, N_res, N_atoms, 3]
            atom_mask: [B, N_res, N_atoms]
        """
        batch_size = tf.shape(pred_coords)[0]
        n_res = tf.shape(pred_coords)[1]
        n_atoms = tf.shape(pred_coords)[2]

        if atom_mask is None:
            atom_mask = tf.ones([batch_size, n_res, n_atoms], dtype=pred_coords.dtype)

        # Build frames
        pred_rot, pred_trans = self.build_frames(pred_coords)
        true_rot, true_trans = self.build_frames(true_coords)

        # CA atoms and mask
        pred_ca = pred_coords[:, :, ATOM_CA, :]  # [B, N_res, 3]
        true_ca = true_coords[:, :, ATOM_CA, :]
        ca_mask = atom_mask[:, :, ATOM_CA]  # [B, N_res]

        # Center using pred/true frame origins
        # pred_ca: [B, N_res, 3] -> [B, N_frames, N_atoms, 3]
        pred_centered = tf.expand_dims(pred_ca, 1) - tf.expand_dims(pred_trans, 2)
        true_centered = tf.expand_dims(true_ca, 1) - tf.expand_dims(true_trans, 2)

        # Transform to local coordinates
        # [B, N_frames, N_atoms, 3] x [B, N_frames, 3, 3] -> [B, N_frames, N_atoms, 3]
        pred_local = tf.einsum('bfaj,bfjk->bfak', pred_centered, pred_rot)
        true_local = tf.einsum('bfaj,bfjk->bfak', true_centered, true_rot)

        # Compute errors
        diff = tf.cast(pred_local, self.compute_dtype) - tf.cast(true_local, self.compute_dtype)
        error = safe_norm(diff, axis=-1)  # [B, N_frames, N_atoms]

        # Clamp
        clamped = tf.minimum(error, self.clamp_distance)
        clamped = tf.cast(clamped, self.compute_dtype)
        # Mask: [B, N_frames] x [B, N_atoms] -> [B, N_frames, N_atoms]
        pair_mask = tf.expand_dims(ca_mask, 2) * tf.expand_dims(ca_mask, 1)
        pair_mask = tf.cast(pair_mask, self.compute_dtype)
        # Mean FAPE per sample, then mean over batch
        fape_per_sample = tf.reduce_sum(clamped * pair_mask, axis=[1, 2]) / (
            tf.reduce_sum(pair_mask, axis=[1, 2]) + self.epsilon
        )

        return tf.reduce_mean(fape_per_sample)


class RMSDLoss(keras.layers.Layer):
    """RMSD loss with differentiable Kabsch alignment (BATCHED)"""

    def __init__(self, epsilon: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = tf.cast(epsilon, self.compute_dtype)

    def kabsch_align(self, pred: tf.Tensor, true: tf.Tensor,
                     mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Align pred to true using Kabsch algorithm (BATCHED)

        Args:
            pred: [B, N, 3]
            true: [B, N, 3]
            mask: [B, N]
        """
        # this does not support half percision and should be float32
        pred = tf.cast(pred, tf.float32)
        true = tf.cast(true, tf.float32)
        mask = tf.cast(mask, tf.float32)
        batch_size = tf.shape(pred)[0]
        n_points = tf.shape(pred)[1]

        if mask is None:
            mask = tf.ones([batch_size, n_points], dtype=pred.dtype)

        mask_sum = tf.reduce_sum(mask, axis=1, keepdims=True) + tf.cast(self.epsilon, tf.float32)
        mask_col = tf.expand_dims(mask, -1)

        # Centroids [B, 3]
        pred_center = tf.reduce_sum(pred * mask_col, axis=1) / mask_sum
        true_center = tf.reduce_sum(true * mask_col, axis=1) / mask_sum

        # Center
        pred_centered = (pred - tf.expand_dims(pred_center, 1)) * mask_col
        true_centered = (true - tf.expand_dims(true_center, 1)) * mask_col

        # Covariance [B, 3, 3]
        H = tf.matmul(pred_centered, true_centered, transpose_a=True)

        # SVD
        s, U, V = tf.linalg.svd(H)

        # Handle reflection
        VU_t = tf.matmul(V, U, transpose_b=True)
        det = tf.linalg.det(VU_t)

        sign_matrix = tf.stack([
            tf.ones([batch_size]),
            tf.ones([batch_size]),
            det
        ], axis=1)
        sign_matrix = tf.linalg.diag(sign_matrix)

        R = tf.matmul(tf.matmul(V, sign_matrix), U, transpose_b=True)

        # Apply
        aligned = tf.matmul(pred - tf.expand_dims(pred_center, 1), R, transpose_b=True)
        aligned = aligned + tf.expand_dims(true_center, 1)

        return tf.cast(aligned, self.compute_dtype)

    def call(self, pred_coords: tf.Tensor, true_coords: tf.Tensor,
             atom_mask: Optional[tf.Tensor] = None, align: bool = True) -> tf.Tensor:
        """
        Compute RMSD loss (BATCHED)

        Args:
            pred_coords: [B, N_res, N_atoms, 3]
            true_coords: [B, N_res, N_atoms, 3]
            atom_mask: [B, N_res, N_atoms]
        """
        pred_coords = tf.cast(pred_coords, self.compute_dtype)
        true_coords = tf.cast(true_coords, self.compute_dtype)
        atom_mask = tf.cast(atom_mask, self.compute_dtype)
        pred_ca = pred_coords[:, :, ATOM_CA, :]  # [B, N_res, 3]
        true_ca = true_coords[:, :, ATOM_CA, :]

        if atom_mask is not None:
            ca_mask = atom_mask[:, :, ATOM_CA]
        else:
            batch_size = tf.shape(pred_ca)[0]
            n_res = tf.shape(pred_ca)[1]
            ca_mask = tf.ones([batch_size, n_res], dtype=pred_ca.dtype)

        if align:
            pred_ca = self.kabsch_align(pred_ca, true_ca, ca_mask)

        # RMSD per sample
        diff_sq = tf.reduce_sum((pred_ca - true_ca) ** 2, axis=-1)  # [B, N_res]
        rmsd_per_sample = tf.sqrt(
            tf.reduce_sum(diff_sq * ca_mask, axis=1) /
            (tf.reduce_sum(ca_mask, axis=1) + self.epsilon)
        )

        return tf.reduce_mean(rmsd_per_sample)


class TorsionAngleLoss(keras.layers.Layer):
    """Loss on backbone torsion angles (BATCHED)"""

    def __init__(self, omega_weight: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.omega_weight = omega_weight

    def compute_dihedral(self, p0: tf.Tensor, p1: tf.Tensor,
                         p2: tf.Tensor, p3: tf.Tensor) -> tf.Tensor:
        """
        Compute dihedral angle (BATCHED)

        Args:
            p0, p1, p2, p3: [B, N, 3]
        """
        b1 = p1 - p0
        b2 = p2 - p1
        b3 = p3 - p2

        n1 = tf.linalg.cross(b1, b2)
        n2 = tf.linalg.cross(b2, b3)

        n1 = safe_normalize(n1, axis=-1)
        n2 = safe_normalize(n2, axis=-1)
        b2_hat = safe_normalize(b2, axis=-1)

        m1 = tf.linalg.cross(n1, b2_hat)

        x = tf.reduce_sum(n1 * n2, axis=-1)
        y = tf.reduce_sum(m1 * n2, axis=-1)

        return tf.atan2(y, x)

    def compute_backbone_torsions(self, coords: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        Compute phi, psi, omega angles (BATCHED)

        Args:
            coords: [B, N_res, N_atoms, 3]
        """
        N = coords[:, :, ATOM_N, :]   # [B, N_res, 3]
        CA = coords[:, :, ATOM_CA, :]
        C = coords[:, :, ATOM_C, :]

        # Phi: C(i-1) - N(i) - CA(i) - C(i)
        phi = self.compute_dihedral(C[:, :-1], N[:, 1:], CA[:, 1:], C[:, 1:])

        # Psi: N(i) - CA(i) - C(i) - N(i+1)
        psi = self.compute_dihedral(N[:, :-1], CA[:, :-1], C[:, :-1], N[:, 1:])

        # Omega: CA(i) - C(i) - N(i+1) - CA(i+1)
        omega = self.compute_dihedral(CA[:, :-1], C[:, :-1], N[:, 1:], CA[:, 1:])

        return {'phi': phi, 'psi': psi, 'omega': omega}

    def periodic_loss(self, pred: tf.Tensor, true: tf.Tensor) -> tf.Tensor:
        """Periodic angular loss (BATCHED): 1 - cos(diff)"""
        diff = pred - true
        return tf.reduce_mean(1.0 - tf.cos(diff))

    def call(self, pred_coords: tf.Tensor, true_coords: tf.Tensor,
             atom_mask: Optional[tf.Tensor] = None) -> Dict[str, tf.Tensor]:
        """
        Compute torsion losses (BATCHED)

        Args:
            pred_coords: [B, N_res, N_atoms, 3]
            true_coords: [B, N_res, N_atoms, 3]
        """
        pred_coords = tf.cast(pred_coords, self.compute_dtype)
        true_coords = tf.cast(true_coords, self.compute_dtype)
        ca_mask = tf.cast(atom_mask[:, :, ATOM_CA], tf.float32)
        pred_torsions = self.compute_backbone_torsions(pred_coords)
        true_torsions = self.compute_backbone_torsions(true_coords)

        phi_loss = self.periodic_loss(pred_torsions['phi'], true_torsions['phi'])
        psi_loss = self.periodic_loss(pred_torsions['psi'], true_torsions['psi'])
        omega_loss = self.periodic_loss(pred_torsions['omega'], true_torsions['omega'])

        total = phi_loss + psi_loss + self.omega_weight * omega_loss

        return {
            'phi': phi_loss,
            'psi': psi_loss,
            'omega': omega_loss,
            'total': total
        }


class StericClashLoss(keras.layers.Layer):
    """Penalizes steric clashes (BATCHED)"""

    def __init__(self, clash_tolerance: float = 0.4,
                 exclude_neighbors: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.clash_tolerance = clash_tolerance
        self.exclude_neighbors = exclude_neighbors
        self.vdw_radii_list = [VDW_RADII[key] for key in VDW_RADII.keys()]
        #self.vdw_radii_list = [
        #    VDW_RADII['N'], VDW_RADII['CA'], VDW_RADII['C'],
        #    VDW_RADII['O'], VDW_RADII['CB']
        #]

    def call(self, pred_coords: tf.Tensor,
             atom_mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Compute steric clash loss (BATCHED)

        Args:
            pred_coords: [B, N_res, N_atoms, 3]
            atom_mask: [B, N_res, N_atoms]
        """
        pred_coords = tf.cast(pred_coords, self.compute_dtype)
        atom_mask = tf.cast(atom_mask, self.compute_dtype)
        batch_size = tf.shape(pred_coords)[0]
        n_res = tf.shape(pred_coords)[1]
        n_atoms = tf.shape(pred_coords)[2]
        n_total = n_res * n_atoms

        if atom_mask is None:
            atom_mask = tf.ones([batch_size, n_res, n_atoms], dtype=pred_coords.dtype)

        # Flatten [B, N_total, 3]
        coords_flat = tf.reshape(pred_coords, [batch_size, n_total, 3])
        mask_flat = tf.reshape(atom_mask, [batch_size, n_total])

        # Pairwise distances [B, N, N]
        diff = tf.expand_dims(coords_flat, 2) - tf.expand_dims(coords_flat, 1)
        distances = safe_norm(diff, axis=-1)

        # VDW radii
        vdw_tensor = tf.constant(self.vdw_radii_list, dtype=pred_coords.dtype)
        radii_per_atom = tf.tile(vdw_tensor, [n_res])  # [N_total]
        min_dist = radii_per_atom[:, None] + radii_per_atom[None, :] - self.clash_tolerance

        # Residue indices
        res_idx = tf.repeat(tf.range(n_res, dtype=tf.int32), n_atoms)
        res_diff = tf.abs(res_idx[:, None] - res_idx[None, :])
        neighbor_mask = tf.cast(res_diff > self.exclude_neighbors, pred_coords.dtype)

        # Full mask [B, N, N]
        pair_mask = tf.expand_dims(mask_flat, 2) * tf.expand_dims(mask_flat, 1)
        pair_mask = pair_mask * neighbor_mask[None, :, :]
        pair_mask = pair_mask * (1.0 - tf.eye(n_total, dtype=pred_coords.dtype)[None, :, :])

        # Clash penalty
        clash = tf.maximum(min_dist[None, :, :] - distances, 0.0) ** 2

        # Mean per sample, then mean over batch
        total_clash = tf.reduce_sum(clash * pair_mask, axis=[1, 2])
        n_pairs = tf.reduce_sum(pair_mask, axis=[1, 2]) + 1e-8
        clash_per_sample = total_clash / n_pairs

        return tf.reduce_mean(clash_per_sample)


class LDDTLoss(keras.layers.Layer):
    """Differentiable LDDT loss (BATCHED)"""

    def __init__(self, cutoff: float = 15.0, thresholds: list = None, **kwargs):
        super().__init__(**kwargs)
        self.cutoff = cutoff
        self.thresholds = thresholds or LDDT_THRESHOLDS

    def call(self, pred_coords: tf.Tensor, true_coords: tf.Tensor,
             atom_mask: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute LDDT loss (BATCHED)

        Args:
            pred_coords: [B, N_res, N_atoms, 3]
            true_coords: [B, N_res, N_atoms, 3]
            atom_mask: [B, N_res, N_atoms]
        """
        pred_coords = tf.cast(pred_coords, self.compute_dtype)
        true_coords = tf.cast(true_coords, self.compute_dtype)
        atom_mask = tf.cast(atom_mask, self.compute_dtype)
        batch_size = tf.shape(pred_coords)[0]
        n_res = tf.shape(pred_coords)[1]

        pred_ca = pred_coords[:, :, ATOM_CA, :]  # [B, N_res, 3]
        true_ca = true_coords[:, :, ATOM_CA, :]

        if atom_mask is not None:
            ca_mask = atom_mask[:, :, ATOM_CA]
        else:
            ca_mask = tf.ones([batch_size, n_res], dtype=pred_ca.dtype)

        # Pairwise distances [B, N_res, N_res]
        pred_diff = tf.expand_dims(pred_ca, 2) - tf.expand_dims(pred_ca, 1)
        true_diff = tf.expand_dims(true_ca, 2) - tf.expand_dims(true_ca, 1)

        pred_dist = safe_norm(pred_diff, axis=-1)
        true_dist = safe_norm(true_diff, axis=-1)

        dist_diff = tf.abs(pred_dist - true_dist)

        # Masks
        cutoff_mask = tf.cast(true_dist < self.cutoff, pred_ca.dtype)
        diag_mask = 1.0 - tf.eye(n_res, batch_shape=[batch_size], dtype=pred_ca.dtype)
        pair_mask = cutoff_mask * diag_mask
        pair_mask = pair_mask * tf.expand_dims(ca_mask, 2) * tf.expand_dims(ca_mask, 1)

        # LDDT per threshold
        lddt_scores = []
        steepness = 5.0
        for thresh in self.thresholds:
            within = tf.sigmoid(steepness * (thresh - dist_diff))
            score = tf.reduce_sum(within * pair_mask, axis=[1, 2]) / (
                tf.reduce_sum(pair_mask, axis=[1, 2]) + 1e-8
            )
            lddt_scores.append(score)

        lddt = tf.reduce_mean(tf.stack(lddt_scores, axis=0), axis=0)  # [B]
        loss = 1.0 - tf.reduce_mean(lddt)
        lddt_score = tf.reduce_mean(lddt)

        return loss, lddt_score


class StructuralLoss(keras.layers.Layer):
    """Combined structural loss (BATCHED)"""

    def __init__(self,
                 fape_weight: float = 1.0,
                 rmsd_weight: float = 1.5,
                 torsion_weight: float = 0.3,
                 clash_weight: float = 0.5,
                 lddt_weight: float = 0.5,
                 fape_clamp: float = 10.0,
                 lddt_cutoff: float = 15.0,
                 **kwargs):
        super().__init__(**kwargs)

        self.loss_weights = {
            'fape': fape_weight,
            'rmsd': rmsd_weight,
            'torsion': torsion_weight,
            'clash': clash_weight,
            'lddt': lddt_weight,
        }

        self.fape_loss = FAPELoss(clamp_distance=fape_clamp)
        self.rmsd_loss = RMSDLoss()
        self.torsion_loss = TorsionAngleLoss()
        self.clash_loss = StericClashLoss()
        self.lddt_loss = LDDTLoss(cutoff=lddt_cutoff)

    def call(self, pred_coords: tf.Tensor, true_coords: tf.Tensor,
             atom_mask: Optional[tf.Tensor] = None) -> Dict[str, tf.Tensor]:
        """
        Compute all losses (BATCHED)

        Args:
            pred_coords: [B, N_res, N_atoms, 3]
            true_coords: [B, N_res, N_atoms, 3]
            atom_mask: [B, N_res, N_atoms]
        """
        losses = {}

        losses['fape'] = self.fape_loss(pred_coords, true_coords, atom_mask)
        losses['rmsd'] = self.rmsd_loss(pred_coords, true_coords, atom_mask)

        torsion = self.torsion_loss(pred_coords, true_coords, atom_mask)
        losses['torsion'] = torsion['total']
        losses['torsion_phi'] = torsion['phi']
        losses['torsion_psi'] = torsion['psi']
        losses['torsion_omega'] = torsion['omega']

        losses['clash'] = self.clash_loss(pred_coords, atom_mask)

        lddt_loss, lddt_score = self.lddt_loss(pred_coords, true_coords, atom_mask)
        losses['lddt'] = lddt_loss
        losses['lddt_score'] = lddt_score

        losses['total'] = (
            self.loss_weights['fape'] * losses['fape'] +
            self.loss_weights['rmsd'] * losses['rmsd'] +
            #self.loss_weights['torsion'] * losses['torsion'] +
            self.loss_weights['clash'] * losses['clash'] +
            self.loss_weights['lddt'] * losses['lddt']
        )

        return losses


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
class BatchedDataLoader:
    def __init__(self, coords, seqs, paired_inds, n_atoms, cutoff=10., pad_token=-2):
        self.coords = coords
        self.seqs = seqs
        self.paired_inds = paired_inds
        self.n_atoms = n_atoms
        self.cutoff = cutoff
        self.pad_token = pad_token
        self.num_samples = len(paired_inds)
        print(f"DataLoader initialized with {self.num_samples} samples")
    
    def _preprocess_single(self, idx):
        pind = self.paired_inds[idx]
        length = pind[2]
        pep_len = pind[3]
        input_coords = self.coords[pind[0]][:length]
        true_coords = self.coords[pind[1]][:length]
        ca_coords = input_coords[:, ATOM_CA, :][np.newaxis, :, :]
        num_res = ca_coords.shape[1]
        adj, _ = build_radius_graph(
            coords=ca_coords,
            mask=np.ones_like(ca_coords, dtype=np.float32)[:, :, 0],
            cutoff=self.cutoff
        )
        adj = adj[0]
        adj_inds = np.argwhere(adj == 1)
        pep_residues = np.arange(num_res - pep_len, num_res)
        mask = np.isin(adj_inds[:, 0], pep_residues)
        remained_inds = np.unique(adj_inds[mask].flatten())
        remained_inds = np.sort(remained_inds)[::-1]
        n_remained = len(remained_inds)
        if n_remained == 0:
            return None
        input_coords = input_coords[remained_inds, :, :]
        true_coords = true_coords[remained_inds, :, :]
        wt_indices = self.seqs[pind[0]][:length][remained_inds]
        mut_indices = self.seqs[pind[1]][:length][remained_inds]
        # Count how many peptide residues in remained_inds
        pep_remained = np.sum(np.isin(remained_inds, pep_residues))
        # Build res_indices: 1 for peptide, 0 for non-peptide, repeated for each atom
        res_indices_single = np.concatenate([
            np.ones(pep_remained, dtype=np.int32),
            np.zeros(n_remained - pep_remained, dtype=np.int32)
        ])
        res_indices = np.repeat(res_indices_single, self.n_atoms)
        return {
            'input_coords': input_coords.astype(np.float32),
            'true_coords': true_coords.astype(np.float32),
            'wt_indices': wt_indices.astype(np.int32),
            'mut_indices': mut_indices.astype(np.int32),
            'res_indices': res_indices,
            'length': n_remained
        }
    
    def _collate_batch(self, batch_data):
        batch_size = len(batch_data)
        max_len = max(d['length'] for d in batch_data)
        max_res_idx_len = max(len(d['res_indices']) for d in batch_data)
        input_coords = np.zeros((batch_size, max_len, self.n_atoms, 3), dtype=np.float32)
        true_coords = np.zeros((batch_size, max_len, self.n_atoms, 3), dtype=np.float32)
        wt_indices = np.full((batch_size, max_len), self.pad_token, dtype=np.int32)
        mut_indices = np.full((batch_size, max_len), self.pad_token, dtype=np.int32)
        res_indices = np.zeros((batch_size, max_res_idx_len), dtype=np.int32)
        atom_mask = np.zeros((batch_size, max_len, self.n_atoms), dtype=np.float32)
        for i, d in enumerate(batch_data):
            l = d['length']
            input_coords[i, :l] = d['input_coords']
            true_coords[i, :l] = d['true_coords']
            wt_indices[i, :l] = d['wt_indices']
            mut_indices[i, :l] = d['mut_indices']
            res_indices[i, :len(d['res_indices'])] = d['res_indices']
            atom_mask[i, :l] = 1.
        mutation_mask = np.where(wt_indices == mut_indices, 0., 1.).astype(np.float32)
        inputs = {
            'coords': tf.constant(input_coords),
            'wt_indices': tf.constant(wt_indices),
            'mut_indices': tf.constant(mut_indices),
            'mutation_mask': tf.constant(mutation_mask),
            'atom_mask': tf.constant(atom_mask),
            'res_indices': tf.constant(res_indices),
        }
        true_coords_tf = tf.constant(true_coords)
        atom_mask_tf = tf.constant(atom_mask)
        return inputs, true_coords_tf, atom_mask_tf
    
    def get_batches(self, batch_size, shuffle=True):
        indices = np.arange(self.num_samples)
        if shuffle:
            np.random.shuffle(indices)
        for start_idx in range(0, len(indices), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            batch_data = []
            for idx in batch_indices:
                try:
                    sample = self._preprocess_single(idx)
                    if sample is not None:
                        batch_data.append(sample)
                except Exception as e:
                    continue
            if len(batch_data) > 0:
                yield self._collate_batch(batch_data)
    
    def get_num_batches(self, batch_size):
        return (self.num_samples + batch_size - 1) // batch_size


def train_batched():
    print("=" * 60)
    print("Batched Training")
    print("=" * 60)
    tf.random.set_seed(42)
    
    # Config
    batch_size = 20
    num_epochs = 3
    n_atoms = NUM_ATOM_TYPES
    
    # Load data
    coords = np.load(os.path.join(OUTPUT_DIR, 'coords.npy'))
    paired = pd.read_csv(os.path.join(OUTPUT_DIR, 'paired.csv'))
    seqs = np.load(os.path.join(OUTPUT_DIR, 'seqs.npy'))
    paired_inds = np.array(paired[['index_1', 'index_2', 'length_1', 'p_len']])
    
    # Create data loader
    data_loader = BatchedDataLoader(coords, seqs, paired_inds, n_atoms, cutoff=9., pad_token=-2)
    num_batches = data_loader.get_num_batches(batch_size)
    
    # Model and optimizer
    model = MutationStructurePredictorAtt(
        node_dim=128, edge_dim=128, hidden_dim=256, heads=8, num_layers=1, cutoff=3., PAD_TOKEN=-2.
    )

    loss_fn = StructuralLoss()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    if mixed_precision:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    # TensorBoard
    log_dir = os.path.join(OUTPUT_DIR, 'logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    train_summary_writer = tf.summary.create_file_writer(log_dir)
    print(f"TensorBoard: tensorboard --logdir {os.path.join(OUTPUT_DIR, 'logs')}")
    
    # Checkpointing
    checkpoint_dir = os.path.join(OUTPUT_DIR, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_loss = float('inf')
    recent_losses = []
    global_step = 0
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}/{num_epochs}")
        epoch_losses = []
        
        for batch_idx, (inputs, true_coords, atom_mask) in enumerate(data_loader.get_batches(batch_size, shuffle=True)):
            with tf.GradientTape() as tape:
                pred, node_att, edge_att = model(inputs, training=True)
                losses = loss_fn(pred, true_coords, atom_mask)
                total_loss = losses['total']
                if mixed_precision:
                    scaled_loss = optimizer.get_scaled_loss(total_loss)
                else:
                    scaled_loss = total_loss
            
            grads = tape.gradient(scaled_loss, model.trainable_variables)
            if mixed_precision:
                grads = optimizer.get_unscaled_gradients(grads)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            loss_val = total_loss.numpy()
            epoch_losses.append(loss_val)
            recent_losses.append(loss_val)
           
            if batch_idx % 1 == 0:
                print(f"  Batch {batch_idx}/{num_batches}, Loss: {loss_val:.6f}, Shape {tf.shape(pred)}")
            
            if global_step % 500 == 0 and global_step > 0:
                avg_recent_loss = np.mean(recent_losses)
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss/avg_500_steps', avg_recent_loss, step=global_step)
                    tf.summary.scalar('loss/best', best_loss, step=global_step)
                if avg_recent_loss < best_loss:
                    best_loss = avg_recent_loss
                    model.save_weights(os.path.join(checkpoint_dir, 'best_model.weights.h5'))
                    print(f"  Step {global_step}: New best! Loss: {best_loss:.6f}")
                else:
                    print(f"  Step {global_step}: Avg: {avg_recent_loss:.6f}, Best: {best_loss:.6f}")
                recent_losses = []
            
            if global_step % 100 == 0:
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss/total', total_loss, step=global_step)
                    for k, v in losses.items():
                        tf.summary.scalar(f'loss/{k}', v, step=global_step)
            
            global_step += 1
        avg_epoch_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch} complete. Avg Loss: {avg_epoch_loss:.6f}, Best: {best_loss:.6f}")
        with train_summary_writer.as_default():
            tf.summary.scalar('epoch/avg_loss', avg_epoch_loss, step=epoch)
        
        if (epoch + 1) % 10 == 0:
            model.save_weights(os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.weights.h5'))
            print(f"Checkpoint saved at epoch {epoch+1}")
    
    model.save_weights(os.path.join(checkpoint_dir, 'final_model.weights.h5'))
    print(f"\nTraining complete! Best loss: {best_loss:.6f}")


if __name__ == "__main__":
    train_batched()