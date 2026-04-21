"""
ig_pipeline.py — Unified Initial Guess Pipeline for AFfine
===========================================================
Implements Tasks 1-3 of the IG Pipeline Expansion:
  Task 1: Flexible coordinate input (PDB or numpy, CA-only / CA+CB / full)
  Task 2: General IG masking via token array (replaces pMHC-specific logic)
  Task 3: Evoformer template masking
All functions are vectorized, memory-efficient, and avoid Python loops
where numpy broadcasting suffices.
"""
import os
import re
import collections
import numpy as np
from typing import Optional, Tuple, List, Union, Dict
from alphafold.common import residue_constants
from Bio import PDB
from Bio.PDB import PDBParser, is_aa, Polypeptide
from predict_utils import load_pdb_coords, fill_afold_coords
# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────
ATOM_TYPE_NUM = residue_constants.atom_type_num  # 37
CA_IDX = residue_constants.atom_order['CA']       # index of CA in atom37
CB_IDX = residue_constants.atom_order['CB']       # index of CB in atom37
N_IDX = residue_constants.atom_order['N']
C_IDX = residue_constants.atom_order['C']
# Three-to-one mapping including MSE→MET
THREE_TO_ONE = {v: k for k, v in residue_constants.restype_1to3.items()}
THREE_TO_ONE['MSE'] = 'M'
# Chain break offset matching AFfine convention
CHAIN_BREAK_OFFSET = 200
# Token convention for mask_array
TOKEN_DEFAULT = 0     # keep coords, eligible if in radius
TOKEN_MASK = -1       # always zero coords / repr
TOKEN_CENTER = -2     # sampling center, coords kept
TOKEN_STABLE = 1      # never touched
# ─────────────────────────────────────────────────────────────────────
# Task 1 — Flexible Coordinate Input
# ─────────────────────────────────────────────────────────────────────
def _estimate_cb_from_ca(ca_coords: np.ndarray, n_res: int) -> np.ndarray:
    """Estimate CB positions ~1.5 Å from CA along a canonical offset.
    For glycine residues the caller should use CA as pseudo-beta instead.
    Args:
        ca_coords: [N_res, 3] CA coordinates.
        n_res: number of residues.
    Returns:
        cb_coords: [N_res, 3] estimated CB coordinates.
    """
    # Canonical CB offset relative to CA in an ideal backbone frame
    # Approximate: CB ≈ CA + 1.5 Å along a fixed direction with small noise
    rng = np.random.RandomState(42)
    canonical_offset = np.array([0.0, 1.0, 1.2], dtype=np.float32)
    canonical_offset = canonical_offset / np.linalg.norm(canonical_offset) * 1.522
    noise = rng.uniform(-0.05, 0.05, size=(n_res, 3)).astype(np.float32)
    cb_coords = ca_coords + canonical_offset[None, :] + noise
    return cb_coords

def _parse_pdb_multichain(pdb_path: str) -> Tuple[
    np.ndarray, np.ndarray, str, np.ndarray, np.ndarray]:
    """Parse a multi-chain PDB into AF2-compatible arrays.
    Uses predict_utils.load_pdb_coords to guarantee residue counts are
    identical to create_single_template_features(). This is critical:
    the alignment TSV, template features, and IG mask must all agree
    on the exact same residue set.
    Handles altLoc (prefer ' ', 'A', '1'), MSE→MET, HETATM filtering,
    chain breaks — all inherited from load_pdb_coords.
    Args:
        pdb_path: path to the PDB file.
    Returns:
        all_positions: [N_res, 37, 3] atom coordinates.
        all_positions_mask: [N_res, 37] presence mask.
        sequence: one-letter sequence (all chains concatenated).
        chain_ids: [N_res] chain ID per residue (str array).
        residue_indices: [N_res] original PDB residue numbers (int array).
    """
    # Use the canonical PDB reader from predict_utils
    chains, all_resids, all_coords, all_name1s = load_pdb_coords(
        pdb_path, allow_chainbreaks=True, allow_skipped_lines=True)
    # Build atom37 coordinate arrays (same function used by template features)
    all_positions, all_positions_mask = fill_afold_coords(
        chains, all_resids, all_coords)
    # Build sequence, chain_ids, residue_indices arrays in residue order
    seq_chars = []
    chain_id_list = []
    resindex_list = []
    for ch in chains:
        for r in all_resids[ch]:
            seq_chars.append(all_name1s[ch][r])
            chain_id_list.append(ch)
            # Extract numeric part of resid (handles insertion codes like ' 123A')
            resid_str = r.strip()
            # PDB resid field is columns 22:27, may contain insertion code
            # Extract leading integer, default to 0 if unparseable
            num_part = ''.join(c for c in resid_str if c.isdigit() or c == '-')
            resindex_list.append(int(num_part) if num_part else 0)
    sequence = ''.join(seq_chars)
    chain_ids = np.array(chain_id_list, dtype='U4')  # U4 for multi-char chain IDs
    residue_indices = np.array(resindex_list, dtype=np.int32)
    return all_positions, all_positions_mask, sequence, chain_ids, residue_indices

def _apply_atom_filter(all_positions: np.ndarray,
                       all_positions_mask: np.ndarray,
                       sequence: str,
                       use_only_CA: bool,
                       use_CA_CB: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Zero-out atoms not requested by the filter mode. Vectorized.
    Args:
        all_positions: [N_res, 37, 3].
        all_positions_mask: [N_res, 37].
        sequence: one-letter sequence.
        use_only_CA: keep only CA.
        use_CA_CB: keep only CA and CB.
    Returns:
        filtered positions and mask (same shapes).
    """
    if not use_only_CA and not use_CA_CB:
        return all_positions, all_positions_mask
    n_res = all_positions.shape[0]
    # Build keep-mask over the 37 atom slots
    keep = np.zeros([ATOM_TYPE_NUM], dtype=np.float32)
    keep[CA_IDX] = 1.0
    if use_CA_CB:
        keep[CB_IDX] = 1.0
    # Broadcast: [37] → [N, 37]
    keep_2d = np.broadcast_to(keep[None, :], (n_res, ATOM_TYPE_NUM))
    all_positions_mask = all_positions_mask * keep_2d
    all_positions = all_positions * keep_2d[:, :, None]
    # For CA-only mode, estimate CB for non-GLY (needed for pseudo-beta)
    if use_only_CA:
        seq_arr = np.array(list(sequence))
        non_gly = (seq_arr != 'G')
        ca_coords = all_positions[:, CA_IDX, :]  # [N, 3]
        cb_est = _estimate_cb_from_ca(ca_coords, n_res)
        # Place estimated CB only for non-GLY and where CA exists
        ca_exists = all_positions_mask[:, CA_IDX] > 0.5
        place = non_gly & ca_exists
        all_positions[place, CB_IDX, :] = cb_est[place]
        all_positions_mask[place, CB_IDX] = 1.0
    return all_positions, all_positions_mask

def _coords_array_to_atom37(coords: np.ndarray,
                            sequence: str,
                            use_only_CA: bool,
                            use_CA_CB: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Convert user-provided coordinate arrays into atom37 format.
    Args:
        coords: [N,3] for CA-only, [N,2,3] for CA+CB, [N,37,3] for full.
        sequence: one-letter sequence.
        use_only_CA: True if coords is [N,3].
        use_CA_CB: True if coords is [N,2,3].
    Returns:
        all_positions: [N, 37, 3].
        all_positions_mask: [N, 37].
    """
    n_res = coords.shape[0]
    all_positions = np.zeros([n_res, ATOM_TYPE_NUM, 3], dtype=np.float32)
    all_positions_mask = np.zeros([n_res, ATOM_TYPE_NUM], dtype=np.float32)
    if use_only_CA:
        # coords: [N, 3]
        assert coords.ndim == 2 and coords.shape[1] == 3, \
            f"CA-only mode expects [N,3], got {coords.shape}"
        all_positions[:, CA_IDX, :] = coords
        all_positions_mask[:, CA_IDX] = 1.0
        # Estimate CB for non-GLY
        seq_arr = np.array(list(sequence))
        non_gly = (seq_arr != 'G')
        cb_est = _estimate_cb_from_ca(coords, n_res)
        all_positions[non_gly, CB_IDX, :] = cb_est[non_gly]
        all_positions_mask[non_gly, CB_IDX] = 1.0
    elif use_CA_CB:
        # coords: [N, 2, 3] where [:,0,:]=CA, [:,1,:]=CB
        assert coords.ndim == 3 and coords.shape[1] == 2, \
            f"CA+CB mode expects [N,2,3], got {coords.shape}"
        all_positions[:, CA_IDX, :] = coords[:, 0, :]
        all_positions_mask[:, CA_IDX] = 1.0
        all_positions[:, CB_IDX, :] = coords[:, 1, :]
        all_positions_mask[:, CB_IDX] = 1.0
        # For GLY, zero out CB (use CA as pseudo-beta downstream)
        seq_arr = np.array(list(sequence))
        gly_mask = (seq_arr == 'G')
        all_positions[gly_mask, CB_IDX, :] = 0.0
        all_positions_mask[gly_mask, CB_IDX] = 0.0
    else:
        # Full atom: coords [N, 37, 3]
        assert coords.ndim == 3 and coords.shape[1] == ATOM_TYPE_NUM, \
            f"Full-atom mode expects [N,37,3], got {coords.shape}"
        all_positions[:] = coords
        # Mask: any atom with nonzero coords is considered present
        all_positions_mask = (np.abs(coords).sum(axis=-1) > 1e-6).astype(np.float32)
    return all_positions, all_positions_mask
def compute_residue_index_with_chain_breaks(
        chain_ids: np.ndarray,
        residue_indices: np.ndarray) -> np.ndarray:
    """Build AF2-style residue_index with +200 offsets at chain boundaries.
    Args:
        chain_ids: [N_res] chain ID per residue.
        residue_indices: [N_res] original PDB residue indices.
    Returns:
        residue_index: [N_res] with chain-break offsets applied.
    """
    residue_index = residue_indices.copy().astype(np.int64)
    # Detect chain boundaries: where chain_id changes
    chain_changes = np.where(chain_ids[1:] != chain_ids[:-1])[0]  # indices before break
    offset = 0
    for break_pos in chain_changes:
        # Add offset to everything after this break
        offset += CHAIN_BREAK_OFFSET
        residue_index[break_pos + 1:] += CHAIN_BREAK_OFFSET
    return residue_index
def parse_structure_input(
        pdb_path: Optional[str] = None,
        coords: Optional[np.ndarray] = None,
        sequence: Optional[str] = None,
        chain_ids: Optional[np.ndarray] = None,
        residue_indices: Optional[np.ndarray] = None,
        use_only_CA: bool = False,
        use_CA_CB: bool = False
) -> Dict[str, object]:
    """Central parsing function for IG/template structural input.
    Accepts EITHER a PDB file OR numpy arrays. Produces standardized
    AF2-compatible atom-position arrays for downstream use by both the
    IG recycling path and the evoformer template path.
    Args:
        pdb_path: path to PDB file (Mode A).
        coords: coordinate array (Mode B). Shape depends on atom mode.
        sequence: one-letter AA sequence, all chains concatenated (Mode B).
        chain_ids: [N_res] chain ID per residue (Mode B).
        residue_indices: [N_res] residue index per residue (Mode B).
        use_only_CA: extract/use only CA atoms.
        use_CA_CB: extract/use CA and CB atoms only.
    Returns:
        dict with keys:
          all_positions      [N_res, 37, 3]
          all_positions_mask  [N_res, 37]
          sequence            str
          chain_ids           [N_res] str array
          residue_indices     [N_res] int array
          residue_index       [N_res] with chain-break offsets
    """
    assert not (use_only_CA and use_CA_CB), \
        "Cannot set both use_only_CA and use_CA_CB."
    # ── Mode A: PDB file ──
    if pdb_path is not None:
        assert os.path.isfile(pdb_path), f"PDB not found: {pdb_path}"
        all_pos, all_mask, sequence, chain_ids, residue_indices = \
            _parse_pdb_multichain(pdb_path)
        all_pos, all_mask = _apply_atom_filter(
            all_pos, all_mask, sequence, use_only_CA, use_CA_CB)
    # ── Mode B: Numpy arrays ──
    elif coords is not None:
        assert sequence is not None, "sequence required for array mode."
        assert chain_ids is not None, "chain_ids required for array mode."
        assert residue_indices is not None, "residue_indices required for array mode."
        n_res = len(sequence)
        assert coords.shape[0] == n_res, \
            f"coords has {coords.shape[0]} residues but sequence has {n_res}."
        assert chain_ids.shape[0] == n_res
        assert residue_indices.shape[0] == n_res
        all_pos, all_mask = _coords_array_to_atom37(
            coords, sequence, use_only_CA, use_CA_CB)
    else:
        raise ValueError("Provide either pdb_path or coords array.")
    # Compute residue_index with chain breaks
    res_index = compute_residue_index_with_chain_breaks(chain_ids, residue_indices)
    return {
        'all_positions': all_pos,
        'all_positions_mask': all_mask,
        'sequence': sequence,
        'chain_ids': chain_ids,
        'residue_indices': residue_indices,
        'residue_index': res_index,
    }
# ─────────────────────────────────────────────────────────────────────
# Task 2 — General IG Masking via Token Array
# ─────────────────────────────────────────────────────────────────────
def parse_residue_spec(spec_str: str,
                       chain_ids: np.ndarray,
                       residue_indices: np.ndarray) -> np.ndarray:
    """Parse a flexible residue specification string into boolean index array.
    Supports formats:
      "B:5-9"       → chain B, residues 5 through 9 inclusive
      "B:5,B:7"     → chain B residues 5 and 7
      "A:*"          → all residues in chain A
      "150,151,152"  → raw 0-indexed positions (no chain prefix)
    Args:
        spec_str: residue specification string.
        chain_ids: [N_res] chain ID array.
        residue_indices: [N_res] residue index array.
    Returns:
        selected: [N_res] boolean array, True for selected residues.
    """
    n_res = chain_ids.shape[0]
    selected = np.zeros(n_res, dtype=bool)
    # Split by comma, but be careful: "B:5-9" has no comma
    parts = [p.strip() for p in spec_str.split(',')]
    for part in parts:
        if ':' in part:
            # Chain:range format
            chain_part, range_part = part.split(':', 1)
            chain_match = (chain_ids == chain_part)
            if range_part == '*':
                # Entire chain
                selected |= chain_match
            elif '-' in range_part:
                # Range: "5-9"
                lo, hi = range_part.split('-', 1)
                lo, hi = int(lo), int(hi)
                in_range = (residue_indices >= lo) & (residue_indices <= hi)
                selected |= (chain_match & in_range)
            else:
                # Single residue number
                res_num = int(range_part)
                selected |= (chain_match & (residue_indices == res_num))
        else:
            # Raw 0-indexed position
            idx = int(part)
            if 0 <= idx < n_res:
                selected[idx] = True
    return selected
def generate_mask_from_structure(
        all_positions: np.ndarray,
        chain_ids: np.ndarray,
        residue_indices: np.ndarray,
        mask_residues: Optional[str] = None,
        sampling_centers: Optional[str] = None,
        stable_residues: Optional[str] = None,
        auto_sampling_radius: Optional[float] = None,
        mask_array_path: Optional[str] = None
) -> np.ndarray:
    """Generate the unified mask_array from structure and user specs.
    Token convention:
       0 = default (keep coords, eligible if in sampling radius)
      -1 = mask (always zero coords / repr)
      -2 = sampling center (coords preserved, center of sampling sphere)
      +1 = stable (never touched)
    Args:
        all_positions: [N_res, 37, 3] from parse_structure_input.
        chain_ids: [N_res] chain IDs.
        residue_indices: [N_res] residue indices.
        mask_residues: spec string for masked residues (→ -1).
        sampling_centers: spec string for sampling centers (→ -2).
        stable_residues: spec string for stable residues (→ +1).
        auto_sampling_radius: if set, auto-detect neighbors of mask
            residues within this radius as sampling centers.
        mask_array_path: path to a pre-computed .npy mask array.
    Returns:
        mask_array: [N_res] int array with tokens.
    """
    n_res = all_positions.shape[0]
    # If explicit mask array file provided, load and return
    if mask_array_path is not None:
        mask_array = np.load(mask_array_path).astype(np.int32)
        assert mask_array.shape[0] == n_res, \
            f"mask_array has {mask_array.shape[0]} entries but structure has {n_res}."
        return mask_array
    mask_array = np.zeros(n_res, dtype=np.int32)
    # Apply mask_residues → token -1
    if mask_residues is not None:
        sel = parse_residue_spec(mask_residues, chain_ids, residue_indices)
        mask_array[sel] = TOKEN_MASK
    # Apply explicit sampling_centers → token -2
    if sampling_centers is not None:
        sel = parse_residue_spec(sampling_centers, chain_ids, residue_indices)
        # Only assign -2 where not already -1
        mask_array[sel & (mask_array != TOKEN_MASK)] = TOKEN_CENTER
    # Auto-detect sampling centers around mask residues
    if auto_sampling_radius is not None and mask_residues is not None:
        mask_positions = np.where(mask_array == TOKEN_MASK)[0]
        if mask_positions.size > 0:
            # Vectorized radius computation using CA coordinates
            all_ca = all_positions[:, CA_IDX, :]           # [N_res, 3]
            center_ca = all_ca[mask_positions]              # [M, 3]
            # Pairwise distances: [M, N_res]
            diff = all_ca[None, :, :] - center_ca[:, None, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=-1))     # [M, N_res]
            in_any_radius = np.any(dists <= auto_sampling_radius, axis=0)  # [N_res]
            # Assign -2 to neighbors that are currently default (0)
            auto_center_sel = in_any_radius & (mask_array == TOKEN_DEFAULT)
            mask_array[auto_center_sel] = TOKEN_CENTER
    # Apply stable_residues → token +1 (overrides auto -2)
    if stable_residues is not None:
        sel = parse_residue_spec(stable_residues, chain_ids, residue_indices)
        mask_array[sel] = TOKEN_STABLE
    return mask_array
def apply_ig_mask(all_positions: np.ndarray,
                  mask_array: np.ndarray) -> np.ndarray:
    """Zero out IG coordinates for mask-token residues. Vectorized.
    Only token -1 gets zeroed. Tokens 0, -2, +1 keep their coordinates.
    Args:
        all_positions: [N_res, 37, 3] atom coordinates.
        mask_array: [N_res] token array.
    Returns:
        masked_positions: [N_res, 37, 3] with mask-token rows zeroed.
    """
    masked_positions = all_positions.copy()
    zero_mask = (mask_array == TOKEN_MASK)  # [N_res] bool
    # Broadcast: [N_res] → [N_res, 1, 1] * [N_res, 37, 3]
    masked_positions[zero_mask] = 0.0
    return masked_positions
def legacy_anchors_to_mask(n_res: int,
                           pep_len: int,
                           anchors: List[int],
                           mhc_len: Optional[int] = None) -> np.ndarray:
    """Convert legacy pMHC anchor/peptide specification to mask_array.
    Backward-compatible mapping:
      MHC residues → 0 (default)
      Peptide anchors → 0 (default)
      Non-anchor peptide → -1 (mask)
    Args:
        n_res: total number of residues.
        pep_len: length of the peptide chain.
        anchors: 1-indexed anchor positions within the peptide.
        mhc_len: number of MHC residues. If None, inferred as n_res - pep_len.
    Returns:
        mask_array: [n_res] int token array.
    """
    if mhc_len is None:
        mhc_len = n_res - pep_len
    mask_array = np.zeros(n_res, dtype=np.int32)
    # Convert 1-indexed peptide anchors to 0-indexed absolute positions
    anchor_abs = set(a - 1 + mhc_len for a in anchors)
    # Peptide region: all non-anchor residues get masked
    for i in range(mhc_len, mhc_len + pep_len):
        if i not in anchor_abs:
            mask_array[i] = TOKEN_MASK
    return mask_array
# ─────────────────────────────────────────────────────────────────────
# Task 3 — Evoformer Template Masking
# ─────────────────────────────────────────────────────────────────────
def apply_template_mask(template_features: dict,
                        evo_mask: np.ndarray) -> dict:
    """Zero out template features for masked residues. Vectorized.
    Modifies the template feature dict in-place (and returns it).
    For residues where evo_mask == -1: zero positions, masks, pseudo-beta.
    The existing SingleTemplateEmbedding in modules.py will then see
    template_mask_2d==0 for those rows/cols, effectively removing them.
    Args:
        template_features: dict of template arrays from compile_template_features.
        evo_mask: [N_res] mask array (same token convention, -1 = mask).
    Returns:
        modified template_features dict.
    """
    masked = (evo_mask == TOKEN_MASK)  # [N_res] bool
    tf = template_features
    # template_all_atom_positions: [N_tmpl, N_res, 37, 3]
    if 'template_all_atom_positions' in tf:
        tf['template_all_atom_positions'][:, masked, :, :] = 0.0
    # template_all_atom_masks: [N_tmpl, N_res, 37]
    if 'template_all_atom_masks' in tf:
        tf['template_all_atom_masks'][:, masked, :] = 0.0
    # template_pseudo_beta: [N_tmpl, N_res, 3]
    if 'template_pseudo_beta' in tf:
        tf['template_pseudo_beta'][:, masked, :] = 0.0
    # template_pseudo_beta_mask: [N_tmpl, N_res]
    if 'template_pseudo_beta_mask' in tf:
        tf['template_pseudo_beta_mask'][:, masked] = 0.0
    return tf
# ─────────────────────────────────────────────────────────────────────
# Task 1 (continued) — Template Feature Builder from Structure
# ─────────────────────────────────────────────────────────────────────
def create_template_features_from_structure(
        all_positions: np.ndarray,
        all_positions_mask: np.ndarray,
        sequence: str,
        chain_ids: np.ndarray = None,
        residue_indices: np.ndarray = None
) -> dict:
    """Build AF2-compatible template feature dict from parsed structure.
    This allows the IG structure to also be injected via the evoformer
    template path (--template_from_ig flag).
    Args:
        all_positions: [N_res, 37, 3] from parse_structure_input.
        all_positions_mask: [N_res, 37].
        sequence: one-letter AA sequence.
        chain_ids: unused here, kept for API consistency.
        residue_indices: unused here, kept for API consistency.
    Returns:
        template feature dict compatible with compile_template_features().
    """
    n_res = all_positions.shape[0]
    # aatype one-hot via HHBLITS_AA_TO_ID
    template_aatype = residue_constants.sequence_to_onehot(
        sequence, residue_constants.HHBLITS_AA_TO_ID)  # [N_res, 22]
    # Compute pseudo-beta: CB for non-GLY, CA for GLY
    seq_arr = np.array(list(sequence))
    is_gly = (seq_arr == 'G')
    ca_coords = all_positions[:, CA_IDX, :]   # [N_res, 3]
    cb_coords = all_positions[:, CB_IDX, :]   # [N_res, 3]
    ca_mask_1d = all_positions_mask[:, CA_IDX]
    cb_mask_1d = all_positions_mask[:, CB_IDX]
    # Pseudo-beta: use CB where available, else CA
    pseudo_beta = np.where(is_gly[:, None], ca_coords, cb_coords)
    pseudo_beta_mask = np.where(is_gly, ca_mask_1d, cb_mask_1d)
    # Fallback: if CB missing for non-GLY, use CA
    cb_missing = (~is_gly) & (cb_mask_1d < 0.5) & (ca_mask_1d > 0.5)
    pseudo_beta[cb_missing] = ca_coords[cb_missing]
    pseudo_beta_mask[cb_missing] = ca_mask_1d[cb_missing]
    # Package with template batch dimension [1, ...]
    template_features = {
        'template_all_atom_positions': all_positions[None, ...],        # [1, N, 37, 3]
        'template_all_atom_masks': all_positions_mask[None, ...],       # [1, N, 37]
        'template_aatype': template_aatype[None, ...],                  # [1, N, 22]
        'template_pseudo_beta': pseudo_beta[None, ...],                 # [1, N, 3]
        'template_pseudo_beta_mask': pseudo_beta_mask[None, ...],       # [1, N]
        'template_domain_names': np.array(['none'.encode()]),
        'template_sequence': np.array([sequence.encode()]),
        'template_sum_probs': np.zeros([1, 1], dtype=np.float32),
    }
    return template_features
# ─────────────────────────────────────────────────────────────────────
# Task 1 (continued) — Updated initial_guess_features
# ─────────────────────────────────────────────────────────────────────
def initial_guess_features_v2(
        structure_output: dict,
        mask_array: Optional[np.ndarray] = None
) -> 'jnp.ndarray':
    """Build IG coordinates from parse_structure_input output.
    Replaces the legacy initial_guess_features() with general-purpose logic.
    Args:
        structure_output: dict from parse_structure_input().
        mask_array: optional [N_res] mask token array. If provided,
            residues with token -1 get coordinates zeroed.
    Returns:
        initial_guess: jnp.ndarray [N_res, 37, 3] ready for modules.py.
    """
    import jax.numpy as jnp
    all_positions = structure_output['all_positions'].copy()
    # Apply IG mask (token -1 → zero coords)
    if mask_array is not None:
        all_positions = apply_ig_mask(all_positions, mask_array)
    # Convert to jax array (same as legacy parse_initial_guess)
    return jnp.array(all_positions, dtype=jnp.float32)
def initial_guess_features_legacy(
        query_seq: str,
        template_pdb_path: str,
        template_sequence: Optional[str] = None,
        aln=None,
        anchors=None,
        peptide_seq: Optional[str] = None,
        ig_mask: Optional[np.ndarray] = None
) -> 'jnp.ndarray':
    """Backward-compatible wrapper around the legacy IG path.
    If ig_mask is provided, uses new masking. Otherwise falls back to
    the original anchor/peptide zeroing via legacy_anchors_to_mask.
    Args:
        query_seq: query amino-acid sequence.
        template_pdb_path: path to the template PDB.
        template_sequence: template sequence (or None to auto-extract).
        aln: alignment tuple (query_aligned, template_aligned).
        anchors: list of 1-indexed anchor positions (legacy pMHC).
        peptide_seq: peptide sequence string (legacy pMHC).
        ig_mask: explicit mask array; if given, overrides anchors logic.
    Returns:
        initial_guess: jnp.ndarray [N_res, 37, 3].
    """
    import jax.numpy as jnp
    from af2_util import get_atom_positions_initial_guess, parse_initial_guess
    # Get raw coordinates via the original alignment-based path
    all_positions, _ = get_atom_positions_initial_guess(
        template_pdb_path, query_seq, template_sequence, aln, anchors, peptide_seq)
    # Apply mask
    if ig_mask is not None:
        all_positions = apply_ig_mask(all_positions, ig_mask)
    elif anchors is not None and peptide_seq is not None:
        # Legacy path: auto-convert to mask array
        n_res = all_positions.shape[0]
        pep_len = len(peptide_seq)
        legacy_mask = legacy_anchors_to_mask(n_res, pep_len, anchors)
        all_positions = apply_ig_mask(all_positions, legacy_mask)
    return parse_initial_guess(all_positions)


