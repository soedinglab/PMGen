"""
sampling_utils.py — Task 4: Sampling Pipeline via Evoformer Representation Masking
===================================================================================
Runs the evoformer ONCE with IG, caches representations, then runs the
structure module N times with stochastically masked representations.
Produces distributions over coordinates, pLDDT, and PAE.
All operations are vectorized / JAX-parallelized where possible.
The full AlphaFold model's Haiku params have keys prefixed with
'alphafold/alphafold_iteration/' (e.g. 'alphafold/alphafold_iteration/
structure_module/single_layer_norm'). When we create a standalone
hk.transform for just the structure module + heads, the module names
start from root (e.g. 'structure_module/single_layer_norm'). This
causes "Unable to retrieve parameter" errors.
FIX: Extract sub-params and strip the prefix before calling sm_apply.
Also fixes:
  - jax.errors.JaxRuntimeError (doesn't exist in older JAX versions)
  - Radius unification between Task 2 and Task 4
"""
import haiku as hk
import os
import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional, Dict, Tuple, Any
from functools import partial
from ig_pipeline import (
    TOKEN_DEFAULT, TOKEN_MASK, TOKEN_CENTER, TOKEN_STABLE, CA_IDX
)
# ─────────────────────────────────────────────────────────────────────
# Phase B: Compute Residue Sets (vectorized, runs once)
# ─────────────────────────────────────────────────────────────────────
def compute_residue_sets(
        mask_array: np.ndarray,
        ig_positions: np.ndarray,
        radius: float
) -> Dict[str, np.ndarray]:
    """Compute MASK_SET, CENTERS, STABLE_SET, SAMPLEABLE from mask_array.
    All distance computations are fully vectorized (no Python loops).
    Args:
        mask_array: [N_res] integer token array.
        ig_positions: [N_res, 37, 3] IG atom coordinates (for CA distances).
        radius: sampling radius in Angstroms around centers.
    Returns:
        dict with boolean arrays:
          mask_set:    [N_res] always zeroed in repr
          centers:     [N_res] sampling centers
          stable_set:  [N_res] never touched
          sampleable:  [N_res] union of centers + eligible-in-radius defaults
          eligible:    [N_res] default-token residues within radius
    """
    n_res = mask_array.shape[0]
    mask_set = (mask_array == TOKEN_MASK)        # -1
    centers = (mask_array == TOKEN_CENTER)        # -2
    stable_set = (mask_array == TOKEN_STABLE)     # +1
    default_set = (mask_array == TOKEN_DEFAULT)   # 0
    # Extract CA coordinates for radius computation
    all_ca = ig_positions[:, CA_IDX, :]            # [N_res, 3]
    center_indices = np.where(centers)[0]
    if center_indices.size > 0:
        center_ca = all_ca[center_indices]          # [M, 3]
        # Vectorized pairwise distance: [M, N_res]
        diff = all_ca[None, :, :] - center_ca[:, None, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        # Any residue within radius of any center
        in_any_radius = np.any(dists <= radius, axis=0)  # [N_res]
    else:
        in_any_radius = np.zeros(n_res, dtype=bool)
    # Eligible: within radius AND default token (not stable, not mask)
    eligible = in_any_radius & default_set
    # Sampleable = centers ∪ eligible
    sampleable = centers | eligible
    return {
        'mask_set': mask_set,
        'centers': centers,
        'stable_set': stable_set,
        'sampleable': sampleable,
        'eligible': eligible,
    }
# ─────────────────────────────────────────────────────────────────────
# Phase C: Stochastic Masking of Cached Representations (JAX)
# ─────────────────────────────────────────────────────────────────────
def _random_subset(key: jnp.ndarray,
                   sampleable_bool: jnp.ndarray,
                   fraction: float) -> jnp.ndarray:
    """Randomly select a fraction of sampleable residues. Pure JAX.
    Args:
        key: JAX PRNG key.
        sampleable_bool: [N_res] boolean, True for sampleable residues.
        fraction: fraction of sampleable residues to select.
    Returns:
        selected: [N_res] boolean, True for selected subset.
    """
    n_res = sampleable_bool.shape[0]
    # Generate uniform random per residue
    rand_vals = jax.random.uniform(key, shape=(n_res,))
    # Select: sampleable AND random < fraction
    selected = sampleable_bool & (rand_vals < fraction)
    return selected
## ── Hardcoded dropout rate for sampling-phase perturbation ──
def generate_masked_representations_single_sample(
        cached_single: jnp.ndarray,
        cached_pair: jnp.ndarray,
        mask_set: jnp.ndarray,
        sampleable: jnp.ndarray,
        sampling_fraction_ig: float,
        sampling_fraction_evo: float,
        key_ig: jnp.ndarray,
        key_evo: jnp.ndarray,
        sampling_dropout_rate: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate one masked (single, pair) representation sample.
    For masked/sampled residues, applies channel-wise dropout instead of
    zeroing. Dropout is in-distribution for the structure module (AF2
    uses dropout during training), so the SM interprets partially-dropped
    channels as noisy-but-plausible signal rather than the pathological
    all-zero case.
    Stable and unselected residues pass through unchanged.
    Args:
        cached_single: [N_res, c_s] cached evoformer single representation.
        cached_pair: [N_res, N_res, c_z] cached evoformer pair representation.
        mask_set: [N_res] bool, always-perturbed residues (token -1).
        sampleable: [N_res] bool, stochastically perturbed (token -2 + eligible).
        sampling_fraction_ig: fraction of sampleable to perturb in single repr.
        sampling_fraction_evo: fraction of sampleable to perturb in pair repr.
        key_ig: PRNG key for single-repr sampling.
        key_evo: PRNG key for pair-repr sampling.
    Returns:
        masked_single: [N_res, c_s]
        masked_pair: [N_res, N_res, c_z]
    """
    rate = sampling_dropout_rate
    keep_rate = 1.0 - rate
    n_res, c_s = cached_single.shape
    c_z = cached_pair.shape[-1]
    # ── Single representation: dropout on selected residues ──
    # Split key: one for subset selection, one for dropout pattern
    key_ig_sel, key_ig_drop = jax.random.split(key_ig)
    ig_sample = _random_subset(key_ig_sel, sampleable, sampling_fraction_ig)
    single_affected = mask_set | ig_sample  # [N_res] bool
    # Channel-wise dropout mask: [N_res, c_s], 1=keep 0=drop
    single_drop_mask = jax.random.bernoulli(
        key_ig_drop, keep_rate, shape=(n_res, c_s)
    ).astype(jnp.float32) / keep_rate  # scale to preserve expected magnitude
    # Apply dropout only to affected residues; others pass through unchanged
    affected_1d = single_affected.astype(jnp.float32)[:, None]  # [N_res, 1]
    masked_single = cached_single * (
        (1.0 - affected_1d) + affected_1d * single_drop_mask
    )
    # ── Pair representation: dropout on affected rows/columns ──
    key_evo_sel, key_evo_drop = jax.random.split(key_evo)
    evo_sample = _random_subset(key_evo_sel, sampleable, sampling_fraction_evo)
    pair_affected = mask_set | evo_sample  # [N_res] bool
    # Entry [i,j] is affected if EITHER i or j is in pair_affected
    keep_1d = (~pair_affected).astype(jnp.float32)
    pair_unaffected_2d = keep_1d[:, None] * keep_1d[None, :]  # [N, N]
    pair_affected_2d = 1.0 - pair_unaffected_2d  # [N, N]
    # Channel-wise dropout mask: [N_res, N_res, c_z]
    pair_drop_mask = jax.random.bernoulli(
        key_evo_drop, keep_rate, shape=(n_res, n_res, c_z)
    ).astype(jnp.float32) / keep_rate
    # Apply dropout only to affected entries; others pass through unchanged
    masked_pair = cached_pair * (
        pair_unaffected_2d[:, :, None] + pair_affected_2d[:, :, None] * pair_drop_mask
    )
    return masked_single, masked_pair

# Vectorized (vmap) version for batch of samples
def _generate_single_sample_vmappable(
        carry: Tuple,
        keys: jnp.ndarray,
        sampling_dropout_rate: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Vmappable wrapper: takes (key_ig, key_evo) and shared carry.
    Args:
        carry: (cached_single, cached_pair, mask_set, sampleable,
                sampling_fraction_ig, sampling_fraction_evo)
        keys: [2] array of two PRNG keys.
    Returns:
        (masked_single, masked_pair)
    """
    (cached_single, cached_pair, mask_set, sampleable,
     frac_ig, frac_evo) = carry
    key_ig, key_evo = keys[0], keys[1]
    return generate_masked_representations_single_sample(
        cached_single, cached_pair,
        mask_set, sampleable,
        frac_ig, frac_evo,
        key_ig, key_evo, sampling_dropout_rate)
def generate_all_masked_representations(
        cached_single: jnp.ndarray,
        cached_pair: jnp.ndarray,
        mask_set: jnp.ndarray,
        sampleable: jnp.ndarray,
        sampling_fraction_ig: float,
        sampling_fraction_evo: float,
        n_samples: int,
        base_seed: int = 42,
        sampling_dropout_rate: float = 0.1,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate N masked representation pairs using vmap for GPU parallelism.
    Args:
        cached_single: [N_res, c_s].
        cached_pair: [N_res, N_res, c_z].
        mask_set: [N_res] bool.
        sampleable: [N_res] bool.
        sampling_fraction_ig: fraction for single repr.
        sampling_fraction_evo: fraction for pair repr.
        n_samples: number of samples to generate.
        base_seed: RNG seed.
    Returns:
        all_singles: [n_samples, N_res, c_s].
        all_pairs: [n_samples, N_res, N_res, c_z].
    """
    # Generate paired PRNG keys: [n_samples, 2]
    master_key = jax.random.PRNGKey(base_seed)
    all_keys = jax.random.split(master_key, n_samples * 2).reshape(n_samples, 2, 2)
    # vmap over the sample dimension
    def single_sample_fn(keys_pair):
        """Process one sample given its two PRNG keys."""
        key_ig, key_evo = keys_pair[0], keys_pair[1]
        return generate_masked_representations_single_sample(
            cached_single, cached_pair,
            mask_set, sampleable,
            sampling_fraction_ig, sampling_fraction_evo,
            key_ig, key_evo, sampling_dropout_rate)
    # Vectorize across n_samples
    batched_fn = jax.vmap(single_sample_fn)
    all_singles, all_pairs = batched_fn(all_keys)
    return all_singles, all_pairs  # [n_samples, N_res, ...], [n_samples, N_res, N_res, ...]
# ─────────────────────────────────────────────────────────────────────
# Phase D: Structure Module + Heads Runner (JAX/Haiku)
# ─────────────────────────────────────────────────────────────────────
def run_structure_module_only(
        params: dict,
        config: Any,
        masked_single: jnp.ndarray,
        masked_pair: jnp.ndarray,
        batch: dict,
        is_training: bool = False
) -> Dict[str, jnp.ndarray]:
    """Run only the structure module + confidence heads on pre-masked repr.
    This is the core architectural change: bypass the evoformer entirely,
    feed cached (masked) representations directly to folding + heads.
    Must be called inside a Haiku context (hk.transform).
    Args:
        params: Haiku parameters (from the full model).
        config: model config.
        masked_single: [N_res, c_s] masked single representation.
        masked_pair: [N_res, N_res, c_z] masked pair representation.
        batch: feature dict with seq_mask, aatype, residue_index etc.
        is_training: training mode flag.
    Returns:
        dict with 'final_atom_positions', 'plddt_logits', 'pae_logits',
        'structure_module' activations.
    """
    import haiku as hk
    from alphafold.model import modules, folding
    c = config.model
    gc = c.global_config
    # Build representations dict matching what AlphaFoldIteration heads expect
    representations = {
        'single': masked_single,
        'pair': masked_pair,
    }
    # ── Structure Module ──
    sm_config = c.heads['structure_module']
    structure_module = folding.StructureModule(sm_config, gc)
    sm_ret = structure_module(representations, batch, is_training)
    # The structure module returns 'final_atom_positions', etc.
    # It also stores internal activations in representations['structure_module']
    representations['structure_module'] = sm_ret.pop('act', sm_ret.get('representations', {}).get('structure_module', masked_single))
    result = {
        'final_atom_positions': sm_ret['final_atom_positions'],
        'final_atom_mask': sm_ret['final_atom_mask'],
    }
    # ── pLDDT Head ──
    # Reads representations['structure_module']
    if 'predicted_lddt' in c.heads and c.heads.get('predicted_lddt.weight', 0.0) > 0:
        lddt_head = modules.PredictedLDDTHead(
            c.heads['predicted_lddt'], gc, name='predicted_lddt')
        lddt_ret = lddt_head(representations, batch, is_training)
        result['plddt_logits'] = lddt_ret['logits']
    # ── PAE Head ──
    # Reads representations['pair'] (the masked pair we supplied)
    if 'predicted_aligned_error' in c.heads:
        pae_head = modules.PredictedAlignedErrorHead(
            c.heads['predicted_aligned_error'], gc,
            name='predicted_aligned_error')
        pae_ret = pae_head(representations, batch, is_training)
        result['pae_logits'] = pae_ret['logits']
    return result
def build_structure_only_fn(config: Any):
    """Build a JIT-compiled function for structure-module-only inference.
    Returns a function: (params, rng, masked_single, masked_pair, batch) → result
    Args:
        config: full model config.
    Returns:
        apply_fn: JIT-compiled Haiku apply function.
    """
    import haiku as hk
    def _forward(masked_single, masked_pair, batch, is_training=False):
        """Inner forward pass for hk.transform."""
        return run_structure_module_only(
            params=None,  # params injected by hk.transform
            config=config,
            masked_single=masked_single,
            masked_pair=masked_pair,
            batch=batch,
            is_training=is_training)
    transformed = hk.transform(_forward)
    apply_fn = jax.jit(transformed.apply)
    return apply_fn, transformed.init
# ─────────────────────────────────────────────────────────────────────
# Phase D (continued): Batched Structure Module over N Samples
# ─────────────────────────────────────────────────────────────────────

"""
sampling_utils.py — PATCH: Fix Haiku parameter mismatch + exception handling
=============================================================================
The full AlphaFold model's Haiku params have keys prefixed with
'alphafold/alphafold_iteration/' (e.g. 'alphafold/alphafold_iteration/
structure_module/single_layer_norm'). When we create a standalone
hk.transform for just the structure module + heads, the module names
start from root (e.g. 'structure_module/single_layer_norm'). This
causes "Unable to retrieve parameter" errors.
FIX: Extract sub-params and strip the prefix before calling sm_apply.
Also fixes:
  - jax.errors.JaxRuntimeError (doesn't exist in older JAX versions)
  - Radius unification between Task 2 and Task 4
"""

def extract_sm_params(full_params: dict) -> dict:
    """Extract structure module + confidence head params from full model.
    The full model's Haiku params have keys like:
        'alphafold/alphafold_iteration/structure_module/...'
        'alphafold/alphafold_iteration/predicted_lddt/...'
        'alphafold/alphafold_iteration/predicted_aligned_error/...'
    The standalone hk.transform expects:
        'structure_module/...'
        'predicted_lddt/...'
        'predicted_aligned_error/...'
    This function strips the 'alphafold/alphafold_iteration/' prefix
    and keeps only the modules needed for SM + heads.
    Args:
        full_params: complete Haiku params dict from RunModel.
    Returns:
        sm_params: stripped params for standalone SM + heads.
    """
    # The prefix used by AlphaFold's module hierarchy
    PREFIX = 'alphafold/alphafold_iteration/'
    # Module prefixes we need for SM + confidence heads
    KEEP_MODULES = (
        'structure_module',
        'predicted_lddt_head',
        'predicted_aligned_error_head',
        'experimentally_resolved_head',
    )
    sm_params = {}
    for key, value in full_params.items():
        # Strip the alphafold iteration prefix
        if key.startswith(PREFIX):
            stripped = key[len(PREFIX):]
        else:
            stripped = key
        # Keep only structure module and head params
        module_name = stripped.split('/')[0]
        if module_name in KEEP_MODULES:
            sm_params[stripped] = value
    if not sm_params:
        raise ValueError(
            f"No structure_module params found. Full param keys sample: "
            f"{list(full_params.keys())[:5]}. Expected prefix: '{PREFIX}'")
    print(f"[Sampling] Extracted {len(sm_params)} param groups for SM + heads "
          f"from {len(full_params)} total")
    return sm_params

def run_sampling_loop(
        params: dict,
        config: Any,
        cached_single: jnp.ndarray,
        cached_pair: jnp.ndarray,
        batch: dict,
        mask_array: np.ndarray,
        ig_positions: np.ndarray,
        radius: float,
        n_samples: int,
        sampling_fraction_ig: float = 0.5,
        sampling_fraction_evo: float = 0.3,
        base_seed: int = 42,
        sampling_dropout_rate: float = 0.1,
) -> Dict[str, np.ndarray]:
    """Full sampling pipeline: generate masks → run SM → collect distributions.
    For memory efficiency, processes samples in chunks when n_samples is large.
    Args:
        params: Haiku model parameters (full model).
        config: full model config.
        cached_single: [N_res, c_s] from evoformer.
        cached_pair: [N_res, N_res, c_z] from evoformer.
        batch: feature dict (seq_mask, aatype, residue_index...).
        mask_array: [N_res] token array from Task 2.
        ig_positions: [N_res, 37, 3] original IG coordinates.
        radius: sampling radius in Angstroms.
        n_samples: number of structure module samples.
        sampling_fraction_ig: fraction for single repr masking.
        sampling_fraction_evo: fraction for pair repr masking.
        base_seed: RNG seed.
    Returns:
        dict with all_coords, all_plddt, all_pae, statistics.
    """
    from alphafold.model import modules as mod
    from alphafold.model import folding
    batch = _squeeze_batch_ensemble(batch)
    # Phase B: compute residue sets
    residue_sets = compute_residue_sets(mask_array, ig_positions, radius)
    mask_set_jax = jnp.array(residue_sets['mask_set'])
    sampleable_jax = jnp.array(residue_sets['sampleable'])
    n_res = cached_single.shape[0]
    # Determine memory-safe chunk size
    c_z = cached_pair.shape[-1]
    mem_per_sample_mb = (n_res * n_res * c_z * 4) / (1024 ** 2)
    max_chunk_mb = 2048
    chunk_size = max(1, min(n_samples, int(max_chunk_mb / max(mem_per_sample_mb, 1))))
    print(f"[Sampling] N_res={n_res}, c_z={c_z}, chunk_size={chunk_size}, "
          f"n_samples={n_samples}")
    # ── FIX 1: Extract and remap params for standalone SM ──
    sm_params = extract_sm_params(params)
    # ── Build structure-only forward function ──
    c_model = config.model
    gc = c_model.global_config
    def _sm_forward(masked_single, masked_pair, batch_inner):
        """Haiku-transformable SM + heads forward pass."""
        representations = {'single': masked_single, 'pair': masked_pair}
        # Structure module
        sm = folding.StructureModule(c_model.heads['structure_module'], gc)
        sm_ret = sm(representations, batch_inner, is_training=False)
        # SM returns act inside 'representations' dict, not at top level
        if 'representations' in sm_ret and 'structure_module' in sm_ret['representations']:
            act = sm_ret['representations']['structure_module']
        elif 'act' in sm_ret:
            act = sm_ret['act']
        else:
            act = masked_single
            print("[WARNING] SM did not return activations, pLDDT will be unreliable")
        representations['structure_module'] = act
        result = {
            'final_atom_positions': sm_ret['final_atom_positions'],
            'final_atom_mask': sm_ret['final_atom_mask'],
        }
        # pLDDT head
        if 'predicted_lddt' in c_model.heads:
            lddt_head = mod.PredictedLDDTHead(c_model.heads['predicted_lddt'], gc)
            result['plddt_logits'] = lddt_head(representations, batch_inner, False)['logits']
        # PAE head
        if 'predicted_aligned_error' in c_model.heads:
            pae_head = mod.PredictedAlignedErrorHead(c_model.heads['predicted_aligned_error'], gc)
            result['pae_logits'] = pae_head(representations, batch_inner, False)['logits']
        return result
    sm_transformed = hk.transform(_sm_forward)
    # ### FIX 1 continued: use sm_params (stripped prefix), not full params
    sm_apply = jax.jit(sm_transformed.apply)
    # Collect results
    all_coords_list = []
    all_plddt_list = []
    all_pae_list = []
    sampling_freq = np.zeros(n_res, dtype=np.float32)
    n_processed = 0
    while n_processed < n_samples:
        current_chunk = min(chunk_size, n_samples - n_processed)
        # Generate masked representations for this chunk
        chunk_singles, chunk_pairs = generate_all_masked_representations(
            cached_single, cached_pair,
            mask_set_jax, sampleable_jax,
            sampling_fraction_ig, sampling_fraction_evo,
            current_chunk,
            base_seed=base_seed + n_processed,
            sampling_dropout_rate=sampling_dropout_rate)
        # Track sampling frequency
        for si in range(current_chunk):
            zeroed = jnp.all(chunk_singles[si] == 0, axis=-1)
            sampling_freq += np.array(zeroed, dtype=np.float32)
        # Run structure module for each sample in the chunk
        try:
            def _run_one(single_pair):
                """Run SM on one (single, pair) sample."""
                s, p = single_pair
                # ### FIX 1: pass sm_params instead of full params
                return sm_apply(sm_params, jax.random.PRNGKey(0), s, p, batch)
            batched_run = jax.vmap(_run_one)
            chunk_results = batched_run((chunk_singles, chunk_pairs))
            # Extract numpy arrays
            all_coords_list.append(np.array(chunk_results['final_atom_positions']))
            if 'plddt_logits' in chunk_results:
                plddt_probs = jax.nn.softmax(chunk_results['plddt_logits'], axis=-1)
                n_bins = plddt_probs.shape[-1]
                bin_centers = jnp.arange(0.5, n_bins, 1.0) / n_bins * 100.0
                plddt_scores = jnp.sum(plddt_probs * bin_centers[None, None, :], axis=-1)
                all_plddt_list.append(np.array(plddt_scores))
            if 'pae_logits' in chunk_results:
                pae_probs = jax.nn.softmax(chunk_results['pae_logits'], axis=-1)
                n_pae_bins = pae_probs.shape[-1]
                pae_bin_centers = jnp.arange(0.5, n_pae_bins) * 0.5
                pae_scores = jnp.sum(pae_probs * pae_bin_centers[None, None, None, :], axis=-1)
                all_pae_list.append(np.array(pae_scores))
        # ### FIX 2: catch RuntimeError (works on all JAX versions) instead of
        # jax.errors.JaxRuntimeError (only exists in newer JAX)
        except (RuntimeError, MemoryError) as e:
            print(f"[Sampling] vmap failed at chunk_size={current_chunk}, "
                  f"falling back to sequential. Error: {type(e).__name__}")
            for si in range(current_chunk):
                # ### FIX 1: pass sm_params here too
                result_i = sm_apply(
                    sm_params, jax.random.PRNGKey(si),
                    chunk_singles[si], chunk_pairs[si], batch)
                all_coords_list.append(np.array(result_i['final_atom_positions'])[None])
                if 'plddt_logits' in result_i:
                    probs = jax.nn.softmax(result_i['plddt_logits'], axis=-1)
                    n_bins = probs.shape[-1]
                    bins = jnp.arange(0.5, n_bins, 1.0) / n_bins * 100.0
                    score = jnp.sum(probs * bins[None, :], axis=-1)
                    all_plddt_list.append(np.array(score)[None])
                if 'pae_logits' in result_i:
                    probs = jax.nn.softmax(result_i['pae_logits'], axis=-1)
                    n_pae_bins = probs.shape[-1]
                    bins = jnp.arange(0.5, n_pae_bins) * 0.5
                    score = jnp.sum(probs * bins[None, None, :], axis=-1)
                    all_pae_list.append(np.array(score)[None])
        n_processed += current_chunk
        print(f"[Sampling] Processed {n_processed}/{n_samples} samples")
    # ── Phase E: Distribution Collection ──
    all_coords = np.concatenate(all_coords_list, axis=0)
    output = {'all_coords': all_coords, 'sampling_freq': sampling_freq}
    if all_plddt_list:
        all_plddt = np.concatenate(all_plddt_list, axis=0)
        output['all_plddt'] = all_plddt
        output['plddt_mean'] = np.mean(all_plddt, axis=0)
        output['plddt_std'] = np.std(all_plddt, axis=0)
    if all_pae_list:
        all_pae = np.concatenate(all_pae_list, axis=0)
        output['all_pae'] = all_pae
        output['pae_mean'] = np.mean(all_pae, axis=0)
    # Coordinate RMSF: per-residue CA position standard deviation
    from ig_pipeline import CA_IDX
    all_ca = all_coords[:, :, CA_IDX, :]  # [n_samples, N_res, 3]
    ca_mean = np.mean(all_ca, axis=0)     # [N_res, 3]
    ca_dev = all_ca - ca_mean[None, :, :] # [n_samples, N_res, 3]
    coord_rmsf = np.sqrt(np.mean(np.sum(ca_dev ** 2, axis=-1), axis=0))
    output['coord_rmsf'] = coord_rmsf
    return output

# ─────────────────────────────────────────────────────────────────────
# Output Writers
# ─────────────────────────────────────────────────────────────────────
def write_ensemble_pdb(
        all_coords: np.ndarray,
        aatype: np.ndarray,
        residue_index: np.ndarray,
        output_path: str,
        plddt_scores: Optional[np.ndarray] = None
):
    """Write multi-model PDB ensemble from sampling results.
    Args:
        all_coords: [n_samples, N_res, 37, 3] coordinates.
        aatype: [N_res] integer amino acid types (0-20).
        residue_index: [N_res] residue indices.
        output_path: path for the output PDB file.
        plddt_scores: optional [n_samples, N_res] pLDDT to use as B-factor.
    """
    from alphafold.common import residue_constants as rc
    restypes = rc.restypes + ['X']
    res_1to3 = lambda r: rc.restype_1to3.get(restypes[r], 'UNK')
    atom_types = rc.atom_types
    n_samples, n_res, _, _ = all_coords.shape
    lines = []
    for model_idx in range(n_samples):
        lines.append(f'MODEL     {model_idx + 1}')
        atom_index = 1
        for i in range(n_res):
            res_name = res_1to3(int(aatype[i]))
            for j, atom_name in enumerate(atom_types):
                # Only write atoms with nonzero coordinates
                pos = all_coords[model_idx, i, j]
                if np.abs(pos).sum() < 1e-6:
                    continue
                b_factor = 0.0
                if plddt_scores is not None:
                    b_factor = float(plddt_scores[model_idx, i])
                name = atom_name if len(atom_name) == 4 else f' {atom_name}'
                line = (f'ATOM  {atom_index:>5} {name:<4} '
                        f'{res_name:>3} A{int(residue_index[i]):>4}    '
                        f'{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}'
                        f'{1.00:>6.2f}{b_factor:>6.2f}          '
                        f'{atom_name[0]:>2}  ')
                lines.append(line)
                atom_index += 1
        lines.append('ENDMDL')
    lines.append('END')
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"[Sampling] Wrote ensemble PDB with {n_samples} models → {output_path}")
def save_sampling_results(
        sampling_output: Dict[str, np.ndarray],
        prefix: str
):
    """Save all sampling results as .npy / .npz files.
    Args:
        sampling_output: dict from run_sampling_loop.
        prefix: output file prefix.
    """
    # Comprehensive archive
    np.savez_compressed(
        f'{prefix}_sampling_results.npz',
        **{k: v for k, v in sampling_output.items()
           if isinstance(v, np.ndarray)})
    # Individual key files for convenience
    for key in ['plddt_mean', 'plddt_std', 'coord_rmsf', 'pae_mean', 'sampling_freq']:
        if key in sampling_output:
            np.save(f'{prefix}_{key}.npy', sampling_output[key])
    print(f"[Sampling] Saved sampling results → {prefix}_sampling_results.npz")


def _squeeze_batch_ensemble(batch: dict) -> dict:
    """Strip the leading num_ensemble dimension from processed_feature_dict.
    AlphaFold's data pipeline produces features with shape [num_ensemble, ...].
    During normal forward passes AlphaFoldIteration slices/averages this away
    before features reach the structure module.  Our standalone SM call bypasses
    that, so we must squeeze it ourselves.
    Takes the first ensemble copy ([0]) for every array-like value whose leading
    dim matches the detected num_ensemble.  Scalars and non-array values are
    passed through unchanged.
    Args:
        batch: raw processed_feature_dict from model_runner.process_features().
    Returns:
        New dict with every tensor reduced to [N_res, ...] (no ensemble dim).
    """
    import numpy as np
    import jax.numpy as jnp
    # ── Detect ensemble size from 'aatype' (guaranteed present, shape=[E, N_res]) ──
    aatype = batch.get('aatype', None)
    if aatype is None or not hasattr(aatype, 'shape') or len(aatype.shape) < 2:
        # Already squeezed or unexpected layout — return as-is
        return batch
    n_ens = aatype.shape[0]
    squeezed = {}
    for k, v in batch.items():
        # Only squeeze array-like values whose first dim == num_ensemble
        if hasattr(v, 'shape') and len(v.shape) >= 2 and v.shape[0] == n_ens:
            squeezed[k] = v[0]
        else:
            squeezed[k] = v
    return squeezed