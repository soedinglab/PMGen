"""
predict_utils_ig.py — Drop-in additions/replacements for predict_utils.py
=========================================================================
Contains the updated predict_structure() and run_alphafold_prediction()
that integrate Tasks 1-4 of the IG pipeline expansion.
Import this module alongside the existing predict_utils, or merge
these functions into predict_utils.py directly.
"""
import os
import json
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from timeit import default_timer as timer
from typing import Optional, Dict, Any, List
from collections import OrderedDict
from alphafold.common import protein, residue_constants
from alphafold.data import pipeline
from ig_pipeline import (
    parse_structure_input,
    generate_mask_from_structure,
    apply_ig_mask,
    apply_template_mask,
    create_template_features_from_structure,
    initial_guess_features_v2,
    initial_guess_features_legacy,
    legacy_anchors_to_mask,
    TOKEN_MASK,
)
from sampling_utils import (
    run_sampling_loop,
    write_ensemble_pdb,
    save_sampling_results,
)
# ─────────────────────────────────────────────────────────────────────
# Updated initial_guess_features (unified entry point)
# ─────────────────────────────────────────────────────────────────────
def initial_guess_features_unified(
        # New-style inputs (Task 1)
        ig_pdb: Optional[str] = None,
        ig_coords_npy: Optional[str] = None,
        ig_sequence: Optional[str] = None,
        ig_chain_ids_npy: Optional[str] = None,
        ig_residue_indices_npy: Optional[str] = None,
        ig_use_only_CA: bool = False,
        ig_use_CA_CB: bool = False,
        ig_mask: Optional[np.ndarray] = None,
        # Legacy inputs (backward compat)
        query_seq: Optional[str] = None,
        template_pdb_path: Optional[str] = None,
        template_sequence: Optional[str] = None,
        aln=None,
        anchors: Optional[List[int]] = None,
        peptide_seq: Optional[str] = None,
) -> Dict[str, Any]:
    """Unified initial guess feature builder supporting both new and legacy modes.
    Returns dict with:
      'initial_guess': jnp.ndarray [N_res, 37, 3]
      'structure_output': dict from parse_structure_input (new mode only)
      'mask_array': np.ndarray [N_res] if masking was applied
    Args:
        ig_pdb: PDB path for new-style IG input.
        ig_coords_npy: path to numpy coordinate array.
        ig_sequence: sequence string (for array mode).
        ig_chain_ids_npy: path to chain IDs array.
        ig_residue_indices_npy: path to residue indices array.
        ig_use_only_CA: CA-only mode flag.
        ig_use_CA_CB: CA+CB mode flag.
        ig_mask: explicit mask array.
        query_seq: legacy query sequence.
        template_pdb_path: legacy template PDB.
        template_sequence: legacy template sequence.
        aln: legacy alignment tuple.
        anchors: legacy anchor positions.
        peptide_seq: legacy peptide sequence.
    Returns:
        dict with 'initial_guess' and metadata.
    """
    result = {}
    # ── New-style path: PDB or numpy arrays ──
    if ig_pdb is not None or ig_coords_npy is not None:
        # Build coordinate arrays
        coords = None
        chain_ids = None
        residue_indices = None
        sequence = ig_sequence
        if ig_coords_npy is not None:
            coords = np.load(ig_coords_npy)
            assert ig_sequence is not None, "--ig_sequence required with --ig_coords_npy"
            assert ig_chain_ids_npy is not None, "--ig_chain_ids_npy required"
            assert ig_residue_indices_npy is not None, "--ig_residue_indices_npy required"
            chain_ids = np.load(ig_chain_ids_npy)
            residue_indices = np.load(ig_residue_indices_npy)
        structure_output = parse_structure_input(
            pdb_path=ig_pdb,
            coords=coords,
            sequence=sequence,
            chain_ids=chain_ids,
            residue_indices=residue_indices,
            use_only_CA=ig_use_only_CA,
            use_CA_CB=ig_use_CA_CB)
        result['structure_output'] = structure_output
        result['initial_guess'] = initial_guess_features_v2(
            structure_output, mask_array=ig_mask)
        if ig_mask is not None:
            result['mask_array'] = ig_mask
    # ── Legacy path: template PDB + alignment ──
    elif template_pdb_path is not None:
        result['initial_guess'] = initial_guess_features_legacy(
            query_seq, template_pdb_path, template_sequence,
            aln, anchors, peptide_seq, ig_mask=ig_mask)
        if ig_mask is not None:
            result['mask_array'] = ig_mask
        elif anchors is not None and peptide_seq is not None:
            n_res = result['initial_guess'].shape[0]
            result['mask_array'] = legacy_anchors_to_mask(
                n_res, len(peptide_seq), anchors)
    else:
        raise ValueError("No IG input provided. Supply --ig_pdb, --ig_coords_npy, "
                         "or legacy template_pdb_path.")
    return result
# ─────────────────────────────────────────────────────────────────────
# Updated predict_structure with template masking + sampling
# ─────────────────────────────────────────────────────────────────────
def predict_structure_v2(
        prefix: str,
        feature_dict: dict,
        model_runners: dict,
        random_seed: int = 0,
        crop_size: Optional[int] = None,
        dump_pdbs: bool = True,
        dump_metrics: bool = True,
        # IG inputs (new-style)
        ig_result: Optional[Dict] = None,
        no_initial_guess: bool = False,
        return_all_outputs: bool = False,
        # Template masking (Task 3)
        evo_template_mask: Optional[np.ndarray] = None,
        # Sampling (Task 4)
        sampling_mode: bool = False,
        n_times_sampling: int = 100,
        radius: float = 8.0,
        sampling_fraction_ig: float = 0.5,
        sampling_fraction_evo: float = 0.3,
        base_seed: int = 42,
        sampling_dropout_rate: float = 0.1,
):
    """Predict structure with full IG pipeline support (Tasks 1-4).
    Args:
        prefix: output file prefix.
        feature_dict: AF2 feature dictionary.
        model_runners: dict of model name → RunModel instances.
        random_seed: random seed for feature processing.
        crop_size: maximum residue count.
        dump_pdbs: write PDB files.
        dump_metrics: write metric .npy files.
        ig_result: dict from initial_guess_features_unified().
        no_initial_guess: skip IG entirely.
        return_all_outputs: save representations etc.
        evo_template_mask: [N_res] mask for template features (Task 3).
        sampling_mode: enable Task 4 sampling pipeline.
        n_times_sampling: number of SM samples.
        radius: sampling radius in Angstroms.
        sampling_fraction_ig: fraction for single repr.
        sampling_fraction_evo: fraction for pair repr.
        base_seed: RNG seed for sampling.
    Returns:
        all_metrics dict keyed by model name.
    """
    # ── Task 3: Apply template mask before feature processing ──
    if evo_template_mask is not None:
        feature_dict = dict(feature_dict)  # shallow copy to avoid mutation
        apply_template_mask(feature_dict, evo_template_mask)
    unrelaxed_pdb_lines = []
    model_names = []
    metric_tags = 'plddt ptm predicted_aligned_error'.split()
    if return_all_outputs or sampling_mode:
        metric_tags = 'plddt ptm predicted_aligned_error representations'.split()
    all_metrics = {}
    metrics = {}
    for model_name, model_runner in model_runners.items():
        start = timer()
        print(f"[IG Pipeline] Running {model_name}")
        # Process features
        processed_feature_dict = model_runner.process_features(
            feature_dict, random_seed=random_seed)
        # ── Run prediction ──
        initial_guess = None
        if not no_initial_guess and ig_result is not None:
            initial_guess = ig_result.get('initial_guess', None)
        prediction_result = model_runner.predict(
            processed_feature_dict,
            initial_guess=initial_guess,
            return_representations=(return_all_outputs or sampling_mode))
        # ── Task 4: Sampling Pipeline ──
        if sampling_mode and ig_result is not None:
            mask_array = ig_result.get('mask_array', None)
            structure_output = ig_result.get('structure_output', None)
            if mask_array is not None and 'representations' in prediction_result:
                cached_single = prediction_result['representations']['single']
                cached_pair = prediction_result['representations']['pair']
                # Get IG positions for radius computation
                if structure_output is not None:
                    ig_positions = structure_output['all_positions']
                else:
                    # Fallback: use predicted positions
                    ig_positions = np.array(prediction_result['structure_module']['final_atom_positions'])
                print(f"[IG Pipeline] Starting sampling: {n_times_sampling} samples, "
                      f"radius={radius}Å")
                # Pad mask_array to match ig_positions if needed (AF pads to crop_size)
                if mask_array.shape[0] < ig_positions.shape[0]:
                    pad_len = ig_positions.shape[0] - mask_array.shape[0]
                    mask_array = np.concatenate([
                        mask_array,
                        np.ones(pad_len, dtype=np.int32)  # +1 = stable (never sampled)
                    ])
                sampling_output = run_sampling_loop(
                    params=model_runner.params,
                    config=model_runner.config,
                    cached_single=cached_single,
                    cached_pair=cached_pair,
                    batch=processed_feature_dict,
                    mask_array=mask_array,
                    ig_positions=ig_positions,
                    radius=radius,
                    n_samples=n_times_sampling,
                    sampling_fraction_ig=sampling_fraction_ig,
                    sampling_fraction_evo=sampling_fraction_evo,
                    base_seed=base_seed,
                    sampling_dropout_rate=sampling_dropout_rate)
                # Save sampling results
                sampling_prefix = f'{prefix}_{model_name}'
                save_sampling_results(sampling_output, sampling_prefix)
                # Write ensemble PDB
                if dump_pdbs:
                    aatype = np.array(processed_feature_dict['aatype'][0])
                    res_idx = np.array(processed_feature_dict['residue_index'][0])
                    plddt = sampling_output.get('all_plddt', None)
                    write_ensemble_pdb(
                        sampling_output['all_coords'],
                        aatype, res_idx,
                        f'{sampling_prefix}_ensemble.pdb',
                        plddt_scores=plddt)
                prediction_result['sampling'] = sampling_output
            else:
                print("[IG Pipeline] WARNING: sampling_mode requires mask_array "
                      "and return_representations. Skipping sampling.")
        # ── Standard output processing ──
        unrelaxed_protein = protein.from_prediction(
            processed_feature_dict, prediction_result)
        unrelaxed_pdb_lines.append(protein.to_pdb(unrelaxed_protein))
        model_names.append(model_name)
        all_metrics[model_name] = {}
        for tag in metric_tags:
            result = prediction_result.get(tag, None)
            metrics.setdefault(tag, []).append(result)
            if result is not None:
                all_metrics[model_name][tag] = result
        elapsed = timer() - start
        plddt_mean = np.mean(prediction_result.get('plddt', [0]))
        print(f"[IG Pipeline] {model_name} pLDDT: {plddt_mean:.2f} Time: {elapsed:.1f}s")
    # ── Rank models by pLDDT and write outputs ──
    plddts = metrics.get('plddt', [])
    if plddts:
        plddt_arr = np.array([p if p is not None else np.zeros(1) for p in plddts])
        lddt_rank = np.mean(plddt_arr, axis=-1).argsort()[::-1]
        for n, r in enumerate(lddt_rank):
            print(f"model_{n+1} mean_pLDDT={np.mean(plddt_arr[r]):.2f}")
            if dump_pdbs:
                pdb_path = f'{prefix}_model_{n+1}_{model_names[r]}.pdb'
                with open(pdb_path, 'w') as f:
                    f.write(unrelaxed_pdb_lines[r])
            if dump_metrics:
                metrics_prefix = f'{prefix}_model_{n+1}_{model_names[r]}'
                for tag in metric_tags:
                    m = metrics[tag][r] if r < len(metrics.get(tag, [])) else None
                    if m is not None:
                        if tag != 'representations':
                            np.save(f'{metrics_prefix}_{tag}.npy', m)
                        else:
                            with open(f'{metrics_prefix}_{tag}.pkl', 'wb') as f:
                                pickle.dump(m, f)
    return all_metrics
# ─────────────────────────────────────────────────────────────────────
# Updated run_alphafold_prediction
# ─────────────────────────────────────────────────────────────────────
def run_alphafold_prediction_v2(
        query_sequence: str,
        msa: list,
        deletion_matrix: list,
        chainbreak_sequence: str,
        template_features: dict,
        model_runners: dict,
        out_prefix: str,
        crop_size: Optional[int] = None,
        dump_pdbs: bool = True,
        dump_metrics: bool = True,
        # IG options
        ig_result: Optional[Dict] = None,
        no_initial_guess: bool = False,
        return_all_outputs: bool = False,
        # Template masking
        evo_template_mask: Optional[np.ndarray] = None,
        # Sampling
        sampling_mode: bool = False,
        n_times_sampling: int = 100,
        radius: float = 8.0,
        sampling_fraction_ig: float = 0.5,
        sampling_fraction_evo: float = 0.3,
        base_seed: int = 42,
        sampling_dropout_rate: float = 0.1,
):
    """Top-level prediction function with full IG pipeline support.
    Builds the AF2 feature dict, applies chain breaks, and delegates
    to predict_structure_v2 for the actual model runs.
    Args:
        query_sequence: target amino acid sequence.
        msa: list of MSA sequences.
        deletion_matrix: deletion matrix.
        chainbreak_sequence: '/' separated chain sequence.
        template_features: compiled template feature dict.
        model_runners: dict of RunModel instances.
        out_prefix: output file prefix.
        crop_size: max residue count for padding.
        dump_pdbs: write PDB outputs.
        dump_metrics: write metric files.
        ig_result: dict from initial_guess_features_unified.
        no_initial_guess: skip IG.
        return_all_outputs: save all outputs.
        evo_template_mask: template mask array (Task 3).
        sampling_mode: enable sampling (Task 4).
        n_times_sampling: number of samples.
        radius: sampling radius.
        sampling_fraction_ig: single repr fraction.
        sampling_fraction_evo: pair repr fraction.
        base_seed: RNG seed.
    Returns:
        all_metrics dict.
    """
    # Build standard AF2 feature dict
    feature_dict = {
        **pipeline.make_sequence_features(
            sequence=query_sequence, description="none",
            num_res=len(query_sequence)),
        **pipeline.make_msa_features(
            msas=[msa], deletion_matrices=[deletion_matrix]),
        **template_features
    }
    # Apply chain break offsets
    Ls = [len(split) for split in chainbreak_sequence.split('/')]
    idx_res = feature_dict['residue_index']
    L_prev = 0
    for L_i in Ls[:-1]:
        idx_res[L_prev + L_i:] += 200
        L_prev += L_i
    feature_dict['residue_index'] = idx_res
    # Delegate to predict_structure_v2
    all_metrics = predict_structure_v2(
        prefix=out_prefix,
        feature_dict=feature_dict,
        model_runners=model_runners,
        crop_size=crop_size,
        dump_pdbs=dump_pdbs,
        dump_metrics=dump_metrics,
        ig_result=ig_result,
        no_initial_guess=no_initial_guess,
        return_all_outputs=return_all_outputs,
        evo_template_mask=evo_template_mask,
        sampling_mode=sampling_mode,
        n_times_sampling=n_times_sampling,
        radius=radius,
        sampling_fraction_ig=sampling_fraction_ig,
        sampling_fraction_evo=sampling_fraction_evo,
        base_seed=base_seed,
        sampling_dropout_rate=sampling_dropout_rate)
    return all_metrics
