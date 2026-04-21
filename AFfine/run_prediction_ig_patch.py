"""
run_prediction_ig_patch.py — CLI patch for run_prediction.py
=============================================================
Contains the new argparse arguments and the updated main-loop logic
that integrates Tasks 1-4 into run_prediction.py.
Merge these into the existing run_prediction.py.
Instructions:
  1. Add the argparse block below AFTER the existing parser.add_argument() calls.
  2. Replace the per-target loop body with the updated version below.
"""
import json
import os
import pandas as pd
import numpy as np
# ═════════════════════════════════════════════════════════════════════
# SECTION 1: New CLI arguments (add after existing parser.add_argument calls)
# ═════════════════════════════════════════════════════════════════════
def _has_valid_template_pdb_dict(targetl) -> bool:
    """Check if targetl row has a valid template_pdb_dict pointing to a file.
    Handles: missing column, NaN value, non-existent file.
    Args:
        targetl: pandas row (from iterrows or itertuples).
    Returns:
        True if template_pdb_dict points to a valid, existing file.
    """
    val = getattr(targetl, 'template_pdb_dict', None)
    if val is None:
        return False
    if isinstance(val, float) and pd.isna(val):
        return False
    return os.path.isfile(str(val))

def add_ig_pipeline_args(parser):
    """Register all IG pipeline CLI arguments on the given argparse parser.
    Call this right after creating the parser and before parse_args().
    Args:
        parser: argparse.ArgumentParser instance.
    """
    # ── Task 1: Flexible coordinate input ──
    ig_group = parser.add_argument_group('IG Pipeline — Task 1: Structure Input')
    ig_group.add_argument('--ig_pdb', type=str, default=None,
        help='PDB file for initial guess input (new-style). '
             'Replaces legacy template_pdb_dict for IG coordinates.')
    ig_group.add_argument('--ig_coords_npy', type=str, default=None,
        help='Numpy array file for IG coordinates. Shape: [N,3] CA-only, '
             '[N,2,3] CA+CB, or [N,37,3] full atoms.')
    ig_group.add_argument('--ig_use_only_CA', action='store_true', default=False,
        help='Extract/use only CA atoms from IG structure. '
             'CB estimated for pseudo-beta computation.')
    ig_group.add_argument('--ig_use_CA_CB', action='store_true', default=False,
        help='Extract/use only CA and CB atoms from IG structure.')
    ig_group.add_argument('--ig_sequence', type=str, default=None,
        help='Amino acid sequence (required for numpy array mode).')
    ig_group.add_argument('--ig_chain_ids_npy', type=str, default=None,
        help='Chain ID numpy array [N_res] (required for array mode).')
    ig_group.add_argument('--ig_residue_indices_npy', type=str, default=None,
        help='Residue index numpy array [N_res] (required for array mode).')
    ig_group.add_argument('--template_from_ig', action='store_true', default=False,
        help='Also inject IG structure as an evoformer template.')
    # ── Task 2: General IG masking ──
    mask_group = parser.add_argument_group('IG Pipeline — Task 2: Masking')
    mask_group.add_argument('--ig_mask_npy', type=str, default=None,
        help='Path to pre-computed mask array .npy file [N_res]. '
             'Tokens: 0=keep, -1=mask, -2=center, +1=stable.')
    mask_group.add_argument('--mask_residues', type=str, default=None,
        help='Residues to mask (token -1). Format: "B:5-9" or "B:150" or '
             '"150,151". Comma-separated for multiple specs.')
    mask_group.add_argument('--sampling_centers', type=str, default=None,
        help='Residues as sampling centers (token -2). Same format as '
             '--mask_residues.')
    mask_group.add_argument('--stable_residues', type=str, default=None,
        help='Residues that are never sampled (token +1). Same format. '
             '"A:*" for entire chain.')
    mask_group.add_argument('--auto_sampling_radius', type=float, default=None,
        help='Auto-detect sampling centers within this radius (Å) of '
             'masked residues. Requires --mask_residues.')
    # ── Task 3: Evoformer template masking ──
    tmpl_group = parser.add_argument_group('IG Pipeline — Task 3: Template Masking')
    tmpl_group.add_argument('--evo_template_mask', type=str, default=None,
        help='Template masking mode. "auto" reuses IG mask. '
             'Or path to separate mask .npy file.')
    # ── Task 4: Sampling pipeline ──
    samp_group = parser.add_argument_group('IG Pipeline — Task 4: Sampling')
    samp_group.add_argument('--sampling_mode', action='store_true', default=False,
        help='Enable stochastic sampling pipeline. Runs evoformer once, '
             'then structure module N times with masked representations.')
    samp_group.add_argument('--n_times_sampling', type=int, default=100,
        help='Number of structure module samples (default: 100).')
    samp_group.add_argument('--radius', type=float, default=8.0,
        help='Sampling radius in Å around sampling centers (default: 8.0).')
    samp_group.add_argument('--sampling_fraction_IG', type=float, default=0.5,
        help='Fraction of sampleable residues zeroed in single repr (default: 0.5).')
    samp_group.add_argument('--sampling_fraction_evo', type=float, default=0.3,
        help='Fraction of sampleable residues zeroed in pair repr (default: 0.3).')
    samp_group.add_argument('--sampling_seed', type=int, default=42,
        help='Base random seed for sampling (default: 42).')
    samp_group.add_argument('--sampling_dropout_rate', type=float, default=0.1,
        help='dropout rate on evoformer final layer on sampled and masked tokens.')
    return parser
# ═════════════════════════════════════════════════════════════════════
# SECTION 2: IG pipeline setup function (call once before the target loop)
# ═════════════════════════════════════════════════════════════════════
def setup_ig_pipeline(args):
    """Parse IG-related args and build reusable pipeline config.
    Call after parse_args(), before the target iteration loop.
    Args:
        args: parsed argparse namespace.
    Returns:
        ig_config: dict with pipeline settings, or None if no IG args set.
    """
    # Detect if any new-style IG args are provided
    has_new_ig = (args.ig_pdb is not None or args.ig_coords_npy is not None)
    has_masking = (args.mask_residues is not None or args.ig_mask_npy is not None)
    if not has_new_ig and not has_masking and not args.sampling_mode:
        return None  # No new IG pipeline features requested
    config = {
        'ig_pdb': args.ig_pdb,
        'ig_coords_npy': args.ig_coords_npy,
        'ig_use_only_CA': args.ig_use_only_CA,
        'ig_use_CA_CB': args.ig_use_CA_CB,
        'ig_sequence': args.ig_sequence,
        'ig_chain_ids_npy': args.ig_chain_ids_npy,
        'ig_residue_indices_npy': args.ig_residue_indices_npy,
        'template_from_ig': args.template_from_ig,
        'ig_mask_npy': args.ig_mask_npy,
        'mask_residues': args.mask_residues,
        'sampling_centers': args.sampling_centers,
        'stable_residues': args.stable_residues,
        'auto_sampling_radius': args.auto_sampling_radius,
        'evo_template_mask': args.evo_template_mask,
        'sampling_mode': args.sampling_mode,
        'n_times_sampling': args.n_times_sampling,
        'radius': args.radius,
        'sampling_fraction_ig': args.sampling_fraction_IG,
        'sampling_fraction_evo': args.sampling_fraction_evo,
        'base_seed': args.sampling_seed,
        'sampling_dropout_rate': args.sampling_dropout_rate,
    }
    return config
# ═════════════════════════════════════════════════════════════════════
# SECTION 3: Per-target IG processing (replaces body of target loop)
# ═════════════════════════════════════════════════════════════════════
def process_target_with_ig_pipeline(
        args,
        ig_config,
        targetl,
        query_sequence,
        query_chainseq,
        all_template_features,
        model_runners,
        outfile_prefix,
        crop_size,
        msa,
        deletion_matrix
):
    """Process a single target using the full IG pipeline.
    This replaces the inner body of the target iteration loop in
    run_prediction.py. Supports both legacy and new-style IG input.
    Args:
        args: parsed CLI args.
        ig_config: dict from setup_ig_pipeline (or None for legacy mode).
        targetl: single row from targets DataFrame.
        query_sequence: target sequence string.
        query_chainseq: chainbreak sequence string (e.g. "MKTL.../AVSW...").
        all_template_features: compiled template features dict.
        model_runners: model runner dict.
        outfile_prefix: output prefix string.
        crop_size: max residue count.
        msa: MSA sequence list.
        deletion_matrix: deletion matrix list.
    Returns:
        all_metrics dict from prediction.
    """
    import predict_utils
    from predict_utils_ig import (
        initial_guess_features_unified,
        run_alphafold_prediction_v2,
    )
    from ig_pipeline import (
        parse_structure_input,
        generate_mask_from_structure,
        create_template_features_from_structure,
    )
    ig_result = None
    evo_template_mask = None
    # ── New-style IG pipeline ──
    if ig_config is not None and (ig_config['ig_pdb'] or ig_config['ig_coords_npy']):
        print("[IG Pipeline] Using new-style IG input")
        # Step 1: Parse structure
        structure_output = parse_structure_input(
            pdb_path=ig_config['ig_pdb'],
            coords=np.load(ig_config['ig_coords_npy']) if ig_config['ig_coords_npy'] else None,
            sequence=ig_config['ig_sequence'],
            chain_ids=np.load(ig_config['ig_chain_ids_npy']) if ig_config['ig_chain_ids_npy'] else None,
            residue_indices=np.load(ig_config['ig_residue_indices_npy']) if ig_config['ig_residue_indices_npy'] else None,
            use_only_CA=ig_config['ig_use_only_CA'],
            use_CA_CB=ig_config['ig_use_CA_CB'])
        # Step 2: Generate mask array (Task 2)
        mask_array = generate_mask_from_structure(
            all_positions=structure_output['all_positions'],
            chain_ids=structure_output['chain_ids'],
            residue_indices=structure_output['residue_indices'],
            mask_residues=ig_config['mask_residues'],
            sampling_centers=ig_config['sampling_centers'],
            stable_residues=ig_config['stable_residues'],
            auto_sampling_radius=ig_config['auto_sampling_radius'],
            mask_array_path=ig_config['ig_mask_npy'])
        # Step 3: Build IG features
        ig_result = initial_guess_features_unified(
            ig_pdb=ig_config['ig_pdb'],
            ig_coords_npy=ig_config['ig_coords_npy'],
            ig_sequence=ig_config['ig_sequence'],
            ig_chain_ids_npy=ig_config['ig_chain_ids_npy'],
            ig_residue_indices_npy=ig_config['ig_residue_indices_npy'],
            ig_use_only_CA=ig_config['ig_use_only_CA'],
            ig_use_CA_CB=ig_config['ig_use_CA_CB'],
            ig_mask=mask_array)
        ig_result['mask_array'] = mask_array
        ig_result['structure_output'] = structure_output
        # Step 3b: Optionally inject IG as evoformer template (Task 1)
        if ig_config['template_from_ig']:
            ig_template_features = create_template_features_from_structure(
                structure_output['all_positions'],
                structure_output['all_positions_mask'],
                structure_output['sequence'])
            # Merge with existing template features
            from predict_utils import compile_template_features
            existing_list = []
            # Unstack existing templates if present
            n_existing = all_template_features.get(
                'template_all_atom_positions', np.zeros((0,))).shape[0]
            for ti in range(n_existing):
                t_feat = {k: v[ti] for k, v in all_template_features.items()
                          if isinstance(v, np.ndarray) and v.ndim > 1}
                existing_list.append(t_feat)
            # Add IG template (squeeze batch dim)
            ig_t = {k: v[0] for k, v in ig_template_features.items()
                    if isinstance(v, np.ndarray) and v.ndim > 1}
            existing_list.append(ig_t)
            if existing_list:
                all_template_features = compile_template_features(existing_list)
        # Step 4: Template masking (Task 3)
        if ig_config['evo_template_mask'] is not None:
            if ig_config['evo_template_mask'] == 'auto':
                # Reuse IG mask for templates
                evo_template_mask = mask_array
            elif os.path.isfile(ig_config['evo_template_mask']):
                evo_template_mask = np.load(ig_config['evo_template_mask'])
            else:
                print(f"[IG Pipeline] WARNING: --evo_template_mask "
                      f"'{ig_config['evo_template_mask']}' not recognized.")
        print(f"[IG Pipeline] mask_array tokens: "
              f"mask(-1)={np.sum(mask_array==-1)}, "
              f"center(-2)={np.sum(mask_array==-2)}, "
              f"stable(+1)={np.sum(mask_array==1)}, "
              f"default(0)={np.sum(mask_array==0)}")
    # ── Legacy IG pipeline (backward compat) ──
    elif not args.no_initial_guess and _has_valid_template_pdb_dict(targetl):
        print("[IG Pipeline] Using legacy template_pdb_dict IG input")
        template_pdb_dict_path = str(targetl.template_pdb_dict)
        with open(template_pdb_dict_path, 'r') as f:
            template_pdb_dict = json.load(f)
        template_pdb_list = template_pdb_dict['template_path']
        aln = (template_pdb_dict['aln_target'][0],
            template_pdb_dict['aln_template'][0])
        anchors = template_pdb_dict['target_anchors'][0]
        peptide_seq = template_pdb_dict['aln_P'][0][0].replace('-', '')
        # Below added for this patch
        mask_array_for_legacy = None
        if hasattr(args, 'pep_sampling') and args.pep_sampling is not None:
            # PMGen: build mask from TARGET sequence, not template
            chain_lens = [len(c) for c in query_chainseq.split('/')]
            pep_len = chain_lens[-1]
            pep_start = sum(chain_lens[:-1])
            total_len = sum(chain_lens)
            mask_array_for_legacy = np.ones(total_len, dtype=np.int32)  # +1 stable

            if args.pep_sampling == 'all':
                sample_idx = list(range(pep_len))
            elif args.pep_sampling == 'non_anchors':
                anchor_idx = set(a - 1 for a in anchors)  # 0-indexed within peptide
                sample_idx = [i for i in range(pep_len) if i not in anchor_idx]
            else:
                sample_idx = [int(x) - 1 for x in args.pep_sampling.split(',')]

            for i in sample_idx:
                mask_array_for_legacy[pep_start + i] = -2
            print(f"[IG Pipeline] PMGen pep_sampling: {len(sample_idx)} peptide "
                  f"positions as centers, {pep_start} MHC residues stable")
        elif ig_config is not None and (ig_config.get('mask_residues') or ig_config.get('ig_mask_npy')):
            # Original AlphaFoldGuesser path (template == target)
            struct_out = parse_structure_input(pdb_path=template_pdb_list[0])
            mask_array_for_legacy = generate_mask_from_structure(
                struct_out['all_positions'], struct_out['chain_ids'],
                struct_out['residue_indices'],
                mask_residues=ig_config['mask_residues'],
                auto_sampling_radius=ig_config['auto_sampling_radius'],
                stable_residues=ig_config['stable_residues'],
                mask_array_path=ig_config['ig_mask_npy'])
        # Above added for this patch
        ig_result = initial_guess_features_unified(
            query_seq=query_sequence,
            template_pdb_path=template_pdb_list[0],
            template_sequence=None,
            aln=aln,
            anchors=anchors,
            peptide_seq=peptide_seq,
            ig_mask=mask_array_for_legacy)
    elif not args.no_initial_guess:
        # No new-style IG, no legacy dict — warn and proceed without IG
        print("[IG Pipeline] WARNING: --no_initial_guess not set but no IG "
            "input found (no --ig_pdb, no template_pdb_dict). "
            "Proceeding without initial guess.")
    # ig_result stays None → run_alphafold_prediction_v2 handles gracefully
    # ── Run prediction (v2 with full pipeline) ──
    all_metrics = run_alphafold_prediction_v2(
        query_sequence=query_sequence,
        msa=msa,
        deletion_matrix=deletion_matrix,
        chainbreak_sequence=query_chainseq,
        template_features=all_template_features,
        model_runners=model_runners,
        out_prefix=outfile_prefix,
        crop_size=crop_size,
        dump_pdbs=not (args.no_pdbs if hasattr(args, 'no_pdbs') else False),
        dump_metrics=not (args.terse if hasattr(args, 'terse') else False),
        ig_result=ig_result,
        no_initial_guess=args.no_initial_guess,
        return_all_outputs=args.return_all_outputs,
        evo_template_mask=evo_template_mask,
        sampling_mode=ig_config['sampling_mode'] if ig_config else False,
        n_times_sampling=ig_config['n_times_sampling'] if ig_config else 100,
        radius=ig_config['radius'] if ig_config else 8.0,
        sampling_fraction_ig=ig_config['sampling_fraction_ig'] if ig_config else 0.5,
        sampling_fraction_evo=ig_config['sampling_fraction_evo'] if ig_config else 0.3,
        base_seed=ig_config['base_seed'] if ig_config else 42,
        sampling_dropout_rate=ig_config['sampling_dropout_rate'] if ig_config else 0.1)
    return all_metrics
# ═════════════════════════════════════════════════════════════════════
# SECTION 4: Example integration snippet (shows how to patch run_prediction.py)
# ═════════════════════════════════════════════════════════════════════
INTEGRATION_EXAMPLE = """
# ── In run_prediction.py, after creating parser: ──
from run_prediction_ig_patch import add_ig_pipeline_args, setup_ig_pipeline, process_target_with_ig_pipeline
parser = add_ig_pipeline_args(parser)
args = parser.parse_args()
# ... existing imports and model_runners setup ...
ig_config = setup_ig_pipeline(args)
# ── In the per-target loop, replace the prediction call: ──
for counter, targetl in targets.iterrows():
    # ... existing target parsing (alignfile, msa, etc.) ...
    all_metrics = process_target_with_ig_pipeline(
        args=args,
        ig_config=ig_config,
        targetl=targetl,
        query_sequence=query_sequence,
        query_chainseq=query_chainseq,
        all_template_features=all_template_features,
        model_runners=model_runners,
        outfile_prefix=outfile_prefix,
        crop_size=crop_size,
        msa=msa,
        deletion_matrix=deletion_matrix)
    # ... existing post-processing ...
"""
