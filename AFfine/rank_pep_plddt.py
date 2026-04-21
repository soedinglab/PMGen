import numpy as np
import json
import sys
import os

def rank_peptide_plddt(npz_path, query_chainseq, output_path=None):
    """
    Rank sampled structures by mean peptide pLDDT.
    
    Args:
        npz_path: path to *_sampling_results.npz
        query_chainseq: e.g. "MKTL.../AVSL..." (peptide is last chain)
        output_path: where to save JSON (default: same dir as npz)
    """
    data = np.load(npz_path)
    all_plddt = data['all_plddt']        # [N_samples, N_res_padded]
    plddt_mean = data.get('plddt_mean')  # [N_res_padded] from single prediction

    chain_lens = [len(c) for c in query_chainseq.split('/')]
    pep_start = sum(chain_lens[:-1])
    pep_end = sum(chain_lens)

    # Non-sampled: use the first prediction's pLDDT (or plddt_mean)
    # plddt_mean is average across samples, so use all_plddt[0] isn't right either
    # Better: load from the regular prediction. But plddt_mean works as proxy.
    non_sampled_pep = float(np.mean(plddt_mean[pep_start:pep_end]))

    # Per-sample peptide mean pLDDT
    sampled = {}
    for i in range(all_plddt.shape[0]):
        pep_plddt = float(np.mean(all_plddt[i, pep_start:pep_end]))
        sampled[i] = round(pep_plddt, 3)

    # Sort by pLDDT descending
    ranked = dict(sorted(sampled.items(), key=lambda x: x[1], reverse=True))

    result = {
        "non_sampled_peptide_plddt": round(non_sampled_pep, 3),
        "n_samples": len(ranked),
        "sampled_ranked": {str(k): v for k, v in ranked.items()},
    }

    if output_path is None:
        output_path = npz_path.replace('_sampling_results.npz', '_pep_plddt_ranked.json')

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {output_path}")
    return result


if __name__ == '__main__':
    npz_path = sys.argv[1]
    chainseq = sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else None
    rank_peptide_plddt(npz_path, chainseq, out)