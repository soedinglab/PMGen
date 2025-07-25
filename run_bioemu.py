# Run on bioemu env
from bioemu.sample import main as sample
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="BioEmu sampling script")

    parser.add_argument('--sequence', type=str, required=True, help='MHC+peptide sequence with no separator. For MHC-II, Alpha+Beta+peptide')
    parser.add_argument('--id', type=str, required=True, help='PMGen id given to sequence input.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory to save the results')
    parser.add_argument('--cache_embeds_dir', required=True, type=str, help='Path to alphafold folder containing ids/*_representations.pkl')
    parser.add_argument( '--bioemu_num_samples', type=int, default=10, help='Sampling rounds in BioEmu. You might get a lower number of structures if --bioemu_filter_samples is active.')
    parser.add_argument('--bioemu_batch_size_100', type=int, default=10, help='Batch size for a sequence of length 100. Actual batch size '                                                                       'scales with the square of sequence length.')
    parser.add_argument('--bioemu_filter_samples', action='store_true', help='Filter out unphysical samples (e.g., long bond distances or steric clashes).')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    sample(sequence=args.sequence,
            id=args.id,
            num_samples=int(args.bioemu_num_samples),
            output_dir=args.output_dir,
            batch_size_100=int(args.bioemu_batch_size_100),
            cache_embeds_dir=args.cache_embeds_dir,
            filter_samples=args.bioemu_filter_samples)