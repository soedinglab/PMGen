#!/usr/bin/env python3
"""
Extract a single model from a multi-model PDB ensemble file.

Usage:
    python extract_model.py ensemble.pdb 5 output_dir/
"""

import sys
import os


def extract_model(input_pdb, model_number, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    header_lines = []
    model_lines = []
    in_target = False
    found = False
    header_done = False

    with open(input_pdb, 'r') as f:
        for line in f:
            if not header_done:
                if line.startswith("MODEL"):
                    header_done = True
                else:
                    header_lines.append(line)
                    continue

            if line.startswith("MODEL"):
                current = int(line.split()[1])
                in_target = (current == model_number)
                if in_target:
                    found = True

            if line.startswith("ENDMDL") and in_target:
                model_lines.append(line)
                break

            if in_target:
                model_lines.append(line)

    if not found:
        sys.exit(f"Model {model_number} not found in {input_pdb}")

    basename = os.path.splitext(os.path.basename(input_pdb))[0]
    output_path = os.path.join(output_dir, f"{basename}_model_{model_number}.pdb")

    with open(output_path, 'w') as f:
        f.writelines(header_lines + model_lines + ["END\n"])

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("Usage: python extract_model.py <input.pdb> <model_number> <output_dir>")

    extract_model(sys.argv[1], int(sys.argv[2]), sys.argv[3])