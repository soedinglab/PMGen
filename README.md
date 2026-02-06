<table align="center" border="0" cellpadding="0" cellspacing="0">
  <tr>
    <td><h1>PMGen: From Structure Prediction to Peptide Generation</h1></td>
    <td><img src="PMGen_logo.png" width="200" alt="PMGen Logo"/></td>
  </tr>
</table>

<p align="center">
  <a href="https://colab.research.google.com/github/soedinglab/PMGen/blob/master/colab.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
  <a href="https://github.com/soedinglab/PMGen">
    <img src="https://img.shields.io/github/stars/soedinglab/PMGen?style=social" alt="GitHub Stars"/>
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-Apache_2.0-blue.svg" alt="License"/>  </a>
</p>

PMGen (Peptide MHC Generator) is a comprehensive pipeline for predicting peptide-MHC (pMHC) complex structures and designing optimized peptide sequences. 

## Key Features

- **Fast & accurate structure prediction** using AlphaFold with template engineering or initial guess mode
- **Peptide sequence design** with structure-aware optimization
- **MHC pseudo-sequence design** for customized allele engineering
- **Iterative peptide optimization** with binding prediction
- **Mutation screening** for systematic variant analysis
- **Batch processing** for multiple pMHC complexes

## Installation

### Requirements

- Python 3.8+
- Conda or Mamba
- Git
Optional
- Modeller (requires a license key - [get it here](https://salilab.org/modeller/registration.html))
- CUDA-enabled GPU (Required for faster Alphafold predictions)

### Setup


```bash
git clone https://github.com/soedinglab/PMGen.git
cd PMGen
bash -l install.sh
#or, for CPU only support run: bash -l install.sh --cpu
conda activate PMGen
```

You will be prompted to enter your Modeller license key. The script automatically:
- Creates the PMGen Conda environment
- Installs all dependencies
- Downloads AlphaFold parameters
- Clones PANDORA and ProteinMPNN

### Optional: Configure NetMHCpan (Recommended)

Install [NetMHCpan](https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/) and [NetMHCIIpan](https://services.healthtech.dtu.dk/services/NetMHCIIpan-4.3/), then edit `user_setting.py`:

```python
netmhcipan_path = '/path/to/netMHCpan'
netmhciipan_path = '/path/to/netMHCIIpan'
```

## Quick Start

### Prepare Input File

Create a tab-separated file (`input.tsv`) with your pMHC data:

```tsv
peptide	mhc_seq	mhc_type	anchors	id
GILGFVFTL	GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQRTDAPKTHMTHHAVSDHEATLRCWALSFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGQEQRYTCHVQHEGLPKPLTLRWE	1		sample1
KLGGALQAK	GSHSLKYFHTSVSRPGRGEPRFISVGYVDDTQFVRFDSDAASPRGEPRAPWVEQEGPEYWDRNTQIFKTNTQTYRENLRIALRYYNQSEAGSHIIQRMYGCDLGPDGRLLRGHDQYAYDGKDYIALNEDLRSWTAADTAAQITQRKWEAAREAEQRRAYLEGTCVEWLRRYLKNGNATLLRTDSPKTHMTHHPISDHEATLRCWALGFYPAEITLTWQRDGEDQTQDTELVETRPAGDRTFQKWAAVVVPSGEEQRYTCHVQHEGLPKPLTLRWE	1		sample2
```

**Columns:**
- `peptide`: Peptide sequence
- `mhc_seq`: MHC sequence (for MHC-II: Alpha/Beta separated by `/`)
- `mhc_type`: 1 for MHC-I, 2 for MHC-II
- `anchors`: Anchor positions (leave empty for prediction)
- `id`: Unique identifier

### Basic Structure Prediction (Recommended)

**Single-threaded mode with initial guess (fastest & most accurate):**

```bash
python run_PMGen.py \
  --mode wrapper \
  --run single \
  --df input.tsv \
  --output_dir output/ \
  --initial_guess
```

This is the **preferred mode** for most users. It uses:
- `--mode wrapper`: Works for one or more than one prediction per run.
- `--run single`: Sequential processing (unparallel)
- `--initial_guess`: Fast and more accurate AlphaFold mode without homology modeling (recommended)

## Common Use Cases

### 1. Structure Prediction with Multiple Models

```bash
python run_PMGen.py \
  --mode wrapper \
  --run single \
  --df input.tsv \
  --output_dir output/ \
  --initial_guess \
  --models model_1_ptm model_2_ptm model_3_ptm
```

### 2. Peptide Sequence Design

Generate optimized peptide variants:

```bash
python run_PMGen.py \
  --mode wrapper \
  --run single \
  --df input.tsv \
  --output_dir output/ \
  --initial_guess \
  --peptide_design \
  --num_sequences_peptide 50 \
  --binder_pred
```

### 3. MHC Pseudo-Sequence Design

Customize MHC binding groove residues:

```bash
python run_PMGen.py \
  --mode wrapper \
  --run single \
  --df input.tsv \
  --output_dir output/ \
  --initial_guess \
  --only_pseudo_sequence_design \
  --num_sequences_mhc 20
```

### 4. Iterative Peptide Optimization

Optimize peptides over multiple rounds:

```bash
python run_PMGen.py \
  --mode wrapper \
  --run single \
  --df input.tsv \
  --output_dir output/ \
  --initial_guess \
  --peptide_design \
  --binder_pred \
  --iterative_peptide_gen 3 \
  --fix_anchors
```

### 5. Mutation Screening

Systematically test point mutations:

```bash
python run_PMGen.py \
  --mode wrapper \
  --run single \
  --df input.tsv \
  --output_dir output/ \
  --initial_guess \
  --mutation_screen \
  --n_mutations 1
```

## Key Options

| Flag | Description |
|------|-------------|
| `--mode wrapper` | Batch processing mode (recommended) |
| `--run single` | Sequential processing (recommended) |
| `--initial_guess` | Fast AF mode without templates (recommended) |
| `--peptide_design` | Enable peptide sequence generation |
| `--only_pseudo_sequence_design` | Design MHC binding groove only |
| `--binder_pred` | Predict binding affinity (requires NetMHCpan) |
| `--fix_anchors` | Keep anchor positions fixed during design |
| `--iterative_peptide_gen N` | Run N rounds of optimization |
| `--mutation_screen` | Systematic mutation analysis |
| `--num_templates` | Number of structural templates (default: 4) |
| `--num_recycles` | AlphaFold recycles (default: 3) |

## Output Structure

```
output/
├── pandora/              # Template structures
├── alphafold/            # Predicted pMHC structures
├── proteinmpnn/          # Designed sequences
│   └── {id}/
│       ├── peptide_design/
│       └── only_pseudo_sequence_design/
└── best_structures/      # Top-ranked models (if --best_structures used)
```

## Citation

```
@article{asgary2025pmgen,
  author = {Asgary, Amir H. and Amirreza and others},
  title = {PMGen: From Peptide-MHC Prediction to Neoantigen Generation},
  journal = {bioRxiv},
  year = {2025},
  month = {11},
  date = {2025-11-14},
  doi = {10.1101/2025.11.14.688404},
  url = {https://doi.org/10.1101/2025.11.14.688404},
  note = {Preprint}
}
```


PMGen uses the following tools and papers:
- **PANDORA**: [Antunes et al., Front. Immunol. 2022](https://www.frontiersin.org/articles/10.3389/fimmu.2022.878762/full)
- **AlphaFold**: [Jumper et al., Nature 2021](https://www.nature.com/articles/s41586-021-03819-2)
- **AFfine**: [Bradley et al., PNAS 2023](https://www.pnas.org/doi/abs/10.1073/pnas.2216697120)
- **ProteinMPNN**: [Dauparas et al., Science 2022](https://www.science.org/doi/10.1126/science.add2187)

## Support

For issues or questions, please open an issue on [GitHub](https://github.com/soedinglab/PMGen).
