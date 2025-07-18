# PMGen

**PMGen** (Peptide-MHC Predictive, Modeling and Generative) is a powerful and flexible framework 
for Peptide-MHC (pMHC) complex modeling, binding prediction, and neoantigen design. It integrates 
cutting-edge tools such as **PANDORA** for template generation, **AlphaFold (via AFfine)** for 
structural prediction, and **ProteinMPNN** for sequence design, enabling researchers to model
pMHC complexes, predict binding interactions, and engineer novel peptide or MHC 
sequences.

## Features

- **Structural Modeling**: Generate accurate 3D models of pMHC complexes using PANDORA and AlphaFold.
- **Binding Prediction**: Predict peptide-MHC binding affinities and anchor positions.
- **Protein Design**: Design new peptide sequences, full MHC sequences, or MHC pseudo-sequences using ProteinMPNN.
- **Batch Processing**: Process multiple pMHC complexes efficiently with parallel execution support.
- **Customizable Workflow**: Options to skip steps (e.g., AlphaFold) or focus on specific tasks (e.g., ProteinMPNN only).

## Installation

**Requirements:**
```
    Python 3.8+
    Conda or Mamba
    CUDA-enabled GPU (required for AlphaFold)
    Modeller (requires a license key)
    Git
```

Follow these steps to set up PMGen on your system:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/soedinglab/PMGen.git
   cd PMGen
   bash -l install.sh
   conda activate PMGen
   ```
 You will be prompted to enter your Modeller license key (required for PANDORA).
 The script creates a Conda environment (PMGen), installs dependencies, clones PANDORA and ProteinMPNN, and sets up necessary configurations.

    
1. **Customize setting file (Optional but Recommended)**:
Install netMHCpan's latest version.
Edit `user_setting.py` to adjust netMHCpan and netMHCIIpan installation paths.



## Usage

PMGen operates in two primary modes:

    modeling: For modeling a single pMHC complex.
    wrapper: For batch processing multiple complexes from a TSV file.

Run the tool with the main script:
bash
`python run_PMGen.py [options]`
Command-Line Options
General Options

    --run [parallel|single]: Execution mode (default: parallel). Note: parallel is only for --mode wrapper.
    --mode [wrapper|modeling]: Operation mode (default: wrapper).
    --output_dir <path>: Directory for output files (required).

Modeling Mode Options

    --peptide <sequence>: Peptide sequence (e.g., SIINFEKL).
    --mhc_seq <sequence>: MHC sequence (MHC-I: single chain; MHC-II: chain_A/chain_B).
    --mhc_fasta <file>: FASTA file with MHC sequence(s).
    --mhc_allele <allele>: MHC allele (e.g., HLA-A*02:01, optional).
    --mhc_type [1|2]: MHC class (1 for MHC-I, 2 for MHC-II).
    --id <identifier>: Unique identifier for the run.
    --anchors <positions>: Anchor positions (e.g., [2,9] for MHC-I).
    --predict_anchor: Predict anchor positions (recommended if --anchors is omitted).

Wrapper Mode Options

    --df <file.tsv>: TSV file with pMHC data (columns: peptide, mhc_seq, mhc_type, anchors, id).

Protein Design Options

    --peptide_design: Enable peptide sequence design.
    --only_pseudo_sequence_design: Design MHC pseudo-sequence only.
    --mhc_design: Design the entire MHC sequence.
    --num_sequences_peptide <int>: Number of peptide sequences to generate (default: 10).
    --num_sequences_mhc <int>: Number of MHC sequences to generate (default: 10).
    --sampling_temp <float>: ProteinMPNN sampling temperature (default: 0.1).
    --batch_size <int>: ProteinMPNN batch size (default: 1).
    --hot_spot_thr <float>: Distance threshold for MHC hot-spots (default: 6.0).

Advanced Options

    --num_templates <int>: Number of templates (default: 4).
    --num_recycles <int>: AlphaFold recycles (default: 3).
    --models <list>: AlphaFold models (default: ['model_2_ptm']).
    --alphafold_param_folder <path>: Path to AlphaFold parameters.
    --fine_tuned_model_path <path>: Path to fine-tuned AlphaFold model.
    --max_ram <int>: Max RAM per job in GB (default: 3, for parallel mode).
    --max_cores <int>: Max CPU cores (default: 4, for parallel mode).
    --no_alphafold: Skip AlphaFold modeling.
    --only_protein_mpnn: Run ProteinMPNN only on existing structures.

## Examples

Here are several practical examples to demonstrate how to use PMGen:

Start by setting up your variables:
```
PEPTIDE='NLVPMVATV'
# MHC-I
MHC_SEQ='AGSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGCYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMCAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQRTDA'
HLAALLEL='HLA-B*5301'
DF='data/example/smaller_wrapper_exmaple.tsv'
# MHC-II
MHC_ALPHA_CHAIN_SEQ='IKEEHVIIQAEFYLNPDQSGEFMFDFDGDEIFHVDMAKKETVWRLEEFGRFASFEAQGALANIAVDKANLEIMTKRSNYTPITNVPPEVTVLTNS'
MHC_BETA_CHAIN_SEQ='DTRPRFLEYSTSECHFFNGTERVRFLDRYFYNQEEYVRFDSDVGEFRAVTELGRPDEEYWNSQKDFLEDRRAAVDTYCRHNYGVGESFTVQRRVH'
HLAALLEL_II='HLA-DRA*01/HLA-DRB1*0101'
```

#### Modelling Runs for Single Predictions:
This mode is used when you want to run PMGen for a single model. We recommend to use `Wrapper` run.
1. Running for a Single Model

Model a single pMHC complex with anchor prediction:
```bash
python run_PMGen.py \
  --mode modeling \
  --peptide "$PEPTIDE" \
  --mhc_seq "$MHC_SEQ" \
  --mhc_type 1 \
  --output_dir outputs/basic_mhci \
  --predict_anchor
```
 Replace `MHC_SEQUENCE` with your MHC sequence (e.g., from a FASTA file).
 Use `--mhc_fasta <file>` instead of `--mhc_seq` if providing a FASTA file.

2. MHC-II Prediction with manual anchors:

We do not recommend to set `--anchors` and it is better to predict them.
```bash
python run_PMGen.py \
  --mode modeling \
  --peptide "$PEPTIDE" \
  --mhc_seq "$MHC_ALPHA_CHAIN_SEQ/$MHC_BETA_CHAIN_SEQ" \
  --mhc_type 2 \
  --output_dir outputs/mhcii_manual \
  --anchors [2,5,7,9] \
  --id "custom_id123"
```

3. Using FASTA input with allele name
```
python run_PMGen.py \
  --mode modeling \
  --peptide "$PEPTIDE" \
  --mhc_fasta mhc_sequence.fasta \
  --mhc_type 1 \
  --mhc_allele "$HLAALLEL" \
  --output_dir outputs/fasta_input \
  --predict_anchor
 ```

#### Using Wrapper (Recommended)

We highly recommed to use a `--df path/dataframe` as input and run the Wrapper mode. You can run it in two
different ways `--run parallel/single`. The first runs them in parallel depending on your
defined resources in `--max_ram , --max_cores` per job. The single mode runs predictions
in a for loop one by one.

You require to make a **tab separated** dataframe same as below:
```aiignore
id      peptide              mhc_allele      mhc_type        anchors
ex1     NLVPMVATV            HLA-B*5301         1              1;8
ex2     AAGASSLLL            HLA-A*0201         1               
ex3     SLLPEPPDAPDAPP       HLA-DRB1*04:01     2
ex3     SLLPEPPDAPDAPP       HLA-DRB1*04:01     2            2;4;6;9
```
Empty anchors rows will be predicted.

4. Basic wrapper mode (serial execution)

```
python run_PMGen.py \
  --mode wrapper \
  --run single \
  --df "$DF" \
  --output_dir outputs/wrapper_serial \
  --num_templates 4 \
  --num_recycles 3
```
5. Parallel wrapper execution with multiple models

The valie models are `model_1_ptm, model_2_ptm, model_3_ptm, model_4_ptm, model_5_ptm,
 model_1, model_2, model_3, model_4, model_5` from original alphafold params. If you
want to try a fine-tuned model, you could provide its path e.g 
`--fine_tuned_model_path AFfine/af_params/params_finetune/params/model_ft_mhc_20640.pkl`
and its name `--models model_2_ptm_ft`. Make sure the name contains `_ft` so it is
interpreted as fine-tuned model.
```
python run_PMGen.py \
  --mode wrapper \
  --run parallel \
  --df "$DF" \
  --output_dir outputs/wrapper_parallel \
  --max_ram 2 \      # GB per job
  --max_cores 16 \   # Total cores to use
  --num_templates 5 \
  --num_recycles 3 \
  --models model_2_ptm model_3_ptm
```
6. Memory-intensive parallel run

```
python run_PMGen.py \
  --mode wrapper \
  --run parallel \
  --df "$DF" \
  --output_dir outputs/wrapper_highmem \
  --max_ram 2 \     
  --max_cores 32 \
  --num_recycles 6 \
  --best_n_templates 4 \
  --n_homology_models 2
```

#### Protein Design:
You can run protein design in both wrapper and single modes.
7. Protein design using a single pMHC:

Enable peptide, MHC, and pseudo-sequence design:
```bash
bash
python run_PMGen.py \
 --mode modeling \
 --peptide "SIINFEKL" \
 --mhc_seq "MHC_ALPHA_CHAIN_SEQ/MHC_BETA_CHAIN_SEQ" \
 --mhc_type 2 \
 --id "design_model" \
 --output_dir "output" \
 --peptide_design \
 --only_pseudo_sequence_design \
 --mhc_design \
 --num_sequences_peptide 20 \
 --num_sequences_mhc 10 \
 --sampling_temp 0.2
```
This generates 20 peptide sequences and 10 MHC sequences (full and pseudo).

8. Protein Design using TSV file - Large number of sequences:

```bash
python run_PMGen.py \
   --mode wrapper \
   --run single   \
   --df "data/example/wrapper_input_example.tsv"   \
   --output_dir wrapper_protdesign   \
   --num_templates 4 \
   --num_recycles 4 \
   --peptide_design \
   --only_pseudo_sequence_design \
   --models model_2_ptm model_1_ptm \
   --only_protein_mpnn \
   --num_sequences_peptide 100 \
   --num_sequences_mhc 50 \
   --batch_size 5
```
9. Iterative Peptide Generation in Wrapper mode with fixed anchors:

```bash
python run_PMGen.py \
  --run parallel \
  --mode wrapper \
  --output_dir outputs/iterative_dataset \
  --df "data/example/wrapper_input_example.tsv" \
  --peptide_design \
  --num_sequences_peptide 10 \
  --binder_pred \
  --fix_anchors \
  --peptide_random_fix_fraction 0 \
  --iterative_peptide_gen 50
```

### Output

Results are saved in --output_dir with the following structure:

    output/pandora/: Template structures from PANDORA.
    output/alignment/: Alignment files for AlphaFold.
    output/alphafold/: AlphaFold-predicted PDB files.
    output/protienmpnn/: ProteinMPNN-designed sequences (if enabled).

Each subdirectory is organized by the id provided.

### Troubleshooting
```
    GPU Not Found: Ensure CUDA is installed (conda install -c nvidia cuda-nvcc) and your GPU is compatible.
    Memory Errors: Reduce --max_ram or --max_cores to match your system's capacity.
    Missing NetMHCpan: Set its path in user_setting.py for anchor prediction, check installation and check tcsh installation.
    Installation Issues: Check the Conda environment and rerun install.sh.
    PANDORA Errors: check output/pandora/id/pandora.log files.
    Alphafold Errors: Mostly relevant to GPU and Jax configuration. Adjust them based on your system.
    ProteinMPNN Errors: biopython, pytorch relavent.
```

For additional help, file an issue at GitHub.

### References
```
 "1. PANDORA - GitHub: https://github.com/X-lab-3D/PANDORA"
 "   Paper: https://www.frontiersin.org/articles/10.3389/fimmu.2022.878762/full"
 "2. AFfine - GitHub: https://github.com/phbradley/alphafold_finetune"
 "   Paper: https://www.pnas.org/doi/abs/10.1073/pnas.2216697120"
 "3. Modeller - Website: https://salilab.org/modeller/"
 "   Paper: A. Fiser, R.K. Do, & A. Sali. Modeling of loops in protein structures, Protein Science 9. 1753-1773, 2000."
 "4. AlphaFold - GitHub: https://github.com/google-deepmind/alphafold"
 "   Paper: https://www.nature.com/articles/s41586-021-03819-2"
 "5. ProteinMPNN - Github https://github.com/dauparas/ProteinMPNN"
 "   Paper: https://www.science.org/doi/10.1126/science.add2187"
```
