

# ParseFold-MHC: peptide-MHC structure prediction pipeline

## Installation
aa

## Set example environment variables
```
PEPTIDE='NLVPMVATV'
# MHC-I
MHC_SEQ='AGSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGCYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMCAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQRTDA'
HLAALLEL='HLA-B*5301'
DF='data/example/wrapper_input_example.tsv'
# MHC-II
MHC_ALPHA_CHAIN_SEQ='IKEEHVIIQAEFYLNPDQSGEFMFDFDGDEIFHVDMAKKETVWRLEEFGRFASFEAQGALANIAVDKANLEIMTKRSNYTPITNVPPEVTVLTNS'
MHC_BETA_CHAIN_SEQ='DTRPRFLEYSTSECHFFNGTERVRFLDRYFYNQEEYVRFDSDVGEFRAVTELGRPDEEYWNSQKDFLEDRRAAVDTYCRHNYGVGESFTVQRRVH'
HLAALLEL_II='HLA-DRA*01/HLA-DRB1*0101'
```

#########################
### Basic Modeling Mode #
#########################

## Example 1: Minimal MHC-I modeling
```
python run_parsefold.py \
  --mode modeling \
  --peptide "$PEPTIDE" \
  --mhc_seq "$MHC_SEQ" \
  --mhc_type 1 \
  --output_dir outputs/basic_mhci \
  --predict_anchor
```
## Example 2: MHC-II with manual anchors
We do not recommend to set anchors and it is better to predict them.
```
python run_parsefold.py \
  --mode modeling \
  --peptide "$PEPTIDE" \
  --mhc_seq "MHC_ALPHA_CHAIN_SEQ/MHC_BETA_CHAIN_SEQ" \
  --mhc_type 2 \
  --output_dir outputs/mhcii_manual \
  --anchors [2,5,7,9] \
  --id "custom_id123"
```
## Example 3: Using FASTA input with allele name
```
python run_parsefold.py \
  --mode modeling \
  --peptide "$PEPTIDE" \
  --mhc_fasta mhc_sequence.fasta \
  --mhc_type 1 \
  --mhc_allele "$HLAALLEL" \
  --output_dir outputs/fasta_input \
  --predict_anchor
 ```

########################
### Wrapper Mode Runs ##
We highly recommed to use a `--df path/dataframe` as input and run the Wrapper mode. You can run it in two
different ways `--run parallel/single`. The first runs them in parallel depending on your
defined resources in `--max_ram , --max_cores` per job. The single mode runs predictions
in a for loop one by one.

########################
## Example 4: Basic wrapper mode (serial execution)
```
python run_parsefold.py \
  --mode wrapper \
  --run single \
  --df "$DF" \
  --output_dir outputs/wrapper_serial \
  --num_templates 4 \
  --num_recycles 3
```
## Example 5: Parallel wrapper execution with resource limits
You can use as many as
```
python run_parsefold.py \
  --mode wrapper \
  --run parallel \
  --df "$DF" \
  --output_dir outputs/wrapper_parallel \
  --max_ram 2 \      # GB per job
  --max_cores 16 \   # Total cores to use
  --num_templates 5 \
  --num_recycles 6 \
  --models model_2_ptm model_3_ptm
```
## Example 6: Memory-intensive parallel run
```
python run_parsefold.py \
  --mode wrapper \
  --run parallel \
  --df "$DF" \
  --output_dir outputs/wrapper_highmem \
  --max_ram 2 \     
  --max_cores 32 \
  --num_recycles 3 \
  --best_n_templates 4 \
  --n_homology_models 2
```