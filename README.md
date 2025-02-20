

# ParseFold-MHC: peptide-MHC structure prediction pipeline

## Installation
**Step 1:**

Make sure you have mamba/conda installed.

**Step 2**

Check your GPU configuration and setting.

**Step 3**
```aiignore
# Clone Repository
git clone https://github.com/AmirAsgary/ParseFold-MHC.git
# install
cd ParseFold-MHC
./install.sh
conda activate parsefold_mhc
```


## How to Use
There are two main ways to use ParseFold so called `--mode wrapper` and 
`--mode modeling`. The default is set on `wrapper` mode. In this mode
You can parallelize multiple predictions and run ParseFold in `--run parallel` mode
which makes template engineering pipline much faster. In `modeling` mode you can
only predict a single structure and the inputs are provided in bash script,
while `wrapper` requires a dataframe. 
### Set example environment variables
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

### Basic Modeling Mode #
#########################

#### Example 1: Minimal MHC-I modeling
```
python run_parsefold.py \
  --mode modeling \
  --peptide "$PEPTIDE" \
  --mhc_seq "$MHC_SEQ" \
  --mhc_type 1 \
  --output_dir outputs/basic_mhci \
  --predict_anchor
```
#### Example 2: MHC-II with manual anchors
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
#### Example 3: Using FASTA input with allele name
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

------------------
### Wrapper Mode Runs ##
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

#### Example 4: Basic wrapper mode (serial execution)
```
python run_parsefold.py \
  --mode wrapper \
  --run single \
  --df "$DF" \
  --output_dir outputs/wrapper_serial \
  --num_templates 4 \
  --num_recycles 3
```
#### Example 5: Parallel wrapper execution with multiple models
The valie models are `model_1_ptm, model_2_ptm, model_3_ptm, model_4_ptm, model_5_ptm,
 model_1, model_2, model_3, model_4, model_5` from original alphafold params. If you
want to try a fine-tuned model, you could provide its path e.g 
`--fine_tuned_model_path AFfine/af_params/params_finetune/params/model_ft_mhc_20640.pkl`
and its name `--models model_2_ptm_ft`. Make sure the name contains `_ft` so it is
interpreted as fine-tuned model.
```
python run_parsefold.py \
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
#### Example 6: Memory-intensive parallel run
```
python run_parsefold.py \
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