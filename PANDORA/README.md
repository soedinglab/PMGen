# PANDORA

![Build](https://github.com/X-lab-3D/PANDORA/actions/workflows/main.yml/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/X-lab-3D/PANDORA/badge.svg?branch=master)](https://coveralls.io/github/X-lab-3D/PANDORA?branch=master)
[![Anaconda-Server Badge](https://anaconda.org/csb-nijmegen/csb-pandora/badges/version.svg)](https://anaconda.org/csb-nijmegen/csb-pandora)
[![Documentation Status](https://readthedocs.org/projects/csb-pandora/badge/?version=latest)](http://csb-pandora.readthedocs.io/?badge=latest)

### Peptide ANchored mODelling fRAmework for peptide-mhc complexes

![PANDORA](https://github.com/DarioMarzella/PANDORA/blob/master/images/flowchart_pMHCI.png?raw=true)

### For Reviewers of the Reversed Peptide Modeling

For the reviewing purposes of our reversed peptide modelling paper, please switch to the [reverse_peptide_MHCII](https://github.com/X-lab-3D/PANDORA/tree/reverse_peptide_MHCII) branch, and follow the [installation instructions](https://github.com/X-lab-3D/PANDORA/tree/reverse_peptide_MHCII?tab=readme-ov-file#github--pypi-installation) and example cases in the REAMDE there. We are actively working on a stable release.

### Contents

- [Overview](#overview)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Docker](#docker)
- [Tutorial](#tutorial)
- [Code Design](#code-design)
- [Output](#output)
- [License](./LICENSE)
- [Issues](#issues)
- [Publication](#publication)

## Overview

PANDORA is anchor restrained modelling pipeline for generating peptide-MHC structures.

It contains multiple functions to pre-process data and it's able to exploit different crucial domain knowledge provided by the user to guide the modelling.

PANDORA documentation can be found at: https://csb-pandora.readthedocs.io/en/latest/


## Dependencies
PANDORA requires MODELLER, python and some python libraries to be installed.
The following installations are required to start PANDORA installation:

- [Python](https://www.python.org/) >=3.7
- conda
- pip3

The (conda) installation process will take care of installing the following dependencies (see [Installation](#installation)):

- [BioPython](https://anaconda.org/conda-forge/biopython)
- [muscle](https://anaconda.org/bioconda/muscle) >= 5.1
- [Modeller](https://anaconda.org/salilab/modeller) >= 9.3
- [Blast](https://anaconda.org/bioconda/blast) >= 10.2
- [pdb2sql](https://github.com/DeepRank/pdb2sql) (Required only for RMSD calculations for evaluation purposes)


## Installation

#### 1. Get a Modeller Key License:
Prior to PANDORA installation, you need to first activate MODELLER's license. Please request MODELLER license at: https://salilab.org/modeller/registration.html

Replace XXXX with your MODELLER License key and run the command:

```
export KEY_MODELLER='XXXX'
```

#### 2. Install PANDORA

### GitHub / Pypi installation

#### 1. Install Modeller:
Prior to PANDORA installation, you need to first activate MODELLER's license. Please request MODELLER license at: https://salilab.org/modeller/registration.html

Replace XXXX with your MODELLER License key and run the command:

```
export KEY_MODELLER='XXXX'
```

Then, install MODELLER with:
```
conda install -y -c salilab modeller
```

Note: You can also follow the instruction in the conda install output to enter your modeller key in the appropriate file afterwars, instead of setting it beforehand.

#### 2. Install Other dependencies
PANDORA relies on muscle (https://anaconda.org/bioconda/muscle) and blast (https://anaconda.org/bioconda/blast) that can be both installed via bioconda.


```
conda install -c bioconda muscle=5.1 blast=2.10
```
For some HPC systems the conda blast installation might not work due to missing libraries. For those cases, you can download blast from their release page and install it (make shure you add it to your PATH, otherwise PANDORA will not be able to find it): https://ftp.ncbi.nlm.nih.gov/blast/executables/LATEST/

Note: Mac M1 processors cannot compile muscle version v5.0 and v5.1 from conda. To instll muscle, you will need to build it from source. You can find the muscle 5 code and the link to how to install from source in [their GitHub repo](https://github.com/rcedgar/muscle).

#### 3. Install PANDORA

Clone the repository:
```
git clone https://github.com/X-lab-3D/PANDORA.git
```
Enter the cloned directory and install the package:
```
cd PANDORA
pip install -e .

```
### Download Template Database
PANDORA needs a PDB template database to work. All the structures are retrieved from [IMGT](http://www.imgt.org/3Dstructure-DB/) database.

The database can be quickly retrieved by zenodo by running the command-line tool:

```bash
pandora-fetch
```

Alternatively, the same function can be done in python:

```python
from PANDORA import Database

Database.install_database()
```

### (Advanced) Generate template Database

You can also generate the database from scratch, downloading and parsing the structures directly from IMGT. This will ensure you to have as many templates as possible, as the quickly-retrievable database will not be re-released often. 

The database can be generated with the command-line tool:

```bash
pandora-create
```

Or with the pythoncode below: 

First, generate the necessary folders by running:
```
python install.py
```
Then, run the following script changing the path to the database you want to generate:

```python
## import requested modules
from PANDORA import Database

## A. Create local Database
db = Database.Database()
db.construct_database(n_jobs=<n_jobs>)
```

Note 1:  By default, the database generation will use one core only. You can sensitevly speed it up by changing the paramenter --num-cores for pandora-create or 'n_jobs' for Database.contruct_database().

Note 2: the database is saved by default into `~/PANDORA_databases/default`. It is possible to modify the folder name (`default`) by creating a `config.json` file in the PANDORA installation folder using `data_folder_name` as a key, and the desired folder name as a value, like in the example below:

```
{"data_folder_name": "<folder_in_Databases>"}
```

Note 3 (For master branch only, conda release v0.9.0): You can download the pre-made database from https://github.com/X-lab-3D/PANDORA_database (pMHC I only, generated on 23/03/2021) and follow the [instructions](https://github.com/X-lab-3D/PANDORA_database/blob/main/README.md). Please be sure you re-path your database as explained in the instructions.

### (Optional) Install NetMHCpan and/or NetMHCIIpan

PANDORA lets the user to predict peptide's anchor residues instead of using conventional predefined anchor residues.
In that case you need to download [NetMHCpan](https://services.healthtech.dtu.dk/cgi-bin/sw_request) (for peptide:MHC class I) and/or [NetMHCIIpan](https://services.healthtech.dtu.dk/cgi-bin/sw_request) (for peptide:MHC class II).

Once you have obtained the download link, you can install them by following the instructions in their .readme files. Note that PANDORA will assume they are installed and callable from command line (as netMHCpan and netMHCIIpan).

## Docker
The Docker release is currently not uploaded on DockerHub and, due to licensing of external software (MODELLER and NetMHCpan), the only way to use at this moment is by building the docker image directly from the recipe.

We advice to use Docker only in case Conda or Git/Pypi installations are not possible in your system.

1)  Get MODELLER lincense (and optional softwares, if desired)
   
First, you will always need to get a modeller license key and, optionally, the netMHCpan and netMHCIIpan installation packages (see [Installation](#installation) for details).

2) Download PANDORA

Download the GitHub repository:
```
git clone https://github.com/X-lab-3D/PANDORA.git
cd PANDORA
```

1) Add your license key
   
Now open the dockerfile and replace the "XXXX" at line 16 with your MODELLER license key.

4) (Optional) Install netMHCpan and/or netMHCIIpan
   
Download netMHCpan or netMHCIIpan directly into the PANDORA repository you cloned, then proceed with the installation as explained in their own .readme files until points 2.a and 2.b.

**For points 2.a and 2.b, make sure you set** NHOME to "/app/netMHCpan-\<version>" and TMPDIR to "/app/netMHCpan-\<version>/tmp" where \<version> is the version of the software you downloaded (same for netMHCIIpan, as "/app/netMHCIIpan-\<version>").

Finally, open the PANDORA dockerfile, uncomment lines 25-26, plus the lines referring to the software you want to install (29-30 for netMHCpan and 33-34 for netMHCIIpan), save and close the file.

In case, later, netMHCpan cannot be found by PANDORA, it might mean this step did not work as intended. 

5) Create a database folder in your local machine

To avoid re-downloading the PANDORA database every time you run your docker container, you will need to dowload the database in a folder synced between the container and the local machine.
To do so, create a folder (wherever you prefer in your machine). We will call the absolute path to this folder \<path/to/local/db>

6) Compile the docker image
   
Make sure you are in the PANDORA repo, then run:
```
docker build -t pandora .
```
You may chance "pandora" to the name you want to give to your docker image.
Also, be advised that including netMHCpan or netMHCIIpan into the image will take docker a considerable longer time.

7) Run the docker container

Run the docker container **while linking the local database folder to the docker one**:
```
docker run -v <path/to/local/db>:/root/PANDORA_databases/default/ -it pandora
```

8) (First time only) Download the PANDORA database
Inside your docker container, run the following to download the database:
```
pandora-fetch
```
It will be downloaded into a folder we previously linked to \<path/to/local/db>, thus will finally end up in your local machine.

9) Enjoy!

Done! Now you are ready to use PANDORA. Just remember to always run the docker container with:
```
docker run -v <path/to/local/db>:/root/PANDORA_databases/default/ -it pandora
```

## Tutorial

#### Differences between pMHC-I modeling and pMHC-II modeling setups
PANDORA is the first package capable of handling both pMHC-I and pMHC-II complexes, but with few setup differences, listed in the table below:


| --- | pMHC-I | pMHC-II |
| --- | --- | --- |
| Anchors | User-provided, predicted with netMHCpan  or automatically assigned on canonical spacing | User-provided or predicted by netMHCIIpan. **Cannot** be automatically assigned.| 
| MHC chains sequences | Alpha chain is called "M_chain_seq" and Beta chain is called "B2M_seq". Alpha chain is required (either as sequence or allele name). B2M is not required and will be retrieved from the selected template if not provided. | Alpha chain is called "M_chain_seq" and Beta chain is called "N_chain_seq". They are both required **except for HLA-DR**, for which the alpha chain will be automatically be assigned as HLA-DRA1*01:01 if not provided |

#### Example 1 : Generating a peptide:MHC complex given the peptide sequence
PANDORA requires at least these information to generate models:
- Peptide sequence
- MHC allele or MHC sequence

Steps:
A. Load the template database (see installation, point 4)

B. Creating a Template object based on the given target information

C. Generating *n* number of pMHC models (Default *n=20*)

Please note that you can specify output directory yourself, otherwise will be generated in a folder named as the case ID in the current working directory.

Command-line:
```bash
pandora-run -m I -i myTestCase -a HLA-A*0201 -p LLFGYPVYV -k 2,9
```
Please run `pandora-run --help` for further information about the arguments.

Python:
```python
## import requested modules
from PANDORA import Target
from PANDORA import Pandora
from PANDORA import Database

## A. Load local Database
db = Database.load()

## B. Create Target object
target = Target(id = 'myTestCase',
    allele_type = 'HLA-A*0201',
    peptide = 'LLFGYPVYV',
    anchors = [2,9])

## C. Perform modelling
case = Pandora.Pandora(target, db)
case.model()
```

Note: The user can also input the MHC chain sequence directly, without having to rely on the allele name. The argument to provide in this case is `M_chain_seq` for chain Alpha and either `B2M_seq` for MHC-I or `N_chain_seq` for MHC-II beta chain.

#### Example 2: Run PANDORA Wrapper on multiple cases (running in parallel on multiple cores)

PANDORA can model large batches of peptides in parallel. You need to provide the following peptide information in a *.tsv* or *.csv* file:

- *Peptide sequence,  MHC Allele name / MHC chain sequence*

Note: you can also add various information to your file, including anchors for each case, templates, IDs.
You can find all the arguments for the master branch version in the [documentation](https://csb-pandora.readthedocs.io/en/latest/PANDORA.Wrapper.html#module-PANDORA.Wrapper.Wrapper).
For other branches, like development, we suggest you to use the python help() function or check directly the [docstring in the source code](https://github.com/X-lab-3D/PANDORA/blob/d43f9d91bee9f793ee7fe4cb10be3cb4e299e36d/PANDORA/Wrapper/Wrapper.py#L147).

The Wrapper class will take care of generating PANDORA target objects and parallelize the modelling on the given number of cores <n_cores>:

Command-line:
```bash
pandora-wrapper -m I -f datafile.tsv -h False  -p 0 -a 1 -d tab
```
Please run `pandora-wrapper --help` for further information about the arguments.

```python
from PANDORA import Database
from PANDORA import Wrapper

## A. Load local database
db = Database.load()

## B. Create the wrapper object. It will also run the modelling for each case.
wrap =  Wrapper.Wrapper('datafile.tsv', db, num_cores=<n_cores>)

```

#### Example 3: Create multiple loop models in a your given directory
There are some options provided that you can input them as arguments to the functions.

For instance:
- Generate more models for your modelling case
- Specify the output directory yourself
- Give your target a name
- Predict anchors by NetMHCpan

Please note that, if *anchors* is not specified or *use_netmhcpan* is set to *False*, PANDORA will automatically assign canonical anchors (P2 and PΩ). This can be done automatically only for pMHC-I structures.

```python
from PANDORA import Target
from PANDORA import Pandora
from PANDORA import Database

## A. load the pregenerated Database  of all pMHC PDBs as templates
db = Database.load()

## B. Create Target object
target = Target(id = 'myTestCase',
    allele_type = ['HLA-B*5301', 'HLA-B*5302'],
    peptide = 'TPYDINQML',
    use_netmhcpan = True)

## C. Perform modelling
case = Pandora.Pandora(target, db, output_dir = '/your/directory/')
case.model(n_loop_models=100)  # Generates 100 models
```

#### Example 4: Model a peptide:MHCI complex with an alpha helix in the peptide

Input domain secondary structure prediction information (Helix/Beta strand):

```python
from PANDORA import Target
from PANDORA import Pandora
from PANDORA import Database

## A. Load pregenerated database of all pMHC PDBs as templates
db = Database.load()

## B. Create Target object
target = Target(id = 'myMHCITestCase',
    allele_type = 'MH1-B*2101',
    peptide = 'TAGQSNYDRL',
    anchors = [2,10],
    helix = ['4', '9'])

## C. Perform modelling
case = Pandora.Pandora(target, db)
case.model(helix=target.helix)
```
#### Example 5: Generating a peptide:MHC class II complex given the peptide sequence

To model a peptide:MHC class II complex, you only need to specify that in *PMHC.Target()* function: as *MHC_class='II'* (By default it is set to model MHC class I).

```python
from PANDORA import Target
from PANDORA import Pandora
from PANDORA import Database

## A. Load pregenerated database of all pMHC PDBs as templates
db = Database.load()

target = Target(id='myMHCIITestCase',
    MHC_class = 'II',
    allele_type = ['HLA-DRA*0102', 'HLA-DRB1*0101'],
    peptide = 'GELIGILNAAKVPAD',
    anchors = [4, 7, 9, 12])

case = Pandora.Pandora(target, db)
case.model()
```
Note: For MHC II, no canonical anchors can be defined. Therefore the user must either install and use NetMHCIIpan or directly input the anchors positions as *anchors* in *PMHC.Target()*

####  Example 6: Generating a peptide:MHC class II complex with a reversed peptide

Reversed peptide binding is often observed in specific alleles of MHC-II, but PANDORA allows you to model reverse peptides for any MHC-II allele. 

**Note**: When performing reversed peptide modeling, **only the anchor residues should be considered in reverse order**, while the rest of the peptide should maintain its normal sequence. This ensures that the reversed template preserves the correct binding interactions at the MHC anchor positions.

Required Information:

- Peptide sequence (normal sequence)

- MHC allele or MHC sequence

- Anchor residues (input in reverse order, descending)

Steps:

- A. Load the template database

- B. Creating a Target object based on the given target information, with the reverse option set to True

- C. Generating the pMHC model using the reversed template

As illustrated in the figure, this example demonstrates a peptide sequence with anchor residues [11, 8, 6, 3] in reverse order.

<img src="https://github.com/X-lab-3D/PANDORA/blob/reverse_peptide_MHCII/images/reverse_peptide.png" alt="Logo" width="500">

How to run PANDORA for a peptide sequence with anchor residues [10, 7, 5, 2] in reverse order:

```python
## Import required modules
from PANDORA import Target
from PANDORA import Pandora
from PANDORA import Database

## A. Load local Database
db = Database.load()

## B. Create Target object with reverse peptide option
target = Target(id='reversedTestCase',
    MHC_class='II',
    peptide='VLQAGFFLLTRIL',
    allele_type=['HLA-DPA1*02:01', 'HLA-DPB1*01:01'],
    anchors=[10, 7, 5, 2],                             # Input anchors in reverse order
    reverse=True)                                      # Flag to indicate the use of reversed peptides

## C. Perform modelling
case = Pandora.Pandora(target, db)
case.model()
```

#### Example 7: Benchmark PANDORA on one modelling case

Evaluate the framework on a target with a known experimental structure:
- Provide the PDB ID for the *Target* class
- Set *benchmark=True* for the modelling
  (calculates L-RMSD to show how far the model is from the near-native structure)

```python
from PANDORA import Target
from PANDORA import Pandora
from PANDORA import Database

## A. Load pregenerated database of all pMHC PDBs as templates
db = Database.load()

## B. Create Target object
target = Target(id='1A1M',
    allele_type=db.MHCI_data['1A1M'].allele_type,
    peptide=db.MHCI_data['1A1M'].peptide,
    anchors = db.MHCI_data['1A1M'].anchors)
    
## C. Perform modelling
case = Pandora.Pandora(target, db)
case.model(benchmark=True)
```

## Code Design
PANDORA has been implemented in an Object-Oriented Design(OOD). Resulting in a comprehensible and user-friendly framework.

see [Class Diagram](https://github.com/DarioMarzella/PANDORA/blob/master/images/class_diagram.png?raw=true)

## Output

The following file structure is prepared to store the output files for each case. Each modelling case is given a specific name based on target and template ID.

Please note that the modelling results consisting genretaed models by default are stored in *./Databases/default/outputs/* directory

- Main outputs: *molpdf_DOPE.tsv, *BL*.pdb, modeller.log(
- Input files prepared for modelling: *contacs_*.list, *.ali*
- *.py* files: MODELLER scripts
- MODELLER by product outputs(Generated during the modelling): *D0*, DL*, *IL*.pdb , , *.ini, *.lrsr, *.rsr, *.sch, ...*

```
Databases
  └── data
     └── outputs                         Default directory to save output
        └── <target_name>_<template_id>  Each user's modelling case is given a specific name

           ├── molpdf_DOPE.tsv           Ranking all models by molpdf and DOPE modeller's scoring functions
           ├── *BL*.pdb                  Final models
           ├── modeller.log              Printing log file generated by MODELLER, describing modelling steps, or any issues arose along modelling

           ├── *.ali                     Alignment file between template(s) and target used for modelling
           ├── contacts_*.list           Contact restraints

           ├── MyLoop.py                 MODELLER script to set loop modelling parameters for the peptide
           ├── cmd_modeller_ini.py       MODELLER script to generate an initial model to extract restraints from
           ├── cmd_modeller.py           MODELLER script to set the main modelling parameters

           ├── *.ini                     Model generated placing the target atoms at the same coordinate as the template's atoms
           ├── *IL*.pdb                  Initial loop model
           └── ...


```

## Issues

If you have questions or find a bug, please report the issue in the [Github issue channel](https://github.com/X-lab-3D/PANDORA/issues).

## Publication
If you use PANDORA, please cite the following paper in your work:

Marzella DF, Parizi FM, Tilborg Dv, Renaud N, Sybrandi D, Buzatu R, Rademaker DT, ‘t Hoen PAC and Xue LC (2022) PANDORA: A Fast, Anchor-Restrained Modelling Protocol for Peptide: MHC Complexes. Front. Immunol. 13:878762. doi: 10.3389/fimmu.2022.878762

https://www.frontiersin.org/articles/10.3389/fimmu.2022.878762/full
