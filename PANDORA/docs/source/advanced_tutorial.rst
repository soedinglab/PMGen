Advanced Tutorial
=================

Multiple levels of parallelization
----------------------------------

PANDORA currently supports two levels of parallelization: per-case and per-loop parallelization.
Per-case parallelization allows the user to spread multiple p-MHC cases accross multiple cores in parallel. It is handled by the Wrapper() class with the argument "num_cores".
Per-loop parallelization allows the user to spread the same case's loop models on multiple cores in parallel. It is handled by MODELLER itself and it is passed through either the Wrapper class or the Pandora.model() function with the argument "n_jobs".
When using the default amount of loop models (20) we advise to only use per-case parllelization for time-efficency and ease-of-use.
In case, instead, the user wants to model large number of loop models, maybe for a small or moderate amount of cases, we advise to increase the per-loop parallelization.
Be advised: the final maximum number of CPU cores used is num_cores * n_jobs, as each parallel case will use n_jobs cores to generate its models.

The following sample script will spread the 2000 loop models of one case over 4 separate CPU cores:

>>> from PANDORA import PMHC
>>> from PANDORA import Pandora
>>> from PANDORA import Database
>>>
>>> ## A. load the pre-generated templates database
>>> db = Database.load()
>>>
>>> ## B. Create Target object
>>> target = PMHC.Target(id='myTestCase'
>>>     allele_type = 'HLA-B*5301',
>>>     peptide = 'TPYDINQML')
>>>
>>> ## C. Perform modelling
>>> case = Pandora.Pandora(target, db)
>>> case.model(n_loop_models=2000, n_jobs=4
>>>            restraints_stdev=0.3)

The following sample script will spread the 2000 loop models of each case over 4 separate CPU cores, and spread all the cases over 8 processes, resulting in a total of 32 used CPU cores:

>>> from PANDORA import Pandora
>>> from PANDORA import Database
>>> from PANDORA import Wrapper
>>>
>>> ## A. load the pre-generated templates database
>>> db = Database.load()
>>>
>>> ## B. Create the wrapper object
>>> wrap =  Wrapper(data_file='datafile.tsv', database=db, 
                    num_cores=8, n_jobs=4, n_loop_models=2000)

Benchmark PANDORA on one p:MHC-I case
-------------------------------------

If you want to evaluate the framework on a target with a known experimental structure:
- Provide the PDB ID for the *Target* class
- Set *benchmark=True* for the modelling
(calculates L-RMSD to show how far the model is from the near-native structure)

>>> from PANDORA import PMHC
>>> from PANDORA import Pandora
>>> from PANDORA import Database
>>>
>>> ## A. Load pregenerated database of all pMHC PDBs as templates
>>> db = Database.load('pandora_Database')
>>>
>>> ## B. Create Target object
>>> target = PMHC.Target('1A1M',
>>>     db.MHCI_data['1A1M'].allele_type,
>>>     db.MHCI_data['1A1M'].peptide,
>>>     anchors = db.MHCI_data['1A1M'].anchors)
>>>
>>> ## C. Perform modelling
>>> case = Pandora.Pandora(target, db)
>>> case.model(benchmark=True)

p:MHC-I complex with an alpha helix in the peptide
--------------------------------------------------

If you have some domain knowledge of the peptide conformation, whether it forms secondary structures other than loop (Helix/Beta strand), the framework will consider that while modelling the peptide:


>>> from PANDORA import PMHC
>>> from PANDORA import Pandora
>>> from PANDORA import Database
>>>
>>> ## A. Load pregenerated database of all pMHC PDBs as templates
>>> db = Database.load('pandora_Database')
>>>
>>> ## B. Create Target object
>>> target = PMHC.Target(
>>>     allele_type = ['MH1-B*2101', 'MH1-B*2101'],
>>>     peptide = 'TAGQSNYDRL',
>>>     anchors = [2,10],
>>>     helix = ['4', '9'])
>>>
>>> ## C. Perform modelling
>>> case = Pandora.Pandora(target, db)
>>> case.model(helix=target.helix)
