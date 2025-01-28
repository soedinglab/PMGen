Tutorial
========

Download Template Database
--------------------------------

PANDORA needs a PDB template database to work.
The database can be quickly retrieved by zenodo by running the command-line tool:

`$ pandora-fetch``


Alternatively, the same function can be done in python:


>>> from PANDORA import Database
>>> Database.install_database()


(Advanced) Generate template Database

You can also generate the database from scratch, downloading and parsing the structures directly from IMGT. This will ensure you to have as many templates as possible, as the quickly-retrievable database will not be re-released often. 

The database can be generated with the command-line tool:

`pandora-create``


Or with the pythoncode below: 


>>> ## import requested modules
>>> from PANDORA import Database
>>> 
>>> ## A. Create local Database
>>> db = Database.Database()
>>> db.construct_database(n_jobs=<n_jobs>)

Note 1:  By default, the database generation will use one core only. You can sensitevly speed it up by changing the paramenter --num-cores for pandora-create ot 'n_jobs' for Database.contruct_database().

Note 2: the database is saved by default into `~/PANDORA_databases/default`. It is possible to modify the folder name (`default`) by creating a `config.json` file in the PANDORA installation folder using `data_folder_name` as a key, and the desired folder name as a value, like in the example below:


`{"data_folder_name": "<folder_in_Databases>"}`


p:MHC-I base example
--------------------

PANDORA requires at least these information to generate models:

- Peptide sequence
- MHC allele

Steps:

A. Load templates database

B. Create a Target object

C. Generate *n* pMHC models (Default n=20)

Please note that you can specify output directory yourself, otherwise will be generated in the current working directory.

>>> ## import requested modules
>>> from PANDORA import PMHC
>>> from PANDORA import Pandora
>>> from PANDORA import Database
>>>
>>> ## A. Create local Database
>>> db = Database.load()
>>>
>>> ## B. Create Target object
>>> target = PMHC.Target(
>>>     allele_type='HLA-A*02:01',
>>>     peptide='LLFGYPVYV',
>>>     anchors = [2,9])
>>>
>>> ## C. Perform modelling
>>> case = Pandora.Pandora(target, db)
>>> case.model()


Run PANDORA on multiple cases (running in parallel on multiple cores)
---------------------------------------------------------------------

PANDORA can model more than one pMHC complex in parallel. You need to provide the following information in a *.tsv* file:

- *Peptide sequence,  Allele name*


>>> from PANDORA import Pandora
>>> from PANDORA import Database
>>> from PANDORA import Wrapper
>>>
>>> ## A. load the pre-generated templates database
>>> db = Database.load()
>>>
>>> ## B. Create the wrapper object
>>> wrap =  Wrapper(data_file='datafile.tsv', database=db, num_cores=128)


p:MHC-II base example
---------------------

>>> from PANDORA import PMHC
>>> from PANDORA import Pandora
>>> from PANDORA import Database
>>>
>>> ## A. Load pre-generated template database
>>> db = Database.load()
>>>
>>> target = PMHC.Target(
>>>     MHC_class='II',
>>>     allele_type = ['HLA-DRA*0101', 'HLA-DRB1*0101'],
>>>     peptide = 'GELIGILNAAKVPAD',
>>>     anchors = [4, 7, 9, 12])
>>>
>>> case = Pandora.Pandora(target, db)
>>> case.model()


High accuracy modelling (advised to set prepare models for MD)
--------------------------------------------------------------

Multiple options can be used to increase the modelling accuracy.

The following table reports the arguments that most impact the modelling quality and computational time. The same arguments can be used for either Pandora.model() or Wrapper()


.. list-table:: Modelling time and quality affecting arguments
   :widths: 15 40 20 15
   :header-rows: 1

   * - Option 
     - Effect 
     - Impact on computational time 
     - Optimal

   * - n_loop_models 
     - Increases the modelling sampling step, generating more loop models 
     - Computational time linearly increases with the increasing of the loop models requested.
     - As high as the user wants
  
   * - loop_refinement 
     - MODELLER loop refinement method 
     - Very small 
     - "very_slow" 

   * - restraints_stdev 
     - If False (by default), the restraints are not flexible at all, locking in place the anchors (for MHC-I) or the whole binding core (for MHC-II) to the template position. An higher stdev allowes the restraints to be stretched to accomodate different anchors. Highly recommended to prevent small clashed that might deeply influence MD experiments quality
     - High, 50% more for pMHC-I cases and up to 90% more for pMHC-II cases
     - 0.2 - 0.3 . Not advised to increase over 0.5

   * - clip_C_domain 
     - False by default. If True, it does not model C-like domain and the Beta-2 microglobulin (if present), generating a 3D-model with only the G-domains and the peptide.
     - Practically none
     - False


Example of high accuracy pMHC-I modelling:

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
>>> case.model(n_loop_models=100, loop_refinement="very_slow",
>>>            restraints_stdev=0.3, clip_C_domain=False)


Fast and light modelling
------------------------

On the other hand, a user might want to quickly generate large amounts of models with a lower focus on accuracy. Here's an example of how to do so:

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
>>> case.model(n_loop_models=20, loop_refinement="very_fast",
>>>            restraints_stdev=False, clip_C_domain=True)

The number of loop models could in theory be reduced to less than 20 models, but we do not advise this solution to not decrease the accuracy too much.
