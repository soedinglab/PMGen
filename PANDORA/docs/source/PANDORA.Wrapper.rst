Wrapper Module
===============


.. automodule:: PANDORA.Wrapper

The Wrapper module is used to run PANDORA in parallel on multiple cases.
All the cases must be provided in a .csv/.tsv file, and various information can be added to them (template, anchor positions, M chain seq).

Wrapper Example:

>>> from PANDORA import Pandora
>>> from PANDORA import Database
>>> from PANDORA import Wrapper
>>>
>>> ## A. Load the pre-generated templates database
>>> db = Database.load()
>>>
>>> ## B. Create and run wrapper with 24 concurrent jobs
>>> wrap =  Wrapper.Wrapper(data_file='datafile.tsv', database=db, IDs_col=0, 
>>>                         peptides_col=1, allele_name_col, n_jobs=24)
>>>



Wrapper
----------------------------------------

.. automodule:: PANDORA.Wrapper.Wrapper
    :members:
    :undoc-members:
    :show-inheritance:
