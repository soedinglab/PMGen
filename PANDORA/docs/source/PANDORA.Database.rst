Database Generation
===================


.. automodule:: PANDORA.Database

This module contains all the tools to compute the template structure and reference sequence databases. The main class used for the data generation is ``PANDORA.Database.Database``.
Through this class you can specify the type of structures you want to download (MHCI and/or II).

Basic building Example:

>>> from PANDORA import Database
>>>
>>> db = Database.Database()
>>> db.construct_database()
>>>

A database can be easily loaded and edited.
Custom structures (i.e. not coming from IMGT) can be also added to the database by using ``Database.add_structure()`` function.
To do so, you will need to read your structure into a Biopython structure object:

>>> from PANDORA import Database
>>>
>>> #Load the database
>>> db = Database.load()
>>>
>>> db.add_structure(id='0000', allele_type=['HLA-A*02:01'],
                      peptide = 'AAALLLAAA', MHC_class = 'I',
                      anchors = [],
                      pdb_path = './PDBs/pMHCI/0000.pdb', pdb = False)

Database
----------------------------------------

.. automodule:: PANDORA.Database.Database
    :members:
    :undoc-members:
    :show-inheritance:

Database_functions
----------------------------------------

.. automodule:: PANDORA.Database.Database_functions
    :noindex:
    :members:
    :undoc-members:
