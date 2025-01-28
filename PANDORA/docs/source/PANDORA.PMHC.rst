Target Generation
=================


.. automodule:: PANDORA.PMHC

This module is used to define PMHC objects (including Target and Template objects) and Model objects.
The class used to generate a PANDORA target is ``PANDORA.PMHC.Target``.

This module also contain the Model class, carrying various information of the output models.
These objects can be saved when using `PANDORA.Pandora.Pandora.model` with the argument ``pickle_out=True``

Basic target Example:

>>> from PANDORA.PMHC import Target
>>>
>>> target = PMHC.Target(id = 'myTestCase',
>>>     MHC_class = 'I',
>>>     allele_type = 'HLA-A*02:01',
>>>     peptide = 'LLFGYPVYV',
>>>     anchors = [2,9])
>>>


PMHC
----------------------------------------

.. automodule:: PANDORA.PMHC.PMHC
    :members:
    :undoc-members:
    :show-inheritance:

Model
----------------------------------------

.. automodule:: PANDORA.PMHC.Model
    :noindex:
    :members:
    :undoc-members:

Anchors
----------------------------------------

.. automodule:: PANDORA.PMHC.Anchors
    :noindex:
    :members:
    :undoc-members:
