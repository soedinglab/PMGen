Main Module (Pandora)
=====================


.. automodule:: PANDORA.Pandora

This is the main PANDORA module, to find the template for a given target and run the modelling, including all the intermediate steps.
The main class used for the data generation is ``PANDORA.Pandora.Pandora``.

Basic building Example:

>>> case = Pandora.Pandora(target, db)
>>> case.model()

Where `target` is a pre-defined PANDORA.PMHC.Target object and `db` is a pre-defined PANDORA.Database.Database object.

Pandora
----------------------------------------

.. automodule:: PANDORA.Pandora.Pandora
    :members:
    :undoc-members:
    :show-inheritance:

Modelling_functions
----------------------------------------

.. automodule:: PANDORA.Pandora.Modelling_functions
    :noindex:
    :members:
    :undoc-members:
