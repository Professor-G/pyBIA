.. _catalog:

Morphological Catalog
===========

.. admonition:: Under Construction (last updated 2025-09-16)

   This documentation is still being written and may change frequently!

If you have a 2D array, but no positions, creating a catalog is quick and easy using the `catalog <https://pybia.readthedocs.io/en/latest/autoapi/pyBIA/catalog/index.html>`_ module:

.. code-block:: python

    from pyBIA import catalog

    cat = catalog.Catalog(data)
    cat.create(save_file=True)

The X and Y pixel arguments can be input if source locations are known, with optional parameters available to control background subtraction, source detection thresholds, and flux calculations. If the error map is provided, the output catalog will contain the photometric error as well, which, after the catalog has been created, can be accessed via the ``cat`` class attribute which will be a dataframe containing all of the calculated flux and morphological features. These computed features can then be used to train a machine learning model using the `ensemble_model <https://pybia.readthedocs.io/en/latest/autoapi/pyBIA/ensemble_model/index.html>`_ module. 