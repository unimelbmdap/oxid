====================
OxID
====================

A python package that takes magenetic data from samples of Ochre to predict the type of iron oxide present

Installation
============

To install, first make sure that you have poetry. See the instructions at https://python-poetry.org/docs/#installation

Then clone the repository:

.. code-block:: bash

    git clone https://github.com/unimelbmdap/oxid.git


Then install the dependencies using poetry:

.. code-block:: bash

    cd oxid
    poetry install

Enter the virtual environment:

.. code-block:: bash

    poetry shell

Usage
=====

To use the OxID on a single sample, you can run the following command:

.. code-block:: bash

    oxid infer --rtsirm <RT-SIRM.dat> --hysteresis <hysteresis.dat> --zfcfc <zfcfc.dat> --plot <plot.png>

You do not need to include each of the input files, but you must include at least one of them. The `--plot` argument is also optional.

To use the OxID on a batch of samples, you can run the following command:

.. code-block:: bash

    oxid infer-csv <input.csv> --output <output.csv>

The input.csv file should have at least one of the following columns: `RT-SIRM`, `Hyster`, `ZFC-FC` (case insensitive and hyphens are optional). 
If the column is not found or the corresponding cell is blank in a row, then OxID will skip that input data type.
The output.csv file will contain the inferred values plus all the columns from the input file.

See more options by running:

.. code-block:: bash

    oxid --help

Credits
=======

This package was created by:

- `Maddison Crombie <https://www.linkedin.com/in/maddison-crombie-26814b12a/?originalSubdomain=au>`_
- `Robert Turnbull <https://findanexpert.unimelb.edu.au/profile/877006-robert-turnbull>`_
- `Rachel Popelka-Filcoff <https://findanexpert.unimelb.edu.au/profile/870256-rachel-popelka-filcoff>`_
