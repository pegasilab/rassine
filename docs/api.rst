Public API
==========

Command line arguments
~~~~~~~~~~~~~~~~~~~~~~

.. argparse::
   :module: rassine.arguments
   :func: argument_parser
   :prog: rassine

     --column_wave : @after
          Name of the spectrum file to reduce. The file can be either a pickle, csv, or txt. For pickle and csv the default colunm name must be ’wave’ and ’flux’ but this name can be changed by the column_wave and column_flux parameters. For fits file, use the Rassine_trigger.py file with preprocessing button.
          Automatic: No
          Type: string

Modules
~~~~~~~

.. automodule:: rassine
   :imported-members:
   :members:
   :undoc-members:
   :exclude-members: norm, interp1d, Slider, Button, RadioButtons, multicpu, Time
