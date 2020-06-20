#########
Utilities
#########

There are several utility classes that are necessary or optional for the
modeling process. The Flow class is required to add component commodities,
variables, and interconnections. The Series and Var classes are used to
introduce time-series data and additional variables to the model.
The Logger class is a helpful tool for the debugging process. Auto-generated
plots of the Plotter class can also be used for result validation.
The different Solar classes provide functionality to generate feed-in time
series data for thermal and electrical solar collectors.

.. toctree::
   :maxdepth: 2
   
   flowSeriesVar
   logger
   plotter
   solar
  
