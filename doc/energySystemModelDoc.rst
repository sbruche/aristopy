###################
Energy System Model
###################

The EnergySystemModel is the main model container.
An instance of the EnergySystemModel class holds the modelled components, the overall pyomo model and the results of the optimization. 
It also provides some features to manipulate the associated component models:

* Relax the integrality of binary variables
* General editing of component variables (e.g. change bounds or domains)
* Reset component variables after applying changes (e.g. relaxation)
* Cluster implemented time series data
* Export and import component configurations
* Add variables, constraints and objective function contributions outside of the component declaration


**EnergySystemModel class**

.. toctree::
   :maxdepth: 1

   energySystemModelClassDoc