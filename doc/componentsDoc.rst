##########
Components
##########

Components can be added to an "EnergySystemModel" class. 
The "Component" class holds the data and the model equations of each component. 
All components have to inherit from the "Component" class.
There are five basic component classes in "aristopy". These are:

* Source class
* Sink class (inherits from Source class),
* Conversion class,
* Bus (Transmission) class,
* Storage class.

**Component (Super) class**

.. toctree::
   :maxdepth: 1

   componentClassDoc

**Basic component classes**

.. toctree::
   :maxdepth: 1
   
   sourceSinkClassDoc
   conversionClassDoc
   busClassDoc
   storageClassDoc
