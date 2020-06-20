###################
Package description
###################

The two main elements of *aristopy* are the EnergySystem and the added Component
instances.
These components are modeled with the help of the superordinate class Component
and the inheriting subclasses Source, Sink, Conversion, Bus, and Storage.
Moreover *aristopy* comes with some auxiliary classes that are required for the
modeling process (Flow, Series, Var), or can be helpful for debugging (Logger),
visualizing results (Plotter), or modeling specific solar components
(SolarThermalCollector, PVSystem).

**Contents:**

.. toctree::
   :maxdepth: 3
	   
   ar_energySystem
   ar_component
   ar_utilities