.. image:: https://www.energietechnik.tu-berlin.de/fileadmin/fg106/Fotos/aristopy_logo_small.png
    :target: https://www.energietechnik.tu-berlin.de/menue/forschung/energiesystemanalyse_und_optimierung/oeb_ensys/
    :width: 300px
    :alt: aristopy Logo
    :align: right

|

Welcome to aristopy's documentation!
====================================

The Python package *aristopy* is a framework for modeling and optimizing the design and operation of energy systems.
The name of the framework is derived from the great Greek thinker Aristotle. For Aristotle, planning and the wise use of human goods represented great virtues. 
Transferred to today's time and the design of energy systems, this implies using appropriate tools that support the planning process and contribute to an optimal use of the available resources (money, fuel, etc.).

**Selected highlights**

* Flexible modeling of energy systems with only a small number of basic components (Source, Sink, Conversion, Bus, Storage) and a comprehensive API.
* Manual scripting of component constraints to enable all types of mathematical modeling classes (linear [LP], mixed-integer linear [MILP], mixed-integer non-linear [MINLP], etc.).
* Declaration of persistent models to quickly run models iteratively after applying small changes (e.g., add an integer-cut constraint).
* Auto-generated visualization of the optimization results with flexible plotting routines.


**How to cite aristopy**

| You are welcome to test *aristopy* and use it for your own purposes. If you publish results based on the application of the package, we kindly ask you to cite this documentation.
| The `tsam <https://github.com/FZJ-IEK3-VSA/tsam>`_ package (time series aggregation module) is used for the clustering of time series.
| The package `pvlib <https://github.com/pvlib/pvlib-python>`_ is applied for calculating solar feed-in time series. Pvlib is not a part of the standard installation of *aristopy* and needs to be obtained separately.

.. toctree::
   :maxdepth: 1
   :caption: Contents:
   
   installation
   examples
   package
   license
   acknowledgment
   

Indices and tables
^^^^^^^^^^^^^^^^^^

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
