<h1>
```diff
- The project has been moved to [GitLab](https://git.tu-berlin.de/etus/public/aristopy)
```
</h1>


<a href="https://www.energietechnik.tu-berlin.de/menue/forschung/energiesystemanalyse_und_optimierung/oeb_ensys/">
<img src="https://raw.githubusercontent.com/sbruche/aristopy/master/docs/misc/aristopy_logo.png" alt="aristopy logo" width="300px"/></a>

# Optimizing energy systems with *aristopy*


The Python package *aristopy* is a framework for modeling and optimizing the
design and operation of energy systems.
The name of the framework is derived from the great Greek thinker Aristotle.
For Aristotle, planning and the wise use of human goods represented great virtues.
Transferred to today's time and the design of energy systems, this implies using
appropriate tools that support the planning process and contribute to an
optimal use of the available resources (money, fuel, etc.).

##  Selected highlights
* Flexible modeling of energy systems with only a small number of basic
  components (Source, Sink, Conversion, Bus, Storage) and a comprehensive API.
* Manual scripting of component constraints to enable all types of
  mathematical modeling classes (linear \[LP\], mixed-integer linear
  \[MILP\], mixed-integer non-linear \[MINLP\], etc.).

* Declaration of persistent models to quickly run models iteratively
  after applying small changes (e.g., add an integer-cut constraint).
* Auto-generated visualization of the optimization results with
  flexible plotting routines.

## Documentation
The package documentation is hosted on readthedocs.org and can be accessed
[here](https://aristopy.readthedocs.io/en/latest/index.html).

## Installation
Before you can create your first optimization model with *aristopy*, you need
to make sure you have Python and *aristopy*, and at least one suitable 
mathematical solver installed on your machine.

The installation of *aristopy* in your current environment can easily be 
executed from the command line via pip: 

```python
pip install aristopy
```

More detailed installation instructions can be found in the 
[documentation](https://aristopy.readthedocs.io/en/latest/installation.html).

## Examples
The code of the first simple example from the examples directory, shown 
below, illustrates the notation of *aristopy*. 
A detailed description of the code is provided in the 
[documentation](https://aristopy.readthedocs.io/en/latest/model_to_get_started.html). 

```python
import aristopy as ar

# Create basic energy system instance
es = ar.EnergySystem(
    number_of_time_steps=3, hours_per_time_step=1,
    interest_rate=0.05, economic_lifetime=20)

# Add a gas source, two different conversion units and sinks
gas_source = ar.Source(
    ensys=es, name='gas_source', commodity_cost=20, outlet=ar.Flow('Fuel'))

gas_boiler = ar.Conversion(
    ensys=es, name='gas_boiler', basic_variable='Heat',
    inlet=ar.Flow('Fuel', 'gas_source'), outlet=ar.Flow('Heat', 'heat_sink'),
    capacity_max=150, capex_per_capacity=60e3,
    user_expressions='Heat == 0.9 * Fuel')

chp_unit = ar.Conversion(
    ensys=es, name='chp_unit', basic_variable='Elec',
    inlet=ar.Flow('Fuel', 'gas_source'),
    outlet=[ar.Flow('Heat', 'heat_sink'), ar.Flow('Elec', 'elec_sink')],
    capacity_max=100, capex_per_capacity=600e3,
    user_expressions=['Heat == 0.5 * Fuel',
                      'Elec == 0.4 * Fuel'])

heat_sink = ar.Sink(
    ensys=es, name='heat_sink', inlet=ar.Flow('Heat'),
    commodity_rate_fix=ar.Series('heat_demand', [100, 200, 150]))

elec_sink = ar.Sink(
    ensys=es, name='elec_sink', inlet=ar.Flow('Elec'), commodity_revenues=30)

# Run the optimization
es.optimize(solver='cbc', results_file='results.json')

# Plot some results
plotter = ar.Plotter('results.json')
plotter.plot_operation('heat_sink', 'Heat', lgd_pos='lower center',
                       bar_lw=0.5, ylabel='Thermal energy [MWh]')
plotter.plot_objective(lgd_pos='lower center')
```

The method *plot_operation* returns a mixed bar and line plot that visualizes 
the operation of a component based on a selected commodity.

<img src="https://raw.githubusercontent.com/sbruche/aristopy/master/docs/misc/operation_plot.png" alt="operation plot" width="600"/>

The method *plot_objective* returns a bar chart that summarizes the cost
contributions of each component to the overall objective function value 
(net present value).

<img src="https://raw.githubusercontent.com/sbruche/aristopy/master/docs/misc/objective_plot.png" alt="objective plot" width="600"/>

## Citing and Contributing
You are welcome to test aristopy and use it for your purposes. If you
publish results based on the application of the package, please
cite this GitHub repository or the [project documentation](
https://aristopy.readthedocs.io/en/latest/index.html) on readthedocs.org.

If you have questions, found a bug, or want to contribute to the development
of *aristopy*, you are invited to open an issue or contact the developers
(stefan-bruche@tu-berlin.de).

## License
[MIT License](https://opensource.org/licenses/MIT)

Copyright (c) 2020 Stefan Bruche (TU Berlin)

## Acknowledgement
This work was developed during the research project "MINLP-Optimization of
Design and Operation of Complex Energy Systems", funded by the German Federal
Ministry for Economic Affairs and Energy (project reference number 03ET4053A).
The funding is gratefully acknowledged.

<a href="https://www.energietechnik.tu-berlin.de/menue/forschung/energiesystemanalyse_und_optimierung/oeb_ensys/">
<img src="https://raw.githubusercontent.com/sbruche/aristopy/master/docs/misc/bmwi_logo.png" alt="BMWi Logo" width="200px"></a>
