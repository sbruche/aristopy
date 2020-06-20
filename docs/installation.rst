############
Installation
############

Before you can create your first optimization model with *aristopy* you need to
install the following three essential components:

1. Python and IDE
2. Aristopy package
3. Solver


**Python and IDE**

The first requirement for using *aristopy* is a working installation of Python
on your machine. *Aristopy* is currently tested with Python 3.6 and 3.7.
Please use one of the many good tutorials on the internet if you need help with
the Python installation. Make sure to add the path of the installation to your
system's environment variables to simplify the call of python, pip,
etc. from the command line. To enhance your workflow with Python, we also
recommend installing an integrated development environment (IDE).
PyCharm has served well for us, but there are numerous other useful and
free software tools available.


**Aristopy package**

You can easily install *aristopy* in your current environment via pip by using
the following command: ::

    >> pip install aristopy

Alternatively, you can create a clone of aristopy's GitHub repository
(provided git is installed) in a local directory of your machine ::

    >> git clone https://github.com/sbruche/aristopy.git

or download a zipped version directly from the `GitHub page
<https://github.com/sbruche/aristopy.git>`_.

After that, you need to go to your local directory and install *aristopy* by
running the setup-file with python ::

    >> python setup.py install

or using pip install [#]_. ::

    >> pip install -e .[dev]


**Solver**

.. note::
    The installation of *aristopy* does not include any solvers. They need to
    be obtained separately, in accordance with the properties of your model,
    the availability of licenses, and your specific preferences.

The availability of a mathematical solver is essential to generate results for
your optimization problem. You need to ensure the solver of choice is suitable
for your model's mathematical class (e.g., if you added non-linear correlations,
you need to use a solver for non-linear problems). We refer to the common
literature for further information on mathematical modeling in general.

There is a great variety of solvers available on the market.
For the use with *aristopy* you have to consider that the solver interface must
be usable for the underlying optimization package
`Pyomo <https://pyomo.readthedocs.io/en/stable/>`_.
A useful way to start is downloading the open-source solvers, available for
free from `AMPL <https://ampl.com/products/solvers/open-source/>`_.
For example, we recommend the solver CBC for mixed-integer problems (MILP)
and the solver ipopt for non-linear (NLP) problems.

If you have access to a license of the powerful, commercial MILP-solvers Gurobi
or CPLEX, you are encouraged to apply them to solve your *aristopy* model
(provided your model is not non-linear). Academic users may be eligible to
receive free licenses for both solvers.
Please note that all installed solvers must be locatable by *aristopy*.
Therefore, it is important to add the path to the solver executables to your
system's environment variables.

.. [#] Argument -e is optional for editable / development mode. Add [dev] if you
   also want to install the extra-dependencies, i.e. sphinx, pytest, etc.
