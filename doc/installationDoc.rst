###############
Getting started
###############

.. TODO: Rework and write with own words!
	In the following, instructions for installing and using aristopy on Windows are given. The installation
	instructions for installing and using aristopy on Linux/macOS systems are however quite similar and can be, hopefully
	easily, derived from the instructions below.

	**Python installation**

	aristopy runs on Python 3 platforms (i.e. Anaconda). Currently, it is advised not to use a Python version exceeding
	Python 3.7.

	**aristopy installation**

	Install via pip by typing

		$ pip install aristopy

	into the command prompt. Alternatively, download or clone a local copy of the repository to your computer

		$ git clone https://github.com/aristopy.git

	and install aristopy (in development mode) in the folder where the setup.py is located with

		$ pip install -e .

	or install directly via python as

		$ python setup.py install

	**Installation of additional packages**

	The Python packages `tsam <https://github.com/FZJ-IEK3-VSA/tsam>`_ and `PYOMO <http://www.pyomo.org/>`_ should be
	installed by pip alongside aristopy.

	**Installation of an optimization solver**

	In theory many solvers can be used (e.g. `GUROBI <http://www.gurobi.com/>`_  or
	`GLPK <https://sourceforge.net/projects/winglpk/files/latest/download>`_). For the installation of GUROBI, follow
	the instructions on the solver's website. GUROBI has, if applicable, an academic license option. For installation
	of GLPK, move the downloaded folder to a desired location. Then, manually append the Environment Variable *Path*
	with the absolute path leading to the folder in which the glpsol.exe is located (c.f. w32/w64 folder, depending on
	operating system type).