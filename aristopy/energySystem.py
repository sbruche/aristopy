#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
**The EnergySystem class**

* Last edited: 2020-06-06
* Created by: Stefan Bruche (TU Berlin)
"""
import os
import time
import json
from collections import OrderedDict

import pandas as pd
import pyomo.environ as pyomo
import pyomo.network as network
import pyomo.opt as opt
from tsam.timeseriesaggregation import TimeSeriesAggregation

from aristopy import utils, logger


class EnergySystem:
    """
    The EnergySystem class is aristopy's main model container. An instance of
    the EnergySystem class holds the modeled components, the overall pyomo
    model and the results of the optimization.
    The EnergySystem class provides features to built and solve the
    optimization problem, manipulate the associated component models, and
    process the results of the optimization. The implemented class methods are:

    - Perform clustering of the implemented time series data (**cluster**)
    - Declare the pyomo optimization model (**declare_model**)
    - Call the main optimization routine (**optimize**)
    - Relax the integrality of binary variables (**relax_integrality**)
    - Edit properties of component variables, e.g., change bounds or domains
      (**edit_component_variables**)
    - Reset component variables after applying changes, e.g., relaxation
      (**reset_component_variables**)
    - Export and import configurations, i.e. component existences and capacities
      (**export_component_configuration**, **import_component_configuration**)
    - Create integer-cut-constraints to exclude the current design solution
      from the solution space and enforce a new design in subsequent model runs
      (**add_design_integer_cut_constraint**)
    - Add variables, constraints and objective function contributions directly
      to the main pyomo model, outside of the component declaration
      (**add_variable** and **add_constraint** and
      **add_objective_function_contribution**)
    """
    def __init__(self, number_of_time_steps=8760, hours_per_time_step=1,
                 interest_rate=0.05, economic_lifetime=20, logging=None):
        """
        Initialize an instance of the EnergySystem class.

        :param number_of_time_steps: Number of considered time steps for
            modeling the dispatch problem. With "hours_per_time_step" the
            share of the modeled year can be calculated. In this way, the cost
            of each time step is scaled and included in the objective function.
            |br| *Default: 8760*
        :type number_of_time_steps: int (>0)

        :param hours_per_time_step: Number of hours per modeled time step.
            |br| *Default: 1*
        :type hours_per_time_step: int (>0)

        :param interest_rate: Value to calculate the present value factor of a
            cost rate that occurs in the future.
            |br| *Default: 0.05 (corresponds to 5%)*
        :type interest_rate: float, int (>=0)

        :param economic_lifetime: Years to consider for calculating the net
            present value of an investment with annual incoming and outgoing
            cash flows.
            |br| *Default: 20*
        :type economic_lifetime:  int (>0)

        :param logging: Specify the behavior of the logging by setting an own
            Logger class instance. User can decide where to log (file/console)
            and what to log (see description of aristopy "Logger").
            |br| *Default: None (display minimal logging in the console)*
        :type logging: None or instance of aristopy's "Logger" class
        """
        # Check user input:
        utils.check_energy_system_input(
            number_of_time_steps, hours_per_time_step, interest_rate,
            economic_lifetime, logging)

        # **********************************************************************
        #   Logging
        # **********************************************************************
        # If no logger instance is passed to the "logging" keyword a default
        # logger is initialized. This will only display errors on the console.
        # Otherwise the passed logger instance is used and a logger for the
        # instance of the energy system class is initialized on "self.log"
        if logging is None:
            self.logger = logger.Logger(default_log_handler='stream',
                                        default_log_level='ERROR')
        else:
            self.logger = logging
        self.log = self.logger.get_logger(self)

        # **********************************************************************
        #   Time and clustering
        # **********************************************************************
        self.number_of_time_steps = number_of_time_steps
        self.hours_per_time_step = hours_per_time_step
        self.number_of_years = number_of_time_steps * hours_per_time_step/8760.0

        # Initialization: Overwritten if 'cluster' function is called
        self.periods = [0]
        self.periods_order = [0]
        self.period_occurrences = [1]
        self.number_of_time_steps_per_period = number_of_time_steps
        self.inter_period_time_steps = [0, 1]  # one before & after only period

        # Flag 'is_data_clustered' indicates if the function 'cluster' has been
        # called before. The flag is reset to False if new components are added.
        self.is_data_clustered = False
        # 'typical_periods' is altered by function 'cluster' to an array ranging
        # from 0 to number_of_typical_periods-1.
        self.typical_periods = [0]

        # **********************************************************************
        #   Economics
        # **********************************************************************
        # The economic criterion net present value represents the objective
        # function value to be maximized. Hence, a present value factor (pvf) is
        # required to calculate the present value of an annuity. The global
        # parameters interest rate and economic lifetime of the energy system
        # investment are used to this end.
        self.pvf = sum(1 / (1 + interest_rate)**n
                       for n in range(1, economic_lifetime+1))

        # **********************************************************************
        #   Optimization
        # **********************************************************************
        # The parameter 'model' holds the pyomo ConcreteModel instance with
        # sets, parameters, variables, constraints and the objective function.
        # It is None during initialization and changed when the functions
        # 'optimize', or 'declare_model' are called.
        # Before the model instance is optimized, a solver instance is assigned
        # to the "solver" attribute. It also stores basic results of the run.
        # The "is_model_declared" flag indicates if the model instance is
        # already declared.
        # The "is_persistent_model_declared" flag states if the model has been
        # declared and assigned to a persistent solver instance.
        self.model = None
        self.run_info = {'solver_name': '',
                         'time_limit': None, 'optimization_specs': '',
                         'model_build_time': 0, 'model_solve_time': 0,
                         'upper_bound': 0, 'lower_bound': 0, 'sense': '',
                         'solver_status': '', 'termination_condition': ''}
        self.solver = None
        self.is_model_declared = False
        self.is_persistent_model_declared = False

        # **********************************************************************
        #   Components
        # **********************************************************************
        # 'components' is a dictionary {component name: component object itself}
        # in which all components of the EnergySystem instance are stored.
        # The pyomo block model object (stored variables and constraints) of a
        # component instance can be accessed via its "block" attribute.
        self.components = {}

        # The 'component_connections' is a dict that stores the connections of
        # the component instances of the energy system model. It is formed from
        # the specified inlets and outlets and the connecting commodity:
        # {arc_name: [source instance, destination instance, commodity_name]}
        self.component_connections = {}

        # 'component_configuration' is a pandas Dataframe to store basic
        # information about the availability and capacity of the modelled
        # components. It is used to export / import the configuration results.
        self.component_configuration = pd.DataFrame(
            index=[utils.BI_EX, utils.BI_MODULE_EX, utils.CAP])

        # DataFrames and dictionaries to store additionally added pyomo objects
        # (variables and constraints) and objective function contributions.
        self.added_constraints = pd.DataFrame(index=['has_time_set',
                                                     'alternative_set', 'rule'])
        self.added_variables = pd.DataFrame(index=['domain', 'has_time_set',
                                                   'alternative_set', 'init',
                                                   'ub', 'lb', 'pyomo'])
        self.added_objective_function_contributions = {}
        self.added_obj_contributions_results = {}

        self.log.info('Initializing EnergySystem completed!')

    def __repr__(self):
        # Define class format for printing and logging
        return '<EnSysMo: "id=%s..%s">' % (hex(id(self))[:3],
                                           hex(id(self))[-3:])

    def add_variable(self, var):
        """
        Function to manually add pyomo variables to the main pyomo model
        (ConcreteModel: model) of the energy system instance via instances
        of aristopy's Var class. The attributes of the variables are stored in
        DataFrame "added_variables" and later initialized during the call of
        function 'optimize', or 'declare_model'.

        :param var: Instances of aristopy's Var class (single or in list)
        """
        self.log.info('Call of function "add_variable"')

        # Check the correctness of the user input
        var_list = utils.check_add_vars_input(var)

        for v in var_list:
            # Wrap everything up in a pandas Series
            series = pd.Series({'has_time_set': v.has_time_set,
                                'alternative_set': v.alternative_set,
                                'domain': v.domain, 'init': v.init,
                                'ub': v.ub, 'lb': v.lb, 'pyomo': None})
            # Add the Series with new name to DataFrame "added_variables"
            self.added_variables[v.name] = series

    def add_constraint(self, rule, name=None, has_time_set=True,
                       alternative_set=None):
        """
        Function to manually add constraints to the main pyomo model after the
        instance has been created. The attributes are stored in the DataFrame
        'added_constraints' and later initialized during the call of function
        'optimize', or 'declare_model'.

        :param rule: A Python function that specifies the constraint with a
            equality or inequality expression. The rule must hold at least
            two arguments: First the energy system instance it is added to (in
            most cases: self), second the ConcreteModel of the instance (model).
            Additional arguments represent sets (e.g., time).
        :type rule: function

        :param name: Name (identifier) of the added constraint. The rule name is
            used if no name is specified.
            |br| *Default: None*
        :type name: str

        :param has_time_set: Is True if the time set of the energy system model
            is also a set of the added constraint.
            |br| *Default: True*
        :type has_time_set: bool

        :param alternative_set: Alternative constraint sets can be added here
            via iterable Python objects (e.g. list).
            |br| *Default: None*
        """
        self.log.info('Call of function "add_constraint"')

        # Check the correctness of the user input
        utils.check_add_constraint(rule, name, has_time_set, alternative_set)
        # The rule name is used as constraint identifier if no name is given
        if name is None:
            name = rule.__name__
        # Put everything together in a pandas Series
        series = pd.Series({'has_time_set': has_time_set,
                            'alternative_set': alternative_set,
                            'rule': rule})
        # Add the Series to the DataFrame "added_constraints"
        self.added_constraints[name] = series

    def add_objective_function_contribution(self, rule, name=None):
        """
        Additional objective function contributions can be added with this
        method. The method requires a Python function input that takes the main
        pyomo model (ConcreteModel: model) and returns a single (scalar) value.

        :param rule: A Python function returning a scalar value which is added
            to the objective function of the model instance. The rule must hold
            exactly two arguments: The energy system instance it is added to (in
            most cases: self), second the ConcreteModel of the instance (model).
        :type rule: function

        :param name: Name (identifier) of the added objective function
            contribution. The rule name is used if no name is specified.
            |br| *Default: None*
        :type name: str
        """
        self.log.info('Call of function "add_objective_function_contribution"')

        # Check the input:
        assert isinstance(name, (str, type(None))), '"name" should be a string!'
        if not callable(rule):
            raise TypeError('The "rule" needs to hold a callable object!')
        if name is None:
            name = rule.__name__
        # Add the rule and the name to a dictionary of the EnergySystem instance
        self.added_objective_function_contributions[name] = rule

    def cluster(self, number_of_typical_periods=4,
                number_of_time_steps_per_period=24,
                cluster_method='hierarchical',
                **kwargs):
        """
        Method for the aggregation and clustering of time series data. First,
        the time series data and their respective weights are collected from all
        components and split into pieces with equal length of
        'number_of_time_steps_per_period'.
        Subsequently, a clustering method is called and each period is assigned
        to one of 'number_of_typical_periods' typical periods. The clustered
        data is later stored in the components.
        The package `tsam <https://github.com/FZJ-IEK3-VSA/tsam>`_ (time series
        aggregation module) is used to perform the clustering.
        The clustering algorithm can be controlled by adding required keyword
        arguments (using 'kwargs' parameter). To learn more about tsam and
        possible keyword arguments see the package `documentation
        <https://tsam.readthedocs.io/en/latest/index.html>`_.

        :param number_of_typical_periods: Number of typical periods to be
            clustered. |br| *Default: 4*
        :type number_of_typical_periods: int (>0)

        :param number_of_time_steps_per_period: Number of time steps per period
            |br| *Default: 24*
        :type number_of_time_steps_per_period: int (>0)

        :param cluster_method: Name of the applied clustering method (e.g.,
            'k_means'). See the tsam documentation for all possible options.
            |br| *Default: 'hierarchical'*
        :type cluster_method: str
        """
        # Check input arguments
        utils.check_cluster_input(number_of_typical_periods,
                                  number_of_time_steps_per_period,
                                  self.number_of_time_steps)

        time_start = time.time()
        self.log.info('Start clustering with %s typical periods and %s time '
                      'steps per period.' % (number_of_typical_periods,
                                             number_of_time_steps_per_period))

        # Get time series data and their respective weights from all components
        # and collect them in two dictionaries
        time_series_data, time_series_weights = {}, {}
        for comp in self.components.values():
            if comp.number_in_group == 1:  # Add only once per group
                data, weights = comp.get_time_series_data_for_aggregation()
                time_series_data.update(data)
                time_series_weights.update(weights)

        # Convert data dictionary to pandas DataFrame
        time_series_data = pd.DataFrame.from_dict(time_series_data)
        # Specific index is not relevant, but tsam requires a uniform index
        time_series_data.index = \
            pd.date_range('2050-01-01 00:30:00',
                          periods=self.number_of_time_steps,
                          freq=(str(self.hours_per_time_step) + 'H'),
                          tz='Europe/Berlin')
        # Reindex axis for reproducibility of TimeSeriesAggregation results
        time_series_data = time_series_data.reindex(
            sorted(time_series_data.columns), axis=1)

        # Set up instance of tsam's TimeSeriesAggregation class and cluster data
        cluster_class = TimeSeriesAggregation(
            timeSeries=time_series_data,
            noTypicalPeriods=number_of_typical_periods,
            hoursPerPeriod=
            number_of_time_steps_per_period * self.hours_per_time_step,
            clusterMethod=cluster_method,
            weightDict=time_series_weights,
            **kwargs)

        # Store clustered time series data in the components
        data = pd.DataFrame.from_dict(cluster_class.clusterPeriodDict)
        for comp in self.components.values():
            comp.set_aggregated_time_series_data(data)

        self.typical_periods = cluster_class.clusterPeriodIdx
        self.number_of_time_steps_per_period = number_of_time_steps_per_period

        self.periods = list(range(int(
            self.number_of_time_steps / number_of_time_steps_per_period)))
        self.inter_period_time_steps = list(range(int(
            self.number_of_time_steps / number_of_time_steps_per_period) + 1))

        self.periods_order = cluster_class.clusterOrder
        self.period_occurrences = [
            (self.periods_order == tp).sum() for tp in self.typical_periods]

        # Set cluster flag to True
        self.is_data_clustered = True
        self.log.info("    Time required for clustering: %.2f sec"
                      % (time.time() - time_start))
        # Debugging:
        self.log.debug('typical_periods: %s' % self.typical_periods)
        self.log.debug('periods: %s' % self.periods)
        self.log.debug('inter_period_time_steps: %s'
                       % self.inter_period_time_steps)
        self.log.debug('periods_order: %s' % self.periods_order)
        self.log.debug('period_occurrences: %s' % self.period_occurrences)

    def declare_time_sets(self, model, use_clustered_data):
        """
        Initialize time parameters and four different time sets.

        The "time_set" represents the general set. The index holds tuples of
        periods and time steps inside of these periods. In case the optimization
        is performed without time series aggregation, the set runs from [(0,0),
        (0,1), ..to.., (0,number_of_time_steps-1)].
        Otherwise: [(0,0), ..., (0,number_of_time_steps_per_period-1), (1,0),
        ..., (number_of_typical_periods-1, number_of_time_steps_per_period-1)].
        |br| The set "intra_period_time_set" holds tuples of periods and points
        in time before, after or between regular time steps inside of a period.
        Hence, the second value runs from 0 to "number_of_time_steps" (without
        aggregation) or "number_of_time_steps_per_period" respectively. |br|
        The third set "intra_period_time_set" is one-dimensional and ranges from
        0 to the overall number of periods plus 1. So if no aggregation is used
        it has only two entries [0, 1]. Otherwise is is ranging from 0 to
        (1 + number_of_time_steps / number_of_time_steps_per_period). |br|
        The "typical_periods_set" is a set ranging from 0 to the number of
        typical periods. If no aggregation is used it only holds 0. |br|

        E.g.: Case with 2 periods and 3 time steps per period |br|
        1) ______(0,0)___(0,1)___(0,2)___________(1,0)___(1,1)___(1,2)_____ |br|
        2) __(0,0)___(0,1)___(0,2)___(0,3)___(1,0)___(1,1)___(1,2)___(1,3)_ |br|
        3) 0___________________________    1    __________________________2 |br|
        4) ______________  0  ___________________________  1  _____________ |br|
        1) time_set, 2) intra_period_time_steps_set,
        3) inter_period_time_steps_set, 4) typical_periods_set |br|

        :param model: Pyomo model instance holding sets, variables, constraints
            and the objective function.
        :type model: pyomo ConcreteModel

        :param use_clustered_data: Declare optimization model with original
            (full scale) time series (=> False) or clustered data (=> True).
        :type use_clustered_data: bool
        """
        self.log.info('    Declare time sets')

        # Set the time series data for all modelled components
        for comp in self.components.values():
            comp.set_time_series_data(use_clustered_data)

        # Reset time-relevant attributes and initialize sets if full scale
        # problem is considered.
        if not use_clustered_data:
            self.periods = [0]
            self.periods_order = [0]
            self.period_occurrences = [1]
            self.number_of_time_steps_per_period = self.number_of_time_steps
            self.inter_period_time_steps = [0, 1]
            self.typical_periods = [0]

            # Define sets: Only period "0" exists
            def init_time_set(m):
                return ((0, t) for t in range(self.number_of_time_steps))

            def init_intra_period_time_set(m):
                return ((0, t) for t in range(self.number_of_time_steps + 1))

        else:
            self.log.info('    Aggregated time series data detected (number of '
                          'typical periods: %s, number of time steps per '
                          'period: %s' % (len(self.typical_periods),
                                          self.number_of_time_steps_per_period))

            # Define sets
            def init_time_set(m):
                return ((p, t) for p in self.typical_periods
                        for t in range(self.number_of_time_steps_per_period))

            def init_intra_period_time_set(m):
                return ((p, t) for p in self.typical_periods
                        for t in range(self.number_of_time_steps_per_period+1))

        # Initialize the four time sets:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # E.g.: Case with 2 periods and 3 time steps per period
        # 1) ______(0,0)___(0,1)___(0,2)___________(1,0)___(1,1)___(1,2)______
        # 2) __(0,0)___(0,1)___(0,2)___(0,3)___(1,0)___(1,1)___(1,2)___(1,3)__
        # 3) 0_______________________________1_______________________________2
        # 4) ________________0_______________________________1________________
        # 1) time_set, 2) intra_period_time_steps_set,
        # 3) inter_period_time_steps_set, 4) typical_periods_set

        model.time_set = pyomo.Set(
            dimen=2, initialize=init_time_set, ordered=True)

        model.intra_period_time_set = pyomo.Set(
            dimen=2, initialize=init_intra_period_time_set, ordered=True)

        model.inter_period_time_set = pyomo.Set(
            initialize=self.inter_period_time_steps, ordered=True)

        model.typical_periods_set = pyomo.Set(
            initialize=self.typical_periods, ordered=True)

    def declare_objective(self, model):
        """
        Method to declare the objective function of the optimization problem.
        The objective function contributions (CAPEX and OPEX) are calculated in
        the components itself, and collected and summarized by the
        'declare_objective' function of the EnergySystem instance.
        The objective function of the optimization is the maximization of the
        net present value.

        :param model: Pyomo model instance holding sets, variables, constraints
            and the objective function.
        :type model: pyomo ConcreteModel
        """
        self.log.info('    Declare objective function')

        def summarize_extra_objective_function_contributions():
            for key, val in self.added_objective_function_contributions.items():
                obj_rule = getattr(self, key)  # get the function
                obj_value = obj_rule(self.model)  # get returned value
                # Add the values to the results dict (used to export results)
                self.added_obj_contributions_results[key] = obj_value
            return sum(self.added_obj_contributions_results.values())

        def objective(m):
            NPV = sum(comp.get_objective_function_contribution(self, model)
                      for comp in self.components.values()) \
                  + summarize_extra_objective_function_contributions()
            return NPV
        model.Obj = pyomo.Objective(rule=objective, sense=pyomo.maximize)

    def declare_model(self, use_clustered_data=False,
                      declare_persistent=False,
                      persistent_solver='gurobi_persistent'):
        """
        Declare the pyomo optimization model of the EnergySystem instance.
        First, all component connections are established by using their input
        and output specifications and the plausibility of connections is
        validated. Then, the ConcreteModel instance is created and time sets,
        component model blocks, variables, ports, constraints, arcs, and the
        objective function are added.

        :param use_clustered_data: Declare optimization model with original
            (full scale) time series (=> False) or clustered data (=> True).
            |br| *Default: False*
        :type use_clustered_data: bool

        :param declare_persistent: States if a persistent model instance should
            be formed. In this case, after model declaration a persistent solver
            instance is created and the declared model instance is assigned.
            |br| *Default: False*
        :type declare_persistent: bool

        :param persistent_solver: Name of the persistent solver to be used.
            Possible options are "gurobi_persistent" and "cplex_persistent".
            Is ignored if keyword "declare_persistent" is False.
            |br| *Default: 'gurobi_persistent'*
        :type persistent_solver: str
        """
        self.log.info('Declare optimization model with '
                      'use_clustered_data=%s and declare_persistent=%s'
                      % (use_clustered_data, declare_persistent))

        time_start = time.time()

        # Check inputs:
        persistent_solver = persistent_solver.lower()
        utils.check_declare_model_input(
            use_clustered_data, self.is_data_clustered,
            declare_persistent, persistent_solver)

        # Initialize flags of model status
        self.is_model_declared = False
        self.is_persistent_model_declared = False

        # ######################################################################
        #   C O M P O N E N T   C O N N E C T I O N S
        # ######################################################################
        # Helper functions:
        # -----------------
        def component_not_found(c_name):
            return('\nThe component "{}" or its instances cannot be found in '
                   'the EnergySystem model.\nThe components considered are: {}'
                   '\n Please check the names of your components and inlet and '
                   'outlet specs!'.format(c_name, list(self.components)))

        def set_connection(source, destination, commodity):
            """
            Add connections (arcs) to the "component_connections" dict.
            Update the "var_connections" dictionaries in the components:
            {'variable name': 'connected arc name'}
            "var_connections" is used for plotting purposes.
            """
            # Add a new connection to "component_connections" if not available
            arc_name = 'arc-{}-{}-{}'.format(
                source.name, destination.name, commodity)

            if arc_name not in self.component_connections:
                self.component_connections[arc_name] = [
                    source, destination, commodity]

            # Get the commodity variables from comp. and raise if not available
            if source.outlet_commod_and_var_names.get(commodity) is None:
                raise ValueError('Commodity "%s" not found at outlet of "%s"'
                                 % (commodity, source.name))
            source_var = source.outlet_commod_and_var_names[commodity]
            if destination.inlet_commod_and_var_names.get(commodity) is None:
                raise ValueError('Commodity "%s" not found at inlet of "%s"'
                                 % (commodity, destination.name))
            destination_var = destination.inlet_commod_and_var_names[commodity]

            # Update dictionary "var_connections" in the source component
            if source.var_connections.get(source_var) is None:
                source.var_connections[source_var] = [arc_name]
            else:
                arc_names = source.var_connections[source_var]
                if arc_name not in arc_names:
                    arc_names.append(arc_name)
                    source.var_connections[source_var] = arc_names

            # Update dictionary "var_connections" in the destination component
            if destination.var_connections.get(destination_var) is None:
                destination.var_connections[destination_var] = [arc_name]
            else:
                arc_names = destination.var_connections[destination_var]
                if arc_name not in arc_names:
                    arc_names.append(arc_name)
                    destination.var_connections[destination_var] = arc_names

        # **********************************************************************
        #   Establish component connections from "inlet" and "outlet"
        # **********************************************************************
        for comp_name, comp in self.components.items():
            # Loop over inlets ...
            for flow in comp.inlet:
                if flow.link is not None:
                    # loop again over all components:
                    found_src = False
                    for c_src in self.components.values():
                        if c_src.group_name == flow.link:
                            found_src = True  # found source
                            set_connection(c_src, comp, flow.commodity)
                    if not found_src:
                        raise ValueError(component_not_found(flow.link))
            # Loop over outlets ...
            for flow in comp.outlet:
                if flow.link is not None:
                    # loop again over all components:
                    found_dest = False
                    for c_dest in self.components.values():
                        if c_dest.group_name == flow.link:
                            found_dest = True  # found destination
                            set_connection(comp, c_dest, flow.commodity)
                    if not found_dest:
                        raise ValueError(component_not_found(flow.link))

        # **********************************************************************
        #   Check connections
        # **********************************************************************
        for name, comp in self.components.items():

            # Check that non-conversion comp. have only one commodity at ports!
            if comp.__class__.__name__ != 'Conversion' and \
                    len(comp.commodities) > 1:
                raise ValueError('Commodity error in "%s". Found more than one '
                                 'commodity: "%s"' % (name, comp.commodities))

            # Check that number of inlets and outlets is correct:
            nbr_of_inlets = len(comp.inlet_commod_and_var_names)
            nbr_of_outlets = len(comp.outlet_commod_and_var_names)
            # Conversion component have at least one inlet and one outlet
            if comp.__class__.__name__ == 'Conversion' \
                    and (nbr_of_inlets < 1 or nbr_of_outlets < 1):
                raise ValueError('"%s" needs at least one inlet and one outlet!'
                                 % name)
            # Source, Storage, Bus components need exactly one outlet
            if comp.__class__.__name__ in ['Source', 'Storage', 'Bus'] and \
                    nbr_of_outlets != 1:
                raise ValueError('"%s" needs one outlet, but "%s" were found!'
                                 % (name, nbr_of_outlets))
            # Sink, Storage, Bus components need exactly one inlet
            if comp.__class__.__name__ in ['Sink', 'Storage', 'Bus'] and \
                    nbr_of_inlets != 1:
                raise ValueError('"%s" needs one inlet, but "%s" were found!'
                                 % (name, nbr_of_inlets))
            # Sinks don't have outlets
            if comp.__class__.__name__ == 'Sink' and nbr_of_outlets != 0:
                raise ValueError('Sink "%s" cannot have outlets!' % name)
            # Sources don't have inlets
            if comp.__class__.__name__ == 'Source' and nbr_of_inlets != 0:
                raise ValueError('Source "%s" cannot have inlets!' % name)

            # Raise if component has inlet or outlet commodity without
            # connection to an arc
            for commod, var in comp.inlet_commod_and_var_names.items():
                if comp.var_connections.get(var) is None:
                    raise ValueError('Inlet commodity "%s" of component "%s" is'
                                     ' unconnected!' % (commod, comp.name))
            for commod, var in comp.outlet_commod_and_var_names.items():
                if comp.var_connections.get(var) is None:
                    raise ValueError('Outlet commodity "%s" of component "%s" '
                                     'is unconnected!' % (commod, comp.name))

        # **********************************************************************
        #   Check basic variable
        # **********************************************************************
        # Usually basic variables are referring to inlet or outlet commodities
        # which are time-related. Scalar variables can also be selected under
        # certain circumstances. Check permissibility of parameter combinations:
        for name, comp in self.components.items():
            # Fine, if basic variable has a time set (--> go to next comp.)
            if comp.variables[comp.basic_variable]['has_time_set']:
                continue
            # Conversion w/o 'opex_operation' and w/o binary operation var.: OK
            elif comp.__class__.__name__ == 'Conversion' and \
                    comp.opex_operation == 0 and not comp.has_bi_op:
                continue
            # Sink or Source components w/o 'opex_operation', w/o binary
            # operation var. and w/o commodity_rates, _cost or _rev. are also OK
            elif comp.__class__.__name__ in ['Sink', 'Source'] and \
                    comp.opex_operation == 0 and not comp.has_bi_op and \
                    comp.op_rate_min is None and comp.op_rate_max is None and \
                    comp.op_rate_fix is None and comp.commodity_cost == 0 and \
                    comp.commodity_cost_time_series is None and \
                    comp.commodity_revenues == 0 and \
                    comp.commodity_revenues_time_series is None:
                continue
            else:
                raise Exception(
                    'The basic variable of component "%s" works without a time '
                    'set. This functionality is only allowed for Sink, Source ' 
                    'and Conversion components w/o binary operation variables, ' 
                    'opex_operation, commodity_rates, _cost, and _revs!' % name)

        # **********************************************************************
        #   Add user expressions
        # **********************************************************************
        # Fill the "user_expressions_dict" with data from comp. initialization
        # (old data / variables are overwritten in 'comp.user_expressions_dict')
        for comp in self.components.values():
            for expr in comp.user_expressions:
                # Create a name for the expression by removing all spaces
                expr_name = expr.replace(' ', '')
                # Disassemble expr into its parts -> e.g. ['Q', '>=', '100']
                expr_pieces = utils.disassemble_user_expression(expr)
                # Append name and list of pieces to the 'user_expressions_dict'
                comp.user_expressions_dict[expr_name] = expr_pieces

        # ######################################################################
        #   P Y O M O   M O D E L
        # ######################################################################
        # Create a pyomo ConcreteModel instance to store the sets, variables,
        # constraints, objective function, etc.
        self.model = pyomo.ConcreteModel()
        # Duals are stored with the name 'dual' in the model instance (might be
        # interesting for analyzing optimization results of linear models).
        self.model.dual = pyomo.Suffix(direction=pyomo.Suffix.IMPORT)

        # **********************************************************************
        #   Declare time sets
        # **********************************************************************
        self.declare_time_sets(self.model, use_clustered_data)

        # **********************************************************************
        #   Declare component blocks, variables, ports and constraints
        # **********************************************************************
        for comp in self.components.values():
            comp.declare_component_model_block(model=self.model)
            comp.declare_component_variables(model=self.model)
            comp.declare_component_ports()
            comp.declare_component_user_constraints(model=self.model)
            comp.declare_component_constraints(ensys=self, model=self.model)

        # **********************************************************************
        #   Declare arcs
        # **********************************************************************
        for arc_name, connection in self.component_connections.items():
            # Get the connected component class instances and the commodity name
            src, dest, commod = connection[0], connection[1], connection[2]
            # Get the ports in the component model blocks for each variable
            outlet = getattr(src.block, 'outlet_' + commod)
            inlet = getattr(dest.block, 'inlet_' + commod)
            # Create an arc to connect two ports
            setattr(self.model, arc_name, network.Arc(src=outlet, dest=inlet))

        # Call model transformation factory: Expand the arcs
        pyomo.TransformationFactory("network.expand_arcs").apply_to(self.model)

        # Workaround for Pyomo-Versions before 5.6.9:
        # Get all constraints that end with '_split' and deactivate them
        # --> Workaround for Port.Extensive function with indexed variables
        # See: https://groups.google.com/forum/#!topic/pyomo-forum/LaoKMhyu9pA
        # for c in model.component_objects(pyomo.Constraint, active=True):
        #     if c.name.endswith('_split'):
        #         c.deactivate()

        # **********************************************************************
        #   Declare additional pyomo objects, added by:
        #       * add_variable
        #       * add_constraint
        #       * add_objective_function_contribution
        # **********************************************************************
        # Additional variables:
        # ~~~~~~~~~~~~~~~~~~~~~
        for var_name in self.added_variables:
            # Get the pandas series from DataFrame and the variable specs
            var_dict = self.added_variables[var_name]
            domain = getattr(pyomo, var_dict['domain'])
            bounds = (var_dict['lb'], var_dict['ub'])
            init = var_dict['init']
            # Differentiation between variables with and without time_set
            if var_dict['has_time_set']:
                setattr(self.model, var_name, pyomo.Var(
                    self.model.time_set, domain=domain, bounds=bounds,
                    initialize=init))
            elif var_dict['alternative_set'] is not None:
                setattr(self.model, var_name, pyomo.Var(
                    var_dict['alternative_set'], domain=domain,
                    bounds=bounds, initialize=init))
            else:
                setattr(self.model, var_name, pyomo.Var(
                    domain=domain, bounds=bounds, initialize=init))
            # Store variable in self.added_variables[var_name]['pyomo']
            pyomo_var = getattr(self.model, var_name)
            self.added_variables[var_name]['pyomo'] = pyomo_var

        # Additional constraints:
        # ~~~~~~~~~~~~~~~~~~~~~~~
        for con_name in self.added_constraints:
            con_dict = self.added_constraints[con_name]
            rule = con_dict['rule']
            # https://stackoverflow.com/questions/972/adding-a-method-to-an-existing-object-instance
            setattr(self, con_name, rule.__get__(self))
            con = getattr(self, con_name)
            # Differentiation between variables with and without time_set
            if con_dict['has_time_set']:
                setattr(self.model, con_name, pyomo.Constraint(
                    self.model.time_set, rule=con))
            elif con_dict['alternative_set'] is not None:
                setattr(self.model, con_name, pyomo.Constraint(
                    con_dict['alternative_set'], rule=con))
            else:
                setattr(self.model, con_name, pyomo.Constraint(rule=con))

        # Additional objective function contributions:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for name, rule in self.added_objective_function_contributions.items():
            setattr(self, name, rule.__get__(self))

        # **********************************************************************
        #   Declare empty constraint list for design integer cuts
        # **********************************************************************
        # Attribute "integer_cut_constraints" in the ConcreteModel is
        # initialized as an empty pyomo ConstraintList while declaring the
        # optimization problem. Integer cut constraints can be added with
        # function "add_design_integer_cut_constraint".
        self.model.integer_cut_constraints = pyomo.ConstraintList()

        # **********************************************************************
        #   Declare objective function
        # **********************************************************************
        self.declare_objective(self.model)

        # **********************************************************************
        #   Built persistent model instance (if "declare_persistent" is True)
        # **********************************************************************
        if declare_persistent:
            # Store persistent solver name
            self.run_info['solver_name'] = persistent_solver

            # Call the persistent solver and assign the solver to "self.solver"
            time_persistent_start = time.time()
            self.solver = opt.SolverFactory(persistent_solver)
            # Create a "gurobipy" model object and add the model to the solver
            self.solver.set_instance(self.model)
            # Set the flag 'is_persistent_model_declared' to True
            self.is_persistent_model_declared = True
            self.log.info('    Time to set instance to persistent solver: %.2f'
                          % (time.time() - time_persistent_start))

        # **********************************************************************
        # Set flag indicating the declaration status of the model to be True
        self.is_model_declared = True

        # Store the time to build the optimization model instance
        self.run_info['model_build_time'] = int(time.time() - time_start)
        self.log.info('    Time to declare optimization model: %.2f'
                      % (time.time() - time_start))

    # ==========================================================================
    #    O P T I M I Z E
    # ==========================================================================
    def optimize(self, declare_model=True, use_clustered_data=False,
                 declare_persistent=False,
                 persistent_solver='gurobi_persistent',
                 solver='gurobi', time_limit=None, optimization_specs='',
                 results_file='results.json', tee=True):
        """
        Call the optimization routine for the EnergySystem instance.
        First, a new pyomo model instance is built (if 'declare_model'=True),
        then the model is delivered to the specified solver and finally, the
        optimization results are exported to a json-file.

        :param declare_model: Declare a new instance of the pyomo ConcreteModel
            (=> True: Call function 'declare_model') or use a previously
            declared model instance if available (=> False).
            |br| *Default: True*
        :type declare_model: bool

        :param use_clustered_data: Declare optimization model with original
            (full scale) time series (=> False) or clustered data (=> True).
            |br| *Default: False*
        :type use_clustered_data: bool

        :param declare_persistent: States if a persistent model instance should
            be formed. In this case, after model declaration a persistent solver
            instance is created and the declared model instance is assigned.
            |br| *Default: False*
        :type declare_persistent: bool

        :param persistent_solver: Name of the persistent solver to be used.
            Possible options are "gurobi_persistent" and "cplex_persistent".
            Is ignored if keyword "declare_persistent" is False.
            |br| *Default: 'gurobi_persistent'*
        :type persistent_solver: str

        :param solver: Name of the applied solver (make sure the solver is
            available on your machine and aristopy can find the path to the
            solver executables).
            |br| *Default: 'gurobi'*
        :type solver: str

        :param time_limit: Limits the total optimization time (in seconds).
            If the time limit is reached, the solver returns with the best
            currently found solution. If None is specified, solver runs until
            the default time limit is exceeded (solver specific) or another
            criteria for abortion is triggered (e.g., reached 'MIPGap').
            |br| *Default: None*
        :type time_limit: int (>0) or None

        :param optimization_specs: Additional solver parameters can be set from
            a string. To find out more about possible solver options check the
            documentation of the applied solver (e.g., see `gurobi documentation
            <https://www.gurobi.com/documentation/9.0/refman/parameters.html>`_)
            |br| E.g.: 'Threads=1 MIPGap=0.01'  |br| *Default: ''*
        :type optimization_specs: str

        :param results_file: Name of the results file (required format: .json).
            |br| *Default: 'results.json'*
        :type results_file: str or None

        :param tee: Show solver output (on screen and/or logfile).
            |br| *Default: True*
        :type tee: bool
        """
        self.log.info('Call of function "optimize"')

        solver, persistent_solver = solver.lower(), persistent_solver.lower()

        if declare_model:
            self.declare_model(declare_persistent=declare_persistent,
                               persistent_solver=persistent_solver,
                               use_clustered_data=use_clustered_data)
        elif not self.is_model_declared:
            raise ValueError('The optimization model is not declared yet. Set '
                             'the argument "declare_model" to True or call the '
                             'function "declare_model" first.')

        time_start = time.time()

        # Check input arguments
        utils.check_optimize_input(use_clustered_data,
                                   declare_persistent, persistent_solver,
                                   self.is_data_clustered, solver,
                                   time_limit, optimization_specs)

        # Add basic input arguments to the 'run_info' dict
        self.run_info['time_limit'] = time_limit
        self.run_info['optimization_specs'] = optimization_specs

        # **********************************************************************
        #   Solve model
        # **********************************************************************
        # Call a solver from the SolverFactory if the model is not persistent
        if not self.is_persistent_model_declared:
            self.run_info['solver_name'] = solver
            self.solver = opt.SolverFactory(solver)

        # Set the time limit if specified (name depends on applied solver)
        time_limit_param_names = {
            'scip': 'limits/time', 'cbc': 'seconds', 'glpk': 'tmlim',
            'gurobi': 'TimeLimit', 'gurobi_persistent': 'TimeLimit',
            'cplex': 'timelimit', 'cplex_persistent': 'timelimit',
            'baron': 'MaxTime'}
        if self.run_info['time_limit'] is not None:
            if self.is_persistent_model_declared:
                param_name = time_limit_param_names[persistent_solver]
                self.solver.options[param_name] = time_limit
            elif not self.is_persistent_model_declared and \
                    solver in time_limit_param_names.keys():
                param_name = time_limit_param_names[solver]
                self.solver.options[param_name] = time_limit

        # Set further solver options and solve the model
        self.solver.set_options(optimization_specs)
        if not self.is_persistent_model_declared:
            self.log.info('Solve non-persistent model using %s' % solver)
            solver_info = self.solver.solve(self.model, tee=tee)
        else:
            self.log.info('Solve persistent model using %s' % solver)
            solver_info = self.solver.solve(tee=tee)

        # Show the solver summary on the screen
        # solver_info.write()

        # Add selected results to the 'run_info' dict
        self.run_info['upper_bound'] = solver_info.problem.upper_bound
        self.run_info['lower_bound'] = solver_info.problem.lower_bound
        self.run_info['sense'] = str(solver_info.problem.sense)
        self.run_info['solver_status'] = str(solver_info.solver.status)
        self.run_info['termination_condition'] = str(
            solver_info.solver.termination_condition)
        self.run_info['model_solve_time'] = int(time.time() - time_start)

        self.log.info('Solve time: %d sec' % self.run_info['model_solve_time'])

        # **********************************************************************
        #   Export results to JSON
        # **********************************************************************
        # Write results to a json-file with a specified name.
        # Delete the results_file if exception is thrown.
        if results_file is not None:
            try:
                with open(results_file, 'w') as f:
                    f.write(json.dumps(self.serialize(), indent=2))
            except (ValueError or json.JSONDecodeError):
                self.log.error('Problem encountered while writing json-export.')
                if os.path.isfile(results_file):
                    os.remove(results_file)

    def export_component_configuration(self):
        """
        This function collects the configuration data of all modeled components
        (results of the optimization) as pandas Series and returns them in a
        pandas DataFrame. The configuration features are (if exist):

        * the binary existence variable (utils.BI_EX),
        * the binary existence variables of modules (utils.BI_MODULE_EX)
        * the component capacity variable (utils.CAP)

        :returns: The configuration of all components of the model instance.
        :rtype: pandas DataFrame
        """
        self.log.info('Call of function "export_component_configuration"')

        for name, comp in self.components.items():
            self.component_configuration[name] = \
                comp.export_component_configuration()
        return self.component_configuration

    def import_component_configuration(self, config, fix_existence=True,
                                       fix_modules=True, fix_capacity=True,
                                       store_previous_variables=True):
        """
        Function to load a pandas DataFrame with configuration specifications
        (binary existence variables and capacity variable values) and fix the
        configuration of the modeled components accordingly.

        :param config: The component configuration of the model instance
            (generated by function 'export_component_configuration').
        :type config: pandas DataFrame

        :param fix_existence: Specify whether the imported (global) binary
            component existence variables should be fixed (if available).
            |br| *Default: True*
        :type fix_existence: bool

        :param fix_modules: Specify whether the imported binary existence
            variables of the component modules should be fixed (if available).
            |br| *Default: True*
        :type fix_modules: bool

        :param fix_capacity: Specify whether the imported component capacity
            variable should be fixed or not.
            |br| *Default: True*
        :type fix_capacity: bool

        :param store_previous_variables: State whether the representation of the
            variables before applying the configuration import should be stored
            in DataFrame "variables_copy" of each component. This representation
            can be used by function "reset_component_variables" to undo changes.
            |br| *Default: True*
        :type store_previous_variables: bool
        """
        self.log.info('Call of function "import_component_configuration"')

        if not isinstance(config, pd.DataFrame):
            raise TypeError('The data needs to be imported as a pd.DataFrame!')
        utils.check_and_set_bool(fix_existence, 'fix_existence'),
        utils.check_and_set_bool(fix_modules, 'fix_modules')
        utils.check_and_set_bool(fix_capacity, 'fix_capacity')
        utils.check_and_set_bool(store_previous_variables,
                                 'store_previous_variables')

        for comp_name in config:
            if comp_name not in self.components.keys():
                self.log.warn('The configuration data of component "%s" is '
                              'imported but the component is not found in the '
                              'EnergySystem instance.' % comp_name)
            else:
                data = config[comp_name]
                comp = self.components[comp_name]
                comp.import_component_configuration(
                    data, fix_existence, fix_modules, fix_capacity,
                    store_previous_variables)

    def add_design_integer_cut_constraint(self, which_instances='all',
                                          include_existence=True,
                                          include_modules=True):
        """
        Function to add an integer cut constraint to the Concrete Model of the
        EnergySystem instance. Hereby, the currently calculated design
        solution is excluded from the solution space. Hence, in a next
        optimization run of the same model instance a different design solution
        has to be determined. The integer cuts are destroyed once a new
        optimization model is declared.

        :param which_instances: State which components should be considered
            while formulating the integer cut constraint. The argument can
            either take the string 'all' or a list of component (group) names.
            |br| *Default: 'all'*
        :type which_instances: str 'all' or list of component (group) names

        :param include_existence: State if the binary existence variables of the
            components should be considered (if available) for formulating the
            integer cut constraint.
            |br| *Default: True*
        :type include_existence: bool

        :param include_modules: State if the binary modules existence variables
            of the components should be considered (if available) for
            formulating the integer cut constraint.
            |br| *Default: True*
        :type include_modules: bool
        """
        self.log.info('Call of function "add_design_integer_cut_constraint"')

        # Check user input
        if which_instances != 'all' and not isinstance(which_instances, list):
            raise TypeError('"which_instances" takes string "all" or holds '
                            'a list of component names!')
        utils.check_and_set_bool(include_existence, 'include_existence')
        utils.check_and_set_bool(include_modules, 'include_modules')

        # Run the 'add_integer_cut_constraint' function with the desired comps
        # https://pyomo.readthedocs.io/en/latest/working_models.html
        icc_expr = 0  # init
        for name, comp in self.components.items():
            if which_instances == 'all' or name in which_instances \
                    or comp.group_name in which_instances:

                # If existence modules binaries should be used and are available
                if include_modules and comp.capacity_per_module is not None:
                    # Check if variable is constructed
                    var_bi_mod_ex = comp.variables[utils.BI_MODULE_EX]['pyomo']
                    if var_bi_mod_ex is not None:
                        # Check in every index in variable if it has a value
                        for idx in var_bi_mod_ex:
                            # Append variable to ICC expression
                            if var_bi_mod_ex[idx].value is not None:
                                if var_bi_mod_ex[idx].value < 0.5:
                                    icc_expr += var_bi_mod_ex[idx]
                                else:
                                    icc_expr += (1 - var_bi_mod_ex[idx])

                # If existence binaries should be used and are available in comp
                if include_existence and comp.has_bi_ex:
                    # ----------------------------------------------------------
                    # *** Special case: ***
                    # If a component has a binary existence variable and binary
                    # module variables, you should not use both for ICC. E.g.
                    # case: all modules do not exist (BI_MOD_EX[i]=0) and global
                    # binary can take both values (0 and 1) but nothing changes.
                    # ==> Use first module binary as global existence indicator
                    if include_modules and comp.capacity_per_module is not None:
                        continue  # skip the rest und go to next item in loop
                    elif not include_modules and comp.capacity_per_module is \
                            not None:
                        var_bi_mod_ex = comp.variables[utils.BI_MODULE_EX][
                            'pyomo']
                        if var_bi_mod_ex is not None:
                            if var_bi_mod_ex[1].value is not None:
                                if var_bi_mod_ex[1].value < 0.5:
                                    icc_expr += var_bi_mod_ex[1]
                                else:
                                    icc_expr += (1 - var_bi_mod_ex[1])
                        continue
                    # ----------------------------------------------------------
                    # Check if variable is constructed and has a value
                    var_bi_ex = comp.variables[utils.BI_EX]['pyomo']
                    if var_bi_ex is not None and var_bi_ex.value is not None:
                        # Append variable to ICC expression
                        if var_bi_ex.value < 0.5:
                            icc_expr += var_bi_ex
                        else:
                            icc_expr += (1 - var_bi_ex)

        # Add the expression to the model instance
        if self.is_model_declared:
            if self.is_persistent_model_declared:
                # add constraint to persistent solver
                self.solver.add_constraint(
                    self.model.integer_cut_constraints.add(icc_expr >= 1))
            else:  # conventional model
                self.model.integer_cut_constraints.add(icc_expr >= 1)
        else:
            raise ValueError('Integer cut constraints can only be applied if a '
                             'model has already been constructed and the '
                             'results are available!')

    def relax_integrality(self, which_instances='all', include_existence=True,
                          include_modules=True, include_time_dependent=True,
                          store_previous_variables=True):
        """
        Function to relax the integrality of the binary variables of the
        modelled components. This means binary variables are declared to be
        'NonNegativeReals with an upper bound of 1. The relaxation can be
        performed for the binary existence variable, the module existence binary
        variables and time-dependent binary variables.

        :param which_instances: State for which components the relaxation of the
            binary variables should be done. The keyword argument can either
            take the string 'all' or a list of component (group) names.
            |br| *Default: 'all'*
        :type which_instances: str 'all' or list of component (group) names

        :param include_existence: State whether the integrality of the binary
            existence variables should be relaxed (if available).
            |br| *Default: True*
        :type include_existence: bool

        :param include_modules: State whether the integrality of the binary
            modules existence variables should be relaxed (if available).
            |br| *Default: True*
        :type include_modules: bool

        :param include_time_dependent: State whether the integrality of the
            time-dependent binary variables should be relaxed. (if available).
            |br| *Default: True*
        :type include_time_dependent: bool

        :param store_previous_variables: State whether the representation of the
            variables before applying the relaxation should be stored in the
            DataFrame "variables_copy" of each component. This representation
            can be used by function "reset_component_variables" to undo changes.
            |br| *Default: True*
        :type store_previous_variables: bool
        """
        self.log.info('Call of function "relax_integrality"')

        # Check user input
        if which_instances != 'all' and not isinstance(which_instances, list):
            raise TypeError('"which_instances" takes string "all" or holds '
                            'a list of component names!')
        utils.check_and_set_bool(include_existence, 'include_existence')
        utils.check_and_set_bool(include_modules, 'include_modules')
        utils.check_and_set_bool(include_time_dependent,
                                 'include_time_dependent')
        utils.check_and_set_bool(store_previous_variables,
                                 'store_previous_variables')

        # Run the 'relax_integrality' function in the desired components
        for name, comp in self.components.items():
            if which_instances == 'all' or name in which_instances \
                    or comp.group_name in which_instances:
                comp.relax_integrality(include_existence, include_modules,
                                       include_time_dependent,
                                       store_previous_variables)

    def edit_component_variables(self, name, which_instances='all',
                                 store_previous_variables=True, **kwargs):
        """
        Method for manipulating the specifications of already defined component
        variables (e.g., change variable domain, add variable bounds, etc.).

        :param name: Name / identifier of the edited variable.
        :type name: str

        :param which_instances: State for which components the relaxation of the
            binary variables should be done. The keyword argument can either
            take the string 'all' or a list of component (group) names.
            |br| *Default: 'all'*
        :type which_instances: str 'all' or list of component (group) names

        :param store_previous_variables: State whether the representation of the
            variables before applying the relaxation should be stored in the
            DataFrame "variables_copy" of each component. This representation
            can be used by function "reset_component_variables" to undo changes.
            |br| *Default: True*
        :type store_previous_variables: bool

        :param kwargs: Additional keyword arguments for editing. Options are:
            'ub' and 'lb' to add an upper or lower variable bound, 'domain' to
            set the variable domain, and 'has_time_set' to define if the
            variable should inherit the global time set of the EnergySystem.
        """
        self.log.info('Call of function "edit_component_variables"')

        # Run the function in the desired components
        for comp in self.components.values():
            if which_instances == 'all' or comp.name in which_instances \
                    or comp.group_name in which_instances:
                comp.edit_variable(name, store_previous_variables, **kwargs)

    def reset_component_variables(self, which_instances='all'):
        """
        Function to reset the variables of the modeled components to their
        state that is stored in the component DataFrame "variables_copy".
        This includes the resetting of the DataFrame "variables" and the pyomo
        variables itself (if constructed).

        :param which_instances: State for which components the variable
            resetting should be done. The keyword argument can either take the
            string 'all' or a list of component (group) names.
            |br| *Default: 'all'*
        :type which_instances: str 'all' or list of component (group) names
        """
        self.log.info('Call of function "reset_component_variables"')
        # Check user input
        if which_instances != 'all' and not isinstance(which_instances, list):
            raise TypeError('The "which_instances" keyword can either take the '
                            'string "all" or a list of component names!')

        # Run the 'reset_variables' function in the desired components
        for name, comp in self.components.items():
            if which_instances == 'all' or name in which_instances \
                    or comp.group_name in which_instances:
                comp.reset_variables()

    # ==========================================================================
    #   Serialize the content of the class instance and its components
    # ==========================================================================
    def serialize(self):
        """
        This method collects all relevant input data and optimization results
        from the EnergySystem instance, the added components and the pyomo
        model instance. The data is arranged in an ordered dictionary, that is
        later exported as a file in JSON-format.

        :return: OrderedDict
        """

        def get_arc_variable_values():
            arc_data = {}
            for arc_name, connection in self.component_connections.items():
                # Get the pyomo block for the expanded arc
                arc_block = getattr(self.model, arc_name + '_expanded')
                # Get connected component class instances and the commodity name
                src, dest, commod = connection[0], connection[1], connection[2]
                # We can only get the commodity variable values from the arc
                # block directly, if the variables are split (not for direct
                # connection of single outlet with single inlet).
                if hasattr(arc_block, commod):
                    arc_var = str(getattr(arc_block, commod).get_values())
                else:
                    # Else: Block only holds one equality constraint (IN == OUT)
                    # --> Directly use the values of the outlet port variable of
                    # the source or the inlet port variable of the destination.
                    source_var = src.outlet_commod_and_var_names[commod]
                    arc_var = str(getattr(src.block, source_var).get_values())
                arc_data[arc_name] = arc_var
            return arc_data

        def get_added_obj_values():
            obj_values = {}
            for key, val in self.added_obj_contributions_results.items():
                obj_values[key] = pyomo.value(val)
            return obj_values

        components = {}
        for name, comp in self.components.items():
            components[name] = comp.serialize()

        return OrderedDict([
            ('model_class', self.__class__.__name__),
            ('number_of_time_steps', self.number_of_time_steps),
            ('hours_per_time_step', self.hours_per_time_step),
            ('is_data_clustered', self.is_data_clustered),
            ('number_of_time_steps_per_period',
             self.number_of_time_steps_per_period),
            ('number_of_typical_periods', len(self.typical_periods)),
            ('total_number_of_periods', len(self.periods)),
            ('periods_order', str(list(self.periods_order))),
            ('period_occurrences', str(self.period_occurrences)),
            ('present value factor', self.pvf),
            ('components', components),
            ('arc_variables', get_arc_variable_values()),
            ('added_objective_function_contributions', get_added_obj_values()),
            ('run_info', self.run_info)
        ])


if __name__ == '__main__':
    es = EnergySystem()
