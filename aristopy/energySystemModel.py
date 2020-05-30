#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
** The EnergySystemModel class **

* Last edited: 2020-01-01
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


# The EnergySystemModel is the main model container. An instance of the
# EnergySystemModel class holds the modelled components, the overall pyomo
# model and the results of the optimization.
# It also provides some features to manipulate the associated component models:
#   * Relax the integrality of binary variables
#   * General editing of component variables (e.g. change bounds or domains)
#   * Reset component variables after applying changes (e.g. relaxation)
#   * Cluster implemented time series data
#   * Export and import component configurations
#   * Add variables, constraints and objective function contributions outside
#     of the component declaration
class EnergySystemModel:
    def __init__(self, number_of_time_steps=8760, hours_per_time_step=1,
                 interest_rate=0.05, economic_lifetime=20, logging=None):
        """
        Initialize an instance of the EnergySystemModel class.

        :param number_of_time_steps: Number of considered time steps for
            modeling the dispatch problem. With "hours_per_time_step" the
            share of the modelled year can be calculated. In this way the cost
            of each time step is scaled and included in the objective function.
            |br| * Default: 8760
        :type number_of_time_steps: integer (>0)

        :param hours_per_time_step: Number of hours per modelled time step.
            |br| * Default: 1
        :type hours_per_time_step: integer (>0)

        :param interest_rate: Value to calculate the present value factor of a
            cost rate that occurs in the future.
            |br| * Default: 0.05 (corresponds to 5%)
        :type interest_rate: float or integer (>=0)

        :param economic_lifetime: Years to consider for calculating the net
            present value of a investment with annual incoming and outgoing
            cash flows. |br| * Default: 20
        :type economic_lifetime:  integer (>0)

        :param logging: Specify the behavior of the logging by setting a own
            Logger class instance. User can decide where to log (file/console)
            and what to log (see description of aristopy "Logger"). With
            default value "None" minimal logging is displayed in the console.
        :type logging: None or instance of aristopy class "Logger"
        """
        # Check user input:
        utils.check_energy_system_model_input(
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

        # Initialization: Overwritten if time series aggregation is performed
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
        # required to calculate the present value of a annuity. The global
        # parameters interest rate and economic lifetime of the energy system
        # investment are used to this end.
        self.pvf = sum(1 / (1 + interest_rate)**n
                       for n in range(1, economic_lifetime+1))

        # **********************************************************************
        #   Optimization
        # **********************************************************************
        # The parameter 'pyM' holds the Pyomo Concrete Model instance containing
        # sets, parameters, variables, constraints and the objective function.
        # It is None during initialization and changed when the functions
        # 'optimize' or 'declare_optimization_problem' are called.
        # Before the model instance is optimized, a solver instance is assigned
        # to the "solver" attribute. It also stores basic results of the run.
        # The "is_model_declared" flag indicates if the model instance is
        # already declared.
        # The "is_persistent_model_declared" flag states if the model has been
        # declared and assigned to a persistent solver instance.
        self.pyM = None
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
        # 'components' is a dictionary (component name: component instance)
        # in which all components of the EnergySystemModel instance are stored.
        # The pyomo block model object (stored variables and constraints) of a
        # component instance can be accessed via its "pyB" attribute.
        self.components = {}  # name of comp: comp object itself

        # The 'component_connections' is a dict that stores the connections of
        # the component instances of the energy system model. It is formed from
        # the specified inlets and outlets and the connecting commodity:
        # {arc_name: [source instance, destination instance, commodity_name]}
        self.component_connections = {}  # {arc_name: [src, dest, commodity]}

        # 'component_configuration' is a pandas Dataframe to store basic results
        # of the availability and capacity of the modelled components. It is
        # used to export (and import) the configuration results.
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

        self.log.info('Initializing EnergySystemModel completed!')

    def __repr__(self):
        # Define class format for printing and logging
        return '<EnSysMo: "id=%s..%s">' % (hex(id(self))[:3],
                                           hex(id(self))[-3:])

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
        :type which_instances: 'all' or list of component (group) names

        :param include_existence: State whether the integrality of the binary
            existence variables should be relaxed (if available).
        :type include_existence: boolean

        :param include_modules: State whether the integrality of the binary
            modules existence variables should be relaxed (if available).
        :type include_modules: boolean

        :param include_time_dependent: State whether the integrality of the
            time-dependent binary variables should be relaxed. (if available).
        :type include_time_dependent: boolean

        :param store_previous_variables: State whether the representation of the
            variables before applying the relaxation should be stored in the
            DataFrame "variables_copy" of each component. This representation
            can be used by function "reset_component_variables" to undo changes.
        :type store_previous_variables: boolean
        """
        self.log.info('Call of function "relax_integrality"')

        # Check user input
        if which_instances != 'all' and not isinstance(which_instances, list):
            raise TypeError('"which_instances" takes string "all" or holds '
                            'a list of component names!')
        utils.is_boolean(include_existence), utils.is_boolean(include_modules)
        utils.is_boolean(include_time_dependent)
        utils.is_boolean(store_previous_variables)

        # Run the 'relax_integrality' function in the desired components
        for name, comp in self.components.items():
            if which_instances == 'all' or name in which_instances \
                    or comp.group_name in which_instances:
                comp.relax_integrality(include_existence, include_modules,
                                       include_time_dependent,
                                       store_previous_variables)

    def edit_component_variables(self, variable, which_instances='all',
                                 store_previous_variables=True, **kwargs):
        # TODO: Add description
        self.log.info('Call of function "edit_component_variables"')

        # Run the function in the desired components
        for name, comp in self.components.items():
            if which_instances == 'all' or name in which_instances \
                    or comp.group_name in which_instances:
                comp.edit_variable(variable, store_previous_variables, **kwargs)

    def reset_component_variables(self, which_instances='all'):
        """
        Function to reset the variables of the modelled components to their
        state that is stored in the component DataFrame "variables_copy".
        This includes the resetting of the DataFrame "variables" and the pyomo
        variables itself (if constructed).

        :param which_instances: State for which components the variable
            resetting should be done. The keyword argument can either take the
            string 'all' or a list of component (group) names.
        :type which_instances: 'all' or list of component (group) names
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

    def export_component_configuration(self):
        """
        This function gets the configuration data (results of the optimization)
        of all modelled components as pandas Series and collects and returns
        them in a pandas DataFrame. The configuration features are (if exist):

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

    def import_component_configuration(self, dataframe, fix_existence=True,
                                       fix_modules=True, fix_capacity=True,
                                       store_previous_variables=True):
        """
        Function to load a pandas DataFrame with configuration specifications
        (binary existence variables and capacity variable values) and fix the
        configuration of the modelled components accordingly.

        :param dataframe: The component configuration of the model instance.
        :type dataframe: pandas DataFrame

        :param fix_existence: Specify whether the imported (global) binary
            component existence variables should be fixed (if available).
        :type fix_existence: boolean

        :param fix_modules: Specify whether the imported binary existence
            variables of the component modules should be fixed (if available).
        :type fix_modules: boolean

        :param fix_capacity: Specify whether the imported component capacity
            variable (of the main commodity) should be fixed or not.
        :type fix_capacity: boolean

        :param store_previous_variables: State whether the representation of the
            variables before applying the configuration import should be stored
            in DataFrame "variables_copy" of each component. This representation
            can be used by function "reset_component_variables" to undo changes.
        :type store_previous_variables: boolean
        """
        self.log.info('Call of function "import_component_configuration"')

        utils.is_dataframe(dataframe), utils.is_boolean(fix_existence),
        utils.is_boolean(fix_modules), utils.is_boolean(fix_capacity),
        utils.is_boolean(store_previous_variables)

        for comp_name in dataframe:
            if comp_name not in self.components.keys():
                self.log.warn('The configuration data of component "%s" is '
                              'imported but the component is not found in the '
                              'energy system model instance.' % comp_name)
            else:
                data = dataframe[comp_name]
                comp = self.components[comp_name]
                comp.import_component_configuration(
                    data, fix_existence, fix_modules, fix_capacity,
                    store_previous_variables)

    def add_variable(self, var):
        """
        Function to manually add pyomo variables to the main model container
        (ConcreteModel: pyM) of the energy system model instance via instances
        of aristopy's Var class. The attributes of the variables are stored in
        DataFrame "added_variables" and later initialized during the function
        call 'declare_optimization_problem'.

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

    def add_constraint(self, name=None, has_time_set=True, alternative_set=None,
                       rule=None):
        """
        Function to manually add constraints to the energy system model after
        the instance has been created. The attributes are stored in the
        DataFrame 'added_constraints' and initialized during the function call
        'declare_optimization_problem'.

        :param name: Name (identifier) of the added constraint. The rule name is
            used if no name is specified.
        :type name: string

        :param has_time_set: Is True if the time set of the energy system model
            is also a set of the added constraint.
            |br| * Default: True
        :type has_time_set: boolean

        :param alternative_set: Alternative constraint sets can be added here
            via iterable Python objects (e.g. list)

        :param rule: A Python function that specifies the constraint with a
            equality or inequality expression. The rule must hold at least
            two arguments: First the energy system model it is added to (in
            most cases: self), second the ConcreteModel of the instance (m).
            Additional arguments represent sets (e.g. time).
        :type rule: Python function
        """
        self.log.info('Call of function "add_constraint"')

        # Check the correctness of the user input
        utils.check_add_constraint(name, has_time_set, alternative_set, rule)
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
        model container (ConcreteModel, pyM) and returns a single value.

        :param rule: A Python function that returns a single value that is added
            to the objective function of the model instance. The rule must hold
            exactly two arguments: The energy system model it is added to (in
            most cases: self), second the ConcreteModel of the instance (m).
        :type rule: Python function

        :param name: Name (identifier) of the added objective function
            contribution. The rule name is used if no name is specified.
        :type name: string
        """
        self.log.info('Call of function "add_objective_function_contribution"')

        # Check the input:
        if name is not None:
            utils.is_string(name)
        if not callable(rule):
            raise TypeError('The "rule" needs to hold a callable object!')
        if name is None:
            name = rule.__name__
        # Add the rule and the name to a dictionary of the EnSys model instance.
        self.added_objective_function_contributions.update({name: rule})

    def add_design_integer_cut_constraint(self, which_instances='all',
                                          include_existence=True,
                                          include_modules=True):
        """
        Function to add an integer cut constraint to the Concrete Model of the
        EnergySystemModel instance. Hereby, the currently calculated design
        solution is excluded from the solution space. Hence, in a next
        optimization run of the same model instance a different design solution
        has to be determined. The integer cuts are destroyed once a new
        optimization problem is declared.

        :param which_instances: State which components should be considered
            while formulating the integer cut constraint. The argument can
            either take the string 'all' or a list of component (group) names.
        :type which_instances: 'all' or list of component (group) names

        :param include_existence: State if the binary existence variables of the
            components should be considered (if available) for formulating the
            integer cut constraint.
        :type include_existence: boolean

        :param include_modules: State if the binary modules existence variables
            of the components should be considered (if available) for
            formulating the integer cut constraint.
        :type include_modules: boolean
        """
        self.log.info('Call of function "add_design_integer_cut_constraint"')

        # Check user input
        if which_instances != 'all' and not isinstance(which_instances, list):
            raise TypeError('"which_instances" takes string "all" or holds '
                            'a list of component names!')
        utils.is_boolean(include_existence), utils.is_boolean(include_modules)

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
                    self.pyM.integer_cut_constraints.add(icc_expr >= 1))
            else:  # conventional model
                self.pyM.integer_cut_constraints.add(icc_expr >= 1)
        else:
            raise ValueError('Integer cut constraints can only be applied if a '
                             'model has already been constructed and the '
                             'results are available!')

    def cluster(self, number_of_typical_periods=7,
                number_of_time_steps_per_period=24,
                cluster_method='hierarchical',
                sort_values=True,
                **kwargs):
        """
        Clusters the time series data of all components considered in the
        EnergySystemModel instance and then stores the clustered data in the
        respective components. For this, the time series data is broken down
        into an ordered sequence of periods (e.g. 365 days) and to each period
        a typical period (e.g. 7 typical days with 24 hours) is assigned.
        For the clustering itself, the tsam package is used
        (cf. https://github.com/FZJ-IEK3-VSA/tsam).
        Additional keyword arguments for the TimeSeriesAggregation instance
        can be added (facilitated by kwargs). As an example: it might be useful
        to add extreme periods to the clustered typical periods.

        (Note: See tsam package documentation for more information.)

        :param number_of_typical_periods: states the number of typical periods
            into which the time series data should be clustered.
            |br| * Default: 7
        :type number_of_typical_periods: strictly positive integer

        :param number_of_time_steps_per_period: number of time steps per period
            |br| * Default: 24
        :type number_of_time_steps_per_period: strictly positive integer

        :param cluster_method: states the method which is used in the tsam
            package for clustering the time series data. Options are for example
            'averaging','k_means','exact k_medoid' or 'hierarchical'.
            |br| * Default: 'hierarchical'
        :type cluster_method: string

        :param sort_values: states if the tsam algorithm should use
            (a) the sorted duration curves (-> True) or
            (b) the original profiles (-> False)
            of the time series data within a period for clustering.
            |br| * Default: True
        :type sort_values: boolean
        """
        # Ensure input arguments fit temporal representation of energy system
        utils.check_clustering_input(number_of_typical_periods,
                                     number_of_time_steps_per_period,
                                     self.number_of_time_steps)

        time_start = time.time()
        self.log.info('Clustering time series data with %s typical periods and '
                      '%s time steps per period.'
                      % (number_of_typical_periods,
                         number_of_time_steps_per_period))

        # Format data to fit the input requirements of the "tsam" package:
        # (a) collect the time series data from all components in a DataFrame
        # with unique column names
        # (b) thereby collect the weights for each time series as well in a dict
        time_series_data, time_series_weights = {}, {}
        for comp in self.components.values():
            if comp.number_in_group == 1:  # Add only once per group
                data, weights = comp.get_data_for_time_series_aggregation()
                time_series_data.update(data)
                time_series_weights.update(weights)

        # Convert data dictionary to pandas DataFrame
        time_series_data = pd.DataFrame.from_dict(time_series_data)
        # Note: Sets index for the time series data.
        # The index is of no further relevance in the energy system model.
        time_series_data.index = \
            pd.date_range('2050-01-01 00:30:00',
                          periods=self.number_of_time_steps,
                          freq=(str(self.hours_per_time_step) + 'H'),
                          tz='Europe/Berlin')

        # Cluster data with tsam package
        # (reindex_axis for reproducibility of TimeSeriesAggregation)
        time_series_data = time_series_data.reindex(
            sorted(time_series_data.columns), axis=1)

        cluster_class = TimeSeriesAggregation(
            timeSeries=time_series_data,
            noTypicalPeriods=number_of_typical_periods,
            hoursPerPeriod=
            number_of_time_steps_per_period * self.hours_per_time_step,
            clusterMethod=cluster_method,
            sortValues=sort_values,
            weightDict=time_series_weights,
            **kwargs)

        # Convert the clustered data to a pandas DataFrame and store the
        # respective clustered time series data in the associated components
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
        # Error in original formulation: see E-Mail (26.9.19)
        self.period_occurrences = [
            (self.periods_order == tp).sum() for tp in self.typical_periods]

        # Set cluster flag True (ensure consistently clustered time series data)
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

    def declare_time_sets(self, pyM, time_series_aggregation):
        """
        Initialize time parameters and four different time sets.

        The "time_set" represents the general set. The index holds tuples of
        periods and time steps inside of these periods. In case the optimization
        is performed without time series aggregation, the set runs from [(0,0),
        (0,1), ..to.., (0,number_of_time_steps-1)].
        Otherwise: [(0,0), ..., (0,number_of_time_steps_per_period-1), (1,0),
        ..., (number_of_typical_periods-1, number_of_time_steps_per_period-1)].
        The set "intra_period_time_set" holds tuples of periods and points in
        time before, after or between regular time steps inside of a period.
        Hence, the second value runs from 0 to "number_of_time_steps" (without
        aggregation) or "number_of_time_steps_per_period" respectively.
        The third set "intra_period_time_set" is one-dimensional and ranges from
        0 to the overall number of periods plus 1. So if no aggregation is used
        it has only two entries [0, 1]. Otherwise is is ranging from 0 to
        (1 + number_of_time_steps / number_of_time_steps_per_period).
        The "typical_periods_set" is a set ranging from 0 to the number of
        typical periods. If no aggregation is used it only holds 0. |br|

        |br| E.g.: Case with 2 periods and 3 time steps per period |br|
        1) ______(0,0)___(0,1)___(0,2)___________(1,0)___(1,1)___(1,2)______|br|
        2) __(0,0)___(0,1)___(0,2)___(0,3)___(1,0)___(1,1)___(1,2)___(1,3)__|br|
        3) 0_______________________________1_______________________________2|br|
        4) ________________0_______________________________1________________|br|
        1) time_set, 2) intra_period_time_steps_set,
        3) inter_period_time_steps_set, 4) typical_periods_set |br|

        :param pyM: Pyomo ConcreteModel instance containing sets, variables,
            constraints and objective.
        :type pyM: pyomo ConcreteModel

        :param time_series_aggregation: states if the optimization of the energy
            system model should be done with
            (a) the full scale time series (False) or
            (b) clustered time series data (True).
            |br| * Default: False
        :type time_series_aggregation: boolean
        """
        self.log.info('    Declare time sets')

        # Set the time series data for all modelled components
        for comp in self.components.values():
            comp.set_time_series_data(time_series_aggregation)

        # Reset time-relevant attributes and initialize sets if full scale
        # problem is considered.
        if not time_series_aggregation:
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

        pyM.time_set = pyomo.Set(
            dimen=2, initialize=init_time_set, ordered=True)

        pyM.intra_period_time_set = pyomo.Set(
            dimen=2, initialize=init_intra_period_time_set, ordered=True)

        pyM.inter_period_time_set = pyomo.Set(
            initialize=self.inter_period_time_steps, ordered=True)

        pyM.typical_periods_set = pyomo.Set(
            initialize=self.typical_periods, ordered=True)

    def declare_objective(self, pyM):
        """
        Declare the objective function by obtaining the contributions to the
        objective function from all components. Currently, the only selectable
        objective function is the maximization of the net present value.
        """
        self.log.info('    Declare objective function')

        def summarize_extra_objective_function_contributions():
            for key, val in self.added_objective_function_contributions.items():
                obj_rule = getattr(self, key)  # get the function
                obj_value = obj_rule(self.pyM)  # get returned value
                # Add the values to the results dict (used to export results)
                self.added_obj_contributions_results[key] = obj_value
            return sum(self.added_obj_contributions_results.values())

        def objective(m):
            NPV = sum(comp.get_objective_function_contribution(self, pyM)
                      for comp in self.components.values()) \
                  + summarize_extra_objective_function_contributions()
            return NPV
        pyM.Obj = pyomo.Objective(rule=objective, sense=pyomo.maximize)

    def declare_optimization_problem(self, time_series_aggregation=False,
                                     persistent_model=False,
                                     persistent_solver='gurobi_persistent'):
        """
        Declare the optimization problem of the specified energy system. First a
        pyomo ConcreteModel instance is created (pyM) and filled with

        * basic time sets,
        * sets, variables and constraints of the components,
        * an objective function,
        * variables and constraints and objective function contributions that \
        are added via 'add_xxx' functions.

        :param time_series_aggregation: states if the optimization of the energy
            system model should be done with
            (a) the full time series (False) or
            (b) clustered time series data (True).
            |br| * Default: False
        :type time_series_aggregation: boolean

        :param persistent_model: Indicates if a persistent model instance should
            be formed. In this case after model declaration a persistent solver
            instance is created and the declared model instance is assigned.
            |br| * Default: False
        :type persistent_model: boolean

        :param persistent_solver: Name of the persistent solver to be used.
            Possible options are "gurobi_persistent" and "cplex_persistent".
            Is ignored if keyword "persistent_model" is False.
            |br| * Default: 'gurobi_persistent'
        :type persistent_solver: string
        """
        self.log.info('Declare optimization problem with '
                      'time_series_aggregation=%s and persistent_model=%s'
                      % (time_series_aggregation, persistent_model))

        # Get starting time of the function call
        time_start = time.time()

        # Check correctness of inputs
        utils.check_declare_optimization_problem_input(
            time_series_aggregation, self.is_data_clustered,
            persistent_model, persistent_solver)

        # Initialize flags of model status
        self.is_model_declared = False
        self.is_persistent_model_declared = False

        # **********************************************************************
        #   Component connections
        # **********************************************************************
        # Helper functions:
        # -----------------
        def component_not_found(c_name):
            return('\nThe component "{}" or its instances cannot be found in '
                   'the energy system model.\nThe components considered are: {}'
                   '\n Please check the names of your components and inlet and '
                   'outlet specs!'.format(c_name, list(self.components)))

        def set_connection(source, destination, commodity):
            """
            Add connections (arcs) to the "component_connections" dict.
            Update the "var_connections" dictionaries in the components:
            {'variable name': 'connected arc name'
            "var_connections" is used for plotting purposes.
            """
            # Add a new connection to "component_connections" if not available
            arc_name = 'arc-{}-{}-{}'.format(source.name,
                                             destination.name, commodity)

            if arc_name not in self.component_connections:
                self.component_connections[arc_name] = [source, destination,
                                                        commodity]

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

        # ----------------------------------------------------------------------
        # Establish component connections from "inlet" and "outlet"
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
        # ----------------------------------------------------------------------

        # Check correctness of connections
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
        #   Basic variable check
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
        #   User Expressions
        # **********************************************************************
        # Fill the "user_expressions_dict" with data from comp. initialization
        for comp in self.components.values():
            for expr in comp.user_expressions:
                comp.add_expression(expr)

        # **********************************************************************
        #   Initialize mathematical model (ConcreteModel) instance
        # **********************************************************************
        # Initialize a pyomo ConcreteModel which will be used to store the
        # mathematical formulation of the model. The ConcreteModel instance is
        # stored in the EnergySystemModel instance, which makes it available for
        # post-processing or debugging.
        # pyM is just a reference (alias) for the object self.pyM.
        # A pyomo Suffix with the name dual is declared to make dual values
        # associated to the model's constraints available after optimization.
        self.pyM = pyomo.ConcreteModel()
        pyM = self.pyM
        pyM.dual = pyomo.Suffix(direction=pyomo.Suffix.IMPORT)

        # Set time sets for the model instance
        self.declare_time_sets(pyM, time_series_aggregation)

        # Declare all components of the energy system model:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for comp in self.components.values():
            comp.declare_component_model_block(pyM=pyM)
            comp.declare_component_variables(pyM=pyM)
            comp.declare_component_ports()
            comp.declare_component_user_constraints(pyM=pyM)
            comp.declare_component_constraints(ensys=self, pyM=pyM)

        # Declare arcs from component connections list:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for arc_name, connection in self.component_connections.items():
            # Get the connected component class instances and the commodity name
            src, dest, commod = connection[0], connection[1], connection[2]
            # Get the ports in the component model blocks for each variable
            outlet = getattr(src.pyB, 'outlet_' + commod)
            inlet = getattr(dest.pyB, 'inlet_' + commod)
            # Create an arc to connect two ports
            setattr(self.pyM, arc_name, network.Arc(src=outlet, dest=inlet))

        # Call model transformation factory: Expand the arcs
        pyomo.TransformationFactory("network.expand_arcs").apply_to(pyM)

        # Workaround for Pyomo-Versions before 5.6.9:
        # Get all constraints that end with '_split' and deactivate them
        # --> Workaround for Port.Extensive function with indexed variables
        # See: https://groups.google.com/forum/#!topic/pyomo-forum/LaoKMhyu9pA
        # for c in pyM.component_objects(pyomo.Constraint, active=True):
        #     if c.name.endswith('_split'):
        #         c.deactivate()

        # **********************************************************************
        #   Declare additional pyomo objects, added by:
        #       * add_variable
        #       * add_constraint
        #       * add_objective_function_contribution
        # **********************************************************************
        def declare_extra_variables():
            for var_name in self.added_variables:
                # Get the pandas series from DataFrame and the variable specs
                var_dict = self.added_variables[var_name]
                domain = getattr(pyomo, var_dict['domain'])
                bounds = (var_dict['lb'], var_dict['ub'])
                init = var_dict['init']
                # Differentiation between variables with and without time_set
                if var_dict['has_time_set']:
                    setattr(self.pyM, var_name, pyomo.Var(
                        self.pyM.time_set, domain=domain, bounds=bounds,
                        initialize=init))
                elif var_dict['alternative_set'] is not None:
                    setattr(self.pyM, var_name, pyomo.Var(
                        var_dict['alternative_set'], domain=domain,
                        bounds=bounds, initialize=init))
                else:
                    setattr(self.pyM, var_name, pyomo.Var(
                        domain=domain, bounds=bounds, initialize=init))
                # Store variable in self.added_variables[var_name]['pyomo']
                pyomo_var = getattr(self.pyM, var_name)
                self.added_variables[var_name]['pyomo'] = pyomo_var

        def declare_extra_constraints():
            for con_name in self.added_constraints:
                con_dict = self.added_constraints[con_name]
                rule = con_dict['rule']
                # https://stackoverflow.com/questions/972/adding-a-method-to-an-existing-object-instance
                setattr(self, con_name, rule.__get__(self))
                con = getattr(self, con_name)
                # Differentiation between variables with and without time_set
                if con_dict['has_time_set']:
                    setattr(self.pyM, con_name, pyomo.Constraint(
                        self.pyM.time_set, rule=con))
                elif con_dict['alternative_set'] is not None:
                    setattr(self.pyM, con_name, pyomo.Constraint(
                        con_dict['alternative_set'], rule=con))
                else:
                    setattr(self.pyM, con_name, pyomo.Constraint(rule=con))

        def declare_extra_objective_function_contributions():
            for name, rule in self.added_objective_function_contributions.\
                    items():
                setattr(self, name, rule.__get__(self))

        declare_extra_variables()
        declare_extra_constraints()
        declare_extra_objective_function_contributions()

        # **********************************************************************
        #   Declare empty constraint list in the model for design integer cuts
        # **********************************************************************
        # Attribute "integer_cut_constraints" in the ConcreteModel is
        # initialized as an empty pyomo ConstraintList while declaring the
        # optimization problem. Integer cut constraints can be added with
        # function "add_design_integer_cut_constraint".
        self.pyM.integer_cut_constraints = pyomo.ConstraintList()

        # **********************************************************************
        #   Declare objective function
        # **********************************************************************
        # Declare objective function by obtaining the contributions to the
        # objective function from all modeling classes
        self.declare_objective(pyM)

        # **********************************************************************
        #   Built persistent model instance (if "persistent_model" is True)
        # **********************************************************************
        if persistent_model:
            # Store persistent solver name
            self.run_info['solver_name'] = persistent_solver

            # Call the persistent solver and assign the solver to "self.solver"
            time_persistent_start = time.time()
            self.solver = opt.SolverFactory(persistent_solver)
            # Create a "gurobipy" model object and add the model to the solver
            self.solver.set_instance(self.pyM)
            # Set the flag 'is_persistent_model_declared' to True
            self.is_persistent_model_declared = True
            self.log.info('    Time to set instance to persistent solver: %.2f'
                          % (time.time() - time_persistent_start))

        # **********************************************************************
        # Set flag indicating the declaration status of the model to be True
        self.is_model_declared = True

        # Store the build time of the optimize function call in the instance
        self.run_info['model_build_time'] = int(time.time() - time_start)
        self.log.info('    Time to declare optimization model: %.2f'
                      % (time.time() - time_start))

    # ==========================================================================
    #    O P T I M I Z E
    # ==========================================================================
    def optimize(self, declares_optimization_problem=True,
                 persistent_model=False, persistent_solver='gurobi_persistent',
                 time_series_aggregation=False, solver='gurobi',
                 time_limit=None, optimization_specs='', warmstart=False,
                 results_file='results.json', tee=True):
        """
        Optimize the specified energy system for which a pyomo ConcreteModel
        instance is built or called upon.
        A pyomo instance is optimized with the specified inputs, and the
        optimization results are further processed.

        :param declares_optimization_problem: states if the optimization problem
            should be declared (True) or not (False).
            |br| (a) If True, the declareOptimizationProblem function is called
            and a pyomo ConcreteModel instance is built.
            |br| (b) If False, previously declared pyomo ConcreteModel instance
            is used. |br| * Default: True
        :type declares_optimization_problem: boolean

        :param persistent_model: Indicates if a persistent model instance should
            be formed. In this case after model declaration a persistent solver
            instance is created and the declared model instance is assigned.
            |br| * Default: False
        :type persistent_model: boolean

        :param persistent_solver: Name of the persistent solver to be used.
            Possible options are "gurobi_persistent" and "cplex_persistent".
            Is ignored if keyword "persistent_model" is False.
            |br| * Default: 'gurobi_persistent'
        :type persistent_solver: string

        :param time_series_aggregation: states if the optimization of the
            energy system model should be done with
            |br| (a) the full time series (False) or
            |br| (b) clustered time series data (True).
            |br| * Default: False
        :type time_series_aggregation: boolean

        :param solver: specifies which solver should solve the optimization
            problem (has to be installed on the machine).
            |br| * Default: 'gurobi'
        :type solver: string

        :param time_limit: if not specified as None, indicates the maximum solve
            time of the optimization problem in seconds (solver dependent
            input). If triggered before an optimal solution is available, the
            best solution obtained up until then (if available) is processed.
            |br| * Default: None
        :type time_limit: strictly positive integer or None

        :param optimization_specs: specifies parameters for the optimization
            solver (see respective solver documentation for more information).
            Example: 'LogToConsole=1 OptimalityTol=1e-6'
            |br| * Default: empty string ('')
        :type optimization_specs: string

        :param warmstart: specifies if a warm start of the optimization should
            be considered (not always supported by the solvers).
            |br| * Default: False
        :type warmstart: boolean

        :param results_file: Name of the results file (format: .json)
        :type results_file: string or None

        :param tee: Show solver output
            |br| * Default: True
        :type tee: Boolean
        """
        self.log.info('Call of function "optimize"')

        if declares_optimization_problem:
            self.declare_optimization_problem(
                persistent_model=persistent_model,
                persistent_solver=persistent_solver,
                time_series_aggregation=time_series_aggregation)
        else:
            if not self.is_model_declared:
                raise TypeError('The optimization problem is not declared yet. '
                                'Set the argument declares_optimization_problem'
                                ' to True or call the function '
                                '"declare_optimization_problem" first.')

        # Get starting time of the optimization to, later on, obtain the total
        # run time of the optimize function call
        time_start = time.time()

        # Check correctness of inputs
        utils.check_optimize_input(time_series_aggregation,
                                   persistent_model, persistent_solver,
                                   self.is_data_clustered, solver,
                                   time_limit, optimization_specs, warmstart)

        # Store keyword arguments in the EnergySystemModel instance
        self.run_info['time_limit'] = time_limit
        self.run_info['optimization_specs'] = optimization_specs

        # **********************************************************************
        #   Solve the specified optimization problem
        # **********************************************************************
        # Call a solver from the SolverFactory if the model is not persistent
        if not self.is_persistent_model_declared:
            self.run_info['solver_name'] = solver
            self.solver = opt.SolverFactory(solver)
        # Set the time limit if specified (only for solver gurobi)
        if self.run_info['time_limit'] is not None and (
                (persistent_model and persistent_solver == 'gurobi_persistent')
                or (not persistent_model and solver == 'gurobi')):
            self.solver.options['time_limit'] = time_limit

        # Solve optimization problem. The optimization solve time is stored and
        # the solver information is printed.
        if not self.is_persistent_model_declared and solver == 'gurobi':
            self.solver.set_options(optimization_specs)
            self.log.info('Solve non-persistent model using %s' % solver)
            solver_info = self.solver.solve(self.pyM, warmstart=warmstart,
                                            tee=tee)
        elif self.is_persistent_model_declared:
            self.log.info('Solve persistent model using %s' % solver)
            self.solver.set_options(optimization_specs)
            solver_info = self.solver.solve(tee=tee)
        else:
            self.log.info('Solve non-persistent model using %s' % solver)
            solver_info = self.solver.solve(self.pyM, tee=tee)

        # Print the solver summary to the screen
        # solver_info.write()

        # Add solve results to the 'run_info' dict
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
        if results_file:
            try:
                with open(results_file, 'w') as f:
                    f.write(json.dumps(self.serialize(), indent=2))
            except (ValueError or json.JSONDecodeError):
                self.log.error('Problem encountered while writing json-export.')
                if os.path.isfile(results_file):
                    os.remove(results_file)

    # ==========================================================================
    #   Serialize the content of the class instance and its components
    # ==========================================================================
    def serialize(self):

        def get_arc_variable_values():
            arc_data = {}
            for arc_name, connection in self.component_connections.items():
                # Get the pyomo block for the expanded arc
                arc_block = getattr(self.pyM, arc_name + '_expanded')
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
                    arc_var = str(getattr(src.pyB, source_var).get_values())
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
    es = EnergySystemModel()
