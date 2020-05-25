#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
** The Component class **

* Last edited: 2020-01-01
* Created by: Stefan Bruche (TU Berlin)
"""
import copy
from collections import OrderedDict
from abc import ABCMeta, abstractmethod
import pandas as pd
from numpy import nan
import operator as oper
import pyomo.environ as pyomo
import pyomo.network as network
from aristopy import utils


'''
In Python even a class is an object. The metaclass is the class of a class.
A metaclass defines how a class behaves. A class is an instance of metaclass.
The default metaclass used to construct a class in Python is 'type'.
The metaclass 'ABCMeta' enables the use of the decorator 'abstractmethod'.
A class that has a metaclass derived from 'ABCMeta' cannot be instantiated
unless all of its abstract methods and properties are overridden.
Hence, the 'Component' class cannot be instantiated itself and the classes
inheriting from 'Component' must override all abstract methods of the parent.
'''
class Component(metaclass=ABCMeta):
    # The Component class includes the general methods and arguments for the
    # components which can be added to the energy system model (source, sink,
    # storage, conversion, ...). All of these components inherit from the
    # Component class.
    def __init__(self, ensys, name, basic_variable,
                 inlet=None, outlet=None,
                 has_existence_binary_var=False, has_operation_binary_var=False,
                 time_series_data=None, time_series_weights=None,
                 scalar_params=None, additional_vars=None,
                 user_expressions=None,
                 capacity=None, capacity_min=None, capacity_max=None,
                 capacity_per_module=None, maximal_module_number=None,
                 capex_per_capacity=0, capex_if_exist=0,
                 opex_per_capacity=0, opex_if_exist=0,
                 instances_in_group=1,
                 group_has_existence_order=True, group_has_operation_order=True
                 ):
        """
        Initialize an instance of the Component class. Note that an instance of
        the class Component itself can not be instantiated since it holds
        abstract methods. Only the inheriting components (source, conversion,
        ...) can be instantiated.

        :param ensys:
        :param name:
        :param basic_variable:
        :param inlet:
        :param outlet:
        :param has_existence_binary_var: (boolean)
        :param has_operation_binary_var: (boolean)
        :param time_series_data: (dict)
        :param time_series_weights: (dict)
        :param scalar_params: (dict)
        :param additional_vars:
        :param user_expressions:
        :param capacity:
        :param capacity_min:
        :param capacity_max:
        :param capacity_per_module:
        :param maximal_module_number:
        :param capex_per_capacity:
        :param capex_if_exist:
        :param opex_per_capacity:
        :param opex_if_exist:
        :param instances_in_group:
        :param group_has_existence_order:
        :param group_has_operation_order:
        """
        # Documentation: see file "components_attributes_description.txt".

        utils.is_energy_system_model_instance(ensys)
        self.ensys = ensys

        # Set general component data
        utils.is_string(name)
        self.name = name  # might be changed by "add_to_energy_system_model"
        self.group_name = name
        self.number_in_group = 1  # dito
        self.pyB = None  # pyomo simple Block object

        # V A R I A B L E S:
        # ------------------
        # Initialize an empty pandas DataFrame to store the component variables
        self.variables = pd.DataFrame(index=['domain', 'has_time_set',
                                             'alternative_set', 'init',
                                             'ub', 'lb', 'pyomo'])
        # Holds a copy of the variables DataFrame for later reset. It is stored
        # while variables are edited, imported or relaxed (if requested).
        self.variables_copy = pd.DataFrame(index=['domain', 'has_time_set',
                                                  'alternative_set', 'init',
                                                  'ub', 'lb'])

        # Check and set input values for capacities:
        self.capacity, self.capacity_min, self.capacity_max, \
            self.capacity_per_module, self.maximal_module_number = \
            utils.check_and_set_capacities(capacity, capacity_min, capacity_max,
                                           capacity_per_module,
                                           maximal_module_number)

        # Specify a capacity variable (with standard name "CAP") if capacities
        # are stated and add it to variables dict.
        # Set 'capacity_max' as upper bound for the capacity variable "CAP".
        # Lower bound is only used if component is built (depends on status of
        # the existence binary variable if specified) --> see constraints!
        self.has_capacity_var = True \
            if self.capacity_min or self.capacity_max is not None else False
        if self.has_capacity_var:
            self._add_var('CAP', has_time_set=False, ub=self.capacity_max)

        if self.capacity_per_module is not None:
            # add binary variables for the existence of modules of a component
            self._add_var(name='BI_MODULE_EX', has_time_set=False,
                          alternative_set=range(1, self.maximal_module_number
                                                + 1),  # 1 to end inclusive
                          domain='Binary')

        # Add binary variables for existence and operation with standard names
        # ("BI_EX", and "BI_OP", respectively) if required:
        self.has_bi_ex = has_existence_binary_var
        if self.has_bi_ex:
            self._add_var(name='BI_EX', domain='Binary', has_time_set=False)

        self.has_bi_op = has_operation_binary_var
        if self.has_bi_op:
            self._add_var(name='BI_OP', domain='Binary', has_time_set=True)

        if self.capacity_max is None and (self.has_bi_op or self.has_bi_ex):
            raise ValueError('An over-estimator is required if a binary '
                             'operation or existence variable is specified. '
                             'Hence, please enter a value for "capacity_max"!')

        # Multiple instances formed and collected in one group
        self.instances_in_group = utils.set_if_positive(instances_in_group)
        utils.is_boolean(group_has_existence_order)
        self.group_has_existence_order = group_has_existence_order
        if self.instances_in_group > 1 and not self.has_bi_ex and \
                self.group_has_existence_order:  # is True
            raise ValueError('Group requires a binary existence variable if an '
                             'existence order is requested!')
        utils.is_boolean(group_has_operation_order)
        self.group_has_operation_order = group_has_operation_order
        if self.instances_in_group > 1 and not self.has_bi_op and \
                self.group_has_operation_order:  # is True
            raise ValueError('Group requires a binary operation variable if an '
                             'operation order is requested!')

        # Additional variables can be added by the 'additional_vars' keyword
        if additional_vars is not None:
            additional_vars = utils.check_and_convert_to_list(additional_vars)
            for var in additional_vars:
                if var not in self.variables.columns:
                    self._add_var(var)

        # P A R A M E T E R S:
        # --------------------
        # Initialize an empty pandas DataFrame to store the component parameters
        self.parameters = pd.DataFrame(index=['tsam_weight', 'has_time_set',
                                              'values', 'full_resolution',
                                              'aggregated'])

        # Add scalar parameters from dictionary 'scalar_params'
        if scalar_params is not None:
            utils.check_user_dict('scalar_params', scalar_params)
            for key, val in scalar_params.items():
                self._add_param(key, init=val)

        # Add time series data from 'time_series_data'
        if time_series_data is not None:
            for key, val in time_series_data.items():
                data = utils.check_and_convert_time_series(ensys, val)
                self._add_param(key, init=data)
        # Add weights for time series data if required. Default weight = 1.
        if time_series_weights is not None:
            utils.check_user_dict('ts_weight_dict', time_series_weights)
            for param, weight in time_series_weights.items():
                if param in self.parameters.columns:
                    self.parameters[param].loc['tsam_weight'] = weight
                else:
                    raise ValueError('Parameter "{}" is unknown for component '
                                     '"{}"!'.format(param, self.name))

        # Check and set the input values for the cost parameters and prevent
        # invalid parameter combinations.
        self.capex_per_capacity = utils.set_if_positive(capex_per_capacity)
        self.capex_if_exist = utils.set_if_positive(capex_if_exist)
        self.opex_per_capacity = utils.set_if_positive(opex_per_capacity)
        self.opex_if_exist = utils.set_if_positive(opex_if_exist)
        if not self.has_capacity_var and (
                self.capex_per_capacity > 0 or self.opex_per_capacity > 0):
            raise ValueError('Make sure there is a capacity restriction (e.g. '
                             '"capacity", "capacity_max") if you use capacity '
                             'related CAPEX or OPEX.')
        if not self.has_bi_ex and (
                self.capex_if_exist > 0 or self.opex_if_exist > 0):
            raise ValueError('Make sure there is an existence binary variable '
                             'if you use existence related CAPEX or OPEX.')

        # Dictionary to store the contributions of the component to the global
        # objective function value
        self.comp_obj_dict = {'capex_capacity': 0, 'capex_exist': 0,
                              'opex_capacity': 0, 'opex_exist': 0,
                              'opex_operation': 0,
                              'com_cost_time_indep': 0,
                              'com_cost_time_dep': 0,
                              'com_rev_time_indep': 0,
                              'com_rev_time_dep': 0,
                              'start_up_cost': 0}

        # U S E R   E X P R E S S I O N S:
        # --------------------------------
        self.user_expressions = None
        # Add 'user_expressions' are converted to a list of strings
        if user_expressions is not None:
            self.user_expressions = utils.check_and_convert_to_list(
                user_expressions)
        # 'user_expression_dict' holds converted expr -> filled in declaration
        self.user_expressions_dict = {}

        # C O M P O N E N T   C O N N E C T I O N S:
        # ------------------------------------------
        # Check and add inlet and outlets (list of aristopy Flows)
        self.inlet = utils.check_and_set_flows(inlet)
        self.outlet = utils.check_and_set_flows(outlet)

        # List of component commodities (present at inlet and outlet)
        self.commodities = []  # init
        # Dicts for commodities and their respective variable names
        self.inlet_commod_and_var_names = {}  # @inlet: {commod: var_name}
        self.outlet_commod_and_var_names = {}  # @outlet:{commod: var_name}

        for flow in self.inlet:
            # Add commodity to the commodities list if not available:
            if flow.commodity not in self.commodities:
                self.commodities.append(flow.commodity)
            # Add variable to variables DataFrame if not available:
            if flow.var_name not in self.variables:
                self._add_var(flow.var_name)
            # Only add commodity it if it is not already declared at inlet
            if self.inlet_commod_and_var_names.get(flow.commodity) is None:
                self.inlet_commod_and_var_names[flow.commodity] = flow.var_name

        for flow in self.outlet:
            # Check variable name at outlet has not already been used at inlet
            if flow.var_name in self.inlet_commod_and_var_names.values():
                raise ValueError(
                    'Commodity "%s" is found at inlet and outlet of '
                    'component "%s" with the same variable name "%s". '
                    'Please use different names for both sides!'
                    % (flow.commodity, name, flow.var_name))
            # Add commodity to the commodities list if not available:
            if flow.commodity not in self.commodities:
                self.commodities.append(flow.commodity)
            # Add variable to variables DataFrame if not available:
            if flow.var_name not in self.variables:
                self._add_var(flow.var_name)
            # Only add commodity it if it is not already declared at outlet
            if self.outlet_commod_and_var_names.get(flow.commodity) is None:
                self.outlet_commod_and_var_names[flow.commodity] = flow.var_name

        # Every component has a basic variable. It is used to restrict
        # capacities, set operation rates and calculate capex and opex. Usually,
        # the basic variable is referring to a commodity on the inlet or the
        # outlet of the component.
        if basic_variable == 'inlet_variable':
            self.basic_variable = self.inlet[0].var_name
        elif basic_variable == 'outlet_variable':
            self.basic_variable = self.outlet[0].var_name
        elif basic_variable in self.variables:
            self.basic_variable = basic_variable
        else:
            raise ValueError('Name of the basic variable for component "%s" is '
                             'unknown. Please use defaults "inlet_variable" or '
                             '"outlet_variable" to use the variable name of a '
                             'connected Flow, or manually add the variable with'
                             ' the "additional_vars" keyword.' % name)

        # Dict used for plotting. Updated during EnergySystemModel declaration.
        self.var_connections = {}  # {var_name: [connected_arc_names]}

        self.log = None  # Init: Local logger of the component instance

    def __repr__(self):
        return '<Component: "%s">' % self.name

    def pprint(self):
        """ Easy access to pretty print functionality of pyomo model Block """
        if isinstance(self.pyB, pyomo.Block):
            self.pyB.pprint()

    # ==========================================================================
    #    A D D   C O M P O N E N T   T O   T H E   E N S Y S - M O D E L
    # ==========================================================================
    def add_to_energy_system_model(self, ensys, group, instances_in_group=1):
        """
        Add the component to an EnergySystemModel instance.

        :param group: TODO: Add description!
        :param instances_in_group: TODO: Add description!
        :param ensys: EnergySystemModel instance representing the energy system
            in which the component should be modeled.
        :type ensys: EnergySystemModel instance
        """
        ensys.is_data_clustered = False  # reset flag

        for nbr in range(1, instances_in_group + 1):
            if instances_in_group > 1:
                # Create a DEEP-copy of self! Shallow copy creates problems
                # since the Component object holds complicated elements
                instance = copy.deepcopy(self)
                # Anyways, we need a reference on the original EnSys model and
                # not a newly created copy of it!
                instance.ensys = self.ensys
                instance.name = group + '_{}'.format(nbr)  # overwrite name
                instance.number_in_group = nbr  # overwrite number_in_group
            else:
                instance = self  # reference self in 'instance'
            # If component name already exists raise an error.
            if instance.name in ensys.components:
                raise ValueError('Component name {} is not unique.'
                                 .format(instance.name))
            else:
                # Add component with its name and all attributes and methods to
                # the dictionary 'components' of the energy system model
                ensys.components.update({instance.name: instance})

                # Define an own logger for every added component instance
                instance.log = instance.ensys.logger\
                    .get_logger(instance)
                instance.log.info('Add component "%s" to the energy system '
                                  'model' % instance.name)

    # ==========================================================================
    #    A D D ,  E D I T,  R E S E T,  F I X ,  R E L A X   V A R I A B L E S
    # ==========================================================================
    def _add_var(self, name, domain='NonNegativeReals', has_time_set=True,
                 alternative_set=None, ub=None, lb=None, init=None):
        # Specify bounds in the DataFrame according to the variable domain
        if domain == 'NonNegativeReals':
            lb = 0
        elif domain == 'Binary':
            lb = 0
            ub = 1
        # Append data to DataFrame
        series = pd.Series({'has_time_set': has_time_set,
                            'alternative_set': alternative_set,
                            'domain': domain, 'init': init, 'ub': ub, 'lb': lb,
                            'pyomo': None})
        self.variables[name] = series

    def relax_integrality(self, include_existence=True, include_modules=True,
                          include_time_dependent=True,
                          store_previous_variables=True):
        """
        Function to relax the integrality of the binary variables. This means
        binary variables are declared to be 'NonNegativeReals with an upper
        bound of 1. This function encompasses the resetting of the DataFrame
        "variables" and the pyomo variables itself (if already constructed).
        The relaxation can be performed for the binary existence variable, the
        module existence binary variables and time-dependent binary variables.

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
            variables before applying the configuration import should be stored
            in DataFrame "variables_copy" of each component. This representation
            can be used by function "reset_variables" to undo changes.
        :type store_previous_variables: boolean
        """
        # Check the input of the function:
        utils.is_boolean(include_existence)
        utils.is_boolean(include_modules)
        utils.is_boolean(include_time_dependent)
        utils.is_boolean(store_previous_variables)

        def relax_variable(var):
            # Manipulate "variables" (also possible if model not declared yet)
            var['domain'] = 'NonNegativeReals'
            var['ub'] = 1
            if self.ensys.is_model_declared:  # model / variables are declared
                domain = getattr(pyomo, 'NonNegativeReals')
                var_pyomo = var['pyomo']
                if var_pyomo is not None and not var_pyomo.is_indexed():
                    var_pyomo.domain = domain
                    var_pyomo.setub(1)
                    # additional step required for persistent model
                    if self.ensys.is_persistent_model_declared:
                        self.ensys.solver.update_var(var_pyomo)
                elif var_pyomo is not None and var_pyomo.is_indexed():
                    for idx in var_pyomo:
                        var_pyomo[idx].domain = domain
                        var_pyomo[idx].setub(1)
                        # additional step required for persistent model
                        if self.ensys.is_persistent_model_declared:
                            self.ensys.solver.update_var(var_pyomo[idx])

        def store_copy_of_variable(name, var):
            series = pd.Series({'has_time_set': var['has_time_set'],
                                'alternative_set': var['alternative_set'],
                                'domain': var['domain'], 'init': var['init'],
                                'ub': var['ub'], 'lb': var['lb']})
            # Only store it if it is not already in the DataFrame -> otherwise
            # a more original version might be overwritten unintentionally
            if name not in self.variables_copy:
                self.variables_copy[name] = series

        # Loop through the DataFrame "variables"
        for v in self.variables.columns:
            # Include the global existence variable:
            if include_existence and v == 'BI_EX':
                if store_previous_variables:
                    store_copy_of_variable(v, self.variables[v])
                relax_variable(self.variables[v])
                # self.log.debug('Relax integrality of variable: "%s"' % v)
            # Include the module existence variables:
            elif include_modules and v == 'BI_MODULE_EX':
                if store_previous_variables:
                    store_copy_of_variable(v, self.variables[v])
                relax_variable(self.variables[v])
                # self.log.debug('Relax integrality of variable: "%s"' % v)
            # Include time-dependent variables
            elif include_time_dependent and self.variables[v]['domain'] \
                    == 'Binary' and self.variables[v]['has_time_set']:
                if store_previous_variables:
                    store_copy_of_variable(v, self.variables[v])
                relax_variable(self.variables[v])
                # self.log.debug('Relax integrality of variable: "%s"' % v)

    # ******   R E S E T   V A R I A B L E S   *********************************
    def reset_variables(self):
        """
        Function to reset the variables to their state stored in the DataFrame
        "variables_copy". This includes the resetting of the DataFrame
        "variables" and the pyomo variables itself (if already constructed).
        """
        # Loop through variables in DataFrame "variables_copy"
        for var_copy_name in self.variables_copy:

            # The variable should be in the DataFrame "variables"
            if var_copy_name in self.variables:

                # Reset the "variables" DataFrame to values in "variables_copy"
                for attr in ['domain', 'has_time_set', 'alternative_set',
                             'init', 'ub', 'lb']:
                    self.variables[var_copy_name][attr] = \
                        self.variables_copy[var_copy_name][attr]

                # Reset the pyomo variable if it has already been constructed
                if self.ensys.is_model_declared:
                    var_pyomo = self.variables[var_copy_name]['pyomo']
                    var_copy_dict = self.variables_copy[var_copy_name]
                    domain = getattr(pyomo, var_copy_dict['domain'])

                    # Reset domain & unfix var (works globally on (non)-indexed)
                    var_pyomo.domain = domain
                    var_pyomo.unfix()

                    # Set lower and upper bounds and update persistent solver
                    if not var_pyomo.is_indexed():
                        var_pyomo.setlb(var_copy_dict['lb'])
                        var_pyomo.setub(var_copy_dict['ub'])
                        if self.ensys.is_persistent_model_declared:
                            self.ensys.solver.update_var(var_pyomo)
                    else:
                        # variable is indexed --> might take a while, especially
                        # if a persistent model needs to be updated idx by idx
                        for idx in var_pyomo:
                            var_pyomo[idx].setlb(var_copy_dict['lb'])
                            var_pyomo[idx].setub(var_copy_dict['ub'])
                            if self.ensys.is_persistent_model_declared:
                                self.ensys.solver.update_var(var_pyomo[idx])

            # Delete the variable from the DataFrame variables_copy after reset
            self.variables_copy.drop(var_copy_name, axis=1, inplace=True)

    # ******   E D I T   V A R I A B L E   *************************************
    def edit_variable(self, variable, store_previous_variables=True, **kwargs):
        """
        Public function on component level for manipulating the specifications
        of already defined component variables.

        :param variable: Name of the variable (str)

        :param store_previous_variables: State whether the representation of the
            variables before applying the edit function should be stored
            in DataFrame "variables_copy" of each component. This representation
            can be used by function "reset_variables" to undo changes.
        :type store_previous_variables: boolean
        """
        # Check the function input (done locally because might be called
        # directly from component).
        utils.check_edit_var_input(variable, store_previous_variables, **kwargs)

        def store_copy_of_variable(name, var):
            series = pd.Series({'has_time_set': var['has_time_set'],
                                'alternative_set': var['alternative_set'],
                                'domain': var['domain'], 'init': var['init'],
                                'ub': var['ub'], 'lb': var['lb']})
            # Only store it if it is not already in the DataFrame -> otherwise
            # a more original version might be overwritten unintentionally
            if name not in self.variables_copy:
                self.variables_copy[name] = series

        # Check if the variable with name "variable" is available for editing:
        if variable not in self.variables.columns:
            self.log.warn('No variable "%s" available for editing' % variable)
        else:
            self.log.info('Edit variable: "%s"' % variable)
            # Store the previous variable setting if requested
            if store_previous_variables:
                store_copy_of_variable(name=variable,
                                       var=self.variables[variable])
            # Write keyword arguments in "variables" DataFrame
            for key, val in kwargs.items():
                self.variables[variable][key] = val

            # Edit the pyomo variable if it has already been constructed
            if self.ensys.is_model_declared:
                var_pyomo = self.variables[variable]['pyomo']
                edit_domain, edit_lb, edit_ub = False, False, False  # init
                if 'domain' in kwargs.keys():
                    domain = getattr(pyomo, kwargs['domain'])
                    edit_domain = True
                if 'lb' in kwargs.keys():
                    lb = kwargs['lb']
                    edit_lb = True
                if 'ub' in kwargs.keys():
                    ub = kwargs['ub']
                    edit_ub = True
                if 'has_time_set' in kwargs.keys():
                    self.log.warn('Note: The model needs to be declared again '
                                  'if the set of variable should be edited.')
                if not var_pyomo.is_indexed():
                    if edit_domain:
                        var_pyomo.domain = domain
                    if edit_lb:
                        var_pyomo.setlb(lb)
                    if edit_ub:
                        var_pyomo.setub(ub)
                    if self.ensys.is_persistent_model_declared and (
                            edit_domain or edit_lb or edit_ub):
                        self.ensys.solver.update_var(var_pyomo)
                else:  # indexed variable
                    for idx in var_pyomo:
                        if edit_domain:
                            var_pyomo[idx].domain = domain
                        if edit_lb:
                            var_pyomo[idx].setlb(lb)
                        if edit_ub:
                            var_pyomo[idx].setub(ub)
                        if self.ensys.is_persistent_model_declared and (
                                edit_domain or edit_lb or edit_ub):
                            self.ensys.solver.update_var(var_pyomo[idx])

    def export_component_configuration(self):
        """
        This function exports the component configuration data (results of the
        optimization) as a pandas Series. The features are (if exist):
        |br| * the binary existence variable (BI_EX),
        |br| * the binary existence variables of modules (BI_MODULE_EX)
        |br| * the component capacity variable (CAP of the main commodity)

        :return: The configuration of the modelled component instance.
        :rtype: pandas Series
        """
        bi_ex_val, bi_mod_ex_val, cap_val = None, None, None

        if self.has_bi_ex:
            bi_ex_val = self.variables['BI_EX']['pyomo'].value

        if self.capacity_per_module is not None:
            var = self.variables['BI_MODULE_EX']['pyomo']
            bi_mod_ex_val = {i: var[i].value for i in range(
                1, self.maximal_module_number + 1)}  # equiv. to pyomo.RangeSet

        if self.has_capacity_var:
            cap_val = self.variables['CAP']['pyomo'].value

        series = pd.Series({'BI_EX': bi_ex_val, 'BI_MODULE_EX': bi_mod_ex_val,
                            'CAP': cap_val})
        # Replace numpy NaNs that might occur here and there with None
        series = series.replace({nan: None})

        return series

    def import_component_configuration(self, data, fix_existence=True,
                                       fix_modules=True, fix_capacity=True,
                                       store_previous_variables=True):
        """
        # TODO: Do i need to round imported values? (especially binary)
        Function to load a pandas Series (index=['BI_EX', 'BI_MODULE_EX',
        'CAP']) with configuration specifications (binary existence variables
        and capacity variable values). The values are used to fix a specific
        component configuration (for example from other model runs).

        :param data: The configuration of the modelled component instance.
        :type data: pandas Series

        :param fix_existence: Specify whether the imported (global) binary
            component existence variable should be fixed (if available).
        :type fix_existence: boolean

        :param fix_modules: Specify whether the imported binary existence
            variable of the component modules should be fixed (if available).
        :type fix_modules: boolean

        :param fix_capacity: Specify whether the imported component capacity
            variable (of the main commodity) should be fixed or not.
        :type fix_capacity: boolean

        :param store_previous_variables: State whether the representation of the
            variables before applying the configuration import should be stored
            in DataFrame "variables_copy" of each component. This representation
            can be used by function "reset_variables" to undo changes.
        :type store_previous_variables: boolean
        """
        utils.is_series(data), utils.is_boolean(fix_existence),
        utils.is_boolean(fix_modules), utils.is_boolean(fix_capacity),
        utils.is_boolean(store_previous_variables)

        def store_copy_of_variable(name, var):
            series = pd.Series({'has_time_set': var['has_time_set'],
                                'alternative_set': var['alternative_set'],
                                'domain': var['domain'], 'init': var['init'],
                                'ub': var['ub'], 'lb': var['lb']})
            # Only store it if it is not already in the DataFrame -> otherwise
            # a more original version might be overwritten unintentionally
            if name not in self.variables_copy:
                self.variables_copy[name] = series

        # EXISTENCE VARIABLE: Fix if required and available in model and data
        if fix_existence and self.has_bi_ex and data['BI_EX'] is not None:
            # Backup of variables in "variables_copy" if requested for resetting
            if store_previous_variables:
                store_copy_of_variable(name='BI_EX',
                                       var=self.variables['BI_EX'])
            # Always possible to set bounds (also if not declared)
            self.variables['BI_EX']['lb'] = data['BI_EX']
            self.variables['BI_EX']['ub'] = data['BI_EX']
            if self.ensys.is_model_declared:  # model / variables are declared
                self.variables['BI_EX']['pyomo'].fix(data['BI_EX'])
                # additional step required for persistent model
                if self.ensys.is_persistent_model_declared:
                    self.ensys.solver.update_var(
                        self.variables['BI_EX']['pyomo'])

        # MODULE EXISTENCE VARIABLE: Fix if required and available
        if fix_modules and self.capacity_per_module is not None \
                and data['BI_MODULE_EX'] is not None:
            # Backup of variables in "variables_copy" if requested for resetting
            if store_previous_variables:
                store_copy_of_variable(name='BI_MODULE_EX',
                                       var=self.variables['BI_MODULE_EX'])
            # Always possible to set bounds (also if not declared)
            self.variables['BI_MODULE_EX']['lb'] = data['BI_MODULE_EX']
            self.variables['BI_MODULE_EX']['ub'] = data['BI_MODULE_EX']
            if self.ensys.is_model_declared:
                for i in range(1, self.maximal_module_number + 1):
                    self.variables['BI_MODULE_EX']['pyomo'][i].fix(
                        data['BI_MODULE_EX'][i])
                    # additional step required for persistent model
                    if self.ensys.is_persistent_model_declared:
                        self.ensys.solver.update_var(
                            self.variables['BI_MODULE_EX']['pyomo'][i])

        # CAPACITY VARIABLE: Fix if required and available in model and data
        if fix_capacity and self.has_capacity_var and data['CAP'] is not None:
            # Backup of variables in "variables_copy" if requested for resetting
            if store_previous_variables:
                store_copy_of_variable(name='CAP', var=self.variables['CAP'])
            # Always possible to set bounds (also if not declared)
            self.variables['CAP']['lb'] = data['CAP']
            self.variables['CAP']['ub'] = data['CAP']
            if self.ensys.is_model_declared:  # model / variables are declared
                self.variables['CAP']['pyomo'].fix(data['CAP'])
                # additional step required for persistent model
                if self.ensys.is_persistent_model_declared:
                    self.ensys.solver.update_var(self.variables['CAP']['pyomo'])

    # ==========================================================================
    #    A D D   P A R A M E T E R
    # ==========================================================================
    def _add_param(self, name, tsam_weight=1, init=None):
        # If input is integer or float, flag 'has_time_set' is set to False.
        # Else, (input: numpy array, list, dict or pandas series) flag -> True
        has_time_set = False \
            if isinstance(init, int) or isinstance(init, float) else True
        # Append data to DataFrame
        series = pd.Series({'tsam_weight': tsam_weight,
                            'has_time_set': has_time_set,
                            'values': init, 'full_resolution': init,
                            'aggregated': None})
        self.parameters[name] = series

    # ==========================================================================
    #    A D D   E X P R E S S I O N
    # ==========================================================================
    def add_expression(self, expression):
        # Create a name for the expression by removing all spaces
        expr_name = expression.replace(' ', '')
        # Disassemble expr into its parts -> e.g. ['Q', '>=', '100']
        expr_pieces = utils.disassemble_user_expression(expression)
        # Append name and list of pieces to the 'user_expressions_dict'
        self.user_expressions_dict.update({expr_name: expr_pieces})

    def set_time_series_data(self, time_series_aggregation):
        """
        Function for setting the time series data in the 'parameters' dictionary
        of a component depending on whether a calculation with aggregated time
        series is requested or not.

        :param time_series_aggregation: time series aggregation requested
            (True) or not (False).
        :type time_series_aggregation: boolean
        """
        for param in self.parameters:
            param_dict = self.parameters[param]
            if param_dict['has_time_set']:
                if time_series_aggregation:
                    # Use aggregated data if time series aggr. is requested
                    param_dict['values'] = param_dict['aggregated']
                else:
                    # else use full resolution data with reformatted index (p,t)
                    data = param_dict['full_resolution']
                    idx = pd.MultiIndex.from_product([[0], data.index])
                    param_dict['values'] = pd.Series(data.values, index=idx)

    def get_data_for_time_series_aggregation(self):
        """
        Get all time series data and their respective weights of a component for
        the time series aggregation.
        """
        data, weights = {}, {}
        for param in self.parameters.columns:
            if self.parameters[param]['has_time_set']:  # is True
                unique_name = self.group_name + '_' + param
                data.update(
                    {unique_name: self.parameters[param]['full_resolution']})
                weights.update(
                    {unique_name: self.parameters[param]['tsam_weight']})
        return data, weights

    def set_aggregated_time_series_data(self, data):
        """
        Set (store) the aggregated time series data in the 'parameters'
        dictionary of the component after applying the time series aggregation.
        """
        # Find time series data of a comp. (identifier starts with comp. name)
        for series_name in data.columns:
            if series_name.startswith(self.group_name):
                # Get the original parameter name by slicing the 'unique_name'
                param_name = series_name[len(self.group_name)+1:]
                # Store aggregated time series data in parameters dict of comp.
                self.parameters[param_name]['aggregated'] = data[series_name]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #   Functions for declaring components of the energy system and their
    #   contributions to the objective function
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def declare_component_model_block(self, pyM):
        """
        Create a pyomo Block and store it in attribute 'pyB' of the component.
        """
        setattr(pyM, self.name, pyomo.Block())
        self.pyB = getattr(pyM, self.name)

    def declare_component_variables(self, pyM):
        """
        Create all variables stored in DataFrame 'variables'.
        """
        for var_name in self.variables:
            # Get the pandas series from DataFrame and the variable specs
            var_dict = self.variables[var_name]
            domain = getattr(pyomo, var_dict['domain'])
            # If iterable object (dict, or list) in 'lb' and 'ub' field
            # -> use a lambda function to specify the variable bounds
            if hasattr(var_dict['lb'], '__iter__') \
                    and hasattr(var_dict['ub'], '__iter__'):
                bounds = (lambda m, i: (var_dict['lb'][i], var_dict['ub'][i]))
            else:
                bounds = (var_dict['lb'], var_dict['ub'])
            init = var_dict['init']

            # Differentiation between variables with and without time_set
            if var_dict['has_time_set']:
                setattr(self.pyB, var_name, pyomo.Var(
                        pyM.time_set, domain=domain, bounds=bounds,
                        initialize=init))

            # e.g. for 'BI_MODULE_EX' or for 'SOC' with 'intra_period_time_set'
            elif var_dict['alternative_set'] is not None:
                try:
                    # Try to find the set in the concrete model instance and
                    # use it for the variable declaration
                    alt_set = getattr(pyM, var_dict['alternative_set'])
                    setattr(self.pyB, var_name, pyomo.Var(
                        alt_set, domain=domain, bounds=bounds, initialize=init))
                except AttributeError:  # attr. is a string but it is not found
                    self.log.error('Pyomo model does not have the attribute '
                                   '"%s"' % var_dict['alternative_set'])
                    raise
                except TypeError:  # provided attribute is not a string
                    try:
                        # Assume the entry in the dict is the desired set
                        setattr(self.pyB, var_name, pyomo.Var(
                            var_dict['alternative_set'], domain=domain,
                            bounds=bounds, initialize=init))
                    except Exception:  # e.g. "object is not iterable"
                        self.log.error('Something went wrong in the declaration'
                                       ' of variable "%s" with the set "%s"' % (
                                        var_name, var_dict['alternative_set']))
                        raise
            # built variable without any set
            else:
                setattr(self.pyB, var_name, pyomo.Var(
                        domain=domain, bounds=bounds, initialize=init))

            # Store variable in self.variables[var_name]['pyomo']
            pyomo_var = getattr(self.pyB, var_name)
            self.variables[var_name]['pyomo'] = pyomo_var

    def declare_component_ports(self):
        """
        Create all ports from dictionaries the 'inlet_commod_and_var_names' and
        'outlet_commod_and_var_names' and add the variables to the ports.
        """
        # Create inlet ports
        for commod, var_name in self.inlet_commod_and_var_names.items():
            # Declare port
            port_name = 'inlet_' + commod
            setattr(self.pyB, port_name, network.Port())
            # Add variable to port
            port = getattr(self.pyB, port_name)
            port.add(getattr(self.pyB, var_name), commod,
                     network.Port.Extensive, include_splitfrac=False)

        # Create outlet ports
        for commod, var_name in self.outlet_commod_and_var_names.items():
            # Declare port
            port_name = 'outlet_' + commod
            setattr(self.pyB, port_name, network.Port())
            # Add variable to port
            port = getattr(self.pyB, port_name)
            port.add(getattr(self.pyB, var_name), commod,
                     network.Port.Extensive, include_splitfrac=False)

    def declare_component_user_constraints(self, pyM):
        reserved_chars = ['*', '/', '+', '-', '(', ')', '==', '>=', '<=', '**']
        for expr_name, expr in self.user_expressions_dict.items():
            has_time_dependency = False  # init
            for i, name in enumerate(expr):
                if name not in reserved_chars:
                    try:
                        # Convert it to float if it is possible without an error
                        expr[i] = float(name)
                    except ValueError:
                        # If it is not a float or a reserved char, it should be
                        # a variable or parameter. Check if the name is
                        # available in DataFrames 'variables' or 'parameters'.
                        if name in self.variables.columns:
                            # write a tuple on position of expression:
                            # Pos 0: The variable (simple or indexed)
                            # Pos 1: The bool value if the var has a time set
                            expr[i] = (self.variables[name].loc['pyomo'],
                                       self.variables[name].loc['has_time_set'])
                            # Set flag 'has_time_dependency' to True if at least
                            # one variable in expression has a time dependency
                            if self.variables[name].loc['has_time_set']:
                                has_time_dependency = True
                        elif name in self.parameters.columns:
                            # write a tuple on position of expression:
                            # Pos 0: The parameter (scalar or time series)
                            # Pos 1: The bool value if the param has a time set
                            expr[i] = (self.parameters[name].loc['values'],
                                       self.parameters[name].loc['has_time_set'])
                            if self.parameters[name].loc['has_time_set']:
                                has_time_dependency = True
                        else:
                            raise ValueError('Variable or parameter "{}" has '
                                             'not been declared. Hence, '
                                             'constructing block "{}" failed.'
                                             .format(name, self.name))

            # TODO: Check if order of the operators in the expr list is legal?

            # Create an empty DataFrame (with or without index) to store the
            # expressions that are simplified in the next step
            df_expr = pd.DataFrame(
                index=(pd.MultiIndex.from_tuples(pyM.time_set)
                       if has_time_dependency
                       else pd.MultiIndex.from_tuples([(0, 0)])),
                columns=range(len(expr)))

            for i, name in enumerate(expr):
                if name in reserved_chars or isinstance(name, float):
                    df_expr[i] = name
                elif isinstance(name, tuple) and not name[1]:
                    # name is parameter or variable and has no time set
                    df_expr[i] = [name[0] for _ in range(len(df_expr.index))]
                elif isinstance(name, tuple) and name[1]:
                    # name is parameter or variable and has a time set
                    df_expr[i] = [name[0][j] for j in df_expr.index]

            # Do the simplification process and get two python dictionaries with
            # left hand side (lhs) and right hand side (rhs) expressions for the
            # whole set (time slice) and the operation sign ('==', '>=', '<=')
            lhs, op, rhs = utils.simplify_user_constraint(df_expr)

            # Construct the user constraint using lhs, op, rhs:
            def built_user_constraint_rule(m, p=None, t=None):
                # If the constraint has no index (set) lhs and rhs hold only one
                # entry at index (0, 0) --> "p", "t" are overwritten with zeros
                p = p if p is not None else 0
                t = t if t is not None else 0
                # Do the final lhs/rhs operation and return the expression
                # Note: looping over rows of pandas DataFrames takes much longer
                if op == '<=':
                    return oper.le(lhs[(p, t)], rhs[(p, t)])
                elif op == '==':
                    return oper.eq(lhs[(p, t)], rhs[(p, t)])
                elif op == '>=':
                    return oper.ge(lhs[(p, t)], rhs[(p, t)])

            # Add user constraint to the component block
            setattr(self.pyB, expr_name, pyomo.Constraint(
                pyM.time_set if has_time_dependency else [None],
                rule=built_user_constraint_rule))

    @abstractmethod
    def declare_component_constraints(self, ensys, pyM):
        """ Declares constraints of a component. """
        raise NotImplementedError

    @abstractmethod
    def get_objective_function_contribution(self, ensys, pyM):
        """ Get contribution to the objective function. """
        raise NotImplementedError

    # [...]

    # ==========================================================================
    # --------------------------------------------------------------------------
    #    D E C L A R A T I O N   O F   V A L I D   C O N S T R A I N T S
    # --------------------------------------------------------------------------
    # ==========================================================================

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #    T I M E   I N D E P E N D E N T   C O N S T R A I N T S
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def con_couple_bi_ex_and_cap(self):
        """
        Couples the global existence binary variable with the capacity variable.
        If component does not exist, capacity must take a value of 0.
        Availability of required 'capacity_max' is checked in initialization.
        |br| ``CAP <= BI_EX * capacity_max``
        """
        if self.has_bi_ex:
            bi_ex = self.variables['BI_EX']['pyomo']
            cap = self.variables['CAP']['pyomo']
            cap_max = self.capacity_max

            def con_couple_bi_ex_and_cap(m):
                return cap <= bi_ex * cap_max
            setattr(self.pyB, 'con_couple_bi_ex_and_cap',
                    pyomo.Constraint(rule=con_couple_bi_ex_and_cap))

    def con_cap_min(self):
        """
        Specify the minimum (nominal) capacity of a component (based on its
        basic variable). If a binary existence variable is declared, the minimal
        capacity is only used if the component exists. |br|
        ``CAP >= capacity_min * BI_EX``  or |br|
        ``CAP >= capacity_min`` |br|
        (The maximal capacity is initially set by an upper variable bound.)
        """
        if self.capacity_min is not None:
            cap = self.variables['CAP']['pyomo']
            cap_min = self.capacity_min

            def con_cap_min(m):
                if self.has_bi_ex:
                    bi_ex = self.variables['BI_EX']['pyomo']
                    return cap >= cap_min * bi_ex
                else:
                    return cap >= cap_min
            setattr(self.pyB, 'con_cap_min', pyomo.Constraint(rule=con_cap_min))

    def con_cap_modular(self):
        """
        The nominal capacity of a component is calculated as a product of the
        capacity per module and the number of existing modules. |br|
        ``CAP == capacity_per_module * summation(BI_MODULE_EX)``
        """
        if self.capacity_per_module is not None:
            cap = self.variables['CAP']['pyomo']
            bi_mod = self.variables['BI_MODULE_EX']['pyomo']
            cap_per_mod = self.capacity_per_module

            def con_cap_modular(m):
                return cap == cap_per_mod * pyomo.summation(bi_mod)
            setattr(self.pyB, 'con_cap_modular', pyomo.Constraint(
                rule=con_cap_modular))

    def con_modular_sym_break(self):
        """
        The next module can only be built if the previous one already exists.
        E.g.: |br| ``BI_MODULE_EX[2] <= BI_MODULE_EX[1]``
        """
        if self.capacity_per_module is not None:
            bi_mod = self.variables['BI_MODULE_EX']['pyomo']

            def con_modular_sym_break(m, nr):
                if nr != 1:
                    return bi_mod[nr] <= bi_mod[nr-1]
                else:
                    return pyomo.Constraint.Skip
            setattr(self.pyB, 'con_modular_sym_break',
                    pyomo.Constraint(pyomo.RangeSet(self.maximal_module_number),
                                     rule=con_modular_sym_break))

    def con_couple_existence_and_modular(self):
        """
        Couples the global existence binary variable with the binary existence
        status of the first module. All other modules are indirectly coupled
        via symmetry break constraints. |br|
        ``BI_EX >= BI_MODULE_EX[1]``
        """
        if self.has_bi_ex and self.capacity_per_module is not None:
            bi_ex = self.variables['BI_EX']['pyomo']
            bi_mod = self.variables['BI_MODULE_EX']['pyomo']

            def con_couple_existence_and_modular(m):
                if self.maximal_module_number >= 1:
                    return bi_ex >= bi_mod[1]
                else:
                    return pyomo.Constraint.Skip
            setattr(self.pyB, 'con_couple_existence_and_modular',
                    pyomo.Constraint(rule=con_couple_existence_and_modular))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #    T I M E   D E P E N D E N T   C O N S T R A I N T S
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def con_bi_var_ex_and_op_relation(self, pyM):
        """
        Relationship between the binary variables for existence and operation.
        A component can only be operated if it does exist. |br|
        ``BI_OP[p, t] <= BI_EX``
        """
        # Only required if both binary variables are considered
        if self.has_bi_ex and self.has_bi_op:
            # Get variables:
            bi_ex = self.variables['BI_EX']['pyomo']
            bi_op = self.variables['BI_OP']['pyomo']

            def con_bi_var_ex_and_op_relation(m, p, t):
                return bi_op[p, t] <= bi_ex
            setattr(self.pyB, 'con_bi_var_ex_and_op_relation',
                    pyomo.Constraint(pyM.time_set,
                                     rule=con_bi_var_ex_and_op_relation))

    @abstractmethod
    def con_operation_limit(self, pyM):
        """
        The operation of a component (MWh) is limit by its nominal power (MW)
        multiplied with the number of hours per time step (not for storage
        because it is already a capacity!)
        (with reference to its basic variable).
        E.g.: |br|
        ``Q[p, t] <= CAP * dt``  (conversion, sink, source) or |br|
        ``Q_SOC[p, t] <= CAP`` (storage) or |br|
        ``Q_INLET[p, t] <= CAP * dt`` (bus)
        """
        raise NotImplementedError

    def con_couple_op_binary_and_basic_var(self, pyM):
        """
        If a binary operation variable is declared, it needs to be coupled with
        the basic operation variable. Therefore, a value for overestimation is
        needed. The capacity_max parameter is used for this. Hence, if not
        specified, an error is raised. E.g.: |br|
        ``Q[p, t] <= capacity_max * BI_OP[p, t] * dt``
        """
        if self.has_bi_op and self.capacity_max is not None:
            # Get variables:
            bi_op = self.variables['BI_OP']['pyomo']
            basic_var = self.variables[self.basic_variable]['pyomo']
            cap_max = self.capacity_max
            dt = self.ensys.hours_per_time_step

            def con_couple_op_binary_and_basic_var(m, p, t):
                return basic_var[p, t] <= cap_max * bi_op[p, t] * dt
            setattr(self.pyB, 'con_couple_op_binary_and_basic_var',
                    pyomo.Constraint(pyM.time_set,
                                     rule=con_couple_op_binary_and_basic_var))

    # ==========================================================================
    #    S E R I A L I Z E
    # ==========================================================================
    def serialize(self):

        return OrderedDict([
            ('model_class', self.__class__.__name__),
            ('group_name', self.group_name),
            ('number_in_group', self.number_in_group),
            ('variables', self.get_variable_values()),
            ('parameters', self.get_parameter_values()),
            ('commodities', self.commodities),
            ('inlet_commod_and_var_names', self.inlet_commod_and_var_names),
            ('outlet_commod_and_var_names', self.outlet_commod_and_var_names),
            ('basic_variable', self.basic_variable),
            ('var_connections', self.var_connections),
            ('comp_obj_dict', self.get_obj_values())
        ])

    def get_variable_values(self):
        var_dict = {}
        for var in self.variables.loc['pyomo']:
            var_dict[var.local_name] = str(var.get_values())
        return var_dict

    def get_parameter_values(self):
        para_dict = {}
        for para in self.parameters:
            if hasattr(self.parameters[para]['values'], '__iter__'):
                # Usually data is stored as a pandas Series --> convert to dict
                para_dict[para] = str(self.parameters[para]['values'].to_dict())
            else:
                # float or integer cannot be converted to dict --> use directly
                para_dict[para] = str(self.parameters[para]['values'])
        return para_dict

    def get_obj_values(self):
        obj_values = {}
        for key, val in self.comp_obj_dict.items():
            # if isinstance(val, int) or isinstance(val, float):
            #     obj_values[key] = val
            # else:  # if pyomo expression
            #     obj_values[key] = pyomo.value(val)
            obj_values[key] = pyomo.value(val)
        return obj_values
