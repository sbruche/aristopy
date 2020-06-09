#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
**The Component class**

* Last edited: 2020-06-14
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
from aristopy import utils, EnergySystem


# In Python even a class is an object. The metaclass is the class of a class.
# A metaclass defines how a class behaves. A class is an instance of metaclass.
# The default metaclass used to construct a class in Python is 'type'.
# The metaclass 'ABCMeta' enables the use of the decorator 'abstractmethod'.
# A class that has a metaclass derived from 'ABCMeta' cannot be instantiated
# unless all of its abstract methods and properties are overridden.
# Hence, the 'Component' class cannot be instantiated itself and the classes
# inheriting from 'Component' must override all abstract methods of the parent.

class Component(metaclass=ABCMeta):
    """
    The component class is the parent class for all energy system components.
    In other words, each instance of Source, Sink, Conversion, Bus and Storage
    inherits the parameters and methods of the component class. The component
    class itself contains abstract methods and can, therefore, not be
    instantiated itself. The abstract methods are later overwritten by the child
    classes to enable instantiation.
    """
    def __init__(self, ensys, name, inlet, outlet, basic_variable,
                 has_existence_binary_var=False, has_operation_binary_var=False,
                 time_series_data=None, scalar_params=None,
                 additional_vars=None, user_expressions=None,
                 capacity=None, capacity_min=None, capacity_max=None,
                 capacity_per_module=None, maximal_module_number=None,
                 capex_per_capacity=0, capex_if_exist=0,
                 opex_per_capacity=0, opex_if_exist=0, opex_operation=0
                 ):
        """
        Initialize an instance of the Component class. Note that an instance of
        the class Component itself can not be instantiated since it holds
        abstract methods.

        :param ensys: EnergySystem instance the component is added to.
        :type ensys: instance of aristopy's EnergySystem class

        :param name: Unique name of the component or the component group.
        :type name: str

        :param inlet: The inlet holds a set of aristopy's Flow instances,
            transporting commodities, entering the component.
            The commodities are creating variables inside of the component.
            The variable name is consistent with the name of the commodity,
            if not specified differently in the Flow instance. |br|
            Example: inlet=aristopy.Flow('ABC', var_name='XYZ') => commodity
            'ABC' is entering the component, and is represented by a newly an
            internally added variable with the name 'XYZ'.
        :type inlet: (list of) instance(s) of aristopy's Flow class, or None

        :param outlet: See description of keyword argument 'inlet'.
        :type outlet: (list of) instance(s) of aristopy's Flow class, or None

        :param basic_variable: Components may have multiple variables, but every
            component has only one basic variable. It is used to restrict
            capacities, set operation rates, and calculate CAPEX and OPEX
            (if available). Usually, the basic variable points to a commodity
            variable on the inlet or the outlet of the component. In this case,
            users need to use string inputs 'inlet_variable' or
            'outlet_variable', respectively. If any other variable should be
            used as the basic variable (e.g., added via keyword argument
            'additional_vars'), users need to set the variable name directly,
            as specified during declaration.
        :type basic_variable: str

        :param has_existence_binary_var: States if the component has a binary
            variable that indicates its existence status. If the parameter is
            set to True, a scalar pyomo binary variable is added to the pyomo
            block model of the component (name specified in utils-file,
            default: 'BI_EX'). It can be used to enable minimal component
            capacities ('capacity_min'), or capacity-independent CAPEX and OPEX
            ('capex_if_exist', 'opex_if_exist').
            |br| *Default: False*
        :type has_existence_binary_var: bool

        :param has_operation_binary_var: States if the component has a binary
            variable that indicates its operational status. If the parameter is
            set to True, a pyomo binary variable, utilizing the global
            EnergySystem time set, is added to the pyomo block model of the
            component (name specified in utils-file, default: 'BI_OP').
            It can be used to enable load-depending conversion rates (via
            'user_expressions'), or minimal part-loads ('min_load_rel'), and
            start-up cost ('start_up_cost') (the last two only for Conversion).
            |br| *Default: False*
        :type has_operation_binary_var: bool

        :param time_series_data: Keyword argument for adding time series data to
            the component by using instances of aristopy’s Series class. This
            data can be used for manual scripting in the 'user_expressions'.
            |br| *Default: None*
        :type time_series_data: (list of) instance(s) of aristopy's Series
            class, or None

        :param scalar_params: Keyword argument for adding scalar parameters to
            the component by using a Python dict {'parameter_name': value}.
            The scalar parameters can be used for manual scripting in the
            'user_expressions'.
            |br| *Default: None*
        :type scalar_params: dict, or None

        :param additional_vars: Keyword argument for adding variables to the
            component, that are not automatically created, e.g., by attaching
            Flows to the inlet or outlet. The variables are provided by adding a
            set of aristopy's Var instances. |br|
            Example: additional_vars=aristopy.Var('ABC', domain='Reals', ub=42)
            |br| *Default: None*
        :type additional_vars: (list of) instance(s) of aristopy's Var class,
            or None

        :param user_expressions: Keyword argument for adding expression-like
            strings, which are converted into pyomo constraints during model
            declaration. User expressions can be applied for various ends, e.g.,
            for specifying commodity conversion rates, or limiting capacities.
            The options for mathematical operators are: '==', '>=', '<=', '**',
            '*', '/', '+', '-', '(', ')'. The expressions can handle variables
            that are created with standard names (see Globals in utils-file,
            e.g., CAP), and the variables and parameters added by the user via
            keyword arguments: 'inlet', 'outlet', 'time_series_data',
            'scalar_params', 'additional_vars'.
            |br| Example: user_expressions=['Q == 0.5*F + 2*BI_OP', 'CAP <= 42']
            |br| *Default: None*
        :type user_expressions: (list of) str, or None

        :param capacity: States the fixed capacity value of the component, if
            the component exists. Hence, if the parameter
            'has_existence_binary_var' is set to True, the value for 'capacity'
            is only applied if the component is selected in the optimal design
            (BI_EX=1).
            |br| *Default: None*
        :type capacity: float or int (>=0), or None

        :param capacity_min: States the minimal component capacity, if the
            component exists. Hence, if the parameter 'has_existence_binary_var'
            is set to True, the value for 'capacity_min' is only applied if the
            component is selected in the optimal design (BI_EX=1).
            |br| *Default: None*
        :type capacity_min: float or int (>=0), or None

        :param capacity_max: States the maximal component capacity. This value
            introduces an upper bound for the capacity variable. It also serves
            as an over-estimator and is required if the parameters
            'has_existence_binary_var' or 'has_operation_binary_var' is True.
            |br| *Default: None*
        :type capacity_max: float or int (>=0), or None

        :param capacity_per_module: If a component is modular, its capacity can
            be modeled with parameter 'capacity_per_module'. If a value is
            introduced, an additional binary variable is created (default name:
            'BI_MODULE_EX'), which indicates the existence of each module.
            |br| *Default: None*
        :type capacity_per_module: float or int (>=0), or None

        :param maximal_module_number: This keyword argument works in combination
            with parameter 'capacity_per_module' and indicates the maximal
            number of modules to be installed in the optimal design.
            |br| *Default: None*
        :type maximal_module_number: int (>0), or None

        :param capex_per_capacity: Parameter to calculate the capital investment
            cost associated with the *CAPACITY* of a component. The final value
            for capacity-related CAPEX is obtained by multiplying
            'capex_per_capacity' with the capacity variable value (CAP).
            |br| *Default: 0*
        :type capex_per_capacity: float or int (>=0)

        :param capex_if_exist: Parameter to calculate the capital investment
            cost associated with the *EXISTENCE* of a component. The final value
            for existence-related CAPEX is obtained by multiplying
            'capex_if_exist' with the binary existence variable value (BI_EX).
            |br| *Default: 0*
        :type capex_if_exist: float or int (>=0)

        :param opex_per_capacity: Parameter to calculate the annual operational
            cost associated with the *CAPACITY* of a component. The final value
            for capacity-related OPEX is obtained by multiplying
            'opex_per_capacity' with the capacity variable value (CAP).
            |br| *Default: 0*
        :type opex_per_capacity: float or int (>=0)

        :param opex_if_exist: Parameter to calculate the annual operational
            cost associated with the *EXISTENCE* of a component. The final value
            for existence-related OPEX is obtained by multiplying
            'opex_if_exist' with the binary existence variable value (BI_EX).
            |br| *Default: 0*
        :type opex_if_exist: float or int (>=0)

        :param opex_operation: Parameter to calculate the annual operational
            cost associated with the *OPERATION* of a component (associated with
            its basic variable). The final value for operation-related OPEX is
            obtained by summarizing the products of 'opex_operation' with the
            time-dependent values of the basic variable.
            |br| *Default: 0*
        :type opex_operation: float or int (>=0)
        """

        # Check and set general attributes:
        if not isinstance(ensys, EnergySystem):
            raise TypeError('Input "ensys" requires an EnergySystem instance.')
        self.ensys = ensys
        assert isinstance(name, str), 'The component "name" should be a string!'
        self.name = name  # might be changed by "add_to_energy_system"
        self.group_name = name
        self.number_in_group = 1  # dito
        self.block = None  # pyomo simple Block object
        self.log = None  # Init: Local logger of the component instance

        # V A R I A B L E S:
        # ------------------
        # Initialize an empty pandas DataFrame to store the component variables.
        self.variables = pd.DataFrame(index=[
            'domain', 'has_time_set', 'alternative_set', 'init', 'ub', 'lb',
            'pyomo'])
        # Holds a copy of the 'variables' DataFrame for later reset. It is
        # stored while variables are edited, imported or relaxed (if requested).
        self.variables_copy = pd.DataFrame(index=[
            'domain', 'has_time_set', 'alternative_set', 'init', 'ub', 'lb'])

        # Check and set input values for capacities:
        self.capacity, self.capacity_min, self.capacity_max, \
            self.capacity_per_module, self.maximal_module_number = \
            utils.check_and_set_capacities(
                capacity, capacity_min, capacity_max,
                capacity_per_module, maximal_module_number)

        # Specify a capacity variable (with standard name -> see: utils.CAP) if
        # capacities are stated and add it to the 'variables' DataFrame.
        # Set 'capacity_max' as upper bound for the capacity variable.
        # Lower bound is only used if component is built (depends on the status
        # of the existence binary variable if specified) --> see constraints!
        self.has_capacity_var = True if self.capacity_min is not None or \
            self.capacity_max is not None else False

        if self.has_capacity_var:
            self.add_var(utils.CAP, has_time_set=False, ub=self.capacity_max)

        if self.capacity_per_module is not None:
            # add binary variables for the existence of modules of a component
            self.add_var(name=utils.BI_MODULE_EX, has_time_set=False,
                         alternative_set=range(
                             1, self.maximal_module_number+1),  # 1 to incl. end
                         domain='Binary')

        # Add binary variables for existence and operation with standard names
        # (see: utils.BI_EX, and utils.BI_OP, respectively) if required:
        self.has_bi_ex = has_existence_binary_var
        if self.has_bi_ex:
            self.add_var(name=utils.BI_EX, domain='Binary', has_time_set=False)

        self.has_bi_op = has_operation_binary_var
        if self.has_bi_op:
            self.add_var(name=utils.BI_OP, domain='Binary', has_time_set=True)

        if self.capacity_max is None and (self.has_bi_op or self.has_bi_ex):
            raise ValueError('An over-estimator is required if a binary '
                             'operation or existence variable is specified. '
                             'Hence, please enter a value for "capacity_max"!')

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
                self.add_var(flow.var_name)
            # Only add commodity it if it is not already declared at inlet:
            if self.inlet_commod_and_var_names.get(flow.commodity) is None:
                self.inlet_commod_and_var_names[flow.commodity] = flow.var_name

        for flow in self.outlet:
            # Check variable name at outlet has not already been used at inlet
            if flow.var_name in self.inlet_commod_and_var_names.values():
                raise ValueError(
                    'Commodity "%s" is found at inlet and outlet of '
                    'component "%s" with the same variable name "%s". '
                    'Please use different variable names for both sides!'
                    % (flow.commodity, name, flow.var_name))
            # Add commodity to the commodities list if not available:
            if flow.commodity not in self.commodities:
                self.commodities.append(flow.commodity)
            # Add variable to variables DataFrame if not available:
            if flow.var_name not in self.variables:
                self.add_var(flow.var_name)
            # Only add commodity it if it is not already declared at outlet:
            if self.outlet_commod_and_var_names.get(flow.commodity) is None:
                self.outlet_commod_and_var_names[flow.commodity] = flow.var_name

        # Dictionary used for plotting. Updated during EnergySystem declaration.
        self.var_connections = {}  # {var_name: [connected_arc_names]}

        # ADDITIONAL VARIABLES can be added by the 'additional_vars' argument
        # via instances of aristopy's Var class (done after inlet/outlet init,
        # to raise an error, if variable name already exists).
        add_var_list = utils.check_add_vars_input(additional_vars)
        for var in add_var_list:
            self.add_var(name=var.name, domain=var.domain,
                         has_time_set=var.has_time_set, ub=var.ub, lb=var.lb,
                         alternative_set=var.alternative_set, init=var.init)

        # Every component has a basic variable. It is used to restrict
        # capacities, set operation rates and calculate capex and opex. Usually,
        # the basic variable is referring to a commodity on the inlet or the
        # outlet of the component. But it can also be set via 'additional_vars'.
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

        # P A R A M E T E R S:
        # --------------------
        # Initialize an empty pandas DataFrame to store the component parameters
        self.parameters = pd.DataFrame(index=['tsam_weight', 'has_time_set',
                                              'values', 'full_resolution',
                                              'aggregated'])

        # Add scalar parameters from dict 'scalar_params' {param_name: value}
        if scalar_params is not None:
            utils.check_scalar_params_dict(scalar_params)
            for key, val in scalar_params.items():
                self.add_param(name=key, data=val)

        # Add time series data from keyword argument 'time_series_data'.
        # Check that it only holds instances of aristopy's Series class.
        series_list = utils.check_and_set_time_series_data(time_series_data)
        for series in series_list:
            # Make sure time series has correct length and index:
            data = utils.check_and_convert_time_series(ensys, series.data)
            self.add_param(series.name, data, series.weighting_factor)

        # Check and set the input values for the cost parameters and prevent
        # invalid parameter combinations.
        self.capex_per_capacity = utils.check_and_set_positive_number(
            capex_per_capacity, 'capex_per_capacity')
        self.capex_if_exist = utils.check_and_set_positive_number(
            capex_if_exist, 'capex_if_exist')
        self.opex_per_capacity = utils.check_and_set_positive_number(
            opex_per_capacity, 'opex_per_capacity')
        self.opex_if_exist = utils.check_and_set_positive_number(
            opex_if_exist, 'opex_if_exist')
        self.opex_operation = utils.check_and_set_positive_number(
            opex_operation, 'opex_operation')
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
        # objective function value:
        self.comp_obj_dict = {
            'capex_capacity': 0, 'capex_exist': 0,
            'opex_capacity': 0, 'opex_exist': 0, 'opex_operation': 0,
            'commodity_cost': 0, 'commodity_revenues': 0,
            'start_up_cost': 0}

        # U S E R   E X P R E S S I O N S:
        # --------------------------------
        # Add 'user_expressions' (type: list of strings, or empty list)
        self.user_expressions = utils.check_and_set_user_expr(user_expressions)
        # 'user_expression_dict' holds converted expr -> filled in declaration
        self.user_expressions_dict = {}

    def __repr__(self):
        return '<Component: "%s">' % self.name

    def pprint(self):
        """ Access the pretty print functionality of the pyomo model Block """
        if isinstance(self.block, pyomo.Block):
            self.block.pprint()

    # ==========================================================================
    #    A D D   C O M P O N E N T   T O   T H E   E N E R G Y   S Y S T E M
    # ==========================================================================
    def add_to_energy_system(self, ensys, group, instances_in_group=1):
        """
        Add the component to the specified instance of the EnergySystem class.
        This implies, creation of copies with unique names if multiple identical
        components should be instantiated ('instances_in_group' > 1), adding the
        component to the dictionary 'components' of the EnergySystem, and
        initializing a Logger instance for each component.

        *Method is not intended for public access!*

        :param ensys: EnergySystem instance the component is added to.
        :param group: Name (str) of the group where the component is added to
            (corresponds to initialization keyword attribute 'name').
        :param instances_in_group: (int > 0) States the number of similar
            component instances that are simultaneously created and arranged in
            a group. That means, the user has the possibility to instantiate
            multiple component instances (only for Conversion!) with identical
            specifications. These components work independently, but may have an
            order for their binary existence and/or operation variables (see:
            'group_has_existence_order', 'group_has_operation_order'). If a
            number larger than 1 is provided, the names of the components are
            extended with integers starting from 1 (e.g., 'conversion_1', ...).
            |br| *Default: 1*
        """
        # Reset flags if new components are added
        ensys.is_model_declared, ensys.is_data_clustered = False, False

        for nbr in range(1, instances_in_group + 1):
            if instances_in_group > 1:
                # Create a DEEP-copy of self! Shallow copy creates problems
                # since the Component object holds complicating elements
                # (e.g., pandas DataFrames)
                instance = copy.deepcopy(self)
                # We always need the reference on the original EnergySystem
                # instance and not on a newly created copy of it!
                instance.ensys = self.ensys
                instance.name = group + '_{}'.format(nbr)  # overwrite name
                instance.number_in_group = nbr  # overwrite number_in_group
            else:
                instance = self  # reference self in 'instance'
            # If component name already exists raise an error.
            if instance.name in ensys.components:
                raise ValueError('Component name "%s" already exists!'
                                 % instance.name)
            else:
                # Add component with its name and all attributes and methods to
                # the dictionary 'components' of the EnergySystem instance
                ensys.components[instance.name] = instance

                # Define an own Logger instance for every added component
                instance.log = instance.ensys.logger.get_logger(instance)
                instance.log.info('Add component "%s" to the EnergySystem'
                                  % instance.name)

    # ==========================================================================
    #    M A N I P U L A T I O N   O F   V A R I A B L E S
    # ==========================================================================
    def add_var(self, name, domain='NonNegativeReals', has_time_set=True,
                alternative_set=None, ub=None, lb=None, init=None):
        """
        Method for adding a variable to the pandas DataFrame 'variables'.

        *Method is not intended for public access!*

        :param name: (str) Name / identifier of the added variable.
        :param domain: (str) A super-set of the values the variable can take on.
            Possible values are: 'Reals', 'NonNegativeReals', 'Binary'.
            |br| *Default: 'NonNegativeReals'*
        :param has_time_set: (bool) Is True, if the time set of the EnergySystem
            instance is also a set of the added variable. |br| * Default: True
        :param alternative_set: Alternative variable sets can be added here via
            iterable Python objects (e.g. list). |br| *Default: None*
        :param ub: (float, int, None) Upper variable bound. |br| *Default: None*
        :param lb: (float, int, None) Lower variable bound. |br| *Default: None*
        :param init: A function or Python object that provides starting values
            for the added variable. |br| *Default: None*
        """
        # Make sure the variable name is unique in the Dataframe
        if name in self.variables:
            raise ValueError('Variable with name "%s" already found in the '
                             'DataFrame. Please use a different name.' % name)
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

    def store_var_copy(self, name, var):
        """
        Method to store the current specifications of a provided variable in
        the pandas DataFrame "variables_copy".

        *Method is not intended for public access!*

        :param name: (str) Name / identifier of the stored variable.
        :param var: (pandas Series) Variable data to store.
        """
        series = pd.Series({'has_time_set': var['has_time_set'],
                            'alternative_set': var['alternative_set'],
                            'domain': var['domain'], 'init': var['init'],
                            'ub': var['ub'], 'lb': var['lb']})
        # Only store it if it is not already in the DataFrame => otherwise
        # a more original version might be overwritten unintentionally.
        if name not in self.variables_copy:
            self.variables_copy[name] = series

    def relax_integrality(self, include_existence=True, include_modules=True,
                          include_time_dependent=True,
                          store_previous_variables=True):
        """
        Method to relax the integrality of the binary variables. This means
        binary variables are declared to be 'NonNegativeReals' with an upper
        bound of 1. This method encompasses the resetting of the DataFrame
        "variables" and the pyomo variables itself (if already constructed).
        The relaxation can be performed for the binary existence variable, the
        module existence binary variables and time-dependent binary variables.

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
            DataFrame "variables_copy". This representation can be used by
            method "reset_variables" to undo changes.
            |br| *Default: True*
        :type store_previous_variables: bool
        """
        # Check the input:
        utils.check_and_set_bool(include_existence, 'include_existence')
        utils.check_and_set_bool(include_modules, 'include_modules')
        utils.check_and_set_bool(include_time_dependent,
                                 'include_time_dependent')
        utils.check_and_set_bool(store_previous_variables,
                                 'store_previous_variables')

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

        # Loop through the DataFrame "variables"
        for v in self.variables.columns:
            # Include the global existence variable:
            if include_existence and v == utils.BI_EX:
                if store_previous_variables:
                    self.store_var_copy(v, self.variables[v])
                relax_variable(self.variables[v])
                # self.log.debug('Relax integrality of variable: "%s"' % v)
            # Include the module existence variables:
            elif include_modules and v == utils.BI_MODULE_EX:
                if store_previous_variables:
                    self.store_var_copy(v, self.variables[v])
                relax_variable(self.variables[v])
                # self.log.debug('Relax integrality of variable: "%s"' % v)
            # Include time-dependent variables
            elif include_time_dependent and self.variables[v]['domain'] \
                    == 'Binary' and self.variables[v]['has_time_set']:
                if store_previous_variables:
                    self.store_var_copy(v, self.variables[v])
                relax_variable(self.variables[v])
                # self.log.debug('Relax integrality of variable: "%s"' % v)

    def edit_variable(self, name, store_previous_variables=True, **kwargs):
        """
        Method on component level for manipulating the specifications of
        already defined component variables (e.g., change variable domain,
        add variable bounds, etc.).

        :param name: Name / identifier of the edited variable.
        :type name: str

        :param store_previous_variables: State whether the representation of the
            variables before applying the edit_variable method should be stored
            in DataFrame "variables_copy" of each component. This representation
            can be used by mehtod "reset_variables" to undo changes.
            |br| *Default: True*
        :type store_previous_variables: bool

        :param kwargs: Additional keyword arguments for editing. Options are:
            'ub' and 'lb' to add an upper or lower variable bound, 'domain' to
            set the variable domain, and 'has_time_set' to define if the
            variable should inherit the global time set of the EnergySystem.
        """
        # Check the method input (done locally because might be called
        # directly from component).
        utils.check_edit_var_input(name, store_previous_variables, **kwargs)

        # Check if the variable with given 'name' is available for editing:
        if name not in self.variables.columns:
            self.log.warn('No variable "%s" available for editing' % name)
        else:
            self.log.info('Edit variable: "%s"' % name)
            # Store the previous variable setting if requested
            if store_previous_variables:
                self.store_var_copy(name=name, var=self.variables[name])
            # Write keyword arguments in "variables" DataFrame
            for key, val in kwargs.items():
                self.variables[name][key] = val

            # Edit the pyomo variable if it has already been constructed
            if self.ensys.is_model_declared:
                var_pyomo = self.variables[name]['pyomo']
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

    def reset_variables(self):
        """
        Method to reset the variables to the state, that is stored in the
        DataFrame "variables_copy". This includes the resetting of the DataFrame
        "variables" and the pyomo variables itself (if already constructed).
        """
        # Loop through variables in DataFrame "variables_copy"
        for var_copy_name in self.variables_copy:

            # The variable should be declared in the DataFrame "variables"
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

            # Delete the variable from the "variables_copy" after resetting
            self.variables_copy.drop(var_copy_name, axis=1, inplace=True)

    # ==========================================================================
    #    E X P O R T   AND   I M P O R T   C O N F I G U R A T I O N
    # ==========================================================================
    def export_component_configuration(self):
        """
        This method exports the component configuration data (results of the
        optimization) as a pandas Series. The features are (if exist):

        * the binary existence variable (utils.BI_EX),
        * the binary existence variables of modules (utils.BI_MODULE_EX)
        * the component capacity variable (of main commodity, utils.CAP)

        :return: The configuration of the modelled component instance.
        :rtype: pandas Series
        """
        bi_ex_val, bi_mod_ex_val, cap_val = None, None, None

        if self.has_bi_ex:
            bi_ex_val = self.variables[utils.BI_EX]['pyomo'].value

        if self.capacity_per_module is not None:
            var = self.variables[utils.BI_MODULE_EX]['pyomo']
            bi_mod_ex_val = {i: var[i].value for i in range(
                1, self.maximal_module_number + 1)}  # equiv. to pyomo.RangeSet

        if self.has_capacity_var:
            cap_val = self.variables[utils.CAP]['pyomo'].value

        series = pd.Series({utils.BI_EX: bi_ex_val,
                            utils.BI_MODULE_EX: bi_mod_ex_val,
                            utils.CAP: cap_val})
        # Replace numpy NaNs that might occur here and there with None
        series = series.replace({nan: None})

        return series

    def import_component_configuration(self, data, fix_existence=True,
                                       fix_modules=True, fix_capacity=True,
                                       store_previous_variables=True):
        """
        Method to load a pandas Series with configuration specifications
        (binary existence variables and capacity variable values). The values
        are used to fix a specific component configuration (for example from
        other model runs).

        :param data: The configuration of the modelled component instance.
        :type data: pandas Series

        :param fix_existence: Specify whether the imported (global) binary
            component existence variable should be fixed (if available).
            |br| *Default: True*
        :type fix_existence: bool

        :param fix_modules: Specify whether the imported binary existence
            variable of the component modules should be fixed (if available).
            |br| *Default: True*
        :type fix_modules: bool

        :param fix_capacity: Specify whether the imported component capacity
            variable (of the main commodity) should be fixed or not.
            |br| *Default: True*
        :type fix_capacity: bool

        :param store_previous_variables: State whether the representation of the
            variables before applying the configuration import should be stored
            in DataFrame "variables_copy" of each component. This representation
            can be used by method "reset_variables" to undo changes.
            |br| *Default: True*
        :type store_previous_variables: bool
        """
        # Check input:
        if not isinstance(data, pd.Series):
            raise TypeError('The data needs to be imported as a pandas Series!')
        utils.check_and_set_bool(fix_existence, 'fix_existence'),
        utils.check_and_set_bool(fix_modules, 'fix_modules')
        utils.check_and_set_bool(fix_capacity, 'fix_capacity'),
        utils.check_and_set_bool(store_previous_variables,
                                 'store_previous_variables')

        # Note: Importing seems to work without rounding => if not: round values

        # EXISTENCE VARIABLE: Fix if required and available in model and data
        if fix_existence and self.has_bi_ex and data[utils.BI_EX] is not None:
            # Backup of variables in "variables_copy" if requested for resetting
            if store_previous_variables:
                self.store_var_copy(name=utils.BI_EX,
                                    var=self.variables[utils.BI_EX])
            # Always possible to set bounds (also if not declared)
            self.variables[utils.BI_EX]['lb'] = data[utils.BI_EX]
            self.variables[utils.BI_EX]['ub'] = data[utils.BI_EX]
            if self.ensys.is_model_declared:  # model / variables are declared
                self.variables[utils.BI_EX]['pyomo'].fix(data[utils.BI_EX])
                # additional step required for persistent model
                if self.ensys.is_persistent_model_declared:
                    self.ensys.solver.update_var(
                        self.variables[utils.BI_EX]['pyomo'])

        # MODULE EXISTENCE VARIABLE: Fix if required and available
        if fix_modules and self.capacity_per_module is not None \
                and data[utils.BI_MODULE_EX] is not None:
            # Backup of variables in "variables_copy" if requested for resetting
            if store_previous_variables:
                self.store_var_copy(name=utils.BI_MODULE_EX,
                                    var=self.variables[utils.BI_MODULE_EX])
            # Always possible to set bounds (also if not declared)
            self.variables[utils.BI_MODULE_EX]['lb'] = data[utils.BI_MODULE_EX]
            self.variables[utils.BI_MODULE_EX]['ub'] = data[utils.BI_MODULE_EX]
            if self.ensys.is_model_declared:
                for i in range(1, self.maximal_module_number + 1):
                    self.variables[utils.BI_MODULE_EX]['pyomo'][i].fix(
                        data[utils.BI_MODULE_EX][i])
                    # additional step required for persistent model
                    if self.ensys.is_persistent_model_declared:
                        self.ensys.solver.update_var(
                            self.variables[utils.BI_MODULE_EX]['pyomo'][i])

        # CAPACITY VARIABLE: Fix if required and available in model and data
        if fix_capacity and self.has_capacity_var \
                and data[utils.CAP] is not None:
            # Backup of variables in "variables_copy" if requested for resetting
            if store_previous_variables:
                self.store_var_copy(name=utils.CAP,
                                    var=self.variables[utils.CAP])
            # Always possible to set bounds (also if not declared)
            self.variables[utils.CAP]['lb'] = data[utils.CAP]
            self.variables[utils.CAP]['ub'] = data[utils.CAP]
            if self.ensys.is_model_declared:  # model / variables are declared
                self.variables[utils.CAP]['pyomo'].fix(data[utils.CAP])
                # additional step required for persistent model
                if self.ensys.is_persistent_model_declared:
                    self.ensys.solver.update_var(
                        self.variables[utils.CAP]['pyomo'])

    # ==========================================================================
    #    P A R A M E T E R S   AND   T I M E   S E R I E S
    # ==========================================================================
    def add_param(self, name, data, tsam_weight=1):
        """
        Method for adding a parameter to the pandas DataFrame 'parameters'.

        *Method is not intended for public access!*

        :param name: (str) Name / identifier of the added parameter.
        :param data: (float, int, list, dict, numpy array, pandas Series) Data.
        :param tsam_weight: (int or float) Weighting factor to use in case the
            provided parameter is a series and an aggregation is requested.
            |br| *Default: 1*
        """
        # Make sure the parameter name is unique in the Dataframe
        if name in self.parameters:
            raise ValueError('Parameter with name "%s" already found in the '
                             'DataFrame. Please use a different name.' % name)
        # If input is integer or float, flag 'has_time_set' is set to False.
        # Else, (input: numpy array, list, dict or pandas series) flag -> True
        has_time_set = False if isinstance(data, (int, float)) else True
        # Append data to DataFrame
        series = pd.Series({'tsam_weight': tsam_weight,
                            'has_time_set': has_time_set,
                            'values': data, 'full_resolution': data,
                            'aggregated': None})
        self.parameters[name] = series

    def set_time_series_data(self, use_clustered_data):
        """
        Sets the time series data in the 'parameters' DataFrame of a component
        depending on whether a calculation with aggregated time series data
        should be performed or with the original data (without clustering).

        *Method is not intended for public access!*

        :param use_clustered_data: Use aggregated data (True), or original
            data (False).
        :type use_clustered_data: bool
        """
        for param in self.parameters:
            param_dict = self.parameters[param]
            if param_dict['has_time_set']:
                if use_clustered_data:
                    # Use aggregated data if time series aggr. is requested
                    param_dict['values'] = param_dict['aggregated']
                else:
                    # else use full resolution data with reformatted index (p,t)
                    data = param_dict['full_resolution']
                    idx = pd.MultiIndex.from_product([[0], data.index])
                    param_dict['values'] = pd.Series(data.values, index=idx)

    def get_time_series_data_for_aggregation(self):
        """
        Collect all time series data and their respective weights and merge
        them in two dictionaries. The returned dictionaries are used for the
        time series aggregation in the "cluster" method of the EnergySystem
        instance.

        *Method is not intended for public access!*

        :returns: (dict) time series data, (dict) time series weights
        """
        data, weights = {}, {}
        for param in self.parameters:
            if self.parameters[param]['has_time_set']:  # is True
                unique_name = self.group_name + '_' + param
                data[unique_name] = self.parameters[param]['full_resolution']
                weights[unique_name] = self.parameters[param]['tsam_weight']
        return data, weights

    def set_aggregated_time_series_data(self, data):
        """
        Store the aggregated time series data in the 'parameters' DataFrame.

        *Method is not intended for public access!*

        :param data: (pandas DataFrame) DataFrame with the aggregated time
            series data of all components.
        """
        # Find time series data of a comp. (identifier starts with group name
        # and the parameter name is in the parameters DataFrame)
        for series_name in data:
            if series_name.startswith(self.group_name) and \
                    series_name[len(self.group_name)+1:] in self.parameters:
                # Get the original parameter name by slicing the 'unique_name'
                param_name = series_name[len(self.group_name)+1:]
                # Store aggregated time series data in parameters dict of comp.
                self.parameters[param_name]['aggregated'] = data[series_name]

    # ==========================================================================
    #    M O D E L   B L O C K   D E C L A R A T I O N
    # ==========================================================================
    def declare_component_model_block(self, model):
        """
        Create a pyomo Block and store it in attribute 'block' of the component.
        The pyomo Block is the container for all pyomo objects of the component.

        *Method is not intended for public access!*

        :param model: Pyomo ConcreteModel of the EnergySystem instance
        """
        setattr(model, self.name, pyomo.Block())
        self.block = getattr(model, self.name)

    # ==========================================================================
    #    V A R I A B L E   D E C L A R A T I O N
    # ==========================================================================
    def declare_component_variables(self, model):
        """
        Create all pyomo variables that are stored in DataFrame 'variables'.

        *Method is not intended for public access!*

        :param model: Pyomo ConcreteModel of the EnergySystem instance
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
                setattr(self.block, var_name, pyomo.Var(
                        model.time_set, domain=domain, bounds=bounds,
                        initialize=init))

            # e.g. for 'BI_MODULE_EX' or for 'SOC' with 'intra_period_time_set'
            elif var_dict['alternative_set'] is not None:
                try:
                    # Try to find the set in the concrete model instance and
                    # use it for the variable declaration
                    alt_set = getattr(model, var_dict['alternative_set'])
                    setattr(self.block, var_name, pyomo.Var(
                        alt_set, domain=domain, bounds=bounds, initialize=init))
                except AttributeError:  # attr. is a string but it is not found
                    self.log.error('Pyomo model does not have the attribute '
                                   '"%s"' % var_dict['alternative_set'])
                    raise
                except TypeError:  # provided attribute is not a string
                    try:
                        # Assume the entry in the dict is the desired set
                        setattr(self.block, var_name, pyomo.Var(
                            var_dict['alternative_set'], domain=domain,
                            bounds=bounds, initialize=init))
                    except Exception:  # e.g. "object is not iterable"
                        self.log.error('Something went wrong in the declaration'
                                       ' of variable "%s" with the set "%s"' % (
                                        var_name, var_dict['alternative_set']))
                        raise
            # built variable without any set
            else:
                setattr(self.block, var_name, pyomo.Var(
                        domain=domain, bounds=bounds, initialize=init))

            # Store variable in self.variables[var_name]['pyomo']
            pyomo_var = getattr(self.block, var_name)
            self.variables[var_name]['pyomo'] = pyomo_var

    # ==========================================================================
    #    P O R T   D E C L A R A T I O N
    # ==========================================================================
    def declare_component_ports(self):
        """
        Create all ports from the dictionaries 'inlet_commod_and_var_names' and
        'outlet_commod_and_var_names', and assign port variables.

        Note: The ports are currently always created with extensive behaviour!

        *Method is not intended for public access!*
        """
        # Create inlet ports
        for commod, var_name in self.inlet_commod_and_var_names.items():
            # Declare port
            port_name = 'inlet_' + commod
            setattr(self.block, port_name, network.Port())
            # Add variable to port
            port = getattr(self.block, port_name)
            port.add(getattr(self.block, var_name), commod,
                     network.Port.Extensive, include_splitfrac=False)

        # Create outlet ports
        for commod, var_name in self.outlet_commod_and_var_names.items():
            # Declare port
            port_name = 'outlet_' + commod
            setattr(self.block, port_name, network.Port())
            # Add variable to port
            port = getattr(self.block, port_name)
            port.add(getattr(self.block, var_name), commod,
                     network.Port.Extensive, include_splitfrac=False)

    # ==========================================================================
    #    U S E R   C O N S T R A I N T   D E C L A R A T I O N
    # ==========================================================================
    def declare_component_user_constraints(self, model):
        """
        Create all constraints, that are introduced by manual scripting in the
        keyword argument 'user_expressions'.

        *Method is not intended for public access!*

        :param model: Pyomo ConcreteModel of the EnergySystem instance
        """
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

            # Create an empty DataFrame (with or without index) to store the
            # expressions that are simplified in the next step
            df_expr = pd.DataFrame(
                index=(pd.MultiIndex.from_tuples(model.time_set)
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
            setattr(self.block, expr_name, pyomo.Constraint(
                model.time_set if has_time_dependency else [None],
                rule=built_user_constraint_rule))

    # ==========================================================================
    #    C O N V E N T I O N A L   C O N S T R A I N T   D E C L A R A T I O N
    # ==========================================================================
    @abstractmethod
    def declare_component_constraints(self, ensys, model):
        """ Abstract method to create (conventional) component constraints. """
        raise NotImplementedError

    # **************************************************************************
    #    Time-independent constraints
    # **************************************************************************
    def con_couple_bi_ex_and_cap(self):
        """
        Constraint to couple the global existence binary variable with the
        capacity variable. If the component does not exist, the capacity must
        take a value of 0. Note: The availability of the required 'capacity_max'
        parameter is checked during initialization.
        E.g.: |br| ``CAP <= BI_EX * capacity_max``

        *Method is not intended for public access!*
        """
        if self.has_bi_ex:
            bi_ex = self.variables[utils.BI_EX]['pyomo']
            cap = self.variables[utils.CAP]['pyomo']
            cap_max = self.capacity_max

            def con_couple_bi_ex_and_cap(m):
                return cap <= bi_ex * cap_max
            setattr(self.block, 'con_couple_bi_ex_and_cap',
                    pyomo.Constraint(rule=con_couple_bi_ex_and_cap))

    def con_cap_min(self):
        """
        Constraint to set the minimum capacity of a component (based on its
        basic variable). If a binary existence variable is declared, the minimal
        capacity is only enforced if the component exists. E.g.: |br|
        ``CAP >= capacity_min * BI_EX``  or |br|
        ``CAP >= capacity_min`` |br|

        *Method is not intended for public access!*
        """
        if self.capacity_min is not None:
            cap = self.variables[utils.CAP]['pyomo']
            cap_min = self.capacity_min

            def con_cap_min(m):
                if self.has_bi_ex:
                    bi_ex = self.variables[utils.BI_EX]['pyomo']
                    return cap >= cap_min * bi_ex
                else:
                    return cap >= cap_min
            setattr(self.block, 'con_cap_min',
                    pyomo.Constraint(rule=con_cap_min))

    def con_cap_modular(self):
        """
        Constraint to calculate the nominal capacity of a component from the
        product of the capacity per module and the number of existing modules.
        E.g.: |br| ``CAP == capacity_per_module * summation(BI_MODULE_EX)``

        *Method is not intended for public access!*
        """
        if self.capacity_per_module is not None:
            cap = self.variables[utils.CAP]['pyomo']
            bi_mod = self.variables[utils.BI_MODULE_EX]['pyomo']
            cap_per_mod = self.capacity_per_module

            def con_cap_modular(m):
                return cap == cap_per_mod * pyomo.summation(bi_mod)
            setattr(self.block, 'con_cap_modular', pyomo.Constraint(
                rule=con_cap_modular))

    def con_modular_sym_break(self):
        """
        Constraint to state, that the next module can only be built if the
        previous one already exists (symmetry break constraint for components
        consisting of multiple modules).
        E.g.: |br| ``BI_MODULE_EX[2] <= BI_MODULE_EX[1]``

        *Method is not intended for public access!*
        """
        if self.capacity_per_module is not None:
            bi_mod = self.variables[utils.BI_MODULE_EX]['pyomo']

            def con_modular_sym_break(m, nr):
                if nr != 1:
                    return bi_mod[nr] <= bi_mod[nr-1]
                else:
                    return pyomo.Constraint.Skip
            setattr(self.block, 'con_modular_sym_break',
                    pyomo.Constraint(pyomo.RangeSet(self.maximal_module_number),
                                     rule=con_modular_sym_break))

    def con_couple_existence_and_modular(self):
        """
        Constraint to couple the global existence binary variable with the
        binary existence status of the first module. All other modules are
        indirectly coupled via symmetry breaks. E.g.: |br|
        ``BI_EX >= BI_MODULE_EX[1]``

        *Method is not intended for public access!*
        """
        if self.has_bi_ex and self.capacity_per_module is not None:
            bi_ex = self.variables[utils.BI_EX]['pyomo']
            bi_mod = self.variables[utils.BI_MODULE_EX]['pyomo']

            def con_couple_existence_and_modular(m):
                if self.maximal_module_number >= 1:
                    return bi_ex >= bi_mod[1]
                else:
                    return pyomo.Constraint.Skip
            setattr(self.block, 'con_couple_existence_and_modular',
                    pyomo.Constraint(rule=con_couple_existence_and_modular))

    # **************************************************************************
    #    Time-dependent constraints
    # **************************************************************************
    def con_bi_var_ex_and_op_relation(self, model):
        """
        Constraint to set a relationship between the binary variables for
        existence and operation. A component can only be operated if it exists.
        E.g.: |br| ``BI_OP[p, t] <= BI_EX``

        *Method is not intended for public access!*
        """
        # Only required if both binary variables are considered
        if self.has_bi_ex and self.has_bi_op:
            bi_ex = self.variables[utils.BI_EX]['pyomo']
            bi_op = self.variables[utils.BI_OP]['pyomo']

            def con_bi_var_ex_and_op_relation(m, p, t):
                return bi_op[p, t] <= bi_ex
            setattr(self.block, 'con_bi_var_ex_and_op_relation',
                    pyomo.Constraint(model.time_set,
                                     rule=con_bi_var_ex_and_op_relation))

    @abstractmethod
    def con_operation_limit(self, model):
        """
        Constraint to limit the operation (value of the basic variable) of a
        component (MWh) by its nominal power (MW) multiplied with the number of
        hours per time step (not for Storage because it is already a capacity!).
        E.g.: |br|
        ``Q[p, t] <= CAP * dt``  (Conversion, Sink, Source) |br|
        ``SOC[p, t] <= CAP`` (Storage) |br|
        ``Q_IN[p, t] <= CAP * dt`` (Bus)

        *Method is not intended for public access!*
        """
        raise NotImplementedError

    def con_couple_op_binary_and_basic_var(self, model):
        """
        Constraint to couple the binary operation variable (is available), with
        the basic variable of the component. Therefore, the "capacity_max"
        parameter serves for overestimation (BigM). E.g.: |br|
        ``Q[p, t] <= capacity_max * BI_OP[p, t] * dt``

        *Method is not intended for public access!*
        """
        if self.has_bi_op:
            bi_op = self.variables[utils.BI_OP]['pyomo']
            basic_var = self.variables[self.basic_variable]['pyomo']
            cap_max = self.capacity_max
            dt = self.ensys.hours_per_time_step

            def con_couple_op_binary_and_basic_var(m, p, t):
                return basic_var[p, t] <= cap_max * bi_op[p, t] * dt
            setattr(self.block, 'con_couple_op_binary_and_basic_var',
                    pyomo.Constraint(model.time_set,
                                     rule=con_couple_op_binary_and_basic_var))

    # ==========================================================================
    #    O B J E C T I V E   F U N C T I O N   C O N T R I B U T I O N
    # ==========================================================================
    def get_objective_function_contribution(self, ensys, model):
        """
        Calculate the objective function contributions of the component and add
        the values to the component dictionary "comp_obj_dict".

        *Method is not intended for public access!*

        :param ensys: Instance of the EnergySystem class
        :param model: Pyomo ConcreteModel of the EnergySystem instance
        """

        # The following part is the same for all components:
        # Alias of the components' objective function dictionary
        obj = self.comp_obj_dict
        # Get general required variables:
        basic_var = self.variables[self.basic_variable]['pyomo']

        # CAPEX :
        # ~~~~~~~
        # CAPEX depending on capacity of component
        if self.capex_per_capacity > 0:
            cap = self.variables[utils.CAP]['pyomo']
            obj['capex_capacity'] = -1 * self.capex_per_capacity * cap

        # CAPEX depending on existence of component
        if self.capex_if_exist > 0:
            bi_ex = self.variables[utils.BI_EX]['pyomo']
            obj['capex_exist'] = -1 * self.capex_if_exist * bi_ex

        # OPEX :
        # ~~~~~~
        # OPEX depending on capacity of component
        if self.opex_per_capacity > 0:
            cap = self.variables[utils.CAP]['pyomo']
            obj['opex_capacity'] = -1 * ensys.pvf * self.opex_per_capacity * cap

        # OPEX depending on existence of component
        if self.opex_if_exist > 0:
            bi_ex = self.variables[utils.BI_EX]['pyomo']
            obj['opex_exist'] = -1 * ensys.pvf * self.opex_if_exist * bi_ex

        # OPEX for operation (related to basic variable!)
        if self.opex_operation > 0:
            obj['opex_operation'] = -1 * ensys.pvf * self.opex_operation * sum(
                basic_var[p, t] * ensys.period_occurrences[p] for p, t in
                model.time_set) / ensys.number_of_years

        return sum(self.comp_obj_dict.values())

    # ==========================================================================
    #    S E R I A L I Z E
    # ==========================================================================
    def serialize(self):
        """
        This method collects all relevant input data and optimization results
        from the Component instance, and returns them in an ordered dictionary.

        :return: OrderedDict
        """
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
