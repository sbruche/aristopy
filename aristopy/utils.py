#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
**Globals, helper and utility functions**

* Last edited: 2020-06-30
* Created by: Stefan Bruche (TU Berlin)
"""
import warnings
import pandas as pd
import numpy as np
import aristopy
import pyomo.environ as pyo

# ==============================================================================
# GLOBAL VARIABLE NAMES:
# ----------------------
# These names (strings) are used to create the constraints of the optimization
# model. They can be applied for manual scripting of expressions inside of the
# components. E.g., user_expressions='CAP <= 42 * BI_EX'
CAP = 'CAP'                    # component capacity variable
BI_EX = 'BI_EX'                # binary existence variable
BI_OP = 'BI_OP'                # binary operation variable
BI_MODULE_EX = 'BI_MODULE_EX'  # binary existence variable of component modules
BI_SU = 'BI_SU'                # binary start-up variable of conversion units
BI_SU_INTER = 'BI_SU_INTER'    # binary start-up var. (inter-period time-steps)
SOC = 'SOC'                    # State of charge var. of storage components
SOC_MAX = 'SOC_MAX'            # State of charge var. for maximal energy content
SOC_MIN = 'SOC_MIN'            # State of charge var. for minimal energy content
SOC_INTER = 'SOC_INTER'        # State of charge var. (inter-period time-steps)
# ==============================================================================


def aggregate_time_series(full_series, number_of_time_steps_to_aggregate,
                          aggregate_by='sum'):
    """
    Helper function to compress the original (full scale) time series to a
    shorter one. Returns an array with aggregated values. For aggregation the
    values can either be summed up ('sum') or averaged ('mean').
    (E.g. s = aggregate_time_series([1, 2, 3, 4], 2, 'sum') >> s = [3, 7]

    :param full_series: Original (full scale) hourly time series to aggregate.
        E.g. two time steps (hours) are summed up or averaged and handled as one
    :type full_series: pandas Series or numpy array or list

    :param number_of_time_steps_to_aggregate: Number of time steps to aggregate.
        Should be an integer divisor of the length of the full series.
    :type number_of_time_steps_to_aggregate: int

    :param aggregate_by: aggregation method to apply. Can either be 'sum'
        (values are summed up) or 'mean' (values are averaged).
    :type aggregate_by: str ('sum' or 'mean')
    """
    # Check the user input:
    if not (isinstance(full_series, (pd.Series, np.ndarray, list))):
        raise ValueError('The "full_series" can either by a pandas Series or a '
                         'numpy ndarray or a Python list.')
    check_positive_int(number_of_time_steps_to_aggregate,
                       'number_of_time_steps_to_aggregate')
    if not (aggregate_by == 'sum' or aggregate_by == 'mean'):
        raise ValueError('Keyword "aggregate_by" can either be "sum" or "mean"')

    # Check that number_of_time_steps_to_aggregate is an integer divisor of len
    if len(full_series) % number_of_time_steps_to_aggregate != 0:
        raise ValueError('The provided series can not be equally dived with '
                         'length {}'.format(number_of_time_steps_to_aggregate))
    else:
        # convert input to numpy array and reshape it e.g. [[1, 2] [3, 4], ...]
        s = np.array(full_series).reshape(-1, number_of_time_steps_to_aggregate)
        if aggregate_by == 'sum':
            # sum everything over axis 1 (horizontal) -> [3 7]
            s = s.sum(1)
        else:  # aggregate_by == 'mean'
            # calculate mean over axis 1 -> [1.5 3.5]
            s = s.mean(1)
        return s


def check_add_constraint(rule, name, has_time_set, alternative_set):
    """ Check the user input for the function 'add_constraint' that can be used
        to manually add constraints to the ConcreteModel instance. """
    if not callable(rule):
        raise TypeError('The "rule" keyword needs to hold a callable object!')

    assert isinstance(name, (str, type(None))), '"name" should be a string'
    assert isinstance(has_time_set, bool), '"has_time_set" should be a boolean'

    if alternative_set is not None and not hasattr(alternative_set, '__iter__'):
        raise TypeError('The "alternative_set" keyword requires a iterable '
                        'Python object!')


def check_add_vars_input(data):
    """
    Function to check the input of components "additional_vars" argument and
    the "add_variable" function of the energy system model instance. Input is
    required as instances of aristopy's Var class (multiple -> arranged in list)

    :return: list (of aristopy Var instances)
    """
    exception = ValueError("Invalid data type for added variable! "
                           "Please provide instances of aristopy's Var class!")
    if data is None:
        return []
    elif isinstance(data, aristopy.Var):
        return [data]
    elif isinstance(data, list):
        if any(not isinstance(item, aristopy.Var) for item in data):
            raise exception
        return data
    else:
        raise exception


def check_and_convert_time_series(ensys, data):
    """
    Function to check time series data. Main tasks:
    * Compare data length with the time index of the EnergySystem instance.
      Raise an error if it is not sufficient, shorten data if it is too long.
    * Convert provided data to pandas Series if necessary and add the time
      index of the EnergySystem instance.

    :return: converted data as pandas Series
    """
    #  If length of data is not sufficient raise an error
    if len(data) < ensys.number_of_time_steps:
        raise ValueError('The length of a time series is not sufficient for the'
                         ' number of time steps of the energy system model.\n'
                         'Time steps energy system model: {}\nLength data: {}\n'
                         'Data: {}'.format(ensys.number_of_time_steps,
                                           len(data), data))

    # Convert to pandas Series if data is in a dictionary (cannot slice dicts)
    if isinstance(data, dict):
        data = pd.Series(data)

    # If length of data is more than needed -> shorten the data, discard rest
    if len(data) > ensys.number_of_time_steps:
        data = data[:ensys.number_of_time_steps]

    # Lists and arrays don't have an index --> add the time index of ensys
    if isinstance(data, list) or isinstance(data, np.ndarray):
        data = pd.Series(data, index=list(range(ensys.number_of_time_steps)))

    # If data is already a pandas Series but the index does not match the time
    # index of the energy system model --> replace the index
    elif isinstance(data, pd.Series) and list(sorted(
            data.index)) != list(range(ensys.number_of_time_steps)):
        data = pd.Series(data.values, index=list(range(
            ensys.number_of_time_steps)))

    return data


def check_and_set_bool(value, name=None):
    """ Check and return boolean value (True or False) """
    param = '' if name is None else '"' + name + '" '  # ending with space!
    if not isinstance(value, bool):
        raise TypeError('Input argument {}requires a boolean value '
                        '(True or False)'.format(param))
    return value


def check_and_set_capacities(cap, cap_min, cap_max, cap_per_mod, max_mod_nbr):
    """  Check input values for capacity arguments. """
    # positive floats or integers if not None
    for val in [cap, cap_min, cap_max, cap_per_mod]:
        if val is not None:
            check_and_set_positive_number(val)
    if max_mod_nbr is not None:
        check_positive_int(max_mod_nbr, 'maximal_module_number')

    # fixed capacity dominates other arguments
    if cap is not None:
        for val in [cap_min, cap_max, cap_per_mod, max_mod_nbr]:
            if val is not None:
                warnings.warn('Fixed capacity value "capacity" is specified. '
                              'Hence, parameters "capacity_min, capacity_max, '
                              'capacity_per_module, maximal_module_number" are '
                              'not required and ignored.')
        cap_min, cap_max = cap, cap  # cap_min = cap_max = cap
        cap_per_mod, max_mod_nbr = None, None

    # if fixed capacities are used (cap_min = cap_max) set cap to the same value
    if cap_min == cap_max and cap is None:
        cap = cap_min
    # raise an error if cap_min is larger than cap_max
    if cap_min is not None and cap_max is not None and cap_min > cap_max:
        raise ValueError('The minimal capacity of a component cannot exceed its'
                         ' maximal capacity!')

    # calculate cap_max from cap_per_mod and max_mod_nbr if needed and possible
    if cap_per_mod is not None and max_mod_nbr is not None and cap_max is None:
        cap_max = cap_per_mod * max_mod_nbr
    # "maximal_module_number" is useless without "capacity_per_module" --> raise
    if cap_per_mod is None and max_mod_nbr is not None:
        raise ValueError('Please specify a value for the "capacity_per_module" '
                         'if you set a value for "maximal_module_number".')
    # Prevent invalid parameter combination and calculate max_mod_nbr if needed
    if cap_per_mod is not None and max_mod_nbr is None:
        if cap_max is None:
            raise ValueError('Cannot work with this combination of arguments. '
                             'If a value for  the "capacity_per_module" is set,'
                             ' an additional value for "maximal_module_number"'
                             ' or "capacity_max" is required.')
        else:
            max_mod_nbr = cap_max // cap_per_mod  # round down to integer!

    return cap, cap_min, cap_max, cap_per_mod, max_mod_nbr


def check_and_set_commodity_rates(comp, rate_min, rate_max, rate_fix):
    """  Check input values for commodity rate arguments and add the data to
        the parameters DataFrame of the component """
    # Show warning if rate_fix is specified and rate_min or rate_max as well.
    # --> set rate_min and rate_max to None.
    if rate_fix is not None and (rate_min or rate_max is not None):
        rate_min, rate_max = None, None
        warnings.warn('\nIf "commodity_rate_fix" is specified, the time series '
                      '"commodity_rate_min" and "commodity_rate_max" are not '
                      'required and are set to None.')

    # Check if data types are correct and add data to 'parameters' DataFrame
    def _check_and_set_rate(data, name):
        if isinstance(data, (int, float)):
            comp.add_param(name, data)
        elif isinstance(data, aristopy.Series):
            conv_data = check_and_convert_time_series(comp.ensys, data.data)
            comp.add_param(data.name, conv_data, data.weighting_factor)
            name = data.name
        elif isinstance(data, type(None)):
            name = None
        else:
            raise ValueError('Found invalid data type for commodity rate in '
                             'component "%s". Please provide float, integer, '
                             'aristopy Series, or None.' % comp.name)
        return name

    rate_min = _check_and_set_rate(rate_min, 'commodity_rate_min')
    rate_max = _check_and_set_rate(rate_max, 'commodity_rate_max')
    rate_fix = _check_and_set_rate(rate_fix, 'commodity_rate_fix')

    # Raise error if min is larger than max at a certain position in the series
    if rate_min and rate_max is not None and not (
            np.array(comp.parameters[rate_max]['full_resolution']) >=
            np.array(comp.parameters[rate_min]['full_resolution'])).all():
        raise ValueError('The series "commodity_rate_min" of component "%s" has'
                         ' at least one value that is larger than '
                         '"commodity_rate_max".' % comp.name)

    return rate_min, rate_max, rate_fix


def check_and_set_cost_and_revenues(comp, data):
    """ Function to check the input for commodity cost and revenues.
        Can handle integers, floats and aristopy's Series class instances.

        Function returns:
        * scalar as a int or float value (or None)
        * series name of time_series in the parameter DF (or None)
    """
    scalar_value, time_series_name = 0, None  # init
    if isinstance(data, (int, float)):
        check_and_set_positive_number(data)  # raise error if value not >= 0
        scalar_value = data
    # if aristopy Series is provided --> add it to the 'parameters' DataFrame
    elif isinstance(data, aristopy.Series):
        conv_data = check_and_convert_time_series(comp.ensys, data.data)
        comp.add_param(data.name, conv_data, data.weighting_factor)
        time_series_name = data.name
    else:
        raise ValueError('Found invalid data type for commodity cost or revenue'
                         ' in component "%s". Please provide float, integer, or'
                         ' aristopy Series.' % comp.name)
    return scalar_value, time_series_name


def check_and_set_flows(data):
    """
    Function to check if the provided data for inlet and outlet keywords only
    contains instances of aristopy's Flow class.

    :return: list of aristopy flow instances
    """
    exception = ValueError("Invalid data type for inlet or outlet argument! "
                           "Please provide instances of aristopy's Flow class!")
    if data is None:
        return []
    elif isinstance(data, aristopy.Flow):
        return [data]
    elif isinstance(data, list):
        flow_list = []
        for item in data:
            if isinstance(item, aristopy.Flow):
                # Loop again over list and check if multiple Flows with same
                # commodity but different variable names are provided in a comp.
                # E.g. prevent, inlet=[ar.Flow('COMM_1', 'snk_1', 'NAME_1'),
                #                      ar.Flow('COMM_1', 'snk_2', 'NAME_2')]
                for flow in flow_list:
                    if flow.commodity == item.commodity and \
                            flow.var_name != item.var_name:
                        raise ValueError('Found Flows with same commodity but '
                                         'different variable names in a comp.')
                flow_list.append(item)
            else:
                raise exception
        return flow_list
    else:
        raise exception


def check_and_set_positive_number(value, name=None):
    """ Check and return positive input value (float, int) """
    param = '' if name is None else '"'+name+'" '  # ending with space!
    if not (isinstance(value, (float, int))) or not value >= 0:
        raise TypeError('Input argument {}requires a positive float or '
                        'integer!'.format(param))
    return value


def check_and_set_range_zero_one(value, name=None):
    """  Check the return positive input value less or equal one. """
    check_and_set_positive_number(value, name)  # type float or int and val >= 0
    param = '' if name is None else '"'+name+'" '  # ending with space!
    if not value <= 1:
        raise ValueError('Maximal value for argument {}is "1".'.format(param))
    return value


def check_and_set_time_series_data(data):
    """
    Function to check if the provided content of the time_series_data argument
    only contains instances of aristopy's Series class.

    :return: list of aristopy Series instances
    """
    exception = ValueError("Invalid data type for time_series_data argument! "
                           "Please provide instances of aristopy Series class!")
    if data is None:
        return []
    elif isinstance(data, aristopy.Series):
        return [data]
    elif isinstance(data, list):
        if any(not isinstance(item, aristopy.Series) for item in data):
            raise exception
        return data
    else:
        raise exception


def check_and_set_user_expr(data):
    """
    Function to check if the provided content of the user_expressions argument
    contains a string, or a list of strings, or None.

    :return: list (of strings)
    """
    exception = ValueError("Invalid data type for user_expressions argument! "
                           "Please provide (list of) strings, or None!")
    if data is None:
        return []
    elif isinstance(data, str):
        return [data]
    elif isinstance(data, list):
        if any(not isinstance(item, str) for item in data):
            raise exception
        return data
    else:
        raise exception


def check_cluster_input(number_of_typical_periods,
                        number_of_time_steps_per_period, number_of_time_steps):
    """ Check correctness of input arguments to the 'cluster' method """
    check_positive_int(
        number_of_typical_periods, 'number_of_typical_periods')
    check_positive_int(
        number_of_time_steps_per_period, 'number_of_time_steps_per_period')
    if not number_of_time_steps % number_of_time_steps_per_period == 0:

        raise ValueError('The total number of model time steps is "%d". This '
                         'number cannot be divided into an integer number of '
                         'periods of length "%d" without a remainder. Consider '
                         'changing parameter "number_of_time_steps_per_period".'
                         % (number_of_time_steps,
                            number_of_time_steps_per_period))
    t_product = number_of_typical_periods * number_of_time_steps_per_period
    if number_of_time_steps < t_product:
        raise ValueError('The total number of model time steps is "%d". The '
                         'product of the requested "number_of_typical_periods" '
                         'and "number_of_time_steps_per_period" is "%d" which '
                         'is out of range.' % (number_of_time_steps, t_product))


def check_declare_model_input(use_clustered_data, is_data_clustered,
                              declare_persistent, persistent_solver):
    """ Check correctness of input arguments to the 'declare_model' method """
    check_and_set_bool(use_clustered_data, 'use_clustered_data')
    if use_clustered_data and not is_data_clustered:
        raise ValueError('Model declaration with clustered time series data is'
                         ' requested ("use_clustered_data"=True), but the model'
                         ' flag states that clustered data is not available. \n'
                         'Please call the method "cluster" first and call '
                         '"declare_model" or "optimize" afterward.')
    check_and_set_bool(declare_persistent, 'declare_persistent')
    if persistent_solver not in ['gurobi_persistent', 'cplex_persistent']:
        raise ValueError('Valid solvers for the persistent interface are '
                         '"gurobi_persistent" and "cplex_persistent"!')


def check_edit_var_input(name, store_vars, **kwargs):
    """ Check input to the Component method 'edit_variable' """
    assert isinstance(name, str), 'The variable name should be a string!'
    check_and_set_bool(store_vars, 'store_previous_variables')
    for key, val in kwargs.items():
        if key not in ['ub', 'lb', 'domain', 'has_time_set', 'init']:
            warnings.warn('Keyword argument "{}" not recognized. Options '
                          'for variable editing are: "domain", "ub", "lb", '
                          '"has_time_set", "init".'.format(key))
        if (key == 'ub' or key == 'lb') and val is not None:
            assert isinstance(val, (float, int)), 'Input requires float or int!'
        if key == 'domain' and val not in [
                'Binary', 'NonNegativeReals', 'Reals']:
            raise TypeError('Select the domain of the variable from '
                            '"Binary", "NonNegativeReals", "Reals"')
        if key == 'has_time_set':
            check_and_set_bool(val, 'has_time_set')


def check_energy_system_input(number_of_time_steps, hours_per_time_step,
                              interest_rate, economic_lifetime, logging):
    """ Check input to initialization method of the EnergySystem class """
    check_positive_int(number_of_time_steps, 'number_of_time_steps')
    check_positive_int(hours_per_time_step, 'hours_per_time_step')
    check_positive_int(economic_lifetime, 'economic_lifetime')
    check_and_set_positive_number(interest_rate, 'interest_rate')
    if logging is not None and not isinstance(logging, aristopy.Logger):
        raise TypeError('"logging" only takes instances of the class "Logger"')


def check_logger_input(logfile, delete_old, default_handler, local_handler,
                       default_level, local_level, screen_to_log):
    """ Check input to initialization method of the Logger class """
    assert isinstance(logfile, str), 'The logfile name should be a string!'
    check_and_set_bool(delete_old, 'delete_old_logs')
    check_and_set_bool(screen_to_log, 'write_screen_output_to_logfile')
    if default_handler not in ['file', 'stream']:
        raise ValueError('Input argument should be "file" or "stream".')
    if default_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        raise ValueError('Options for logger levels are: "DEBUG", "INFO", '
                         '"WARNING", "ERROR", "CRITICAL"')
    for key, val in local_handler.items():
        if key not in ['EnergySystem', 'Source', 'Sink', 'Conversion',
                       'Storage', 'Bus']:
            raise ValueError('Keys for dictionary "local_log_handler" are: '
                             '"EnergySystem", "Source", "Sink", '
                             '"Conversion", "Storage", "Bus"')
        if val not in ['file', 'stream']:
            raise ValueError('Input argument should be "file" or "stream".')
    for key, val in local_level.items():
        if key not in ['EnergySystem', 'Source', 'Sink', 'Conversion',
                       'Storage', 'Bus']:
            raise ValueError('Keys for dictionary "local_log_level" are: '
                             '"EnergySystem", "Source", "Sink", '
                             '"Conversion", "Storage", "Bus"')
        if val not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError('Options for logger levels are: "DEBUG", "INFO", '
                             '"WARNING", "ERROR", "CRITICAL"')


def check_positive_int(value, name=None, zero_is_invalid=True):
    """ Check that the input value is a positive (and non-zero) integer. """
    param = '' if name is None else '"'+name+'" '  # ending with space!
    if zero_is_invalid and (not isinstance(value, int) or not value > 0):
        raise ValueError('Input argument {}requires a positive and non-zero '
                         'integer.'.format(param))
    elif not zero_is_invalid and (not isinstance(value, int) or not value >= 0):
        raise ValueError('Input argument {}requires a positive integer (>=0).'
                         .format(param))


def check_optimize_input(use_clustered_data, declare_persistent,
                         persistent_solver, is_data_clustered,
                         solver, time_limit, optimization_specs):
    """ Check correctness of input arguments to the 'optimize' method """
    check_declare_model_input(use_clustered_data, is_data_clustered,
                              declare_persistent, persistent_solver)
    assert isinstance(solver, str), 'The "solver" parameter should be a string!'
    if time_limit is not None:
        check_positive_int(time_limit, 'time_limit')
    assert isinstance(optimization_specs, str), \
        'The "optimization_specs" parameter should be a string!'


def check_plot_operation_input(data, comp, commod, scale, single_period, level,
                               show_plot, save_plot, file_name):
    """ Function to check the input for Plotter method 'plot_operation' """
    if comp not in data['components']:
        raise Exception('Component not found in JSON-File!')
    if commod not in data['components'][comp]['commodities']:
        raise Exception('Could not find commodity "{}" in component {}!'
                        .format(commod, comp))
    if single_period is not None and single_period not in list(range(
            data['number_of_typical_periods'])):
        raise Exception('Period index for plotting is out of range!')
    if level != 1 and level != 2:
        raise Exception('Level of detail can take values 1 or 2!')
    assert isinstance(file_name, str), 'The "file_name" should be a string!'
    check_and_set_bool(scale, 'scale_to_hourly_resolution')
    check_and_set_bool(show_plot, 'show_plot')
    check_and_set_bool(save_plot, 'save_plot')


def check_scalar_params_dict(data):
    """ Function to check the input of the "scalar_params" argument. """
    if not isinstance(data, dict):
        raise ValueError('"scalar_params" need to be provided as a dictionary!')
    for key, val in data.items():
        if not isinstance(key, str):
            raise TypeError('Check "scalar_params". Keys need to be strings!')
        assert isinstance(val, (float, int)), 'Input requires float or int!'


def io_error_message(class_name, comp_name, io_name):
    """ Error message to return for missing inlet / outlet specifications """
    return ValueError('%s "%s" requires at least on %s Flow (not None!). Please'
                      ' check your specifications for keyword argument %s.'
                      % (class_name, comp_name, io_name, io_name))


# ==============================================================================
#    U S E R   C O N S T R A I N T S
# ==============================================================================
def disassemble_user_expression(expr):
    """ Disassemble a user-specified expression in its pieces. """
    expr = expr.replace(' ', '')  # remove all spaces
    # List of reserved chars that are used for string splitting:
    reserved_chars_1 = ['*', '/', '+', '-', '(', ')']
    reserved_chars_2 = ['==', '>=', '<=', '**']
    reserved_chars_3 = ['sum', 'sin', 'cos', 'exp', 'log']
    # Do the string splitting. Exemplary output:
    # ['Q', '==', 'xx', '*', '(', 'F', '/', '0.3', ')', '**', '3']
    expr_part = []
    while len(expr) > 0:
        store_string = ''
        for i in range(len(expr)):
            if expr[i:i + 3] in reserved_chars_3:  # three signs in a row
                if store_string:  # is not empty
                    expr_part.append(store_string)
                expr_part.append(expr[i:i + 3])  # reserved chars
                expr = expr[len(store_string) + 3:len(expr)]
                break
            elif expr[i:i + 2] in reserved_chars_2:  # two signs in a row
                if store_string:  # is not empty
                    expr_part.append(store_string)
                expr_part.append(expr[i:i + 2])  # reserved chars
                expr = expr[len(store_string) + 2:len(expr)]
                break
            elif expr[i] in reserved_chars_1:  # one sign
                if store_string:  # is not empty
                    expr_part.append(store_string)
                expr_part.append(expr[i:i + 1])
                expr = expr[len(store_string) + 1:len(expr)]
                break
            else:
                store_string += expr[i]
                # If string is in last position of expression
                if len(expr) == len(store_string):
                    expr_part.append(store_string)
                    expr = expr[len(store_string):len(expr)]  # -> expr = ''
    return expr_part


def simplify_user_constraint(df, time_step_weights):
    """
    Simplify a user-defined constraint in multiple iterations until the provided
    DataFrame holds only 3 columns (left hand side, le/eq/ge, right hand side).
    ==> The data from the simplified DataFrame is returned as python dicts for
        "lhs" and "rhs" and a string for the operation sign ('==', '>=', '<=').

    Exemplary string representation of the original user_expression:
    '-Q == (1 + F * 0.8) ** 3'

    Stepwise simplification process of the original DataFrame:
         0            1   2  3    4  5            6  7    8  9   10   11
    0 0  -  comp.Q[0,0]  ==  (  1.0  +  comp.F[0,0]  *  0.8  )  **  3.0
      1  -  comp.Q[0,1]  ==  (  1.0  +  comp.F[0,1]  *  0.8  )  **  3.0
      2  -  comp.Q[0,2]  ==  (  1.0  +  comp.F[0,2]  *  0.8  )  **  3.0
      [...]
         0            1   2  3    4  5                6  7   8    9
    0 0  -  comp.Q[0,0]  ==  (  1.0  +  0.8*comp.F[0,0]  )  **  3.0
      1  -  comp.Q[0,1]  ==  (  1.0  +  0.8*comp.F[0,1]  )  **  3.0
      2  -  comp.Q[0,2]  ==  (  1.0  +  0.8*comp.F[0,2]  )  **  3.0
      [...]
         0            1   2  3                      4  5   6    7
    0 0  -  comp.Q[0,0]  ==  (  1.0 + 0.8*comp.F[0,0]  )  **  3.0
      1  -  comp.Q[0,1]  ==  (  1.0 + 0.8*comp.F[0,1]  )  **  3.0
      2  -  comp.Q[0,2]  ==  (  1.0 + 0.8*comp.F[0,2]  )  **  3.0
      [...]
         0            1   2                      3   4    5
    0 0  -  comp.Q[0,0]  ==  1.0 + 0.8*comp.F[0,0]  **  3.0
      1  -  comp.Q[0,1]  ==  1.0 + 0.8*comp.F[0,1]  **  3.0
      2  -  comp.Q[0,2]  ==  1.0 + 0.8*comp.F[0,2]  **  3.0
      [...]
         0            1   2                             3
    0 0  -  comp.Q[0,0]  ==  (1.0 + 0.8*comp.F[0,0])**3.0
      1  -  comp.Q[0,1]  ==  (1.0 + 0.8*comp.F[0,1])**3.0
      2  -  comp.Q[0,2]  ==  (1.0 + 0.8*comp.F[0,2])**3.0
      [...]
                     0   1                             2
    0 0  - comp.Q[0,0]  ==  (1.0 + 0.8*comp.F[0,0])**3.0
      1  - comp.Q[0,1]  ==  (1.0 + 0.8*comp.F[0,1])**3.0
      2  - comp.Q[0,2]  ==  (1.0 + 0.8*comp.F[0,2])**3.0
      [...]
    """
    debug = False  # prints simplification progress on the console if True

    # Simplify the DataFrame until it has only 3 columns (LHS <==> RHS)
    while_iter_count = 0  # init counter
    found_sum_operator = False  # init
    dropped_index = False  # init

    while len(df.columns) > 3 and while_iter_count < 100:
        # Increase counter by 1 for each iteration in the while-loop
        while_iter_count += 1
        # Flag to check if mathematical operation has been performed
        # -> if True, jump to beginning of the while-loop (continue)
        operation_done = False  # init
        found_minus_sign = False  # init
        df.columns = range(len(df.columns))  # rename the columns
        first_row = df.iloc[0]  # list of first row entries
        if debug: print(df.head(3).to_string())
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 1. Look for a part of the expression in brackets
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop first_row from left, look for first ')' and remember last '('
        last_opening_idx = 0  # init
        first_closing_idx = len(first_row)  # init
        for idx, obj in first_row.items():
            if isinstance(obj, str) and obj == '(':
                last_opening_idx = idx
            if isinstance(obj, str) and obj == ')':
                first_closing_idx = idx
                break
        # Get slice of 'first_row' (in brackets) or whole row if no brackets
        first_row_slice = first_row[last_opening_idx:first_closing_idx + 1]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 2. Look for exponents in the slice
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Look for power symbol ('**') in expression part
        for idx, obj in first_row_slice.items():
            if isinstance(obj, str) and obj == '**':
                # Find names of surrounding elements of '**' operator
                lhs_idx = idx - 1
                rhs_idx = idx + 1
                rhs_obj = first_row_slice[rhs_idx]
                # If rhs is a minus sign, next element is considered
                if isinstance(rhs_obj, str) and rhs_obj == '-':
                    rhs_idx = idx + 2
                    found_minus_sign = True
                # Do the power operation, depending on the flag
                # 'found_minus_sign' and delete surrounding columns
                if found_minus_sign:
                    df[idx] = df[lhs_idx] ** (-1 * df[rhs_idx])
                    df.drop([idx + 2, idx + 1, idx - 1], axis=1, inplace=True)
                else:
                    df[idx] = df[lhs_idx] ** df[rhs_idx]
                    df.drop([idx + 1, idx - 1], axis=1, inplace=True)
                # Set flag 'operation_done' to True to continue while
                operation_done = True
                break  # the for-loop
        # Jump to the beginning of while-loop if flag is True
        if operation_done:
            continue
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 3. Look for division and multiplication
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Look for division or multiplication symbols in expression part
        for idx, obj in first_row_slice.items():
            if isinstance(obj, str) and (obj == '/' or obj == '*'):
                # Find names of surrounding elements of '/' or '*' operator
                lhs_idx = idx - 1
                rhs_idx = idx + 1
                rhs_obj = first_row_slice[rhs_idx]
                # If rhs is a minus sign, next element is considered
                if isinstance(rhs_obj, str) and rhs_obj == '-':
                    rhs_idx = idx + 2
                    found_minus_sign = True
                # Do the operation, depending on the flag 'found_minus_sign'
                # and the found operator and delete surrounding columns
                if found_minus_sign:
                    if obj == '/':
                        df[idx] = df[lhs_idx] / (-1 * df[rhs_idx])
                    else:  # obj == '*'
                        df[idx] = df[lhs_idx] * (-1 * df[rhs_idx])
                    df.drop([idx + 2, idx + 1, idx - 1], axis=1, inplace=True)
                else:
                    if obj == '/':
                        df[idx] = df[lhs_idx] / df[rhs_idx]
                    else:  # obj == '*'
                        df[idx] = df[lhs_idx] * df[rhs_idx]
                    df.drop([idx + 1, idx - 1], axis=1, inplace=True)
                # Set flag 'operation_done' to True to continue while
                operation_done = True
                break  # the for-loop
        # Jump to the beginning of while-loop if flag is True
        if operation_done:
            continue
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 4. Look for summation and subtraction
        #    (--> Combinations: -+ or ++ are not allowed!)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Look for summation and subtraction symbols in expression part
        for idx, obj in first_row_slice.items():
            if isinstance(obj, str) and (obj == '+' or obj == '-'):
                # ------------------------------------------------------
                # Special case I: Plus or minus operator in first position
                if idx == 0:
                    if obj == '-':
                        df[1] = -1 * df[1]
                    # else --> obj == '+' --> pass
                    df.drop([0], axis=1, inplace=True)  # delete first col
                    operation_done = True
                    break  # the for-loop
                # ------------------------------------------------------
                # Find names of surrounding elements of '+' or '-' operator
                lhs_idx = idx - 1
                lhs_obj = first_row_slice[lhs_idx]
                rhs_idx = idx + 1
                rhs_obj = first_row_slice[rhs_idx]
                # ------------------------------------------------------
                # Special case II: Plus or minus operator is right beside an
                # opening bracket or an (non)-equality sign.
                if isinstance(lhs_obj, str) and (
                        lhs_obj == '(' or lhs_obj == '==' or
                        lhs_obj == '>=' or lhs_obj == '<='):
                    if obj == '-':
                        df[rhs_idx] = -1 * df[rhs_idx]
                    # else --> obj == '+' --> pass
                    df.drop([idx], axis=1, inplace=True)
                    operation_done = True
                    break  # the for-loop
                # ------------------------------------------------------
                # If rhs is a minus sign, next element is considered
                if isinstance(rhs_obj, str) and rhs_obj == '-':
                    rhs_idx = idx + 2
                    found_minus_sign = True
                # Do the operation, depending on the flag 'found_minus_sign'
                # and the found operator and delete surrounding columns
                if found_minus_sign:
                    if obj == '+':
                        df[idx] = df[lhs_idx] + (-1 * df[rhs_idx])
                    else:  # obj == '-'
                        df[idx] = df[lhs_idx] - (-1 * df[rhs_idx])
                    df.drop([idx + 2, idx + 1, idx - 1], axis=1, inplace=True)
                else:
                    if obj == '+':
                        df[idx] = df[lhs_idx] + df[rhs_idx]
                    else:  # obj == '-'
                        df[idx] = df[lhs_idx] - df[rhs_idx]
                    df.drop([idx + 1, idx - 1], axis=1, inplace=True)
                # Set flag 'operation_done' to True to continue while
                operation_done = True
                break  # the for-loop
        # Jump to the beginning of while-loop if flag is True
        if operation_done:
            continue
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 5. Delete enclosing brackets
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # This point it reached, if an expression slice does not contain
        # any more possibilities to perform the mathematical operations
        # 'pow', 'div', 'mul', 'add', 'sub'. Check if the term is still
        # enclosed by brackets --> remove them.
        idx = first_row_slice.keys()

        if first_row[idx[0]] == '(' and first_row[idx[-1]] == ')':

            # Special case for operators: SUM, SIN, COS, EXP, LOG.
            # These operators are always on the left side of an opening bracket.
            # Check if there is one of these operators in front of the brackets:
            if idx[0] != 0 and isinstance(first_row[idx[0]-1], str) and \
                    first_row[idx[0]-1] in ['sum', 'sin', 'cos', 'exp', 'log']:
                # Get the name of the operator
                operator_left_of_bracket = first_row[idx[0]-1]

                # Use simple python sum function if operator is 'sum':
                if operator_left_of_bracket == 'sum':
                    # Only do 'sum' if it is not a scalar value (all the same).
                    # Compare the expression string representations of the first
                    # two expression rows (after some string modifications).
                    exception = 'The data in the brackets of "sum"-operator ' \
                                'is scalar (constant or not time-dependent). ' \
                                '"sum"-operator is useless in this case. ' \
                                'Consider rewriting the user_expression.'
                    all_rows = df[idx[1]].to_list()
                    if len(all_rows) >= 2:
                        row_0 = _expr_string_adjustment(all_rows[0].__str__())
                        row_1 = _expr_string_adjustment(all_rows[1].__str__())
                        if row_0 == row_1:
                            raise ValueError(exception)
                    # Don't allow sum for time-independent constraints. Actually
                    # it would work, but it doesn't make sense [CAP == sum(42)].
                    elif len(all_rows) == 1:
                        raise ValueError(exception)
                    # Do sum operation if exception is not raised:
                    # Also consider the weight of each time-step (only important
                    # if the data is clustered).
                    df[idx[1]] = sum(weight * row_data for weight, row_data
                                     in zip(time_step_weights, all_rows))
                    found_sum_operator = True  # set flag for post-processing

                # Do the math with Pyomo's intrinsic functions for sin, log, ...
                elif operator_left_of_bracket == 'sin':
                    df[idx[1]] = [pyo.sin(k) for k in df[idx[1]].to_list()]
                elif operator_left_of_bracket == 'cos':
                    df[idx[1]] = [pyo.cos(k) for k in df[idx[1]].to_list()]
                elif operator_left_of_bracket == 'exp':
                    df[idx[1]] = [pyo.exp(k) for k in df[idx[1]].to_list()]
                elif operator_left_of_bracket == 'log':
                    df[idx[1]] = [pyo.log(k) for k in df[idx[1]].to_list()]
                # If there was an operator in front of the brackets --> drop it
                df.drop([idx[0]-1], axis=1, inplace=True)

            # Now drop the surrounding brackets
            df.drop([idx[0], idx[-1]], axis=1, inplace=True)
            # Set flag 'operation_done' to True to continue while
            operation_done = True
        # Jump to the beginning of while-loop if flag is True
        if operation_done:
            continue

    # Outside of while-loop:
    # ~~~~~~~~~~~~~~~~~~~~~~
    # Raise an error if too many iterations are performed on one constraint!
    if while_iter_count >= 100:
        raise ValueError('The function "simplify_user_constraint" needed too '
                         'many iterations. The simplification process stopped '
                         'before the constraint could be simplified to the '
                         'desired form "LHS <==> RHS". Check your constraints!')

    # Final renaming of last three columns to [0, 1, 2]
    df.columns = range(len(df.columns))
    if debug: print(df.head(3).to_string())
    if debug: print('Number of while-loops:', str(while_iter_count))

    # Special case: If there was at least one 'sum' operation, it is possible
    # that the constraint is not time-dependent anymore. Assume 'Q' is an time-
    # dependent variable. We will loose time-dependency in case A and not in
    # case B: A) 'CAP == sum(Q)';  B) 'CAP == Q + sum(Q)'.
    # Expression string representations are compared to check if the expressions
    # are the same (first two rows are enough). Note: The strings can have the
    # same content but different order and additional brackets sometimes
    # => adjust expression string representation to enable comparison.
    if found_sum_operator and len(df[0]) >= 2:
        lhs_row_0 = _expr_string_adjustment(df[0].iloc[0].__str__())
        lhs_row_1 = _expr_string_adjustment(df[0].iloc[1].__str__())
        rhs_row_0 = _expr_string_adjustment(df[2].iloc[0].__str__())
        rhs_row_1 = _expr_string_adjustment(df[2].iloc[1].__str__())
        if lhs_row_0 == lhs_row_1 and rhs_row_0 == rhs_row_1:
            df.drop(df.index[1:], axis=0, inplace=True)
            dropped_index = True
            if debug: print(df.head(3).to_string())

    # Convert the first and the third column to python dictionaries and get the
    # global operation sign from the first row of the middle column.
    lhs = df[0].to_dict()
    op = df[1].iloc[0]  # done only once --> it's the same operator in all rows
    rhs = df[2].to_dict()

    # Return the left and right hand side dicts and the operator sign
    return lhs, op, rhs, dropped_index


def _expr_string_adjustment(expr):
    # Remove all (necessary and unnecessary) opening and closing brackets
    # (sometimes pandas adds them at strange positions in a few rows), split the
    # expressions at spaces (they are between operation signs and all other
    # objects, except brackets and indices), and sort the string alphabetically.
    expr = expr.replace('(', '').replace(')', '')
    expr = expr.split()
    expr.sort()
    return expr
# ==============================================================================
