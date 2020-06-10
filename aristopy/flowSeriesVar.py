#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#    F L O W ,   S E R I E S ,   V A R
# ==============================================================================
"""
* File name: flowSeriesVar.py
* Last edited: 2020-06-14
* Created by: Stefan Bruche (TU Berlin)

The instances of the Flow class are required to add commodities and the
associated variables to the modeled components and to create the component
interconnections. The Series and Var classes are used to introduce time-series
data and additional variables to the pyomo model instance.
"""
import pandas as pd
import numpy as np


# ******************************************************************************
#    T H E   F L O W   C L A S S
# ******************************************************************************
class Flow:
    def __init__(self, commodity, link=None, var_name='commodity_name',
                 **kwargs):
        """
        A Flow represents a connection point of the inlet or the outlet of a
        component. The Flow has a single commodity that is entering or leaving
        the component. Additionally, information on the variable name of the
        commodity inside the component can be provided (default: use the name of
        the commodity) and the linking component (source or destination of the
        Flow) can be stated.

        :param commodity: Name (identifier) of the commodity
        :type commodity: str

        :param link: Component name of the connected Flow source or destination
            needs to be stated on one or both sides of the Flow
            |br| *Default: None*
        :type link: str, or None

        :param var_name: Variable name of the Flow commodity as used inside the
            component, e.g. for scripting of user expressions.
            |br| *Default: 'commodity_name' => i.e. use name of the commodity*
        :type var_name: str
        """

        # Check input argument types
        assert isinstance(commodity, str), '"commodity" should be a string'
        assert isinstance(link, (str, type(None))), '"link" should be a string'
        assert isinstance(var_name, str), '"var_name" should be a string'

        self.commodity = commodity
        self.link = link
        self.var_name = commodity if var_name == 'commodity_name' else var_name

        # Add and process possible kwargs here (if any); not applied right now
        for key, val in kwargs.items():
            if key not in []:
                raise ValueError('Unknown keyword argument "%s" provided' % key)

        # Right now it is only possible to have one commodity per Flow!
        # Might be extended to allow lists of commodities, var_names, kwargs if
        # they are attached to one flow (like pipe with mass flow and enthalpy)


# ******************************************************************************
#    T H E   S E R I E S   C L A S S
# ******************************************************************************
class Series:
    def __init__(self, name, data, weighting_factor=1.0):
        """
        The Series class is used to add time series data to the model.
        Time series data can be required in the Source class to implement data
        for keyword arguments 'commodity_rate_min',´'commodity_rate_max',
        'commodity_rate_fix'. Additionally, they might be needed to a add time-
        dependent commodity cost or revenues, or generally for scripting of
        user expressions (added via 'time_series_data' argument).

        :param name: Name (identifier) of the time series data instance. Can be
            used for scripting of user expressions.
        :type name: str

        :param data: Time series data
        :type data: list, dict, numpy array, pandas Series

        :param weighting_factor: Weighting factor to use for the clustering
            |br| *Default: 1.0*
        :param weighting_factor: float or int
        """

        # Check input arguments:
        assert isinstance(name, str), '"name" should be a string'
        assert isinstance(data, (list, dict, np.ndarray, pd.Series)), \
            'Expected "data" with type list, dict, numpy array or pandas Series'
        assert isinstance(weighting_factor, (int, float)), \
            'Expected "weighting_factor" as integer or float'
        assert 0 <= weighting_factor <= 1, \
            'Expected weighting factor value between 0 and 1'

        self.name = name
        self.data = data
        self.weighting_factor = weighting_factor


# ******************************************************************************
#    T H E   V A R   ( V A R I A B L E )   C L A S S
# ******************************************************************************
class Var:
    def __init__(self, name, domain='NonNegativeReals', has_time_set=True,
                 alternative_set=None, ub=None, lb=None, init=None):
        """
        Class to manually add pyomo variables to a component (via argument
        "additional_vars"), or the main model container (ConcreteModel: model)
        of the EnergySystem instance (via function "add_variable").

        :param name: Name (identifier) of the added variable
        :type name: str

        :param domain: A super-set of the values the variable can take on.
            Possible values are: 'Reals', 'NonNegativeReals', 'Binary'.
            |br| *Default: 'NonNegativeReals'*
        :type domain: str

        :param has_time_set: Is True if the time set of the EnergySystem
            instance is also a set of the added variable.
            |br| *Default: True*
        :type has_time_set: bool

        :param alternative_set: Alternative variable sets can be added here via
            iterable Python objects (e.g. list)
            |br| *Default: None*

        :param ub: Upper variable bound.
            |br| *Default: None*
        :type ub: float or int

        :param lb: Lower variable bound.
            |br| *Default: None*
        :type lb: float or int

        :param init: A function or Python object that provides starting values
            for the added variable.
            |br| *Default: None*
        """

        # Check input arguments:
        assert isinstance(name, str), '"name" should be a string'
        assert domain in ['Binary', 'NonNegativeReals', 'Reals'], \
            'Select variable domain from "Binary", "NonNegativeReals", "Reals"'
        assert isinstance(has_time_set, bool), 'Expected "has_time_set as bool'
        assert isinstance(ub, (int, float, type(None))), 'Expected "ub" as ' \
                                                         'numeric value or None'
        assert isinstance(lb, (int, float, type(None))), 'Expected "lb" as ' \
                                                         'numeric value or None'
        if alternative_set is not None and not hasattr(alternative_set,
                                                       '__iter__'):
            raise TypeError('"alternative_set" requires an iterable object!')

        self.name = name
        self.domain = domain
        self.has_time_set = has_time_set
        self.alternative_set = alternative_set
        self.ub = ub
        self.lb = lb
        self.init = init  # no input checks applied for initialization function


if __name__ == '__main__':
    flow = Flow('HEAT')
    series = Series(name='my_data', data=[1, 2, 3], weighting_factor=0.8)
    var = Var('TEST_VAR', ub=42)
