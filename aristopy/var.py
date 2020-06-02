#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
** The Var (variable) class **

* Last edited: 2020-06-01
* Created by: Stefan Bruche (TU Berlin)
"""


class Var:
    def __init__(self, name, domain='NonNegativeReals', has_time_set=True,
                 alternative_set=None, ub=None, lb=None, init=None):
        """
        Class to manually add pyomo variables to a component (via argument
        "additional_vars"), or the main model container (ConcreteModel: model)
        of the EnergySystem instance (via function "add_variable").

        :param name: Name (identifier) of the added variable
        :type name: string

        :param domain: A super-set of the values the variable can take on.
            Possible values are: 'Reals', 'NonNegativeReals', 'Binary'.
            |br| * Default: 'NonNegativeReals'
        :type domain: string

        :param has_time_set: Is True if the time set of the energy system model
            is also a set of the added variable.
            |br| * Default: True
        :type has_time_set: boolean

        :param alternative_set: Alternative variable sets can be added here via
            iterable Python objects (e.g. list)

        :param ub: Upper variable bound.
        :type ub: Number (integer or float)

        :param lb: Lower variable bound.
        :type lb: Number (integer or float)

        :param init: A function or Python object that provides starting values
            for the added variable.
        """

        # Check correctness of input types
        assert isinstance(name, str), 'Expected variable "name" as a string'
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

    var = Var('TEST_VAR', ub=42)
