#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
** The Flow class **

* Last edited: 2020-06-01
* Created by: Stefan Bruche (TU Berlin)
"""


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

        :param commodity: Name of the commodity (string)
        :param link: Component name of the connected Flow source or destination
            needs to be stated on one or both sides of the Flow
            (string, default: None)
        :param var_name: Variable name of the Flow commodity as used inside the
            component, e.g. for scripting of user expressions.
            (string, default: 'commodity_name', i.e. use name of the commodity)
        """

        # Check correctness of input types
        assert isinstance(commodity, str), 'Expected "commodity" string'
        assert isinstance(link, (str, type(None))), 'Expected "link" string'
        assert isinstance(var_name, str), 'Expected "var_name" string'

        self.commodity = commodity
        self.link = link
        self.var_name = commodity if var_name == 'commodity_name' else var_name

        # Add and process possible kwargs here (if any)
        for key, val in kwargs.items():
            if key not in []:
                raise ValueError('Unknown keyword argument "%s" provided' % key)

        # Right now it is only possible to have one commodity per Flow!
        # Could be extended to allow lists of commodities, var_names, kwargs if
        # they are attached to one flow (like pipe with mass flow and enthalpy)


if __name__ == '__main__':
    flow = Flow('HEAT')
