#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#    B U S
# ==============================================================================
"""
* File name: bus.py
* Last edited: 2020-06-14
* Created by: Stefan Bruche (TU Berlin)

A Bus component collects and transfers a commodity. Bus components can also
be used to model transmission lines between different sites.
"""
import pyomo.environ as pyomo
from aristopy import utils
from aristopy.component import Component


class Bus(Component):
    def __init__(self, ensys, name, inlet, outlet,
                 basic_variable='inlet_variable',
                 has_existence_binary_var=None,
                 time_series_data=None, scalar_params=None,
                 additional_vars=None, user_expressions=None,
                 capacity=None, capacity_min=None, capacity_max=None,
                 capex_per_capacity=0, capex_if_exist=0,
                 opex_per_capacity=0, opex_if_exist=0, opex_operation=0,
                 losses=0
                 ):
        """
        Initialize an instance of the Bus class.

        .. note::
           See the documentation of the :class:`Component
           <aristopy.component.Component>` class for a description of all
           keyword arguments and inherited methods.

        :param losses: Factor to specify the relative loss of the transported
            commodity between inlet and outlet per hour of operation
            (0 => no loss; 1 => 100% loss).
            |br| *Default: 0*
        :type losses: float or int (0<=value<=1)
        """

        # Prevent None at inlet & outlet! (Flows are checked in Component init)
        if inlet is None:
            raise utils.io_error_message('Bus', name, 'inlet')
        if outlet is None:
            raise utils.io_error_message('Bus', name, 'outlet')

        Component.__init__(self, ensys=ensys, name=name,
                           inlet=inlet, outlet=outlet,
                           basic_variable=basic_variable,
                           has_existence_binary_var=has_existence_binary_var,
                           time_series_data=time_series_data,
                           scalar_params=scalar_params,
                           additional_vars=additional_vars,
                           user_expressions=user_expressions,
                           capacity=capacity, capacity_min=capacity_min,
                           capacity_max=capacity_max,
                           capex_per_capacity=capex_per_capacity,
                           capex_if_exist=capex_if_exist,
                           opex_per_capacity=opex_per_capacity,
                           opex_if_exist=opex_if_exist,
                           opex_operation=opex_operation
                           )

        # Check and set additional input arguments
        self.losses = utils.check_and_set_range_zero_one(losses, 'losses')

        # Store the names for the loading and unloading variables
        self.inlet_variable = self.inlet[0].var_name
        self.outlet_variable = self.outlet[0].var_name

        # Last step: Add the component to the EnergySystem instance
        self.add_to_energy_system(ensys, name)

    def __repr__(self):
        return '<Bus: "%s">' % self.name

    # ==========================================================================
    #    C O N V E N T I O N A L   C O N S T R A I N T   D E C L A R A T I O N
    # ==========================================================================
    def declare_component_constraints(self, ensys, model):
        """
        Method to declare all component constraints.

        The following constraint methods are inherited from the Component class
        and are not documented in this sub-class:

        * :meth:`con_couple_bi_ex_and_cap
          <aristopy.component.Component.con_couple_bi_ex_and_cap>`
        * :meth:`con_cap_min <aristopy.component.Component.con_cap_min>`

        *Method is not intended for public access!*

        :param ensys: Instance of the EnergySystem class
        :param model: Pyomo ConcreteModel of the EnergySystem instance
        """
        # Time-independent constraints :
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.con_couple_bi_ex_and_cap()
        self.con_cap_min()

        # Time-dependent constraints :
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.con_operation_limit(model)
        self.con_bus_balance(model)

    # **************************************************************************
    #    Time-dependent constraints
    # **************************************************************************
    def con_operation_limit(self, model):
        """
        The basic variable of a component is limited by its nominal capacity.
        This means, the operation (commodity at inlet or outlet) of a bus
        (MWh) is limited by its nominal power (MW) multiplied with the number of
        hours per time step. E.g.: |br|
        ``Q_IN[p, t] <= CAP * dt``

        *Method is not intended for public access!*
        """
        # Only required if component has a capacity variable
        if self.has_capacity_var:
            cap = self.variables[utils.CAP]['pyomo']
            basic_var = self.variables[self.basic_variable]['pyomo']
            dt = self.ensys.hours_per_time_step

            def con_operation_limit(m, p, t):
                return basic_var[p, t] <= cap * dt

            setattr(self.block, 'con_operation_limit', pyomo.Constraint(
                model.time_set, rule=con_operation_limit))

    def con_bus_balance(self, model):
        """
        The quantity of the commodity at the outlet must equal the quantity at
        the inlet minus the the transmission loss share. A bus component cannot
        store a commodity (correction with "hours_per_time_step" not needed).
        E.g.: |br| ``Q_OUT[p, t] == Q_IN[p, t] * (1 - losses)``

        *Method is not intended for public access!*
        """
        inlet_var = self.variables[self.inlet_variable]['pyomo']
        outlet_var = self.variables[self.outlet_variable]['pyomo']

        def con_bus_balance(m, p, t):
            return outlet_var[p, t] == inlet_var[p, t] * (1 - self.losses)

        setattr(self.block, 'con_bus_balance', pyomo.Constraint(
                model.time_set, rule=con_bus_balance))

    # ==========================================================================
    #    S E R I A L I Z E
    # ==========================================================================
    def serialize(self):
        """
        This method collects all relevant input data and optimization results
        from the Component instance, and returns them in an ordered dictionary.

        :return: OrderedDict
        """
        comp_dict = super().serialize()
        comp_dict['inlet_variable'] = self.inlet_variable
        comp_dict['outlet_variable'] = self.outlet_variable
        return comp_dict
