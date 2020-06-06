#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
**The Bus class**

* Last edited: 2020-06-06
* Created by: Stefan Bruche (TU Berlin)
"""
import pyomo.environ as pyomo
from aristopy import utils
from aristopy.component import Component


class Bus(Component):
    # A Bus component collects and transfers a commodity.
    # They can also be used to model transmission lines between different sites.
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

        :param losses:
            |br| *Default: 0*
        :type losses:
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

        # Check and set bus (transmission) specific input arguments
        self.losses = utils.set_if_between_zero_and_one(losses)  # relative loss

        # Store the names for the loading and unloading variables
        self.inlet_variable = self.inlet[0].var_name
        self.outlet_variable = self.outlet[0].var_name

        # Last step: Add the component to the EnergySystem instance
        self.add_to_energy_system(ensys, name)

    def __repr__(self):
        return '<Bus: "%s">' % self.name

    def declare_component_constraints(self, ensys, model):
        """
        Declare time independent and dependent constraints.

        :param ensys: EnergySystem instance representing the energy system
            in which the component should be added.
        :type ensys: EnergySystem class instance

        :param model: Pyomo ConcreteModel which stores the mathematical
            formulation of the energy system model.
        :type model: Pyomo ConcreteModel
        """

        # Time independent constraints:
        # -----------------------------
        self.con_couple_bi_ex_and_cap()
        self.con_cap_min()

        # Time dependent constraints:
        # ---------------------------
        self.con_operation_limit(model)
        self.con_bus_balance(model)


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #    A D D I T I O N A L   T I M E   D E P E N D E N T   C O N S .
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def con_operation_limit(self, model):
        """
        The operation of a bus comp. (inlet variable!) is limit by its nominal
        power (MW) multiplied with the number of hours per time step.
        E.g.: |br| ``Q_IN[p, t] <= CAP * dt``
        """
        # Only required if component has a capacity variable
        if self.has_capacity_var:
            # Get variables:
            cap = self.variables[utils.CAP]['pyomo']
            inlet_var = self.variables[self.inlet_variable]['pyomo']
            dt = self.ensys.hours_per_time_step

            def con_operation_limit(m, p, t):
                return inlet_var[p, t] <= cap * dt

            setattr(self.block, 'con_operation_limit', pyomo.Constraint(
                model.time_set, rule=con_operation_limit))

    def con_bus_balance(self, model):
        """
        The sum of outlets must equal the sum of the inlets minus the share of
        the transmission losses. A bus component cannot store a commodity.
        E.g.: |br| ``Q_OUT[p, t] == Q_IN[p, t] * (1 - losses)``
        (correction with "hours_per_time_step" not needed)
        """
        # Get variables:
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
        comp_dict = super().serialize()
        comp_dict['inlet_variable'] = self.inlet_variable
        comp_dict['outlet_variable'] = self.outlet_variable
        return comp_dict
