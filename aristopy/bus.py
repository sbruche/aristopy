#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
** The Bus class **

* Last edited: 2020-01-01
* Created by: Stefan Bruche (TU Berlin)
"""
import pyomo.environ as pyomo

from aristopy import utils
from aristopy.component import Component


class Bus(Component):
    # A Bus component collects and transfers a commodity.
    # They can also be used to model transmission lines between different sites.
    def __init__(self, ensys, name, basic_commodity,
                 inlet_connections=None, outlet_connections=None,
                 existence_binary_var=None,
                 time_series_data=None,
                 time_series_weights=None,
                 scalar_params=None, additional_vars=None,
                 user_expressions=None,
                 capacity=None, capacity_min=None, capacity_max=None,
                 # fix_existence=None, oder 1 oder 0
                 capex_per_capacity=0, capex_if_exist=0,
                 opex_per_capacity=0, opex_if_exist=0, opex_operation=0,
                 losses=0
                 ):
        """
        Initialize a bus component.

        :param ensys:
        :param name:
        :param basic_commodity:
        :param inlet_connections:
        :param outlet_connections:
        :param existence_binary_var:
        :param time_series_data:
        :param time_series_weights:
        :param scalar_params:
        :param additional_vars:
        :param user_expressions:
        :param capacity:
        :param capacity_min:
        :param capacity_max:
        :param capex_per_capacity:
        :param capex_if_exist:
        :param opex_per_capacity:
        :param opex_if_exist:
        :param losses:
        """

        Component.__init__(self, ensys, name, basic_commodity,
                           existence_binary_var=existence_binary_var,
                           time_series_data=time_series_data,
                           time_series_weights=time_series_weights,
                           scalar_params=scalar_params,
                           additional_vars=additional_vars,
                           user_expressions=user_expressions,
                           capacity=capacity, capacity_min=capacity_min,
                           capacity_max=capacity_max,
                           capex_per_capacity=capex_per_capacity,
                           capex_if_exist=capex_if_exist,
                           opex_per_capacity=opex_per_capacity,
                           opex_if_exist=opex_if_exist
                           )

        # Check and set bus (transmission) specific input arguments
        self.opex_operation = utils.set_if_positive(opex_operation)
        self.losses = utils.set_if_between_zero_and_one(losses)  # relative loss

        # Declare create two variables. One for loading and one for unloading.
        self.inlet_variable = self.basic_commodity + '_IN'
        self.outlet_variable = self.basic_commodity + '_OUT'
        self._add_var(self.inlet_variable)
        self._add_var(self.outlet_variable)

        # Check and add inlet and outlet connections
        self.inlet_connections = utils.check_and_convert_to_list(
            inlet_connections)
        self.outlet_connections = utils.check_and_convert_to_list(
            outlet_connections)
        self.inlet_ports_and_vars = \
            {self.basic_commodity: self.inlet_variable}
        self.outlet_ports_and_vars = \
            {self.basic_commodity: self.outlet_variable}

        # Last step: Add the component to the energy system model instance
        self.add_to_energy_system_model(ensys, name)

    def __repr__(self):
        return '<Bus: "%s">' % self.name

    def declare_component_constraints(self, ensys, pyM):
        """
        Declare time independent and dependent constraints.

        :param ensys: EnergySystemModel instance representing the energy system
            in which the component should be added.
        :type ensys: EnergySystemModel class instance

        :param pyM: Pyomo ConcreteModel which stores the mathematical
            formulation of the energy system model.
        :type pyM: Pyomo ConcreteModel
        """

        # Time independent constraints:
        # -----------------------------
        self.con_couple_bi_ex_and_cap()
        self.con_cap_min()

        # Time dependent constraints:
        # ---------------------------
        self.con_operation_limit(pyM)
        self.con_bus_balance(pyM)

    def get_objective_function_contribution(self, ensys, pyM):
        """ Get contribution to the objective function. """

        # Alias of the components' objective function dictionary
        obj = self.comp_obj_dict

        # Get the inlet variable:
        inlet_var = self.variables[self.inlet_variable]['pyomo']
        # Check if the bus is unconnected. If this is True, don't calculate the
        # objective function contributions (could create infeasibilities!)
        if len(self.var_connections.keys()) == 0:
            self.log.warn('Found an unconnected component! Skipped possible '
                          'objective function contributions.')
            return 0

        # ---------------
        #   C A P E X
        # ---------------
        # CAPEX depending on capacity
        if self.capex_per_capacity > 0:
            cap = self.variables[self.capacity_variable]['pyomo']
            obj['capex_capacity'] = -1 * self.capex_per_capacity * cap

        # CAPEX depending on existence of component
        if self.capex_if_exist > 0:
            bi_ex = self.variables[self.bi_ex]['pyomo']
            obj['capex_exist'] = -1 * self.capex_if_exist * bi_ex
        # ---------------
        #   O P E X
        # ---------------
        # OPEX depending on capacity
        if self.opex_per_capacity > 0:
            cap = self.variables[self.capacity_variable]['pyomo']
            obj['opex_capacity'] = -1 * ensys.pvf * self.opex_per_capacity * cap

        # OPEX depending on existence of component
        if self.opex_if_exist > 0:
            bi_ex = self.variables[self.bi_ex]['pyomo']
            obj['opex_exist'] = -1 * ensys.pvf * self.opex_if_exist * bi_ex

        # OPEX for operating the bus: Associated with the INLET variable!
        if self.opex_operation > 0:
            obj['opex_operation'] = -1 * ensys.pvf * self.opex_operation * sum(
                inlet_var[p, t] * ensys.period_occurrences[p] for p, t in
                pyM.time_set) / ensys.number_of_years

        return sum(obj.values())

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #    A D D I T I O N A L   T I M E   D E P E N D E N T   C O N S .
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def con_operation_limit(self, pyM):
        """
        The operation of a bus comp. (inlet variable!) is limit by its nominal
        power (MW) multiplied with the number of hours per time step.
        E.g.: |br| ``Q_IN[p, t] <= Q_CAP * dt``
        """
        # Only required if component has a capacity variable
        if self.capacity_variable is not None:
            # Get variables:
            cap = self.variables[self.capacity_variable]['pyomo']
            inlet_var = self.variables[self.inlet_variable]['pyomo']
            dt = self.ensys.hours_per_time_step

            def con_operation_limit(m, p, t):
                return inlet_var[p, t] <= cap * dt

            setattr(self.pyB, 'con_operation_limit', pyomo.Constraint(
                pyM.time_set, rule=con_operation_limit))

    def con_bus_balance(self, pyM):
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

        setattr(self.pyB, 'con_bus_balance', pyomo.Constraint(
                pyM.time_set, rule=con_bus_balance))

    # ==========================================================================
    #    S E R I A L I Z E
    # ==========================================================================
    def serialize(self):
        comp_dict = super().serialize()
        comp_dict['inlet_variable'] = self.inlet_variable
        comp_dict['outlet_variable'] = self.outlet_variable
        return comp_dict
