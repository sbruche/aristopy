#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
** The Source and Sink classes **

* Last edited: 2020-01-01
* Created by: Stefan Bruche (TU Berlin)
"""
import pyomo.environ as pyomo
from aristopy.component import Component
from aristopy import utils


class Source(Component):
    # Source components transfer commodities over the boundary into the system.
    def __init__(self, ensys, name, basic_variable='outlet_variable',
                 inlet=None, outlet=None,
                 has_existence_binary_var=False, has_operation_binary_var=False,
                 commodity_rate_min=None, commodity_rate_max=None,
                 commodity_rate_fix=None,
                 time_series_data=None, scalar_params=None,
                 additional_vars=None, user_expressions=None,
                 capacity=None, capacity_min=None, capacity_max=None,
                 capex_per_capacity=0, capex_if_exist=0,
                 opex_per_capacity=0, opex_if_exist=0, opex_operation=0,
                 commodity_cost=0, commodity_revenues=0
                 ):

        """
        Initialize a source component.

        :param ensys:
        :param name:
        :param basic_variable:
        :param inlet: ** Only accepts None for instances of Source **
        :param outlet:
        :param has_existence_binary_var:
        :param has_operation_binary_var: Should not be used in this Component!
        :param commodity_rate_min:
        :param commodity_rate_max:
        :param commodity_rate_fix:
        :param time_series_data:
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
        :param opex_operation:
        :param commodity_cost:
        :param commodity_revenues:
        """
        # The 'inlet' keyword should not be changed from default value 'None'
        if self.__class__ == Source and inlet is not None:
            raise ValueError('Source "%s" cannot have inlet Flows!' % name)

        Component.__init__(self, ensys, name,
                           inlet=inlet, outlet=outlet,
                           basic_variable=basic_variable,
                           has_existence_binary_var=has_existence_binary_var,
                           has_operation_binary_var=has_operation_binary_var,
                           time_series_data=time_series_data,
                           scalar_params=scalar_params,
                           additional_vars=additional_vars,
                           user_expressions=user_expressions,
                           capacity=capacity,
                           capacity_min=capacity_min,
                           capacity_max=capacity_max,
                           capex_per_capacity=capex_per_capacity,
                           capex_if_exist=capex_if_exist,
                           opex_per_capacity=opex_per_capacity,
                           opex_if_exist=opex_if_exist,
                           opex_operation=opex_operation
                           )

        # Check and set sink / source specific input arguments
        self.commodity_cost, self.commodity_cost_time_series = \
            utils.check_and_set_cost_and_revenues(self, commodity_cost)
        self.commodity_revenues, self.commodity_revenues_time_series = \
            utils.check_and_set_cost_and_revenues(self, commodity_revenues)

        # Check and set time series for commodity rates (if available)
        self.op_rate_min, self.op_rate_max, self.op_rate_fix = \
            utils.check_and_set_commodity_rates(
                self, commodity_rate_min, commodity_rate_max,
                commodity_rate_fix)

        # Last step: Add the component to the energy system model instance
        self.add_to_energy_system_model(ensys, name)

    def __repr__(self):
        return '<Source: "%s">' % self.name

    # **************************************************************************
    #   Declare component constraints
    # **************************************************************************
    def declare_component_constraints(self, ensys, pyM):
        """
        Declare time independent and dependent constraints.

        :param ensys: EnergySystem instance representing the energy system
            in which the component should be added.
        :type ensys: EnergySystem class instance

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
        self.con_bi_var_ex_and_op_relation(pyM)
        self.con_operation_limit(pyM)
        self.con_couple_op_binary_and_basic_var(pyM)
        self.con_commodity_rate_min(pyM)
        self.con_commodity_rate_max(pyM)
        self.con_commodity_rate_fix(pyM)

    def get_objective_function_contribution(self, ensys, pyM):
        """ Get contribution to the objective function. """
        # Check if the component is completely unconnected. If this is True,
        # don't use the objective function contributions of this component
        # (could create infeasibilities!)
        if len(self.var_connections.keys()) == 0:
            self.log.warn('Found an unconnected component! Skipped possible '
                          'objective function contributions.')
            return 0

        # Call function in "Component" class and calculate CAPEX and OPEX
        super().get_objective_function_contribution(ensys, pyM)

        # --------------------------------
        #   COMMODITY COST AND REVENUES
        # --------------------------------
        # Get the basic variable:
        basic_var = self.variables[self.basic_variable]['pyomo']

        # Time-independent cost of a commodity (scalar cost value)
        if self.commodity_cost > 0:
            self.comp_obj_dict['commodity_cost'] = \
                -1 * ensys.pvf * self.commodity_cost * sum(
                    basic_var[p, t] * ensys.period_occurrences[p]
                    for p, t in pyM.time_set) / ensys.number_of_years

        # Time-dependent cost of a commodity (time series cost values)
        if self.commodity_cost_time_series is not None:
            cost_ts = self.parameters[self.commodity_cost_time_series]['values']
            self.comp_obj_dict['commodity_cost'] = -1 * ensys.pvf * sum(
                cost_ts[p, t] * basic_var[p, t] * ensys.period_occurrences[p]
                for p, t in pyM.time_set) / ensys.number_of_years

        # Time-independent revenues for of a commodity (scalar revenue value)
        if self.commodity_revenues > 0:
            self.comp_obj_dict['commodity_revenues'] = \
                ensys.pvf * self.commodity_revenues * sum(
                    basic_var[p, t] * ensys.period_occurrences[p]
                    for p, t in pyM.time_set) / ensys.number_of_years

        # Time-dependent revenues for a commodity (time series revenue values)
        if self.commodity_revenues_time_series is not None:
            rev_ts = self.parameters[
                self.commodity_revenues_time_series]['values']
            self.comp_obj_dict['commodity_revenues'] = ensys.pvf * sum(
                rev_ts[p, t] * basic_var[p, t] * ensys.period_occurrences[p]
                for p, t in pyM.time_set) / ensys.number_of_years

        return sum(self.comp_obj_dict.values())

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #    A D D I T I O N A L   T I M E   D E P E N D E N T   C O N S .
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def con_operation_limit(self, pyM):
        """
        The operation variable (ref.: basic variable / basic commodity) of a
        sink / source unit (MWh) is limit by its nominal power (MW) multiplied
        with the number of hours per time step. E.g.: |br|
        ``Q[p, t] <= CAP * dt``
        """
        # Only required if component has a capacity variable
        if self.has_capacity_var:
            # Get variables:
            cap = self.variables[utils.CAP]['pyomo']
            basic_var = self.variables[self.basic_variable]['pyomo']
            has_time_set = self.variables[self.basic_variable]['has_time_set']
            dt = self.ensys.hours_per_time_step

            def con_operation_limit(m, p, t):
                if has_time_set:
                    return basic_var[p, t] <= cap * dt
                else:
                    return basic_var <= cap

            setattr(self.pyB, 'con_operation_limit', pyomo.Constraint(
                pyM.time_set, rule=con_operation_limit))

    def con_commodity_rate_min(self, pyM):
        """
        The basic variable of a component needs to have a minimal value of
        "commodity_rate_min" in every time step. E.g.: |br|
        ``Q[p, t] >= op_rate_min[p, t]``
        (No correction with "hours_per_time_step" needed because it should
        already be included in the time series for "commodity_rate_min")
        """
        # Only required if component has a time series for "commodity_rate_min".
        if self.op_rate_min is not None:
            # Get variables:
            op_min = self.parameters[self.op_rate_min]['values']
            basic_var = self.variables[self.basic_variable]['pyomo']
            has_time_set = self.parameters[self.op_rate_min]['has_time_set']

            def con_commodity_rate_min(m, p, t):
                if has_time_set:
                    return basic_var[p, t] >= op_min[p, t]
                else:
                    return basic_var[p, t] >= op_min
            setattr(self.pyB, 'con_commodity_rate_min', pyomo.Constraint(
                pyM.time_set, rule=con_commodity_rate_min))

    def con_commodity_rate_max(self, pyM):
        """
        The basic variable of a component can have a maximal value of
        "commodity_rate_max" in every time step. E.g.: |br|
        ``Q[p, t] <= op_rate_max[p, t]``
        (No correction with "hours_per_time_step" needed because it should
        already be included in the time series for "commodity_rate_max")
        """
        # Only required if component has a time series for "commodity_rate_max".
        if self.op_rate_max is not None:
            # Get variables:
            op_max = self.parameters[self.op_rate_max]['values']
            basic_var = self.variables[self.basic_variable]['pyomo']
            has_time_set = self.parameters[self.op_rate_max]['has_time_set']

            def con_commodity_rate_max(m, p, t):
                if has_time_set:
                    return basic_var[p, t] <= op_max[p, t]
                else:
                    return basic_var[p, t] <= op_max
            setattr(self.pyB, 'con_commodity_rate_max', pyomo.Constraint(
                pyM.time_set, rule=con_commodity_rate_max))

    def con_commodity_rate_fix(self, pyM):
        """
        The basic variable of a component needs to have a value of
        "commodity_rate_fix" in every time step. E.g.: |br|
        ``Q[p, t] == op_rate_fix[p, t]``
        (No correction with "hours_per_time_step" needed because it should
        already be included in the time series for "commodity_rate_fix")
        """
        # Only required if component has a time series for "commodity_rate_fix"
        if self.op_rate_fix is not None:
            # Get variables:
            op_fix = self.parameters[self.op_rate_fix]['values']
            basic_var = self.variables[self.basic_variable]['pyomo']
            has_time_set = self.parameters[self.op_rate_fix]['has_time_set']

            def con_commodity_rate_fix(m, p, t):
                if has_time_set:
                    return basic_var[p, t] == op_fix[p, t]
                else:
                    return basic_var[p, t] == op_fix
            setattr(self.pyB, 'con_commodity_rate_fix', pyomo.Constraint(
                pyM.time_set, rule=con_commodity_rate_fix))

    # ==========================================================================
    #    S E R I A L I Z E
    # ==========================================================================
    def serialize(self):
        comp_dict = super().serialize()
        comp_dict['commodity_rate_min'] = self.op_rate_min
        comp_dict['commodity_rate_max'] = self.op_rate_max
        comp_dict['commodity_rate_fix'] = self.op_rate_fix
        comp_dict['commodity_cost_time_series'] = \
            self.commodity_cost_time_series
        comp_dict['commodity_revenues_time_series'] = \
            self.commodity_revenues_time_series
        return comp_dict


class Sink(Source):
    # Sink components transfer commodities over the boundary out of the system.
    def __init__(self, ensys, name, basic_variable='inlet_variable',
                 inlet=None,
                 has_existence_binary_var=None, has_operation_binary_var=None,
                 commodity_rate_min=None, commodity_rate_max=None,
                 commodity_rate_fix=None,
                 time_series_data=None, scalar_params=None,
                 additional_vars=None, user_expressions=None,
                 capacity=None, capacity_min=None, capacity_max=None,
                 capex_per_capacity=0, capex_if_exist=0,
                 opex_per_capacity=0, opex_if_exist=0, opex_operation=0,
                 commodity_cost=0, commodity_revenues=0
                 ):
        """
        Initialize a sink component. The Sink class inherits from the Source
        class. Both have the same input parameters with only one exception.
        The Sink has an "inlet" attribute instead of "outlet".

        :param inlet: TODO: Add description!
        """

        Source.__init__(self, ensys, name, basic_variable=basic_variable,
                        inlet=inlet,
                        has_existence_binary_var=has_existence_binary_var,
                        has_operation_binary_var=has_operation_binary_var,
                        commodity_rate_min=commodity_rate_min,
                        commodity_rate_max=commodity_rate_max,
                        commodity_rate_fix=commodity_rate_fix,
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
                        opex_operation=opex_operation,
                        commodity_cost=commodity_cost,
                        commodity_revenues=commodity_revenues)

    def __repr__(self):
        return '<Sink: "%s">' % self.name
