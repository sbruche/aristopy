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
    def __init__(self, ensys, name, basic_commodity, outlet_connections=None,
                 existence_binary_var=None, operation_binary_var=None,
                 commodity_rate_min=None, commodity_rate_max=None,
                 commodity_rate_fix=None,
                 time_series_data=None, time_series_weights=None,
                 scalar_params=None, additional_vars=None,
                 user_expressions=None,
                 capacity=None, capacity_min=None, capacity_max=None,
                 capex_per_capacity=0, capex_if_exist=0,
                 opex_per_capacity=0, opex_if_exist=0, opex_operation=0,
                 commodity_cost=0, commodity_cost_time_series=None,
                 commodity_revenues=0, commodity_revenues_time_series=None
                 ):
        """
        Initialize a source component.

        :param ensys:
        :param name:
        :param basic_commodity:
        :param outlet_connections:
        :param existence_binary_var:
        :param operation_binary_var: **Should not be used in this Component!**
        :param commodity_rate_min:
        :param commodity_rate_max:
        :param commodity_rate_fix:
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
        :param opex_operation:
        :param commodity_cost:
        :param commodity_cost_time_series:
        :param commodity_revenues:
        :param commodity_revenues_time_series:
        """

        Component.__init__(self, ensys, name, basic_commodity,
                           existence_binary_var=existence_binary_var,
                           operation_binary_var=operation_binary_var,
                           time_series_data=time_series_data,
                           time_series_weights=time_series_weights,
                           scalar_params=scalar_params,
                           additional_vars=additional_vars,
                           user_expressions=user_expressions,
                           capacity=capacity,
                           capacity_min=capacity_min,
                           capacity_max=capacity_max,
                           capex_per_capacity=capex_per_capacity,
                           capex_if_exist=capex_if_exist,
                           opex_per_capacity=opex_per_capacity,
                           opex_if_exist=opex_if_exist
                           )
        # todo:Check if list or string, if string convert to list with one entry
        if self.__class__ == Source:
            # Check and add outlet connections
            self.outlet_connections = utils.check_and_convert_to_list(
                outlet_connections)
            self.basic_variable = self.basic_commodity + '_OUT'
            self.outlet_ports_and_vars = \
                {self.basic_commodity: self.basic_variable}
            self._add_var(self.basic_variable)

        # Check and set sink / source specific input arguments
        self.opex_operation = utils.set_if_positive(opex_operation)
        self.commodity_cost = utils.set_if_positive(commodity_cost)
        self.commodity_revenues = utils.set_if_positive(commodity_revenues)
        self.commodity_cost_time_series = utils.check_existence_in_dataframe(
            commodity_cost_time_series, self.parameters)
        self.commodity_revenues_time_series = \
            utils.check_existence_in_dataframe(commodity_revenues_time_series,
                                               self.parameters)

        # Check and set time series for commodity rates (if available)
        self.op_rate_min, self.op_rate_max, self.op_rate_fix = \
            utils.check_commodity_rates(self, commodity_rate_min,
                                        commodity_rate_max, commodity_rate_fix)

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
        self.con_bi_var_ex_and_op_relation(pyM)
        self.con_operation_limit(pyM)
        self.con_couple_op_binary_and_basic_var(pyM)
        self.con_commodity_rate_min(pyM)
        self.con_commodity_rate_max(pyM)
        self.con_commodity_rate_fix(pyM)

    def get_objective_function_contribution(self, ensys, pyM):
        """ Get contribution to the objective function. """

        # Alias of the components' objective function dictionary
        obj = self.comp_obj_dict
        # Get general required variables:
        basic_var = self.variables[self.basic_variable]['pyomo']

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

        # OPEX depending on existence of sink / source
        if self.opex_if_exist > 0:
            bi_ex = self.variables[self.bi_ex]['pyomo']
            obj['opex_exist'] = -1 * ensys.pvf * self.opex_if_exist * bi_ex

        # OPEX for operating the sink / source
        if self.opex_operation > 0:
            obj['opex_operation'] = -1 * ensys.pvf * self.opex_operation * sum(
                basic_var[p, t] * ensys.period_occurrences[p] for p, t in
                pyM.time_set) / ensys.number_of_years
        # ---------------
        #   M I S C
        # ---------------
        # Time-independent cost of a commodity (scalar cost value)
        if self.commodity_cost is not None:
            obj['com_cost_time_indep'] = \
                -1 * ensys.pvf * self.commodity_cost * sum(
                    basic_var[p, t] * ensys.period_occurrences[p]
                    for p, t in pyM.time_set) / ensys.number_of_years

        # Time-dependent cost of a commodity (time series cost values)
        if self.commodity_cost_time_series is not None:
            cost_ts = self.parameters[self.commodity_cost_time_series]['values']
            obj['com_cost_time_dep'] = -1 * ensys.pvf * sum(
                cost_ts[p, t] * basic_var[p, t] * ensys.period_occurrences[p]
                for p, t in pyM.time_set) / ensys.number_of_years

        # Time-independent revenues for of a commodity (scalar revenue value)
        if self.commodity_revenues is not None:
            obj['com_rev_time_indep'] = \
                ensys.pvf * self.commodity_revenues * sum(
                    basic_var[p, t] * ensys.period_occurrences[p]
                    for p, t in pyM.time_set) / ensys.number_of_years

        # Time-dependent revenues for a commodity (time series revenue values)
        if self.commodity_revenues_time_series is not None:
            rev_ts = self.parameters[
                self.commodity_revenues_time_series]['values']
            obj['com_rev_time_dep'] = ensys.pvf * sum(
                rev_ts[p, t] * basic_var[p, t] * ensys.period_occurrences[p]
                for p, t in pyM.time_set) / ensys.number_of_years

        return sum(obj.values())

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #    A D D I T I O N A L   T I M E   D E P E N D E N T   C O N S .
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def con_operation_limit(self, pyM):
        """
        The operation variable (ref.: basic variable / basic commodity) of a
        sink / source unit (MWh) is limit by its nominal power (MW) multiplied
        with the number of hours per time step. E.g.: |br|
        ``Q[p, t] <= Q_CAP * dt``
        """
        # Only required if component has a capacity variable
        if self.capacity_variable is not None:
            # Get variables:
            cap = self.variables[self.capacity_variable]['pyomo']
            basic_var = self.variables[self.basic_variable]['pyomo']
            dt = self.ensys.hours_per_time_step

            def con_operation_limit(m, p, t):
                return basic_var[p, t] <= cap * dt

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

            def con_commodity_rate_min(m, p, t):
                return basic_var[p, t] >= op_min[p, t]
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

            def con_commodity_rate_max(m, p, t):
                return basic_var[p, t] <= op_max[p, t]
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

            def con_commodity_rate_fix(m, p, t):
                return basic_var[p, t] == op_fix[p, t]
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
        return comp_dict


class Sink(Source):
    # Sink components transfer commodities over the boundary out of the system.
    def __init__(self, ensys, name, basic_commodity, inlet_connections=None,
                 existence_binary_var=None, operation_binary_var=None,
                 commodity_rate_min=None, commodity_rate_max=None,
                 commodity_rate_fix=None,
                 time_series_data=None, time_series_weights=None,
                 scalar_params=None, additional_vars=None,
                 user_expressions=None,
                 capacity=None, capacity_min=None, capacity_max=None,
                 capex_per_capacity=0, capex_if_exist=0,
                 opex_per_capacity=0, opex_if_exist=0, opex_operation=0,
                 commodity_cost=0, commodity_cost_time_series=None,
                 commodity_revenues=0, commodity_revenues_time_series=None
                 ):
        """
        Initialize a sink component. The Sink class inherits from the Source
        class. Both have the same input parameters with only one exception.
        The Sink has an "inlet_connections" attribute instead of
        "inlet_connections".

        :param inlet_connections: TODO: Add description!
        """

        Source.__init__(self, ensys, name, basic_commodity,
                        existence_binary_var=existence_binary_var,
                        operation_binary_var=operation_binary_var,
                        commodity_rate_min=commodity_rate_min,
                        commodity_rate_max=commodity_rate_max,
                        commodity_rate_fix=commodity_rate_fix,
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
                        opex_if_exist=opex_if_exist,
                        opex_operation=opex_operation,
                        commodity_cost=commodity_cost,
                        commodity_cost_time_series=commodity_cost_time_series,
                        commodity_revenues=commodity_revenues,
                        commodity_revenues_time_series=
                        commodity_revenues_time_series)

        if self.__class__ == Sink:
            # Check and add inlet connections
            self.inlet_connections = utils.check_and_convert_to_list(
                inlet_connections)
            self.basic_variable = self.basic_commodity + '_IN'
            self.inlet_ports_and_vars = \
                {self.basic_commodity: self.basic_variable}
            self._add_var(self.basic_variable)

    def __repr__(self):
        return '<Sink: "%s">' % self.name
