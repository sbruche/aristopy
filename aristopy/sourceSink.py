#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
** The Source and Sink classes **

* Last edited: 2020-01-01
* Created by: Stefan Bruche (TU Berlin)
"""
import warnings
import pyomo.environ as pyomo
import pyomo.network as network
from aristopy.component import Component
from aristopy import utils


class Source(Component):
    # Source components transfer commodities over the boundary into the system.
    def __init__(self, ensys, name, basic_variable, inlets=None, outlets=None,
                 existence_binary_var=None, operation_binary_var=None,
                 operation_rate_min=None, operation_rate_max=None,
                 operation_rate_fix=None,
                 time_series_data_dict=None, time_series_weight_dict=None,
                 scalar_params_dict=None, additional_vars=None,
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
        :param basic_variable:
        :param inlets:
        :param outlets:
        :param existence_binary_var:
        :param operation_binary_var: **Should not be used in this Component!**
        :param operation_rate_min:
        :param operation_rate_max:
        :param operation_rate_fix:
        :param time_series_data_dict:
        :param time_series_weight_dict:
        :param scalar_params_dict:
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

        Component.__init__(self, ensys, name, basic_variable,
                           inlets=inlets, outlets=outlets,
                           existence_binary_var=existence_binary_var,
                           operation_binary_var=operation_binary_var,
                           operation_rate_min=operation_rate_min,
                           operation_rate_max=operation_rate_max,
                           operation_rate_fix=operation_rate_fix,
                           time_series_data_dict=time_series_data_dict,
                           time_series_weight_dict=time_series_weight_dict,
                           scalar_params_dict=scalar_params_dict,
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

        self.modeling_class = 'Src'

        if self.__class__ == Source and inlets is not None:
            warnings.warn('A "source" should better not have inlet streams!')

        # Check and set sink / source specific input arguments
        self.opex_operation = utils.set_if_positive(opex_operation)
        self.commodity_cost = utils.set_if_positive(commodity_cost)
        self.commodity_revenues = utils.set_if_positive(commodity_revenues)
        self.commodity_cost_time_series = utils.check_existence_in_dataframe(
            commodity_cost_time_series, self.parameters)
        self.commodity_revenues_time_series = \
            utils.check_existence_in_dataframe(commodity_revenues_time_series,
                                               self.parameters)

        # Last step: Add the component to the energy system model instance
        self.add_to_energy_system_model(ensys, name)

    def __repr__(self):
        return '<Source: "%s">' % self.name

    def declare_component_ports(self):
        """
        Create all ports from dict 'ports_and_vars' and add variables to ports.
        """
        # Create ports and assign variables to ports
        for port_name, var_name in self.ports_and_vars.items():
            # Declare port
            setattr(self.pyB, port_name, network.Port())
            # Add variable to port
            port = getattr(self.pyB, port_name)
            port.add(getattr(self.pyB, var_name), var_name, port.Extensive)

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
        self.con_operation_rate_min(pyM)
        self.con_operation_rate_max(pyM)
        self.con_operation_rate_fix(pyM)

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
                pyM.time_set) / ensys.number_of_years  # * ensys.hours_per_time_step
        # ---------------
        #   M I S C
        # ---------------
        # Time-independent cost of a commodity (scalar cost value)
        if self.commodity_cost is not None:
            obj['com_cost_time_indep'] = \
                -1 * ensys.pvf * self.commodity_cost * sum(
                    basic_var[p, t] * ensys.period_occurrences[p]
                    for p, t in pyM.time_set) / ensys.number_of_years  # * ensys.hours_per_time_step

        # Time-dependent cost of a commodity (time series cost values)
        if self.commodity_cost_time_series is not None:
            cost_ts = self.parameters[self.commodity_cost_time_series]['values']
            obj['com_cost_time_dep'] = -1 * ensys.pvf * sum(
                cost_ts[p, t] * basic_var[p, t] * ensys.period_occurrences[p]
                for p, t in pyM.time_set) / ensys.number_of_years  # * ensys.hours_per_time_step

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
                for p, t in pyM.time_set) / ensys.number_of_years  # * ensys.hours_per_time_step

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


class Sink(Source):
    # Sink components transfer commodities over the boundary out of the system.
    def __init__(self, ensys, name, basic_variable, inlets=None, outlets=None,
                 existence_binary_var=None,  operation_binary_var=None,
                 operation_rate_min=None, operation_rate_max=None,
                 operation_rate_fix=None,
                 time_series_data_dict=None, time_series_weight_dict=None,
                 scalar_params_dict=None, additional_vars=None,
                 user_expressions=None,
                 capacity=None, capacity_min=None, capacity_max=None,
                 capex_per_capacity=0, capex_if_exist=0,
                 opex_per_capacity=0, opex_if_exist=0, opex_operation=0,
                 commodity_cost=0, commodity_cost_time_series=None,
                 commodity_revenues=0, commodity_revenues_time_series=None
                 ):
        """
        Initialize a sink component. The Sink class inherits from the Source
        class. Both have the same input parameters. See the Source class for a
        description of the input parameters.
        """

        Source.__init__(self, ensys, name, basic_variable, inlets, outlets,
                        existence_binary_var,  operation_binary_var,
                        operation_rate_min, operation_rate_max,
                        operation_rate_fix,
                        time_series_data_dict, time_series_weight_dict,
                        scalar_params_dict, additional_vars, user_expressions,
                        capacity, capacity_min, capacity_max,
                        capex_per_capacity, capex_if_exist,
                        opex_per_capacity, opex_if_exist, opex_operation,
                        commodity_cost, commodity_cost_time_series,
                        commodity_revenues, commodity_revenues_time_series)

        self.modeling_class = 'Snk'

        if self.__class__ == Sink and outlets is not None:
            warnings.warn('A "sink" should better not have outlet streams!')

    def __repr__(self):
        return '<Sink: "%s">' % self.name
