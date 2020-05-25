#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
** The Conversion class **

* Last edited: 2020-01-01
* Created by: Stefan Bruche (TU Berlin)
"""
import pyomo.environ as pyomo
from aristopy.component import Component
from aristopy import utils


class Conversion(Component):
    # A Conversion component converts commodities into each other.
    def __init__(self, ensys, name, basic_variable,
                 inlet=None, outlet=None,
                 has_existence_binary_var=None, has_operation_binary_var=None,
                 time_series_data=None, time_series_weights=None,
                 scalar_params=None, additional_vars=None,
                 user_expressions=None,
                 capacity=None, capacity_min=None, capacity_max=None,
                 capacity_per_module=None, maximal_module_number=None,
                 capex_per_capacity=0, capex_if_exist=0,
                 opex_per_capacity=0, opex_if_exist=0, opex_operation=0,
                 start_up_cost=0, min_load_rel=None, instances_in_group=1,
                 group_has_existence_order=True, group_has_operation_order=True,
                 use_inter_period_formulation=True
                 ):
        """
        Initialize a conversion component.

        :param ensys:
        :param name:
        :param basic_variable:
        :param inlet:
        :param outlet:
        :param has_existence_binary_var:
        :param has_operation_binary_var:
        :param time_series_data:
        :param time_series_weights:
        :param scalar_params:
        :param additional_vars:
        :param user_expressions:
        :param capacity:
        :param capacity_min:
        :param capacity_max:
        :param capacity_per_module:
        :param maximal_module_number:
        :param capex_per_capacity:
        :param capex_if_exist:
        :param opex_per_capacity:
        :param opex_if_exist:
        :param opex_operation:
        :param start_up_cost:
        :param min_load_rel:
        :param instances_in_group:
        :param group_has_existence_order:
        :param group_has_operation_order:
        """

        Component.__init__(self, ensys, name, basic_variable=basic_variable,
                           inlet=inlet, outlet=outlet,
                           has_existence_binary_var=has_existence_binary_var,
                           has_operation_binary_var=has_operation_binary_var,
                           time_series_data=time_series_data,
                           time_series_weights=time_series_weights,
                           scalar_params=scalar_params,
                           additional_vars=additional_vars,
                           user_expressions=user_expressions,
                           capacity=capacity,
                           capacity_min=capacity_min,
                           capacity_max=capacity_max,
                           capacity_per_module=capacity_per_module,
                           maximal_module_number=maximal_module_number,
                           capex_per_capacity=capex_per_capacity,
                           capex_if_exist=capex_if_exist,
                           opex_per_capacity=opex_per_capacity,
                           opex_if_exist=opex_if_exist,
                           instances_in_group=instances_in_group,
                           group_has_existence_order=group_has_existence_order,
                           group_has_operation_order=group_has_operation_order
                           )

        # Check and set the value for the minimal relative part-load
        if min_load_rel is not None:
            utils.is_positive_number(min_load_rel)
            assert min_load_rel <= 1, 'Maximal value for "min_load_rel" is 1!'
            if not self.has_bi_op:
                raise ValueError('Minimal part-loads require the availability '
                                 'of a binary operation variable.')
        self.min_load_rel = min_load_rel

        # Check and set more conversion specific input arguments
        self.opex_operation = utils.set_if_positive(opex_operation)
        self.start_up_cost = utils.set_if_positive(start_up_cost)  # [€/Start]
        utils.is_boolean(use_inter_period_formulation)  # check input
        self.use_inter_period_formulation = use_inter_period_formulation

        if self.start_up_cost != 0:
            if not self.has_bi_op:
                raise ValueError('Start-up cost require the availability '
                                 'of a binary operation variable.')
            # Else: add Start-up binary variable
            self._add_var(name='BI_SU', domain='Binary', has_time_set=True)
            # If inter-period-formulation is requested: add another binary var
            if self.use_inter_period_formulation:
                self._add_var(name='BI_SU_INTER', domain='Binary',
                              has_time_set=False, init=0,
                              alternative_set='inter_period_time_set')

        # Last step: Add the component to the energy system model instance
        self.add_to_energy_system_model(ensys, name, instances_in_group)

    def __repr__(self):
        return '<Conversion: "%s">' % self.name

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
        self.con_cap_modular()
        self.con_modular_sym_break()
        self.con_couple_existence_and_modular()
        self.con_existence_sym_break(ensys)

        # Time dependent constraints:
        # ---------------------------
        self.con_bi_var_ex_and_op_relation(pyM)
        self.con_operation_limit(pyM)
        self.con_couple_op_binary_and_basic_var(pyM)
        self.con_min_load_rel(pyM)
        self.con_start_up_cost(ensys, pyM)
        self.con_start_up_cost_inter(ensys, pyM)
        self.con_operation_sym_break(ensys, pyM)

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
            cap = self.variables['CAP']['pyomo']
            obj['capex_capacity'] = -1 * self.capex_per_capacity * cap

        # CAPEX depending on existence of component
        if self.capex_if_exist > 0:
            bi_ex = self.variables['BI_EX']['pyomo']
            obj['capex_exist'] = -1 * self.capex_if_exist * bi_ex
        # ---------------
        #   O P E X
        # ---------------
        # OPEX depending on capacity
        if self.opex_per_capacity > 0:
            cap = self.variables['CAP']['pyomo']
            obj['opex_capacity'] = -1 * ensys.pvf * self.opex_per_capacity * cap

        # OPEX depending on existence of component
        if self.opex_if_exist > 0:
            bi_ex = self.variables['BI_EX']['pyomo']
            obj['opex_exist'] = -1 * ensys.pvf * self.opex_if_exist * bi_ex

        # OPEX for operating the conversion unit
        if self.opex_operation > 0:
            obj['opex_operation'] = -1 * ensys.pvf * self.opex_operation * sum(
                basic_var[p, t] * ensys.period_occurrences[p] for p, t in
                pyM.time_set) / ensys.number_of_years
        # ---------------
        #   M I S C
        # ---------------
        # Start-up cost
        if self.start_up_cost > 0:
            bi_su = self.variables['BI_SU']['pyomo']  # only available if '>0'
            start_cost_intra = -1 * ensys.pvf * self.start_up_cost * sum(
                bi_su[p, t] * ensys.period_occurrences[p] for p, t in
                pyM.time_set) / ensys.number_of_years
            if self.use_inter_period_formulation and ensys.is_data_clustered:
                bi_su_inter = self.variables['BI_SU_INTER']['pyomo']
                start_cost_inter = -1 * ensys.pvf * self.start_up_cost * pyomo.\
                    summation(bi_su_inter) / ensys.number_of_years
            else:
                start_cost_inter = 0
            obj['start_up_cost'] = start_cost_intra + start_cost_inter
        return sum(obj.values())

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #    A D D I T I O N A L   T I M E   I N D E P E N D E N T   C O N S .
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def con_existence_sym_break(self, ensys):
        """
        TODO: Add description!
        """
        if self.number_in_group > 1 and self.group_has_existence_order:
            bi_ex = self.variables['BI_EX']['pyomo']
            # Get 'bi_ex' of the previous component in the sequence
            prior = self.group_name + '_{}'.format(self.number_in_group - 1)
            prior_comp = ensys.components[prior]
            bi_ex_prior = prior_comp.variables['BI_EX']['pyomo']

            def con_existence_sym_break(m):
                return bi_ex <= bi_ex_prior
            setattr(self.pyB, 'con_existence_sym_break',
                    pyomo.Constraint(rule=con_existence_sym_break))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #    A D D I T I O N A L   T I M E   D E P E N D E N T   C O N S .
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def con_operation_sym_break(self, ensys, pyM):
        """
        TODO: Add description!
        """
        if self.number_in_group > 1 and self.group_has_operation_order:
            # Get variables:
            bi_op = self.variables['BI_OP']['pyomo']
            prior = self.group_name + '_{}'.format(self.number_in_group - 1)
            prior_comp = ensys.components[prior]
            bi_op_prior = prior_comp.variables['BI_OP']['pyomo']

            def con_operation_sym_break(m, p, t):
                return bi_op[p, t] <= bi_op_prior[p, t]
            setattr(self.pyB, 'con_operation_sym_break', pyomo.Constraint(
                pyM.time_set, rule=con_operation_sym_break))

    def con_operation_limit(self, pyM):
        """
        The operation variable (ref.: basic commodity) of a conversion unit
        (MWh) is limit by its nominal power (MW) multiplied with the number of
        hours per time step. E.g.: |br|
        ``Q[p, t] <= Q_CAP * dt``
        """
        # Only required if component has a capacity variable
        if self.has_capacity_var:
            # Get variables:
            cap = self.variables['CAP']['pyomo']
            basic_var = self.variables[self.basic_variable]['pyomo']
            dt = self.ensys.hours_per_time_step

            def con_operation_limit(m, p, t):
                return basic_var[p, t] <= cap * dt

            setattr(self.pyB, 'con_operation_limit', pyomo.Constraint(
                pyM.time_set, rule=con_operation_limit))

    def con_min_load_rel(self, pyM):
        """
        Currently minimal part-loads can only be used if a binary variable for
        operation is defined and fixed capacities of the conversion units are
        specified (cap_min = cap_max). E.g.: |br|
        ``Q[p, t] >= capacity * BI_OP[p, t] * min_load_rel * dt``
        """
        # Only required if minimal part-loads should be modelled.
        if self.min_load_rel is not None:
            if self.capacity is not None:  # fixed capacity!

                bi_op = self.variables['BI_OP']['pyomo']
                basic_var = self.variables[self.basic_variable]['pyomo']
                cap = self.capacity
                min_load = self.min_load_rel
                dt = self.ensys.hours_per_time_step

                def con_min_load_rel(m, p, t):
                    return \
                        basic_var[p, t] >= cap * bi_op[p, t] * min_load * dt

                setattr(self.pyB, 'con_min_load_rel', pyomo.Constraint(
                    pyM.time_set, rule=con_min_load_rel))
            else:
                # TODO: Add this to the init check!
                raise NotImplementedError(
                    'Minimal part-loads with flexible unit capacity can be '
                    'modelled but it requires some effort. This feature has'
                    ' not been implemented yet. Please consider the '
                    '"user_expressions" attribute to model it yourself.\n '
                    'E.g.: Q_CAP_OR_OFF[p, t] == CAP * BI_OP[p, t] * dt '
                    '--> Linearization required (Glover)!,\n '
                    'Q[p, t] >= Q_CAP_OR_OFF[p, t] * min_load_rel * dt')

    def con_start_up_cost(self, ensys, pyM):
        """
        Constraint to determine the status of the binary start-up variable. If
        the operational status of a component changes from OFF (bi_op=0) to ON
        (bi_op=1) from one time step to the next, the binary start-up variable
        must take a value of 1. For shut-down and remaining ON or OFF, status
        variable can take both values but objective function forces it to be 0.
        |br| ``0 <= BI_OP[t-1] - BI_OP[t] + BI_SU[t]``
        """
        # Only if the start-up cost value is larger than 0
        if self.start_up_cost > 0:
            # Get binary variables:
            bi_op = self.variables['BI_OP']['pyomo']
            bi_su = self.variables['BI_SU']['pyomo']

            def con_start_up_cost(m, p, t):
                if t != 0:  # not in first time step of a period
                    return 0 <= bi_op[p, t - 1] - bi_op[p, t] + bi_su[p, t]
                else:
                    return pyomo.Constraint.Skip
            setattr(self.pyB, 'con_start_up_cost', pyomo.Constraint(
                pyM.time_set, rule=con_start_up_cost))

    def con_start_up_cost_inter(self, ensys, pyM):
        """
        Todo: Add description
        """
        # Only if start-up cost are applied, the inter-period formulation is
        # requested and the data is clustered.
        if self.start_up_cost > 0 and self.use_inter_period_formulation \
                and ensys.is_data_clustered:
            # Get binary variables:
            bi_op = self.variables['BI_OP']['pyomo']
            bi_su_inter = self.variables['BI_SU_INTER']['pyomo']

            def con_start_up_cost_inter(m, p):
                # not for first period, because there is no precursor to use
                if not p == ensys.periods[0]:

                    typ_period = ensys.periods_order[p]
                    prev_typ_period = ensys.periods_order[p-1]
                    last_ts_idx = pyM.time_set.last()[1]

                    return 0 <= bi_op[prev_typ_period, last_ts_idx] - bi_op[
                        typ_period, 0] + bi_su_inter[p]
                else:
                    return pyomo.Constraint.Skip
            setattr(self.pyB, 'con_start_up_cost_inter', pyomo.Constraint(
                ensys.periods, rule=con_start_up_cost_inter))

    # ==========================================================================
    #    S E R I A L I Z E
    # ==========================================================================
    def serialize(self):
        comp_dict = super().serialize()
        return comp_dict
