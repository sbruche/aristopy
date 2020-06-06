#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
**The Conversion class**

* Last edited: 2020-06-06
* Created by: Stefan Bruche (TU Berlin)
"""
import pyomo.environ as pyomo
from aristopy.component import Component
from aristopy import utils


class Conversion(Component):
    """
    A conversion component takes commodities at the inlet and provides other
    commodities at the outlet after an internal conversion.
    """
    def __init__(self, ensys, name, inlet, outlet, basic_variable,
                 has_existence_binary_var=None, has_operation_binary_var=None,
                 time_series_data=None, scalar_params=None,
                 additional_vars=None, user_expressions=None,
                 capacity=None, capacity_min=None, capacity_max=None,
                 capacity_per_module=None, maximal_module_number=None,
                 capex_per_capacity=0, capex_if_exist=0,
                 opex_per_capacity=0, opex_if_exist=0, opex_operation=0,
                 start_up_cost=0, min_load_rel=None,
                 instances_in_group=1,
                 group_has_existence_order=True, group_has_operation_order=True,
                 use_inter_period_formulation=True
                 ):
        """
        Initialize an instance of the Conversion class.

        *See the documentation of the Component class for a description of all
        keyword arguments and inherited methods.*

        :param start_up_cost:
            |br| *Default: 0*
        :type start_up_cost:

        :param min_load_rel:
            |br| *Default: None*
        :type min_load_rel:

        :param instances_in_group: States the number of similar component
            instances that are simultaneously created and arranged in a group.
            That means, the user has the possibility to instantiate multiple
            component instances (only for Conversion!) with identical
            specifications. These components work independently, but may have an
            order for their binary existence and/or operation variables (see:
            'group_has_existence_order', 'group_has_operation_order'). If a
            number larger than 1 is provided, the names of the components are
            extended with integers starting from 1 (e.g., 'conversion_1', ...).
            |br| *Default: 1*
        :type instances_in_group: int (>0)

        :param group_has_existence_order:
            |br| *Default: True*
        :type group_has_existence_order: bool

        :param group_has_operation_order:
            |br| *Default: True*
        :type group_has_operation_order: bool

        :param use_inter_period_formulation:
            |br| *Default: True*
        :type use_inter_period_formulation: bool
        """

        # Prevent None at inlet & outlet! (Flows are checked in Component init)
        if inlet is None:
            raise utils.io_error_message('Conversion', name, 'inlet')
        if outlet is None:
            raise utils.io_error_message('Conversion', name, 'outlet')

        Component.__init__(self, ensys=ensys, name=name,
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
                           capacity_per_module=capacity_per_module,
                           maximal_module_number=maximal_module_number,
                           capex_per_capacity=capex_per_capacity,
                           capex_if_exist=capex_if_exist,
                           opex_per_capacity=opex_per_capacity,
                           opex_if_exist=opex_if_exist,
                           opex_operation=opex_operation
                           )

        # Check and set the value for the minimal relative part-load
        if min_load_rel is not None:
            utils.is_positive_number(min_load_rel)
            assert min_load_rel <= 1, 'Maximal value for "min_load_rel" is 1!'
            if not self.has_bi_op:
                raise ValueError('Minimal part-loads require the availability '
                                 'of a operation binary variable '
                                 '("has_operation_binary_var=True").')
        self.min_load_rel = min_load_rel

        # Check and set more conversion specific input arguments
        self.start_up_cost = utils.set_if_positive(start_up_cost)  # [€/Start]
        utils.is_boolean(use_inter_period_formulation)  # check input
        self.use_inter_period_formulation = use_inter_period_formulation

        # Multiple instances formed and collected in one group
        utils.is_strictly_positive_int(instances_in_group)
        self.instances_in_group = instances_in_group
        utils.is_boolean(group_has_existence_order)
        self.group_has_existence_order = group_has_existence_order
        if self.instances_in_group > 1 and not self.has_bi_ex and \
                self.group_has_existence_order:  # is True
            raise ValueError('Group requires a binary existence variable if an '
                             'existence order is requested!')
        utils.is_boolean(group_has_operation_order)
        self.group_has_operation_order = group_has_operation_order
        if self.instances_in_group > 1 and not self.has_bi_op and \
                self.group_has_operation_order:  # is True
            raise ValueError('Group requires a binary operation variable if an '
                             'operation order is requested!')

        # Add variables for Start-up
        if self.start_up_cost != 0:
            if not self.has_bi_op:
                raise ValueError('Start-up cost require the availability '
                                 'of a binary operation variable.')
            # Else: add Start-up binary variable
            self.add_var(name=utils.BI_SU, domain='Binary', has_time_set=True)
            # If inter-period-formulation is requested: add another binary var
            if self.use_inter_period_formulation:
                self.add_var(name=utils.BI_SU_INTER, domain='Binary',
                             has_time_set=False, init=0,
                             alternative_set='inter_period_time_set')

        # Last step: Add the component to the EnergySystem instance
        self.add_to_energy_system(ensys, name, instances_in_group)

    def __repr__(self):
        return '<Conversion: "%s">' % self.name

    # ==========================================================================
    #    C O N V E N T I O N A L   C O N S T R A I N T   D E C L A R A T I O N
    # ==========================================================================
    def declare_component_constraints(self, ensys, model):
        """
        Method to declare all component constraints.

        *Method is not intended for public access!*

        :param ensys: Instance of the EnergySystem class
        :param model: Pyomo ConcreteModel of the EnergySystem instance
        """
        # Time-independent constraints :
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.con_couple_bi_ex_and_cap()
        self.con_cap_min()
        self.con_cap_modular()
        self.con_modular_sym_break()
        self.con_couple_existence_and_modular()
        self.con_existence_sym_break(ensys)

        # Time-dependent constraints :
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.con_bi_var_ex_and_op_relation(model)
        self.con_operation_limit(model)
        self.con_couple_op_binary_and_basic_var(model)
        self.con_min_load_rel(model)
        self.con_start_up_cost(ensys, model)
        self.con_start_up_cost_inter(ensys, model)
        self.con_operation_sym_break(ensys, model)

    # **************************************************************************
    #    Time-independent constraints
    # **************************************************************************
    def con_existence_sym_break(self, ensys):
        """
        TODO: Add description!
        """
        if self.number_in_group > 1 and self.group_has_existence_order:
            bi_ex = self.variables[utils.BI_EX]['pyomo']
            # Get 'bi_ex' of the previous component in the sequence
            prior = self.group_name + '_{}'.format(self.number_in_group - 1)
            prior_comp = ensys.components[prior]
            bi_ex_prior = prior_comp.variables[utils.BI_EX]['pyomo']

            def con_existence_sym_break(m):
                return bi_ex <= bi_ex_prior
            setattr(self.block, 'con_existence_sym_break',
                    pyomo.Constraint(rule=con_existence_sym_break))

    # **************************************************************************
    #    Time-dependent constraints
    # **************************************************************************
    def con_operation_sym_break(self, ensys, model):
        """
        TODO: Add description!
        """
        if self.number_in_group > 1 and self.group_has_operation_order:
            # Get variables:
            bi_op = self.variables[utils.BI_OP]['pyomo']
            prior = self.group_name + '_{}'.format(self.number_in_group - 1)
            prior_comp = ensys.components[prior]
            bi_op_prior = prior_comp.variables[utils.BI_OP]['pyomo']

            def con_operation_sym_break(m, p, t):
                return bi_op[p, t] <= bi_op_prior[p, t]
            setattr(self.block, 'con_operation_sym_break', pyomo.Constraint(
                model.time_set, rule=con_operation_sym_break))

    def con_operation_limit(self, model):
        """
        The operation variable (ref.: basic commodity) of a conversion unit
        (MWh) is limit by its nominal power (MW) multiplied with the number of
        hours per time step. E.g.: |br|
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

            setattr(self.block, 'con_operation_limit', pyomo.Constraint(
                model.time_set, rule=con_operation_limit))

    def con_min_load_rel(self, model):
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

                setattr(self.block, 'con_min_load_rel', pyomo.Constraint(
                    model.time_set, rule=con_min_load_rel))
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

    def con_start_up_cost(self, ensys, model):
        """
        Constraint to determine the status of the binary start-up variable. If
        the operational status of a component changes from OFF (bi_op=0) to ON
        (bi_op=1) from one time step to the next, the binary start-up variable
        must take a value of 1. For shut-down and remaining ON or OFF, status
        variable can take both values but objective function forces it to be 0.
        E.g.: |br| ``0 <= BI_OP[t-1] - BI_OP[t] + BI_SU[t]``
        """
        # Only if the start-up cost value is larger than 0
        if self.start_up_cost > 0:
            # Get binary variables:
            bi_op = self.variables[utils.BI_OP]['pyomo']
            bi_su = self.variables[utils.BI_SU]['pyomo']

            def con_start_up_cost(m, p, t):
                if t != 0:  # not in first time step of a period
                    return 0 <= bi_op[p, t - 1] - bi_op[p, t] + bi_su[p, t]
                else:
                    return pyomo.Constraint.Skip
            setattr(self.block, 'con_start_up_cost', pyomo.Constraint(
                model.time_set, rule=con_start_up_cost))

    def con_start_up_cost_inter(self, ensys, model):
        """
        Todo: Add description
        """
        # Only if start-up cost are applied, the inter-period formulation is
        # requested and the data is clustered.
        if self.start_up_cost > 0 and self.use_inter_period_formulation \
                and ensys.is_data_clustered:
            # Get binary variables:
            bi_op = self.variables[utils.BI_OP]['pyomo']
            bi_su_inter = self.variables[utils.BI_SU_INTER]['pyomo']

            def con_start_up_cost_inter(m, p):
                # not for first period, because there is no precursor to use
                if not p == ensys.periods[0]:

                    typ_period = ensys.periods_order[p]
                    prev_typ_period = ensys.periods_order[p-1]
                    last_ts_idx = model.time_set.last()[1]

                    return 0 <= bi_op[prev_typ_period, last_ts_idx] - bi_op[
                        typ_period, 0] + bi_su_inter[p]
                else:
                    return pyomo.Constraint.Skip
            setattr(self.block, 'con_start_up_cost_inter', pyomo.Constraint(
                ensys.periods, rule=con_start_up_cost_inter))

    # ==========================================================================
    #    O B J E C T I V E   F U N C T I O N   C O N T R I B U T I O N
    # ==========================================================================
    def get_objective_function_contribution(self, ensys, model):
        """
        Calculate the objective function contributions of the component and add
        the values to the component dictionary "comp_obj_dict".

        *Method is not intended for public access!*

        :param ensys: Instance of the EnergySystem class
        :param model: Pyomo ConcreteModel of the EnergySystem instance
        """
        # Call method in "Component" class and calculate CAPEX and OPEX
        super().get_objective_function_contribution(ensys, model)

        # START-UP COST :
        # ~~~~~~~~~~~~~~~
        if self.start_up_cost > 0:
            bi_su = self.variables[utils.BI_SU]['pyomo']  # only avail. if '>0'
            start_cost_intra = -1 * ensys.pvf * self.start_up_cost * sum(
                bi_su[p, t] * ensys.period_occurrences[p] for p, t in
                model.time_set) / ensys.number_of_years
            if self.use_inter_period_formulation and ensys.is_data_clustered:
                bi_su_inter = self.variables[utils.BI_SU_INTER]['pyomo']
                start_cost_inter = -1 * ensys.pvf * self.start_up_cost * pyomo.\
                    summation(bi_su_inter) / ensys.number_of_years
            else:
                start_cost_inter = 0
            self.comp_obj_dict['start_up_cost'] = \
                start_cost_intra + start_cost_inter

        return sum(self.comp_obj_dict.values())

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
        return comp_dict
